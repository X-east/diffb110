"""Compare current reconstruction against true-bond oracle reconstruction."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from rdkit import Chem, RDLogger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIFFSBDD_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_ROOT = Path(__file__).resolve().parent
for extra in (PROJECT_ROOT, DIFFSBDD_ROOT, ANALYSIS_ROOT):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import constants
from dataset import ProcessedLigandPocketDataset
from edge_data_oracle_report import (
    _atom_symbols,
    _build_oracle_mol,
    _lookup_reference_length,
)
from molecule_builder import build_molecule

RDLogger.DisableLog("rdApp.*")

RDKIT_BOND_TYPE_TO_ID = {
    Chem.BondType.SINGLE: 1,
    Chem.BondType.DOUBLE: 2,
    Chem.BondType.TRIPLE: 3,
    Chem.BondType.AROMATIC: 4,
}


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    """Write rows to CSV with stable column order."""
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _true_edges(sample: dict[str, object]) -> dict[tuple[int, int], int]:
    """Return undirected true ligand bonds from halfedge labels."""
    halfedge_index = sample["ligand_halfedge_index"]
    halfedge_type = sample["ligand_halfedge_type"]
    edges = {}
    for edge_pos, edge_type_tensor in enumerate(halfedge_type):
        edge_type = int(edge_type_tensor)
        if edge_type <= 0:
            continue
        atom_i = int(halfedge_index[0, edge_pos])
        atom_j = int(halfedge_index[1, edge_pos])
        edges[tuple(sorted((atom_i, atom_j)))] = edge_type
    return edges


def _mol_edges(mol: Chem.Mol | None) -> dict[tuple[int, int], int]:
    """Return undirected RDKit molecule bonds as internal bond type ids."""
    if mol is None:
        return {}
    edges = {}
    for bond in mol.GetBonds():
        edge_type = RDKIT_BOND_TYPE_TO_ID.get(bond.GetBondType(), 0)
        if edge_type <= 0:
            continue
        atom_i = int(bond.GetBeginAtomIdx())
        atom_j = int(bond.GetEndAtomIdx())
        edges[tuple(sorted((atom_i, atom_j)))] = edge_type
    return edges


def _safe_sanitize(mol: Chem.Mol | None) -> tuple[Chem.Mol | None, str | None]:
    """Sanitize a copy of a molecule and return the error type if sanitization fails."""
    if mol is None:
        return None, "no_mol"
    copied = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(copied)
    except Exception as exc:  # noqa: BLE001 - RDKit exposes several exception classes
        return copied, type(exc).__name__
    return copied, None


def _edge_metrics(
    predicted_edges: dict[tuple[int, int], int],
    true_edges: dict[tuple[int, int], int],
) -> dict[str, float | int | str]:
    """Calculate pair-level precision/recall/F1 and bond-type agreement."""
    predicted_pairs = set(predicted_edges)
    true_pairs = set(true_edges)
    matched_pairs = predicted_pairs & true_pairs
    precision = len(matched_pairs) / max(1, len(predicted_pairs))
    recall = len(matched_pairs) / max(1, len(true_pairs))
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    type_matches = sum(
        int(predicted_edges[pair] == true_edges[pair])
        for pair in matched_pairs
    )
    type_accuracy = "" if not matched_pairs else type_matches / len(matched_pairs)
    return {
        "pred_bonds": len(predicted_pairs),
        "true_bonds": len(true_pairs),
        "matched_bonds": len(matched_pairs),
        "extra_bonds": len(predicted_pairs - true_pairs),
        "missed_bonds": len(true_pairs - predicted_pairs),
        "edge_precision": precision,
        "edge_recall": recall,
        "edge_f1": f1,
        "bond_type_accuracy_on_matched": type_accuracy,
    }


def _bond_length_metrics(
    edges: dict[tuple[int, int], int],
    atom_symbols: list[str],
    coords: torch.Tensor,
    tolerance: float,
) -> dict[str, float | int]:
    """Calculate bond length deviation against simple reference tables."""
    checked = 0
    bad = 0
    abs_errors = []
    for (atom_i, atom_j), edge_type in edges.items():
        atom_a = atom_symbols[atom_i]
        atom_b = atom_symbols[atom_j]
        ref_length = _lookup_reference_length(atom_a, atom_b, edge_type)
        if ref_length is None:
            continue
        length = float(torch.linalg.norm(coords[atom_i] - coords[atom_j]).item())
        abs_error = abs(length - ref_length)
        checked += 1
        bad += int(abs_error > tolerance)
        abs_errors.append(abs_error)
    mean_abs_error = sum(abs_errors) / len(abs_errors) if abs_errors else 0.0
    return {
        "checked_bond_lengths": checked,
        "bad_bond_lengths": bad,
        "bad_bond_length_rate": bad / max(1, checked),
        "mean_bond_length_abs_error": mean_abs_error,
    }


def _fragment_metrics(mol: Chem.Mol | None) -> dict[str, float | int]:
    """Return simple fragment statistics."""
    if mol is None or mol.GetNumAtoms() == 0:
        return {
            "num_fragments": 0,
            "largest_fragment_atoms": 0,
            "largest_fragment_atom_frac": 0.0,
        }
    fragments = Chem.GetMolFrags(mol, sanitizeFrags=False)
    largest = max((len(fragment) for fragment in fragments), default=0)
    return {
        "num_fragments": len(fragments),
        "largest_fragment_atoms": largest,
        "largest_fragment_atom_frac": largest / max(1, mol.GetNumAtoms()),
    }


def _try_build_current(
    coords: torch.Tensor,
    atom_types: torch.Tensor,
    dataset_info: dict[str, object],
    use_openbabel: bool,
) -> tuple[Chem.Mol | None, str | None]:
    """Build a molecule with the current DiffSBDD reconstruction function."""
    try:
        return build_molecule(
            coords,
            atom_types,
            dataset_info,
            add_coords=True,
            use_openbabel=use_openbabel,
        ), None
    except Exception as exc:  # noqa: BLE001 - reconstruction failures are recorded
        return None, type(exc).__name__


def _format_float(value: object) -> object:
    """Format floats for stable CSV output."""
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def _summarize(rows: list[dict[str, object]], methods: list[str]) -> list[dict[str, object]]:
    """Aggregate per-molecule reconstruction rows by method."""
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)

    summary_rows = []
    for method in methods:
        method_rows = grouped[method]
        n = len(method_rows)
        numeric_fields = [
            "sanitize_success",
            "pred_bonds",
            "matched_bonds",
            "extra_bonds",
            "missed_bonds",
            "edge_precision",
            "edge_recall",
            "edge_f1",
            "bad_bond_length_rate",
            "mean_bond_length_abs_error",
            "num_fragments",
            "largest_fragment_atom_frac",
        ]
        row = {"method": method, "samples": n}
        for field in numeric_fields:
            values = [
                float(item[field])
                for item in method_rows
                if item[field] != ""
            ]
            row[f"mean_{field}"] = sum(values) / len(values) if values else 0.0
        type_values = [
            float(item["bond_type_accuracy_on_matched"])
            for item in method_rows
            if item["bond_type_accuracy_on_matched"] != ""
        ]
        row["mean_bond_type_accuracy_on_matched"] = (
            sum(type_values) / len(type_values) if type_values else 0.0
        )
        summary_rows.append(row)
    return summary_rows


def _plot_summary(outdir: Path, summary_rows: list[dict[str, object]]) -> None:
    """Write compact bar plots for key reconstruction metrics."""
    methods = [str(row["method"]) for row in summary_rows]
    plot_specs = [
        ("mean_sanitize_success", "sanitize success", "sanitize_success_by_method.png"),
        ("mean_edge_f1", "edge F1", "edge_f1_by_method.png"),
        ("mean_bad_bond_length_rate", "bad bond length rate", "bad_bond_length_by_method.png"),
        ("mean_largest_fragment_atom_frac", "largest fragment atom fraction", "fragment_by_method.png"),
    ]
    for field, ylabel, filename in plot_specs:
        values = [float(row[field]) for row in summary_rows]
        plt.figure(figsize=(7, 4))
        plt.bar(methods, values, color=["#4C78A8", "#F58518", "#54A24B"][:len(methods)])
        plt.ylabel(ylabel)
        plt.xticks(rotation=20, ha="right")
        plt.ylim(0, max(1.0, max(values) * 1.15 if values else 1.0))
        plt.tight_layout()
        plt.savefig(outdir / filename, dpi=180)
        plt.close()


def generate_report(args: argparse.Namespace) -> Path:
    """Run reconstruction ablation and write reports."""
    dataset_info = constants.dataset_params[args.dataset]
    dataset = ProcessedLigandPocketDataset(args.npz_path, center=False)
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    methods = ["openbabel_current", "edm_distance_rule", "true_bond_oracle"]
    limit = len(dataset) if args.max_samples is None else min(args.max_samples, len(dataset))
    rows = []

    for sample_idx in range(limit):
        sample = dataset[sample_idx]
        coords = sample["lig_coords"].float()
        atom_types = sample["lig_one_hot"].argmax(dim=-1).long()
        atom_symbols = _atom_symbols(sample["lig_one_hot"], dataset_info["atom_decoder"])
        true_edges = _true_edges(sample)

        candidates = []
        openbabel_mol, openbabel_build_error = _try_build_current(
            coords, atom_types, dataset_info, use_openbabel=True)
        candidates.append(("openbabel_current", openbabel_mol, openbabel_build_error))

        edm_mol, edm_build_error = _try_build_current(
            coords, atom_types, dataset_info, use_openbabel=False)
        candidates.append(("edm_distance_rule", edm_mol, edm_build_error))

        oracle_mol, oracle_error = _build_oracle_mol(
            atom_symbols,
            coords,
            sample["ligand_halfedge_index"],
            sample["ligand_halfedge_type"],
        )
        candidates.append(("true_bond_oracle", oracle_mol, oracle_error))

        for method, raw_mol, build_error in candidates:
            mol, sanitize_error = _safe_sanitize(raw_mol)
            final_error = build_error or sanitize_error or ""
            predicted_edges = true_edges if method == "true_bond_oracle" else _mol_edges(mol)
            edge_metrics = _edge_metrics(predicted_edges, true_edges)
            length_metrics = _bond_length_metrics(
                predicted_edges,
                atom_symbols,
                coords,
                args.length_tolerance,
            )
            fragment_metrics = _fragment_metrics(mol)
            smiles = ""
            if mol is not None and sanitize_error is None:
                smiles = Chem.MolToSmiles(mol)

            row = {
                "sample_idx": sample_idx,
                "name": sample["names"],
                "method": method,
                "num_atoms": int(sample["num_lig_atoms"]),
                "sanitize_success": int(final_error == ""),
                "error": final_error,
                "smiles": smiles,
            }
            row.update(edge_metrics)
            row.update(length_metrics)
            row.update(fragment_metrics)
            rows.append({key: _format_float(value) for key, value in row.items()})

    per_molecule_fields = [
        "sample_idx",
        "name",
        "method",
        "num_atoms",
        "sanitize_success",
        "error",
        "pred_bonds",
        "true_bonds",
        "matched_bonds",
        "extra_bonds",
        "missed_bonds",
        "edge_precision",
        "edge_recall",
        "edge_f1",
        "bond_type_accuracy_on_matched",
        "checked_bond_lengths",
        "bad_bond_lengths",
        "bad_bond_length_rate",
        "mean_bond_length_abs_error",
        "num_fragments",
        "largest_fragment_atoms",
        "largest_fragment_atom_frac",
        "smiles",
    ]
    _write_csv(outdir / "reconstruction_ablation_per_molecule.csv", rows, per_molecule_fields)

    summary_rows = _summarize(rows, methods)
    summary_fields = ["method", "samples"] + [
        key
        for key in summary_rows[0]
        if key not in {"method", "samples"}
    ]
    _write_csv(
        outdir / "reconstruction_ablation_summary.csv",
        [
            {key: _format_float(value) for key, value in row.items()}
            for row in summary_rows
        ],
        summary_fields,
    )
    _plot_summary(outdir, summary_rows)

    report_path = outdir / "reconstruction_ablation_report.md"
    with report_path.open("w") as handle:
        handle.write("# Reconstruction Ablation Report\n\n")
        handle.write(f"- npz_path: `{args.npz_path}`\n")
        handle.write(f"- dataset: `{args.dataset}`\n")
        handle.write(f"- samples: {limit}\n")
        handle.write(f"- length_tolerance: {args.length_tolerance}\n\n")
        handle.write("| method | sanitize | edge_f1 | bad_bond_length | fragments | extra | missed |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for row in summary_rows:
            handle.write(
                f"| {row['method']} "
                f"| {float(row['mean_sanitize_success']):.3f} "
                f"| {float(row['mean_edge_f1']):.3f} "
                f"| {float(row['mean_bad_bond_length_rate']):.3f} "
                f"| {float(row['mean_num_fragments']):.3f} "
                f"| {float(row['mean_extra_bonds']):.3f} "
                f"| {float(row['mean_missed_bonds']):.3f} |\n"
            )
        handle.write("\n![sanitize success](sanitize_success_by_method.png)\n\n")
        handle.write("![edge f1](edge_f1_by_method.png)\n\n")
        handle.write("![bad bond length](bad_bond_length_by_method.png)\n\n")
        handle.write("![fragment](fragment_by_method.png)\n")

    return report_path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz-path", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        choices=sorted(constants.dataset_params.keys()),
        default="crossdock_full",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("analysis") / f"reconstruction_ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--length-tolerance", type=float, default=0.25)
    return parser.parse_args()


if __name__ == "__main__":
    report = generate_report(parse_args())
    print(f"Wrote reconstruction ablation report to {report}")
