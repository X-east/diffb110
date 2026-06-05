"""Generate oracle reconstruction and bond statistics for DiffSBDD edge data."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter, defaultdict
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
for extra in (PROJECT_ROOT, DIFFSBDD_ROOT):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import constants
from dataset import ProcessedLigandPocketDataset
from edge_diffusion.oracle_reconstruction import build_oracle_mol
from edge_utils import BOND_TYPE_NAMES

RDLogger.DisableLog("rdApp.*")

REFERENCE_BOND_LENGTHS = {
    1: constants.bonds1,
    2: constants.bonds2,
    3: constants.bonds3,
}


def _atom_symbols(one_hot: torch.Tensor, atom_decoder: list[str]) -> list[str]:
    """Convert ligand one-hot atom features to element symbols."""
    atom_indices = one_hot.argmax(dim=-1).cpu().tolist()
    return [atom_decoder[int(idx)] for idx in atom_indices]


def _lookup_reference_length(atom_a: str, atom_b: str, bond_type: int) -> float | None:
    """Return reference bond length in Angstrom when available."""
    if bond_type == 4:
        return 1.40
    bond_table = REFERENCE_BOND_LENGTHS.get(bond_type)
    if bond_table is None:
        return None
    if atom_a in bond_table and atom_b in bond_table[atom_a]:
        return bond_table[atom_a][atom_b] / 100.0
    if atom_b in bond_table and atom_a in bond_table[atom_b]:
        return bond_table[atom_b][atom_a] / 100.0
    return None


def _build_oracle_mol(
    atom_symbols: list[str],
    positions: torch.Tensor,
    halfedge_index: torch.Tensor,
    halfedge_type: torch.Tensor,
) -> tuple[Chem.Mol | None, str | None]:
    """Build an RDKit molecule using true halfedge labels.

    Kept as a compatibility wrapper for existing analysis scripts. New code
    should call ``edge_diffusion.oracle_reconstruction.build_oracle_mol``.
    """
    result = build_oracle_mol(atom_symbols, positions, halfedge_index, halfedge_type)
    return result.mol, result.sanitize_error


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    """Write rows to CSV with stable column order."""
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_sdf_record(
    handle,
    mol: Chem.Mol,
    properties: dict[str, object],
) -> None:
    """Write one SDF record without forcing RDKit kekulization."""
    mol_block = Chem.MolToMolBlock(mol, kekulize=False)
    handle.write(mol_block)
    if not mol_block.endswith("\n"):
        handle.write("\n")
    for key, value in properties.items():
        handle.write(f">  <{key}>\n{value}\n\n")
    handle.write("$$$$\n")


def _plot_bond_type_counts(path: Path, counts: Counter[int]) -> None:
    """Save a bond type count bar plot."""
    keys = sorted(BOND_TYPE_NAMES)
    values = [counts.get(key, 0) for key in keys]
    labels = [BOND_TYPE_NAMES[key] for key in keys]
    plt.figure(figsize=(7, 4))
    plt.bar(labels, values, color="#4C78A8")
    plt.ylabel("count")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_bond_lengths(path: Path, lengths: list[float]) -> None:
    """Save a bond length histogram."""
    plt.figure(figsize=(7, 4))
    if lengths:
        plt.hist(lengths, bins=50, color="#59A14F", alpha=0.9)
    plt.xlabel("bond length (Angstrom)")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def generate_report(args: argparse.Namespace) -> Path:
    """Run oracle reconstruction and write statistics files."""
    dataset_info = constants.dataset_params[args.dataset]
    dataset = ProcessedLigandPocketDataset(args.npz_path, center=False)
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    sdf_path = getattr(args, "sdf_path", None)
    sdf_handle = Path(sdf_path).open("w") if sdf_path is not None else None

    limit = len(dataset) if args.max_samples is None else min(args.max_samples, len(dataset))
    bond_type_counts: Counter[int] = Counter()
    molecule_rows = []
    length_rows = []
    length_values = []
    sanitize_success = 0
    unsupported = 0
    sdf_write_success = 0
    sdf_write_error = 0
    sanitize_error_counts: Counter[str] = Counter()
    initial_sanitize_error_counts: Counter[str] = Counter()
    repair_strategy_counts: Counter[str] = Counter()
    bad_length_count = 0
    checked_length_count = 0
    pair_length_values: dict[str, list[float]] = defaultdict(list)

    try:
        for sample_idx in range(limit):
            sample = dataset[sample_idx]
            atom_symbols = _atom_symbols(sample["lig_one_hot"], dataset_info["atom_decoder"])
            halfedge_type = sample["ligand_halfedge_type"]
            halfedge_index = sample["ligand_halfedge_index"]
            coords = sample["lig_coords"].float()

            sample_counts = Counter(int(x) for x in halfedge_type.tolist())
            bond_type_counts.update(sample_counts)

            for edge_pos, edge_type_tensor in enumerate(halfedge_type):
                edge_type = int(edge_type_tensor)
                if edge_type <= 0:
                    continue
                atom_i = int(halfedge_index[0, edge_pos])
                atom_j = int(halfedge_index[1, edge_pos])
                length = float(torch.linalg.norm(coords[atom_i] - coords[atom_j]).item())
                length_values.append(length)
                atom_a, atom_b = atom_symbols[atom_i], atom_symbols[atom_j]
                pair_key = "-".join(sorted((atom_a, atom_b))) + f"|{BOND_TYPE_NAMES[edge_type]}"
                pair_length_values[pair_key].append(length)
                ref_length = _lookup_reference_length(atom_a, atom_b, edge_type)
                is_bad = False
                if ref_length is not None:
                    checked_length_count += 1
                    is_bad = abs(length - ref_length) > args.length_tolerance
                    bad_length_count += int(is_bad)
                length_rows.append({
                    "sample_idx": sample_idx,
                    "atom_i": atom_i,
                    "atom_j": atom_j,
                    "atom_a": atom_a,
                    "atom_b": atom_b,
                    "bond_type": BOND_TYPE_NAMES[edge_type],
                    "length": f"{length:.6f}",
                    "reference_length": "" if ref_length is None else f"{ref_length:.6f}",
                    "is_bad_length": int(is_bad),
                })

            reconstruction = build_oracle_mol(atom_symbols, coords, halfedge_index, halfedge_type)
            mol = reconstruction.mol
            sanitize_error = reconstruction.sanitize_error
            initial_sanitize_error = reconstruction.initial_sanitize_error
            repair_strategy = reconstruction.repair_strategy
            initial_sanitize_error_counts[initial_sanitize_error or "success"] += 1
            repair_strategy_counts[repair_strategy] += 1
            if sanitize_error is None:
                sanitize_success += 1
                sanitize_error_counts["success"] += 1
            if sanitize_error == "unsupported_atom_type":
                unsupported += 1
            if sanitize_error is not None:
                sanitize_error_counts[sanitize_error] += 1
            if sdf_handle is not None and mol is not None:
                try:
                    _write_sdf_record(
                        sdf_handle,
                        mol,
                        {
                            "_Name": str(sample["names"]),
                            "sample_idx": sample_idx,
                            "oracle_sanitize_success": int(sanitize_error is None),
                            "initial_sanitize_error": "" if initial_sanitize_error is None else initial_sanitize_error,
                            "sanitize_error": "" if sanitize_error is None else sanitize_error,
                            "repair_strategy": repair_strategy,
                        },
                    )
                    sdf_write_success += 1
                except Exception:  # noqa: BLE001 - keep report running and count failures
                    sdf_write_error += 1
            molecule_rows.append({
                "sample_idx": sample_idx,
                "name": sample["names"],
                "num_atoms": int(sample["num_lig_atoms"]),
                "num_bonds": int((halfedge_type > 0).sum().item()),
                "num_halfedges": int(len(halfedge_type)),
                "oracle_sanitize_success": int(sanitize_error is None),
                "initial_sanitize_error": "" if initial_sanitize_error is None else initial_sanitize_error,
                "sanitize_error": "" if sanitize_error is None else sanitize_error,
                "repair_strategy": repair_strategy,
                "smiles": "" if mol is None or sanitize_error is not None else Chem.MolToSmiles(mol),
            })
    finally:
        if sdf_handle is not None:
            sdf_handle.close()

    bond_count_rows = [
        {
            "bond_type_id": key,
            "bond_type": BOND_TYPE_NAMES[key],
            "count": bond_type_counts.get(key, 0),
            "fraction": (
                f"{bond_type_counts.get(key, 0) / max(1, sum(bond_type_counts.values())):.8f}"
            ),
        }
        for key in sorted(BOND_TYPE_NAMES)
    ]
    pair_rows = [
        {
            "pair_bond_type": key,
            "count": len(values),
            "mean_length": f"{sum(values) / len(values):.6f}",
            "min_length": f"{min(values):.6f}",
            "max_length": f"{max(values):.6f}",
        }
        for key, values in sorted(pair_length_values.items())
    ]
    sanitize_error_rows = [
        {
            "sanitize_error": error_name,
            "count": count,
            "fraction": f"{count / max(1, limit):.8f}",
        }
        for error_name, count in sorted(sanitize_error_counts.items())
    ]
    initial_sanitize_error_rows = [
        {
            "initial_sanitize_error": error_name,
            "count": count,
            "fraction": f"{count / max(1, limit):.8f}",
        }
        for error_name, count in sorted(initial_sanitize_error_counts.items())
    ]
    repair_strategy_rows = [
        {
            "repair_strategy": strategy,
            "count": count,
            "fraction": f"{count / max(1, limit):.8f}",
        }
        for strategy, count in sorted(repair_strategy_counts.items())
    ]

    _write_csv(
        outdir / "oracle_reconstruction.csv",
        molecule_rows,
        [
            "sample_idx", "name", "num_atoms", "num_bonds", "num_halfedges",
            "oracle_sanitize_success", "initial_sanitize_error",
            "sanitize_error", "repair_strategy", "smiles",
        ],
    )
    _write_csv(
        outdir / "bond_type_counts.csv",
        bond_count_rows,
        ["bond_type_id", "bond_type", "count", "fraction"],
    )
    _write_csv(
        outdir / "bond_length_by_type.csv",
        length_rows,
        [
            "sample_idx", "atom_i", "atom_j", "atom_a", "atom_b",
            "bond_type", "length", "reference_length", "is_bad_length",
        ],
    )
    _write_csv(
        outdir / "bond_pair_length_summary.csv",
        pair_rows,
        ["pair_bond_type", "count", "mean_length", "min_length", "max_length"],
    )
    _write_csv(
        outdir / "sanitize_error_counts.csv",
        sanitize_error_rows,
        ["sanitize_error", "count", "fraction"],
    )
    _write_csv(
        outdir / "initial_sanitize_error_counts.csv",
        initial_sanitize_error_rows,
        ["initial_sanitize_error", "count", "fraction"],
    )
    _write_csv(
        outdir / "repair_strategy_counts.csv",
        repair_strategy_rows,
        ["repair_strategy", "count", "fraction"],
    )
    _plot_bond_type_counts(outdir / "bond_type_distribution.png", bond_type_counts)
    _plot_bond_lengths(outdir / "bond_length_distribution.png", length_values)

    total_halfedges = sum(bond_type_counts.values())
    total_real_bonds = total_halfedges - bond_type_counts.get(0, 0)
    report = outdir / "edge_data_oracle_report.md"
    with report.open("w") as handle:
        handle.write("# Edge Data Oracle Report\n\n")
        handle.write(f"- npz_path: `{args.npz_path}`\n")
        handle.write(f"- dataset: `{args.dataset}`\n")
        handle.write(f"- samples: {limit}\n")
        handle.write(f"- oracle_sanitize_success: {sanitize_success}/{limit}\n")
        handle.write(f"- unsupported_atom_type: {unsupported}\n")
        if sdf_path is not None:
            handle.write(f"- oracle_sdf: `{sdf_path}`\n")
            handle.write(f"- oracle_sdf_write_success: {sdf_write_success}/{limit}\n")
            handle.write(f"- oracle_sdf_write_error: {sdf_write_error}\n")
        handle.write(f"- total_halfedges: {total_halfedges}\n")
        handle.write(f"- total_real_bonds: {total_real_bonds}\n")
        handle.write(
            f"- no_bond_fraction: {bond_type_counts.get(0, 0) / max(1, total_halfedges):.6f}\n"
        )
        handle.write(
            f"- bad_bond_length_rate: {bad_length_count / max(1, checked_length_count):.6f}"
            f" ({bad_length_count}/{checked_length_count})\n\n"
        )
        handle.write("## Sanitize Error Counts\n\n")
        handle.write("Final sanitize status after oracle reconstruction repairs.\n\n")
        handle.write("| sanitize_error | count | fraction |\n")
        handle.write("|---|---:|---:|\n")
        for row in sanitize_error_rows:
            handle.write(
                f"| {row['sanitize_error']} | {row['count']} | {row['fraction']} |\n"
            )
        handle.write("\n")
        handle.write("## Initial Sanitize Error Counts\n\n")
        handle.write("| initial_sanitize_error | count | fraction |\n")
        handle.write("|---|---:|---:|\n")
        for row in initial_sanitize_error_rows:
            handle.write(
                f"| {row['initial_sanitize_error']} | {row['count']} | {row['fraction']} |\n"
            )
        handle.write("\n")
        handle.write("## Repair Strategy Counts\n\n")
        handle.write("| repair_strategy | count | fraction |\n")
        handle.write("|---|---:|---:|\n")
        for row in repair_strategy_rows:
            handle.write(
                f"| {row['repair_strategy']} | {row['count']} | {row['fraction']} |\n"
            )
        handle.write("\n")
        handle.write("![bond type distribution](bond_type_distribution.png)\n\n")
        handle.write("![bond length distribution](bond_length_distribution.png)\n")

    return report


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
        default=Path("analysis") / f"edge_data_oracle_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--length-tolerance", type=float, default=0.25)
    parser.add_argument("--sdf-path", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    report_path = generate_report(parse_args())
    print(f"Wrote oracle report to {report_path}")
