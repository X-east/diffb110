"""Compare repaired oracle molecules against original ligand SDF files."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
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
from edge_utils import BOND_TYPE_NAMES, BOND_TYPE_TO_ID

RDLogger.DisableLog("rdApp.*")


@dataclass(frozen=True)
class MoleculeFacts:
    """Compact molecule facts used for original-vs-oracle comparison."""

    smiles_iso: str
    smiles_no_stereo: str
    num_atoms: int
    num_bonds: int
    total_formal_charge: int
    formal_charges: tuple[int, ...]
    aromatic_atom_flags: tuple[int, ...]
    aromatic_atom_count: int
    aromatic_bond_count: int
    bond_map: dict[tuple[int, int], int]
    bond_type_counts: dict[str, int]


def ligand_sdf_relpath_from_name(name: str) -> Path:
    """Extract the ligand SDF relative path from a processed CrossDocked name."""
    marker = ".pdb_"
    if marker not in name:
        raise ValueError(f"Cannot parse ligand SDF path from name: {name}")
    relpath = name.split(marker, 1)[1]
    if not relpath.endswith(".sdf"):
        raise ValueError(f"Parsed ligand path is not an SDF: {relpath}")
    return Path(relpath)


def _atom_symbols(one_hot: torch.Tensor, atom_decoder: list[str]) -> list[str]:
    """Convert one-hot atom features to element symbols."""
    atom_indices = one_hot.argmax(dim=-1).cpu().tolist()
    return [atom_decoder[int(idx)] for idx in atom_indices]


def _load_original_sdf(path: Path) -> tuple[Chem.Mol | None, str | None]:
    """Load, sanitize, and dehydrogenate an original ligand SDF."""
    if not path.exists():
        return None, "missing_sdf"
    mol = Chem.MolFromMolFile(str(path), sanitize=False, removeHs=False)
    if mol is None:
        return None, "mol_from_file_failed"
    try:
        Chem.SanitizeMol(mol)
        mol = Chem.RemoveHs(mol, sanitize=True)
        Chem.SanitizeMol(mol)
    except Exception as exc:  # noqa: BLE001 - report exact RDKit failure class
        return None, type(exc).__name__
    return mol, None


def _canonical_smiles(mol: Chem.Mol, isomeric: bool) -> str:
    """Return canonical SMILES with optional stereochemistry."""
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric)


def _molecule_facts(mol: Chem.Mol) -> MoleculeFacts:
    """Extract compact atom, charge, aromaticity, and bond facts."""
    formal_charges = tuple(atom.GetFormalCharge() for atom in mol.GetAtoms())
    aromatic_atom_flags = tuple(int(atom.GetIsAromatic()) for atom in mol.GetAtoms())
    bond_map: dict[tuple[int, int], int] = {}
    bond_type_counts = {name: 0 for name in BOND_TYPE_NAMES.values()}
    aromatic_bond_count = 0
    for bond in mol.GetBonds():
        atom_i = int(bond.GetBeginAtomIdx())
        atom_j = int(bond.GetEndAtomIdx())
        bond_id = BOND_TYPE_TO_ID.get(bond.GetBondType(), -1)
        key = tuple(sorted((atom_i, atom_j)))
        bond_map[key] = bond_id
        bond_name = BOND_TYPE_NAMES.get(bond_id, "unknown")
        bond_type_counts[bond_name] = bond_type_counts.get(bond_name, 0) + 1
        aromatic_bond_count += int(bond.GetIsAromatic() or bond_id == 4)

    return MoleculeFacts(
        smiles_iso=_canonical_smiles(mol, isomeric=True),
        smiles_no_stereo=_canonical_smiles(mol, isomeric=False),
        num_atoms=mol.GetNumAtoms(),
        num_bonds=mol.GetNumBonds(),
        total_formal_charge=sum(formal_charges),
        formal_charges=formal_charges,
        aromatic_atom_flags=aromatic_atom_flags,
        aromatic_atom_count=sum(aromatic_atom_flags),
        aromatic_bond_count=aromatic_bond_count,
        bond_map=bond_map,
        bond_type_counts=bond_type_counts,
    )


def _compare_bond_maps(
    original: dict[tuple[int, int], int],
    oracle: dict[tuple[int, int], int],
) -> dict[str, object]:
    """Compare original and oracle bond pair/type maps."""
    original_pairs = set(original)
    oracle_pairs = set(oracle)
    shared_pairs = original_pairs & oracle_pairs
    type_mismatch_count = sum(1 for pair in shared_pairs if original[pair] != oracle[pair])
    return {
        "bond_pair_set_match": int(original_pairs == oracle_pairs),
        "bond_type_map_match": int(original == oracle),
        "missing_bond_count": len(original_pairs - oracle_pairs),
        "extra_bond_count": len(oracle_pairs - original_pairs),
        "bond_type_mismatch_count": type_mismatch_count,
    }


def _count_vector_differences(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    """Count index-wise differences when vectors have the same length."""
    if len(left) != len(right):
        return max(len(left), len(right))
    return sum(int(a != b) for a, b in zip(left, right))


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    """Write rows to CSV with stable columns."""
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _rate(rows: list[dict[str, object]], key: str) -> float:
    """Return the mean of a 0/1 column."""
    if not rows:
        return 0.0
    return sum(int(row[key]) for row in rows) / len(rows)


def _plot_match_rates(path: Path, summary_rows: list[dict[str, object]]) -> None:
    """Plot global match rates."""
    labels = [row["metric"] for row in summary_rows if row["kind"] == "rate"]
    values = [float(row["value"]) for row in summary_rows if row["kind"] == "rate"]
    plt.figure(figsize=(9, 4.5))
    plt.bar(labels, values, color="#4C78A8")
    plt.ylabel("match rate")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_strategy_rates(path: Path, rows: list[dict[str, object]]) -> None:
    """Plot SMILES and bond-map match rates by repair strategy."""
    strategies = [row["repair_strategy"] for row in rows]
    smiles = [float(row["smiles_no_stereo_match_rate"]) for row in rows]
    bond_maps = [float(row["bond_type_map_match_rate"]) for row in rows]
    x_positions = list(range(len(strategies)))
    width = 0.38
    plt.figure(figsize=(8, 4.5))
    plt.bar([x - width / 2 for x in x_positions], smiles, width=width, label="SMILES no stereo")
    plt.bar([x + width / 2 for x in x_positions], bond_maps, width=width, label="bond type map")
    plt.xticks(x_positions, strategies, rotation=20, ha="right")
    plt.ylabel("match rate")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def run_audit(args: argparse.Namespace) -> Path:
    """Run the full original SDF vs repaired oracle audit."""
    dataset_info = constants.dataset_params[args.dataset]
    dataset = ProcessedLigandPocketDataset(args.npz_path, center=False)
    args.outdir.mkdir(parents=True, exist_ok=True)
    limit = len(dataset) if args.max_samples is None else min(args.max_samples, len(dataset))

    rows = []
    original_error_counts: Counter[str] = Counter()
    repair_strategy_counts: Counter[str] = Counter()
    for sample_idx in range(limit):
        sample = dataset[sample_idx]
        name = str(sample["names"])
        row: dict[str, object] = {
            "sample_idx": sample_idx,
            "name": name,
        }
        try:
            ligand_relpath = ligand_sdf_relpath_from_name(name)
            original_path = args.raw_ligand_root / ligand_relpath
            row["original_sdf"] = str(original_path)
        except ValueError as exc:
            original_path = None
            row["original_sdf"] = ""
            row["original_load_error"] = type(exc).__name__

        atom_symbols = _atom_symbols(sample["lig_one_hot"], dataset_info["atom_decoder"])
        oracle_result = build_oracle_mol(
            atom_symbols=atom_symbols,
            positions=sample["lig_coords"].float(),
            halfedge_index=sample["ligand_halfedge_index"],
            halfedge_type=sample["ligand_halfedge_type"],
        )
        row["oracle_sanitize_success"] = int(oracle_result.sanitize_success)
        row["initial_sanitize_error"] = oracle_result.initial_sanitize_error or ""
        row["oracle_sanitize_error"] = oracle_result.sanitize_error or ""
        row["repair_strategy"] = oracle_result.repair_strategy
        repair_strategy_counts[oracle_result.repair_strategy] += 1

        original_mol = None
        original_error = row.get("original_load_error", "")
        if original_path is not None:
            original_mol, original_error = _load_original_sdf(original_path)
        row["original_load_success"] = int(original_mol is not None)
        row["original_load_error"] = original_error or ""
        original_error_counts[original_error or "success"] += 1

        if original_mol is None or oracle_result.mol is None or not oracle_result.sanitize_success:
            row.update(_empty_comparison_fields())
            rows.append(row)
            continue

        original = _molecule_facts(original_mol)
        oracle = _molecule_facts(oracle_result.mol)
        bond_comparison = _compare_bond_maps(original.bond_map, oracle.bond_map)
        formal_charge_diff_count = _count_vector_differences(
            original.formal_charges, oracle.formal_charges
        )
        aromatic_atom_diff_count = _count_vector_differences(
            original.aromatic_atom_flags, oracle.aromatic_atom_flags
        )
        row.update({
            "original_smiles_iso": original.smiles_iso,
            "oracle_smiles_iso": oracle.smiles_iso,
            "original_smiles_no_stereo": original.smiles_no_stereo,
            "oracle_smiles_no_stereo": oracle.smiles_no_stereo,
            "smiles_iso_match": int(original.smiles_iso == oracle.smiles_iso),
            "smiles_no_stereo_match": int(original.smiles_no_stereo == oracle.smiles_no_stereo),
            "num_atoms_original": original.num_atoms,
            "num_atoms_oracle": oracle.num_atoms,
            "num_atoms_match": int(original.num_atoms == oracle.num_atoms),
            "num_bonds_original": original.num_bonds,
            "num_bonds_oracle": oracle.num_bonds,
            "num_bonds_match": int(original.num_bonds == oracle.num_bonds),
            "total_formal_charge_original": original.total_formal_charge,
            "total_formal_charge_oracle": oracle.total_formal_charge,
            "total_formal_charge_match": int(
                original.total_formal_charge == oracle.total_formal_charge
            ),
            "formal_charge_vector_match": int(original.formal_charges == oracle.formal_charges),
            "formal_charge_diff_count": formal_charge_diff_count,
            "aromatic_atom_count_original": original.aromatic_atom_count,
            "aromatic_atom_count_oracle": oracle.aromatic_atom_count,
            "aromatic_atom_count_match": int(
                original.aromatic_atom_count == oracle.aromatic_atom_count
            ),
            "aromatic_atom_vector_match": int(
                original.aromatic_atom_flags == oracle.aromatic_atom_flags
            ),
            "aromatic_atom_diff_count": aromatic_atom_diff_count,
            "aromatic_bond_count_original": original.aromatic_bond_count,
            "aromatic_bond_count_oracle": oracle.aromatic_bond_count,
            "aromatic_bond_count_match": int(
                original.aromatic_bond_count == oracle.aromatic_bond_count
            ),
            **bond_comparison,
        })
        rows.append(row)

    fieldnames = _audit_fieldnames()
    _write_csv(args.outdir / "oracle_original_sdf_audit.csv", rows, fieldnames)
    summary_rows = _build_summary_rows(rows, original_error_counts, repair_strategy_counts)
    _write_csv(args.outdir / "oracle_original_sdf_audit_summary.csv", summary_rows, [
        "kind", "metric", "value", "count", "denominator",
    ])
    strategy_rows = _build_strategy_rows(rows)
    _write_csv(args.outdir / "oracle_original_sdf_by_repair_strategy.csv", strategy_rows, [
        "repair_strategy", "count", "smiles_no_stereo_match_rate",
        "smiles_iso_match_rate", "charge_vector_match_rate",
        "aromatic_atom_vector_match_rate", "bond_type_map_match_rate",
    ])
    mismatch_rows = [row for row in rows if int(row["smiles_no_stereo_match"]) == 0]
    _write_csv(
        args.outdir / "oracle_original_sdf_smiles_mismatches.csv",
        mismatch_rows,
        fieldnames,
    )
    summary_json = {
        "npz_path": str(args.npz_path),
        "raw_ligand_root": str(args.raw_ligand_root),
        "dataset": args.dataset,
        "samples": limit,
        "summary": summary_rows,
        "by_repair_strategy": strategy_rows,
    }
    (args.outdir / "oracle_original_sdf_audit_summary.json").write_text(
        json.dumps(summary_json, indent=2, ensure_ascii=False)
    )
    _plot_match_rates(args.outdir / "oracle_original_sdf_match_rates.png", summary_rows)
    _plot_strategy_rates(args.outdir / "oracle_original_sdf_strategy_rates.png", strategy_rows)
    report_path = _write_report(args.outdir, args, limit, summary_rows, strategy_rows)
    return report_path


def _empty_comparison_fields() -> dict[str, object]:
    """Return default comparison values for failed original/oracle loads."""
    fields = {}
    for key in _comparison_fieldnames():
        fields[key] = "" if key.endswith("smiles_iso") or key.endswith("smiles_no_stereo") else 0
    return fields


def _audit_fieldnames() -> list[str]:
    """Return stable per-molecule audit columns."""
    return [
        "sample_idx", "name", "original_sdf", "original_load_success",
        "original_load_error", "oracle_sanitize_success", "initial_sanitize_error",
        "oracle_sanitize_error", "repair_strategy", *_comparison_fieldnames(),
    ]


def _comparison_fieldnames() -> list[str]:
    """Return stable comparison-only columns."""
    return [
        "original_smiles_iso", "oracle_smiles_iso",
        "original_smiles_no_stereo", "oracle_smiles_no_stereo",
        "smiles_iso_match", "smiles_no_stereo_match",
        "num_atoms_original", "num_atoms_oracle", "num_atoms_match",
        "num_bonds_original", "num_bonds_oracle", "num_bonds_match",
        "total_formal_charge_original", "total_formal_charge_oracle",
        "total_formal_charge_match", "formal_charge_vector_match",
        "formal_charge_diff_count", "aromatic_atom_count_original",
        "aromatic_atom_count_oracle", "aromatic_atom_count_match",
        "aromatic_atom_vector_match", "aromatic_atom_diff_count",
        "aromatic_bond_count_original", "aromatic_bond_count_oracle",
        "aromatic_bond_count_match", "bond_pair_set_match",
        "bond_type_map_match", "missing_bond_count", "extra_bond_count",
        "bond_type_mismatch_count",
    ]


def _build_summary_rows(
    rows: list[dict[str, object]],
    original_error_counts: Counter[str],
    repair_strategy_counts: Counter[str],
) -> list[dict[str, object]]:
    """Build global summary rows."""
    denominator = len(rows)
    summary_rows: list[dict[str, object]] = []
    for metric in [
        "original_load_success", "oracle_sanitize_success",
        "smiles_no_stereo_match", "smiles_iso_match", "num_atoms_match",
        "num_bonds_match", "total_formal_charge_match",
        "formal_charge_vector_match", "aromatic_atom_count_match",
        "aromatic_atom_vector_match", "aromatic_bond_count_match",
        "bond_pair_set_match", "bond_type_map_match",
    ]:
        count = sum(int(row[metric]) for row in rows)
        summary_rows.append({
            "kind": "rate",
            "metric": metric,
            "value": f"{count / max(1, denominator):.8f}",
            "count": count,
            "denominator": denominator,
        })
    for error_name, count in sorted(original_error_counts.items()):
        summary_rows.append({
            "kind": "original_load_error",
            "metric": error_name,
            "value": f"{count / max(1, denominator):.8f}",
            "count": count,
            "denominator": denominator,
        })
    for strategy, count in sorted(repair_strategy_counts.items()):
        summary_rows.append({
            "kind": "repair_strategy",
            "metric": strategy,
            "value": f"{count / max(1, denominator):.8f}",
            "count": count,
            "denominator": denominator,
        })
    return summary_rows


def _build_strategy_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Build grouped match rates by repair strategy."""
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["repair_strategy"])].append(row)
    strategy_rows = []
    for strategy, group_rows in sorted(grouped.items()):
        strategy_rows.append({
            "repair_strategy": strategy,
            "count": len(group_rows),
            "smiles_no_stereo_match_rate": f"{_rate(group_rows, 'smiles_no_stereo_match'):.8f}",
            "smiles_iso_match_rate": f"{_rate(group_rows, 'smiles_iso_match'):.8f}",
            "charge_vector_match_rate": f"{_rate(group_rows, 'formal_charge_vector_match'):.8f}",
            "aromatic_atom_vector_match_rate": f"{_rate(group_rows, 'aromatic_atom_vector_match'):.8f}",
            "bond_type_map_match_rate": f"{_rate(group_rows, 'bond_type_map_match'):.8f}",
        })
    return strategy_rows


def _write_report(
    outdir: Path,
    args: argparse.Namespace,
    limit: int,
    summary_rows: list[dict[str, object]],
    strategy_rows: list[dict[str, object]],
) -> Path:
    """Write the Markdown audit report."""
    report_path = outdir / "oracle_original_sdf_audit_report.md"
    rate_rows = [row for row in summary_rows if row["kind"] == "rate"]
    rate_by_metric = {str(row["metric"]): float(row["value"]) for row in rate_rows}
    with report_path.open("w") as handle:
        handle.write("# Oracle Original SDF Audit Report\n\n")
        handle.write(f"- npz_path: `{args.npz_path}`\n")
        handle.write(f"- raw_ligand_root: `{args.raw_ligand_root}`\n")
        handle.write(f"- dataset: `{args.dataset}`\n")
        handle.write(f"- samples: {limit}\n\n")
        handle.write("## Global Match Rates\n\n")
        handle.write("| metric | count | denominator | rate |\n")
        handle.write("|---|---:|---:|---:|\n")
        for row in rate_rows:
            handle.write(
                f"| {row['metric']} | {row['count']} | {row['denominator']} | {row['value']} |\n"
            )
        handle.write("\n## By Repair Strategy\n\n")
        handle.write(
            "| repair_strategy | count | no-stereo SMILES | iso SMILES | charge vector | "
            "aromatic atom vector | bond type map |\n"
        )
        handle.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for row in strategy_rows:
            handle.write(
                f"| {row['repair_strategy']} | {row['count']} | "
                f"{row['smiles_no_stereo_match_rate']} | {row['smiles_iso_match_rate']} | "
                f"{row['charge_vector_match_rate']} | {row['aromatic_atom_vector_match_rate']} | "
                f"{row['bond_type_map_match_rate']} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "- Atom counts, bond counts, and bond pair sets matching at 100% means the processed "
            "halfedge connectivity is aligned with the original SDF atom order.\n"
        )
        if rate_by_metric.get("smiles_no_stereo_match", 0.0) < 0.999:
            handle.write(
                "- No-stereo SMILES is not fully preserved. The repaired oracle decoder should "
                "not be treated as chemically faithful for all molecules.\n"
            )
        if rate_by_metric.get("bond_type_map_match", 0.0) < 0.999:
            handle.write(
                "- Bond type/aromaticity mismatch indicates that some RDKit repair paths sanitize "
                "the molecule by changing aromatic/bond semantics.\n"
            )
        if rate_by_metric.get("formal_charge_vector_match", 0.0) < 0.999:
            handle.write(
                "- Formal charge mismatch indicates that charges should be read from original SDF "
                "metadata rather than inferred only during repair.\n"
            )
        handle.write(
            "- Next engineering step: store original formal charges/aromatic atom flags or reuse "
            "original SDF-derived molecule semantics in the edge bundle before training a faithful "
            "edge decoder.\n"
        )
        handle.write("\n![match rates](oracle_original_sdf_match_rates.png)\n\n")
        handle.write("![strategy rates](oracle_original_sdf_strategy_rates.png)\n\n")
        handle.write("## Files\n\n")
        handle.write("- [oracle_original_sdf_audit.csv](oracle_original_sdf_audit.csv)\n")
        handle.write("- [oracle_original_sdf_smiles_mismatches.csv](oracle_original_sdf_smiles_mismatches.csv)\n")
        handle.write("- [oracle_original_sdf_by_repair_strategy.csv](oracle_original_sdf_by_repair_strategy.csv)\n")
        handle.write("- [oracle_original_sdf_audit_summary.csv](oracle_original_sdf_audit_summary.csv)\n")
        handle.write("- [oracle_original_sdf_audit_summary.json](oracle_original_sdf_audit_summary.json)\n")
    return report_path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--npz-path",
        type=Path,
        default=Path("data/crossdocked_raw/processed_edge_noH_full/train.npz"),
    )
    parser.add_argument(
        "--raw-ligand-root",
        type=Path,
        default=Path("data/crossdocked_raw/crossdocked_pocket10"),
    )
    parser.add_argument("--dataset", choices=sorted(constants.dataset_params.keys()), default="crossdock_full")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("analysis") / f"oracle_original_sdf_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    report = run_audit(parse_args())
    print(f"Wrote oracle original SDF audit report to {report}")
