"""Audit semantic-aware oracle reconstruction against original ligand SDF files."""

from __future__ import annotations

import argparse
import csv
import json
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
import numpy as np
import torch
from rdkit import Chem, RDLogger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIFFSBDD_ROOT = Path(__file__).resolve().parents[1]
for extra in (PROJECT_ROOT, DIFFSBDD_ROOT):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import constants
from analysis.oracle_original_sdf_audit import (
    _atom_symbols,
    _compare_bond_maps,
    _count_vector_differences,
    _load_original_sdf,
    _molecule_facts,
    ligand_sdf_relpath_from_name,
)
from dataset import ProcessedLigandPocketDataset
from edge_diffusion import build_oracle_mol, build_semantic_oracle_mol

RDLogger.DisableLog("rdApp.*")

COMPARISON_METRICS = [
    "smiles_no_stereo_match",
    "smiles_iso_match",
    "formal_charge_vector_match",
    "aromatic_atom_vector_match",
    "bond_type_map_match",
]


def run_semantic_oracle_audit(args: argparse.Namespace) -> Path:
    """Compare default and semantic-aware oracle reconstruction."""
    dataset_info = constants.dataset_params[args.dataset]
    dataset = ProcessedLigandPocketDataset(args.npz_path, center=False)
    semantic_data = np.load(args.semantic_npz, allow_pickle=True)
    args.outdir.mkdir(parents=True, exist_ok=True)
    limit = len(dataset) if args.max_samples is None else min(args.max_samples, len(dataset))

    rows: list[dict[str, object]] = []
    default_strategy_counts: Counter[str] = Counter()
    semantic_strategy_counts: Counter[str] = Counter()
    original_error_counts: Counter[str] = Counter()
    name_mismatch_count = 0

    for sample_idx in range(limit):
        sample = dataset[sample_idx]
        name = str(sample["names"])
        semantic_name = str(semantic_data["names"][sample_idx])
        name_match = int(name == semantic_name)
        name_mismatch_count += int(not name_match)
        original_path = _original_path_for_sample(
            name=name,
            semantic_data=semantic_data,
            sample_idx=sample_idx,
            raw_ligand_root=args.raw_ligand_root,
        )
        original_mol, original_error = _load_original_sdf(original_path)
        original_error_counts[original_error or "success"] += 1

        atom_symbols = _atom_symbols(sample["lig_one_hot"], dataset_info["atom_decoder"])
        default_result = build_oracle_mol(
            atom_symbols=atom_symbols,
            positions=sample["lig_coords"].float(),
            halfedge_index=sample["ligand_halfedge_index"],
            halfedge_type=sample["ligand_halfedge_type"],
        )
        semantic_result = build_semantic_oracle_mol(
            atom_symbols=atom_symbols,
            positions=sample["lig_coords"].float(),
            halfedge_index=sample["ligand_halfedge_index"],
            halfedge_type=sample["ligand_halfedge_type"],
            formal_charges=semantic_data["original_formal_charge"][sample_idx],
            total_hs=semantic_data["original_total_h"][sample_idx],
            explicit_hs=semantic_data["original_explicit_h"][sample_idx],
            no_implicit_flags=semantic_data["original_no_implicit"][sample_idx],
            aromatic_atom_flags=semantic_data["original_aromatic_atom"][sample_idx],
            chiral_tags=semantic_data["original_chiral_tag"][sample_idx],
            bond_pairs=semantic_data["original_bond_pair"][sample_idx],
            bond_stereo=semantic_data["original_bond_stereo"][sample_idx],
        )
        default_strategy_counts[default_result.repair_strategy] += 1
        semantic_strategy_counts[semantic_result.repair_strategy] += 1

        row: dict[str, object] = {
            "sample_idx": sample_idx,
            "name": name,
            "semantic_name_match": name_match,
            "original_sdf": str(original_path),
            "original_load_success": int(original_mol is not None),
            "original_load_error": original_error or "",
            "default_sanitize_success": int(default_result.sanitize_success),
            "default_initial_sanitize_error": default_result.initial_sanitize_error or "",
            "default_sanitize_error": default_result.sanitize_error or "",
            "default_repair_strategy": default_result.repair_strategy,
            "semantic_sanitize_success": int(semantic_result.sanitize_success),
            "semantic_initial_sanitize_error": semantic_result.initial_sanitize_error or "",
            "semantic_sanitize_error": semantic_result.sanitize_error or "",
            "semantic_repair_strategy": semantic_result.repair_strategy,
        }
        if original_mol is None:
            row.update(_empty_candidate_fields("default"))
            row.update(_empty_candidate_fields("semantic"))
            rows.append(row)
            continue

        original_facts = _molecule_facts(original_mol)
        row.update(_candidate_fields(
            "default",
            original_facts,
            default_result.mol if default_result.sanitize_success else None,
        ))
        row.update(_candidate_fields(
            "semantic",
            original_facts,
            semantic_result.mol if semantic_result.sanitize_success else None,
        ))
        for metric in COMPARISON_METRICS:
            row[f"{metric}_delta"] = (
                int(row[f"semantic_{metric}"]) - int(row[f"default_{metric}"])
            )
        rows.append(row)

    _write_csv(args.outdir / "semantic_oracle_reconstruction_audit.csv", rows, _audit_fieldnames())
    improved_rows = [
        row for row in rows
        if int(row["smiles_no_stereo_match_delta"]) > 0
    ]
    regressed_rows = [
        row for row in rows
        if int(row["smiles_no_stereo_match_delta"]) < 0
    ]
    _write_csv(
        args.outdir / "semantic_oracle_smiles_improvements.csv",
        improved_rows,
        _audit_fieldnames(),
    )
    _write_csv(
        args.outdir / "semantic_oracle_smiles_regressions.csv",
        regressed_rows,
        _audit_fieldnames(),
    )
    summary = _build_summary(
        args=args,
        rows=rows,
        default_strategy_counts=default_strategy_counts,
        semantic_strategy_counts=semantic_strategy_counts,
        original_error_counts=original_error_counts,
        name_mismatch_count=name_mismatch_count,
    )
    (args.outdir / "semantic_oracle_reconstruction_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    _write_summary_csv(args.outdir / "semantic_oracle_reconstruction_summary.csv", summary)
    _plot_match_rates(args.outdir / "semantic_oracle_match_rates.png", summary)
    _plot_strategy_counts(args.outdir / "semantic_oracle_repair_strategies.png", summary)
    return _write_report(args.outdir, summary)


def _original_path_for_sample(
    name: str,
    semantic_data: np.lib.npyio.NpzFile,
    sample_idx: int,
    raw_ligand_root: Path,
) -> Path:
    """Return original SDF path from sidecar or processed sample name."""
    sidecar_path = str(semantic_data["original_sdf"][sample_idx])
    if sidecar_path:
        return Path(sidecar_path)
    return raw_ligand_root / ligand_sdf_relpath_from_name(name)


def _candidate_fields(
    prefix: str,
    original_facts,
    candidate_mol: Chem.Mol | None,
) -> dict[str, object]:
    """Build comparison fields for one reconstructed candidate."""
    if candidate_mol is None:
        return _empty_candidate_fields(prefix)
    candidate_facts = _molecule_facts(candidate_mol)
    bond_comparison = _compare_bond_maps(original_facts.bond_map, candidate_facts.bond_map)
    formal_charge_diff_count = _count_vector_differences(
        original_facts.formal_charges,
        candidate_facts.formal_charges,
    )
    aromatic_atom_diff_count = _count_vector_differences(
        original_facts.aromatic_atom_flags,
        candidate_facts.aromatic_atom_flags,
    )
    return {
        f"{prefix}_smiles_iso": candidate_facts.smiles_iso,
        f"{prefix}_smiles_no_stereo": candidate_facts.smiles_no_stereo,
        f"{prefix}_smiles_iso_match": int(
            original_facts.smiles_iso == candidate_facts.smiles_iso
        ),
        f"{prefix}_smiles_no_stereo_match": int(
            original_facts.smiles_no_stereo == candidate_facts.smiles_no_stereo
        ),
        f"{prefix}_formal_charge_vector_match": int(
            original_facts.formal_charges == candidate_facts.formal_charges
        ),
        f"{prefix}_formal_charge_diff_count": formal_charge_diff_count,
        f"{prefix}_aromatic_atom_vector_match": int(
            original_facts.aromatic_atom_flags == candidate_facts.aromatic_atom_flags
        ),
        f"{prefix}_aromatic_atom_diff_count": aromatic_atom_diff_count,
        f"{prefix}_bond_type_map_match": int(bond_comparison["bond_type_map_match"]),
        f"{prefix}_bond_pair_set_match": int(bond_comparison["bond_pair_set_match"]),
        f"{prefix}_bond_type_mismatch_count": int(
            bond_comparison["bond_type_mismatch_count"]
        ),
    }


def _empty_candidate_fields(prefix: str) -> dict[str, object]:
    """Return empty comparison fields for missing candidates."""
    return {
        f"{prefix}_smiles_iso": "",
        f"{prefix}_smiles_no_stereo": "",
        f"{prefix}_smiles_iso_match": 0,
        f"{prefix}_smiles_no_stereo_match": 0,
        f"{prefix}_formal_charge_vector_match": 0,
        f"{prefix}_formal_charge_diff_count": 0,
        f"{prefix}_aromatic_atom_vector_match": 0,
        f"{prefix}_aromatic_atom_diff_count": 0,
        f"{prefix}_bond_type_map_match": 0,
        f"{prefix}_bond_pair_set_match": 0,
        f"{prefix}_bond_type_mismatch_count": 0,
    }


def _audit_fieldnames() -> list[str]:
    """Return stable per-molecule audit columns."""
    base = [
        "sample_idx", "name", "semantic_name_match", "original_sdf",
        "original_load_success", "original_load_error",
        "default_sanitize_success", "default_initial_sanitize_error",
        "default_sanitize_error", "default_repair_strategy",
        "semantic_sanitize_success", "semantic_initial_sanitize_error",
        "semantic_sanitize_error", "semantic_repair_strategy",
    ]
    comparison = []
    for prefix in ["default", "semantic"]:
        comparison.extend([
            f"{prefix}_smiles_iso", f"{prefix}_smiles_no_stereo",
            f"{prefix}_smiles_iso_match", f"{prefix}_smiles_no_stereo_match",
            f"{prefix}_formal_charge_vector_match", f"{prefix}_formal_charge_diff_count",
            f"{prefix}_aromatic_atom_vector_match", f"{prefix}_aromatic_atom_diff_count",
            f"{prefix}_bond_type_map_match", f"{prefix}_bond_pair_set_match",
            f"{prefix}_bond_type_mismatch_count",
        ])
    deltas = [f"{metric}_delta" for metric in COMPARISON_METRICS]
    return [*base, *comparison, *deltas]


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    """Write rows to CSV."""
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _build_summary(
    args: argparse.Namespace,
    rows: list[dict[str, object]],
    default_strategy_counts: Counter[str],
    semantic_strategy_counts: Counter[str],
    original_error_counts: Counter[str],
    name_mismatch_count: int,
) -> dict[str, object]:
    """Build JSON-serializable summary."""
    denominator = len(rows)
    match_rates: dict[str, dict[str, float | int]] = {}
    for metric in COMPARISON_METRICS:
        default_count = sum(int(row[f"default_{metric}"]) for row in rows)
        semantic_count = sum(int(row[f"semantic_{metric}"]) for row in rows)
        match_rates[metric] = {
            "default_count": default_count,
            "semantic_count": semantic_count,
            "denominator": denominator,
            "default_rate": default_count / max(1, denominator),
            "semantic_rate": semantic_count / max(1, denominator),
            "delta_count": semantic_count - default_count,
            "delta_rate": (semantic_count - default_count) / max(1, denominator),
        }
    return {
        "npz_path": str(args.npz_path),
        "semantic_npz": str(args.semantic_npz),
        "raw_ligand_root": str(args.raw_ligand_root),
        "dataset": args.dataset,
        "samples": denominator,
        "name_mismatch_count": name_mismatch_count,
        "original_error_counts": dict(sorted(original_error_counts.items())),
        "match_rates": match_rates,
        "default_strategy_counts": dict(sorted(default_strategy_counts.items())),
        "semantic_strategy_counts": dict(sorted(semantic_strategy_counts.items())),
        "smiles_no_stereo_improvements": sum(
            int(row["smiles_no_stereo_match_delta"] > 0) for row in rows
        ),
        "smiles_no_stereo_regressions": sum(
            int(row["smiles_no_stereo_match_delta"] < 0) for row in rows
        ),
    }


def _write_summary_csv(path: Path, summary: dict[str, object]) -> None:
    """Write compact summary CSV."""
    rows = []
    for metric, values in summary["match_rates"].items():
        row = {"metric": metric}
        row.update(values)
        rows.append(row)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "metric", "default_count", "semantic_count", "denominator",
                "default_rate", "semantic_rate", "delta_count", "delta_rate",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _plot_match_rates(path: Path, summary: dict[str, object]) -> None:
    """Plot default vs semantic-aware match rates."""
    metrics = list(summary["match_rates"])
    default_values = [
        float(summary["match_rates"][metric]["default_rate"])
        for metric in metrics
    ]
    semantic_values = [
        float(summary["match_rates"][metric]["semantic_rate"])
        for metric in metrics
    ]
    x_positions = list(range(len(metrics)))
    width = 0.38
    plt.figure(figsize=(9, 4.8))
    plt.bar([x - width / 2 for x in x_positions], default_values, width, label="default")
    plt.bar([x + width / 2 for x in x_positions], semantic_values, width, label="semantic")
    plt.xticks(x_positions, metrics, rotation=25, ha="right")
    plt.ylabel("match rate")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_strategy_counts(path: Path, summary: dict[str, object]) -> None:
    """Plot default vs semantic-aware repair strategy counts."""
    default_counts = summary["default_strategy_counts"]
    semantic_counts = summary["semantic_strategy_counts"]
    strategies = sorted(set(default_counts) | set(semantic_counts))
    x_positions = list(range(len(strategies)))
    width = 0.38
    plt.figure(figsize=(8, 4.5))
    plt.bar(
        [x - width / 2 for x in x_positions],
        [default_counts.get(strategy, 0) for strategy in strategies],
        width,
        label="default",
    )
    plt.bar(
        [x + width / 2 for x in x_positions],
        [semantic_counts.get(strategy, 0) for strategy in strategies],
        width,
        label="semantic",
    )
    plt.xticks(x_positions, strategies, rotation=20, ha="right")
    plt.ylabel("molecule count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _write_report(outdir: Path, summary: dict[str, object]) -> Path:
    """Write Markdown audit report."""
    report_path = outdir / "semantic_oracle_reconstruction_audit_report.md"
    with report_path.open("w") as handle:
        handle.write("# Semantic Oracle Reconstruction Audit\n\n")
        handle.write(f"- npz_path: `{summary['npz_path']}`\n")
        handle.write(f"- semantic_npz: `{summary['semantic_npz']}`\n")
        handle.write(f"- raw_ligand_root: `{summary['raw_ligand_root']}`\n")
        handle.write(f"- dataset: `{summary['dataset']}`\n")
        handle.write(f"- samples: {summary['samples']}\n")
        handle.write(f"- name_mismatch_count: {summary['name_mismatch_count']}\n")
        handle.write(
            f"- smiles_no_stereo_improvements: {summary['smiles_no_stereo_improvements']}\n"
        )
        handle.write(
            f"- smiles_no_stereo_regressions: {summary['smiles_no_stereo_regressions']}\n\n"
        )
        handle.write("## Match Rates\n\n")
        handle.write(
            "| metric | default count | semantic count | denominator | default rate | "
            "semantic rate | delta |\n"
        )
        handle.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for metric, values in summary["match_rates"].items():
            handle.write(
                f"| {metric} | {values['default_count']} | {values['semantic_count']} | "
                f"{values['denominator']} | {values['default_rate']:.8f} | "
                f"{values['semantic_rate']:.8f} | {values['delta_count']} |\n"
            )
        handle.write("\n## Default Repair Strategies\n\n")
        handle.write("| strategy | count |\n")
        handle.write("|---|---:|\n")
        for strategy, count in summary["default_strategy_counts"].items():
            handle.write(f"| {strategy} | {count} |\n")
        handle.write("\n## Semantic-Aware Repair Strategies\n\n")
        handle.write("| strategy | count |\n")
        handle.write("|---|---:|\n")
        for strategy, count in summary["semantic_strategy_counts"].items():
            handle.write(f"| {strategy} | {count} |\n")
        handle.write("\n## Files\n\n")
        handle.write("- [semantic_oracle_reconstruction_audit.csv](semantic_oracle_reconstruction_audit.csv)\n")
        handle.write("- [semantic_oracle_smiles_improvements.csv](semantic_oracle_smiles_improvements.csv)\n")
        handle.write("- [semantic_oracle_smiles_regressions.csv](semantic_oracle_smiles_regressions.csv)\n")
        handle.write("- [semantic_oracle_reconstruction_summary.csv](semantic_oracle_reconstruction_summary.csv)\n")
        handle.write("- [semantic_oracle_reconstruction_summary.json](semantic_oracle_reconstruction_summary.json)\n")
        handle.write("\n![match rates](semantic_oracle_match_rates.png)\n\n")
        handle.write("![repair strategies](semantic_oracle_repair_strategies.png)\n")
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
        "--semantic-npz",
        type=Path,
        default=Path("analysis/edge_semantic_bundle_20260602_n1000/train_edge_semantic.npz"),
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
        default=Path("analysis")
        / f"semantic_oracle_reconstruction_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument("--max-samples", type=int, default=1000)
    return parser.parse_args()


if __name__ == "__main__":
    report = run_semantic_oracle_audit(parse_args())
    print(f"Wrote semantic oracle reconstruction audit report to {report}")
