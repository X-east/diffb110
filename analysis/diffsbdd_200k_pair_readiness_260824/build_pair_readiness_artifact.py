#!/usr/bin/env python3
"""Build a small, reproducible audit of the 200k collection and current E2PO pilot."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, QED, Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold


ROOT = Path(__file__).resolve().parents[2]
COLLECTION = (
    ROOT
    / "runs"
    / "diffsbdd_collect"
    / "diffsbdd-original-t1000-8gpu-100k-trajectory_20260730_194643"
)
BASELINE_SDF = (
    ROOT
    / "analysis"
    / "diffsbdd_filtered_bc_rank40_allpreferred_8gpu_100epoch_monitor_260822"
    / "baseline_original"
    / "generated_1000_attempts.sdf"
)
E2PO_ROOT = ROOT / "analysis" / "diffsbdd_global_e2po_t500_v1" / "formal_run_gpu5" / "generation"
OUT = Path(__file__).resolve().parent

RDLogger.DisableLog("rdApp.*")


def _load_json(path: Path) -> dict[str, Any]:
    """Load one JSON evidence file."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _collection_rows() -> list[dict[str, Any]]:
    """Return cumulative collection counts and the derived second-100k increment."""
    original = _load_json(COLLECTION / "summary_100k_original.json")
    cumulative = _load_json(COLLECTION / "summary.json")
    rows: list[dict[str, Any]] = []
    for label, summary in (("original_100k", original), ("cumulative_200k", cumulative)):
        rows.append(
            {
                "stage": label,
                "fixed_attempts": int(summary["expected_fixed_attempts"]),
                "completed_shards": int(summary["completed_shards_total"]),
                "rdkit_valid": int(summary["rdkit_valid_total"]),
                "invalid_or_nonvalid": int(summary["expected_fixed_attempts"] - summary["rdkit_valid_total"]),
                "rdkit_valid_rate": float(summary["rdkit_valid_total"] / summary["expected_fixed_attempts"]),
                "trajectory_bytes": int(summary["shard_bytes_total"]),
                "checkpoint_sha256": str(summary["checkpoint_sha256"]),
                "num_timesteps": int(summary["num_timesteps"]),
                "world_size": int(summary["world_size"]),
            }
        )
    rows.append(
        {
            "stage": "derived_added_after_original",
            "fixed_attempts": rows[1]["fixed_attempts"] - rows[0]["fixed_attempts"],
            "completed_shards": rows[1]["completed_shards"] - rows[0]["completed_shards"],
            "rdkit_valid": rows[1]["rdkit_valid"] - rows[0]["rdkit_valid"],
            "invalid_or_nonvalid": rows[1]["invalid_or_nonvalid"] - rows[0]["invalid_or_nonvalid"],
            "rdkit_valid_rate": (rows[1]["rdkit_valid"] - rows[0]["rdkit_valid"])
            / (rows[1]["fixed_attempts"] - rows[0]["fixed_attempts"]),
            "trajectory_bytes": rows[1]["trajectory_bytes"] - rows[0]["trajectory_bytes"],
            "checkpoint_sha256": rows[1]["checkpoint_sha256"],
            "num_timesteps": rows[1]["num_timesteps"],
            "world_size": rows[1]["world_size"],
        }
    )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a list of flat dictionaries as CSV."""
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sdf_summary(label: str, path: Path) -> dict[str, Any]:
    """Recompute fixed-attempt quality descriptors for one SDF."""
    supplier = Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False, strictParsing=False)
    total = 0
    valid = 0
    smiles: set[str] = set()
    scaffolds: set[str] = set()
    values: dict[str, list[float]] = {
        "carbon_fraction": [],
        "hetero_atoms": [],
        "logp": [],
        "tpsa": [],
        "qed": [],
    }
    for mol in supplier:
        total += 1
        if mol is None:
            continue
        try:
            Chem.SanitizeMol(mol)
            valid += 1
            canonical = Chem.MolToSmiles(mol, canonical=True)
            smiles.add(canonical)
            scaffolds.add(MurckoScaffold.MurckoScaffoldSmiles(mol=mol))
            heavy = mol.GetNumHeavyAtoms()
            carbon = sum(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms())
            values["carbon_fraction"].append(carbon / heavy if heavy else 0.0)
            values["hetero_atoms"].append(float(rdMolDescriptors.CalcNumHeteroatoms(mol)))
            values["logp"].append(float(Crippen.MolLogP(mol)))
            values["tpsa"].append(float(rdMolDescriptors.CalcTPSA(mol)))
            values["qed"].append(float(QED.qed(mol)))
        except Exception:
            continue
    row: dict[str, Any] = {
        "source": label,
        "path": str(path),
        "attempts": 1000,
        "sdf_records": total,
        "sanitize_valid": valid,
        "valid_rate": valid / 1000,
        "unique_smiles": len(smiles),
        "murcko_scaffolds": len(scaffolds),
    }
    for key, items in values.items():
        row[key] = sum(items) / len(items) if items else float("nan")
    return row


def _plot_collection(rows: list[dict[str, Any]]) -> None:
    """Plot collection yield using fixed-attempt denominators."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    labels = [row["stage"].replace("_", "\n") for row in rows]
    rates = [100.0 * row["rdkit_valid_rate"] for row in rows]
    valid = [row["rdkit_valid"] for row in rows]
    axes[0].bar(labels, rates, color=["#4c78a8", "#59a14f", "#f28e2b"])
    axes[0].set_ylabel("RDKit-valid rate (%)")
    axes[0].set_ylim(0, 80)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(labels, valid, color=["#4c78a8", "#59a14f", "#f28e2b"])
    axes[1].set_ylabel("RDKit-valid endpoints")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("DiffSBDD fixed-attempt collection audit", fontsize=14)
    fig.savefig(OUT / "collection_yield.png", dpi=180)
    fig.savefig(OUT / "collection_yield.svg")
    plt.close(fig)


def _plot_e2po(rows: list[dict[str, Any]]) -> None:
    """Plot current T=500 fixed-soft-label E2PO quality drift."""
    labels = [row["source"] for row in rows]
    x = list(range(len(rows)))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].plot(x, [row["valid_rate"] for row in rows], marker="o", label="valid rate")
    axes[0].plot(x, [row["murcko_scaffolds"] / 1000 for row in rows], marker="s", label="scaffolds / 1000")
    axes[0].set_ylabel("Rate or scaffold count / 1000")
    axes[0].set_ylim(bottom=0)
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25)
    for key, marker in (("carbon_fraction", "o"), ("qed", "s")):
        axes[1].plot(x, [row[key] for row in rows], marker=marker, label=key)
    axes[1].set_ylabel("Mean descriptor")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25)
    axes[0].set_xticks(x, labels, rotation=35, ha="right")
    axes[1].set_xticks(x, labels, rotation=35, ha="right")
    fig.suptitle("Current T=500 fixed-soft-label E2PO: quality drift", fontsize=14)
    fig.savefig(OUT / "current_e2po_quality_drift.png", dpi=180)
    fig.savefig(OUT / "current_e2po_quality_drift.svg")
    plt.close(fig)


def main() -> None:
    """Build CSV and plot artifacts."""
    collection_rows = _collection_rows()
    _write_csv(OUT / "collection_summary.csv", collection_rows)
    _plot_collection(collection_rows)

    quality_rows = [_sdf_summary("Original", BASELINE_SDF)]
    for epoch in range(1, 6):
        quality_rows.append(_sdf_summary(f"E2PO_epoch_{epoch}", E2PO_ROOT / f"epoch_{epoch:04d}" / "molecules.sdf"))
    _write_csv(OUT / "current_e2po_quality.csv", quality_rows)
    _plot_e2po(quality_rows)

    manifest = {
        "collection_run": str(COLLECTION),
        "collection_completion_report": str(COLLECTION / "completion_report.json"),
        "e2po_run": str(E2PO_ROOT.parent),
        "baseline_sdf": str(BASELINE_SDF),
        "interpretation": "descriptive audit; no affinity or activity claim",
    }
    with (OUT / "artifact_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
