#!/usr/bin/env python3
"""Quantify current DiffSBDD E2PO endpoint and generation quality."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
TRAINING_SIDECAR = ROOT / "analysis/offline_iql_labels_alidiff_unidock_affinity_8gpu_full_260824/reward_sidecar.csv"
GENERATION_METRICS = ROOT / "analysis/diffsbdd_global_e2po_good_bad_xv_smoke_260825/generation_molecule_metrics.csv"
DOCKING_METRICS = ROOT / "analysis/diffsbdd_global_e2po_good_bad_xv_smoke_260825/generation_docking_metrics.csv"


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return one numeric column with invalid values represented as NaN."""
    return pd.to_numeric(frame[column], errors="coerce")


def boolean(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return one tolerant boolean column."""
    return frame[column].astype(str).str.lower().isin({"1", "true", "yes"})


def rate(count: int, denominator: int) -> float:
    """Return a percentage value for one fixed denominator."""
    return float(100.0 * count / denominator) if denominator else float("nan")


def add_metric(rows: list[dict[str, object]], name: str, count: int, denominator: int, scope: str) -> None:
    """Append one count/rate row."""
    rows.append(
        {
            "metric": name,
            "count": int(count),
            "denominator": int(denominator),
            "rate_percent": rate(int(count), int(denominator)),
            "scope": scope,
        }
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a table and keep column ordering stable."""
    pd.DataFrame(rows).to_csv(path, index=False)


def training_input_tables(sidecar: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Build training-input validity, failure, and known-risk summaries."""
    rows: list[dict[str, object]] = []
    total = len(sidecar)
    for column in (
        "reconstruction_success",
        "sanitize_success",
        "rdkit_valid",
        "docking_input_success",
        "docking_success",
        "pb_evaluated",
        "pb_valid",
    ):
        add_metric(rows, column, int(boolean(sidecar, column).sum()), total, "all_200k_endpoints")

    invalid = sidecar[~boolean(sidecar, "rdkit_valid")]
    failure_rows: list[dict[str, object]] = []
    for reason, group in invalid.groupby(invalid["failure_reason"].fillna("<NA>"), sort=False):
        failure_rows.append(
            {
                "failure_reason": str(reason),
                "count": int(len(group)),
                "rate_all_endpoints_percent": rate(len(group), total),
                "rate_invalid_endpoints_percent": rate(len(group), len(invalid)),
            }
        )

    valid = sidecar[boolean(sidecar, "rdkit_valid")].copy()
    valid_rows: list[dict[str, object]] = []
    valid_risks = {
        "three_member_ring_count>0": numeric(valid, "three_member_ring_count") > 0,
        "fused_ring_count>0": numeric(valid, "fused_ring_count") > 0,
        "complex_ring_count>0": numeric(valid, "complex_ring_count") > 0,
        "brenk_count>0": numeric(valid, "brenk_count") > 0,
        "carbon_fraction>0.85": numeric(valid, "carbon_fraction") > 0.85,
        "docking_raw>=-5": numeric(valid, "docking_raw") >= -5,
        "docking_raw>=-8": numeric(valid, "docking_raw") >= -8,
    }
    for name, mask in valid_risks.items():
        add_metric(valid_rows, name, int(mask.fillna(False).sum()), len(valid), "rdkit_valid_training_endpoints")
    ring_union = (
        valid_risks["three_member_ring_count>0"]
        | valid_risks["fused_ring_count>0"]
        | valid_risks["complex_ring_count>0"]
    ).fillna(False)
    add_metric(valid_rows, "ring_or_complex_union", int(ring_union.sum()), len(valid), "rdkit_valid_training_endpoints")
    add_metric(
        valid_rows,
        "ring_or_complex_or_carbon_union",
        int((ring_union | valid_risks["carbon_fraction>0.85"]).sum()),
        len(valid),
        "rdkit_valid_training_endpoints",
    )

    descriptor_rows: list[dict[str, object]] = []
    for column in (
        "carbon_fraction",
        "hetero_atom_count",
        "ring_count",
        "brenk_count",
        "docking_raw",
        "ligand_efficiency",
        "qed",
        "sa_raw",
        "logp",
        "tpsa",
    ):
        values = numeric(valid, column).dropna()
        descriptor_rows.append(
            {
                "metric": column,
                "n": int(len(values)),
                "mean": float(values.mean()),
                "median": float(values.median()),
                "p10": float(values.quantile(0.10)),
                "p90": float(values.quantile(0.90)),
            }
        )

    facts = {
        "total_endpoints": total,
        "invalid_endpoints": int(len(invalid)),
        "rdkit_valid_endpoints": int(len(valid)),
        "invalid_failure_counts": {str(row["failure_reason"]): int(row["count"]) for row in failure_rows},
        "posebusters_evaluated": int(boolean(sidecar, "pb_evaluated").sum()),
        "note": "fragment_count for valid rows is after largest-fragment docking preparation; fragmented raw endpoints are counted from failure_reason.",
    }
    return (
        pd.DataFrame(rows),
        pd.DataFrame(failure_rows),
        {"risk": pd.DataFrame(valid_rows), "descriptor": pd.DataFrame(descriptor_rows), "facts": facts},
    )


def generation_tables(generation: pd.DataFrame, docking: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Build post-training generation, docking, and Original-relative tables."""
    generation_rows: list[dict[str, object]] = []
    docking_rows: list[dict[str, object]] = []
    for variant, group in generation.groupby("variant", sort=False):
        valid = group[numeric(group, "rdkit_valid").fillna(0) == 1].copy()
        attempts = len(group)
        valid_n = len(valid)
        flags = {
            "fragment_rate_valid": numeric(valid, "fragment_count") > 1,
            "bad_bond_rate_valid": numeric(valid, "bad_bond_length_fraction") > 0.05,
            "three_member_rate_valid": numeric(valid, "three_member_ring_count") > 0,
            "large_ring_rate_valid": numeric(valid, "large_ring_count") > 0,
            "halogen_bad_rate_valid": numeric(valid, "halogen_bad_position_count") > 0,
            "guard_violation_rate_valid": numeric(valid, "chemistry_guard_violation_count") > 0,
            "brenk_alert_rate_valid": numeric(valid, "brenk_alert") > 0,
            "carbon_over_0.85_rate_valid": numeric(valid, "carbon_fraction") > 0.85,
        }
        noncomposition = (
            flags["fragment_rate_valid"]
            | flags["bad_bond_rate_valid"]
            | flags["three_member_rate_valid"]
            | flags["large_ring_rate_valid"]
            | flags["halogen_bad_rate_valid"]
        ).fillna(False)
        any_guard_or_pathology = (noncomposition | flags["guard_violation_rate_valid"]).fillna(False)
        row: dict[str, object] = {
            "variant": variant,
            "attempts": attempts,
            "rdkit_valid": valid_n,
            "valid_rate_attempts_percent": rate(valid_n, attempts),
            "noncomposition_pathology_count_valid": int(noncomposition.sum()),
            "noncomposition_pathology_rate_valid_percent": rate(int(noncomposition.sum()), valid_n),
            "guard_or_pathology_count_valid": int(any_guard_or_pathology.sum()),
            "guard_or_pathology_rate_valid_percent": rate(int(any_guard_or_pathology.sum()), valid_n),
        }
        for name, mask in flags.items():
            row[name] = rate(int(mask.fillna(False).sum()), valid_n)
            row[name.replace("_rate_valid", "_count_valid")] = int(mask.fillna(False).sum())
        for column in (
            "carbon_fraction",
            "hetero_atom_count",
            "heavy_atom_count",
            "mw",
            "logp",
            "tpsa",
            "qed",
            "sa_score",
            "ring_count",
        ):
            values = numeric(valid, column).dropna()
            row[f"{column}_mean_valid"] = float(values.mean())
            row[f"{column}_median_valid"] = float(values.median())
        generation_rows.append(row)

    docking["rdkit_valid_num"] = numeric(docking, "rdkit_valid").fillna(0)
    docking["docking_success_num"] = numeric(docking, "docking_success").fillna(0)
    docking["score_num"] = numeric(docking, "docking_score")
    docking["le_num"] = numeric(docking, "docking_ligand_efficiency")
    for variant, group in docking.groupby("variant", sort=False):
        valid = group[group["rdkit_valid_num"] == 1]
        finite = valid[np.isfinite(valid["score_num"])]
        le = finite["le_num"].dropna()
        row = {
            "variant": variant,
            "attempts": len(group),
            "backend_success": int(group["docking_success_num"].sum()),
            "rdkit_valid": len(valid),
            "finite_valid_scores": len(finite),
            "score_mean_valid": float(finite["score_num"].mean()),
            "score_median_valid": float(finite["score_num"].median()),
            "score_p10_valid": float(finite["score_num"].quantile(0.10)),
            "score_p90_valid": float(finite["score_num"].quantile(0.90)),
            "le_mean_valid": float(le.mean()),
            "le_median_valid": float(le.median()),
            "le_p90_valid": float(le.quantile(0.90)),
        }
        for threshold in (-10, -12, -13):
            count = int((finite["score_num"] <= threshold).sum())
            row[f"score_le_{abs(threshold)}_count_valid"] = count
            row[f"score_le_{abs(threshold)}_rate_valid_percent"] = rate(count, len(finite))
        docking_rows.append(row)

    generation_summary = pd.DataFrame(generation_rows)
    docking_summary = pd.DataFrame(docking_rows)
    original = generation_summary[generation_summary["variant"] == "original"].iloc[0]
    comparison_rows: list[dict[str, object]] = []
    for _, row in generation_summary.iterrows():
        comparison_rows.append(
            {
                "variant": row["variant"],
                "valid_rate_attempts_delta_pp": row["valid_rate_attempts_percent"] - original["valid_rate_attempts_percent"],
                "guard_violation_delta_pp": row["guard_violation_rate_valid"] - original["guard_violation_rate_valid"],
                "noncomposition_pathology_delta_pp": row["noncomposition_pathology_rate_valid_percent"] - original["noncomposition_pathology_rate_valid_percent"],
                "large_ring_delta_pp": row["large_ring_rate_valid"] - original["large_ring_rate_valid"],
                "brenk_delta_pp": row["brenk_alert_rate_valid"] - original["brenk_alert_rate_valid"],
                "carbon_fraction_delta": row["carbon_fraction_mean_valid"] - original["carbon_fraction_mean_valid"],
                "tpsa_delta": row["tpsa_mean_valid"] - original["tpsa_mean_valid"],
                "logp_delta": row["logp_mean_valid"] - original["logp_mean_valid"],
            }
        )

    facts = {
        "generation_attempts_per_variant": sorted({int(x) for x in generation_summary["attempts"]}),
        "docking_backend_success_per_variant": {
            str(row["variant"]): int(row["backend_success"]) for _, row in docking_summary.iterrows()
        },
        "note": "molecule-level rates use RDKit-valid denominator; validity uses fixed attempts denominator.",
    }
    return generation_summary, docking_summary, pd.DataFrame(comparison_rows), facts


def plot_outputs(
    generation: pd.DataFrame,
    docking: pd.DataFrame,
    training_valid_risk: pd.DataFrame,
) -> None:
    """Write compact plots for the Markdown work record."""
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 14, "axes.labelsize": 12})
    order = ["original", "x_only", "v_only", "native_joint"]
    colors = ["#4C78A8", "#72B7B2", "#F2A541", "#E45756"]
    gen = generation.set_index("variant").loc[order].reset_index()
    dock = docking.set_index("variant").loc[order].reset_index()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].bar(gen["variant"], gen["valid_rate_attempts_percent"], color=colors)
    axes[0, 0].set_title("RDKit-valid rate; fixed 1,000 attempts")
    axes[0, 0].set_ylabel("percent")
    axes[0, 0].set_ylim(0, 80)
    axes[0, 1].bar(gen["variant"], gen["guard_violation_rate_valid"], color=colors, label="guard")
    axes[0, 1].bar(gen["variant"], gen["large_ring_rate_valid"], color="#9D755D", alpha=0.65, label="large ring")
    axes[0, 1].set_title("Structural flags; valid-only")
    axes[0, 1].set_ylabel("percent")
    axes[0, 1].legend(frameon=False)
    axes[1, 0].bar(gen["variant"], gen["carbon_fraction_mean_valid"], color=colors)
    axes[1, 0].set_title("Mean carbon fraction; valid-only")
    axes[1, 0].set_ylabel("fraction")
    axes[1, 1].bar(gen["variant"], gen["tpsa_mean_valid"], color=colors)
    axes[1, 1].set_title("Mean TPSA; valid-only")
    axes[1, 1].set_ylabel("TPSA")
    for axis in axes.flat:
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.25)
    fig.savefig(OUT / "post_training_quality_comparison.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    axes[0].bar(dock["variant"], dock["score_median_valid"], color=colors)
    axes[0].set_title("UniDock median score")
    axes[0].set_ylabel("lower is better")
    axes[1].plot(dock["variant"], dock["score_le_10_rate_valid_percent"], "o-", label="<= -10")
    axes[1].plot(dock["variant"], dock["score_le_12_rate_valid_percent"], "o-", label="<= -12")
    axes[1].plot(dock["variant"], dock["score_le_13_rate_valid_percent"], "o-", label="<= -13")
    axes[1].set_title("Docking threshold rates")
    axes[1].set_ylabel("percent; valid-only")
    axes[1].legend(frameon=False)
    axes[2].bar(dock["variant"], dock["le_median_valid"], color=colors)
    axes[2].set_title("Ligand-efficiency median")
    axes[2].set_ylabel("-score / heavy atom")
    for axis in axes:
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.25)
    fig.savefig(OUT / "post_training_docking_comparison.png", dpi=220)
    plt.close(fig)

    risk = training_valid_risk[training_valid_risk["metric"].isin(
        ["three_member_ring_count>0", "fused_ring_count>0", "complex_ring_count>0", "brenk_count>0", "carbon_fraction>0.85"]
    )]
    fig, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    axis.bar(risk["metric"], risk["rate_percent"], color="#5B8FF9")
    axis.set_title("Training-input known risk flags; 135,498 valid endpoints")
    axis.set_ylabel("percent")
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=0.25)
    fig.savefig(OUT / "training_input_known_risks.png", dpi=220)
    plt.close(fig)


def main() -> None:
    """Run the complete read-only current-quality analysis."""
    OUT.mkdir(parents=True, exist_ok=True)
    sidecar = pd.read_csv(TRAINING_SIDECAR, low_memory=False)
    generation = pd.read_csv(GENERATION_METRICS, low_memory=False)
    docking = pd.read_csv(DOCKING_METRICS, low_memory=False)
    training_summary, failure_summary, training_details = training_input_tables(sidecar)
    generation_summary, docking_summary, comparison, generation_facts = generation_tables(generation, docking)
    training_summary.to_csv(OUT / "training_input_validity_summary.csv", index=False)
    failure_summary.to_csv(OUT / "training_input_failure_summary.csv", index=False)
    training_details["risk"].to_csv(OUT / "training_input_known_risks.csv", index=False)
    training_details["descriptor"].to_csv(OUT / "training_input_descriptor_summary.csv", index=False)
    generation_summary.to_csv(OUT / "post_training_generation_summary.csv", index=False)
    docking_summary.to_csv(OUT / "post_training_docking_summary.csv", index=False)
    comparison.to_csv(OUT / "post_vs_original_deltas.csv", index=False)
    facts = {"training": training_details["facts"], "generation": generation_facts}
    (OUT / "analysis_facts.json").write_text(json.dumps(facts, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_outputs(generation_summary, docking_summary, training_details["risk"])
    print(json.dumps(facts, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
