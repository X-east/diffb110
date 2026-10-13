"""Quantify composition, topology, and structural risk in DiffSBDD E2PO outputs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "analysis/diffsbdd_global_e2po_good_bad_xv_smoke_260825/generation_molecule_metrics.csv"
OUT = Path(__file__).resolve().parent
SEED = 20260829
BOOTSTRAPS = 2000

CONTINUOUS_METRICS = [
    "heavy_atom_count",
    "atom_count",
    "mw",
    "carbon_fraction",
    "hetero_atom_count",
    "hetero_atom_fraction",
    "logp",
    "tpsa",
    "hba",
    "hbd",
    "qed",
    "sa_score",
    "ring_count",
    "aromatic_ring_count",
    "three_member_ring_count",
    "max_fused_ring_system_size",
    "spiro_atom_count",
    "bridgehead_atom_count",
    "large_ring_count",
    "fragment_count",
    "bad_bond_length_fraction",
]

RATE_METRICS = [
    "brenk_alert",
    "chemistry_guard_violation_count",
    "fragment_count",
    "three_member_ring_count",
    "large_ring_count",
]

# This is an explicitly operational diagnostic, not a universal molecular
# complexity definition. All terms are size/topology burden indicators.
INDEX_METRICS = [
    "heavy_atom_count",
    "ring_count",
    "aromatic_ring_count",
    "three_member_ring_count",
    "max_fused_ring_system_size",
    "spiro_atom_count",
    "bridgehead_atom_count",
    "large_ring_count",
]

EVENT_METRICS = [
    "any_fragment",
    "any_three_member_ring",
    "any_large_ring",
    "any_guard_violation",
    "any_brenk_alert",
]


def _safe_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convert requested columns to numeric and fail on missing columns."""
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise KeyError(f"missing required columns: {missing}")
    result = frame.copy()
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def _valid_frame(frame: pd.DataFrame, variant: str) -> pd.DataFrame:
    """Return the fixed-attempt RDKit-valid rows for one variant."""
    selected = frame[(frame["variant"] == variant) & (frame["rdkit_valid"] == 1)].copy()
    if selected.empty:
        raise ValueError(f"no valid rows for variant={variant}")
    return selected


def _bootstrap_mean_difference(
    values: np.ndarray,
    reference: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Return mean difference and an independent-sample bootstrap 95% CI."""
    values = values[np.isfinite(values)]
    reference = reference[np.isfinite(reference)]
    observed = float(values.mean() - reference.mean())
    draws = np.empty(BOOTSTRAPS, dtype=float)
    for index in range(BOOTSTRAPS):
        draws[index] = (
            rng.choice(values, size=values.size, replace=True).mean()
            - rng.choice(reference, size=reference.size, replace=True).mean()
        )
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return observed, float(lower), float(upper)


def _cohen_d(values: np.ndarray, reference: np.ndarray) -> float:
    """Return pooled-SD standardized mean difference."""
    values = values[np.isfinite(values)]
    reference = reference[np.isfinite(reference)]
    pooled = np.sqrt(
        ((values.size - 1) * values.var(ddof=1) + (reference.size - 1) * reference.var(ddof=1))
        / max(values.size + reference.size - 2, 1)
    )
    if pooled == 0:
        return 0.0
    return float((values.mean() - reference.mean()) / pooled)


def _complexity_index(frame: pd.DataFrame, original: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add an Original-standardized size/topology burden index."""
    usable: list[str] = []
    for metric in INDEX_METRICS:
        scale = float(original[metric].std(ddof=1))
        if np.isfinite(scale) and scale > 0:
            usable.append(metric)
            frame[f"index_z_{metric}"] = (frame[metric] - original[metric].mean()) / scale
    frame["size_topology_burden_index"] = frame[[f"index_z_{metric}" for metric in usable]].mean(axis=1)
    return frame, usable


def _summarize(frame: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Return valid-only mean, median, SD, and fixed-attempt counts."""
    rows: list[dict[str, object]] = []
    for variant in sorted(frame["variant"].unique()):
        all_rows = frame[frame["variant"] == variant]
        valid = _valid_frame(frame, variant)
        for metric in metrics:
            values = valid[metric].to_numpy(dtype=float)
            rows.append(
                {
                    "variant": variant,
                    "metric": metric,
                    "attempts": int(len(all_rows)),
                    "valid_n": int(len(valid)),
                    "mean_valid": float(np.nanmean(values)),
                    "median_valid": float(np.nanmedian(values)),
                    "sd_valid": float(np.nanstd(values, ddof=1)),
                    "valid_rate_attempts": float(len(valid) / len(all_rows)),
                }
            )
    return pd.DataFrame(rows)


def _effects(frame: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Compare every trained variant with Original using bootstrap intervals."""
    rng = np.random.default_rng(SEED)
    original = _valid_frame(frame, "original")
    rows: list[dict[str, object]] = []
    for variant in sorted(frame["variant"].unique()):
        if variant == "original":
            continue
        valid = _valid_frame(frame, variant)
        for metric in metrics:
            values = valid[metric].to_numpy(dtype=float)
            reference = original[metric].to_numpy(dtype=float)
            difference, lower, upper = _bootstrap_mean_difference(values, reference, rng)
            ref_mean = float(np.nanmean(reference))
            rows.append(
                {
                    "variant": variant,
                    "metric": metric,
                    "original_mean": ref_mean,
                    "variant_mean": float(np.nanmean(values)),
                    "difference_vs_original": difference,
                    "relative_change_percent": float(100.0 * difference / ref_mean) if abs(ref_mean) > 1.0e-12 else np.nan,
                    "bootstrap_ci95_low": lower,
                    "bootstrap_ci95_high": upper,
                    "cohen_d": _cohen_d(values, reference),
                    "ci_excludes_zero": bool(lower > 0 or upper < 0),
                }
            )
    return pd.DataFrame(rows)


def _correlations(frame: pd.DataFrame) -> pd.DataFrame:
    """Quantify whether high carbon fraction tracks topology burden."""
    rows: list[dict[str, object]] = []
    for variant in sorted(frame["variant"].unique()):
        valid = _valid_frame(frame, variant)
        for metric in INDEX_METRICS + ["size_topology_burden_index", "tpsa", "hetero_atom_count", "hetero_atom_fraction", "guard_violation"]:
            if metric == "guard_violation":
                values = valid["chemistry_guard_violation_count"]
                correlation_column = "chemistry_guard_violation_count"
            else:
                values = valid[metric]
                correlation_column = metric
            rows.append(
                {
                    "variant": variant,
                    "x": "carbon_fraction",
                    "y": metric,
                    "spearman_r": float(valid[["carbon_fraction", correlation_column]].corr(method="spearman").iloc[0, 1]),
                    "n": int(len(valid)),
                }
            )
    return pd.DataFrame(rows)


def _plot_index(summary: pd.DataFrame, out: Path) -> None:
    """Plot the diagnostic size/topology burden index."""
    values = summary[summary["metric"] == "size_topology_burden_index"].copy()
    order = ["original", "x_only", "v_only", "native_joint"]
    values["variant"] = pd.Categorical(values["variant"], categories=order, ordered=True)
    values = values.sort_values("variant")
    original_mean = float(values.loc[values["variant"] == "original", "mean_valid"].iloc[0])
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    x = np.arange(len(values))
    colors = ["#4C78A8", "#72B7B2", "#E45756", "#F2CF5B"]
    ax.bar(x, values["mean_valid"], color=colors, width=0.68)
    ax.axhline(original_mean, color="#333333", linestyle="--", linewidth=1.8, label="Original valid mean")
    for index, value in enumerate(values["mean_valid"]):
        ax.text(index, value + 0.035 if value >= 0 else value - 0.11, f"{value:.2f}", ha="center", fontsize=13)
    ax.set_xticks(x, ["Original", "x-only", "v-only", "native-joint"])
    ax.set_ylabel("Original-standardized size/topology burden")
    ax.set_title("E2PO outputs: size/topology burden index")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=11)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=220)
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)


def _plot_effect_heatmap(effects: pd.DataFrame, out: Path) -> None:
    """Plot standardized mean differences for composition and topology metrics."""
    metrics = [
        "carbon_fraction",
        "hetero_atom_count",
        "tpsa",
        "heavy_atom_count",
        "ring_count",
        "aromatic_ring_count",
        "large_ring_count",
        "chemistry_guard_violation_count",
    ]
    labels = ["C fraction", "Hetero atoms", "TPSA", "Heavy atoms", "Rings", "Aromatic rings", "Large rings", "Guard violation"]
    variants = ["x_only", "v_only", "native_joint"]
    matrix = np.full((len(variants), len(metrics)), np.nan)
    for i, variant in enumerate(variants):
        for j, metric in enumerate(metrics):
            row = effects[(effects["variant"] == variant) & (effects["metric"] == metric)]
            if not row.empty:
                matrix[i, j] = float(row["cohen_d"].iloc[0])
    fig, ax = plt.subplots(figsize=(13.2, 4.9))
    vmax = max(1.0, float(np.nanmax(np.abs(matrix))))
    image = ax.imshow(matrix, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isfinite(matrix[i, j]):
                color = "white" if abs(matrix[i, j]) > 0.55 * vmax else "black"
                ax.text(j, i, f"{matrix[i, j]:+.2f}", ha="center", va="center", color=color, fontsize=12)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(variants)), ["x-only", "v-only", "native-joint"])
    ax.set_title("Standardized change versus Original (Cohen's d; valid molecules)")
    fig.colorbar(image, ax=ax, label="Cohen's d")
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=220)
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)


def main() -> None:
    """Run the quantitative complexity analysis and write all evidence."""
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(INPUT)
    frame = _safe_numeric(frame, ["rdkit_valid", *CONTINUOUS_METRICS, "brenk_alert", "chemistry_guard_violation_count"])
    frame["any_fragment"] = (frame["fragment_count"] > 1).astype(float)
    frame["any_three_member_ring"] = (frame["three_member_ring_count"] > 0).astype(float)
    frame["any_large_ring"] = (frame["large_ring_count"] > 0).astype(float)
    frame["any_guard_violation"] = (frame["chemistry_guard_violation_count"] > 0).astype(float)
    frame["any_brenk_alert"] = (frame["brenk_alert"] > 0).astype(float)
    frame, index_metrics_used = _complexity_index(frame, _valid_frame(frame, "original"))
    all_metrics = CONTINUOUS_METRICS + EVENT_METRICS + ["size_topology_burden_index"]
    summary = _summarize(frame, all_metrics)
    effects = _effects(frame, all_metrics)
    correlations = _correlations(frame)
    summary.to_csv(OUT / "complexity_metric_summary.csv", index=False)
    effects.to_csv(OUT / "complexity_effects_vs_original.csv", index=False)
    correlations.to_csv(OUT / "carbon_topology_correlations.csv", index=False)
    _plot_index(summary, OUT / "size_topology_burden_index")
    _plot_effect_heatmap(effects, OUT / "complexity_effect_heatmap")

    key_metrics = [
        "carbon_fraction",
        "hetero_atom_count",
        "hetero_atom_fraction",
        "heavy_atom_count",
        "ring_count",
        "aromatic_ring_count",
        "large_ring_count",
        "three_member_ring_count",
        "max_fused_ring_system_size",
        "fragment_count",
        "bad_bond_length_fraction",
        "chemistry_guard_violation_count",
        "any_fragment",
        "any_three_member_ring",
        "any_large_ring",
        "any_guard_violation",
        "any_brenk_alert",
        "size_topology_burden_index",
    ]
    key = summary[summary["metric"].isin(key_metrics)].copy()
    key.to_csv(OUT / "key_complexity_summary.csv", index=False)
    report_facts = {
        "input": str(INPUT),
        "variants": sorted(frame["variant"].unique().tolist()),
        "attempts_per_variant": {variant: int((frame["variant"] == variant).sum()) for variant in sorted(frame["variant"].unique())},
        "valid_counts": {variant: int(len(_valid_frame(frame, variant))) for variant in sorted(frame["variant"].unique())},
        "index_definition": "mean of Original-valid z-scores for size/topology metrics; higher means greater operational burden",
        "index_metrics_used": index_metrics_used,
        "bootstrap_replicates": BOOTSTRAPS,
        "seed": SEED,
        "interpretation_boundary": "The index is a transparent diagnostic, not a universal molecular complexity or synthesizability score.",
    }
    (OUT / "analysis_facts.json").write_text(json.dumps(report_facts, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
