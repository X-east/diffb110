"""Create five-model benchmark and DiffSBDD-IQL evidence figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/home/md666/data/disk6/DiffB110")
OUT = ROOT / "analysis/diffsbdd_iql_midterm_20260817"
AUDIT = ROOT / "analysis/midterm_defense_data_audit_20260730"
FORMAL = ROOT / "analysis/benchmark/benchmark5/8gjl_c1101_benchmark_20260706_1000steps_3rep_formal"
FULL = ROOT / "analysis/benchmark/benchmark5/five_model_chemblfull_umap_20260817/results/analysis_v001"

MODELS = ["DiffGui", "DiffSBDD", "TargetDiff", "PMDM", "GCDM-SBDD"]
COLORS = {
    "DiffGui": "#E45756",
    "DiffSBDD": "#4C78A8",
    "TargetDiff": "#F58518",
    "PMDM": "#54A24B",
    "GCDM-SBDD": "#B279A2",
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def save(fig: plt.Figure, name: str) -> None:
    """Save a readable PNG and editable SVG."""
    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def plot_funnel() -> None:
    """Plot fixed-attempt generation and validity funnel."""
    table = pd.read_csv(AUDIT / "generation_funnel_repeat_summary.csv")
    table = table[table["model"].isin(MODELS)].copy()
    stages = [
        ("sanitize_valid", "Sanitize-valid"),
        ("unique_smiles", "Unique SMILES"),
        ("pb_valid", "PB-valid"),
        ("full_redocking_success", "Full redocking"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=True)
    for ax, (stage, label) in zip(axes.flat, stages):
        sub = table[table["stage"] == stage].set_index("model").loc[MODELS]
        x = np.arange(len(MODELS))
        means = sub["success_rate_mean"].to_numpy() * 100
        errors = sub["success_rate_std"].to_numpy() * 100
        ax.bar(x, means, yerr=errors, capsize=4, color=[COLORS[m] for m in MODELS])
        ax.set_title(label, fontsize=12, weight="bold")
        ax.set_xticks(x, MODELS, rotation=25, ha="right")
        ax.set_ylim(0, 108)
        ax.set_ylabel("Rate per fixed attempt (%)")
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    fig.suptitle("Five-model fixed-attempt generation funnel", fontsize=15, weight="bold")
    fig.tight_layout()
    save(fig, "five_model_generation_funnel")


def plot_docking() -> None:
    """Plot raw-pose and full-redocking outcomes using the formal three-repeat summary."""
    labels = MODELS
    raw_median = np.array([-9.774, -5.837, -9.051, -2.454, 16.480])
    full_median = np.array([-11.009, -9.580, -11.087, -9.663, -6.526])
    full_threshold = np.array([71.30, 39.07, 70.11, 41.06, 2.48])
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    panels = [
        (axes[0], raw_median, "Raw-pose QVina median (kcal/mol)", "lower is better"),
        (axes[1], full_median, "Full-redocking median (kcal/mol)", "lower is better"),
        (axes[2], full_threshold, "Full-redocking rate <= -10 (%)", "higher is better"),
    ]
    for ax, values, title, note in panels:
        ax.bar(x, values, color=[COLORS[m] for m in labels])
        ax.set_title(title, fontsize=11, weight="bold")
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.text(0.02, 0.96, note, transform=ax.transAxes, va="top", fontsize=9, color="#555555")
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
        for i, value in enumerate(values):
            ax.text(i, value + (0.7 if value >= 0 else -0.7), f"{value:.2f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=9)
    fig.suptitle("Five-model target-pocket evaluation", fontsize=15, weight="bold")
    fig.tight_layout()
    save(fig, "five_model_docking_comparison")


def plot_chembl() -> None:
    """Plot full-ChEMBL UMAP coverage and nearest-distance summaries."""
    coverage = pd.read_csv(FULL / "coverage_summary.csv").set_index("model").loc[MODELS]
    distance = pd.read_csv(FULL / "distance_to_chembl_summary.csv").set_index("model").loc[MODELS]
    x = np.arange(len(MODELS))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.9))
    axes[0].bar(
        x,
        coverage["inside_hdr_95_mean"].to_numpy() * 100,
        yerr=coverage["inside_hdr_95_std"].to_numpy() * 100,
        capsize=4,
        color=[COLORS[m] for m in MODELS],
    )
    axes[0].set_title("ChEMBL36 95% HDR coverage", fontsize=12, weight="bold")
    axes[0].set_ylabel("Generated molecules inside HDR (%)")
    axes[0].set_ylim(0, 75)
    axes[1].bar(
        x,
        distance["median_mean"].to_numpy(),
        yerr=distance["median_std"].to_numpy(),
        capsize=4,
        color=[COLORS[m] for m in MODELS],
    )
    axes[1].set_title("Nearest ChEMBL36 UMAP distance", fontsize=12, weight="bold")
    axes[1].set_ylabel("Median distance (lower is closer)")
    for ax in axes:
        ax.set_xticks(x, MODELS, rotation=25, ha="right")
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    fig.suptitle("Five-model chemical-space reference analysis", fontsize=15, weight="bold")
    fig.tight_layout()
    save(fig, "five_model_chembl_overlap")


def plot_pathology() -> None:
    """Plot structure-risk rates from the three-repeat raw-quality audit."""
    metrics = ["Rich-carbon", "Brenk alert", "SA > 6", "Rings >= 6"]
    values = {
        "DiffGui": [8.25, 25.95, 6.73, 23.06],
        "DiffSBDD": [3.00, 58.44, 6.60, 5.95],
        "TargetDiff": [15.70, 40.35, 9.35, 13.48],
        "PMDM": [6.48, 44.49, 23.94, 23.69],
        "GCDM-SBDD": [4.84, 75.86, 1.90, 0.00],
    }
    x = np.arange(len(metrics))
    width = 0.19
    fig, ax = plt.subplots(figsize=(12.5, 5))
    for i, model in enumerate(MODELS):
        ax.bar(x + (i - 1.5) * width, values[model], width, label=model, color=COLORS[model])
    ax.set_xticks(x, metrics)
    ax.set_ylabel("Rate among valid generated molecules (%)")
    ax.set_ylim(0, 85)
    ax.set_title("Structure-risk indicators remain model-specific", fontsize=14, weight="bold")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    ax.legend(ncol=4, frameon=False)
    fig.tight_layout()
    save(fig, "five_model_structure_risk")


def plot_iql_gates() -> None:
    """Plot evidence status without implying completed IQL actor results."""
    labels = [
        "100k trajectory\ncollection",
        "Full episode\nindex",
        "Formal reward\nsidecar accepted",
        "Held-out critic\ngo/no-go",
        "IQL actor\nindependent eval",
    ]
    values = [1, 1, 0, 0, 0]
    colors = ["#4C78A8", "#4C78A8", "#D9D9D9", "#D9D9D9", "#D9D9D9"]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    x = np.arange(len(labels))
    ax.bar(x, values, color=colors, width=0.65)
    for i, value in enumerate(values):
        ax.text(i, 0.06 if value else 0.04, "accepted" if value else "pending", ha="center", va="bottom", fontsize=10, weight="bold", color="#1F1F1F")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.2)
    ax.set_yticks([])
    ax.set_title("DiffSBDD offline-IQL evidence gates (2026-08-17 snapshot)", fontsize=14, weight="bold")
    ax.text(0.5, -0.22, "Pending gates are not failures; they are prerequisites before formal actor claims.", transform=ax.transAxes, ha="center", color="#555555")
    fig.tight_layout()
    save(fig, "diffsbdd_iql_evidence_gates")


def main() -> None:
    """Generate all figures."""
    OUT.mkdir(parents=True, exist_ok=True)
    plot_funnel()
    plot_docking()
    plot_chembl()
    plot_pathology()
    plot_iql_gates()


if __name__ == "__main__":
    main()
