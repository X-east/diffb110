from __future__ import annotations

"""Plot ChEMBL classifier score comparison for DiffGui and DiffSBDD samples."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DIFFGUI_CSV = HERE / "diffgui_500_scores.csv"
DIFFSBDD_CSV = Path("analysis/diffsbdd_8gpu_5000_score_20260608/generated_5000_scores.csv")


def _valid_probs(csv_path: Path, column: str) -> np.ndarray:
    """Load finite probabilities for scoring-valid molecules."""
    df = pd.read_csv(csv_path)
    valid = df[df["scoring_valid"] == 1]
    values = pd.to_numeric(valid[column], errors="coerce").dropna().to_numpy(dtype=float)
    return values[np.isfinite(values)]


def _rate(values: np.ndarray, threshold: float) -> float:
    """Return fraction above a probability threshold."""
    return float(np.mean(values >= threshold)) if values.size else 0.0


def main() -> None:
    """Create the comparison figure."""
    dg_v02 = _valid_probs(DIFFGUI_CSV, "p_good_v02")
    ds_v02 = _valid_probs(DIFFSBDD_CSV, "p_good_v02")
    dg_v021 = _valid_probs(DIFFGUI_CSV, "p_good_v021")
    ds_v021 = _valid_probs(DIFFSBDD_CSV, "p_good_v021")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    bins = np.linspace(0.0, 1.0, 41)
    colors = {"DiffGui": "#2f7f5f", "DiffSBDD": "#3268a8"}

    for ax, dg, ds, title, xlabel in [
        (axes[0], dg_v02, ds_v02, "Classifier v0.2", "P_good_v0.2"),
        (axes[1], dg_v021, ds_v021, "Classifier v0.2.1", "P_good_v0.2.1"),
    ]:
        ax.hist(ds, bins=bins, density=True, alpha=0.48, color=colors["DiffSBDD"], label=f"DiffSBDD n={len(ds)}")
        ax.hist(dg, bins=bins, density=True, alpha=0.55, color=colors["DiffGui"], label=f"DiffGui n={len(dg)}")
        ax.axvline(0.5, color="#222222", linestyle="--", linewidth=1.0)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("density")
        ax.legend(frameon=False, fontsize=8)

    labels = ["v0.2 >=0.5", "v0.2 >=0.8", "v0.2.1 >=0.5", "v0.2.1 >=0.8"]
    dg_rates = [_rate(dg_v02, 0.5), _rate(dg_v02, 0.8), _rate(dg_v021, 0.5), _rate(dg_v021, 0.8)]
    ds_rates = [_rate(ds_v02, 0.5), _rate(ds_v02, 0.8), _rate(ds_v021, 0.5), _rate(ds_v021, 0.8)]
    x = np.arange(len(labels))
    width = 0.36
    axes[2].bar(x - width / 2, ds_rates, width, color=colors["DiffSBDD"], label="DiffSBDD")
    axes[2].bar(x + width / 2, dg_rates, width, color=colors["DiffGui"], label="DiffGui")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=25, ha="right")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_ylabel("fraction")
    axes[2].set_title("High-score fraction")
    axes[2].legend(frameon=False, fontsize=8)
    for xpos, value in zip(x - width / 2, ds_rates):
        axes[2].text(xpos, value + 0.02, f"{value:.2f}", ha="center", va="bottom", fontsize=7)
    for xpos, value in zip(x + width / 2, dg_rates):
        axes[2].text(xpos, value + 0.02, f"{value:.2f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle("ChEMBL classifier scores: DiffGui vs DiffSBDD", y=1.03, fontsize=13)
    fig.tight_layout()
    fig.savefig(HERE / "diffgui_vs_diffsbdd_classifier_distribution.png", dpi=200, bbox_inches="tight")
    fig.savefig(HERE / "diffgui_vs_diffsbdd_classifier_distribution.svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
