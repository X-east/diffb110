from __future__ import annotations

"""Plot score and descriptor summaries for the 5000-molecule SDF scoring run."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "generated_5000_scores.csv"


def main() -> None:
    """Create compact score distribution figures."""
    df = pd.read_csv(CSV_PATH)
    valid = df[df["scoring_valid"] == 1].copy()

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, col, title in [
        (axes[0, 0], "p_good_v02", "v0.2 P(good)"),
        (axes[0, 1], "p_good_v021", "v0.2.1 P(good)"),
        (axes[1, 0], "dynamic_score_v02", "v0.2 dynamic rank score"),
        (axes[1, 1], "guarded_score", "guarded score"),
    ]:
        ax.hist(valid[col].dropna(), bins=50, color="#3268a8", alpha=0.85)
        ax.set_title(title)
        ax.set_xlabel(col)
        ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(HERE / "score_distributions.png", dpi=180)
    fig.savefig(HERE / "score_distributions.svg")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, col, title in [
        (axes[0], "carbon_fraction", "carbon fraction"),
        (axes[1], "tpsa", "TPSA"),
        (axes[2], "logp", "LogP"),
    ]:
        ax.scatter(valid[col], valid["guarded_score"], s=8, alpha=0.35, color="#2f7f5f")
        ax.set_xlabel(col)
        ax.set_ylabel("guarded_score")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(HERE / "score_vs_properties.png", dpi=180)
    fig.savefig(HERE / "score_vs_properties.svg")
    plt.close(fig)

    status_counts = df["sanitize_status"].value_counts().reset_index()
    status_counts.columns = ["sanitize_status", "count"]
    status_counts.to_csv(HERE / "sanitize_status_counts.csv", index=False)


if __name__ == "__main__":
    main()
