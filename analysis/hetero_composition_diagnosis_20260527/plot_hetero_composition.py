#!/usr/bin/env python3
"""Create plots for the C/O-only hetero-composition diagnosis."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-diffb110")

import matplotlib.pyplot as plt
import pandas as pd


ANALYSIS_DIR = Path(__file__).resolve().parent


def pct(series: pd.Series) -> pd.Series:
    return series.astype(float) * 100.0


def plot_global_comparison() -> None:
    summary = pd.read_csv(ANALYSIS_DIR / "global_composition_summary.csv")
    labels = {
        "chembl36_cleaned": "ChEMBL cleaned",
        "chembl36_strict_property_pool": "ChEMBL strict",
        "out_valid_generated": "OUT generated",
    }
    metrics = [
        ("co_only_high_c_low_o_rate", "C/O-only high-C O1-2"),
        ("co_only_mostly_c_o1_3_rate", "C/O-only mostly-C O1-3"),
        ("no_non_o_hetero_rate", "No non-O hetero"),
        ("has_nitrogen_rate", "Has nitrogen"),
        ("has_halogen_rate", "Has halogen"),
    ]
    data = summary.set_index("label")[[m[0] for m in metrics]].rename(index=labels)
    data.columns = [m[1] for m in metrics]
    data = data * 100.0

    ax = data.T.plot(kind="bar", figsize=(12, 6), width=0.78)
    ax.set_ylabel("Molecules (%)")
    ax.set_title("Global C/O-only and non-oxygen heteroatom comparison")
    ax.set_ylim(0, max(100, data.max().max() * 1.15))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / "global_co_only_comparison.png", dpi=220)
    plt.close()


def plot_top_reward_runs() -> None:
    df = pd.read_csv(ANALYSIS_DIR / "top_reward_composition_summary.csv")
    metric = "co_only_mostly_c_o1_3_rate"
    df = df[df["top_n"].ge(100)].sort_values(metric, ascending=False).head(15)
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 7))
    y = range(len(df))
    ax.barh(y, pct(df[metric]), color="#c5543d", label="C/O-only mostly-C O1-3")
    ax.scatter(pct(df["has_nitrogen_rate"]), y, color="#2b6cb0", label="Has nitrogen", zorder=3)
    ax.scatter(
        pct(df["has_halogen_rate"]),
        y,
        color="#2f855a",
        label="Has halogen",
        zorder=3,
    )
    ax.set_yticks(list(y))
    ax.set_yticklabels(df["run"])
    ax.invert_yaxis()
    ax.set_xlabel("Top reward decile molecules (%)")
    ax.set_title("Top reward regions enriched for C/O-only mostly-carbon molecules")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / "top_reward_co_only_runs.png", dpi=220)
    plt.close()


def plot_hetero_signatures() -> None:
    df = pd.read_csv(ANALYSIS_DIR / "hetero_signature_counts.csv")
    df = df[df["field"].eq("hetero_signature")]
    keep = ["chembl36_strict_property_pool", "out_valid_generated"]
    df = df[df["dataset"].isin(keep)]
    top_values = (
        df[df["dataset"].eq("out_valid_generated")]
        .sort_values("fraction", ascending=False)
        .head(12)["value"]
        .tolist()
    )
    df = df[df["value"].isin(top_values)]
    pivot = (
        df.pivot_table(index="value", columns="dataset", values="fraction", fill_value=0.0)
        .reindex(top_values)
        * 100.0
    )
    pivot = pivot.rename(
        columns={
            "chembl36_strict_property_pool": "ChEMBL strict",
            "out_valid_generated": "OUT generated",
        }
    )

    ax = pivot.plot(kind="bar", figsize=(12, 6), width=0.78)
    ax.set_ylabel("Molecules (%)")
    ax.set_xlabel("Hetero signature")
    ax.set_title("Top generated hetero signatures vs strict ChEMBL pool")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / "hetero_signature_comparison.png", dpi=220)
    plt.close()


def update_report_links() -> None:
    path = ANALYSIS_DIR / "co_only_composition_report.md"
    text = path.read_text()
    block = """\n## 图示\n\n![Global C/O-only comparison](global_co_only_comparison.png)\n\n![Top reward C/O-only runs](top_reward_co_only_runs.png)\n\n![Hetero signature comparison](hetero_signature_comparison.png)\n"""
    if "## 图示" not in text:
        text = text.replace("\n## 五、结论用法\n", block + "\n## 五、结论用法\n")
        path.write_text(text)


def main() -> None:
    plot_global_comparison()
    plot_top_reward_runs()
    plot_hetero_signatures()
    update_report_links()


if __name__ == "__main__":
    main()
