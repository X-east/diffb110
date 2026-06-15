from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


MODEL_ORDER = [
    "Native8J6",
    "DiffSBDD",
    "DiffGui",
    "DePPA",
    "PMDM",
    "DiffInt",
    "DiffPharma",
]


def _load_docking(docking_root: Path) -> pd.DataFrame:
    rows = []
    for summary_path in sorted(docking_root.glob("*/run_summary.json")):
        with summary_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        rows.append(
            {
                "model": data["model"],
                "dock_total": data["total_ligands"],
                "dock_success": data["success_count"],
                "dock_success_rate": data["success_rate"],
                "dock_score_mean": data.get("score_mean"),
                "dock_score_median": data.get("score_median"),
                "dock_score_best": data.get("score_min_best"),
            }
        )
    return pd.DataFrame(rows)


def _load_quality(evaluation_root: Path) -> pd.DataFrame:
    summary_csv = evaluation_root / "six_models" / "model_summary.csv"
    if not summary_csv.exists():
        return pd.DataFrame()
    columns = [
        "model",
        "sdf_records",
        "rdkit_readable",
        "sanitize_valid",
        "sanitize_rate_records",
        "unique_rate_valid",
        "heavy_atoms_mean",
        "ring_count_mean",
        "aromatic_ring_count_mean",
        "three_member_ring_count_mean",
        "carbon_fraction_mean",
        "hetero_atoms_mean",
        "qed_mean",
        "logp_mean",
        "tpsa_mean",
        "has_aromatic_ring_rate",
        "has_three_member_ring_rate",
    ]
    quality = pd.read_csv(summary_csv)
    return quality[[col for col in columns if col in quality.columns]]


def _ordered(df: pd.DataFrame) -> pd.DataFrame:
    order = {model: idx for idx, model in enumerate(MODEL_ORDER)}
    return df.assign(_order=df["model"].map(order).fillna(999)).sort_values("_order").drop(columns="_order")


def _plot_docking(summary: pd.DataFrame, out_dir: Path, label: str, file_prefix: str) -> None:
    dock = summary.dropna(subset=["dock_score_mean"]).copy()
    dock = _ordered(dock)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = range(len(dock))
    ax.bar(x, dock["dock_score_mean"], color="#4C78A8", label="mean")
    ax.scatter(x, dock["dock_score_best"], color="#F58518", zorder=3, label="best")
    native = dock.loc[dock["model"] == "Native8J6", "dock_score_mean"]
    if not native.empty:
        ax.axhline(native.iloc[0], color="#54A24B", linestyle="--", linewidth=1.5, label="native 8J6")
    ax.set_xticks(list(x))
    ax.set_xticklabels(dock["model"], rotation=35, ha="right")
    ax.set_ylabel("UniDock score (lower is better)")
    ax.set_title(f"{label} docking")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / f"{file_prefix}_docking_scores.png", dpi=220)
    plt.close(fig)


def _plot_quality(summary: pd.DataFrame, out_dir: Path, label: str, file_prefix: str) -> None:
    quality = summary[summary["model"] != "Native8J6"].copy()
    quality = _ordered(quality)
    metrics = [
        ("sanitize_rate_records", "sanitize rate", (0, 1.05)),
        ("carbon_fraction_mean", "carbon fraction", (0, 1.05)),
        ("ring_count_mean", "ring count mean", None),
        ("qed_mean", "QED mean", (0, 1.05)),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4))
    for ax, (col, title, ylim) in zip(axes.ravel(), metrics):
        if col not in quality:
            ax.axis("off")
            continue
        ax.bar(quality["model"], quality[col], color="#72B7B2")
        ax.set_title(title)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(f"{label} molecule quality", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / f"{file_prefix}_quality_summary.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--label", default="5XZR/8J6 benchmark")
    parser.add_argument("--file-prefix", default="benchmark")
    args = parser.parse_args()

    out_dir = args.out_dir or args.run_root
    out_dir.mkdir(parents=True, exist_ok=True)

    quality = _load_quality(args.run_root / "evaluation")
    docking = _load_docking(args.run_root / "docking" / "unidock_fast_5xzr_8j6")
    if quality.empty:
        summary = docking
    else:
        summary = pd.merge(quality, docking, how="outer", on="model")
    summary = _ordered(summary)
    summary.to_csv(out_dir / "benchmark_summary.csv", index=False)

    _plot_docking(summary, out_dir, args.label, args.file_prefix)
    if not quality.empty:
        _plot_quality(summary, out_dir, args.label, args.file_prefix)

    print(json.dumps({"rows": int(len(summary)), "out_dir": str(out_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
