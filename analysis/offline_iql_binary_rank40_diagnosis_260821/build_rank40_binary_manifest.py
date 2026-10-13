"""Freeze a valid-only rank-40% preferred/non-preferred label for diagnosis."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


INPUT = Path("analysis/offline_iql_scaffold_split_260821/scaffold_split_manifest.csv.gz")
OUTPUT_DIR = Path("analysis/offline_iql_binary_rank40_diagnosis_260821")
QUANTILE = 0.40


def parse_bool(value: object) -> bool:
    """Parse common CSV boolean encodings."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def main() -> None:
    """Build the frozen binary manifest and audit class composition."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(INPUT, compression="gzip")
    frame["structural_hard_bad"] = frame["structural_hard_bad"].map(parse_bool)
    frame["umap_nearest_chembl_distance"] = pd.to_numeric(
        frame["umap_nearest_chembl_distance"], errors="raise"
    )
    train_distance = frame.loc[
        frame["split"] == "train", "umap_nearest_chembl_distance"
    ]
    threshold = float(train_distance.quantile(QUANTILE, interpolation="linear"))
    frame["rank40_candidate"] = frame["umap_nearest_chembl_distance"] <= threshold
    frame["binary_label"] = "non_preferred"
    frame.loc[
        frame["rank40_candidate"] & ~frame["structural_hard_bad"], "binary_label"
    ] = "preferred"
    frame["binary_target"] = frame["binary_label"].map(
        {"preferred": 1.0, "non_preferred": -1.0}
    )

    output_columns = [
        "trajectory_id",
        "molecule_id",
        "canonical_smiles",
        "rdkit_valid",
        "trajectory_shard",
        "mol_index",
        "umap_nearest_chembl_distance",
        "structural_hard_bad",
        "structural_hard_bad_reasons",
        "scaffold_group",
        "scaffold_fold",
        "split",
        "rank40_candidate",
        "binary_label",
        "binary_target",
    ]
    frame[output_columns].to_csv(
        OUTPUT_DIR / "binary_rank40_manifest.csv.gz",
        index=False,
        compression="gzip",
    )

    summary_rows: list[dict[str, object]] = []
    for split in ("train", "validation", "test"):
        subset = frame.loc[frame["split"] == split]
        for label in ("preferred", "non_preferred"):
            label_subset = subset.loc[subset["binary_label"] == label]
            summary_rows.append(
                {
                    "split": split,
                    "binary_label": label,
                    "rows": len(label_subset),
                    "fraction_within_split": len(label_subset) / len(subset),
                    "mean_distance": float(label_subset["umap_nearest_chembl_distance"].mean()),
                    "median_distance": float(label_subset["umap_nearest_chembl_distance"].median()),
                    "hard_bad_count": int(label_subset["structural_hard_bad"].sum()),
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / "binary_rank40_summary.csv", index=False)

    metadata = {
        "input": str(INPUT),
        "valid_only": True,
        "invalid_rows_included": False,
        "rule": (
            "preferred = train-frozen nearest-ChEMBL UMAP distance <= train 40th percentile "
            "and structural_hard_bad == false; non_preferred = every remaining valid row"
        ),
        "quantile": QUANTILE,
        "train_frozen_threshold": threshold,
        "counts": summary.to_dict(orient="records"),
        "scaffold_overlap_note": "Reuses the existing scaffold-held-out folds; no group overlap was observed.",
    }
    (OUTPUT_DIR / "binary_rank40_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    colors = {"preferred": "#2F6B9A", "non_preferred": "#C96B3C"}
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    count_frame = summary.pivot(index="split", columns="binary_label", values="rows").loc[
        ["train", "validation", "test"]
    ]
    count_frame[["preferred", "non_preferred"]].plot.bar(
        ax=axes[0],
        color=[colors["preferred"], colors["non_preferred"]],
        rot=0,
        width=0.76,
    )
    axes[0].set_title("Rank-40% binary labels by scaffold split", fontsize=16)
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Valid molecules", fontsize=13)
    axes[0].legend(title="label", fontsize=10)

    values = [
        frame.loc[
            (frame["split"] == split) & (frame["binary_label"] == label),
            "umap_nearest_chembl_distance",
        ].to_numpy()
        for split in ("train", "validation", "test")
        for label in ("preferred", "non_preferred")
    ]
    positions = [0, 1, 3, 4, 6, 7]
    box = axes[1].boxplot(
        values,
        positions=positions,
        widths=0.72,
        patch_artist=True,
        showfliers=False,
    )
    for index, patch in enumerate(box["boxes"]):
        patch.set_facecolor(colors["preferred"] if index % 2 == 0 else colors["non_preferred"])
        patch.set_alpha(0.72)
    axes[1].axhline(threshold, color="#555555", linestyle="--", linewidth=1.8, label=f"train q40={threshold:.4f}")
    axes[1].set_xticks([0.5, 3.5, 6.5], ["train", "validation", "test"])
    axes[1].set_title("Distance distribution under frozen train threshold", fontsize=16)
    axes[1].set_ylabel("Nearest ChEMBL UMAP distance", fontsize=13)
    axes[1].legend(fontsize=10)
    for axis in axes:
        axis.grid(axis="y", alpha=0.22)
        axis.tick_params(labelsize=11)
    figure.suptitle("Valid-only preferred vs non-preferred label audit", fontsize=19)
    figure.savefig(OUTPUT_DIR / "binary_rank40_label_audit.png", dpi=190)
    figure.savefig(OUTPUT_DIR / "binary_rank40_label_audit.svg")
    plt.close(figure)


if __name__ == "__main__":
    main()
