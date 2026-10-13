"""Re-plot facets / drift / manifest from saved coordinate CSVs.

Reads the already-computed UMAP projection outputs and reproduces the three
figure products that failed only because of a matplotlib keyword typo. No UMAP
fit is re-run; this is pure plotting + light summary recomputation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

OUT_DIR = Path(__file__).resolve().parent
ORIGINAL_COLOR = "#3498DB"
STEP_4000_COLOR = "#E74C3C"
CHEMBL_COLOR = "#A8ADB3"
FACET_GROUPS = [
    "Original", "step_002000", "step_004000", "step_006000",
    "step_010000", "step_014000", "step_018000", "final",
]
STEP_COLORMAP = "viridis"


def _save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _group_color(label: str, step: int, norm: Normalize, cmap) -> str:
    if label == "Original":
        return ORIGINAL_COLOR
    if label == "step_004000":
        return STEP_4000_COLOR
    return cmap(norm(step))


def main() -> None:
    chembl_xy = pd.read_csv(OUT_DIR / "chembl_reference_coordinates.csv.gz").to_numpy()
    generated = pd.read_csv(OUT_DIR / "e2po_projected_coordinates.csv.gz")

    # ---- facets ----
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 8.0), sharex=True, sharey=True)
    for axis, label in zip(axes.ravel(), FACET_GROUPS):
        axis.scatter(
            chembl_xy[:, 0], chembl_xy[:, 1], s=0.5, color=CHEMBL_COLOR,
            alpha=0.09, linewidths=0, rasterized=True,
        )
        group = generated[generated["group"] == label]
        colour = ORIGINAL_COLOR if label == "Original" else (
            STEP_4000_COLOR if label == "step_004000" else "#7B4EA8"
        )
        axis.scatter(
            group["umap1"], group["umap2"], s=8.0, color=colour,
            alpha=0.55, linewidths=0, rasterized=True,
        )
        axis.set_title(f"{label}  |  n={len(group):,}", fontsize=16)
        axis.set_xlabel("UMAP 1", fontsize=14)
        axis.set_ylabel("UMAP 2", fontsize=14)
        axis.tick_params(labelsize=12)
    fig.suptitle("E2PO checkpoint coverage in one shared ChEMBL36 reference map", fontsize=20)
    fig.text(
        0.5, 0.008,
        "All panels share one fixed ChEMBL-fitted UMAP. Coordinate distance is descriptive "
        "neighbourhood structure, not efficacy or molecular quality.",
        ha="center", fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.955))
    _save_figure(fig, OUT_DIR / "e2po_chembl_umap_facets")

    # ---- drift (recompute summary from per-molecule distances) ----
    chembl_self_tree = None
    from scipy.spatial import cKDTree
    tree = cKDTree(chembl_xy)
    chembl_self = tree.query(chembl_xy, k=2, workers=-1)[0][:, 1]
    chembl_p95 = float(np.quantile(chembl_self, 0.95))

    rows = []
    for label in generated["group"].unique():
        g = generated[generated["group"] == label]
        step = int(g["step"].iloc[0])
        umap_dist = g["nearest_chembl_umap_distance"].to_numpy()
        rows.append({
            "group": label, "step": step, "n": int(len(g)),
            "nearest_chembl_umap_distance_median": float(np.median(umap_dist)),
            "nearest_chembl_tanimoto_distance_median": float(
                np.median(g["nearest_chembl_tanimoto_distance"].to_numpy())),
            "fraction_within_chembl_umap_p95": float(np.mean(umap_dist <= chembl_p95)),
        })
    summary = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.4))
    panels = [
        ("nearest_chembl_umap_distance_median", "Nearest ChEMBL distance (UMAP map)"),
        ("nearest_chembl_tanimoto_distance_median", "Nearest ChEMBL distance (ECFP4 Tanimoto)"),
        ("fraction_within_chembl_umap_p95", "Fraction inside ChEMBL P95 radius"),
    ]
    original = summary[summary["group"] == "Original"].iloc[0]
    train = summary[summary["step"] > 0].sort_values("step")
    for axis, (column, title) in zip(axes, panels):
        axis.plot(
            train["step"].to_numpy(), train[column].to_numpy(),
            marker="o", color="#7B4EA8", linewidth=2.2, markersize=7,
            label="E2PO checkpoints",
        )
        baseline = float(original[column])
        axis.axhline(baseline, color=ORIGINAL_COLOR, linestyle="--", linewidth=2.0,
                     alpha=0.9, label=f"Original: {baseline:.4f}")
        best = train.loc[train["group"] == "step_004000", column]
        if len(best):
            axis.scatter([4000], [float(best.iloc[0])], s=170, facecolors="none",
                         edgecolors=STEP_4000_COLOR, linewidths=2.4, zorder=5,
                         label="step_0040 (recommended)")
        axis.set_title(title, fontsize=15)
        axis.set_xlabel("training step", fontsize=13)
        axis.legend(fontsize=10)
    fig.suptitle("Distance-to-ChEMBL metrics across E2PO training", fontsize=18)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save_figure(fig, OUT_DIR / "e2po_chembl_drift_metrics")

    # ---- manifest ----
    summary_sorted = summary.sort_values("step")
    manifest = {
        "description": "E2PO checkpoints projected onto a fixed ChEMBL36 100k UMAP map",
        "chembl_reference_n": int(len(chembl_xy)),
        "generated_total_n": int(len(generated)),
        "groups": summary_sorted[["group", "step", "n"]].to_dict(orient="records"),
        "chembl_p95_umap_radius": chembl_p95,
        "outputs": [
            "chembl_reference_coordinates.csv.gz",
            "e2po_projected_coordinates.csv.gz",
            "chembl_distance_summary.csv",
            "e2po_chembl_umap_overview.png",
            "e2po_chembl_umap_facets.png",
            "e2po_chembl_drift_metrics.png",
        ],
    }
    (OUT_DIR / "projection_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("facets + drift + manifest regenerated")
    print(summary_sorted.to_string(index=False))


if __name__ == "__main__":
    main()
