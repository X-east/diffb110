#!/usr/bin/env python3
"""Spatial distribution of oversized rings, extreme fused rings, structural alerts,
and the true out-of-ChEMBL mass, in the fixed joint ChEMBL36/5-model UMAP.

Two important reframings versus the strict Tukey-fence view (analyze_umap_anomalies.py):

1. "Inside ChEMBL space" is defined by ChEMBL36's OWN internal spread
   (second-nearest-neighbour p95 = 0.0247), not by the generated-distance Tukey
   fence. Using that reference, a large fraction of generated molecules already
   lie outside the ChEMBL envelope.
2. The three structural issues (oversized rings, extreme fused rings, structural
   alerts) are population-scale, not tail phenomena. This script maps where each
   kind sits in UMAP and quantifies the out-of-ChEMBL mass per model.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/diffb110_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


LOGGER = logging.getLogger("five_model_umap_categories")

MODEL_ORDER = ["DiffGui", "DiffSBDD", "PMDM", "GCDM-SBDD", "TargetDiff"]
MODEL_COLORS = {
    "DiffGui": "#3B82C4",
    "DiffSBDD": "#D95F59",
    "PMDM": "#D9A441",
    "GCDM-SBDD": "#8E78B7",
    "TargetDiff": "#3D9B72",
}
CHEMBL_COLOR = "#C9CDD2"
CHEMBL_SELF_NN_P95 = 0.024680944121926826  # from analyze_umap_distances.py

# category display
CATEGORY_STYLE = {
    "oversized_ring": {"color": "#E67E22", "label": "Oversized ring (>=9 mem.)"},
    "extreme_fused": {"color": "#8E44AD", "label": "Extreme fused/polycyclic (fused>=5 or rings>=6)"},
    "structural_alert": {"color": "#E74C3C", "label": "Structural alert (PAINS/NIH/Brenk)"},
}


def parse_args() -> argparse.Namespace:
    default_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordinates", type=Path, default=default_dir / "joint_umap_coordinates.csv.gz")
    parser.add_argument("--distances", type=Path, default=default_dir / "nearest_chembl_distance_per_generated_molecule.csv.gz")
    parser.add_argument(
        "--quality", type=Path,
        default=default_dir / ".." / "8gjl_c1101_benchmark_20260706_1000steps_3rep_formal"
        / "raw_quality_vs_chembl_20260728" / "per_molecule_raw_quality.csv.gz",
    )
    parser.add_argument("--output-dir", type=Path, default=default_dir)
    return parser.parse_args()


def _configure_logging(output_dir: Path) -> None:
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "category_analysis.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(stream)
    root.addHandler(file_handler)


def _configure_plotting() -> None:
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 13, "axes.titlesize": 15, "axes.labelsize": 14,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
        "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 0.8,
        "legend.frameon": False, "svg.fonttype": "none", "pdf.fonttype": 42,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def _load(args: argparse.Namespace) -> pd.DataFrame:
    coords = pd.read_csv(args.coordinates, low_memory=False)
    if len(coords) != 110_910:
        raise ValueError(f"expected 110,910 coordinate rows, found {len(coords):,}")
    dist = pd.read_csv(args.distances, low_memory=False)
    quality = pd.read_csv(args.quality, low_memory=False)
    quality = quality[quality["rdkit_valid"] == 1].copy()
    merged = dist.merge(quality, on=["model", "source_sdf", "source_index"], how="left", suffixes=("", "_q"))
    return coords, merged


def _category_flags(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()

    def oversized_ring(r):
        return (pd.notna(r.get("large_ring_count")) and r["large_ring_count"] > 0) or \
               (pd.notna(r["max_ring_size"]) and r["max_ring_size"] >= 9)

    def extreme_fused(r):
        return (pd.notna(r["max_fused_ring_system_size"]) and r["max_fused_ring_system_size"] >= 5) or \
               (pd.notna(r["total_ring_count"]) and r["total_ring_count"] >= 6)

    def structural_alert(r):
        return (pd.notna(r.get("pains_alert_count")) and r["pains_alert_count"] > 0) or \
               (pd.notna(r.get("nih_alert_count")) and r["nih_alert_count"] > 0) or \
               (pd.notna(r.get("brenk_alert_count")) and r["brenk_alert_count"] > 0)

    frame["oversized_ring"] = frame.apply(oversized_ring, axis=1)
    frame["extreme_fused"] = frame.apply(extreme_fused, axis=1)
    frame["structural_alert"] = frame.apply(structural_alert, axis=1)
    frame["any_category"] = frame[["oversized_ring", "extreme_fused", "structural_alert"]].any(axis=1)
    frame["inside_chembl_envelope"] = frame["nearest_chembl_umap_distance"] <= CHEMBL_SELF_NN_P95
    return frame


def _summary_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total = len(frame)
    inside = int(frame["inside_chembl_envelope"].sum())

    overview = pd.DataFrame([{
        "metric": "generated molecules", "value": total, "fraction": 1.0,
    }, {
        "metric": "inside ChEMBL self-p95 envelope (<=0.0247)", "value": inside,
        "fraction": inside / total,
    }, {
        "metric": "OUTSIDE ChEMBL envelope", "value": total - inside,
        "fraction": (total - inside) / total,
    }, {
        "metric": "oversized_ring", "value": int(frame["oversized_ring"].sum()),
        "fraction": frame["oversized_ring"].mean(),
    }, {
        "metric": "extreme_fused", "value": int(frame["extreme_fused"].sum()),
        "fraction": frame["extreme_fused"].mean(),
    }, {
        "metric": "structural_alert", "value": int(frame["structural_alert"].sum()),
        "fraction": frame["structural_alert"].mean(),
    }, {
        "metric": "any of three categories", "value": int(frame["any_category"].sum()),
        "fraction": frame["any_category"].mean(),
    }])

    by_model = []
    for model in MODEL_ORDER:
        sub = frame[frame["model"] == model]
        n = len(sub)
        by_model.append({
            "model": model, "n": n,
            "inside_envelope": int(sub["inside_chembl_envelope"].sum()),
            "outside_envelope": int((~sub["inside_chembl_envelope"]).sum()),
            "outside_rate": float((~sub["inside_chembl_envelope"]).mean()),
            "oversized_ring": int(sub["oversized_ring"].sum()),
            "extreme_fused": int(sub["extreme_fused"].sum()),
            "structural_alert": int(sub["structural_alert"].sum()),
            "any_category": int(sub["any_category"].sum()),
            "any_category_rate": float(sub["any_category"].mean()),
        })
    by_model = pd.DataFrame(by_model)

    # where each category sits: nearest model centroid
    coords = pd.read_csv(args_coords, low_memory=False)
    centroids = {m: coords[coords["model"] == m][["umap1", "umap2"]].mean().to_numpy() for m in MODEL_ORDER}
    xy = frame[["umap1", "umap2"]].to_numpy()
    dist_mat = np.stack([np.sqrt(((xy - c) ** 2).sum(1)) for c in centroids.values()], 1)
    nearest = np.argmin(dist_mat, 1)
    frame["nearest_model_centroid"] = [MODEL_ORDER[i] for i in nearest]

    sit_rows = []
    for cat in ["oversized_ring", "extreme_fused", "structural_alert"]:
        sub = frame[frame[cat]]
        vc = sub["nearest_model_centroid"].value_counts().reindex(MODEL_ORDER).fillna(0).astype(int)
        rec = {"category": cat}
        for m in MODEL_ORDER:
            rec[m] = int(vc[m])
        rec["n"] = len(sub)
        sit_rows.append(rec)
    sit = pd.DataFrame(sit_rows)
    return overview, by_model, sit


def _save(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _backdrop(ax, coords):
    chembl = coords[coords["model"] == "ChEMBL36"]
    ax.scatter(chembl["umap1"], chembl["umap2"], s=1.0, c=CHEMBL_COLOR, alpha=0.06, zorder=0)
    for m in MODEL_ORDER:
        g = coords[coords["model"] == m]
        ax.scatter(g["umap1"], g["umap2"], s=2.5, c=MODEL_COLORS[m], alpha=0.18, zorder=1)


def _plot_category_panels(coords, frame, out: Path) -> None:
    cats = ["oversized_ring", "extreme_fused", "structural_alert"]
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 12.0))
    axes = axes.ravel()
    for ax, cat in zip(axes, cats):
        _backdrop(ax, coords)
        sub = frame[frame[cat]]
        ax.scatter(sub["umap1"], sub["umap2"], s=10, c=CATEGORY_STYLE[cat]["color"],
                   alpha=0.55, edgecolors="none", zorder=3, label=f"{CATEGORY_STYLE[cat]['label']} (n={len(sub)})")
        ax.set_title(CATEGORY_STYLE[cat]["label"], loc="left", fontsize=14)
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
        ax.legend(loc="upper right", fontsize=10.5)
    # 4th panel: combined continuous distance map (the "outside" mass)
    ax = axes[3]
    _backdrop(ax, coords)
    gen = frame
    sc = ax.scatter(gen["umap1"], gen["umap2"], s=4, c=gen["nearest_chembl_umap_distance"],
                   cmap="RdYlBu_r", vmin=0, vmax=0.20, alpha=0.7, zorder=3)
    cb = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("UMAP dist. to nearest ChEMBL36")
    ax.axhline(np.nan, label="")  # placeholder
    ax.set_title("All generated molecules colored by distance to ChEMBL", loc="left", fontsize=14)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    fig.suptitle("Where do oversized rings, extreme fused rings, structural alerts, and out-of-ChEMBL molecules sit?",
                 x=0.02, ha="left", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, out / "category_spatial_panels")


def _plot_outside_mass(frame, by_model, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.2))
    bm = by_model.set_index("model").loc[MODEL_ORDER].reset_index()

    ax = axes[0]
    x = np.arange(len(bm))
    ax.bar(x, bm["inside_envelope"], color="#7FB3D5", label="Inside ChEMBL envelope")
    ax.bar(x, bm["outside_envelope"], bottom=bm["inside_envelope"], color="#E74C3C",
           label="Outside ChEMBL envelope")
    for i, (ins, out_n) in enumerate(zip(bm["inside_envelope"], bm["outside_envelope"])):
        ax.text(i, ins + out_n + 30, f"{out_n}/{ins+out_n}\n({100*out_n/(ins+out_n):.0f}% out)",
                ha="center", fontsize=10.5)
    ax.set_xticks(x, bm["model"], rotation=20, ha="right")
    ax.set_ylabel("Molecule count")
    ax.set_title("a  Inside vs OUTSIDE ChEMBL envelope (p95 self-NN = 0.0247)", loc="left")
    ax.legend(fontsize=11)

    ax = axes[1]
    x = np.arange(len(bm))
    w = 0.27
    ax.bar(x - w, bm["oversized_ring"], w, color=CATEGORY_STYLE["oversized_ring"]["color"], label="Oversized ring")
    ax.bar(x, bm["extreme_fused"], w, color=CATEGORY_STYLE["extreme_fused"]["color"], label="Extreme fused")
    ax.bar(x + w, bm["structural_alert"], w, color=CATEGORY_STYLE["structural_alert"]["color"], label="Structural alert")
    ax.set_xticks(x, bm["model"], rotation=20, ha="right")
    ax.set_ylabel("Molecule count")
    ax.set_title("b  Three structural-issue counts per model", loc="left")
    ax.legend(fontsize=11)

    fig.tight_layout()
    _save(fig, out / "outside_mass_and_categories")


def main() -> None:
    global args_coords
    args = parse_args()
    args_coords = args.coordinates
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _configure_logging(args.output_dir)
    _configure_plotting()

    LOGGER.info("loading coordinate / distance / quality tables")
    coords, frame = _load(args)
    frame = _category_flags(frame)
    LOGGER.info("generated molecules: %d", len(frame))

    overview, by_model, sit = _summary_tables(frame)
    overview.to_csv(args.output_dir / "category_overview.csv", index=False)
    by_model.to_csv(args.output_dir / "category_by_model.csv", index=False)
    sit.to_csv(args.output_dir / "category_spatial_location.csv", index=False)

    LOGGER.info("\n%s", overview.to_string(index=False))
    LOGGER.info("\n%s", by_model.to_string(index=False))
    LOGGER.info("\n%s", sit.to_string(index=False))

    _plot_category_panels(coords, frame, args.output_dir)
    _plot_outside_mass(frame, by_model, args.output_dir)

    manifest = {
        "analysis": "spatial distribution of oversized rings / extreme fused rings / structural alerts and out-of-ChEMBL mass",
        "chembl_envelope_boundary": CHEMBL_SELF_NN_P95,
        "envelope_definition": "ChEMBL36 second-nearest-neighbour UMAP distance p95 (internal spread of ChEMBL itself)",
        "category_definitions": {
            "oversized_ring": "large_ring_count>0 OR max_ring_size>=9",
            "extreme_fused": "max_fused_ring_system_size>=5 OR total_ring_count>=6",
            "structural_alert": "pains_alert_count>0 OR nih_alert_count>0 OR brenk_alert_count>0",
        },
        "overview": overview.to_dict(orient="records"),
        "outputs": [
            "category_overview.csv", "category_by_model.csv", "category_spatial_location.csv",
            "category_spatial_panels.png", "outside_mass_and_categories.png",
        ],
        "interpretation_boundary": "UMAP is descriptive; 'inside/outside ChEMBL' is relative to this fixed 2D projection and not a statement of activity, docking, or absolute quality.",
    }
    with (args.output_dir / "manifest_categories.json").open("w", encoding="utf-8") as h:
        json.dump(manifest, h, indent=2, ensure_ascii=False)
    LOGGER.info("completed successfully")


if __name__ == "__main__":
    main()
