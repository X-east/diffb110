#!/usr/bin/env python3
"""Dedicated structural-pathology breakdown for the two RL models: DiffSBDD and DiffGui.

Uses the SAME fixed joint ChEMBL36 / 5-model UMAP coordinates and the same category
definitions as analyze_umap_categories.py, but restricts the view to DiffSBDD
(retained RL comparison) and DiffGui (current RL base). Provides a focused set of
figures and tables:

  1. diffsbdd_diffgui_category_location.png  -- 2x3 grid (rows=model, cols=category),
     each panel highlights one pathology kind over that model's own molecules plus a
     faint ChEMBL36 backdrop.
  2. diffsbdd_diffgui_envelope_distance.png  -- per-model UMAP colored by distance to
     nearest ChEMBL36, with the ChEMBL self-p95 envelope boundary annotated.
  3. diffsbdd_diffgui_property_profile.png   -- median property comparison of pathology
     vs clean molecules within each model.

All distances and categories are computed on the already-fitted 2D projection; no
refitting is performed.
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


LOGGER = logging.getLogger("diffsbdd_diffgui_breakdown")

MODELS = ["DiffSBDD", "DiffGui"]
MODEL_COLORS = {"DiffSBDD": "#D95F59", "DiffGui": "#3B82C4"}
CHEMBL_COLOR = "#C9CDD2"
CHEMBL_SELF_NN_P95 = 0.024680944121926826  # from analyze_umap_distances.py

CATEGORY_STYLE = {
    "oversized_ring": {"color": "#E67E22", "label": "Oversized ring (>=9 mem.)"},
    "extreme_fused": {"color": "#8E44AD", "label": "Extreme fused/polycyclic (fused>=5 or rings>=6)"},
    "structural_alert": {"color": "#E74C3C", "label": "Structural alert (PAINS/NIH/Brenk)"},
}

PROFILE_PROPS = [
    ("mol_wt", "MW"),
    ("mol_log_p", "LogP"),
    ("qed", "QED"),
    ("sa_score", "SA"),
    ("tpsa", "TPSA"),
    ("total_ring_count", "Rings"),
    ("max_ring_size", "MaxRing"),
    ("carbon_fraction", "C-frac"),
    ("num_heteroatoms", "HeteroAtm"),
]


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
    file_handler = logging.FileHandler(output_dir / "diffsbdd_diffgui_breakdown.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(stream)
    root.addHandler(file_handler)


def _configure_plotting() -> None:
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 10, "axes.titlesize": 11.5, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 0.8,
        "legend.frameon": False, "svg.fonttype": "none", "pdf.fonttype": 42,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def _load(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    coords = pd.read_csv(args.coordinates, low_memory=False)
    if len(coords) != 110_910:
        raise ValueError(f"expected 110,910 coordinate rows, found {len(coords):,}")
    dist = pd.read_csv(args.distances, low_memory=False)
    quality = pd.read_csv(args.quality, low_memory=False)
    quality = quality[quality["rdkit_valid"] == 1].copy()
    merged = dist.merge(quality, on=["model", "source_sdf", "source_index"], how="left", suffixes=("", "_q"))
    merged = merged[merged["model"].isin(MODELS)].copy()
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


def _save(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _backdrop(ax, coords):
    chembl = coords[coords["model"] == "ChEMBL36"]
    ax.scatter(chembl["umap1"], chembl["umap2"], s=1.0, c=CHEMBL_COLOR, alpha=0.05, zorder=0)
    for m in MODELS:
        g = coords[coords["model"] == m]
        ax.scatter(g["umap1"], g["umap2"], s=2.5, c=MODEL_COLORS[m], alpha=0.22, zorder=1)


def _plot_category_location(coords, frame, out: Path) -> None:
    cats = ["oversized_ring", "extreme_fused", "structural_alert"]
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 10.5))
    for ri, model in enumerate(MODELS):
        mcoords = coords[coords["model"] == model]
        mframe = frame[frame["model"] == model]
        for ci, cat in enumerate(cats):
            ax = axes[ri, ci]
            _backdrop(ax, coords)
            norm = mframe[~mframe[cat]]
            ax.scatter(norm["umap1"], norm["umap2"], s=6, c=MODEL_COLORS[model],
                       alpha=0.30, zorder=2, label=f"{model} clean (n={len(norm)})")
            sub = mframe[mframe[cat]]
            ax.scatter(sub["umap1"], sub["umap2"], s=11, c=CATEGORY_STYLE[cat]["color"],
                       alpha=0.7, edgecolors="none", zorder=3,
                       label=f"{CATEGORY_STYLE[cat]['label']} (n={len(sub)})")
            n_out = int((~sub["inside_chembl_envelope"]).sum())
            ax.set_title(f"{model} | {cat.split('_')[0]} {cat.split('_')[1]}  ({n_out} outside env)",
                         loc="left", fontsize=10)
            ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
            ax.legend(loc="upper right", fontsize=6.8, framealpha=0.9)
    fig.suptitle("DiffSBDD vs DiffGui: where do structural-pathology molecules sit in the fixed joint UMAP?",
                 x=0.02, ha="left", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, out / "diffsbdd_diffgui_category_location")


def _plot_envelope_distance(coords, frame, out: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 11.0))
    for ri, model in enumerate(MODELS):
        ax = axes[ri]
        _backdrop(ax, coords)
        sub = frame[frame["model"] == model]
        sc = ax.scatter(sub["umap1"], sub["umap2"], s=7, c=sub["nearest_chembl_umap_distance"],
                        cmap="RdYlBu_r", vmin=0, vmax=0.15, alpha=0.75, zorder=3)
        nc = int((~sub["inside_chembl_envelope"]).sum())
        inside = int(sub["inside_chembl_envelope"].sum())
        ax.set_title(f"{model}: distance to nearest ChEMBL36  "
                     f"[{inside} inside (<=0.0247) / {nc} outside = {100*nc/len(sub):.1f}% out]",
                     loc="left", fontsize=11)
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
        ax.text(0.02, 0.02, f"ChEMBL self-p95 boundary = {CHEMBL_SELF_NN_P95:.4f}\n"
                            f"(molecules beyond this lie outside the ChEMBL envelope)",
                transform=ax.transAxes, fontsize=8.5, va="bottom",
                bbox=dict(boxstyle="round", fc="white", ec="#888888", alpha=0.85))
        cb = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.02)
        cb.set_label("UMAP dist. to nearest ChEMBL36")
    fig.suptitle("DiffSBDD vs DiffGui: per-molecule distance to ChEMBL36 in the fixed joint UMAP",
                 x=0.02, ha="left", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, out / "diffsbdd_diffgui_envelope_distance")


def _plot_property_profile(frame, out: Path) -> None:
    fig, axes = plt.subplots(1, len(PROFILE_PROPS), figsize=(2.55 * len(PROFILE_PROPS), 4.3))
    for ai, (col, lbl) in enumerate(PROFILE_PROPS):
        ax = axes[ai]
        x = np.arange(len(MODELS))
        w = 0.38
        clean_vals = []
        path_vals = []
        for m in MODELS:
            sub = frame[frame["model"] == m]
            clean = sub[~sub["any_category"]][col].dropna()
            path = sub[sub["any_category"]][col].dropna()
            clean_vals.append(clean.median() if len(clean) else np.nan)
            path_vals.append(path.median() if len(path) else np.nan)
        ax.bar(x - w / 2, clean_vals, w, color="#7FB3D5", label="Clean")
        ax.bar(x + w / 2, path_vals, w, color="#E74C3C", label="Pathology")
        ax.set_xticks(x, MODELS, rotation=15)
        ax.set_title(lbl, fontsize=10)
        for xi, (cv, pv) in enumerate(zip(clean_vals, path_vals)):
            if np.isfinite(cv):
                ax.text(xi - w / 2, cv, f"{cv:.2f}", ha="center", va="bottom", fontsize=6.5)
            if np.isfinite(pv):
                ax.text(xi + w / 2, pv, f"{pv:.2f}", ha="center", va="bottom", fontsize=6.5)
        if ai == 0:
            ax.legend(fontsize=7.5)
    fig.suptitle("Median property profile: pathology vs clean molecules (within each model)",
                 x=0.02, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, out / "diffsbdd_diffgui_property_profile")


def _summary_tables(frame: pd.DataFrame, coords: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for model in MODELS:
        sub = frame[frame["model"] == model]
        n = len(sub)
        rows.append({
            "model": model, "n": n,
            "inside_envelope": int(sub["inside_chembl_envelope"].sum()),
            "outside_envelope": int((~sub["inside_chembl_envelope"]).sum()),
            "outside_rate": float((~sub["inside_chembl_envelope"]).mean()),
            "oversized_ring": int(sub["oversized_ring"].sum()),
            "oversized_ring_rate": float(sub["oversized_ring"].mean()),
            "extreme_fused": int(sub["extreme_fused"].sum()),
            "extreme_fused_rate": float(sub["extreme_fused"].mean()),
            "structural_alert": int(sub["structural_alert"].sum()),
            "structural_alert_rate": float(sub["structural_alert"].mean()),
            "any_category": int(sub["any_category"].sum()),
            "any_category_rate": float(sub["any_category"].mean()),
        })
    breakdown = pd.DataFrame(rows)

    # location: within each model, how far out is each category (median ChEMBL dist) and outside rate
    loc_rows = []
    for model in MODELS:
        msub = frame[frame["model"] == model]
        for cat in ["oversized_ring", "extreme_fused", "structural_alert"]:
            sub = msub[msub[cat]]
            if len(sub) == 0:
                continue
            loc_rows.append({
                "model": model, "category": cat, "n": len(sub),
                "median_chembl_dist": float(sub["nearest_chembl_umap_distance"].median()),
                "outside_envelope": int((~sub["inside_chembl_envelope"]).sum()),
                "outside_rate": float((~sub["inside_chembl_envelope"]).mean()),
            })
    location = pd.DataFrame(loc_rows)

    # property profile CSV
    prof_rows = []
    for model in MODELS:
        sub = frame[frame["model"] == model]
        clean = sub[~sub["any_category"]]
        path = sub[sub["any_category"]]
        rec = {"model": model, "group": "clean", "n": len(clean)}
        rec.update({col: float(clean[col].median()) if clean[col].notna().any() else np.nan for col, _ in PROFILE_PROPS})
        prof_rows.append(rec)
        rec = {"model": model, "group": "pathology", "n": len(path)}
        rec.update({col: float(path[col].median()) if path[col].notna().any() else np.nan for col, _ in PROFILE_PROPS})
        prof_rows.append(rec)
    profile = pd.DataFrame(prof_rows)
    return breakdown, location, profile


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _configure_logging(args.output_dir)
    _configure_plotting()

    LOGGER.info("loading coordinate / distance / quality tables (DiffSBDD + DiffGui only)")
    coords, frame = _load(args)
    frame = _category_flags(frame)
    LOGGER.info("model molecules: %s", frame["model"].value_counts().to_dict())

    breakdown, location, profile = _summary_tables(frame, coords)
    breakdown.to_csv(args.output_dir / "diffsbdd_diffgui_category_breakdown.csv", index=False)
    location.to_csv(args.output_dir / "diffsbdd_diffgui_category_location.csv", index=False)
    profile.to_csv(args.output_dir / "diffsbdd_diffgui_property_profile.csv", index=False)

    LOGGER.info("\n%s", breakdown.to_string(index=False))
    LOGGER.info("\n%s", location.to_string(index=False))
    LOGGER.info("\n%s", profile.to_string(index=False))

    _plot_category_location(coords, frame, args.output_dir)
    _plot_envelope_distance(coords, frame, args.output_dir)
    _plot_property_profile(frame, args.output_dir)

    manifest = {
        "analysis": "dedicated DiffSBDD vs DiffGui structural-pathology breakdown on fixed joint UMAP",
        "scope_models": MODELS,
        "chembl_envelope_boundary": CHEMBL_SELF_NN_P95,
        "envelope_definition": "ChEMBL36 second-nearest-neighbour UMAP distance p95 (internal spread of ChEMBL itself)",
        "category_definitions": {
            "oversized_ring": "large_ring_count>0 OR max_ring_size>=9",
            "extreme_fused": "max_fused_ring_system_size>=5 OR total_ring_count>=6",
            "structural_alert": "pains_alert_count>0 OR nih_alert_count>0 OR brenk_alert_count>0",
        },
        "breakdown": breakdown.to_dict(orient="records"),
        "outputs": [
            "diffsbdd_diffgui_category_breakdown.csv",
            "diffsbdd_diffgui_category_location.csv",
            "diffsbdd_diffgui_property_profile.csv",
            "diffsbdd_diffgui_category_location.png",
            "diffsbdd_diffgui_envelope_distance.png",
            "diffsbdd_diffgui_property_profile.png",
        ],
        "interpretation_boundary": "UMAP is descriptive; 'inside/outside ChEMBL' is relative to this fixed 2D projection and not a statement of activity, docking, or absolute quality.",
    }
    with (args.output_dir / "manifest_diffsbdd_diffgui.json").open("w", encoding="utf-8") as h:
        json.dump(manifest, h, indent=2, ensure_ascii=False)
    LOGGER.info("completed successfully")


if __name__ == "__main__":
    main()
