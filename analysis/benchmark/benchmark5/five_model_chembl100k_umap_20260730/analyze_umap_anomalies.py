#!/usr/bin/env python3
"""Identify and characterize anomalous generated molecules in the fixed joint UMAP.

An anomalous molecule is a sanitize-valid generated molecule that lies far from
the ChEMBL36 100k cloud in the fixed 2D joint ECFP4-UMAP projection developed in
``plot_five_model_chembl100k_umap.py``. This script does NOT refit UMAP; it only
consumes the already-published coordinates, the per-molecule nearest-ChEMBL
distance, and the per-molecule raw-quality descriptors for the same 10,910
generated molecules.

Two anomaly tiers are used, both derived from the generated-distance distribution
(``nearest_chembl_umap_distance``):

* **anomaly**  : above the Tukey upper fence  Q3 + 1.5*IQR
* **extreme**  : above the Tukey extreme fence  Q3 + 3.0*IQR

The script then answers the two questions in section 3.5:

1. Where do anomalous molecules sit in UMAP space?
   (UMAP extents, centroid offset, nearest-group assignment, local clustering)
2. What kinds of anomalous molecules are there?
   (per-model origin, a rule-based molecular-feature taxonomy, and example SMILES)
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


LOGGER = logging.getLogger("five_model_umap_anomaly")

MODEL_ORDER = ["DiffGui", "DiffSBDD", "PMDM", "GCDM-SBDD", "TargetDiff"]
GROUP_ORDER = ["ChEMBL36", *MODEL_ORDER]
MODEL_COLORS = {
    "DiffGui": "#3B82C4",
    "DiffSBDD": "#D95F59",
    "PMDM": "#D9A441",
    "GCDM-SBDD": "#8E78B7",
    "TargetDiff": "#3D9B72",
}
CHEMBL_COLOR = "#9AA0A6"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    default_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coordinates",
        type=Path,
        default=default_dir / "joint_umap_coordinates.csv.gz",
    )
    parser.add_argument(
        "--distances",
        type=Path,
        default=default_dir / "nearest_chembl_distance_per_generated_molecule.csv.gz",
    )
    parser.add_argument(
        "--quality",
        type=Path,
        default=default_dir
        / ".."
        / "8gjl_c1101_benchmark_20260706_1000steps_3rep_formal"
        / "raw_quality_vs_chembl_20260728"
        / "per_molecule_raw_quality.csv.gz",
    )
    parser.add_argument("--output-dir", type=Path, default=default_dir)
    return parser.parse_args()


def _configure_logging(output_dir: Path) -> None:
    """Configure terminal and durable file logging."""
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(
        output_dir / "anomaly_analysis.log", mode="w", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    root.addHandler(stream)
    root.addHandler(file_handler)


def _configure_plotting() -> None:
    """Apply slide-readable Python plotting defaults."""
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 13,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


# --- molecular-feature taxonomy -------------------------------------------------
# Each predicate returns True when the molecule belongs to that anomaly "kind".
# Priority order (most structurally severe first) decides the *primary* kind;
# co-occurrence is still recorded separately.
KIND_RULES = [
    (
        "carbon_greasy",
        "All-carbon or carbon-enriched, low-heteroatom molecules (greasy debris).",
        lambda r: bool(r["all_carbon"] == 1)
        or bool(r["carbon_enriched_low_hetero"] == 1)
        or (pd.notna(r["carbon_fraction"]) and r["carbon_fraction"] > 0.92),
    ),
    (
        "hypercyclic",
        "Very high ring counts or large fused ring systems.",
        lambda r: (pd.notna(r["total_ring_count"]) and r["total_ring_count"] >= 5)
        or (pd.notna(r["max_fused_ring_system_size"]) and r["max_fused_ring_system_size"] >= 5)
        or (pd.notna(r["ring_density"]) and r["ring_density"] > 0.45),
    ),
    (
        "oversized",
        "Molecular weight or heavy-atom count far above drug-like range.",
        lambda r: (pd.notna(r["mol_wt"]) and r["mol_wt"] > 500)
        or (pd.notna(r["heavy_atoms"]) and r["heavy_atoms"] > 45),
    ),
    (
        "hydrophobic",
        "Extreme lipophilicity (cLogP well above 5).",
        lambda r: pd.notna(r["mol_log_p"]) and r["mol_log_p"] > 5.0,
    ),
    (
        "alert_laden",
        "Carries PAINS / NIH / Brenk structural-alert substructures.",
        lambda r: (
            (pd.notna(r["pains_alert_count"]) and r["pains_alert_count"] > 0)
            or (pd.notna(r["nih_alert_count"]) and r["nih_alert_count"] > 0)
            or (pd.notna(r["brenk_alert_count"]) and r["brenk_alert_count"] > 0)
        ),
    ),
    (
        "tiny",
        "Very small molecules (MW below 150).",
        lambda r: pd.notna(r["mol_wt"]) and r["mol_wt"] < 150,
    ),
    (
        "halogen_rich",
        "Unusually high halogen fraction.",
        lambda r: pd.notna(r["halogen_fraction"]) and r["halogen_fraction"] > 0.30,
    ),
    (
        "low_complexity",
        "Low Bertz structural complexity for its size.",
        lambda r: pd.notna(r["bertz_complexity"]) and r["bertz_complexity"] < 200,
    ),
]


def _assign_kinds(row: pd.Series) -> tuple[str, list[str]]:
    """Return (primary_kind, all_kinds) for one anomaly molecule."""
    matched = [name for name, _desc, pred in KIND_RULES if pred(row)]
    if not matched:
        return "other_off_manifold", ["other_off_manifold"]
    return matched[0], matched


def _load_and_merge(args: argparse.Namespace) -> pd.DataFrame:
    """Load coordinates, distances, raw quality, and merge on the generation key."""
    coords = pd.read_csv(args.coordinates, low_memory=False)
    if len(coords) != 110_910:
        raise ValueError(f"expected 110,910 coordinate rows, found {len(coords):,}")
    dist = pd.read_csv(args.distances, low_memory=False)
    quality = pd.read_csv(args.quality, low_memory=False)
    quality = quality[quality["rdkit_valid"] == 1].copy()

    merged = dist.merge(
        quality,
        on=["model", "source_sdf", "source_index"],
        how="left",
        suffixes=("", "_q"),
    )
    missing = merged["mol_wt"].isna().sum()
    if missing:
        LOGGER.warning("property columns missing for %d / %d molecules", missing, len(merged))
    return merged


def _flag_anomalies(frame: pd.DataFrame) -> pd.DataFrame:
    """Add anomaly tiers and per-molecule kind annotations."""
    d = frame["nearest_chembl_umap_distance"].to_numpy(dtype=float)
    q1, q3 = np.percentile(d, [25, 75])
    iqr = q3 - q1
    fence = q3 + 1.5 * iqr
    fence_extreme = q3 + 3.0 * iqr
    frame = frame.copy()
    frame["anomaly_tier"] = np.where(
        d > fence_extreme, "extreme", np.where(d > fence, "anomaly", "normal")
    )
    frame["is_anomaly"] = frame["anomaly_tier"] != "normal"
    meta = {
        "distance_min": float(d.min()),
        "distance_median": float(np.median(d)),
        "distance_mean": float(d.mean()),
        "distance_q90": float(np.percentile(d, 90)),
        "distance_q95": float(np.percentile(d, 95)),
        "distance_q99": float(np.percentile(d, 99)),
        "distance_max": float(d.max()),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "tukey_fence": float(fence),
        "tukey_extreme_fence": float(fence_extreme),
        "n_anomaly": int((frame["anomaly_tier"] == "anomaly").sum()),
        "n_extreme": int((frame["anomaly_tier"] == "extreme").sum()),
        "n_total_generated": int(len(frame)),
    }

    kinds = frame[frame["is_anomaly"]].apply(_assign_kinds, axis=1)
    primary = pd.Series([k[0] for k in kinds], index=kinds.index)
    allk = pd.Series([";".join(k[1]) for k in kinds], index=kinds.index)
    frame["primary_kind"] = "normal"
    frame.loc[primary.index, "primary_kind"] = primary
    frame["all_kinds"] = ""
    frame.loc[allk.index, "all_kinds"] = allk
    return frame, meta


def _nearest_group(frame: pd.DataFrame) -> pd.DataFrame:
    """Assign each anomaly to its nearest group centroid in UMAP space."""
    coords = pd.read_csv(args_for_centroid.coordinates, low_memory=False)
    centroids = {
        g: coords[coords["model"] == g][["umap1", "umap2"]].mean().to_numpy()
        for g in GROUP_ORDER
    }
    xy = frame[["umap1", "umap2"]].to_numpy()
    rows = []
    for g in GROUP_ORDER:
        c = centroids[g]
        rows.append(np.sqrt(((xy - c) ** 2).sum(axis=1)))
    dist_mat = np.stack(rows, axis=1)
    nearest = np.argmin(dist_mat, axis=1)
    frame = frame.copy()
    frame["nearest_group_centroid"] = [GROUP_ORDER[i] for i in nearest]
    # distance to ChEMBL centroid (periphery measure)
    chembl_c = centroids["ChEMBL36"]
    frame["umap_distance_to_chembl_centroid"] = np.sqrt(
        ((xy - chembl_c) ** 2).sum(axis=1)
    )
    return frame


def _summarize_by_model(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-model anomaly counts and rates."""
    rows = []
    for model in MODEL_ORDER:
        sub = frame[frame["model"] == model]
        n_total = len(sub)
        n_anom = int(sub["is_anomaly"].sum())
        n_ext = int((sub["anomaly_tier"] == "extreme").sum())
        rows.append(
            {
                "model": model,
                "n_generated": n_total,
                "n_anomaly": n_anom,
                "n_extreme": n_ext,
                "anomaly_rate": n_anom / n_total if n_total else float("nan"),
                "extreme_rate": n_ext / n_total if n_total else float("nan"),
                "median_nearest_chembl_distance": float(sub["nearest_chembl_umap_distance"].median()),
            }
        )
    return pd.DataFrame(rows)


def _property_profile(frame: pd.DataFrame) -> pd.DataFrame:
    """Compare anomaly vs normal generated on key descriptors + category prevalence."""
    anom = frame[frame["is_anomaly"]]
    normal = frame[~frame["is_anomaly"]]
    props = [
        ("mol_wt", "Molecular weight"),
        ("mol_log_p", "cLogP"),
        ("heavy_atoms", "Heavy atoms"),
        ("total_ring_count", "Ring count"),
        ("carbon_fraction", "Carbon fraction"),
        ("tpsa", "TPSA"),
        ("qed", "QED"),
        ("sa_score", "SA score"),
        ("fraction_csp3", "Frac. sp3"),
        ("num_rotatable_bonds", "Rotatable bonds"),
        ("bertz_complexity", "Bertz complexity"),
    ]
    rows = []
    for col, label in props:
        a = pd.to_numeric(anom[col], errors="coerce").dropna()
        n = pd.to_numeric(normal[col], errors="coerce").dropna()
        rows.append(
            {
                "property": label,
                "anomaly_median": float(a.median()) if len(a) else float("nan"),
                "normal_median": float(n.median()) if len(n) else float("nan"),
                "anomaly_mean": float(a.mean()) if len(a) else float("nan"),
                "normal_mean": float(n.mean()) if len(n) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _kind_breakdown(frame: pd.DataFrame, by_model: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Primary-kind counts and per-kind co-occurrence flags, plus per-model kind matrix."""
    anom = frame[frame["is_anomaly"]].copy()
    primary_counts = anom["primary_kind"].value_counts()
    # ordered by our taxonomy appearance
    primary_order = [name for name, _d, _p in KIND_RULES] + ["other_off_manifold"]
    primary_counts = primary_counts.reindex([k for k in primary_order if k in primary_counts.index])
    primary_df = primary_counts.rename_axis("primary_kind").reset_index(name="count")
    primary_df["fraction_of_anomalies"] = primary_df["count"] / len(anom)

    # per-model x kind matrix (a molecule may count in multiple kinds via all_kinds)
    matrix_rows = []
    for model in MODEL_ORDER:
        sub = anom[anom["model"] == model]
        rec = {"model": model, "n_anomaly": len(sub)}
        for name, _d, _p in KIND_RULES:
            rec[name] = int(sub["all_kinds"].str.contains(rf"\b{name}\b").sum())
        rec["other_off_manifold"] = int((sub["primary_kind"] == "other_off_manifold").sum())
        matrix_rows.append(rec)
    matrix_df = pd.DataFrame(matrix_rows)
    return primary_df, matrix_df


def _save_figure(fig: plt.Figure, stem: Path) -> None:
    """Export one Python-rendered figure in four formats."""
    fig.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def _plot_overview(coords: pd.DataFrame, frame: pd.DataFrame, output_dir: Path) -> None:
    """UMAP overview: ChEMBL faint, 5 model clouds, anomalies as red stars."""
    fig, ax = plt.subplots(figsize=(11.6, 8.4))
    chembl = coords[coords["model"] == "ChEMBL36"]
    ax.scatter(
        chembl["umap1"], chembl["umap2"], s=1.5, c=CHEMBL_COLOR, alpha=0.10, label="ChEMBL36 100k", zorder=1,
    )
    for model in MODEL_ORDER:
        g = coords[coords["model"] == model]
        ax.scatter(
            g["umap1"], g["umap2"], s=4, c=MODEL_COLORS[model], alpha=0.45, label=model, zorder=2,
        )
    anom = frame[frame["is_anomaly"]]
    sc = ax.scatter(
        anom["umap1"], anom["umap2"],
        s=14 + 60 * (anom["nearest_chembl_umap_distance"] / anom["nearest_chembl_umap_distance"].max()),
        c="#E74C3C", edgecolors="black", linewidths=0.3, marker="*",
        label=f"Anomalies (n={len(anom)})", zorder=4,
    )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("a  Anomalous generated molecules in the joint ChEMBL36/5-model UMAP", loc="left")
    ax.legend(loc="upper right", fontsize=11, markerscale=1.4)
    fig.text(
        0.5, 0.005,
        "Red stars = molecules whose nearest-ChEMBL UMAP distance exceeds the Tukey fence (Q3+1.5*IQR). "
        "Star size scales with the distance.",
        ha="center", fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    _save_figure(fig, output_dir / "anomaly_umap_overview")


def _plot_spatial_hist(frame: pd.DataFrame, output_dir: Path) -> None:
    """2D density of anomalies + nearest-group membership pie for anomalies."""
    anom = frame[frame["is_anomaly"]]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.0))

    ax = axes[0]
    ax.hist2d(anom["umap1"], anom["umap2"], bins=40, cmap="Reds", cmin=1)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("b  Spatial density of anomalies", loc="left")

    ax = axes[1]
    counts = anom["nearest_group_centroid"].value_counts().reindex(GROUP_ORDER).fillna(0)
    colors = [MODEL_COLORS.get(g, CHEMBL_COLOR) for g in counts.index]
    ax.bar(counts.index, counts.values, color=colors)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.5, int(v), ha="center", fontsize=11)
    ax.set_ylabel("Anomaly count")
    ax.set_title("c  Nearest group centroid for each anomaly", loc="left")
    ax.tick_params(axis="x", rotation=25)

    fig.tight_layout()
    _save_figure(fig, output_dir / "anomaly_spatial_summary")


def _plot_kind_bars(primary_df: pd.DataFrame, matrix_df: pd.DataFrame, output_dir: Path) -> None:
    """Bar chart of primary kinds and a per-model x kind heatmap."""
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.4))

    ax = axes[0]
    ax.bar(primary_df["primary_kind"], primary_df["count"], color="#E74C3C")
    for i, v in enumerate(primary_df["count"]):
        ax.text(i, v + 0.5, int(v), ha="center", fontsize=11)
    ax.set_xticklabels(primary_df["primary_kind"], rotation=30, ha="right")
    ax.set_ylabel("Anomaly count")
    ax.set_title("d  Primary anomaly kind (mutually exclusive)", loc="left")

    ax = axes[1]
    kind_cols = [name for name, _d, _p in KIND_RULES] + ["other_off_manifold"]
    mat = matrix_df.set_index("model")[kind_cols]
    im = ax.imshow(mat.values, aspect="auto", cmap="OrRd")
    ax.set_xticks(range(len(kind_cols)))
    ax.set_xticklabels(kind_cols, rotation=40, ha="right", fontsize=10)
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(mat.index, fontsize=11)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = int(mat.values[i, j])
            if val:
                ax.text(j, i, val, ha="center", va="center", fontsize=10,
                        color="black" if val < mat.values.max() * 0.6 else "white")
    ax.set_title("e  Anomaly kinds by source model (co-occurrence counted)", loc="left")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="count")

    fig.tight_layout()
    _save_figure(fig, output_dir / "anomaly_kind_breakdown")


def _plot_property_compare(profile: pd.DataFrame, frame: pd.DataFrame, output_dir: Path) -> None:
    """Violin/box comparison of key properties: anomalies vs normal generated."""
    anom = frame[frame["is_anomaly"]]
    normal = frame[~frame["is_anomaly"]]
    props = [
        ("mol_wt", "MW"),
        ("mol_log_p", "cLogP"),
        ("heavy_atoms", "Heavy atoms"),
        ("total_ring_count", "Rings"),
        ("carbon_fraction", "C frac."),
        ("tpsa", "TPSA"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.2))
    axes = axes.ravel()
    for ax, (col, label) in zip(axes, props):
        a = pd.to_numeric(anom[col], errors="coerce").dropna()
        n = pd.to_numeric(normal[col], errors="coerce").dropna()
        ax.violinplot([n, a], showmedians=True, widths=0.8)
        ax.set_xticks([1, 2], ["normal", "anomaly"])
        ax.set_title(label, loc="left", fontsize=14)
    fig.suptitle("f  Property distributions: anomalous vs normal generated molecules", x=0.02, ha="left", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save_figure(fig, output_dir / "anomaly_property_comparison")


def _examples(frame: pd.DataFrame, n_per_kind: int = 5) -> pd.DataFrame:
    """Pick the most extreme (largest distance) examples per primary kind."""
    anom = frame[frame["is_anomaly"]]
    rows = []
    for kind in anom["primary_kind"].unique():
        sub = anom[anom["primary_kind"] == kind].nlargest(n_per_kind, "nearest_chembl_umap_distance")
        for _, r in sub.iterrows():
            rows.append(
                {
                    "primary_kind": kind,
                    "model": r["model"],
                    "molecule_id": r["molecule_id"],
                    "canonical_smiles": r["canonical_smiles"],
                    "nearest_chembl_umap_distance": r["nearest_chembl_umap_distance"],
                    "mol_wt": r["mol_wt"],
                    "mol_log_p": r["mol_log_p"],
                    "total_ring_count": r["total_ring_count"],
                    "carbon_fraction": r["carbon_fraction"],
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    """Run the anomalous-molecule analysis and export all artifacts."""
    global args_for_centroid
    args = parse_args()
    args_for_centroid = argparse.Namespace(coordinates=args.coordinates)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _configure_logging(args.output_dir)
    _configure_plotting()

    LOGGER.info("loading & merging coordinate / distance / quality tables")
    frame = _load_and_merge(args)
    LOGGER.info("merged generated molecules: %d", len(frame))

    frame, meta = _flag_anomalies(frame)
    LOGGER.info(
        "anomaly fence=%.4f extreme fence=%.4f  n_anomaly=%d n_extreme=%d",
        meta["tukey_fence"], meta["tukey_extreme_fence"], meta["n_anomaly"], meta["n_extreme"],
    )

    frame = _nearest_group(frame)

    by_model = _summarize_by_model(frame)
    profile = _property_profile(frame)
    primary_df, matrix_df = _kind_breakdown(frame, by_model)
    examples = _examples(frame)

    # anomaly molecule export (all anomaly rows + key columns)
    export_cols = [
        "model", "repeat", "molecule_id", "canonical_smiles", "source_sdf", "source_index",
        "umap1", "umap2", "nearest_chembl_umap_distance", "umap_distance_to_chembl_centroid",
        "anomaly_tier", "primary_kind", "all_kinds", "nearest_group_centroid",
        "mol_wt", "mol_log_p", "heavy_atoms", "total_ring_count", "aromatic_ring_count",
        "max_fused_ring_system_size", "carbon_fraction", "halogen_fraction", "tpsa",
        "qed", "sa_score", "num_rotatable_bonds", "bertz_complexity",
        "pains_alert_count", "nih_alert_count", "brenk_alert_count",
    ]
    anomaly_export = frame[frame["is_anomaly"]][export_cols].copy()
    anomaly_export = anomaly_export.sort_values("nearest_chembl_umap_distance", ascending=False)

    out = args.output_dir
    by_model.to_csv(out / "anomaly_summary_by_model.csv", index=False)
    profile.to_csv(out / "anomaly_property_profile.csv", index=False)
    primary_df.to_csv(out / "anomaly_kind_primary.csv", index=False)
    matrix_df.to_csv(out / "anomaly_kind_by_model.csv", index=False)
    examples.to_csv(out / "anomaly_examples_by_kind.csv", index=False)
    anomaly_export.to_csv(out / "anomaly_molecules.csv.gz", index=False, compression="gzip")

    # figures
    coords = pd.read_csv(args.coordinates, low_memory=False)
    _plot_overview(coords, frame, out)
    _plot_spatial_hist(frame, out)
    _plot_kind_bars(primary_df, matrix_df, out)
    _plot_property_compare(profile, frame, out)

    # manifest fragment
    manifest_path = out / "manifest_anomaly.json"
    manifest = {
        "analysis": "anomalous generated molecules in the fixed joint ChEMBL36/5-model UMAP",
        "distance_definition": "Euclidean distance in 2D UMAP from each generated point to its nearest ChEMBL36 point",
        "anomaly_tiers": {
            "anomaly": {"rule": "nearest_chembl_umap_distance > Q3 + 1.5*IQR",
                         "threshold": meta["tukey_fence"], "n": meta["n_anomaly"]},
            "extreme": {"rule": "nearest_chembl_umap_distance > Q3 + 3.0*IQR",
                         "threshold": meta["tukey_extreme_fence"], "n": meta["n_extreme"]},
        },
        "generated_distance_distribution": {k: v for k, v in meta.items()
                                             if k.startswith("distance") or k in {"q1", "q3", "iqr"}},
        "taxonomy": {name: desc for name, desc, _p in KIND_RULES}
        | {"other_off_manifold": "Off-manifold but none of the structural rules fired"},
        "outputs": [
            "anomaly_molecules.csv.gz", "anomaly_summary_by_model.csv",
            "anomaly_property_profile.csv", "anomaly_kind_primary.csv",
            "anomaly_kind_by_model.csv", "anomaly_examples_by_kind.csv",
            "anomaly_umap_overview.png", "anomaly_spatial_summary.png",
            "anomaly_kind_breakdown.png", "anomaly_property_comparison.png",
        ],
        "interpretation_boundary": (
            "UMAP is a descriptive neighborhood projection. Anomaly = off-manifold in this "
            "specific 2D map; it is not a statement about target activity, docking, or absolute "
            "molecular quality."
        ),
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    LOGGER.info("by-model anomaly rates:\n%s", by_model.to_string(index=False))
    LOGGER.info("primary kinds:\n%s", primary_df.to_string(index=False))
    LOGGER.info("completed successfully")


if __name__ == "__main__":
    main()
