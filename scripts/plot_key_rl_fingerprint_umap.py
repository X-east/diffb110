#!/usr/bin/env python3
"""Joint ECFP4-UMAP analysis of key RL runs against one frozen ChEMBL sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from multiprocessing import Pool
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/diffb110_matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/diffb110_numba_cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem


RDLogger.DisableLog("rdApp.error")
LOGGER = logging.getLogger("key_rl_fingerprint_umap")
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RL = ROOT / "analysis/training_record_review_20260727/carbon_drift_molecule_level.csv.gz"
DEFAULT_EXISTING_EMBEDDINGS = (
    ROOT
    / "analysis/benchmark/benchmark5"
    / "8gjl_c1101_benchmark_20260706_1000steps_3rep_formal"
    / "raw_quality_vs_chembl_20260728/umap_embeddings.csv"
)
DEFAULT_OUTPUT = ROOT / "analysis/key_rl_fingerprint_umap_20260729"

METHOD_ORDER = [
    "DiffSBDD dock+classifier",
    "DiffGui Next6-04",
    "DiffGui DePPA t200",
    "DiffGui PB-fast",
    "DePPA original",
]
METHOD_SHORT = {
    "DiffSBDD dock+classifier": "DiffSBDD dock+classifier",
    "DiffGui Next6-04": "DiffGui Next6-04",
    "DiffGui DePPA t200": "DiffGui DePPA t200",
    "DiffGui PB-fast": "DiffGui PB-fast",
    "DePPA original": "DePPA original",
}
COHORT_COLORS = {"early": "#3B82C4", "middle": "#9B8AC4", "late": "#D95F59"}
CHEMBL_COLOR = "#B8B8B8"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rl-csv", type=Path, default=DEFAULT_RL)
    parser.add_argument("--existing-embeddings", type=Path, default=DEFAULT_EXISTING_EMBEDDINGS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chembl-label", default="ChEMBL")
    parser.add_argument("--chembl-sample-size", type=int, default=30000)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--nbits", type=int, default=2048)
    parser.add_argument("--n-neighbors", type=int, default=20)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--metric", default="cosine")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--workers", type=int, default=18)
    return parser.parse_args()


def _configure_plotting() -> None:
    """Apply the project publication plotting defaults."""
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _sample_hash(smiles: list[str]) -> str:
    """Hash the ordered ChEMBL sample so every downstream analysis can verify identity."""
    payload = "\n".join(smiles).encode("utf-8") + b"\n"
    return hashlib.sha256(payload).hexdigest()


def _load_frozen_chembl(args: argparse.Namespace) -> pd.DataFrame:
    """Reuse the exact ordered ChEMBL sample from the existing five-model UMAP."""
    source = pd.read_csv(args.existing_embeddings, usecols=["label", "smiles"])
    sample = source[source["label"] == args.chembl_label][["smiles"]].copy()
    sample = sample.rename(columns={"smiles": "canonical_smiles"}).reset_index(drop=True)
    if len(sample) < args.chembl_sample_size:
        raise ValueError(
            f"expected at least {args.chembl_sample_size} frozen ChEMBL rows, found {len(sample)}"
        )
    sample = sample.head(args.chembl_sample_size).copy()
    if sample["canonical_smiles"].isna().any():
        raise ValueError("frozen ChEMBL sample contains missing SMILES")
    sample.insert(0, "sample_order", np.arange(len(sample), dtype=int))
    sample["source_embeddings"] = str(args.existing_embeddings)
    return sample


def _load_rl_molecules(path: Path) -> pd.DataFrame:
    """Load traced sanitize-valid RL molecules without deduplicating their distribution."""
    columns = ["method", "epoch", "cohort", "canonical_smiles", "source_file", "source_row"]
    frame = pd.read_csv(path, usecols=columns)
    frame = frame[
        frame["method"].isin(METHOD_ORDER)
        & frame["canonical_smiles"].notna()
        & frame["cohort"].isin(["early", "middle", "late"])
    ].copy()
    frame["epoch"] = pd.to_numeric(frame["epoch"], errors="raise").astype(int)
    frame["input_order"] = np.arange(len(frame), dtype=int)
    return frame


def _fingerprint_task(task: tuple[int, str, int, int]) -> tuple[int, np.ndarray | None]:
    """Calculate one binary Morgan/ECFP4 fingerprint."""
    index, smiles, radius, nbits = task
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return index, None
        bit_vector = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
        array = np.zeros((nbits,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(bit_vector, array)
        return index, array
    except Exception:
        return index, None


def _fingerprints(smiles: list[str], radius: int, nbits: int, workers: int) -> tuple[np.ndarray, np.ndarray]:
    """Calculate fingerprints in input order and return valid indexes plus matrix."""
    tasks = [(index, value, radius, nbits) for index, value in enumerate(smiles)]
    with Pool(processes=workers) as pool:
        results = pool.map(_fingerprint_task, tasks, chunksize=128)
    valid = [(index, fp) for index, fp in results if fp is not None]
    indexes = np.asarray([item[0] for item in valid], dtype=int)
    matrix = np.vstack([item[1] for item in valid]).astype(np.uint8, copy=False)
    return indexes, matrix


def _group_summary(embeddings: pd.DataFrame) -> pd.DataFrame:
    """Summarize coordinate centroids and within-group spread for each training cohort."""
    chembl = embeddings[embeddings["dataset"] == "ChEMBL"][["umap1", "umap2"]]
    chembl_centroid = chembl.mean().to_numpy()
    rows: list[dict[str, Any]] = []
    for (method, cohort), group in embeddings[embeddings["dataset"] == "RL"].groupby(
        ["method", "cohort"], sort=False
    ):
        coords = group[["umap1", "umap2"]].to_numpy()
        centroid = coords.mean(axis=0)
        rows.append(
            {
                "method": method,
                "cohort": cohort,
                "n": int(len(group)),
                "epoch_min": int(group["epoch"].min()),
                "epoch_max": int(group["epoch"].max()),
                "centroid_umap1": float(centroid[0]),
                "centroid_umap2": float(centroid[1]),
                "centroid_distance_to_chembl": float(np.linalg.norm(centroid - chembl_centroid)),
                "within_group_rms_radius": float(np.sqrt(np.mean(np.sum((coords - centroid) ** 2, axis=1)))),
            }
        )
    return pd.DataFrame(rows)


def _epoch_centroids(embeddings: pd.DataFrame) -> pd.DataFrame:
    """Calculate one joint-UMAP centroid per training epoch."""
    rl = embeddings[embeddings["dataset"] == "RL"]
    summary = (
        rl.groupby(["method", "epoch", "cohort"], as_index=False)
        .agg(n=("canonical_smiles", "size"), umap1=("umap1", "mean"), umap2=("umap2", "mean"))
    )
    return summary


def _save_figure(fig: plt.Figure, stem: Path) -> None:
    """Save raster and editable vector exports from Python only."""
    fig.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_facets(embeddings: pd.DataFrame, epochs: pd.DataFrame, output_dir: Path) -> None:
    """Plot one comparable ChEMBL/early/late panel for each key RL training."""
    chembl = embeddings[embeddings["dataset"] == "ChEMBL"]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.6), sharex=True, sharey=True)
    axes_flat = axes.ravel()
    for axis, method in zip(axes_flat, METHOD_ORDER):
        group = embeddings[(embeddings["dataset"] == "RL") & (embeddings["method"] == method)]
        axis.scatter(
            chembl["umap1"], chembl["umap2"], s=1.0, color=CHEMBL_COLOR,
            alpha=0.10, linewidths=0, rasterized=True,
        )
        for cohort in ["early", "late"]:
            subset = group[group["cohort"] == cohort]
            axis.scatter(
                subset["umap1"], subset["umap2"], s=7.0,
                color=COHORT_COLORS[cohort], alpha=0.55, linewidths=0,
                label=f"{cohort.capitalize()} (n={len(subset):,})", rasterized=True,
            )
        trajectory = epochs[epochs["method"] == method].sort_values("epoch")
        if len(trajectory) > 1:
            axis.plot(trajectory["umap1"], trajectory["umap2"], color="#333333", linewidth=0.8, alpha=0.7)
            axis.scatter(
                trajectory["umap1"], trajectory["umap2"], c=trajectory["epoch"],
                cmap="Greys", s=8, linewidths=0, zorder=4,
            )
        axis.set_title(METHOD_SHORT[method], fontsize=14)
        axis.legend(loc="best", fontsize=9.5, markerscale=1.4)
        axis.set_xlabel("UMAP1")
        axis.set_ylabel("UMAP2")
    axes_flat[-1].axis("off")
    fig.suptitle("RL chemical-space drift against one frozen ChEMBL36 sample", fontsize=17, y=0.995)
    fig.text(
        0.5, 0.005,
        "ECFP4 (radius 2, 2048 bits); joint UMAP; grey = identical 30,000-molecule ChEMBL reference",
        ha="center", fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.98))
    _save_figure(fig, output_dir / "key_rl_early_late_vs_frozen_chembl_umap")


def _plot_trajectory(epochs: pd.DataFrame, output_dir: Path) -> None:
    """Plot epoch-centroid trajectories alone for easier drift inspection."""
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 8.4), sharex=True, sharey=True)
    axes_flat = axes.ravel()
    for axis, method in zip(axes_flat, METHOD_ORDER):
        trajectory = epochs[epochs["method"] == method].sort_values("epoch")
        scatter = axis.scatter(
            trajectory["umap1"], trajectory["umap2"], c=trajectory["epoch"],
            cmap="viridis", s=18, linewidths=0,
        )
        if len(trajectory) > 1:
            axis.plot(trajectory["umap1"], trajectory["umap2"], color="#555555", linewidth=0.8, alpha=0.65)
        axis.set_title(METHOD_SHORT[method], fontsize=14)
        axis.set_xlabel("UMAP1")
        axis.set_ylabel("UMAP2")
        colorbar = fig.colorbar(scatter, ax=axis, fraction=0.045, pad=0.03)
        colorbar.set_label("Epoch", fontsize=10)
        colorbar.ax.tick_params(labelsize=9)
    axes_flat[-1].axis("off")
    fig.suptitle("Epoch-centroid trajectories in the shared ECFP4-UMAP space", fontsize=17, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    _save_figure(fig, output_dir / "key_rl_epoch_centroid_trajectories")


def _write_report(
    output_dir: Path,
    sample_hash: str,
    chembl_count: int,
    rl_input: pd.DataFrame,
    summary: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    """Write a durable report with lineage, counts, and embedded figures."""
    count_table = (
        rl_input.groupby(["method", "cohort"], as_index=False)
        .size()
        .pivot(index="method", columns="cohort", values="size")
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    count_table = count_table.set_index("method").reindex(METHOD_ORDER).fillna(0).astype(int)
    for cohort in ["early", "middle", "late"]:
        if cohort not in count_table.columns:
            count_table[cohort] = 0
    change_rows: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        method_summary = summary[summary["method"] == method].set_index("cohort")
        if not {"early", "late"}.issubset(method_summary.index):
            continue
        early = method_summary.loc["early"]
        late = method_summary.loc["late"]
        displacement = np.hypot(
            late["centroid_umap1"] - early["centroid_umap1"],
            late["centroid_umap2"] - early["centroid_umap2"],
        )
        change_rows.append(
            {
                "method": method,
                "displacement": float(displacement),
                "distance_early": float(early["centroid_distance_to_chembl"]),
                "distance_late": float(late["centroid_distance_to_chembl"]),
                "distance_change": float(
                    late["centroid_distance_to_chembl"] - early["centroid_distance_to_chembl"]
                ),
                "radius_early": float(early["within_group_rms_radius"]),
                "radius_late": float(late["within_group_rms_radius"]),
                "radius_change": float(
                    late["within_group_rms_radius"] - early["within_group_rms_radius"]
                ),
            }
        )
    count_lines = [
        "| Training | Early | Middle | Late |",
        "|---|---:|---:|---:|",
    ]
    for method, row in count_table.iterrows():
        count_lines.append(
            f"| {method} | {row['early']:,} | {row['middle']:,} | {row['late']:,} |"
        )
    change_lines = [
        "| Training | Centroid displacement | Distance to ChEMBL, early -> late | Change | RMS radius, early -> late | Change |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in change_rows:
        change_lines.append(
            f"| {row['method']} | {row['displacement']:.3f} | "
            f"{row['distance_early']:.3f} -> {row['distance_late']:.3f} | "
            f"{row['distance_change']:+.3f} | "
            f"{row['radius_early']:.3f} -> {row['radius_late']:.3f} | "
            f"{row['radius_change']:+.3f} |"
        )
    lines = [
        "# Key RL Fingerprint UMAP Versus Frozen ChEMBL36",
        "",
        "## Scope and identity controls",
        "",
        f"- ChEMBL sample count: `{chembl_count}`",
        f"- Ordered ChEMBL SMILES SHA-256: `{sample_hash}`",
        "- The ChEMBL list is extracted from the existing five-model UMAP point table and reused without resampling.",
        "- RL duplicates are retained because point density and mode collapse are part of the learned distribution.",
        f"- Fingerprint: ECFP4, radius `{args.radius}`, `{args.nbits}` bits.",
        f"- Joint UMAP: metric `{args.metric}`, n_neighbors `{args.n_neighbors}`, min_dist `{args.min_dist}`, random_state `{args.random_state}`.",
        "",
        "## Input counts",
        "",
        *count_lines,
        "",
        f"All `{len(rl_input):,}` RL rows are sanitize-valid molecules with traceable canonical SMILES. Exact duplicates are retained.",
        "",
        "## Shared-space views",
        "",
        "![Key RL early and late cohorts](key_rl_early_late_vs_frozen_chembl_umap.png)",
        "",
        "![Epoch centroid trajectories](key_rl_epoch_centroid_trajectories.png)",
        "",
        "## Early-to-late changes",
        "",
        *change_lines,
        "",
        "- `DePPA original` shows the clearest directed drift away from the ChEMBL centroid while its distribution contracts.",
        "- `DiffGui PB-fast` moves closer by centroid but expands strongly; this is not evidence of improved chemical-space coverage or molecular quality.",
        "- `DiffGui Next6-04` and `DiffGui DePPA t200` show smaller, non-catastrophic shifts in this projection. DiffSBDD changes little but ends slightly farther away.",
        "",
        "## Interpretation boundary",
        "",
        "UMAP is a neighborhood visualization, not a calibrated molecular-distance scale. Centroid distances are descriptive within this one fitted coordinate system only. Chemical validity and applicability require the separate descriptor, alert, geometry, and docking analyses.",
        "",
        "## Evidence files",
        "",
        "- [Frozen ChEMBL sample](chembl_sample_30000_frozen.csv.gz)",
        "- [All point coordinates](joint_umap_embeddings.csv.gz)",
        "- [Cohort summary](cohort_umap_summary.csv)",
        "- [Epoch centroid coordinates](epoch_centroids.csv)",
        "- [Run manifest](manifest.json)",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Build the frozen-reference joint fingerprint embedding and figures."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.output_dir / "run.log", mode="w", encoding="utf-8"),
        ],
    )
    _configure_plotting()

    chembl = _load_frozen_chembl(args)
    chembl_smiles = chembl["canonical_smiles"].astype(str).tolist()
    sample_hash = _sample_hash(chembl_smiles)
    chembl.to_csv(args.output_dir / "chembl_sample_30000_frozen.csv.gz", index=False, compression="gzip")
    LOGGER.info("frozen ChEMBL n=%s ordered_sha256=%s", len(chembl), sample_hash)

    rl = _load_rl_molecules(args.rl_csv)
    LOGGER.info("RL input rows=%s methods=%s", len(rl), rl["method"].nunique())
    all_smiles = chembl_smiles + rl["canonical_smiles"].astype(str).tolist()
    valid_indexes, fingerprint_matrix = _fingerprints(
        all_smiles, args.radius, args.nbits, args.workers
    )
    if len(valid_indexes) != len(all_smiles):
        LOGGER.warning("fingerprint failures=%s", len(all_smiles) - len(valid_indexes))
    LOGGER.info("fitting joint UMAP rows=%s bits=%s", *fingerprint_matrix.shape)
    reducer = umap.UMAP(
        n_components=2,
        metric=args.metric,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        random_state=args.random_state,
        low_memory=True,
        n_jobs=1,
    )
    coordinates = reducer.fit_transform(fingerprint_matrix)

    metadata_rows: list[dict[str, Any]] = []
    for order, smiles in enumerate(chembl_smiles):
        metadata_rows.append(
            {
                "dataset": "ChEMBL",
                "method": "ChEMBL36",
                "epoch": np.nan,
                "cohort": "reference",
                "canonical_smiles": smiles,
                "source_file": str(args.existing_embeddings),
                "source_row": order,
            }
        )
    for row in rl.to_dict("records"):
        metadata_rows.append(
            {
                "dataset": "RL",
                "method": row["method"],
                "epoch": row["epoch"],
                "cohort": row["cohort"],
                "canonical_smiles": row["canonical_smiles"],
                "source_file": row["source_file"],
                "source_row": row["source_row"],
            }
        )
    metadata = pd.DataFrame(metadata_rows).iloc[valid_indexes].reset_index(drop=True)
    metadata["umap1"] = coordinates[:, 0]
    metadata["umap2"] = coordinates[:, 1]
    metadata.to_csv(args.output_dir / "joint_umap_embeddings.csv.gz", index=False, compression="gzip")

    summary = _group_summary(metadata)
    summary.to_csv(args.output_dir / "cohort_umap_summary.csv", index=False)
    epoch_summary = _epoch_centroids(metadata)
    epoch_summary.to_csv(args.output_dir / "epoch_centroids.csv", index=False)
    _plot_facets(metadata, epoch_summary, args.output_dir)
    _plot_trajectory(epoch_summary, args.output_dir)

    manifest = {
        "chembl_source_embeddings": str(args.existing_embeddings),
        "chembl_sample_rows": int(len(chembl)),
        "chembl_ordered_smiles_sha256": sample_hash,
        "rl_source": str(args.rl_csv),
        "rl_rows_input": int(len(rl)),
        "fingerprint_valid_rows": int(len(metadata)),
        "methods": METHOD_ORDER,
        "fingerprint": {"type": "Morgan/ECFP4", "radius": args.radius, "nbits": args.nbits},
        "umap": {
            "joint_fit": True,
            "metric": args.metric,
            "n_neighbors": args.n_neighbors,
            "min_dist": args.min_dist,
            "random_state": args.random_state,
        },
        "duplicates_retained": True,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_report(args.output_dir, sample_hash, len(chembl), rl, summary, args)
    LOGGER.info("completed output=%s", args.output_dir)


if __name__ == "__main__":
    main()
