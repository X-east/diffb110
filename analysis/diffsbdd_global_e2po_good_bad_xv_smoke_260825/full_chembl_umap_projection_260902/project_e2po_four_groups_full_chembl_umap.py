#!/usr/bin/env python3
"""Project the four DiffSBDD Global-E2PO groups into the frozen full-ChEMBL UMAP.

The five-model benchmark (260819) uses a UMAP reducer fitted on 2,734,034 standardized
ChEMBL36 molecules, while the E2PO four-group report (260825) uses a different reducer
fitted on only the first 100k ChEMBL rows. Their coordinates therefore have different
scales and cannot be compared point-to-point.

This script re-projects the four E2PO groups (Original / x_only / v_only / native_joint)
through the five-model reducer, and recomputes the exact same descriptive metrics
(ChEMBL HDR membership and nearest-ChEMBL 2D distance) so that all nine groups live in
one coordinate system.

No UMAP fit is performed; the reducer is only used for ``transform``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import multiprocessing as mp
import platform
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import AllChem
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

RDLogger.DisableLog("rdApp.error")
LOGGER = logging.getLogger("project_e2po_four_groups_full_chembl_umap")

FIVE_MODELS: tuple[str, ...] = ("DiffGui", "DiffSBDD", "PMDM", "GCDM-SBDD", "TargetDiff")
E2PO_GROUPS: tuple[str, ...] = ("original", "x_only", "v_only", "native_joint")
GROUP_LABELS: dict[str, str] = {
    "original": "Original",
    "x_only": "x_only",
    "v_only": "v_only",
    "native_joint": "native_joint",
}
COLORS: dict[str, str] = {
    "DiffGui": "#3B82C4",
    "DiffSBDD": "#D95F59",
    "PMDM": "#D9A441",
    "GCDM-SBDD": "#8E78B7",
    "TargetDiff": "#3D9B72",
    "Original": "#555555",
    "x_only": "#8C564B",
    "v_only": "#E377C2",
    "native_joint": "#17BECF",
}
DISCLAIMER = "UMAP is descriptive chemical-space context, not an activity, docking, or IQL-label criterion."


def fingerprint_task(item: tuple[int, str]) -> tuple[int, np.ndarray | None]:
    """Return the ECFP4 on-bit indices for one SMILES, or None when unparsable."""
    index, smiles = item
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return index, None
    return index, np.fromiter(AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=2048).GetOnBits(), dtype=np.int32)


def build_csr(rows: Sequence[np.ndarray | None]) -> sparse.csr_matrix:
    """Stack per-molecule on-bit index arrays into a dense-width CSR matrix."""
    filled: list[np.ndarray] = [row if row is not None else np.empty(0, dtype=np.int32) for row in rows]
    indptr = np.zeros(len(filled) + 1, dtype=np.int64)
    for i, row in enumerate(filled):
        indptr[i + 1] = indptr[i] + len(row)
    indices = np.concatenate(filled) if indptr[-1] else np.empty(0, dtype=np.int32)
    return sparse.csr_matrix((np.ones(len(indices), dtype=np.float32), indices, indptr), shape=(len(filled), 2048))


def density_grid(points: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts, x_edges, y_edges = np.histogram2d(points[:, 0], points[:, 1], bins=bins)
    return counts.T, x_edges, y_edges


def hdr_membership(
    points: np.ndarray, density: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray, mass: float
) -> np.ndarray:
    """Flag points inside the smallest ChEMBL density region holding ``mass`` of the total."""
    ordered = np.sort(density.ravel())[::-1]
    threshold = ordered[min(np.searchsorted(np.cumsum(ordered) / ordered.sum(), mass), len(ordered) - 1)]
    x_bin = np.searchsorted(x_edges, points[:, 0], side="right") - 1
    y_bin = np.searchsorted(y_edges, points[:, 1], side="right") - 1
    valid = (x_bin >= 0) & (x_bin < density.shape[1]) & (y_bin >= 0) & (y_bin < density.shape[0])
    result = np.zeros(len(points), dtype=bool)
    result[valid] = density[y_bin[valid], x_bin[valid]] >= threshold
    return result


def draw_density(ax: plt.Axes, density: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray) -> None:
    nonzero = density[density > 0]
    ax.imshow(
        density,
        origin="lower",
        extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]),
        cmap="Greys",
        norm=mcolors.LogNorm(vmin=max(1, np.quantile(nonzero, 0.05)), vmax=density.max()),
        alpha=0.78,
        aspect="auto",
        rasterized=True,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonicalize_sdf(sdf_path: Path) -> list[str | None]:
    """Return canonical SMILES for every record in an SDF, None where RDKit cannot sanitize."""
    smiles: list[str | None] = []
    supplier = Chem.SDMolSupplier(str(sdf_path), sanitize=True, removeHs=True)
    for molecule in supplier:
        if molecule is None:
            smiles.append(None)
            continue
        try:
            smiles.append(Chem.MolToSmiles(molecule))
        except Exception:  # pragma: no cover - defensive: RDKit can raise on exotic valence
            smiles.append(None)
    return smiles


def load_e2po_molecules(generation_root: Path, expected_valid: dict[str, int]) -> pd.DataFrame:
    """Read the four E2PO SDFs and keep only RDKit-sanitizable molecules."""
    records: list[dict[str, Any]] = []
    for group in E2PO_GROUPS:
        sdf_path = generation_root / group / "molecules.sdf"
        if not sdf_path.is_file():
            raise FileNotFoundError(f"missing generation SDF: {sdf_path}")
        smiles_list = canonicalize_sdf(sdf_path)
        kept = 0
        for index, smiles in enumerate(smiles_list):
            if smiles is None:
                continue
            records.append(
                {
                    "row_id": len(records),
                    "group": GROUP_LABELS[group],
                    "repeat": "gen1",
                    "source_sdf": str(sdf_path),
                    "source_index": index,
                    "canonical_smiles": smiles,
                }
            )
            kept += 1
        expected = expected_valid.get(group)
        if expected is not None and kept != expected:
            raise ValueError(f"{group}: projected {kept} molecules but generation_summary expects {expected}")
        LOGGER.info("%s: sdf_records=%d sanitize_valid=%d", group, len(smiles_list), kept)
    return pd.DataFrame.from_records(records)


def expected_valid_counts(summary_csv: Path) -> dict[str, int]:
    """Read the four-group sanitize_valid counts used as a cross-check."""
    frame = pd.read_csv(summary_csv)
    return {str(row.variant): int(row.sanitize_valid) for row in frame.itertuples()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--generation-summary", type=Path, required=True)
    parser.add_argument("--five-model-metrics", type=Path, required=True)
    parser.add_argument("--five-model-by-repeat", type=Path, required=True)
    parser.add_argument("--reducer", type=Path, required=True)
    parser.add_argument("--chembl-xy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=28)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.output_dir / "projection.log", encoding="utf-8"),
        ],
        force=True,
    )

    for path in (args.generation_root, args.generation_summary, args.five_model_metrics, args.five_model_by_repeat, args.reducer, args.chembl_xy):
        if not path.exists():
            raise FileNotFoundError(f"missing input: {path}")

    expected = expected_valid_counts(args.generation_summary)
    e2po = load_e2po_molecules(args.generation_root, expected)
    LOGGER.info("e2po valid molecules=%d", len(e2po))

    tasks = list(enumerate(e2po["canonical_smiles"].astype(str)))
    with mp.Pool(args.workers) as pool:
        results = pool.map(fingerprint_task, tasks, chunksize=64)
    rows: list[np.ndarray | None] = [None] * len(results)
    for index, bits in results:
        rows[index] = bits
    matrix = build_csr(rows)
    LOGGER.info("query CSR shape=%s nnz=%d", matrix.shape, matrix.nnz)

    LOGGER.info("loading reducer %s", args.reducer)
    reducer = joblib.load(args.reducer)
    projected = reducer.transform(matrix.toarray().astype(np.float32, copy=False)).astype(np.float32)
    if not np.isfinite(projected).all():
        raise ValueError("non-finite projected coordinates")
    e2po["umap1"], e2po["umap2"] = projected[:, 0], projected[:, 1]
    LOGGER.info("projection complete: %d molecules", len(e2po))

    chembl_xy = np.load(args.chembl_xy, mmap_mode="r")
    chembl = np.asarray(chembl_xy, dtype=np.float32)
    LOGGER.info("chembl coordinates shape=%s", chembl.shape)
    density, x_edges, y_edges = density_grid(chembl, 500)
    generated_xy = e2po[["umap1", "umap2"]].to_numpy(np.float32)
    for mass in (0.5, 0.9, 0.95, 0.99):
        e2po[f"inside_chembl_hdr_{int(mass * 100)}"] = hdr_membership(generated_xy, density, x_edges, y_edges, mass)
    neighbors = NearestNeighbors(n_neighbors=1, algorithm="kd_tree", n_jobs=args.workers).fit(chembl)
    e2po["nearest_chembl_umap_distance"] = neighbors.kneighbors(generated_xy, return_distance=True)[0][:, 0]

    e2po.to_csv(args.output_dir / "e2po_four_groups_projected_metrics.csv.gz", index=False, compression="gzip")

    five = pd.read_csv(args.five_model_metrics)
    five = five[five["model"].isin(FIVE_MODELS)].copy()
    combined = pd.concat(
        [
            five[["model", "repeat", "canonical_smiles", "umap1", "umap2", "inside_chembl_hdr_95", "nearest_chembl_umap_distance"]].rename(columns={"model": "group"}),
            e2po[["group", "repeat", "canonical_smiles", "umap1", "umap2", "inside_chembl_hdr_95", "nearest_chembl_umap_distance"]],
        ],
        ignore_index=True,
    )
    combined.to_csv(args.output_dir / "combined_nine_groups_projected_metrics.csv.gz", index=False, compression="gzip")

    per_repeat = combined.groupby(["group", "repeat"]).agg(
        n=("umap1", "size"),
        inside_hdr95=("inside_chembl_hdr_95", "mean"),
        median_umap_distance=("nearest_chembl_umap_distance", "median"),
        centroid_umap1=("umap1", "mean"),
        centroid_umap2=("umap2", "mean"),
    ).reset_index()
    per_repeat.to_csv(args.output_dir / "combined_by_repeat.csv", index=False)

    order = [*FIVE_MODELS, *[GROUP_LABELS[g] for g in E2PO_GROUPS]]
    summary = (
        per_repeat.groupby("group")
        .agg(
            n_mean=("n", "mean"),
            n_sd=("n", "std"),
            hdr95_mean=("inside_hdr95", "mean"),
            hdr95_sd=("inside_hdr95", "std"),
            median_distance_mean=("median_umap_distance", "mean"),
            median_distance_sd=("median_umap_distance", "std"),
            centroid_umap1=("centroid_umap1", "mean"),
            centroid_umap2=("centroid_umap2", "mean"),
            n_repeats=("repeat", "nunique"),
        )
        .reindex(order)
        .reset_index()
    )
    chembl_center = chembl.mean(axis=0)
    summary["distance_to_chembl_center"] = np.hypot(
        summary["centroid_umap1"] - chembl_center[0], summary["centroid_umap2"] - chembl_center[1]
    )
    original_centroid = summary.loc[summary.group.eq("Original"), ["centroid_umap1", "centroid_umap2"]].to_numpy()
    summary["centroid_shift_vs_Original"] = np.hypot(
        summary["centroid_umap1"] - original_centroid[0, 0], summary["centroid_umap2"] - original_centroid[0, 1]
    )
    diffsbdd_centroid = summary.loc[summary.group.eq("DiffSBDD"), ["centroid_umap1", "centroid_umap2"]].to_numpy()
    summary["centroid_shift_vs_DiffSBDD"] = np.hypot(
        summary["centroid_umap1"] - diffsbdd_centroid[0, 0], summary["centroid_umap2"] - diffsbdd_centroid[0, 1]
    )
    summary.to_csv(args.output_dir / "combined_umap_summary.csv", index=False)
    LOGGER.info("summary written for %d groups", len(summary))

    _draw_overview(args.output_dir, combined, chembl, density, x_edges, y_edges, order)
    _draw_facets(args.output_dir, combined, chembl, density, x_edges, y_edges, order)
    _draw_bars(args.output_dir, summary, order)
    _draw_centroids(args.output_dir, summary, chembl_center, order)

    manifest = {
        "status": "complete",
        "purpose": "Put the four DiffSBDD Global-E2PO groups into the same frozen full-ChEMBL UMAP as the five-model benchmark.",
        "reducer": str(args.reducer.resolve()),
        "reducer_sha256": sha256(args.reducer),
        "chembl_coordinates": str(args.chembl_xy.resolve()),
        "five_model_metrics": str(args.five_model_metrics.resolve()),
        "fit_performed": False,
        "fingerprint": {"type": "ECFP4", "radius": 2, "n_bits": 2048},
        "counts": {
            "e2po_projected": int(len(e2po)),
            "by_group": {label: int((e2po.group == label).sum()) for label in GROUP_LABELS.values()},
            "five_model_rows": int(len(five)),
            "chembl_rows": int(chembl.shape[0]),
        },
        "workers": args.workers,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "rdkit": rdBase.rdkitVersion,
            "joblib": joblib.__version__,
        },
        "interpretation_boundary": DISCLAIMER,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (args.output_dir / "completion.json").write_text(
        json.dumps({"status": "complete", "groups": len(summary), "e2po_molecules": int(len(e2po))}, indent=2),
        encoding="utf-8",
    )
    LOGGER.info("all outputs written to %s", args.output_dir)


def _draw_overview(
    output_dir: Path,
    combined: pd.DataFrame,
    chembl: np.ndarray,
    density: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    order: Iterable[str],
) -> None:
    fig, ax = plt.subplots(figsize=(12, 8.6))
    draw_density(ax, density, x_edges, y_edges)
    for group in order:
        subset = combined[combined.group.eq(group)]
        ax.scatter(
            subset.umap1,
            subset.umap2,
            s=9,
            alpha=0.55,
            linewidths=0,
            color=COLORS[group],
            label=f"{group} (n={len(subset):,})",
            rasterized=True,
        )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("Five benchmark models and four E2PO groups in one frozen full-ChEMBL36 UMAP")
    ax.legend(markerscale=2, frameon=True, fontsize=9, loc="upper right")
    fig.text(0.5, 0.01, DISCLAIMER, ha="center")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(output_dir / "combined_umap_overview.png", dpi=350)
    fig.savefig(output_dir / "combined_umap_overview.svg")
    plt.close(fig)


def _draw_facets(
    output_dir: Path,
    combined: pd.DataFrame,
    chembl: np.ndarray,
    density: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    order: Iterable[str],
) -> None:
    groups = list(order)
    fig, axes = plt.subplots(3, 4, figsize=(20, 13), sharex=True, sharey=True)
    for ax, group in zip(axes.flat, groups):
        draw_density(ax, density, x_edges, y_edges)
        subset = combined[combined.group.eq(group)]
        ax.scatter(subset.umap1, subset.umap2, s=8, color=COLORS[group], alpha=0.62, linewidths=0, rasterized=True)
        ax.set_title(f"{group} | n={len(subset):,}")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
    for ax in axes.flat[len(groups):]:
        draw_density(ax, density, x_edges, y_edges)
        ax.set_title(f"Frozen ChEMBL36 density | n={len(chembl):,}")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
    fig.suptitle("Nine groups in one frozen full-ChEMBL36 UMAP coordinate system", fontsize=16)
    fig.text(0.5, 0.005, DISCLAIMER, ha="center")
    fig.tight_layout(rect=(0, 0.025, 1, 0.96))
    fig.savefig(output_dir / "combined_umap_facets.png", dpi=300)
    fig.savefig(output_dir / "combined_umap_facets.svg")
    plt.close(fig)


def _draw_bars(output_dir: Path, summary: pd.DataFrame, order: Iterable[str]) -> None:
    groups = list(order)
    subset = summary.set_index("group").reindex(groups)
    x = np.arange(len(groups))
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 6.4))
    axes[0].bar(x, subset.n_mean, color=[COLORS[g] for g in groups])
    axes[0].set_ylabel("Molecules entering the map")
    axes[0].set_title("Embedding yield (E2PO: 1 generation each)")
    axes[1].bar(x, subset.hdr95_mean * 100, yerr=subset.hdr95_sd.fillna(0) * 100, color=[COLORS[g] for g in groups], capsize=4)
    axes[1].set_ylabel("Inside ChEMBL 95% HDR (%)")
    axes[1].set_title("High-density ChEMBL overlap")
    axes[2].bar(
        x,
        subset.median_distance_mean,
        yerr=subset.median_distance_sd.fillna(0),
        color=[COLORS[g] for g in groups],
        capsize=4,
    )
    axes[2].set_ylabel("Nearest ChEMBL 2D distance (median)")
    axes[2].set_title("Local UMAP neighbourhood")
    for ax in axes:
        ax.set_xticks(x, groups, rotation=30, ha="right")
    fig.text(
        0.5,
        0.01,
        "Error bars are SD across three repeats; the four E2PO groups have a single generation, so no error bar.",
        ha="center",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(output_dir / "combined_umap_bars.png", dpi=300)
    fig.savefig(output_dir / "combined_umap_bars.svg")
    plt.close(fig)


def _draw_centroids(output_dir: Path, summary: pd.DataFrame, chembl_center: np.ndarray, order: Iterable[str]) -> None:
    groups = list(order)
    subset = summary.set_index("group").reindex(groups)
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.4))
    axes[0].scatter(chembl_center[0], chembl_center[1], s=180, c="k", marker="*", label="ChEMBL36 centroid")
    for group in groups:
        axes[0].scatter(subset.loc[group, "centroid_umap1"], subset.loc[group, "centroid_umap2"], s=110, color=COLORS[group], label=group)
    axes[0].set_xlabel("UMAP 1")
    axes[0].set_ylabel("UMAP 2")
    axes[0].set_title("Group centroids in the shared coordinate system")
    axes[0].legend(fontsize=8)

    x = np.arange(len(groups))
    axes[1].bar(x, subset["distance_to_chembl_center"], color=[COLORS[g] for g in groups])
    axes[1].set_xticks(x, groups, rotation=30, ha="right")
    axes[1].set_ylabel("Centroid distance to ChEMBL36 centroid")
    axes[1].set_title("Chemical-space distance to the ChEMBL centre")
    fig.text(0.5, 0.01, DISCLAIMER, ha="center")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(output_dir / "combined_umap_centroids.png", dpi=300)
    fig.savefig(output_dir / "combined_umap_centroids.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
