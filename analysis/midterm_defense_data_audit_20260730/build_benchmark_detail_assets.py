#!/usr/bin/env python3
"""Build detailed benchmark assets for the 2026-07-30 midterm record."""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
from decimal import Decimal, ROUND_HALF_UP
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
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, Draw, Lipinski, QED, rdMolDescriptors
from scipy import sparse
import umap


LOGGER = logging.getLogger("midterm_benchmark_assets")
ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent
FORMAL_DIR = (
    ROOT
    / "analysis/benchmark/benchmark5"
    / "8gjl_c1101_benchmark_20260706_1000steps_3rep_formal"
)
GENERATED_CSV = FORMAL_DIR / "raw_quality_vs_chembl_20260728/per_molecule_raw_quality.csv.gz"
CHEMBL_30K = ROOT / "analysis/key_rl_fingerprint_umap_20260729/chembl_sample_30000_frozen.csv.gz"

MODEL_ORDER = ["DiffGui", "DiffSBDD", "PMDM", "GCDM-SBDD", "TargetDiff"]
MODEL_DIRS = {
    "DiffGui": "diffgui",
    "DiffSBDD": "diffsbdd",
    "PMDM": "pmdm",
    "GCDM-SBDD": "gcdm_sbdd",
    "TargetDiff": "targetdiff",
}
MODEL_COLORS = {
    "DiffGui": "#3977A8",
    "DiffSBDD": "#C75D54",
    "PMDM": "#C39536",
    "GCDM-SBDD": "#8068A8",
    "TargetDiff": "#398568",
}
CHEMBL_COLOR = "#A7ADB2"


def _configure_logging() -> None:
    """Configure terminal and durable file logging."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    for handler in [
        logging.StreamHandler(),
        logging.FileHandler(OUTPUT_DIR / "build_benchmark_detail_assets.log", mode="w", encoding="utf-8"),
    ]:
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)


def _configure_plotting() -> None:
    """Apply report-ready plotting defaults."""
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 13,
            "axes.titlesize": 17,
            "axes.labelsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _sha256(path: Path) -> str:
    """Return the SHA-256 checksum of one source file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonicalize(smiles: str) -> tuple[str, Chem.Mol | None]:
    """Return canonical isomeric SMILES and its molecule, if readable."""
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        return "", None
    return Chem.MolToSmiles(molecule, isomericSmiles=True, canonical=True), molecule


def _descriptor_row(molecule: Chem.Mol) -> dict[str, float | int]:
    """Calculate concise descriptors for one displayed top-scoring molecule."""
    return {
        "molecular_weight": float(Descriptors.MolWt(molecule)),
        "logp": float(Crippen.MolLogP(molecule)),
        "qed": float(QED.qed(molecule)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(molecule)),
        "hba": int(Lipinski.NumHAcceptors(molecule)),
        "hbd": int(Lipinski.NumHDonors(molecule)),
        "rotatable_bonds": int(Lipinski.NumRotatableBonds(molecule)),
        "heavy_atoms": int(molecule.GetNumHeavyAtoms()),
        "rings": int(rdMolDescriptors.CalcNumRings(molecule)),
    }


def build_top10_assets() -> pd.DataFrame:
    """Select unique full-redocking Top 10 molecules and render one grid per model."""
    output_rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        frames: list[pd.DataFrame] = []
        for repeat in ["rep1", "rep2", "rep3"]:
            csv_path = FORMAL_DIR / "docking" / repeat / MODEL_DIRS[model] / "docking_results.csv"
            frame = pd.read_csv(csv_path)
            frame["repeat"] = repeat
            frame["docking_csv"] = str(csv_path)
            frames.append(frame)
        merged = pd.concat(frames, ignore_index=True)
        merged = merged[merged["success"].eq(True) & merged["score"].notna()].copy()
        canonicalized = merged["input_smiles"].map(_canonicalize)
        merged["canonical_smiles"] = canonicalized.map(lambda item: item[0])
        merged["rdmol"] = canonicalized.map(lambda item: item[1])
        merged = merged[merged["rdmol"].notna() & merged["canonical_smiles"].ne("")]
        merged = merged.sort_values(["score", "repeat", "benchmark_index"], ascending=[True, True, True])
        merged = merged.drop_duplicates("canonical_smiles", keep="first").head(10).copy()
        merged["rank"] = np.arange(1, len(merged) + 1)

        molecules = merged["rdmol"].tolist()
        legends = [
            f"#{rank}  {score:.3f} kcal/mol\n{repeat} | ID {index}"
            for rank, score, repeat, index in zip(
                merged["rank"], merged["score"], merged["repeat"], merged["benchmark_index"]
            )
        ]
        draw_options = Draw.MolDrawOptions()
        draw_options.legendFontSize = 30
        image = Draw.MolsToGridImage(
            molecules,
            molsPerRow=5,
            subImgSize=(480, 400),
            legends=legends,
            drawOptions=draw_options,
            useSVG=False,
        )
        image.save(OUTPUT_DIR / f"top10_2d_{MODEL_DIRS[model]}.png")

        for row in merged.itertuples(index=False):
            values = {
                "model": model,
                "rank": int(row.rank),
                "repeat": row.repeat,
                "benchmark_index": int(row.benchmark_index),
                "ligand_stem": row.ligand_stem,
                "full_redocking_score_kcal_mol": float(row.score),
                "canonical_smiles": row.canonical_smiles,
                "source_sdf": row.source_sdf,
                "docked_output_file": row.output_file,
                "docking_csv": row.docking_csv,
            }
            values.update(_descriptor_row(row.rdmol))
            output_rows.append(values)
        LOGGER.info("rendered %s Top 10 from %d successful rows", model, len(merged))

    output = pd.DataFrame(output_rows)
    output.to_csv(OUTPUT_DIR / "benchmark_full_redocking_top10_per_model.csv", index=False)
    summary = (
        output.groupby("model", sort=False)
        .agg(
            top10_best=("full_redocking_score_kcal_mol", "min"),
            top10_median=("full_redocking_score_kcal_mol", "median"),
            top10_worst=("full_redocking_score_kcal_mol", "max"),
            molecular_weight_median=("molecular_weight", "median"),
            logp_median=("logp", "median"),
            qed_median=("qed", "median"),
            tpsa_median=("tpsa", "median"),
        )
        .reindex(MODEL_ORDER)
        .reset_index()
    )
    summary.to_csv(OUTPUT_DIR / "benchmark_full_redocking_top10_summary.csv", index=False)
    return output


def _fingerprint_task(task: tuple[int, str]) -> tuple[int, Any | None]:
    """Calculate one ECFP4 bit vector while preserving input order."""
    index, smiles = task
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return index, None
    return index, AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=2048)


def _fingerprint_frame(frame: pd.DataFrame, workers: int = 18) -> tuple[pd.DataFrame, list[Any]]:
    """Calculate ECFP4 fingerprints for a molecule table."""
    tasks = list(enumerate(frame["canonical_smiles"].astype(str).tolist()))
    with Pool(processes=workers) as pool:
        results = pool.map(_fingerprint_task, tasks, chunksize=128)
    valid = [(index, fp) for index, fp in results if fp is not None]
    kept = frame.iloc[[index for index, _ in valid]].reset_index(drop=True)
    return kept, [fp for _, fp in valid]


def _to_csr(fingerprints: list[Any]) -> sparse.csr_matrix:
    """Convert RDKit bit vectors to a compact sparse matrix."""
    indptr = np.zeros(len(fingerprints) + 1, dtype=np.int64)
    for index, fingerprint in enumerate(fingerprints):
        indptr[index + 1] = indptr[index] + fingerprint.GetNumOnBits()
    indices = np.empty(int(indptr[-1]), dtype=np.int32)
    for index, fingerprint in enumerate(fingerprints):
        start, stop = int(indptr[index]), int(indptr[index + 1])
        indices[start:stop] = np.fromiter(fingerprint.GetOnBits(), dtype=np.int32, count=stop - start)
    return sparse.csr_matrix(
        (np.ones(len(indices), dtype=np.float32), indices, indptr),
        shape=(len(fingerprints), 2048),
    )


def _save_figure(fig: plt.Figure, stem: Path) -> None:
    """Save one Python figure in preview, vector, and print formats."""
    fig.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(
        stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def build_full_redocking_median_plot() -> None:
    """Plot repeat-level full-redocking medians after largest-fragment preprocessing."""
    summary_path = FORMAL_DIR / "formal_3rep_docking_model_summary.csv"
    summary = pd.read_csv(summary_path).set_index("model").reindex(MODEL_ORDER)
    means = summary["score_median_mean"].to_numpy(dtype=float)
    errors = summary["score_median_std"].to_numpy(dtype=float)

    fig, axis = plt.subplots(figsize=(11.2, 6.4))
    positions = np.arange(len(MODEL_ORDER))
    bars = axis.bar(
        positions,
        means,
        yerr=errors,
        color=[MODEL_COLORS[model] for model in MODEL_ORDER],
        width=0.72,
        capsize=5,
        error_kw={"elinewidth": 1.5, "capthick": 1.5},
    )
    for bar, mean, error in zip(bars, means, errors):
        mean_label = Decimal(str(mean)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
        error_label = Decimal(str(error)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            mean - error - 0.28,
            f"{mean_label} +/- {error_label}",
            ha="center",
            va="top",
            fontsize=12,
        )
    axis.axhline(0, color="#303030", linewidth=0.9)
    axis.set_xticks(positions, MODEL_ORDER, rotation=20, ha="right")
    axis.set_ylabel("UniDock score median (kcal/mol)")
    axis.set_ylim(min(means - errors) - 1.25, 0.45)
    axis.set_title(
        "Full-redocking median after largest-fragment preprocessing",
        loc="left",
    )
    axis.grid(axis="y", color="#D5D8DC", linewidth=0.7, alpha=0.7)
    axis.set_axisbelow(True)
    fig.tight_layout()
    _save_figure(fig, OUTPUT_DIR / "model_comparison_full_docking_median")


def build_chembl30k_umap() -> pd.DataFrame:
    """Fit one joint ECFP4-UMAP for ChEMBL36 30k and all five generator groups."""
    chembl = pd.read_csv(CHEMBL_30K, usecols=["sample_order", "canonical_smiles"])
    chembl["model"] = "ChEMBL36"
    chembl["repeat"] = pd.NA
    chembl["molecule_id"] = "chembl30k_" + chembl["sample_order"].astype(str)

    generated = pd.read_csv(
        GENERATED_CSV,
        usecols=["row_id", "repeat", "model", "sanitize_success", "canonical_smiles"],
    )
    generated = generated[
        generated["sanitize_success"].eq(1)
        & generated["model"].isin(MODEL_ORDER)
        & generated["canonical_smiles"].notna()
    ].copy()
    generated["molecule_id"] = "generated_" + generated["row_id"].astype(str)

    chembl, chembl_fp = _fingerprint_frame(chembl)
    generated, generated_fp = _fingerprint_frame(generated)
    matrix = _to_csr([*chembl_fp, *generated_fp])
    LOGGER.info("fitting joint UMAP matrix=%s", matrix.shape)
    coordinates = umap.UMAP(
        n_neighbors=20,
        min_dist=0.1,
        n_components=2,
        metric="cosine",
        random_state=42,
        low_memory=True,
        verbose=True,
    ).fit_transform(matrix)

    identity = pd.concat(
        [
            chembl[["model", "repeat", "molecule_id", "canonical_smiles"]],
            generated[["model", "repeat", "molecule_id", "canonical_smiles"]],
        ],
        ignore_index=True,
    )
    identity["umap1"] = coordinates[:, 0]
    identity["umap2"] = coordinates[:, 1]
    identity.to_csv(OUTPUT_DIR / "five_model_chembl30k_joint_umap_coordinates.csv.gz", index=False)

    summary_rows: list[dict[str, Any]] = []
    for group_name in ["ChEMBL36", *MODEL_ORDER]:
        group = identity[identity["model"].eq(group_name)]
        xy = group[["umap1", "umap2"]].to_numpy()
        centroid = xy.mean(axis=0)
        summary_rows.append(
            {
                "group": group_name,
                "n": int(len(group)),
                "unique_canonical_smiles": int(group["canonical_smiles"].nunique()),
                "unique_fraction": float(group["canonical_smiles"].nunique() / len(group)),
                "centroid_umap1": float(centroid[0]),
                "centroid_umap2": float(centroid[1]),
                "within_group_rms_radius": float(
                    np.sqrt(np.mean(np.sum((xy - centroid) ** 2, axis=1)))
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / "five_model_chembl30k_group_summary.csv", index=False)

    chembl_xy = identity[identity["model"].eq("ChEMBL36")]
    fig, axis = plt.subplots(figsize=(11.8, 7.0))
    axis.scatter(
        chembl_xy["umap1"], chembl_xy["umap2"], s=1.1, color=CHEMBL_COLOR,
        alpha=0.15, linewidths=0, rasterized=True, label=f"ChEMBL36 (n={len(chembl_xy):,})",
    )
    for model in MODEL_ORDER:
        group = identity[identity["model"].eq(model)]
        axis.scatter(
            group["umap1"], group["umap2"], s=7, color=MODEL_COLORS[model],
            alpha=0.46, linewidths=0, rasterized=True, label=f"{model} (n={len(group):,})",
        )
    axis.set_xlabel("UMAP 1")
    axis.set_ylabel("UMAP 2")
    axis.set_title("Joint ECFP4 chemical space: five original models vs ChEMBL36 30k", loc="left")
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=3, markerscale=2)
    _save_figure(fig, OUTPUT_DIR / "five_model_chembl30k_umap_overview")

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.2), sharex=True, sharey=True)
    for axis, model in zip(axes.flat, MODEL_ORDER):
        group = identity[identity["model"].eq(model)]
        axis.scatter(
            chembl_xy["umap1"], chembl_xy["umap2"], s=0.8, color=CHEMBL_COLOR,
            alpha=0.10, linewidths=0, rasterized=True,
        )
        axis.scatter(
            group["umap1"], group["umap2"], s=6, color=MODEL_COLORS[model],
            alpha=0.52, linewidths=0, rasterized=True,
        )
        axis.set_title(f"{model} (n={len(group):,})", color=MODEL_COLORS[model], loc="left")
    legend_axis = axes.flat[-1]
    legend_axis.axis("off")
    legend_axis.text(
        0.05, 0.72,
        "Common joint embedding\nChEMBL36: n=30,000\nGenerated: n=10,910\nECFP4: radius 2, 2048 bits\nUMAP: cosine, neighbors 20\nseed: 42",
        va="top", ha="left", linespacing=1.45,
    )
    fig.supxlabel("UMAP 1", y=0.03)
    fig.supylabel("UMAP 2", x=0.04)
    fig.suptitle("Model-specific coverage against the same ChEMBL36 30k background", x=0.07, ha="left")
    fig.tight_layout(rect=(0.04, 0.05, 1, 0.95))
    _save_figure(fig, OUTPUT_DIR / "five_model_chembl30k_umap_facets")

    manifest = {
        "analysis": "joint ECFP4-UMAP of five formal benchmark models and frozen ChEMBL36 30k",
        "counts": {
            "chembl": int(len(chembl)),
            "generated_total": int(len(generated)),
            "generated_by_model": generated.groupby("model").size().reindex(MODEL_ORDER).to_dict(),
        },
        "sources": {
            "chembl_30k": str(CHEMBL_30K),
            "generated_molecules": str(GENERATED_CSV),
            "sha256": {"chembl_30k": _sha256(CHEMBL_30K), "generated_molecules": _sha256(GENERATED_CSV)},
        },
        "projection": {
            "fingerprint": "ECFP4 radius=2 n_bits=2048",
            "fit_scope": "one joint fit on frozen ChEMBL36 30k plus all sanitize-valid five-model molecules",
            "algorithm": "UMAP",
            "metric": "cosine",
            "n_neighbors": 20,
            "min_dist": 0.1,
            "random_state": 42,
        },
        "interpretation_boundary": "UMAP is descriptive; distance is not target efficacy or absolute molecular quality.",
    }
    with (OUTPUT_DIR / "five_model_chembl30k_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    return summary


def main() -> None:
    """Build every detailed benchmark asset."""
    _configure_logging()
    _configure_plotting()
    RDLogger.DisableLog("rdApp.warning")
    top10 = build_top10_assets()
    summary = build_chembl30k_umap()
    LOGGER.info("complete top10_rows=%d umap_groups=%d", len(top10), len(summary))


if __name__ == "__main__":
    main()
