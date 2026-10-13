#!/usr/bin/env python3
"""Fit separate generated-only ECFP4-UMAP maps for cumulative 500-step rounds.

The three scopes are the first 3 rounds (rep4-rep6), first 6 rounds
(rep4-rep9), and first 9 rounds (rep4-rep9 plus rep13-rep15).  Each scope is
fit independently using the same UMAP configuration.  ChEMBL is deliberately
not included in these fits so that the map describes differences among the
five generated-model distributions rather than a ChEMBL-anchored reference
space.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import logging
import os
import pickle
import platform
import sys
from collections import OrderedDict
from multiprocessing import Pool
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/diffb110_matplotlib_260831")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/diffb110_umap_numba_cache_260831")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from scipy import sparse

RDLogger.DisableLog("rdApp.*")

import umap


ROOT = Path("/data/disk6/DiffB110")
OUTPUT_DEFAULT = ROOT / "analysis/benchmark/benchmark5/five_model_500_cumulative_generated_umap_260831"
MODEL_ORDER = ["DiffGui", "DiffSBDD", "PMDM", "GCDM-SBDD", "TargetDiff"]
MODEL_FILES = {
    "DiffGui": "diffgui.sdf",
    "DiffSBDD": "diffsbdd.sdf",
    "PMDM": "pmdm.sdf",
    "GCDM-SBDD": "gcdm.sdf",
    "TargetDiff": "targetdiff.sdf",
}
MODEL_COLORS = {
    "DiffGui": "#3B82C4",
    "DiffSBDD": "#D95F59",
    "PMDM": "#D9A441",
    "GCDM-SBDD": "#8E78B7",
    "TargetDiff": "#3D9B72",
}
REPEAT_ORDER = ["rep4", "rep5", "rep6", "rep7", "rep8", "rep9", "rep13", "rep14", "rep15"]
SCOPES = OrderedDict(
    [
        ("first3", ["rep4", "rep5", "rep6"]),
        ("first6", ["rep4", "rep5", "rep6", "rep7", "rep8", "rep9"]),
        ("first9", REPEAT_ORDER.copy()),
    ]
)
ATTEMPTS_PER_MODEL_REPEAT = 1000
RADIUS = 2
N_BITS = 2048
N_COMPONENTS = 2
METRIC = "cosine"
N_NEIGHBORS = 20
MIN_DIST = 0.1
RANDOM_STATE = 42
TRANSFORM_SEED = 42

LOGGER = logging.getLogger("fit_cumulative_umap")


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_logging(output_dir: Path) -> None:
    """Configure readable terminal and durable file logging."""
    output_dir.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "run.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(stream)
    root_logger.addHandler(file_handler)


def configure_plotting() -> None:
    """Apply the established readable Python plotting defaults."""
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 13,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "legend.frameon": False,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def source_path(repeat: str, model: str) -> Path:
    """Return the frozen aggregate SDF for one completed 500-step cell."""
    if repeat in {"rep4", "rep5", "rep6"}:
        directory = ROOT / "analysis/benchmark/benchmark5/five_model_repeats_500steps_260826"
        seeds = {"rep4": "rep4_seed22_29", "rep5": "rep5_seed30_37", "rep6": "rep6_seed38_45"}
    elif repeat in {"rep7", "rep8", "rep9"}:
        directory = ROOT / "analysis/benchmark/benchmark5/five_model_repeats_500steps_260826_cont1"
        seeds = {"rep7": "rep7_seed46_53", "rep8": "rep8_seed54_61", "rep9": "rep9_seed62_69"}
    elif repeat in {"rep13", "rep14", "rep15"}:
        directory = ROOT / "analysis/benchmark/benchmark5/five_model_repeats_500steps_260829"
        seeds = {"rep13": "rep13_seed94_101", "rep14": "rep14_seed102_109", "rep15": "rep15_seed110_117"}
    else:
        raise ValueError(f"unknown repeat: {repeat}")
    return directory / seeds[repeat] / "aggregated" / MODEL_FILES[model]


def source_specs() -> list[dict[str, Any]]:
    """Build and validate the 45 source-SDF specifications."""
    specs: list[dict[str, Any]] = []
    for repeat in REPEAT_ORDER:
        for model in MODEL_ORDER:
            path = source_path(repeat, model)
            if not path.exists():
                raise FileNotFoundError(path)
            specs.append(
                {
                    "step": 500,
                    "repeat": repeat,
                    "model": model,
                    "sdf": path,
                    "attempts": ATTEMPTS_PER_MODEL_REPEAT,
                }
            )
    return specs


def safe_canonical_smiles(mol: Chem.Mol) -> str:
    """Return a canonical non-isomeric SMILES for a sanitized molecule."""
    clean = Chem.RemoveHs(mol)
    return Chem.MolToSmiles(clean, canonical=True, isomericSmiles=False)


def read_source(spec: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read one SDF and retain fixed-attempt provenance for every written record."""
    rows: list[dict[str, Any]] = []
    supplier = Chem.SDMolSupplier(str(spec["sdf"]), sanitize=False, removeHs=False)
    for source_index, raw_mol in enumerate(supplier):
        row: dict[str, Any] = {
            "step": spec["step"],
            "repeat": spec["repeat"],
            "model": spec["model"],
            "source_sdf": str(spec["sdf"]),
            "source_index": source_index,
            "attempts": spec["attempts"],
            "rdkit_readable": int(raw_mol is not None),
            "sanitize_valid": 0,
            "canonical_smiles": "",
            "failure_reason": "",
        }
        if raw_mol is not None:
            try:
                mol = Chem.Mol(raw_mol)
                Chem.SanitizeMol(mol)
                row["canonical_smiles"] = safe_canonical_smiles(mol)
                row["sanitize_valid"] = 1
            except Exception as exc:  # noqa: BLE001
                row["failure_reason"] = type(exc).__name__
        rows.append(row)
    valid = [row for row in rows if row["sanitize_valid"]]
    summary = {
        "step": spec["step"],
        "repeat": spec["repeat"],
        "model": spec["model"],
        "attempts": spec["attempts"],
        "sdf_records": len(rows),
        "rdkit_readable": sum(row["rdkit_readable"] for row in rows),
        "sanitize_valid": len(valid),
        "valid_per_attempt": len(valid) / spec["attempts"],
        "unique_smiles": len({row["canonical_smiles"] for row in valid}),
        "source_sdf": str(spec["sdf"]),
        "source_sha256": sha256(spec["sdf"]),
    }
    return rows, summary


def collect_sources(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collect all source rows and fixed-attempt repeat summaries."""
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for spec in source_specs():
        rows, summary = read_source(spec)
        all_rows.extend(rows)
        summaries.append(summary)
        LOGGER.info(
            "source repeat=%s model=%s records=%d valid=%d",
            spec["repeat"], spec["model"], summary["sdf_records"], summary["sanitize_valid"],
        )
    molecule_frame = pd.DataFrame(all_rows)
    summary_frame = pd.DataFrame(summaries)
    molecule_frame.to_csv(output_dir / "all_source_records.csv.gz", index=False, compression="gzip")
    summary_frame.to_csv(output_dir / "source_summary.csv", index=False)
    return molecule_frame, summary_frame


def fingerprint_from_smiles(smiles: str) -> Any:
    """Calculate one ECFP4 bit vector."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, RADIUS, nBits=N_BITS)


def _fingerprint_task(task: tuple[int, str]) -> tuple[int, Any | None]:
    """Calculate one fingerprint while preserving row order."""
    index, smiles = task
    try:
        return index, fingerprint_from_smiles(smiles)
    except Exception:  # noqa: BLE001
        return index, None


def calculate_fingerprints(frame: pd.DataFrame, workers: int) -> tuple[pd.DataFrame, list[Any]]:
    """Calculate ECFP4 vectors for valid rows and reject impossible fingerprints."""
    valid = frame[frame["sanitize_valid"].eq(1)].copy().reset_index(drop=True)
    tasks = list(enumerate(valid["canonical_smiles"].astype(str).tolist()))
    LOGGER.info("calculating ECFP4 fingerprints n=%d workers=%d", len(tasks), workers)
    with Pool(processes=workers) as pool:
        results = pool.map(_fingerprint_task, tasks, chunksize=128)
    keep = [(index, fp) for index, fp in results if fp is not None]
    if len(keep) != len(valid):
        keep_indices = [index for index, _ in keep]
        valid = valid.iloc[keep_indices].copy().reset_index(drop=True)
        LOGGER.warning("dropped %d valid rows without ECFP4", len(tasks) - len(keep))
    fingerprints = [fp for _, fp in keep]
    valid["valid_row_id"] = np.arange(len(valid), dtype=np.int64)
    return valid, fingerprints


def fingerprints_to_csr(fingerprints: list[Any]) -> sparse.csr_matrix:
    """Convert RDKit bit vectors to sparse CSR format."""
    indptr = np.zeros(len(fingerprints) + 1, dtype=np.int64)
    for index, fingerprint in enumerate(fingerprints):
        indptr[index + 1] = indptr[index] + fingerprint.GetNumOnBits()
    indices = np.empty(int(indptr[-1]), dtype=np.int32)
    for index, fingerprint in enumerate(fingerprints):
        start, stop = int(indptr[index]), int(indptr[index + 1])
        indices[start:stop] = np.fromiter(fingerprint.GetOnBits(), dtype=np.int32, count=stop - start)
    return sparse.csr_matrix(
        (np.ones(len(indices), dtype=np.float32), indices, indptr),
        shape=(len(fingerprints), N_BITS),
    )


def summarize_coordinates(frame: pd.DataFrame, coordinates: np.ndarray) -> pd.DataFrame:
    """Summarize each model and repeat within one fitted coordinate system."""
    table = frame.copy()
    table["umap1"] = coordinates[:, 0]
    table["umap2"] = coordinates[:, 1]
    rows: list[dict[str, Any]] = []
    for repeat in REPEAT_ORDER:
        for model in MODEL_ORDER:
            group = table[(table["repeat"] == repeat) & (table["model"] == model)]
            if group.empty:
                continue
            xy = group[["umap1", "umap2"]].to_numpy()
            centroid = xy.mean(axis=0)
            radius = np.sqrt(np.sum((xy - centroid) ** 2, axis=1))
            rows.append(
                {
                    "repeat": repeat,
                    "model": model,
                    "n": len(group),
                    "unique_smiles": group["canonical_smiles"].nunique(),
                    "centroid_umap1": centroid[0],
                    "centroid_umap2": centroid[1],
                    "rms_radius": float(np.sqrt(np.mean(radius**2))),
                    "median_radius": float(np.median(radius)),
                }
            )
    for model in MODEL_ORDER:
        group = table[table["model"] == model]
        xy = group[["umap1", "umap2"]].to_numpy()
        centroid = xy.mean(axis=0)
        radius = np.sqrt(np.sum((xy - centroid) ** 2, axis=1))
        rows.append(
            {
                "repeat": "pooled",
                "model": model,
                "n": len(group),
                "unique_smiles": group["canonical_smiles"].nunique(),
                "centroid_umap1": centroid[0],
                "centroid_umap2": centroid[1],
                "rms_radius": float(np.sqrt(np.mean(radius**2))),
                "median_radius": float(np.median(radius)),
            }
        )
    return pd.DataFrame(rows), table


def save_scope_manifest(
    scope_dir: Path,
    scope_name: str,
    repeats: list[str],
    source_summary: pd.DataFrame,
    n_valid: int,
    elapsed_seconds: float,
) -> None:
    """Save reproducibility metadata for one independent fit."""
    selected = source_summary[source_summary["repeat"].isin(repeats)].copy()
    payload = {
        "status": "complete",
        "scope": scope_name,
        "repeats": repeats,
        "fit_scope": "all sanitize-valid generated molecules from selected 500-step five-model repeats",
        "n_valid_fit_rows": n_valid,
        "attempts_per_model_repeat": ATTEMPTS_PER_MODEL_REPEAT,
        "models": MODEL_ORDER,
        "fingerprint": {"type": "ECFP4", "radius": RADIUS, "n_bits": N_BITS},
        "umap": {
            "metric": METRIC,
            "n_components": N_COMPONENTS,
            "n_neighbors": N_NEIGHBORS,
            "min_dist": MIN_DIST,
            "random_state": RANDOM_STATE,
            "transform_seed": TRANSFORM_SEED,
            "low_memory": True,
            "fit_threads": 1,
        },
        "source_summary": selected.to_dict(orient="records"),
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "umap_learn": umap.__version__,
        },
        "elapsed_seconds": elapsed_seconds,
        "interpretation_boundary": "Coordinates are descriptive within this fitted map only; maps from first3, first6, and first9 have independent axes and cannot be compared pointwise.",
    }
    (scope_dir / "manifest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def fit_scope(
    scope_name: str,
    repeats: list[str],
    records: pd.DataFrame,
    source_summary: pd.DataFrame,
    output_dir: Path,
    workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit and export one cumulative generated-only UMAP scope."""
    import time

    started = time.time()
    scope_dir = output_dir / scope_name
    scope_dir.mkdir(parents=True, exist_ok=True)
    selected = records[records["repeat"].isin(repeats)].copy()
    valid, fingerprints = calculate_fingerprints(selected, workers)
    matrix = fingerprints_to_csr(fingerprints)
    LOGGER.info("fit scope=%s repeats=%s shape=%s", scope_name, ",".join(repeats), matrix.shape)
    reducer = umap.UMAP(
        n_components=N_COMPONENTS,
        metric=METRIC,
        n_neighbors=N_NEIGHBORS,
        min_dist=MIN_DIST,
        random_state=RANDOM_STATE,
        transform_seed=TRANSFORM_SEED,
        low_memory=True,
        n_jobs=1,
        verbose=True,
    )
    coordinates = reducer.fit_transform(matrix)
    if coordinates.shape != (len(valid), 2) or not np.isfinite(coordinates).all():
        raise ValueError(f"invalid coordinates for {scope_name}: {coordinates.shape}")
    coordinate_summary, coordinate_table = summarize_coordinates(valid, coordinates)
    valid.to_csv(scope_dir / "fit_input_records.csv.gz", index=False, compression="gzip")
    coordinate_table.to_csv(scope_dir / "coordinates.csv.gz", index=False, compression="gzip")
    coordinate_summary.to_csv(scope_dir / "coordinate_summary.csv", index=False)
    sparse.save_npz(scope_dir / "fit_ecfp4_csr.npz", matrix)
    np.save(scope_dir / "coordinates.npy", coordinates.astype(np.float32))
    with (scope_dir / "umap_model.joblib").open("wb") as handle:
        import joblib

        joblib.dump(reducer, handle, compress=3)
    save_scope_manifest(scope_dir, scope_name, repeats, source_summary, len(valid), time.time() - started)
    (scope_dir / "completion.json").write_text(json.dumps({"status": "complete", "stage": "fit"}, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("completed scope=%s n_valid=%d elapsed=%.1fs", scope_name, len(valid), time.time() - started)
    return coordinate_summary, coordinate_table


def save_summary_tables(
    scope_summaries: dict[str, pd.DataFrame],
    source_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Write cross-scope count tables without comparing independent axes."""
    rows: list[dict[str, Any]] = []
    for scope_name, repeats in SCOPES.items():
        selected = source_summary[source_summary["repeat"].isin(repeats)]
        for model in MODEL_ORDER:
            group = selected[selected["model"] == model]
            rows.append(
                {
                    "scope": scope_name,
                    "repeat_count": len(repeats),
                    "repeats": ",".join(repeats),
                    "model": model,
                    "attempts": int(group["attempts"].sum()),
                    "sanitize_valid": int(group["sanitize_valid"].sum()),
                    "valid_per_attempt": float(group["sanitize_valid"].sum() / group["attempts"].sum()),
                    "unique_smiles_within_repeat_sum": int(group["unique_smiles"].sum()),
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "cumulative_validity_summary.csv", index=False)
    pooled_rows: list[dict[str, Any]] = []
    for scope_name, summary_frame in scope_summaries.items():
        pooled = summary_frame[summary_frame["repeat"] == "pooled"]
        for row in pooled.to_dict(orient="records"):
            row["scope"] = scope_name
            pooled_rows.append(row)
    pd.DataFrame(pooled_rows).to_csv(output_dir / "cumulative_coordinate_summary.csv", index=False)


def plot_facets(scope_tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Plot each independently fitted map in a 3-by-5 descriptive facet grid."""
    configure_plotting()
    fig, axes = plt.subplots(3, 5, figsize=(22, 13), squeeze=False)
    for row_index, (scope_name, table) in enumerate(scope_tables.items()):
        for col_index, model in enumerate(MODEL_ORDER):
            axis = axes[row_index, col_index]
            group = table[table["model"] == model]
            axis.scatter(
                group["umap1"],
                group["umap2"],
                s=4,
                alpha=0.28,
                color=MODEL_COLORS[model],
                linewidths=0,
                rasterized=True,
            )
            centroid = group[["umap1", "umap2"]].mean().to_numpy()
            axis.scatter([centroid[0]], [centroid[1]], s=48, color="black", marker="x", linewidths=1.5)
            axis.set_title(f"{model}\n{scope_name} | n={len(group):,}")
            if row_index == len(SCOPES) - 1:
                axis.set_xlabel("UMAP 1")
            if col_index == 0:
                axis.set_ylabel("UMAP 2")
            axis.grid(alpha=0.12, linewidth=0.5)
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.075, top=0.91, hspace=0.62, wspace=0.28)
    fig.suptitle("500-step cumulative generated-only ECFP4-UMAP fits", fontsize=21, y=0.995)
    fig.text(
        0.5,
        0.005,
        "Each row is an independent fit: first3 = rep4-rep6; first6 = rep4-rep9; first9 = rep4-rep9 + rep13-rep15. Axes are not comparable across rows.",
        ha="center",
        fontsize=11,
    )
    for suffix in ["png", "svg", "pdf"]:
        fig.savefig(output_dir / f"cumulative_generated_umap_facets.{suffix}", dpi=350 if suffix == "png" else None, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_validity(summary_path: Path, output_dir: Path) -> None:
    """Plot fixed-attempt validity across cumulative scopes."""
    configure_plotting()
    summary = pd.read_csv(summary_path)
    fig, axis = plt.subplots(figsize=(12, 6))
    x = np.arange(len(MODEL_ORDER))
    width = 0.24
    for index, scope_name in enumerate(SCOPES):
        group = summary[summary["scope"] == scope_name].set_index("model").reindex(MODEL_ORDER)
        axis.bar(
            x + (index - 1) * width,
            group["valid_per_attempt"] * 100,
            width,
            color=[MODEL_COLORS[model] for model in MODEL_ORDER],
            alpha=[0.95, 0.72, 0.5][index],
            label=scope_name,
        )
    axis.set_xticks(x)
    axis.set_xticklabels(MODEL_ORDER, rotation=25, ha="right")
    axis.set_ylabel("Sanitize-valid / fixed attempts (%)")
    axis.set_ylim(0, 105)
    axis.set_title("500-step cumulative generation validity")
    axis.legend(title="Cumulative scope")
    fig.savefig(output_dir / "cumulative_validity.png", dpi=350, bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / "cumulative_validity.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_report(output_dir: Path, source_summary: pd.DataFrame) -> None:
    """Write a concise report with tables, figures, and interpretation limits."""
    summary = pd.read_csv(output_dir / "cumulative_validity_summary.csv")
    lines = [
        "# 500步累计前三轮、前六轮、前九轮生成分子联合UMAP",
        "",
        "本分析按上一版配置分别拟合三个独立的 generated-only ECFP4-UMAP。拟合输入仅为五模型的 RDKit sanitize-valid 生成分子，不加入 ChEMBL reference。",
        "",
        "## 配置",
        "",
        "| 参数 | 值 |",
        "|---|---|",
        f"| Fingerprint | ECFP4, radius={RADIUS}, nBits={N_BITS} |",
        f"| UMAP metric | {METRIC} |",
        f"| n_neighbors | {N_NEIGHBORS} |",
        f"| min_dist | {MIN_DIST} |",
        f"| random_state | {RANDOM_STATE} |",
        "| fit threads | 1 |",
        "| 失败结构 | 不进入指纹/UMAP，但保留在固定 attempts 统计中 |",
        "",
        "## 累计范围",
        "",
        "| scope | repeats | fixed attempts | valid fit rows |",
        "|---|---|---:|---:|",
    ]
    for scope_name, repeats in SCOPES.items():
        selected = source_summary[source_summary["repeat"].isin(repeats)]
        lines.append(f"| {scope_name} | {', '.join(repeats)} | {int(selected['attempts'].sum()):,} | {int(selected['sanitize_valid'].sum()):,} |")
    lines += [
        "",
        "![Cumulative generated-only UMAP facets](cumulative_generated_umap_facets.png)",
        "",
        "每一行是一次独立 UMAP fit。三行的 UMAP 轴、方向、尺度和坐标距离不能进行点对点比较；图只用于观察每个累计数据集内部的五模型相对分布。",
        "",
        "![Cumulative validity](cumulative_validity.png)",
        "",
        "## 固定 attempts 汇总",
        "",
        "| scope | model | attempts | sanitize-valid | valid / attempts |",
        "|---|---|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(f"| {row.scope} | {row.model} | {int(row.attempts):,} | {int(row.sanitize_valid):,} | {row.valid_per_attempt:.4f} |")
    lines += [
        "",
        "## 解释边界",
        "",
        "- 这次拟合的目的，是减少 ChEMBL reference 对生成组间分布图的支配，增加五模型内部差异的可见性。",
        "- 更明显的二维分离不等于模型质量更高，也不等于 docking、活性、药物性或结构更简单。",
        "- 跨累计范围的定量比较应使用原始 ECFP4/Tanimoto、合法率、unique SMILES、Murcko scaffold 和其他独立指标；不能比较三个独立 UMAP 的坐标距离。",
        "- 500 步第一批的 `rep4–rep8`、后续 `rep9`、以及追加的 `rep13–rep15` 作为固定来源保留；失败的 retry 目录不计入。",
        "",
        "## 输出",
        "",
        "- `source_summary.csv`: 45 个模型-重复来源的固定 attempts 和合法性摘要。",
        "- `all_source_records.csv.gz`: 所有 SDF 写入记录及其来源身份。",
        "- `first3/`, `first6/`, `first9/`: 各自的坐标、fit 输入、稀疏 ECFP4 矩阵、UMAP 模型和 manifest。",
        "- `cumulative_validity_summary.csv`: 三个累计范围的模型级固定 attempts 汇总。",
        "- `cumulative_coordinate_summary.csv`: 各独立坐标系内的 centroid 和半径摘要。",
    ]
    (output_dir / "analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    """Run collection, three independent fits, plots, and report generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--workers", type=int, default=18)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    configure_logging(output_dir)
    LOGGER.info("command=%s", " ".join(sys.argv))
    LOGGER.info("output_dir=%s workers=%d", output_dir, args.workers)
    records, source_summary = collect_sources(output_dir)
    scope_summaries: dict[str, pd.DataFrame] = {}
    scope_tables: dict[str, pd.DataFrame] = {}
    for scope_name, repeats in SCOPES.items():
        coordinate_summary, coordinate_table = fit_scope(
            scope_name, repeats, records, source_summary, output_dir, args.workers
        )
        scope_summaries[scope_name] = coordinate_summary
        scope_tables[scope_name] = coordinate_table
    save_summary_tables(scope_summaries, source_summary, output_dir)
    plot_facets(scope_tables, output_dir)
    plot_validity(output_dir / "cumulative_validity_summary.csv", output_dir)
    write_report(output_dir, source_summary)
    payload = {
        "status": "complete",
        "stage": "all",
        "scopes": {name: repeats for name, repeats in SCOPES.items()},
        "source_count": len(source_summary),
        "total_source_attempts": int(source_summary["attempts"].sum()),
        "total_sanitize_valid": int(source_summary["sanitize_valid"].sum()),
        "workers": args.workers,
        "source_manifest_sha256": sha256(output_dir / "source_summary.csv"),
    }
    (output_dir / "manifest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "completion.json").write_text(json.dumps({"status": "complete", "stage": "all"}, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("completed all cumulative fits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
