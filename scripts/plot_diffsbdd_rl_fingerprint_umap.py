#!/usr/bin/env python3
"""Compare traced DiffSBDD RL runs with one frozen ChEMBL ECFP4-UMAP reference."""

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
LOGGER = logging.getLogger("diffsbdd_rl_fingerprint_umap")
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHEMBL = ROOT / "analysis/key_rl_fingerprint_umap_20260729/chembl_sample_30000_frozen.csv.gz"
DEFAULT_LEGACY = ROOT / "analysis/diffsbdd_rl/rl_high_carbon_sparse_reward_20260604/selected_epoch_molecule_flags.csv.gz"
DEFAULT_OUTPUT = ROOT / "analysis/diffsbdd_rl_fingerprint_umap_20260729"
GUARD_RUN = ROOT / "runs/diffsbdd/diffsbdd-8gpu-b64-anticarbon-guard-20260611_20260611_155548"
SINGLE_GLOB = "runs/diffsbdd/diffsbdd-single-reward-*/epochs/epoch_*/molecules.csv"

LEGACY_NAMES = {
    "dock_classifier_0528": "dock+classifier 0528",
    "dock_classifier_0529": "dock+classifier 0529",
    "classifier_only_0526": "classifier only",
    "dock_prior_logp_0525": "dock+ECFP/FCFP+logP",
    "le_prior_logp_0526": "LE+ECFP/FCFP+logP",
}
SINGLE_REWARDS = [
    "docking",
    "qed",
    "sa",
    "tpsa_distribution",
    "ecfp4_prior",
    "fcfp4_prior",
    "logp_distribution",
    "ligand_efficiency",
    "chembl_classifier",
]
SINGLE_NAMES = {
    "docking": "single: docking",
    "qed": "single: QED",
    "sa": "single: SA",
    "tpsa_distribution": "single: TPSA dist.",
    "ecfp4_prior": "single: ECFP4 prior",
    "fcfp4_prior": "single: FCFP4 prior",
    "logp_distribution": "single: logP dist.",
    "ligand_efficiency": "single: ligand efficiency",
    "chembl_classifier": "single: ChEMBL classifier",
}
METHOD_ORDER = list(LEGACY_NAMES.values()) + ["anti-carbon guard b64"] + [
    SINGLE_NAMES[item] for item in SINGLE_REWARDS
]
CHEMBL_HASH = "f6c79b6c939f40cae78e6363a5ab3ff2091da1758efddb7f70d119ce81af51c3"


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chembl-csv", type=Path, default=DEFAULT_CHEMBL)
    parser.add_argument("--legacy-csv", type=Path, default=DEFAULT_LEGACY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=18)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--nbits", type=int, default=2048)
    parser.add_argument("--n-neighbors", type=int, default=20)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def _configure_plotting() -> None:
    """Configure compact publication-style plots."""
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "figure.facecolor": "white",
        }
    )


def _ordered_hash(smiles: list[str]) -> str:
    """Hash ordered SMILES with the established newline convention."""
    return hashlib.sha256(("\n".join(smiles) + "\n").encode("utf-8")).hexdigest()


def _cohort_for_epoch(epoch: int, minimum: int, maximum: int) -> str:
    """Assign exact first and last epochs to early/late and all others to middle."""
    if epoch == minimum:
        return "early"
    if epoch == maximum:
        return "late"
    return "middle"


def _normalize_rows(frame: pd.DataFrame, method: str, source_file: str) -> pd.DataFrame:
    """Normalize one molecule table while retaining invalid attempts for accounting."""
    output = frame[["epoch", "smiles", "rdkit_valid"]].copy()
    output["epoch"] = pd.to_numeric(output["epoch"], errors="raise").astype(int)
    output["rdkit_valid"] = pd.to_numeric(output["rdkit_valid"], errors="coerce").fillna(0).astype(bool)
    output["method"] = method
    output["source_file"] = source_file
    output["source_row"] = np.arange(len(output), dtype=int)
    return output


def _load_rl_attempts(legacy_path: Path) -> pd.DataFrame:
    """Load selected legacy runs, anti-carbon guard, and the nine single-reward runs."""
    frames: list[pd.DataFrame] = []
    legacy = pd.read_csv(
        legacy_path,
        usecols=["run_label", "epoch", "smiles", "rdkit_valid"],
        low_memory=False,
    )
    for run_label, method in LEGACY_NAMES.items():
        subset = legacy[legacy["run_label"] == run_label]
        if subset.empty:
            raise ValueError(f"missing legacy DiffSBDD run: {run_label}")
        frames.append(_normalize_rows(subset, method, str(legacy_path)))

    guard_files = sorted(GUARD_RUN.glob("epochs/epoch_*/molecules.csv"))
    if not guard_files:
        raise FileNotFoundError(f"no anti-carbon guard molecule tables under {GUARD_RUN}")
    for path in guard_files:
        frames.append(
            _normalize_rows(
                pd.read_csv(path, usecols=["epoch", "smiles", "rdkit_valid"]),
                "anti-carbon guard b64",
                str(path),
            )
        )

    single_files = sorted(ROOT.glob(SINGLE_GLOB))
    if len(single_files) != 90:
        raise ValueError(f"expected 90 single-reward epoch tables, found {len(single_files)}")
    for path in single_files:
        run_name = path.parents[2].name
        reward = next((item for item in SINGLE_REWARDS if f"-{item}_" in run_name), None)
        if reward is None:
            raise ValueError(f"cannot identify single reward from {run_name}")
        frames.append(
            _normalize_rows(
                pd.read_csv(path, usecols=["epoch", "smiles", "rdkit_valid"]),
                SINGLE_NAMES[reward],
                str(path),
            )
        )
    attempts = pd.concat(frames, ignore_index=True)
    bounds = attempts.groupby("method")["epoch"].agg(["min", "max"])
    attempts["cohort"] = [
        _cohort_for_epoch(epoch, int(bounds.loc[method, "min"]), int(bounds.loc[method, "max"]))
        for method, epoch in zip(attempts["method"], attempts["epoch"])
    ]
    return attempts


def _fingerprint_task(task: tuple[int, str, int, int]) -> tuple[int, np.ndarray | None]:
    """Calculate one Morgan fingerprint."""
    index, smiles, radius, nbits = task
    try:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return index, None
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nBits=nbits)
        array = np.zeros((nbits,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fingerprint, array)
        return index, array
    except Exception:
        return index, None


def _fingerprints(smiles: list[str], radius: int, nbits: int, workers: int) -> tuple[np.ndarray, np.ndarray]:
    """Calculate fingerprints in input order."""
    tasks = [(index, value, radius, nbits) for index, value in enumerate(smiles)]
    with Pool(processes=workers) as pool:
        results = pool.map(_fingerprint_task, tasks, chunksize=128)
    valid = [(index, fp) for index, fp in results if fp is not None]
    return (
        np.asarray([item[0] for item in valid], dtype=int),
        np.vstack([item[1] for item in valid]).astype(np.uint8, copy=False),
    )


def _save_figure(figure: plt.Figure, stem: Path) -> None:
    """Save raster and vector figure formats."""
    figure.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _summaries(embedding: pd.DataFrame, attempts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build validity, cohort geometry, and epoch-centroid summaries."""
    validity = (
        attempts.groupby(["method", "epoch", "cohort"], as_index=False)
        .agg(attempts=("smiles", "size"), sanitize_valid=("rdkit_valid", "sum"))
    )
    validity["sanitize_valid_rate"] = validity["sanitize_valid"] / validity["attempts"]
    chembl_centroid = embedding[embedding["dataset"] == "ChEMBL"][["umap1", "umap2"]].mean().to_numpy()
    rows: list[dict[str, Any]] = []
    rl = embedding[embedding["dataset"] == "RL"]
    for (method, cohort), group in rl.groupby(["method", "cohort"], sort=False):
        coordinates = group[["umap1", "umap2"]].to_numpy()
        centroid = coordinates.mean(axis=0)
        rows.append(
            {
                "method": method,
                "cohort": cohort,
                "n": len(group),
                "centroid_umap1": centroid[0],
                "centroid_umap2": centroid[1],
                "centroid_distance_to_chembl": np.linalg.norm(centroid - chembl_centroid),
                "within_group_rms_radius": np.sqrt(np.mean(np.sum((coordinates - centroid) ** 2, axis=1))),
            }
        )
    cohort = pd.DataFrame(rows)
    epoch = (
        rl.groupby(["method", "epoch", "cohort"], as_index=False)
        .agg(n=("smiles", "size"), umap1=("umap1", "mean"), umap2=("umap2", "mean"))
    )
    return validity, cohort, epoch


def _plot_family(
    embedding: pd.DataFrame,
    epoch: pd.DataFrame,
    methods: list[str],
    title: str,
    output_stem: Path,
) -> None:
    """Plot comparable early/late facets for one family of DiffSBDD runs."""
    chembl = embedding[embedding["dataset"] == "ChEMBL"]
    columns = 3
    rows = int(np.ceil(len(methods) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(10.8, 3.25 * rows), sharex=True, sharey=True)
    flat = np.asarray(axes).reshape(-1)
    for axis, method in zip(flat, methods):
        group = embedding[(embedding["dataset"] == "RL") & (embedding["method"] == method)]
        axis.scatter(chembl["umap1"], chembl["umap2"], s=1, color="#B8B8B8", alpha=0.10, linewidths=0, rasterized=True)
        for cohort, color in [("early", "#3B82C4"), ("late", "#D95F59")]:
            subset = group[group["cohort"] == cohort]
            axis.scatter(
                subset["umap1"], subset["umap2"], s=8, color=color, alpha=0.58,
                linewidths=0, label=f"{cohort.capitalize()} (n={len(subset):,})", rasterized=True,
            )
        trajectory = epoch[epoch["method"] == method].sort_values("epoch")
        if len(trajectory) > 1:
            axis.plot(trajectory["umap1"], trajectory["umap2"], color="#333333", linewidth=0.7, alpha=0.65)
        axis.set_title(method, fontsize=8)
        axis.set_xlabel("UMAP1")
        axis.set_ylabel("UMAP2")
        axis.legend(loc="best", fontsize=6)
    for axis in flat[len(methods):]:
        axis.axis("off")
    figure.suptitle(title, fontsize=11, y=0.995)
    figure.text(
        0.5, 0.004,
        "ECFP4 (radius 2, 2048 bits); joint UMAP; grey = identical frozen 30,000-molecule ChEMBL sample",
        ha="center", fontsize=7,
    )
    figure.tight_layout(rect=(0, 0.025, 1, 0.98))
    _save_figure(figure, output_stem)


def _write_report(output_dir: Path, attempts: pd.DataFrame, validity: pd.DataFrame, cohort: pd.DataFrame) -> None:
    """Write the concise analysis report with embedded plots."""
    early_late = validity[validity["cohort"].isin(["early", "late"])].copy()
    valid_pivot = early_late.pivot(index="method", columns="cohort", values="sanitize_valid").reindex(METHOD_ORDER)
    rate_pivot = early_late.pivot(index="method", columns="cohort", values="sanitize_valid_rate").reindex(METHOD_ORDER)
    geometry = cohort[cohort["cohort"].isin(["early", "late"])].pivot(
        index="method", columns="cohort", values="centroid_distance_to_chembl"
    ).reindex(METHOD_ORDER)
    lines = [
        "# DiffSBDD RL fingerprint UMAP versus frozen ChEMBL36",
        "",
        "## Identity and scope",
        "",
        "- Frozen ChEMBL sample: `30,000` molecules.",
        f"- Ordered ChEMBL SMILES SHA-256: `{CHEMBL_HASH}`.",
        f"- DiffSBDD runs: `{attempts['method'].nunique()}`; generated attempts represented: `{len(attempts):,}`.",
        "- Fingerprints and UMAP distributions use only `rdkit_valid=True` molecules; validity denominators retain every generated attempt.",
        "- ECFP4 radius 2, 2048 bits; joint cosine UMAP, n_neighbors=20, min_dist=0.1, random_state=42.",
        "",
        "## Multi-objective and guard runs",
        "",
        "![Multi-objective DiffSBDD runs](diffsbdd_multiobjective_vs_frozen_chembl_umap.png)",
        "",
        "## Single-reward runs",
        "",
        "![Single-reward DiffSBDD runs](diffsbdd_single_reward_vs_frozen_chembl_umap.png)",
        "",
        "## Early-to-late summary",
        "",
        "| Training | valid early | valid late | validity early -> late | distance to ChEMBL early -> late |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHOD_ORDER:
        early_n = valid_pivot.loc[method].get("early", np.nan)
        late_n = valid_pivot.loc[method].get("late", np.nan)
        early_rate = rate_pivot.loc[method].get("early", np.nan)
        late_rate = rate_pivot.loc[method].get("late", np.nan)
        early_dist = geometry.loc[method].get("early", np.nan)
        late_dist = geometry.loc[method].get("late", np.nan)
        count_text = lambda value: "0" if pd.isna(value) else f"{int(value):,}"
        rate_text = lambda value: "NA" if pd.isna(value) else f"{value:.1%}"
        dist_text = lambda value: "NA" if pd.isna(value) else f"{value:.3f}"
        lines.append(
            f"| {method} | {count_text(early_n)} | {count_text(late_n)} | "
            f"{rate_text(early_rate)} -> {rate_text(late_rate)} | "
            f"{dist_text(early_dist)} -> {dist_text(late_dist)} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "UMAP is a neighborhood visualization rather than a calibrated chemical-distance metric. A smaller centroid distance does not by itself establish improved validity, drug-likeness, synthesis feasibility, or chemical-space coverage. Interpret it together with validity, within-cohort spread, carbon enrichment, TPSA, rings, and structural alerts.",
            "",
            "## Evidence files",
            "",
            "- [All joint coordinates](joint_umap_embeddings.csv.gz)",
            "- [Per-epoch validity denominators](epoch_validity_summary.csv)",
            "- [Cohort geometry](cohort_umap_summary.csv)",
            "- [Selected-epoch centroid trajectories](epoch_centroids.csv)",
            "- [Run manifest](manifest.json)",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run the full DiffSBDD RL fingerprint-space comparison."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(args.output_dir / "run.log", mode="w")],
    )
    _configure_plotting()
    chembl = pd.read_csv(args.chembl_csv)
    chembl_smiles = chembl["canonical_smiles"].astype(str).tolist()
    actual_hash = _ordered_hash(chembl_smiles)
    if len(chembl) != 30000 or actual_hash != CHEMBL_HASH:
        raise ValueError(f"frozen ChEMBL identity mismatch: n={len(chembl)} hash={actual_hash}")
    attempts = _load_rl_attempts(args.legacy_csv)
    valid_rl = attempts[attempts["rdkit_valid"] & attempts["smiles"].notna()].copy()
    valid_rl["canonical_smiles"] = valid_rl["smiles"].map(
        lambda value: Chem.MolToSmiles(Chem.MolFromSmiles(str(value)), canonical=True)
    )
    LOGGER.info(
        "loaded methods=%s attempts=%s sanitize_valid=%s",
        attempts["method"].nunique(), len(attempts), len(valid_rl),
    )
    all_smiles = chembl_smiles + valid_rl["canonical_smiles"].tolist()
    valid_indexes, matrix = _fingerprints(all_smiles, args.radius, args.nbits, args.workers)
    if len(valid_indexes) != len(all_smiles):
        raise ValueError(f"unexpected fingerprint failures: {len(all_smiles) - len(valid_indexes)}")
    LOGGER.info("fitting joint UMAP rows=%s bits=%s", *matrix.shape)
    coordinates = umap.UMAP(
        n_components=2,
        metric="cosine",
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        random_state=args.random_state,
        low_memory=True,
        n_jobs=1,
    ).fit_transform(matrix)
    chembl_meta = pd.DataFrame(
        {
            "dataset": "ChEMBL",
            "method": "ChEMBL36",
            "epoch": np.nan,
            "cohort": "reference",
            "smiles": chembl_smiles,
            "source_file": str(args.chembl_csv),
            "source_row": np.arange(len(chembl)),
        }
    )
    rl_meta = valid_rl[["method", "epoch", "cohort", "smiles", "source_file", "source_row"]].copy()
    rl_meta.insert(0, "dataset", "RL")
    embedding = pd.concat([chembl_meta, rl_meta], ignore_index=True)
    embedding["umap1"] = coordinates[:, 0]
    embedding["umap2"] = coordinates[:, 1]
    embedding.to_csv(args.output_dir / "joint_umap_embeddings.csv.gz", index=False, compression="gzip")
    validity, cohort, epoch = _summaries(embedding, attempts)
    validity.to_csv(args.output_dir / "epoch_validity_summary.csv", index=False)
    cohort.to_csv(args.output_dir / "cohort_umap_summary.csv", index=False)
    epoch.to_csv(args.output_dir / "epoch_centroids.csv", index=False)
    multi_methods = list(LEGACY_NAMES.values()) + ["anti-carbon guard b64"]
    single_methods = [SINGLE_NAMES[item] for item in SINGLE_REWARDS]
    _plot_family(
        embedding, epoch, multi_methods,
        "DiffSBDD multi-objective and guard RL drift",
        args.output_dir / "diffsbdd_multiobjective_vs_frozen_chembl_umap",
    )
    _plot_family(
        embedding, epoch, single_methods,
        "DiffSBDD single-reward RL drift",
        args.output_dir / "diffsbdd_single_reward_vs_frozen_chembl_umap",
    )
    manifest = {
        "chembl_frozen_csv": str(args.chembl_csv),
        "chembl_rows": len(chembl),
        "chembl_ordered_smiles_sha256": actual_hash,
        "rl_attempts": len(attempts),
        "rl_sanitize_valid": len(valid_rl),
        "methods": METHOD_ORDER,
        "legacy_source": str(args.legacy_csv),
        "anti_carbon_guard_source": str(GUARD_RUN),
        "single_reward_source_glob": SINGLE_GLOB,
        "fingerprint": {"type": "Morgan/ECFP4", "radius": args.radius, "nbits": args.nbits},
        "umap": {
            "joint_fit": True,
            "metric": "cosine",
            "n_neighbors": args.n_neighbors,
            "min_dist": args.min_dist,
            "random_state": args.random_state,
        },
        "duplicates_retained": True,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _write_report(args.output_dir, attempts, validity, cohort)
    LOGGER.info("completed output=%s", args.output_dir)


if __name__ == "__main__":
    main()
