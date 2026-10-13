#!/usr/bin/env python3
"""UMAP chemical-space plots: each generation model vs the ChEMBL36 reference.

Method (grounded in literature):
  * Fingerprint: Morgan/ECFP4 (2048-bit, radius=2), binarized. This is the
    standard representation for chemical-space visualization in de novo
    molecular-generation evaluation (ChemPlot, Felton et al., J Cheminform
    2021; GuacaMol-style benchmarks compare generated molecules against the
    ChEMBL chemical space).
  * Reduction: UMAP -> 2D. UMAP is preferred over t-SNE for chemical-space
    navigation (Wagen 2022; 2025 review "Chemical space visual navigation in
    the era of deep learning").
  * Distance: metric="cosine". Literature often uses "jaccard" on *binary*
    RDKFingerprints, but for ~40k points the exact jaccard nearest-neighbour
    graph is O(N^2) and infeasible; cosine on binarized Morgan is the standard
    fast surrogate and preserves the same neighbourhood structure.

One joint UMAP is fit on (ChEMBL sample + all 5 models) so the five plots share
a single coordinate system and model positions are directly comparable.
"""

from __future__ import annotations

import argparse
import logging
import os
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/diffb110_matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
import umap

RDLogger.DisableLog("rdApp.error")
LOGGER = logging.getLogger("chem_space_umap")

ROOT = Path(__file__).resolve().parents[1]
MODEL_ORDER = ["DiffGui", "DiffSBDD", "PMDM", "GCDM-SBDD", "TargetDiff"]
MODEL_COLORS = {
    "DiffGui": "#3498DB",
    "DiffSBDD": "#E74C3C",
    "PMDM": "#F39C12",
    "GCDM-SBDD": "#9B59B6",
    "TargetDiff": "#2ECC71",
}
CHEMBL_COLOR = "#BFBFBF"
DEFAULT_CHEMBL = ROOT / "data/chembl_standardized_all/chembl_36_standardized_all.csv.gz"
DEFAULT_GEN = (
    ROOT
    / "analysis/benchmark/benchmark5/8gjl_c1101_benchmark_20260706_1000steps_3rep_formal/raw_quality_vs_chembl_20260728/per_molecule_raw_quality.csv.gz"
)
DEFAULT_OUT = (
    ROOT
    / "analysis/benchmark/benchmark5/8gjl_c1101_benchmark_20260706_1000steps_3rep_formal/raw_quality_vs_chembl_20260728"
)


def smiles_to_morgan(task: tuple) -> tuple[str, np.ndarray | None]:
    """Compute a binarized Morgan fingerprint for one SMILES string."""
    smiles, radius, nbits = task
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles, None
        bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
        arr = np.frombuffer(bv.ToBitString().encode(), dtype=np.uint8) - ord("0")
        return smiles, arr.astype(np.float32)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("fingerprint failed for %s: %s", smiles, exc)
        return smiles, None


def main() -> None:
    """Build Morgan fingerprints, fit a joint UMAP, and emit one plot per model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chembl-csv", default=str(DEFAULT_CHEMBL))
    parser.add_argument("--gen-csv", default=str(DEFAULT_GEN))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--chembl-sample", type=int, default=30000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-workers", type=int, default=18)
    parser.add_argument("--nbits", type=int, default=2048)
    parser.add_argument("--radius", type=int, default=2)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("loading ChEMBL reference: %s", args.chembl_csv)
    chembl = pd.read_csv(args.chembl_csv, compression="gzip", usecols=["canonical_smiles"])
    chembl = chembl.sample(min(args.chembl_sample, len(chembl)), random_state=args.random_state)
    chembl_smiles = chembl["canonical_smiles"].tolist()
    LOGGER.info("ChEMBL sampled: %d", len(chembl_smiles))

    LOGGER.info("loading generated molecules: %s", args.gen_csv)
    gen = pd.read_csv(
        args.gen_csv,
        compression="gzip",
        usecols=["model", "sanitize_success", "canonical_smiles"],
    )
    gen = gen[gen["sanitize_success"] == 1].reset_index(drop=True)
    LOGGER.info("generated sanitize-valid: %d", len(gen))
    gen_smiles = gen["canonical_smiles"].tolist()
    gen_model = gen["model"].tolist()

    all_smiles = chembl_smiles + gen_smiles
    all_labels = ["ChEMBL"] * len(chembl_smiles) + gen_model
    LOGGER.info("computing Morgan fingerprints for %d molecules (%d workers)", len(all_smiles), args.n_workers)
    tasks = [(s, args.radius, args.nbits) for s in all_smiles]
    with Pool(args.n_workers) as pool:
        results = pool.map(smiles_to_morgan, tasks)

    valid_smiles: list[str] = []
    valid_labels: list[str] = []
    fps: list[np.ndarray] = []
    for i, (smi, fp) in enumerate(results):
        if fp is not None:
            valid_smiles.append(smi)
            valid_labels.append(all_labels[i])
            fps.append(fp)
    X = np.vstack(fps)
    LOGGER.info("valid fingerprints: %d (dropped %d)", X.shape[0], len(results) - X.shape[0])

    LOGGER.info("fitting UMAP on %d x %d", X.shape[0], X.shape[1])
    reducer = umap.UMAP(
        n_components=2,
        metric="cosine",
        n_neighbors=20,
        min_dist=0.1,
        random_state=args.random_state,
    )
    coords = reducer.fit_transform(X)
    df = pd.DataFrame(
        {"label": valid_labels, "umap1": coords[:, 0], "umap2": coords[:, 1], "smiles": valid_smiles}
    )
    emb_path = out_dir / "umap_embeddings.csv"
    df.to_csv(emb_path, index=False)
    LOGGER.info("saved embeddings: %s", emb_path)

    chem_cent = df[df["label"] == "ChEMBL"][["umap1", "umap2"]].mean().to_numpy()
    rows = []
    for model in MODEL_ORDER:
        m = df[df["label"] == model][["umap1", "umap2"]]
        if len(m) == 0:
            continue
        mc = m.mean().to_numpy()
        rows.append(
            {
                "model": model,
                "n": len(m),
                "centroid_umap1": mc[0],
                "centroid_umap2": mc[1],
                "centroid_dist_to_chembl": float(np.linalg.norm(mc - chem_cent)),
            }
        )
    cd = pd.DataFrame(rows)
    cd.to_csv(out_dir / "umap_centroid_distances.csv", index=False)
    LOGGER.info("centroid distances:\n%s", cd.to_string(index=False))

    chem = df[df["label"] == "ChEMBL"]
    for model in MODEL_ORDER:
        m = df[df["label"] == model]
        if len(m) == 0:
            LOGGER.warning("no molecules for %s, skip", model)
            continue
        fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=300)
        ax.scatter(chem["umap1"], chem["umap2"], s=3, c=CHEMBL_COLOR, alpha=0.18, linewidths=0, label="ChEMBL36")
        ax.scatter(m["umap1"], m["umap2"], s=11, c=MODEL_COLORS[model], alpha=0.55, linewidths=0, label=model)
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.set_title(f"{model} vs ChEMBL36 (ECFP4, UMAP)")
        ax.legend(loc="best", frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for ext in ("png", "svg"):
            path = out_dir / f"chemspace_{model}_vs_chembl_umap.{ext}"
            fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        LOGGER.info("saved plot for %s", model)

    LOGGER.info("done")


if __name__ == "__main__":
    main()
