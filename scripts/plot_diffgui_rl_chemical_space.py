#!/usr/bin/env python3
"""DiffGui chemical-space shift: pretrained vs RL-trained, vs ChEMBL36.

Method (same as plot_chemical_space_umap.py, grounded in literature):
  * ECFP4 (Morgan radius=2 / 2048-bit, binarized)
  * UMAP 2D, metric="cosine" (fast surrogate for jaccard on ~40k pts)
  * One joint UMAP over (ChEMBL 30k sample + DiffGui-pretrained + DiffGui-RL)
    so the three groups share one coordinate system.

Pre-RL DiffGui : benchmark5 formal 3rep, model==DiffGui, sanitize_success==1
                 (generated 2026-07-01, before the DePPA RL runs)
Post-RL DiffGui: final epochs of the completed 100-ep DePPA run
                 diffgui-deppa-pbfast-cpu4-lr1e5-wd1e4-b8x7-t200-100ep_20260715_2018
                 (epoch_0100 is the last; we aggregate the last `--rl-epochs`
                 epochs as the final-policy distribution). Pocket = 8GJL_C1101.
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
LOGGER = logging.getLogger("diffgui_rl_space")

ROOT = Path(__file__).resolve().parents[1]
CHEMBL = ROOT / "data/chembl_standardized_all/chembl_36_standardized_all.csv.gz"
PRE_CSV = (
    ROOT
    / "analysis/benchmark/benchmark5/8gjl_c1101_benchmark_20260706_1000steps_3rep_formal/raw_quality_vs_chembl_20260728/per_molecule_raw_quality.csv.gz"
)
RL_RUN = (
    ROOT
    / "runs/diffgui-deppa-pbfast-cpu4-lr1e5-wd1e4-b8x7-t200-100ep_20260715_2018"
)
OUT = RL_RUN / "chemical_space_vs_pretrained"

PRE_COLOR = "#3498DB"   # DiffGui pretrained (blue, matches project palette)
RL_COLOR = "#E67E22"    # DiffGui RL (orange)
CHEMBL_COLOR = "#BFBFBF"


def smiles_to_morgan(task: tuple) -> tuple[str, str, np.ndarray | None]:
    """Compute a binarized ECFP4 fingerprint for one (label, smiles)."""
    label, smiles, radius, nbits = task
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return label, smiles, None
        bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
        arr = np.frombuffer(bv.ToBitString().encode(), dtype=np.uint8) - ord("0")
        return label, smiles, arr.astype(np.float32)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("fingerprint failed for %s: %s", smiles, exc)
        return label, smiles, None


def extract_rl_smiles(run_dir: Path, k: int) -> list[str]:
    """Parse canonical SMILES from the last k epoch molecules.sdf files."""
    ep_dirs = sorted(
        run_dir.glob("epochs/epoch_*"), key=lambda p: int(p.name.split("_")[1])
    )
    out: list[str] = []
    for e in ep_dirs[-k:]:
        sdf = e / "molecules.sdf"
        if not sdf.exists():
            continue
        sup = Chem.SDMolSupplier(str(sdf), sanitize=True, removeHs=True)
        for mol in sup:
            if mol is None:
                continue
            try:
                smi = Chem.MolToSmiles(mol)
            except Exception:  # noqa: BLE001
                continue
            if smi:
                out.append(smi)
    LOGGER.info("RL run %s: parsed %d SMILES from last %d epochs", run_dir.name, len(out), k)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chembl-csv", default=str(CHEMBL))
    parser.add_argument("--pre-csv", default=str(PRE_CSV))
    parser.add_argument("--rl-run", default=str(RL_RUN))
    parser.add_argument("--out-dir", default=str(OUT))
    parser.add_argument("--chembl-sample", type=int, default=30000)
    parser.add_argument("--rl-epochs", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-workers", type=int, default=16)
    parser.add_argument("--nbits", type=int, default=2048)
    parser.add_argument("--radius", type=int, default=2)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("loading ChEMBL reference")
    chembl = pd.read_csv(args.chembl_csv, compression="gzip", usecols=["canonical_smiles"])
    chembl = chembl.sample(min(args.chembl_sample, len(chembl)), random_state=args.random_state)
    chembl_smiles = chembl["canonical_smiles"].tolist()
    LOGGER.info("ChEMBL sampled: %d", len(chembl_smiles))

    LOGGER.info("loading pretrained DiffGui (benchmark5)")
    pre = pd.read_csv(
        args.pre_csv, compression="gzip",
        usecols=["model", "sanitize_success", "canonical_smiles"],
    )
    pre = pre[(pre["model"] == "DiffGui") & (pre["sanitize_success"] == 1)].reset_index(drop=True)
    pre_smiles = pre["canonical_smiles"].tolist()
    LOGGER.info("pretrained DiffGui sanitize-valid: %d", len(pre_smiles))

    LOGGER.info("extracting RL-trained DiffGui (last %d epochs)", args.rl_epochs)
    rl_smiles = extract_rl_smiles(Path(args.rl_run), args.rl_epochs)
    rl_smiles = list(dict.fromkeys(rl_smiles))  # dedupe
    LOGGER.info("RL DiffGui unique sanitize-valid: %d", len(rl_smiles))

    labels = ["ChEMBL"] * len(chembl_smiles) + ["DiffGui_pre"] * len(pre_smiles) + ["DiffGui_RL"] * len(rl_smiles)
    smiles = chembl_smiles + pre_smiles + rl_smiles
    tasks = [(lab, s, args.radius, args.nbits) for lab, s in zip(labels, smiles)]
    LOGGER.info("computing ECFP4 for %d molecules (%d workers)", len(smiles), args.n_workers)
    with Pool(args.n_workers) as pool:
        results = pool.map(smiles_to_morgan, tasks)
    rows = []
    for lab, smi, fp in results:
        if fp is not None:
            rows.append((lab, smi, fp))
    X = np.vstack([r[2] for r in rows])
    LOGGER.info("valid fingerprints: %d (dropped %d)", X.shape[0], len(results) - X.shape[0])

    LOGGER.info("fitting joint UMAP %d x %d", X.shape[0], X.shape[1])
    reducer = umap.UMAP(
        n_components=2, metric="cosine", n_neighbors=20, min_dist=0.1,
        random_state=args.random_state,
    )
    coords = reducer.fit_transform(X)
    df = pd.DataFrame(
        {"label": [r[0] for r in rows], "umap1": coords[:, 0], "umap2": coords[:, 1], "smiles": [r[1] for r in rows]}
    )
    df.to_csv(out_dir / "umap_embeddings_pre_vs_rl.csv", index=False)

    def centroid(label: str) -> np.ndarray:
        return df[df["label"] == label][["umap1", "umap2"]].mean().to_numpy()

    c_chem = centroid("ChEMBL")
    c_pre = centroid("DiffGui_pre")
    c_rl = centroid("DiffGui_RL")
    summary = pd.DataFrame(
        [
            {"group": "ChEMBL", "n": int((df.label == "ChEMBL").sum()), "centroid_umap1": c_chem[0], "centroid_umap2": c_chem[1], "dist_to_chembl": 0.0},
            {"group": "DiffGui_pretrained", "n": int((df.label == "DiffGui_pre").sum()), "centroid_umap1": c_pre[0], "centroid_umap2": c_pre[1], "dist_to_chembl": float(np.linalg.norm(c_pre - c_chem))},
            {"group": "DiffGui_RL", "n": int((df.label == "DiffGui_RL").sum()), "centroid_umap1": c_rl[0], "centroid_umap2": c_rl[1], "dist_to_chembl": float(np.linalg.norm(c_rl - c_chem))},
        ]
    )
    summary.to_csv(out_dir / "umap_group_summary.csv", index=False)
    # shift pre -> RL
    shift = float(np.linalg.norm(c_rl - c_pre))
    LOGGER.info("centroid shift pretrained -> RL: %.4f (UMAP units)", shift)
    LOGGER.info("\n%s", summary.to_string(index=False))

    pre = df[df["label"] == "DiffGui_pre"]
    rl = df[df["label"] == "DiffGui_RL"]
    chem = df[df["label"] == "ChEMBL"]

    # Plot 1: three groups together
    fig, ax = plt.subplots(figsize=(6.6, 5.4), dpi=300)
    ax.scatter(chem["umap1"], chem["umap2"], s=3, c=CHEMBL_COLOR, alpha=0.18, linewidths=0, label="ChEMBL36")
    ax.scatter(pre["umap1"], pre["umap2"], s=10, c=PRE_COLOR, alpha=0.5, linewidths=0, label="DiffGui pretrained")
    ax.scatter(rl["umap1"], rl["umap2"], s=14, c=RL_COLOR, alpha=0.7, linewidths=0, marker="^", label="DiffGui RL (final)")
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
    ax.set_title("DiffGui chemical space: pretrained vs RL vs ChEMBL (ECFP4, UMAP)")
    ax.legend(loc="best", frameon=False)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"diffgui_pretrained_vs_rl_vs_chembl_umap.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Plot 2: focused pretrained vs RL overlay (no ChEMBL)
    fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=300)
    ax.scatter(pre["umap1"], pre["umap2"], s=12, c=PRE_COLOR, alpha=0.5, linewidths=0, label=f"pretrained (n={len(pre)})")
    ax.scatter(rl["umap1"], rl["umap2"], s=16, c=RL_COLOR, alpha=0.75, linewidths=0, marker="^", label=f"RL final (n={len(rl)})")
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
    ax.set_title("DiffGui: pretrained vs RL-trained distribution (ECFP4, UMAP)")
    ax.legend(loc="best", frameon=False)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"diffgui_pretrained_vs_rl_umap.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    LOGGER.info("done; outputs in %s", out_dir)


if __name__ == "__main__":
    main()
