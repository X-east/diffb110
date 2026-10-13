#!/usr/bin/env python3
"""Diversity proxy for DiffGui pretrained vs RL: mean pairwise ECFP4 Tanimoto
and median nearest-neighbour Tanimoto (higher = more collapsed / less diverse)."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.error")
LOGGER = logging.getLogger("rl_diversity")
ROOT = Path(__file__).resolve().parents[1]
PRE_CSV = ROOT / "analysis/benchmark/benchmark5/8gjl_c1101_benchmark_20260706_1000steps_3rep_formal/raw_quality_vs_chembl_20260728/per_molecule_raw_quality.csv.gz"
RL_RUN = ROOT / "runs/diffgui-deppa-pbfast-cpu4-lr1e5-wd1e4-b8x7-t200-100ep_20260715_2018"


def fps_from_smiles(smiles: list[str], radius=2, nbits=2048) -> np.ndarray:
    out = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
        arr = np.frombuffer(bv.ToBitString().encode(), dtype=np.uint8) - ord("0")
        out.append(arr.astype(np.float32))
    return np.vstack(out)


def diversity_metrics(X: np.ndarray) -> dict:
    n = X.shape[0]
    sim = (X @ X.T)  # shared-bit counts (binary)
    # Tanimoto = c / (2*nbits - c)
    nbits = X.shape[1]
    denom = 2 * nbits - sim
    tan = np.triu(sim / np.where(denom == 0, 1, denom), k=1)
    iu = np.triu_indices(n, k=1)
    mean_pair = float(tan[iu].mean())
    # nearest-neighbour Tanimoto (exclude self = 1.0)
    np.fill_diagonal(sim, -1)
    nn = sim.max(axis=1)
    nn_tan = nn / np.where(2 * nbits - nn == 0, 1, 2 * nbits - nn)
    return {"n": n, "mean_pairwise_tanimoto": mean_pair, "median_nn_tanimoto": float(np.median(nn_tan))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre-csv", default=str(PRE_CSV))
    ap.add_argument("--rl-run", default=str(RL_RUN))
    ap.add_argument("--rl-epochs", type=int, default=20)
    ap.add_argument("--out", default=str(RL_RUN / "chemical_space_vs_pretrained" / "diversity_compare.csv"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    pre = pd.read_csv(args.pre_csv, compression="gzip", usecols=["model", "sanitize_success", "canonical_smiles"])
    pre = pre[(pre.model == "DiffGui") & (pre.sanitize_success == 1)]
    pre_smiles = pre["canonical_smiles"].tolist()

    ep_dirs = sorted(RL_RUN.glob("epochs/epoch_*"), key=lambda p: int(p.name.split("_")[1]))
    rl_smiles: list[str] = []
    for e in ep_dirs[-args.rl_epochs:]:
        sdf = e / "molecules.sdf"
        if not sdf.exists():
            continue
        for mol in Chem.SDMolSupplier(str(sdf), sanitize=True, removeHs=True):
            if mol is None:
                continue
            smi = Chem.MolToSmiles(mol)
            if smi:
                rl_smiles.append(smi)
    rl_smiles = list(dict.fromkeys(rl_smiles))

    X_pre = fps_from_smiles(pre_smiles)
    X_rl = fps_from_smiles(rl_smiles)
    LOGGER.info("pretrained fps: %d, RL fps: %d", X_pre.shape[0], X_rl.shape[0])

    m_pre = diversity_metrics(X_pre)
    m_rl = diversity_metrics(X_rl)
    df = pd.DataFrame([{"group": "DiffGui_pretrained", **m_pre}, {"group": "DiffGui_RL", **m_rl}])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    LOGGER.info("\n%s", df.to_string(index=False))


if __name__ == "__main__":
    main()
