#!/usr/bin/env python3
"""用现成分子跑 LogP 分布奖励：baseline 5827 + Experiment #6 Epoch20."""
from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, SDMolSupplier

# ── 8-bin scheme (same as LogPDistributionReward) ──
BIN_EDGES = np.array([-np.inf, 0.5, 1.5, 2.5, 3.0, 3.5, 4.5, 5.5, np.inf])
N_BINS = 8
# ChEMBL reference proportions (from logp_binning_analysis.py)
REF = np.array([0.044, 0.093, 0.206, 0.140, 0.143, 0.227, 0.108, 0.039])
SCALE = 12.0
MIN_REWARD = -2.0
MAX_REWARD = 2.0

BIN_LABELS = [
    "(-∞,0.5)", "[0.5,1.5)", "[1.5,2.5)", "[2.5,3.0)",
    "[3.0,3.5)", "[3.5,4.5)", "[4.5,5.5)", "(5.5,+∞)",
]


def load_smiles_from_sdf(sdf_path: str) -> list[str | None]:
    """Extract SMILES from SDF file via RDKit."""
    suppl = Chem.SDMolSupplier(sdf_path)
    smiles = []
    for mol in suppl:
        if mol is None:
            smiles.append(None)
        else:
            try:
                smiles.append(Chem.MolToSmiles(mol))
            except Exception:
                smiles.append(None)
    return smiles


def load_smiles_from_csv(csv_path: str) -> list[str | None]:
    """Extract SMILES from molecules.csv (filters rdkit_valid=1)."""
    smiles = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            valid = row.get("rdkit_valid", "0")
            smi = row.get("smiles", "").strip()
            if valid == "1" and smi and smi != "None":
                smiles.append(smi)
            else:
                smiles.append(None)
    return smiles


def compute_logp_bins(smiles_list: list[str | None]) -> tuple[np.ndarray, list[float]]:
    """Compute LogP and bin counts for a list of SMILES."""
    counts = np.zeros(N_BINS, dtype=np.float64)
    logp_vals = []
    for smi in smiles_list:
        if smi is None:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            logp = Descriptors.MolLogP(mol)
            logp_vals.append(logp)
            idx = int(np.digitize(logp, BIN_EDGES)) - 1
            if 0 <= idx < N_BINS:
                counts[idx] += 1
        except Exception:
            pass
    return counts, logp_vals


def analyze_dataset(name: str, smiles_list: list[str | None]):
    """Analyze one dataset: print proportions, gaps, typical rewards."""
    counts, logp_vals = compute_logp_bins(smiles_list)
    n_valid = int(counts.sum())

    # Laplace smoothing (consistent with reward code)
    batch_props = (counts + 1) / (n_valid + N_BINS)

    print(f"\n{'=' * 80}")
    print(f"  {name}")
    print(f"  Total SMILES: {len(smiles_list)}, Valid: {n_valid}")
    print(f"  LogP range: [{min(logp_vals):.2f}, {max(logp_vals):.2f}]")
    print(f"  LogP mean={np.mean(logp_vals):.2f}, std={np.std(logp_vals):.2f}")
    print(f"{'=' * 80}")

    print(f"  {'Bin':<13} {'Ref%':>6} {'Count':>7} {'Batch%':>7} {'Laplace%':>8} {'Gap':>8} {'Reward':>8}")
    print(f"  {'-' * 65}")

    for i in range(N_BINS):
        raw_pct = counts[i] / n_valid * 100 if n_valid > 0 else 0
        lap_pct = batch_props[i] * 100
        gap = REF[i] - batch_props[i]
        r = np.clip(SCALE * gap, MIN_REWARD, MAX_REWARD)
        marker = " ***" if abs(r) > 1.0 else ""
        print(f"  {BIN_LABELS[i]:<13} {REF[i]*100:>5.1f}% {int(counts[i]):>7} {raw_pct:>6.1f}% {lap_pct:>7.1f}% {gap:>+7.4f} {r:>+7.3f}{marker}")

    # Summary stats
    print(f"\n  Reward range: [{MIN_REWARD}, {MAX_REWARD}]")
    print(f"  Typical |gap|: {np.mean(np.abs(REF - batch_props)):.4f} "
          f"→ reward magnitude ~ {SCALE * np.mean(np.abs(REF - batch_props)):.3f}")
    print(f"  Max |gap|: {np.max(np.abs(REF - batch_props)):.4f} "
          f"→ max reward magnitude ~ {SCALE * np.max(np.abs(REF - batch_props)):.3f}")

    # Per-molecule reward distribution summary
    per_mol_rewards = []
    for logp in logp_vals:
        idx = int(np.digitize(logp, BIN_EDGES)) - 1
        if 0 <= idx < N_BINS:
            gap = REF[idx] - batch_props[idx]
            r = np.clip(SCALE * gap, MIN_REWARD, MAX_REWARD)
            per_mol_rewards.append(r)

    if per_mol_rewards:
        arr = np.array(per_mol_rewards)
        print(f"\n  分子级 reward 分布:")
        print(f"    mean={arr.mean():+.3f}, std={arr.std():.3f}")
        print(f"    min={arr.min():+.3f}, max={arr.max():+.3f}")
        print(f"    P5={np.percentile(arr, 5):+.3f}, "
              f"P50={np.percentile(arr, 50):+.3f}, "
              f"P95={np.percentile(arr, 95):+.3f}")
        pos_pct = (arr > 0).sum() / len(arr) * 100
        neg_pct = (arr < 0).sum() / len(arr) * 100
        print(f"    正奖励: {pos_pct:.1f}%, 负奖励: {neg_pct:.1f}%, "
              f"零: {(arr == 0).sum() / len(arr) * 100:.1f}%")


# ── Main ──
if __name__ == "__main__":
    # Dataset 1: Baseline 5827 (SDF)
    sdf_path = "/home/md666/data/disk6/DiffB110/runs/260407_generate_ligands_10000/260407generated_10000_filtered.sdf"
    if os.path.exists(sdf_path):
        print("Loading SDF...")
        sdf_smiles = load_smiles_from_sdf(sdf_path)
        valid_sdf = [s for s in sdf_smiles if s is not None]
        print(f"  SDF: {len(sdf_smiles)} total, {len(valid_sdf)} valid")
        analyze_dataset("Baseline (SDF, 5827 molecules)", sdf_smiles)
    else:
        print(f"SDF not found: {sdf_path}")

    # Dataset 2: Experiment #6 Epoch20 (CSV)
    csv_path = "/home/md666/data/disk6/DiffB110/runs/6gpu-chembl-only_20260521_154957/epochs/epoch_0020/molecules.csv"
    if os.path.exists(csv_path):
        print("\nLoading CSV...")
        csv_smiles = load_smiles_from_csv(csv_path)
        valid_csv = [s for s in csv_smiles if s is not None]
        print(f"  CSV: {len(csv_smiles)} total, {len(valid_csv)} valid")
        analyze_dataset("Experiment #6 Epoch20 (CSV)", csv_smiles)
    else:
        print(f"CSV not found: {csv_path}")
