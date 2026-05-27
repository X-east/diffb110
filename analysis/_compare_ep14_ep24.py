#!/usr/bin/env python3
"""Compare epoch 14 vs epoch 24: actual molecular properties."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen

RUN_DIR = Path("/home/md666/data/disk6/DiffB110/OUT/7gpu-classifier-only_20260526_210715")
OUT_DIR = RUN_DIR / "analysis_figures"

def load_valid_smiles(epoch):
    df = pd.read_csv(RUN_DIR / f"epochs/epoch_{epoch:04d}/molecules.csv")
    valid = df[df["rdkit_valid"] == 1]
    smiles_list = valid["smiles"].tolist()
    high_reward = valid[valid["chembl_classifier_reward"] > 0.5]
    low_reward = valid[valid["chembl_classifier_reward"] < 0]
    return smiles_list, high_reward["smiles"].tolist(), low_reward["smiles"].tolist()

def compute_props(smiles_list):
    mols = []
    logps, molwts, tpsas, hba, hbd, rot = [], [], [], [], [], []
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        mols.append(m)
        logps.append(Crippen.MolLogP(m))
        molwts.append(Descriptors.MolWt(m))
        tpsas.append(Descriptors.TPSA(m))
        hba.append(Descriptors.NumHAcceptors(m))
        hbd.append(Descriptors.NumHDonors(m))
        rot.append(Descriptors.NumRotatableBonds(m))
    return mols, logps, molwts, tpsas, hba, hbd, rot

# ── Load ──
print("Loading molecules...")
ep14_all, ep14_high, ep14_low = load_valid_smiles(14)
ep24_all, ep24_high, ep24_low = load_valid_smiles(24)

print(f"Epoch 14: {len(ep14_all)} valid, {len(ep14_high)} high-reward, {len(ep14_low)} low-reward")
print(f"Epoch 24: {len(ep24_all)} valid, {len(ep24_high)} high-reward, {len(ep24_low)} low-reward")

ep14_props = compute_props(ep14_all)
ep24_props = compute_props(ep24_all)

logp14, mw14, tpsa14, hba14, hbd14, rot14 = ep14_props[1:]
logp24, mw24, tpsa24, hba24, hbd24, rot24 = ep24_props[1:]

# ── Plot ──
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Molecule Property Distribution: Epoch 14 vs Epoch 24\n(computed on valid molecules only)", fontsize=13, fontweight="bold")

bins = 30
alpha = 0.6

pairs = [
    (axes[0, 0], logp14, logp24, "LogP", "#2980b9", "#e74c3c"),
    (axes[0, 1], mw14, mw24, "Molecular Weight", "#27ae60", "#e67e22"),
    (axes[0, 2], tpsa14, tpsa24, "TPSA", "#8e44ad", "#f39c12"),
    (axes[1, 0], hba14, hba24, "H-Bond Acceptors", "#16a085", "#c0392b"),
    (axes[1, 1], hbd14, hbd24, "H-Bond Donors", "#2980b9", "#d35400"),
    (axes[1, 2], rot14, rot24, "Rotatable Bonds", "#2c3e50", "#e74c3c"),
]

for ax, d14, d24, title, c14, c24 in pairs:
    ax.hist(d14, bins=bins, alpha=alpha, color=c14, label=f"Ep14 (n={len(d14)})", density=True, edgecolor="black", linewidth=0.3)
    ax.hist(d24, bins=bins, alpha=alpha, color=c24, label=f"Ep24 (n={len(d24)})", density=True, edgecolor="black", linewidth=0.3)
    ax.set_title(title)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25)

plt.tight_layout()
plt.savefig(OUT_DIR / "15_ep14_vs_ep24_histograms.png", dpi=150)
plt.close()
print("  [15] ep14_vs_ep24_histograms")

# ── Summary stats ──
print(f"\n{'='*60}")
print("Property comparison (mean ± std)")
print(f"{'='*60}")
for name, d14, d24 in [
    ("LogP", logp14, logp24),
    ("MW", mw14, mw24),
    ("TPSA", tpsa14, tpsa24),
    ("HBA", hba14, hba24),
    ("HBD", hbd14, hbd24),
    ("RotBonds", rot14, rot24),
]:
    print(f"  {name:>10s}  Ep14: {np.mean(d14):7.2f} ± {np.std(d14):.2f}  |  Ep24: {np.mean(d24):7.2f} ± {np.std(d24):.2f}")

# ── Show example high-reward molecules from each ──
print(f"\nEp14 high-reward examples (first 10):")
for s in ep14_high[:10]:
    m = Chem.MolFromSmiles(s)
    if m:
        print(f"  LogP={Crippen.MolLogP(m):.1f}  TPSA={Descriptors.TPSA(m):.0f}  MW={Descriptors.MolWt(m):.0f}  {s}")

print(f"\nEp24 high-reward examples (first 10):")
for s in ep24_high[:10]:
    m = Chem.MolFromSmiles(s)
    if m:
        print(f"  LogP={Crippen.MolLogP(m):.1f}  TPSA={Descriptors.TPSA(m):.0f}  MW={Descriptors.MolWt(m):.0f}  {s}")

# ── High reward LogP distribution shift ──
fig, ax = plt.subplots(figsize=(8, 5))
hp14 = [Crippen.MolLogP(Chem.MolFromSmiles(s)) for s in ep14_high if Chem.MolFromSmiles(s)]
hp24 = [Crippen.MolLogP(Chem.MolFromSmiles(s)) for s in ep24_high if Chem.MolFromSmiles(s)]
ax.hist(hp14, bins=15, alpha=0.6, color="#2ecc71", label=f"Ep14 high-reward (n={len(hp14)})", density=True, edgecolor="black", lw=0.3)
ax.hist(hp24, bins=15, alpha=0.6, color="#f39c12", label=f"Ep24 high-reward (n={len(hp24)})", density=True, edgecolor="black", lw=0.3)
ax.axvline(np.mean(hp14), color="#2ecc71", ls="--", lw=1.5, label=f"Ep14 mean={np.mean(hp14):.2f}")
ax.axvline(np.mean(hp24), color="#f39c12", ls="--", lw=1.5, label=f"Ep24 mean={np.mean(hp24):.2f}")
ax.set_xlabel("LogP")
ax.set_title("LogP Distribution: High-Reward Molecules (chembl_classifier > 0.5)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(OUT_DIR / "16_highreward_logp_shift.png", dpi=150)
plt.close()
print("  [16] highreward_logp_shift")

print(f"\nDone. 2 plots → {OUT_DIR}/")
