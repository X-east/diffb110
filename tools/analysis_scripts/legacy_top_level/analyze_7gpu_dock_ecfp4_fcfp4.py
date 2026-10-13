#!/usr/bin/env python3
"""分析 7gpu-dock-ecfp4-fcfp4_20260525_212140 — 聚焦环复杂度、原子组成、化学多样性"""

import json
import os
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

# RDKit imports
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol

# ---------------------------------------------------------------------------
# Paths & output
# ---------------------------------------------------------------------------
ROOT = Path("/home/md666/data/disk6/DiffB110/runs/7gpu-dock-ecfp4-fcfp4_20260525_212140")
CSV_PATH = ROOT / "training_summary.csv"
EPOCH_DIR = ROOT / "epochs"
SAVE_DIR = ROOT / "analysis_figures"
SAVE_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Palette & font sizes
# ---------------------------------------------------------------------------
C_BLUE   = "#4C72B0"
C_ORANGE = "#DD8452"
C_GREEN  = "#55A868"
C_RED    = "#C44E52"
C_PURPLE = "#8172B2"
C_CYAN   = "#64B5CD"
C_BROWN  = "#937860"

FS = 13
FS_TITLE = 15
FS_SUPTITLE = 18
FS_LEGEND = 10
FS_TICK = 11

# ---------------------------------------------------------------------------
# Load training summary
# ---------------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)

def parse_failure_counts(s):
    try:
        return json.loads(s.replace("'", '"'))
    except Exception:
        return {}

fc_df = df["failure_counts"].apply(parse_failure_counts).apply(pd.Series).fillna(0).astype(int)
for col in fc_df.columns:
    df[col] = fc_df[col]

epochs_arr = df["epoch"].values
N_EPOCH = len(epochs_arr)

# ---------------------------------------------------------------------------
# Load molecule data from selected epochs
# ---------------------------------------------------------------------------
SAMPLE_EPOCHS = [1, 10, 20, 30, 40, 50, 62]
mol_data = {}
for ep in SAMPLE_EPOCHS:
    csv_p = EPOCH_DIR / f"epoch_{ep:04d}" / "molecules.csv"
    if csv_p.exists():
        mdf = pd.read_csv(csv_p)
        mdf = mdf[mdf["rdkit_valid"] == 1].copy()
        mol_data[ep] = mdf

# ---------------------------------------------------------------------------
# RDKit analysis helpers
# ---------------------------------------------------------------------------
def analyze_smiles(smiles_str):
    """从 SMILES 计算原子组成、环数、芳环数、Bertz复杂度等"""
    mol = Chem.MolFromSmiles(smiles_str)
    if mol is None:
        return None

    # Atom counts
    atom_counts = Counter(atom.GetSymbol() for atom in mol.GetAtoms())
    total_atoms = sum(atom_counts.values())
    nC = atom_counts.get("C", 0)
    nO = atom_counts.get("O", 0)
    nN = atom_counts.get("N", 0)
    nS = atom_counts.get("S", 0)
    nF = atom_counts.get("F", 0)
    nCl = atom_counts.get("Cl", 0)
    nBr = atom_counts.get("Br", 0)
    nP = atom_counts.get("P", 0)
    n_hetero = total_atoms - nC
    carbon_frac = nC / total_atoms if total_atoms > 0 else 0

    # Ring analysis
    ri = mol.GetRingInfo()
    n_rings = ri.NumRings()
    n_arom_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    n_aliph_rings = n_rings - n_arom_rings

    # Ring sizes
    ring_sizes = []
    for ring_atoms in ri.AtomRings():
        ring_sizes.append(len(ring_atoms))

    # Fused ring detection: rings sharing >=2 atoms
    n_fused_sets = 0
    n_rings_in_fused = 0
    ring_sets = [set(r) for r in ri.AtomRings()]
    visited = [False] * len(ring_sets)
    for i in range(len(ring_sets)):
        if visited[i]:
            continue
        fused_group = {i}
        changed = True
        while changed:
            changed = False
            for j in range(len(ring_sets)):
                if j in fused_group or visited[j]:
                    continue
                for k in fused_group:
                    if len(ring_sets[k] & ring_sets[j]) >= 2:
                        fused_group.add(j)
                        changed = True
                        break
        if len(fused_group) >= 2:
            n_fused_sets += 1
            for idx in fused_group:
                visited[idx] = True
                n_rings_in_fused += 1
        else:
            for idx in fused_group:
                visited[idx] = True

    n_spiro = 0  # spiro rings (share exactly 1 atom)
    for i in range(len(ring_sets)):
        for j in range(i + 1, len(ring_sets)):
            if len(ring_sets[i] & ring_sets[j]) == 1:
                n_spiro += 1

    # Murcko scaffold
    try:
        scaffold = GetScaffoldForMol(mol)
        scaffold_smiles = Chem.MolToSmiles(scaffold) if scaffold else ""
        scf_atoms = scaffold.GetNumAtoms() if scaffold else 0
        scf_rings = rdMolDescriptors.CalcNumRings(scaffold) if scaffold else 0
    except Exception:
        scaffold_smiles = ""
        scf_atoms = 0
        scf_rings = 0

    # Other properties
    bertz = Descriptors.BertzCT(mol)
    num_rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    num_bridgeheads = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    num_spiro_atoms = rdMolDescriptors.CalcNumSpiroAtoms(mol)

    # Largest ring size
    max_ring_size = max(ring_sizes) if ring_sizes else 0
    min_ring_size = min(ring_sizes) if ring_sizes else 0

    # Unique atom types
    unique_atoms = len(set(atom.GetSymbol() for atom in mol.GetAtoms()))

    return {
        "n_atoms": total_atoms,
        "nC": nC, "nO": nO, "nN": nN, "nS": nS,
        "nF": nF, "nCl": nCl, "nBr": nBr, "nP": nP,
        "n_hetero": n_hetero,
        "carbon_frac": carbon_frac,
        "n_rings": n_rings,
        "n_arom_rings": n_arom_rings,
        "n_aliph_rings": n_aliph_rings,
        "n_fused_sets": n_fused_sets,
        "n_rings_in_fused": n_rings_in_fused,
        "n_spiro": n_spiro,
        "max_ring_size": max_ring_size,
        "min_ring_size": min_ring_size,
        "bertz_ct": bertz,
        "n_rot_bonds": num_rot_bonds,
        "n_bridgeheads": num_bridgeheads,
        "n_spiro_atoms": num_spiro_atoms,
        "scf_atoms": scf_atoms,
        "scf_rings": scf_rings,
        "unique_atoms": unique_atoms,
    }


# Apply RDKit analysis to each epoch's valid molecules
print("Running RDKit analysis on sampled epochs...")
epoch_analysis = {}
for ep, mdf in mol_data.items():
    smiles_list = mdf["smiles"].dropna().tolist()
    results = []
    for smi in smiles_list:
        res = analyze_smiles(smi)
        if res:
            results.append(res)
    epoch_analysis[ep] = pd.DataFrame(results)
    print(f"  Epoch {ep}: {len(results)} valid molecules analyzed")

# Compute epoch-level means
epoch_means = {}
for ep, edf in epoch_analysis.items():
    epoch_means[ep] = edf.mean().to_dict()

# ===========================================================================
# FIGURE 1: Training curves (8-panel overview)
# ===========================================================================
fig1, axes = plt.subplots(3, 3, figsize=(22, 16))
fig1.suptitle(f"7GPU dock+ecfp4+fcfp4 Training Overview  (epochs 1–{N_EPOCH})",
              fontsize=FS_SUPTITLE, fontweight="bold", y=1.01)

# 1a: PPO Loss
ax = axes[0, 0]
ax.plot(epochs_arr, df["avg_loss"], color=C_BLUE, linewidth=1.8, marker="o", markersize=3)
ax.set_ylabel("Avg Loss", fontsize=FS)
ax.set_title("(a) PPO Loss", fontsize=FS_TITLE, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=FS_TICK)

# 1b: RDKit Validity
ax = axes[0, 1]
ax.plot(epochs_arr, df["rdkit_valid_ratio"] * 100, color=C_GREEN, linewidth=1.8, marker="s", markersize=3)
ax.set_ylabel("Validity (%)", fontsize=FS)
ax.set_title("(b) RDKit Validity Ratio", fontsize=FS_TITLE, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=FS_TICK)

# 1c: Reward Mean
ax = axes[0, 2]
ax.plot(epochs_arr, df["reward_mean"], color=C_ORANGE, linewidth=1.8, marker="^", markersize=3)
ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
ax.set_ylabel("Reward Mean", fontsize=FS)
ax.set_title("(c) Reward Mean", fontsize=FS_TITLE, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=FS_TICK)

# 1d: Docking Score
ax = axes[1, 0]
ax.plot(epochs_arr, df["avg_docking"], color=C_RED, linewidth=1.8, marker="v", markersize=3)
ax.set_ylabel("Avg Docking Score", fontsize=FS)
ax.set_title("(d) Avg Docking Score (more negative = better)", fontsize=FS_TITLE, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=FS_TICK)

# 1e: ChEMBL ECFP4 Similarity
ax = axes[1, 1]
ax.plot(epochs_arr, df["avg_chembl_prior_ecfp4_similarity"], color=C_PURPLE, linewidth=1.8, marker="D", markersize=3, label="ECFP4")
ax.plot(epochs_arr, df["avg_chembl_prior_fcfp4_similarity"], color=C_CYAN, linewidth=1.8, marker="s", markersize=3, label="FCFP4")
ax.set_ylabel("Avg Tanimoto Similarity", fontsize=FS)
ax.set_title("(e) ChEMBL Prior Similarity", fontsize=FS_TITLE, fontweight="bold")
ax.legend(fontsize=FS_LEGEND)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=FS_TICK)

# 1f: Failure counts stacked
ax = axes[1, 2]
if "success" in df.columns:
    ax.fill_between(epochs_arr, 0, df["success"], label="Success", color=C_GREEN, alpha=0.7)
    y_base = df["success"].values
    if "fragmented_molecule" in df.columns:
        ax.fill_between(epochs_arr, y_base, y_base + df["fragmented_molecule"],
                        label="Fragmented", color=C_ORANGE, alpha=0.7)
        y_base = y_base + df["fragmented_molecule"].values
    if "sanitize_failed" in df.columns:
        ax.fill_between(epochs_arr, y_base, y_base + df["sanitize_failed"],
                        label="Sanitize Failed", color=C_RED, alpha=0.7)
    if "FAILURE_DOCKING_BACKEND" in df.columns:
        y_base_final = y_base + df["sanitize_failed"].values if "sanitize_failed" in df.columns else y_base
        ax.fill_between(epochs_arr, y_base_final, y_base_final + df["FAILURE_DOCKING_BACKEND"],
                        label="Dock Backend Fail", color=C_BROWN, alpha=0.5)
ax.set_ylabel("Molecule Count", fontsize=FS)
ax.set_title("(f) Failure Breakdown", fontsize=FS_TITLE, fontweight="bold")
ax.legend(loc="upper right", fontsize=FS_LEGEND - 1)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=FS_TICK)

# 1g: Carbon fraction trend
ax = axes[2, 0]
ax.plot(epochs_arr, df["carbon_fraction"], color=C_BROWN, linewidth=2.0, marker="o", markersize=3)
ax.axhline(y=0.75, color="gray", linestyle="--", alpha=0.4, label="75% C (ChEMBL-like)")
ax.set_ylabel("Carbon Fraction", fontsize=FS)
ax.set_title("(g) Carbon Fraction (→ collapse signal)", fontsize=FS_TITLE, fontweight="bold")
ax.legend(fontsize=FS_LEGEND)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=FS_TICK)

# 1h: Heteroatom count
ax = axes[2, 1]
ax.plot(epochs_arr, df["hetero_atom_mean"], color=C_GREEN, linewidth=2.0, marker="s", markersize=3)
ax.set_ylabel("Heteroatom Count", fontsize=FS)
ax.set_title("(h) Avg Heteroatom Count", fontsize=FS_TITLE, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=FS_TICK)

# 1i: LogP trend
ax = axes[2, 2]
ax.plot(epochs_arr, df["logp_mean"], color=C_PURPLE, linewidth=1.8, marker="D", markersize=3)
ax.set_ylabel("Avg LogP", fontsize=FS)
ax.set_title("(i) Avg LogP", fontsize=FS_TITLE, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=FS_TICK)

plt.tight_layout()
fig1.savefig(SAVE_DIR / "01_training_overview.png", dpi=150, bbox_inches="tight")
plt.close(fig1)
print("[OK] 01_training_overview.png")

# ===========================================================================
# FIGURE 2: Ring complexity analysis (per-epoch)
# ===========================================================================
fig2, axes = plt.subplots(2, 3, figsize=(20, 13))
fig2.suptitle("Ring Complexity Evolution", fontsize=FS_SUPTITLE, fontweight="bold", y=1.01)

ep_list = sorted(epoch_means.keys())
x_ticks = ep_list

# 2a: Total rings
ax = axes[0, 0]
vals = [epoch_means[ep]["n_rings"] for ep in ep_list]
ax.bar(ep_list, vals, color=C_BLUE, alpha=0.8, edgecolor="white")
ax.plot(ep_list, vals, "o-", color=C_RED, markersize=6, linewidth=2)
ax.set_ylabel("Avg Rings", fontsize=FS)
ax.set_title("(a) Total Ring Count", fontsize=FS_TITLE, fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")
ax.tick_params(labelsize=FS_TICK)

# 2b: Aromatic vs Aliphatic rings
ax = axes[0, 1]
arom_vals = [epoch_means[ep]["n_arom_rings"] for ep in ep_list]
aliph_vals = [epoch_means[ep]["n_aliph_rings"] for ep in ep_list]
x = np.arange(len(ep_list))
w = 0.35
ax.bar(x - w/2, arom_vals, w, label="Aromatic", color=C_PURPLE, alpha=0.8)
ax.bar(x + w/2, aliph_vals, w, label="Aliphatic", color=C_ORANGE, alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(ep_list)
ax.set_ylabel("Avg Count", fontsize=FS)
ax.set_title("(b) Aromatic vs Aliphatic Rings", fontsize=FS_TITLE, fontweight="bold")
ax.legend(fontsize=FS_LEGEND)
ax.grid(True, alpha=0.3, axis="y")
ax.tick_params(labelsize=FS_TICK)

# 2c: Fused ring systems
ax = axes[0, 2]
fused_vals = [epoch_means[ep]["n_fused_sets"] for ep in ep_list]
rings_fused_vals = [epoch_means[ep]["n_rings_in_fused"] for ep in ep_list]
ax.plot(ep_list, fused_vals, "o-", color=C_RED, linewidth=2, markersize=6, label="Fused ring systems")
ax.plot(ep_list, rings_fused_vals, "s--", color=C_BLUE, linewidth=2, markersize=6, label="Rings in fused")
ax.set_ylabel("Count", fontsize=FS)
ax.set_title("(c) Fused Ring Systems", fontsize=FS_TITLE, fontweight="bold")
ax.legend(fontsize=FS_LEGEND)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=FS_TICK)

# 2d: Bridgehead atoms
ax = axes[1, 0]
bh_vals = [epoch_means[ep]["n_bridgeheads"] for ep in ep_list]
ax.bar(ep_list, bh_vals, color=C_BROWN, alpha=0.8, edgecolor="white")
ax.plot(ep_list, bh_vals, "o-", color="black", markersize=6, linewidth=2)
ax.set_ylabel("Avg Bridgeheads", fontsize=FS)
ax.set_title("(d) Bridgehead Atoms (fused ring complexity)", fontsize=FS_TITLE, fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")
ax.tick_params(labelsize=FS_TICK)

# 2e: Spiro atoms
ax = axes[1, 1]
spiro_vals = [epoch_means[ep]["n_spiro_atoms"] for ep in ep_list]
spiro_set_vals = [epoch_means[ep]["n_spiro"] for ep in ep_list]
ax.bar(ep_list, spiro_vals, color=C_CYAN, alpha=0.8, edgecolor="white", label="Spiro atoms")
ax.plot(ep_list, spiro_set_vals, "s--", color=C_RED, linewidth=2, markersize=6, label="Spiro pairs")
ax.set_ylabel("Count", fontsize=FS)
ax.set_title("(e) Spiro Atoms & Pairs", fontsize=FS_TITLE, fontweight="bold")
ax.legend(fontsize=FS_LEGEND)
ax.grid(True, alpha=0.3, axis="y")
ax.tick_params(labelsize=FS_TICK)

# 2f: Max ring size
ax = axes[1, 2]
max_rs_vals = [epoch_means[ep]["max_ring_size"] for ep in ep_list]
min_rs_vals = [epoch_means[ep]["min_ring_size"] for ep in ep_list]
ax.plot(ep_list, max_rs_vals, "o-", color=C_RED, linewidth=2, markersize=6, label="Max ring size")
ax.plot(ep_list, min_rs_vals, "s-", color=C_BLUE, linewidth=2, markersize=6, label="Min ring size")
ax.axhline(y=6, color="gray", linestyle="--", alpha=0.4, label="6-membered")
ax.axhline(y=5, color="orange", linestyle="--", alpha=0.4, label="5-membered")
ax.set_ylabel("Ring Size", fontsize=FS)
ax.set_title("(f) Ring Size Range", fontsize=FS_TITLE, fontweight="bold")
ax.legend(fontsize=FS_LEGEND)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=FS_TICK)

plt.tight_layout()
fig2.savefig(SAVE_DIR / "02_ring_complexity.png", dpi=150, bbox_inches="tight")
plt.close(fig2)
print("[OK] 02_ring_complexity.png")

# ===========================================================================
# FIGURE 3: Atom composition analysis
# ===========================================================================
fig3, axes = plt.subplots(2, 3, figsize=(20, 13))
fig3.suptitle("Atom Composition Evolution (by epoch)", fontsize=FS_SUPTITLE, fontweight="bold", y=1.01)

# 3a: Element distribution (stacked bar) for each epoch
ax = axes[0, 0]
elements = ["C", "O", "N", "S", "F", "Cl", "Br", "P"]
colors_elem = {"C": "#555555", "O": C_RED, "N": C_BLUE, "S": "#FFD700",
               "F": C_GREEN, "Cl": C_GREEN, "Br": "#8B0000", "P": C_ORANGE}
x = np.arange(len(ep_list))
bottom = np.zeros(len(ep_list))
for elem in ["C", "O", "N", "S"]:
    vals = [epoch_means[ep].get(f"n{elem}", 0) for ep in ep_list]
    ax.bar(x, vals, bottom=bottom, label=elem, color=colors_elem[elem], alpha=0.85, edgecolor="white")
    bottom += vals
ax.set_xticks(x)
ax.set_xticklabels(ep_list)
ax.set_ylabel("Avg Atom Count", fontsize=FS)
ax.set_title("(a) Major Elements (C,O,N,S)", fontsize=FS_TITLE, fontweight="bold")
ax.legend(fontsize=FS_LEGEND, ncol=4)
ax.grid(True, alpha=0.3, axis="y")
ax.tick_params(labelsize=FS_TICK)

# 3b: Carbon fraction per epoch
ax = axes[0, 1]
cf_vals = [epoch_means[ep]["carbon_frac"] for ep in ep_list]
ax.bar(ep_list, cf_vals, color=C_BROWN, alpha=0.8, edgecolor="white")
ax.plot(ep_list, cf_vals, "o-", color=C_RED, markersize=6, linewidth=2)
ax.axhline(y=0.70, color="green", linestyle="--", alpha=0.5, label="~70% (balanced)")
ax.axhline(y=0.85, color="red", linestyle="--", alpha=0.5, label="85% (danger zone)")
ax.set_ylabel("Carbon Fraction", fontsize=FS)
ax.set_title("(b) Carbon Fraction → Collapse Signal", fontsize=FS_TITLE, fontweight="bold")
ax.legend(fontsize=FS_LEGEND)
ax.grid(True, alpha=0.3, axis="y")
ax.tick_params(labelsize=FS_TICK)

# 3c: Heteroatom count
ax = axes[0, 2]
het_vals = [epoch_means[ep]["n_hetero"] for ep in ep_list]
ax.bar(ep_list, het_vals, color=C_GREEN, alpha=0.8, edgecolor="white")
ax.plot(ep_list, het_vals, "o-", color=C_RED, markersize=6, linewidth=2)
ax.set_ylabel("Avg Heteroatoms", fontsize=FS)
ax.set_title("(c) Heteroatom Count", fontsize=FS_TITLE, fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")
ax.tick_params(labelsize=FS_TICK)

# 3d: Unique atom types per molecule
ax = axes[1, 0]
ua_vals = [epoch_means[ep]["unique_atoms"] for ep in ep_list]
ax.bar(ep_list, ua_vals, color=C_PURPLE, alpha=0.8, edgecolor="white")
ax.plot(ep_list, ua_vals, "o-", color=C_RED, markersize=6, linewidth=2)
ax.set_ylabel("Unique Element Types", fontsize=FS)
ax.set_title("(d) Element Diversity per Molecule", fontsize=FS_TITLE, fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")
ax.tick_params(labelsize=FS_TICK)

# 3e: O/C ratio
ax = axes[1, 1]
oc_vals = []
for ep in ep_list:
    nC = epoch_means[ep].get("nC", 1)
    nO = epoch_means[ep].get("nO", 0)
    oc_vals.append(nO / nC if nC > 0 else 0)
ax.bar(ep_list, oc_vals, color=C_CYAN, alpha=0.8, edgecolor="white")
ax.plot(ep_list, oc_vals, "o-", color=C_RED, markersize=6, linewidth=2)
ax.set_ylabel("O/C Ratio", fontsize=FS)
ax.set_title("(e) Oxygen-to-Carbon Ratio", fontsize=FS_TITLE, fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")
ax.tick_params(labelsize=FS_TICK)

# 3f: N atom count
ax = axes[1, 2]
n_vals = [epoch_means[ep].get("nN", 0) for ep in ep_list]
ax.bar(ep_list, n_vals, color=C_BLUE, alpha=0.8, edgecolor="white")
ax.plot(ep_list, n_vals, "o-", color=C_RED, markersize=6, linewidth=2)
ax.set_ylabel("Avg N Atoms", fontsize=FS)
ax.set_title("(f) Nitrogen Count", fontsize=FS_TITLE, fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")
ax.tick_params(labelsize=FS_TICK)

plt.tight_layout()
fig3.savefig(SAVE_DIR / "03_atom_composition.png", dpi=150, bbox_inches="tight")
plt.close(fig3)
print("[OK] 03_atom_composition.png")

# ===========================================================================
# FIGURE 4: Chemical property distributions (TPA, HBA/HBD, LogP, complexity)
# ===========================================================================
fig4, axes = plt.subplots(2, 3, figsize=(20, 13))
fig4.suptitle("Chemical Property Trends", fontsize=FS_SUPTITLE, fontweight="bold", y=1.01)

# 4a: TPSA trend
ax = axes[0, 0]
ax.plot(epochs_arr, df["tpsa_mean"], color=C_RED, linewidth=2.0, marker="o", markersize=3, label="Mean")
ax.fill_between(epochs_arr, df["tpsa_mean"] - df["tpsa_std"], df["tpsa_mean"] + df["tpsa_std"],
                color=C_RED, alpha=0.2, label="±1σ")
ax.set_ylabel("TPSA (Å²)", fontsize=FS)
ax.set_title("(a) Topological Polar Surface Area", fontsize=FS_TITLE, fontweight="bold")
ax.legend(fontsize=FS_LEGEND)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=FS_TICK)

# 4b: HBA/HBD
ax = axes[0, 1]
ax.plot(epochs_arr, df["hba_mean"], color=C_BLUE, linewidth=2.0, marker="o", markersize=3, label="HBA")
ax.plot(epochs_arr, df["hbd_mean"], color=C_RED, linewidth=2.0, marker="s", markersize=3, label="HBD")
ax.set_ylabel("Count", fontsize=FS)
ax.set_title("(b) H-Bond Acceptors & Donors", fontsize=FS_TITLE, fontweight="bold")
ax.legend(fontsize=FS_LEGEND)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=FS_TICK)

# 4c: LogP distribution
ax = axes[0, 2]
ax.plot(epochs_arr, df["logp_mean"], color=C_PURPLE, linewidth=2.0, marker="o", markersize=3, label="Mean")
ax.fill_between(epochs_arr, df["logp_mean"] - df["logp_std"], df["logp_mean"] + df["logp_std"],
                color=C_PURPLE, alpha=0.2, label="±1σ")
ax.axhline(y=2.5, color="gray", linestyle="--", alpha=0.5, label="ChEMBL median ~2.5")
ax.axhline(y=5.0, color="red", linestyle="--", alpha=0.5, label="Lipinski limit=5")
ax.set_ylabel("LogP", fontsize=FS)
ax.set_title("(c) LogP Trend", fontsize=FS_TITLE, fontweight="bold")
ax.legend(fontsize=FS_LEGEND)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=FS_TICK)

# 4d: Rotatable bonds
ax = axes[1, 0]
rb_vals = [epoch_means[ep]["n_rot_bonds"] for ep in ep_list]
ax.bar(ep_list, rb_vals, color=C_GREEN, alpha=0.8, edgecolor="white")
ax.plot(ep_list, rb_vals, "o-", color=C_RED, markersize=6, linewidth=2)
ax.set_ylabel("Avg Rotatable Bonds", fontsize=FS)
ax.set_title("(d) Rotatable Bonds (flexibility)", fontsize=FS_TITLE, fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")
ax.tick_params(labelsize=FS_TICK)

# 4e: Bertz CT (complexity index)
ax = axes[1, 1]
bertz_vals = [epoch_means[ep]["bertz_ct"] for ep in ep_list]
ax.bar(ep_list, bertz_vals, color=C_ORANGE, alpha=0.8, edgecolor="white")
ax.plot(ep_list, bertz_vals, "o-", color=C_RED, markersize=6, linewidth=2)
ax.set_ylabel("Bertz CT", fontsize=FS)
ax.set_title("(e) Bertz Complexity Index", fontsize=FS_TITLE, fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")
ax.tick_params(labelsize=FS_TICK)

# 4f: Scaffold analysis
ax = axes[1, 2]
scf_atom_vals = [epoch_means[ep]["scf_atoms"] for ep in ep_list]
scf_ring_vals = [epoch_means[ep]["scf_rings"] for ep in ep_list]
ax.plot(ep_list, scf_atom_vals, "o-", color=C_BLUE, linewidth=2, markersize=6, label="Scaffold atoms")
ax.plot(ep_list, scf_ring_vals, "s-", color=C_RED, linewidth=2, markersize=6, label="Scaffold rings")
ax.set_ylabel("Count", fontsize=FS)
ax.set_title("(f) Murcko Scaffold Size", fontsize=FS_TITLE, fontweight="bold")
ax.legend(fontsize=FS_LEGEND)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=FS_TICK)

plt.tight_layout()
fig4.savefig(SAVE_DIR / "04_chemical_properties.png", dpi=150, bbox_inches="tight")
plt.close(fig4)
print("[OK] 04_chemical_properties.png")

# ===========================================================================
# FIGURE 5: Final epoch (62) detailed distributions
# ===========================================================================
ep_last = 62
edf_last = epoch_analysis[ep_last]

fig5, axes = plt.subplots(2, 3, figsize=(20, 13))
fig5.suptitle(f"Epoch {ep_last} Molecule Distributions (n={len(edf_last)})",
              fontsize=FS_SUPTITLE, fontweight="bold", y=1.01)

# 5a: Ring count histogram
ax = axes[0, 0]
ax.hist(edf_last["n_rings"], bins=range(0, int(edf_last["n_rings"].max()) + 2),
        color=C_BLUE, alpha=0.8, edgecolor="white")
ax.axvline(x=edf_last["n_rings"].mean(), color="red", linestyle="--", linewidth=2,
           label=f'Mean={edf_last["n_rings"].mean():.1f}')
ax.set_xlabel("Ring Count", fontsize=FS)
ax.set_ylabel("Molecules", fontsize=FS)
ax.set_title("(a) Ring Count Distribution", fontsize=FS_TITLE, fontweight="bold")
ax.legend(fontsize=FS_LEGEND)
ax.grid(True, alpha=0.3, axis="y")
ax.tick_params(labelsize=FS_TICK)

# 5b: Carbon fraction histogram
ax = axes[0, 1]
bins_cf = np.linspace(0.6, 1.0, 31)
ax.hist(edf_last["carbon_frac"], bins=bins_cf, color=C_BROWN, alpha=0.8, edgecolor="white")
ax.axvline(x=edf_last["carbon_frac"].mean(), color="red", linestyle="--", linewidth=2,
           label=f'Mean={edf_last["carbon_frac"].mean():.3f}')
ax.axvline(x=0.70, color="green", linestyle="--", alpha=0.5, label="Balanced (~0.70)")
ax.set_xlabel("Carbon Fraction", fontsize=FS)
ax.set_ylabel("Molecules", fontsize=FS)
ax.set_title("(b) Carbon Fraction Distribution", fontsize=FS_TITLE, fontweight="bold")
ax.legend(fontsize=FS_LEGEND)
ax.grid(True, alpha=0.3, axis="y")
ax.tick_params(labelsize=FS_TICK)

# 5c: Heteroatom count histogram
ax = axes[0, 2]
ax.hist(edf_last["n_hetero"], bins=range(0, int(edf_last["n_hetero"].max()) + 2),
        color=C_GREEN, alpha=0.8, edgecolor="white")
ax.axvline(x=edf_last["n_hetero"].mean(), color="red", linestyle="--", linewidth=2,
           label=f'Mean={edf_last["n_hetero"].mean():.1f}')
ax.set_xlabel("Heteroatom Count", fontsize=FS)
ax.set_ylabel("Molecules", fontsize=FS)
ax.set_title("(c) Heteroatom Distribution", fontsize=FS_TITLE, fontweight="bold")
ax.legend(fontsize=FS_LEGEND)
ax.grid(True, alpha=0.3, axis="y")
ax.tick_params(labelsize=FS_TICK)

# 5d: Fused ring sets
ax = axes[1, 0]
ax.hist(edf_last["n_fused_sets"], bins=range(0, int(edf_last["n_fused_sets"].max()) + 2),
        color=C_RED, alpha=0.8, edgecolor="white")
ax.axvline(x=edf_last["n_fused_sets"].mean(), color="black", linestyle="--", linewidth=2,
           label=f'Mean={edf_last["n_fused_sets"].mean():.2f}')
ax.set_xlabel("Fused Ring Systems", fontsize=FS)
ax.set_ylabel("Molecules", fontsize=FS)
ax.set_title("(d) Fused Ring Systems per Molecule", fontsize=FS_TITLE, fontweight="bold")
ax.legend(fontsize=FS_LEGEND)
ax.grid(True, alpha=0.3, axis="y")
ax.tick_params(labelsize=FS_TICK)

# 5e: Bridgeheads histogram
ax = axes[1, 1]
ax.hist(edf_last["n_bridgeheads"], bins=range(0, int(edf_last["n_bridgeheads"].max()) + 2),
        color=C_PURPLE, alpha=0.8, edgecolor="white")
ax.axvline(x=edf_last["n_bridgeheads"].mean(), color="red", linestyle="--", linewidth=2,
           label=f'Mean={edf_last["n_bridgeheads"].mean():.2f}')
ax.set_xlabel("Bridgehead Atoms", fontsize=FS)
ax.set_ylabel("Molecules", fontsize=FS)
ax.set_title("(e) Bridgehead Atom Distribution", fontsize=FS_TITLE, fontweight="bold")
ax.legend(fontsize=FS_LEGEND)
ax.grid(True, alpha=0.3, axis="y")
ax.tick_params(labelsize=FS_TICK)

# 5f: Atom types histogram
ax = axes[1, 2]
from collections import Counter
all_atom_types = Counter()
for _, row in edf_last.iterrows():
    all_atom_types["C"] += row["nC"]
    all_atom_types["O"] += row["nO"]
    all_atom_types["N"] += row["nN"]
    all_atom_types["S"] += row["nS"]
atoms_sorted = sorted(all_atom_types.items(), key=lambda x: x[1], reverse=True)
labels = [a[0] for a in atoms_sorted]
values = [a[1] for a in atoms_sorted]
colors = [colors_elem.get(l, "#999999") for l in labels]
ax.bar(labels, values, color=colors, alpha=0.8, edgecolor="white")
for i, (label, val) in enumerate(zip(labels, values)):
    ax.text(i, val + max(values) * 0.01, f"{val}\n({val/max(values)*100:.1f}%)",
            ha="center", fontsize=9, fontweight="bold")
ax.set_ylabel("Total Atom Count", fontsize=FS)
ax.set_title("(f) Overall Atom Composition", fontsize=FS_TITLE, fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")
ax.tick_params(labelsize=FS_TICK)

plt.tight_layout()
fig5.savefig(SAVE_DIR / "05_epoch62_distributions.png", dpi=150, bbox_inches="tight")
plt.close(fig5)
print("[OK] 05_epoch62_distributions.png")

# ===========================================================================
# FIGURE 6: Molecule property scatter at epoch 62
# ===========================================================================
fig6, axes = plt.subplots(2, 2, figsize=(16, 14))
fig6.suptitle(f"Epoch {ep_last}: Property Cross-Relationships", fontsize=FS_SUPTITLE, fontweight="bold", y=1.01)

# 6a: Rings vs Heteroatoms
ax = axes[0, 0]
sc = ax.scatter(edf_last["n_rings"], edf_last["n_hetero"],
                c=edf_last["carbon_frac"], cmap="RdYlGn_r", alpha=0.5, s=20, edgecolors="none",
                vmin=0.6, vmax=1.0)
plt.colorbar(sc, ax=ax, label="Carbon Fraction")
ax.set_xlabel("Ring Count", fontsize=FS)
ax.set_ylabel("Heteroatom Count", fontsize=FS)
ax.set_title("(a) Rings vs Heteroatoms  (color = C fraction)", fontsize=FS_TITLE, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=FS_TICK)

# 6b: Carbon fraction vs Bertz CT
ax = axes[0, 1]
sc = ax.scatter(edf_last["carbon_frac"], edf_last["bertz_ct"],
                c=edf_last["n_rings"], cmap="viridis", alpha=0.5, s=20, edgecolors="none")
plt.colorbar(sc, ax=ax, label="Ring Count")
ax.set_xlabel("Carbon Fraction", fontsize=FS)
ax.set_ylabel("Bertz Complexity", fontsize=FS)
ax.set_title("(b) Carbon Fraction vs Complexity", fontsize=FS_TITLE, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=FS_TICK)

# 6c: Rings vs fused systems
ax = axes[1, 0]
sc = ax.scatter(edf_last["n_rings"], edf_last["n_fused_sets"],
                c=edf_last["n_bridgeheads"], cmap="Reds", alpha=0.5, s=20, edgecolors="none")
plt.colorbar(sc, ax=ax, label="Bridgeheads")
ax.set_xlabel("Total Rings", fontsize=FS)
ax.set_ylabel("Fused Ring Systems", fontsize=FS)
ax.set_title("(c) Rings vs Fused Systems  (color = bridgeheads)", fontsize=FS_TITLE, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=FS_TICK)

# 6d: Atoms vs Unique Elements
ax = axes[1, 1]
sc = ax.scatter(edf_last["n_atoms"], edf_last["unique_atoms"],
                c=edf_last["n_hetero"], cmap="plasma", alpha=0.5, s=20, edgecolors="none")
plt.colorbar(sc, ax=ax, label="Heteroatom Count")
ax.set_xlabel("Total Atoms", fontsize=FS)
ax.set_ylabel("Unique Element Types", fontsize=FS)
ax.set_title("(d) Molecular Size vs Element Diversity", fontsize=FS_TITLE, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=FS_TICK)

plt.tight_layout()
fig6.savefig(SAVE_DIR / "06_epoch62_cross_relationships.png", dpi=150, bbox_inches="tight")
plt.close(fig6)
print("[OK] 06_epoch62_cross_relationships.png")

# ===========================================================================
# FIGURE 7: Compare first vs last epoch violin distributions
# ===========================================================================
fig7, axes = plt.subplots(2, 3, figsize=(20, 13))
fig7.suptitle(f"Epoch 1 → Epoch {ep_last} Distribution Shift", fontsize=FS_SUPTITLE, fontweight="bold", y=1.01)

edf_first = epoch_analysis[1]

comparison_pairs = [
    ("n_rings", "Ring Count", C_BLUE),
    ("carbon_frac", "Carbon Fraction", C_BROWN),
    ("n_hetero", "Heteroatom Count", C_GREEN),
    ("n_fused_sets", "Fused Ring Systems", C_RED),
    ("n_bridgeheads", "Bridgehead Atoms", C_PURPLE),
    ("unique_atoms", "Unique Element Types", C_CYAN),
]

for idx, (col, title, color) in enumerate(comparison_pairs):
    ax = axes[idx // 3, idx % 3]
    data = [edf_first[col].values, edf_last[col].values]
    parts = ax.violinplot(data, positions=[0, 1], showmeans=True, showmedians=False, widths=0.6)
    for pc in parts["bodies"]:
        pc.set_facecolor(color)
        pc.set_alpha(0.5)
    for i, d in enumerate(data):
        ax.scatter(np.full_like(d, i) + np.random.uniform(-0.05, 0.05, len(d)),
                   d, s=3, alpha=0.15, color="black")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Epoch 1", f"Epoch {ep_last}"])
    ax.set_ylabel(title, fontsize=FS)
    ax.set_title(f"({chr(97+idx)}) {title}", fontsize=FS_TITLE, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(labelsize=FS_TICK)
    # Add mean text
    ax.text(0, ax.get_ylim()[0], f"μ={data[0].mean():.3f}", ha="center", va="bottom", fontsize=9, color="blue")
    ax.text(1, ax.get_ylim()[0], f"μ={data[1].mean():.3f}", ha="center", va="bottom", fontsize=9, color="blue")

plt.tight_layout()
fig7.savefig(SAVE_DIR / "07_epoch1_vs_last_violin.png", dpi=150, bbox_inches="tight")
plt.close(fig7)
print("[OK] 07_epoch1_vs_last_violin.png")

# ===========================================================================
# FIGURE 8: Summary table
# ===========================================================================
fig8, ax = plt.subplots(figsize=(16, 8))
ax.axis("off")

# Build summary rows from epoch analysis
def fmt_change(v1, v_last):
    delta = v_last - v1
    if abs(delta) < 0.001:
        return "→ 0"
    direction = "↑" if delta > 0 else "↓"
    return f"{direction} {abs(delta):.3f}"

def val_str(d, key, fmt=".3f"):
    return f"{d.get(key, 0):{fmt}}"

rows = [
    ["Category", "Metric", "Epoch 1", f"Epoch {ep_last}", "Change"],
    # Ring complexity
    ["Ring\nComplexity", "Avg Rings", val_str(epoch_means[1], "n_rings", ".1f"),
     val_str(epoch_means[ep_last], "n_rings", ".1f"),
     fmt_change(epoch_means[1]["n_rings"], epoch_means[ep_last]["n_rings"])],
    ["", "Aromatic Rings", val_str(epoch_means[1], "n_arom_rings", ".1f"),
     val_str(epoch_means[ep_last], "n_arom_rings", ".1f"),
     fmt_change(epoch_means[1]["n_arom_rings"], epoch_means[ep_last]["n_arom_rings"])],
    ["", "Fused Ring Sets", val_str(epoch_means[1], "n_fused_sets", ".2f"),
     val_str(epoch_means[ep_last], "n_fused_sets", ".2f"),
     fmt_change(epoch_means[1]["n_fused_sets"], epoch_means[ep_last]["n_fused_sets"])],
    ["", "Bridgeheads", val_str(epoch_means[1], "n_bridgeheads", ".1f"),
     val_str(epoch_means[ep_last], "n_bridgeheads", ".1f"),
     fmt_change(epoch_means[1]["n_bridgeheads"], epoch_means[ep_last]["n_bridgeheads"])],
    ["", "Spiro Atoms", val_str(epoch_means[1], "n_spiro_atoms", ".2f"),
     val_str(epoch_means[ep_last], "n_spiro_atoms", ".2f"),
     fmt_change(epoch_means[1]["n_spiro_atoms"], epoch_means[ep_last]["n_spiro_atoms"])],
    # Atom composition
    ["Atom\nComposition", "Carbon Fraction", val_str(epoch_means[1], "carbon_frac"),
     val_str(epoch_means[ep_last], "carbon_frac"),
     fmt_change(epoch_means[1]["carbon_frac"], epoch_means[ep_last]["carbon_frac"])],
    ["", "Total Atoms", val_str(epoch_means[1], "n_atoms", ".1f"),
     val_str(epoch_means[ep_last], "n_atoms", ".1f"),
     fmt_change(epoch_means[1]["n_atoms"], epoch_means[ep_last]["n_atoms"])],
    ["", "Oxygen Count", val_str(epoch_means[1], "nO", ".1f"),
     val_str(epoch_means[ep_last], "nO", ".1f"),
     fmt_change(epoch_means[1]["nO"], epoch_means[ep_last]["nO"])],
    ["", "Nitrogen Count", val_str(epoch_means[1], "nN", ".1f"),
     val_str(epoch_means[ep_last], "nN", ".1f"),
     fmt_change(epoch_means[1]["nN"], epoch_means[ep_last]["nN"])],
    ["", "Unique Elements", val_str(epoch_means[1], "unique_atoms", ".1f"),
     val_str(epoch_means[ep_last], "unique_atoms", ".1f"),
     fmt_change(epoch_means[1]["unique_atoms"], epoch_means[ep_last]["unique_atoms"])],
    # Molecule properties
    ["Molecule\nProperties", "Bertz CT", val_str(epoch_means[1], "bertz_ct", ".1f"),
     val_str(epoch_means[ep_last], "bertz_ct", ".1f"),
     fmt_change(epoch_means[1]["bertz_ct"], epoch_means[ep_last]["bertz_ct"])],
    ["", "Rotatable Bonds", val_str(epoch_means[1], "n_rot_bonds", ".1f"),
     val_str(epoch_means[ep_last], "n_rot_bonds", ".1f"),
     fmt_change(epoch_means[1]["n_rot_bonds"], epoch_means[ep_last]["n_rot_bonds"])],
    ["", "Scaffold Atoms", val_str(epoch_means[1], "scf_atoms", ".1f"),
     val_str(epoch_means[ep_last], "scf_atoms", ".1f"),
     fmt_change(epoch_means[1]["scf_atoms"], epoch_means[ep_last]["scf_atoms"])],
    # Training metrics (from summary)
    ["Training", "RDKit Validity", f"{df['rdkit_valid_ratio'].iloc[0]:.2%}",
     f"{df['rdkit_valid_ratio'].iloc[-1]:.2%}",
     fmt_change(df['rdkit_valid_ratio'].iloc[0], df['rdkit_valid_ratio'].iloc[-1])],
    ["", "Docking Score", f"{df['avg_docking'].iloc[0]:.2f}",
     f"{df['avg_docking'].iloc[-1]:.2f}",
     fmt_change(df['avg_docking'].iloc[0], df['avg_docking'].iloc[-1])],
    ["", "ECFP4 Similarity", f"{df['avg_chembl_prior_ecfp4_similarity'].iloc[0]:.4f}",
     f"{df['avg_chembl_prior_ecfp4_similarity'].iloc[-1]:.4f}",
     fmt_change(df['avg_chembl_prior_ecfp4_similarity'].iloc[0], df['avg_chembl_prior_ecfp4_similarity'].iloc[-1])],
    ["", "LogP Mean", f"{df['logp_mean'].iloc[0]:.2f}",
     f"{df['logp_mean'].iloc[-1]:.2f}",
     fmt_change(df['logp_mean'].iloc[0], df['logp_mean'].iloc[-1])],
    ["", "TPSA", f"{df['tpsa_mean'].iloc[0]:.2f}",
     f"{df['tpsa_mean'].iloc[-1]:.2f}",
     fmt_change(df['tpsa_mean'].iloc[0], df['tpsa_mean'].iloc[-1])],
]

table = ax.table(cellText=rows, loc="center", cellLoc="center",
                 colWidths=[0.15, 0.22, 0.15, 0.18, 0.18])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.8)

# Style header
for j in range(len(rows[0])):
    cell = table[0, j]
    cell.set_facecolor("#2E4057")
    cell.set_text_props(color="white", fontweight="bold", fontsize=12)

# Color-code categories
for i in range(1, len(rows)):
    for j in range(len(rows[0])):
        cell = table[i, j]
        if rows[i][0]:  # category header
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", fontweight="bold")
    # Color change column (last column index = 4)
    change = rows[i][-1]
    if "↑" in change:
        table[i, 4].set_facecolor("#FFC7CE")
    elif "↓" in change:
        table[i, 4].set_facecolor("#C6EFCE")
    elif "→" in change:
        table[i, 4].set_facecolor("#E2EFDA")

ax.set_title(f"Training Progress Summary  |  Epoch 1 → Epoch {ep_last}",
             fontsize=FS_SUPTITLE, fontweight="bold", pad=20)

plt.tight_layout()
fig8.savefig(SAVE_DIR / "08_summary_table.png", dpi=150, bbox_inches="tight")
plt.close(fig8)
print("[OK] 08_summary_table.png")

# ===========================================================================
# FIGURE 9: Example SMILES gallery (epoch 1 vs epoch 62)
# ===========================================================================
fig9, axes = plt.subplots(2, 1, figsize=(20, 10))
fig9.suptitle("Example Generated SMILES: Epoch 1 vs Epoch 62", fontsize=FS_SUPTITLE, fontweight="bold", y=1.02)

for ax, ep, mdf, title_text in [
    (axes[0], 1, mol_data[1], f"Epoch 1  ({len(mol_data[1])} valid, n_rings={epoch_means[1]['n_rings']:.1f}, "
     f"C-frac={epoch_means[1]['carbon_frac']:.3f}, heteroatom={epoch_means[1]['n_hetero']:.1f})"),
    (axes[1], ep_last, mol_data[ep_last], f"Epoch {ep_last}  ({len(mol_data[ep_last])} valid, "
     f"n_rings={epoch_means[ep_last]['n_rings']:.1f}, C-frac={epoch_means[ep_last]['carbon_frac']:.3f}, "
     f"heteroatom={epoch_means[ep_last]['n_hetero']:.1f})"),
]:
    ax.axis("off")
    ax.set_title(title_text, fontsize=FS_TITLE, fontweight="bold", pad=10)

    # Pick 20 representative molecules sorted by docking score
    sample_mols = mdf.dropna(subset=["smiles"]).head(20)
    text_lines = []
    for _, row in sample_mols.iterrows():
        smi = row["smiles"]
        dock = row.get("docking_raw", 0)
        text_lines.append(f"  {smi}  (dock={dock:.1f})")

    ax.text(0.02, 0.98, "\n".join(text_lines), transform=ax.transAxes,
            fontsize=8, fontfamily="monospace", verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

plt.tight_layout()
fig9.savefig(SAVE_DIR / "09_example_smiles.png", dpi=150, bbox_inches="tight")
plt.close(fig9)
print("[OK] 09_example_smiles.png")

# ===========================================================================
# Console summary
# ===========================================================================
print()
print("=" * 80)
print(f"  7GPU dock+ecfp4+fcfp4 Training Analysis")
print("=" * 80)
print(f"  Run          : 7gpu-dock-ecfp4-fcfp4_20260525_212140")
print(f"  Epochs       : 1 → {N_EPOCH}")
print(f"  Reward weights: docking=0.5, chembl_ecfp4=0.2, chembl_fcfp4=0.1, logp=0.2")
print()
print(f"  {'Metric':<35s} {'Epoch 1':>12s} {'Epoch '+str(ep_last):>12s} {'Change':>12s}")
print("  " + "-" * 71)

summary_items = [
    ("RDKit Validity", f"{df['rdkit_valid_ratio'].iloc[0]:.1%}", f"{df['rdkit_valid_ratio'].iloc[-1]:.1%}"),
    ("Docking Score", f"{df['avg_docking'].iloc[0]:.1f}", f"{df['avg_docking'].iloc[-1]:.1f}"),
    ("ECFP4 Similarity", f"{df['avg_chembl_prior_ecfp4_similarity'].iloc[0]:.3f}", f"{df['avg_chembl_prior_ecfp4_similarity'].iloc[-1]:.3f}"),
    ("LogP Mean", f"{df['logp_mean'].iloc[0]:.1f}", f"{df['logp_mean'].iloc[-1]:.1f}"),
    ("TPSA", f"{df['tpsa_mean'].iloc[0]:.1f}", f"{df['tpsa_mean'].iloc[-1]:.1f}"),
    ("Carbon Fraction", f"{df['carbon_fraction'].iloc[0]:.3f}", f"{df['carbon_fraction'].iloc[-1]:.3f}"),
    ("Heteroatom Count", f"{df['hetero_atom_mean'].iloc[0]:.1f}", f"{df['hetero_atom_mean'].iloc[-1]:.1f}"),
    ("--- RDKit Detailed ---", "", ""),
    ("Avg Rings", f"{epoch_means[1]['n_rings']:.1f}", f"{epoch_means[ep_last]['n_rings']:.1f}"),
    ("Fused Ring Systems", f"{epoch_means[1]['n_fused_sets']:.2f}", f"{epoch_means[ep_last]['n_fused_sets']:.2f}"),
    ("Bridgeheads", f"{epoch_means[1]['n_bridgeheads']:.1f}", f"{epoch_means[ep_last]['n_bridgeheads']:.1f}"),
    ("Bertz CT", f"{epoch_means[1]['bertz_ct']:.1f}", f"{epoch_means[ep_last]['bertz_ct']:.1f}"),
    ("Unique Elements", f"{epoch_means[1]['unique_atoms']:.1f}", f"{epoch_means[ep_last]['unique_atoms']:.1f}"),
    ("Oxygen Count", f"{epoch_means[1]['nO']:.1f}", f"{epoch_means[ep_last]['nO']:.1f}"),
    ("Nitrogen Count", f"{epoch_means[1]['nN']:.1f}", f"{epoch_means[ep_last]['nN']:.1f}"),
]

for name, v1, vlast in summary_items:
    print(f"  {name:<35s} {v1:>12s} {vlast:>12s}", end="")
    if "---" not in name:
        try:
            n1 = float(v1)
            n_last = float(vlast)
            delta = n_last - n1
            if abs(delta) < 0.001:
                print(f"  {'→ 0':>10s}")
            else:
                direction = "↑" if delta > 0 else "↓"
                print(f"  {direction} {abs(delta):.3f}")
        except ValueError:
            print()
    else:
        print()

print()
print(f"  Figures saved to: {SAVE_DIR}/")
for fname in sorted(os.listdir(SAVE_DIR)):
    if fname.endswith(".png"):
        print(f"    - {fname}")
print("=" * 80)
