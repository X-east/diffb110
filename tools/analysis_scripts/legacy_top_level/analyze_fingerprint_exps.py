#!/usr/bin/env python3
"""Analyze fingerprint experiments #5, #6, #7 with tailored plots per experiment.

Usage:
    python analysis/analyze_fingerprint_exps.py 5
    python analysis/analyze_fingerprint_exps.py 6
    python analysis/analyze_fingerprint_exps.py 7
"""

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Experiment configs
# ---------------------------------------------------------------------------
EXPS = {
    5: {
        "root": "/home/md666/data/disk6/DiffB110/runs/6gpu-chembl-only_20260521_112200",
        "name": "ECFP4+FCFP4 (无docking)",
        "short": "#5 指纹拆分",
        "epochs": 18,
        "total_mol": 504,
        "has_docking": False,
        "has_diversity": False,
    },
    6: {
        "root": "/home/md666/data/disk6/DiffB110/runs/6gpu-chembl-only_20260521_154957",
        "name": "Docking+ECFP4+FCFP4",
        "short": "#6 docking回归",
        "epochs": 20,
        "total_mol": 504,
        "has_docking": True,
        "has_diversity": False,
    },
    7: {
        "root": "/home/md666/data/disk6/DiffB110/runs/7gpu-dock-ecfp4-fcfp4_20260521_212927",
        "name": "Docking+ECFP4+FCFP4+Diversity (坍塌)",
        "short": "#7 含diversity",
        "epochs": 50,
        "total_mol": 504,
        "has_docking": True,
        "has_diversity": True,
    },
}

# ---------------------------------------------------------------------------
# Fonts (larger)
# ---------------------------------------------------------------------------
FS = 14
FS_TITLE = 17
FS_SUPTITLE = 20
FS_LEGEND = 12
FS_TICK = 12

C_BLUE = "#4C72B0"
C_ORANGE = "#DD8452"
C_GREEN = "#55A868"
C_RED = "#C44E52"
C_PURPLE = "#8172B2"
C_BROWN = "#937860"
C_CYAN = "#64B5CD"

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python analysis/analyze_fingerprint_exps.py <5|6|7>")
        sys.exit(1)
    exp_id = int(sys.argv[1])
    cfg = EXPS[exp_id]

    ROOT = Path(cfg["root"])
    CSV_PATH = ROOT / "training_summary.csv"
    EPOCH_DIR = ROOT / "epochs"
    NE = cfg["epochs"]
    TOTAL = cfg["total_mol"]

    # Load summary
    df = pd.read_csv(CSV_PATH)
    def parse_failure_counts(s):
        try:
            return json.loads(s.replace("'", '"'))
        except Exception:
            return {}
    fc_df = df["failure_counts"].apply(parse_failure_counts).apply(pd.Series).fillna(0).astype(int)
    for col in fc_df.columns:
        df[col] = fc_df[col]
    epochs = df["epoch"].values

    # Load last epoch molecules
    last_ep_path = EPOCH_DIR / f"epoch_{NE:04d}" / "molecules.csv"
    mol_last = pd.read_csv(last_ep_path)
    valid_last = mol_last[mol_last["rdkit_valid"] == 1].copy()
    n_valid = len(valid_last)

    print(f"Experiment #{exp_id}: {cfg['name']}")
    print(f"  Epochs: {NE}, Valid molecules at epoch{NE}: {n_valid}/{TOTAL}")

    # =======================================================================
    # FIGURE 1: Training Trend (6-panel)
    # =======================================================================
    fig1, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig1.suptitle(f"#{exp_id} {cfg['short']} Training Curves  (epochs 1–{NE})",
                  fontsize=FS_SUPTITLE, fontweight="bold", y=1.02)

    # 1a: Loss
    ax = axes[0, 0]
    ax.plot(epochs, df["avg_loss"], color=C_BLUE, linewidth=2.0, marker="o", markersize=4)
    ax.set_xlabel("Epoch", fontsize=FS); ax.set_ylabel("PPO Loss", fontsize=FS)
    ax.set_title("(a) PPO Loss", fontsize=FS_TITLE, fontweight="bold")
    ax.grid(True, alpha=0.3); ax.tick_params(labelsize=FS_TICK)

    # 1b: Validity
    ax = axes[0, 1]
    ax.plot(epochs, df["rdkit_valid_ratio"] * 100, color=C_GREEN, linewidth=2.0, marker="s", markersize=4)
    ax.set_xlabel("Epoch", fontsize=FS); ax.set_ylabel("Validity (%)", fontsize=FS)
    ax.set_title("(b) RDKit Validity", fontsize=FS_TITLE, fontweight="bold")
    ax.grid(True, alpha=0.3); ax.tick_params(labelsize=FS_TICK)

    # 1c: ECFP4 Similarity
    ax = axes[0, 2]
    ax.plot(epochs, df["avg_chembl_prior_ecfp4_similarity"], color=C_PURPLE, linewidth=2.0, marker="D", markersize=4)
    ax.set_xlabel("Epoch", fontsize=FS); ax.set_ylabel("ECFP4 Similarity", fontsize=FS)
    ax.set_title("(c) ECFP4 CheMBL Similarity", fontsize=FS_TITLE, fontweight="bold")
    ax.grid(True, alpha=0.3); ax.tick_params(labelsize=FS_TICK)

    # 1d: FCFP4 Similarity
    ax = axes[1, 0]
    ax.plot(epochs, df["avg_chembl_prior_fcfp4_similarity"], color=C_ORANGE, linewidth=2.0, marker="D", markersize=4)
    ax.set_xlabel("Epoch", fontsize=FS); ax.set_ylabel("FCFP4 Similarity", fontsize=FS)
    ax.set_title("(d) FCFP4 CheMBL Similarity", fontsize=FS_TITLE, fontweight="bold")
    ax.grid(True, alpha=0.3); ax.tick_params(labelsize=FS_TICK)

    # 1e: Failure breakdown
    ax = axes[1, 1]
    ax.fill_between(epochs, 0, df["success"], label="Success", color=C_GREEN, alpha=0.7)
    ax.fill_between(epochs, df["success"], df["success"] + df["fragmented_molecule"],
                    label="Fragmented", color=C_ORANGE, alpha=0.7)
    total_fails = df["success"] + df["fragmented_molecule"] + df["sanitize_failed"]
    ax.fill_between(epochs, total_fails - df["sanitize_failed"], total_fails,
                    label="Sanitize Failed", color=C_RED, alpha=0.7)
    ax.set_xlabel("Epoch", fontsize=FS); ax.set_ylabel("Count", fontsize=FS)
    ax.set_title("(e) Failure Breakdown", fontsize=FS_TITLE, fontweight="bold")
    ax.legend(fontsize=FS_LEGEND, loc="upper right")
    ax.grid(True, alpha=0.3); ax.tick_params(labelsize=FS_TICK)

    # 1f: Docking or Reward Mean
    ax = axes[1, 2]
    if cfg["has_docking"]:
        ax.plot(epochs, df["avg_docking"], color=C_RED, linewidth=2.0, marker="o", markersize=4)
        ax.set_ylabel("Avg Docking Score", fontsize=FS)
        ax.set_title("(f) Avg Docking Score", fontsize=FS_TITLE, fontweight="bold")
    else:
        ax.plot(epochs, df["reward_mean"], color=C_RED, linewidth=2.0, marker="o", markersize=4)
        ax.set_ylabel("Reward Mean", fontsize=FS)
        ax.set_title("(f) Reward Mean", fontsize=FS_TITLE, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=FS)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3); ax.tick_params(labelsize=FS_TICK)

    plt.tight_layout()
    fig1.savefig(ROOT / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print("[OK] training_curves.png")

    # =======================================================================
    # FIGURE 2: Latest Epoch Molecule Analysis (2×2)
    # =======================================================================
    fig2, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig2.suptitle(f"Epoch {NE} Molecule Analysis  ({TOTAL} total, {n_valid} valid)",
                  fontsize=FS_SUPTITLE, fontweight="bold", y=1.02)

    if cfg["has_docking"]:
        # (a) Docking score histogram
        ax = axes[0, 0]
        dock_bins = np.linspace(-15, 0, 25)
        ax.hist(valid_last["docking_raw"], bins=dock_bins, color=C_RED, alpha=0.8, edgecolor="white", linewidth=0.5)
        ax.axvline(x=valid_last["docking_raw"].mean(), color="black", linestyle="--", linewidth=2.0,
                   label=f'Mean = {valid_last["docking_raw"].mean():.2f}')
        ax.set_xlabel("Docking Score", fontsize=FS); ax.set_ylabel("Count", fontsize=FS)
        ax.set_title("(a) Docking Score Distribution", fontsize=FS_TITLE, fontweight="bold")
        ax.legend(fontsize=FS_LEGEND); ax.grid(True, alpha=0.3); ax.tick_params(labelsize=FS_TICK)
    else:
        # No docking: show ECFP4 reward hist
        ax = axes[0, 0]
        bins = np.linspace(-2, 2, 25)
        ax.hist(valid_last["chembl_prior_ecfp4_reward"], bins=bins, color=C_PURPLE, alpha=0.8, edgecolor="white", linewidth=0.5)
        ax.axvline(x=valid_last["chembl_prior_ecfp4_reward"].mean(), color="black", linestyle="--", linewidth=2.0,
                   label=f'Mean = {valid_last["chembl_prior_ecfp4_reward"].mean():.3f}')
        ax.set_xlabel("ECFP4 Reward", fontsize=FS); ax.set_ylabel("Count", fontsize=FS)
        ax.set_title("(a) ECFP4 Reward Distribution", fontsize=FS_TITLE, fontweight="bold")
        ax.legend(fontsize=FS_LEGEND); ax.grid(True, alpha=0.3); ax.tick_params(labelsize=FS_TICK)

    # (b) ECFP4 Similarity histogram
    ax = axes[0, 1]
    sim_bins = np.linspace(0.1, 0.6, 25)
    ax.hist(valid_last["chembl_prior_ecfp4_similarity"], bins=sim_bins, color=C_PURPLE, alpha=0.8, edgecolor="white", linewidth=0.5)
    ax.axvline(x=valid_last["chembl_prior_ecfp4_similarity"].mean(), color="black", linestyle="--", linewidth=2.0,
               label=f'Mean = {valid_last["chembl_prior_ecfp4_similarity"].mean():.3f}')
    ax.set_xlabel("ECFP4 Tanimoto Similarity", fontsize=FS); ax.set_ylabel("Count", fontsize=FS)
    ax.set_title("(b) ECFP4 Similarity Distribution", fontsize=FS_TITLE, fontweight="bold")
    ax.legend(fontsize=FS_LEGEND); ax.grid(True, alpha=0.3); ax.tick_params(labelsize=FS_TICK)

    # (c) FCFP4 Similarity histogram
    ax = axes[1, 0]
    sim_bins2 = np.linspace(0.1, 0.7, 25)
    ax.hist(valid_last["chembl_prior_fcfp4_similarity"], bins=sim_bins2, color=C_ORANGE, alpha=0.8, edgecolor="white", linewidth=0.5)
    ax.axvline(x=valid_last["chembl_prior_fcfp4_similarity"].mean(), color="black", linestyle="--", linewidth=2.0,
               label=f'Mean = {valid_last["chembl_prior_fcfp4_similarity"].mean():.3f}')
    ax.set_xlabel("FCFP4 Tanimoto Similarity", fontsize=FS); ax.set_ylabel("Count", fontsize=FS)
    ax.set_title("(c) FCFP4 Similarity Distribution", fontsize=FS_TITLE, fontweight="bold")
    ax.legend(fontsize=FS_LEGEND); ax.grid(True, alpha=0.3); ax.tick_params(labelsize=FS_TICK)

    # (d) Molecule size or Diversity vs Docking scatter
    ax = axes[1, 1]
    if cfg["has_diversity"]:
        # For #7: scatter docking vs reward, colored by rdkit_valid
        colors = valid_last["rdkit_valid"].map({1: C_GREEN, 0: C_RED})
        ax.scatter(valid_last["docking_raw"], valid_last["reward_total"],
                   c=colors, alpha=0.5, s=15, edgecolors="none")
        ax.set_xlabel("Docking Score", fontsize=FS); ax.set_ylabel("Total Reward", fontsize=FS)
        ax.set_title("(d) Docking vs Reward  (green=valid)", fontsize=FS_TITLE, fontweight="bold")
        ax.grid(True, alpha=0.3); ax.tick_params(labelsize=FS_TICK)
    else:
        smiles_lens = valid_last["smiles"].dropna().str.len()
        ax.hist(smiles_lens, bins=20, color=C_GREEN, alpha=0.8, edgecolor="white", linewidth=0.5)
        ax.set_xlabel("SMILES Length", fontsize=FS); ax.set_ylabel("Count", fontsize=FS)
        ax.set_title("(d) Molecular Size (SMILES length)", fontsize=FS_TITLE, fontweight="bold")
        ax.grid(True, alpha=0.3); ax.tick_params(labelsize=FS_TICK)

    plt.tight_layout()
    fig2.savefig(ROOT / f"epoch{NE}_molecule_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"[OK] epoch{NE}_molecule_analysis.png")

    # =======================================================================
    # FIGURE 3: Distribution Evolution (2-panel: similarity trends or more)
    # =======================================================================
    # Sample epochs for boxplots
    if NE <= 20:
        sample_eps = [1, 5, 10, 15, NE]
    else:
        sample_eps = [1, 5, 10, 15, 20, 25, 30, 40, 50]
    sample_eps = sorted(set([e for e in sample_eps if e <= NE]))

    ecfp4_sims = []; fcfp4_sims = []; valid_s_eps = []
    for ep in sample_eps:
        p = EPOCH_DIR / f"epoch_{ep:04d}" / "molecules.csv"
        if p.exists():
            m = pd.read_csv(p)
            v = m[m["rdkit_valid"] == 1]
            ecfp4_sims.append(v["chembl_prior_ecfp4_similarity"].values)
            fcfp4_sims.append(v["chembl_prior_fcfp4_similarity"].values)
            valid_s_eps.append(ep)

    fig3, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig3.suptitle(f"#{exp_id} Distribution Evolution  (sampled epochs)",
                  fontsize=FS_SUPTITLE, fontweight="bold", y=1.02)

    # 3a: ECFP4 similarity boxplot
    ax = axes[0]
    bp1 = ax.boxplot(ecfp4_sims, labels=valid_s_eps, patch_artist=True, widths=0.6,
                     medianprops={"color": "black", "linewidth": 1.5})
    for patch, color in zip(bp1["boxes"], plt.cm.Purples(np.linspace(0.3, 0.9, len(valid_s_eps)))):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax.set_xlabel("Epoch", fontsize=FS); ax.set_ylabel("ECFP4 Similarity", fontsize=FS)
    ax.set_title("(a) ECFP4 Similarity", fontsize=FS_TITLE, fontweight="bold")
    ax.grid(True, alpha=0.3); ax.tick_params(labelsize=FS_TICK)

    # 3b: FCFP4 similarity boxplot
    ax = axes[1]
    bp2 = ax.boxplot(fcfp4_sims, labels=valid_s_eps, patch_artist=True, widths=0.6,
                     medianprops={"color": "black", "linewidth": 1.5})
    for patch, color in zip(bp2["boxes"], plt.cm.Oranges(np.linspace(0.3, 0.9, len(valid_s_eps)))):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax.set_xlabel("Epoch", fontsize=FS); ax.set_ylabel("FCFP4 Similarity", fontsize=FS)
    ax.set_title("(b) FCFP4 Similarity", fontsize=FS_TITLE, fontweight="bold")
    ax.grid(True, alpha=0.3); ax.tick_params(labelsize=FS_TICK)

    plt.tight_layout()
    fig3.savefig(ROOT / "distribution_evolution.png", dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print("[OK] distribution_evolution.png")

    # =======================================================================
    # FIGURE 4: Key Metric Summary Table
    # =======================================================================
    fig4, ax = plt.subplots(figsize=(12, 5))
    ax.axis("off")

    if cfg["has_docking"]:
        rows = [
            ["Metric", "Epoch 1", f"Epoch {NE}", "Change"],
            ["PPO Loss", f"{df['avg_loss'].iloc[0]:.4f}", f"{df['avg_loss'].iloc[-1]:.4f}",
             f"{df['avg_loss'].iloc[-1] - df['avg_loss'].iloc[0]:.4f}"],
            ["RDKit Valid", f"{df['rdkit_valid_ratio'].iloc[0]:.2%}", f"{df['rdkit_valid_ratio'].iloc[-1]:.2%}",
             f"+{(df['rdkit_valid_ratio'].iloc[-1] - df['rdkit_valid_ratio'].iloc[0])*100:.1f}pp"],
            ["Avg Docking", f"{df['avg_docking'].iloc[0]:.2f}", f"{df['avg_docking'].iloc[-1]:.2f}",
             f"{df['avg_docking'].iloc[-1] - df['avg_docking'].iloc[0]:+.2f}"],
            ["ECFP4 Sim", f"{df['avg_chembl_prior_ecfp4_similarity'].iloc[0]:.4f}",
             f"{df['avg_chembl_prior_ecfp4_similarity'].iloc[-1]:.4f}",
             f"{df['avg_chembl_prior_ecfp4_similarity'].iloc[-1] - df['avg_chembl_prior_ecfp4_similarity'].iloc[0]:+.4f}"],
            ["FCFP4 Sim", f"{df['avg_chembl_prior_fcfp4_similarity'].iloc[0]:.4f}",
             f"{df['avg_chembl_prior_fcfp4_similarity'].iloc[-1]:.4f}",
             f"{df['avg_chembl_prior_fcfp4_similarity'].iloc[-1] - df['avg_chembl_prior_fcfp4_similarity'].iloc[0]:+.4f}"],
            ["Valid Mols", f"{int(df['success'].iloc[0])}/{TOTAL}",
             f"{int(df['success'].iloc[-1])}/{TOTAL}",
             f"+{int(df['success'].iloc[-1] - df['success'].iloc[0])}"],
            ["Fragmented", f"{int(df['fragmented_molecule'].iloc[0])}",
             f"{int(df['fragmented_molecule'].iloc[-1])}",
             f"{int(df['fragmented_molecule'].iloc[-1] - df['fragmented_molecule'].iloc[0]):+d}"],
            ["Sanitize Fail", f"{int(df['sanitize_failed'].iloc[0])}",
             f"{int(df['sanitize_failed'].iloc[-1])}",
             f"{int(df['sanitize_failed'].iloc[-1] - df['sanitize_failed'].iloc[0]):+d}"],
        ]
    else:
        rows = [
            ["Metric", "Epoch 1", f"Epoch {NE}", "Change"],
            ["PPO Loss", f"{df['avg_loss'].iloc[0]:.4f}", f"{df['avg_loss'].iloc[-1]:.4f}",
             f"{df['avg_loss'].iloc[-1] - df['avg_loss'].iloc[0]:.4f}"],
            ["RDKit Valid", f"{df['rdkit_valid_ratio'].iloc[0]:.2%}", f"{df['rdkit_valid_ratio'].iloc[-1]:.2%}",
             f"+{(df['rdkit_valid_ratio'].iloc[-1] - df['rdkit_valid_ratio'].iloc[0])*100:.1f}pp"],
            ["ECFP4 Sim", f"{df['avg_chembl_prior_ecfp4_similarity'].iloc[0]:.4f}",
             f"{df['avg_chembl_prior_ecfp4_similarity'].iloc[-1]:.4f}",
             f"{df['avg_chembl_prior_ecfp4_similarity'].iloc[-1] - df['avg_chembl_prior_ecfp4_similarity'].iloc[0]:+.4f}"],
            ["FCFP4 Sim", f"{df['avg_chembl_prior_fcfp4_similarity'].iloc[0]:.4f}",
             f"{df['avg_chembl_prior_fcfp4_similarity'].iloc[-1]:.4f}",
             f"{df['avg_chembl_prior_fcfp4_similarity'].iloc[-1] - df['avg_chembl_prior_fcfp4_similarity'].iloc[0]:+.4f}"],
            ["Valid Mols", f"{int(df['success'].iloc[0])}/{TOTAL}",
             f"{int(df['success'].iloc[-1])}/{TOTAL}",
             f"+{int(df['success'].iloc[-1] - df['success'].iloc[0])}"],
            ["Fragmented", f"{int(df['fragmented_molecule'].iloc[0])}",
             f"{int(df['fragmented_molecule'].iloc[-1])}",
             f"{int(df['fragmented_molecule'].iloc[-1] - df['fragmented_molecule'].iloc[0]):+d}"],
            ["Sanitize Fail", f"{int(df['sanitize_failed'].iloc[0])}",
             f"{int(df['sanitize_failed'].iloc[-1])}",
             f"{int(df['sanitize_failed'].iloc[-1] - df['sanitize_failed'].iloc[0]):+d}"],
        ]

    table = ax.table(cellText=rows, loc="center", cellLoc="center", colWidths=[0.2, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False); table.set_fontsize(14); table.scale(1, 1.8)
    for i in range(len(rows[0])):
        cell = table[0, i]; cell.set_facecolor("#4472C4")
        cell.set_text_props(color="white", fontweight="bold")
    for i in range(1, len(rows)):
        cell = table[i, 3]; val = rows[i][3]
        if val.startswith("+"): cell.set_facecolor("#C6EFCE")
        elif val.startswith("-"): cell.set_facecolor("#FFC7CE")
    ax.set_title(f"#{exp_id} {cfg['short']}  |  Epoch 1 → Epoch {NE}",
                 fontsize=FS_SUPTITLE, fontweight="bold", pad=20)
    plt.tight_layout()
    fig4.savefig(ROOT / "summary_table.png", dpi=150, bbox_inches="tight")
    plt.close(fig4)
    print("[OK] summary_table.png")

    # =======================================================================
    # FIGURE 5 (EXP #7 only): Collapse Analysis - Docking + Validity over epochs
    # =======================================================================
    if exp_id == 7:
        fig5, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig5.suptitle("#7 Collapse Analysis  (50 epochs, diversity caused collapse)",
                      fontsize=FS_SUPTITLE, fontweight="bold", y=1.02)

        # 5a: Validity + Docking over epochs (dual axis)
        ax = axes[0, 0]
        ax2 = ax.twinx()
        l1 = ax.plot(epochs, df["rdkit_valid_ratio"] * 100, color=C_GREEN, linewidth=2, marker="o", markersize=4, label="Validity %")
        l2 = ax2.plot(epochs, df["avg_docking"], color=C_RED, linewidth=2, marker="s", markersize=4, label="Docking")
        ax.set_xlabel("Epoch", fontsize=FS)
        ax.set_ylabel("Validity (%)", fontsize=FS, color=C_GREEN)
        ax2.set_ylabel("Avg Docking", fontsize=FS, color=C_RED)
        ax.set_title("(a) Validity & Docking Trend", fontsize=FS_TITLE, fontweight="bold")
        ax.grid(True, alpha=0.3); ax.tick_params(labelsize=FS_TICK); ax2.tick_params(labelsize=FS_TICK)
        ax.legend(l1 + l2, [l.get_label() for l in l1 + l2], fontsize=FS_LEGEND)

        # 5b: ECFP4/FCFP4 Similarity over epochs
        ax = axes[0, 1]
        ax.plot(epochs, df["avg_chembl_prior_ecfp4_similarity"], color=C_PURPLE, linewidth=2, label="ECFP4 Sim")
        ax.plot(epochs, df["avg_chembl_prior_fcfp4_similarity"], color=C_ORANGE, linewidth=2, label="FCFP4 Sim")
        ax.set_xlabel("Epoch", fontsize=FS); ax.set_ylabel("Similarity", fontsize=FS)
        ax.set_title("(b) ECFP4 & FCFP4 Similarity", fontsize=FS_TITLE, fontweight="bold")
        ax.legend(fontsize=FS_LEGEND); ax.grid(True, alpha=0.3); ax.tick_params(labelsize=FS_TICK)

        # 5c: Failure counts (abs values)
        ax = axes[1, 0]
        ax.plot(epochs, df["success"], color=C_GREEN, linewidth=2, label="Success")
        ax.plot(epochs, df["fragmented_molecule"], color=C_ORANGE, linewidth=2, label="Fragmented")
        ax.plot(epochs, df["sanitize_failed"], color=C_RED, linewidth=2, label="Sanitize Fail")
        ax.set_xlabel("Epoch", fontsize=FS); ax.set_ylabel("Count", fontsize=FS)
        ax.set_title("(c) Molecule Status Counts", fontsize=FS_TITLE, fontweight="bold")
        ax.legend(fontsize=FS_LEGEND); ax.grid(True, alpha=0.3); ax.tick_params(labelsize=FS_TICK)

        # 5d: Epoch 1 vs 50 ECFP4 Sim comparison (histogram overlay)
        ax = axes[1, 1]
        mol1 = pd.read_csv(EPOCH_DIR / "epoch_0001" / "molecules.csv")
        valid1 = mol1[mol1["rdkit_valid"] == 1]
        ax.hist(valid1["chembl_prior_ecfp4_similarity"], bins=20, alpha=0.6, color=C_BLUE, label=f"Epoch 1 (n={len(valid1)})")
        ax.hist(valid_last["chembl_prior_ecfp4_similarity"], bins=20, alpha=0.6, color=C_RED, label=f"Epoch 50 (n={n_valid})")
        ax.set_xlabel("ECFP4 Similarity", fontsize=FS); ax.set_ylabel("Count", fontsize=FS)
        ax.set_title("(d) ECFP4 Sim: Epoch 1 vs 50", fontsize=FS_TITLE, fontweight="bold")
        ax.legend(fontsize=FS_LEGEND); ax.grid(True, alpha=0.3); ax.tick_params(labelsize=FS_TICK)

        plt.tight_layout()
        fig5.savefig(ROOT / "collapse_analysis.png", dpi=150, bbox_inches="tight")
        plt.close(fig5)
        print("[OK] collapse_analysis.png")

    # =======================================================================
    # Console summary
    # =======================================================================
    print()
    print("=" * 70)
    print(f"  #{exp_id} {cfg['short']}")
    print("=" * 70)
    print(f"  Run     : {cfg['root']}")
    print(f"  Epochs  : {NE} / {TOTAL} molecules/epoch, {cfg['name']}")
    print()
    print(f"  {'Metric':<35s} {'Epoch 1':>10s} {'Epoch '+str(NE):>10s} {'Change':>10s}")
    print("  " + "-"*35 + " " + "-"*10 + " " + "-"*10 + " " + "-"*10)
    print(f"  {'PPO Loss':<35s} {df['avg_loss'].iloc[0]:>10.4f} {df['avg_loss'].iloc[-1]:>10.4f} {df['avg_loss'].iloc[-1]-df['avg_loss'].iloc[0]:>+10.4f}")
    print(f"  {'RDKit Validity':<35s} {df['rdkit_valid_ratio'].iloc[0]*100:>9.1f}% {df['rdkit_valid_ratio'].iloc[-1]*100:>9.1f}% {(df['rdkit_valid_ratio'].iloc[-1]-df['rdkit_valid_ratio'].iloc[0])*100:>+9.1f}pp")
    print(f"  {'ECFP4 Similarity':<35s} {df['avg_chembl_prior_ecfp4_similarity'].iloc[0]:>10.4f} {df['avg_chembl_prior_ecfp4_similarity'].iloc[-1]:>10.4f} {df['avg_chembl_prior_ecfp4_similarity'].iloc[-1]-df['avg_chembl_prior_ecfp4_similarity'].iloc[0]:>+10.4f}")
    print(f"  {'FCFP4 Similarity':<35s} {df['avg_chembl_prior_fcfp4_similarity'].iloc[0]:>10.4f} {df['avg_chembl_prior_fcfp4_similarity'].iloc[-1]:>10.4f} {df['avg_chembl_prior_fcfp4_similarity'].iloc[-1]-df['avg_chembl_prior_fcfp4_similarity'].iloc[0]:>+10.4f}")
    if cfg["has_docking"]:
        print(f"  {'Avg Docking':<35s} {df['avg_docking'].iloc[0]:>10.2f} {df['avg_docking'].iloc[-1]:>10.2f} {df['avg_docking'].iloc[-1]-df['avg_docking'].iloc[0]:>+10.2f}")
    print(f"  {'Valid Molecules':<35s} {int(df['success'].iloc[0]):>5d}/{TOTAL} {int(df['success'].iloc[-1]):>5d}/{TOTAL} {int(df['success'].iloc[-1]-df['success'].iloc[0]):>+6d}")
    print(f"  {'Fragmented':<35s} {int(df['fragmented_molecule'].iloc[0]):>10d} {int(df['fragmented_molecule'].iloc[-1]):>10d} {int(df['fragmented_molecule'].iloc[-1]-df['fragmented_molecule'].iloc[0]):>+10d}")
    print(f"  {'Sanitize Failed':<35s} {int(df['sanitize_failed'].iloc[0]):>10d} {int(df['sanitize_failed'].iloc[-1]):>10d} {int(df['sanitize_failed'].iloc[-1]-df['sanitize_failed'].iloc[0]):>+10d}")
    print()
    print(f"  Files saved to: {ROOT}/")
    print(f"    - training_curves.png")
    print(f"    - epoch{NE}_molecule_analysis.png")
    print(f"    - distribution_evolution.png")
    print(f"    - summary_table.png")
    if exp_id == 7:
        print(f"    - collapse_analysis.png")
    print("=" * 70)

if __name__ == "__main__":
    main()
