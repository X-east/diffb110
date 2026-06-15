#!/usr/bin/env python3
"""
比较 5 个数据集的核心分子属性：
  ChEMBL (155万) | Baseline (5827) | #5 Epoch18 | #6 Epoch20 | #7 Epoch50

用法：python analysis/compare_all_properties.py
"""

import csv, gzip, os, sys, math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch
import numpy as np

# CJK font
for _fp in [
    '/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc',
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
]:
    if os.path.exists(_fp):
        try:
            fm.fontManager.addfont(_fp)
            _fn = fm.FontProperties(fname=_fp).get_name()
            plt.rcParams['font.sans-serif'] = [_fn] + plt.rcParams.get('font.sans-serif', [])
            break
        except Exception:
            pass
plt.rcParams['axes.unicode_minus'] = False

# RDKit
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, Crippen, rdMolDescriptors
# SA Score
try:
    from rdkit.Contrib.SA_Score import sascorer
except ImportError:
    sascorer = None

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CHEMBL_PROP = "/home/md666/data/disk6/DiffB110/data/chembl/chembl_36_properties.csv.gz"
BASELINE_CSV = "/home/md666/data/disk6/DiffB110/OUT/260407_generate_ligands_10000/guacamol_eval/260407generated_10000_properties.csv"
EXP5_CSV = "/home/md666/data/disk6/DiffB110/OUT/6gpu-chembl-only_20260521_112200/epochs/epoch_0018/molecules.csv"
EXP6_CSV = "/home/md666/data/disk6/DiffB110/OUT/6gpu-chembl-only_20260521_154957/epochs/epoch_0020/molecules.csv"
EXP7_CSV = "/home/md666/data/disk6/DiffB110/OUT/7gpu-dock-ecfp4-fcfp4_20260521_212927/epochs/epoch_0050/molecules.csv"
OUTPUT_DIR = "/home/md666/data/disk6/DiffB110/analysis/"

# Property mapping: (display_name, col_baseline, col_chembl, compute_fn)
# compute_fn(smiles) → float
PROPERTIES = [
    ("MW",        "MolWt",   "mol_wt",   lambda m: Descriptors.ExactMolWt(m)),
    ("LogP",      "LogP",    "mol_log_p", lambda m: Crippen.MolLogP(m)),
    ("TPSA",      "TPSA",    "tpsa",     lambda m: Descriptors.TPSA(m)),
    ("HBA",       "HBA",     "num_h_acceptors", lambda m: rdMolDescriptors.CalcNumHBA(m)),
    ("HBD",       "HBD",     "num_h_donors",    lambda m: rdMolDescriptors.CalcNumHBD(m)),
    ("RotBonds",  "RotBonds","num_rotatable_bonds", lambda m: rdMolDescriptors.CalcNumRotatableBonds(m)),
    ("Rings",     "NumRings","num_rings", lambda m: rdMolDescriptors.CalcNumRings(m)),
    ("QED",       "QED",     None,       lambda m: QED.qed(m)),
    ("SA_Score",  "SA_Score",None,       lambda m: sascorer.calculateScore(m) / 10.0 if sascorer else float("nan")),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def compute_props_for_smiles(smiles_list):
    """Compute all PROPERTIES for a list of SMILES. Return dict[name→list]."""
    results = {name: [] for name, _, _, _ in PROPERTIES}
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                for name, _, _, _ in PROPERTIES:
                    results[name].append(float("nan"))
                continue
            for name, _, _, fn in PROPERTIES:
                try:
                    val = fn(mol)
                except Exception:
                    val = float("nan")
                results[name].append(val)
        except Exception:
            for name, _, _, _ in PROPERTIES:
                results[name].append(float("nan"))
    return results


def stats_str(arr):
    """return 'mean±std' string"""
    a = np.array(arr, dtype=float)
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return "N/A"
    return f"{np.mean(a):.3f}±{np.std(a):.3f}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading ChEMBL (sampling 100k)...")
    chembl_data = {name: [] for name, _, _, _ in PROPERTIES}
    chembl_cols = {name: col for name, _, col, _ in PROPERTIES}
    chembl_total = 0
    with gzip.open(CHEMBL_PROP, "rt") as f:
        reader = csv.DictReader(f)
        for row in reader:
            chembl_total += 1
            if chembl_total > 100000:
                break
            for name, _, col_name, _ in PROPERTIES:
                if col_name is None:
                    continue
                val = row.get(col_name, "").strip()
                try:
                    chembl_data[name].append(float(val))
                except (ValueError, TypeError):
                    chembl_data[name].append(float("nan"))
    print(f"  ChEMBL: sampled {len(chembl_data['MW'])} molecules")

    print("Loading Baseline (5827)...")
    base_data = {name: [] for name, _, _, _ in PROPERTIES}
    with open(BASELINE_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for name, col_name, _, _ in PROPERTIES:
                if col_name is None:
                    continue
                val = row.get(col_name, "").strip()
                try:
                    base_data[name].append(float(val))
                except (ValueError, TypeError):
                    base_data[name].append(float("nan"))
    print(f"  Baseline: {len(base_data['MW'])} molecules")

    print("Loading experiment molecules & computing properties...")
    exp_configs = [
        ("#5 Epoch18", EXP5_CSV),
        ("#6 Epoch20", EXP6_CSV),
        ("#7 Epoch50", EXP7_CSV),
    ]
    exp_results = {}
    for label, csv_path in exp_configs:
        smiles = []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("rdkit_valid", "1") == "1":
                    smi = row.get("smiles", "").strip()
                    if smi:
                        smiles.append(smi)
        print(f"  {label}: {len(smiles)} valid molecules")
        exp_results[label] = compute_props_for_smiles(smiles)

    # -------------------------------------------------------------------
    # Build table
    # -------------------------------------------------------------------
    datasets = [
        ("ChEMBL\n155万",  chembl_data),
        ("Baseline\n5827", base_data),
        ("#5\nEpoch18",    exp_results["#5 Epoch18"]),
        ("#6\nEpoch20",    exp_results["#6 Epoch20"]),
        ("#7\nEpoch50",    exp_results["#7 Epoch50"]),
    ]

    # Table data
    row_labels = [name for name, _, _, _ in PROPERTIES]

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.axis("off")

    # Header
    col_labels = ["属性"] + [d[0] for d in datasets]
    cell_text = []
    for rname in row_labels:
        row = [rname]
        for _, data in datasets:
            vals = data.get(rname, [])
            arr = np.array(vals, dtype=float)
            arr = arr[~np.isnan(arr)]
            if len(arr) == 0:
                row.append("N/A")
            else:
                row.append(f"{np.mean(arr):.3f}±{np.std(arr):.3f}")
        cell_text.append(row)

    table = ax.table(cellText=cell_text, colLabels=col_labels,
                     loc="center", cellLoc="center",
                     colWidths=[0.10, 0.18, 0.18, 0.18, 0.18, 0.18])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.7)

    # Style header
    header_color = "#2C3E50"
    for i in range(len(col_labels)):
        cell = table[0, i]
        cell.set_facecolor(header_color)
        cell.set_text_props(color="white", fontweight="bold", fontsize=11)

    # Alternating row colors
    for row_idx in range(1, len(cell_text) + 1):
        for col_idx in range(len(col_labels)):
            cell = table[row_idx, col_idx]
            bg = "#F2F3F4" if row_idx % 2 == 0 else "#FFFFFF"
            cell.set_facecolor(bg)
            cell.set_edgecolor("#BDC3C7")
            cell.set_linewidth(0.5)
            # Highlight the best per row
            if col_idx >= 1 and cell_text[row_idx - 1][col_idx] != "N/A":
                vals = [float(cell_text[row_idx - 1][c].split("±")[0])
                        for c in range(1, len(col_labels))
                        if cell_text[row_idx - 1][c] != "N/A"]
                if vals:
                    this_val = float(cell_text[row_idx - 1][col_idx].split("±")[0])
                    best = min(vals) if row_labels[row_idx - 1] in ("MW", "LogP", "SA_Score") else max(vals)
                    if abs(this_val - best) < 1e-6:
                        cell.set_facecolor("#D5F5E3")

    ax.set_title("Core Molecular Properties | ChEMBL vs Baseline vs DDPO Training",
                 fontsize=15, fontweight="bold", pad=20)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "property_comparison_table.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n[OK] Table saved: {out_path}")

    # Also print CSV
    csv_path = os.path.join(OUTPUT_DIR, "property_comparison_table.csv")
    with open(csv_path, "w") as f:
        f.write(",".join(col_labels) + "\n")
        for row in cell_text:
            f.write(",".join(row) + "\n")
    print(f"[OK] CSV saved: {csv_path}")

    # Console summary
    print("\n" + "=" * 80)
    print(f"{'属性':<12}", end="")
    for dname, _ in datasets:
        print(f"{dname.replace(chr(10), ' '):>18}", end="")
    print()
    print("-" * 80)
    for rname in row_labels:
        print(f"{rname:<12}", end="")
        for _, data in datasets:
            vals = data.get(rname, [])
            arr = np.array(vals, dtype=float)
            arr = arr[~np.isnan(arr)]
            if len(arr) == 0:
                print(f"{'N/A':>18}", end="")
            else:
                print(f"{np.mean(arr):>9.3f}±{np.std(arr):>7.3f}", end="")
        print()
    print("=" * 80)


if __name__ == "__main__":
    main()
