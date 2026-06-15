#!/usr/bin/env python3
"""
ChEMBL 155万分子全面性质分析：直方图、相关性热力图、散点图、完整统计表。

用法：
  python analysis/chembl_full_analysis.py [--sample N]

选项：
  --sample N    只采样 N 万分子（默认全量 155 万）
  --no-plots    只出统计表，跳过画图
  --output-dir  输出目录（默认 analysis/chembl_full_analysis/）

产出：
  - statistics.csv       : 33 个描述子的完整统计（P1/P5/P50/P99 等）
  - histogram_all.png    : 所有描述子的直方图面板
  - histogram_<name>.png : 单个描述子的高清直方图
  - correlation_heatmap.png : 相关性矩阵热力图
  - scatter_matrix.png   : 6 个核心属性的散点图矩阵
"""

import argparse
import csv
import gzip
import os
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

warnings.filterwarnings("ignore")

# ── CJK Font ──
for _fp in [
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJKsc-VF.otf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
]:
    if os.path.exists(_fp):
        try:
            fm.fontManager.addfont(_fp)
            _fn = fm.FontProperties(fname=_fp).get_name()
            plt.rcParams["font.sans-serif"] = [_fn] + plt.rcParams.get("font.sans-serif", [])
            break
        except Exception:
            pass
plt.rcParams["axes.unicode_minus"] = False

# ── Paths ──
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CHEMBL = SCRIPT_DIR.parent / "data" / "chembl" / "chembl_36_properties.csv.gz"
DEFAULT_OUTDIR = SCRIPT_DIR / "chembl_full_analysis"

# ── Descriptor groups for organized visualization ──
GROUP_MAIN = [
    "mol_wt", "mol_log_p", "tpsa", "heavy_atoms", "num_heteroatoms",
    "num_carbon_atoms", "carbon_fraction", "fraction_csp3",
]
GROUP_HBOND = [
    "num_h_acceptors", "num_h_donors", "num_rotatable_bonds",
]
GROUP_RINGS = [
    "num_rings", "num_aromatic_rings", "num_aliphatic_rings",
    "num_saturated_rings", "num_aromatic_heterocycles", "num_saturated_heterocycles",
]
GROUP_TOPO = [
    "chi0v", "chi1v", "chi0n", "chi1n",
    "kappa1", "kappa2", "kappa3", "hall_kier_alpha",
]
GROUP_OTHER = [
    "exact_mol_wt", "num_valence_electrons", "num_radical_electrons",
    "num_spiro_atoms", "num_bridgehead_atoms", "formal_charge",
]
GROUP_LIPINSKI = [
    "lipinski_violations", "lipinski_heavy_violations",
]

ALL_GROUPS = [
    ("Physicochemical", GROUP_MAIN),
    ("H-Bond & Flexibility", GROUP_HBOND),
    ("Rings & Cycles", GROUP_RINGS),
    ("Topological Indices", GROUP_TOPO),
    ("Other Structural", GROUP_OTHER),
    ("Lipinski Violations", GROUP_LIPINSKI),
]

# Flat list of all descriptors (ordered by groups)
ORDERED_DESCRIPTORS = []
for _gname, _group in ALL_GROUPS:
    ORDERED_DESCRIPTORS.extend(_group)

# Friendly labels
DESCRIPTOR_LABELS = {
    "mol_wt": "Molecular Weight",
    "mol_log_p": "LogP (Crippen)",
    "tpsa": "TPSA",
    "heavy_atoms": "Heavy Atoms",
    "num_heteroatoms": "Heteroatoms",
    "num_carbon_atoms": "Carbon Atoms",
    "carbon_fraction": "Carbon Fraction",
    "fraction_csp3": "Fraction Csp3",
    "num_h_acceptors": "H-Bond Acceptors",
    "num_h_donors": "H-Bond Donors",
    "num_rotatable_bonds": "Rotatable Bonds",
    "num_rings": "Total Rings",
    "num_aromatic_rings": "Aromatic Rings",
    "num_aliphatic_rings": "Aliphatic Rings",
    "num_saturated_rings": "Saturated Rings",
    "num_aromatic_heterocycles": "Aromatic Heterocycles",
    "num_saturated_heterocycles": "Saturated Heterocycles",
    "chi0v": "Chi0v",
    "chi1v": "Chi1v",
    "chi0n": "Chi0n",
    "chi1n": "Chi1n",
    "kappa1": "Kappa1",
    "kappa2": "Kappa2",
    "kappa3": "Kappa3",
    "hall_kier_alpha": "Hall-Kier Alpha",
    "exact_mol_wt": "Exact MW",
    "num_valence_electrons": "Valence Electrons",
    "num_radical_electrons": "Radical Electrons",
    "num_spiro_atoms": "Spiro Atoms",
    "num_bridgehead_atoms": "Bridgehead Atoms",
    "formal_charge": "Formal Charge",
    "lipinski_violations": "Lipinski Violations",
    "lipinski_heavy_violations": "Lipinski Heavy Violations",
}


# ══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ══════════════════════════════════════════════════════════════════════════════
def load_data(csv_path: str, max_rows: int | None = None):
    """Load ChEMBL properties CSV into numpy structured array."""
    print(f"Loading: {csv_path}")
    rows = []
    with gzip.open(csv_path, "rt") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_rows and i >= max_rows:
                break
            parsed = {}
            for col in ORDERED_DESCRIPTORS:
                try:
                    parsed[col] = float(row[col])
                except (KeyError, ValueError):
                    parsed[col] = float("nan")
            rows.append(tuple(parsed[col] for col in ORDERED_DESCRIPTORS))
            if (i + 1) % 500000 == 0:
                print(f"  ... {i + 1:,} rows loaded")
    dtype = [(name, np.float64) for name in ORDERED_DESCRIPTORS]
    data = np.array(rows, dtype=dtype)
    print(f"Loaded: {len(data):,} molecules × {len(ORDERED_DESCRIPTORS)} descriptors")
    return data


# ══════════════════════════════════════════════════════════════════════════════
# Statistics
# ══════════════════════════════════════════════════════════════════════════════
def compute_statistics(data: np.ndarray) -> list[dict]:
    """Compute comprehensive statistics for each descriptor."""
    results = []
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    for name in ORDERED_DESCRIPTORS:
        arr = data[name]
        valid = arr[~np.isnan(arr)]
        n_valid = len(valid)
        n_nan = len(arr) - n_valid
        if n_valid == 0:
            results.append({"descriptor": name, "n_valid": 0, "n_nan": n_nan})
            continue
        p_vals = np.percentile(valid, percentiles)
        stats = {
            "descriptor": name,
            "label": DESCRIPTOR_LABELS.get(name, name),
            "count_total": len(arr),
            "n_valid": n_valid,
            "n_nan": n_nan,
            "pct_valid": n_valid / len(arr) * 100,
            "min": float(np.min(valid)),
            "max": float(np.max(valid)),
            "mean": float(np.mean(valid)),
            "median": float(np.median(valid)),
            "std": float(np.std(valid, ddof=1)),
            "var": float(np.var(valid, ddof=1)),
            "skewness": float(_skew(valid)),
            "kurtosis": float(_kurt(valid)),
            "range": float(np.max(valid) - np.min(valid)),
            "iqr": float(np.percentile(valid, 75) - np.percentile(valid, 25)),
        }
        for p, pct in zip(p_vals, percentiles):
            stats[f"p{pct}"] = float(p)
        results.append(stats)
    return results


def _skew(arr: np.ndarray) -> float:
    """Sample skewness."""
    n = len(arr)
    if n < 3:
        return float("nan")
    m = np.mean(arr)
    s = np.std(arr, ddof=1)
    if s == 0:
        return 0.0
    return float((np.sum((arr - m) ** 3) / n) / (s**3))


def _kurt(arr: np.ndarray) -> float:
    """Excess kurtosis (Fisher)."""
    n = len(arr)
    if n < 4:
        return float("nan")
    m = np.mean(arr)
    s = np.std(arr, ddof=1)
    if s == 0:
        return float("nan")
    k = float((np.sum((arr - m) ** 4) / n) / (s**4))
    return k - 3.0


def save_statistics(stats_list: list[dict], outdir: Path):
    """Save statistics to CSV."""
    csv_path = outdir / "statistics.csv"
    if not stats_list:
        return
    keys = list(stats_list[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(stats_list)
    print(f"[OK] Statistics saved: {csv_path}")


def print_statistics_summary(stats_list: list[dict]):
    """Print a compact summary table."""
    header = f"{'Descriptor':<30} {'Mean':>10} {'Std':>8} {'Min':>8} {'Max':>8} {'P5':>8} {'P50':>8} {'P95':>8} {'Skew':>7}"
    sep = "-" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for s in stats_list:
        if s["n_valid"] == 0:
            continue
        print(
            f"{s['label']:<30} "
            f"{s['mean']:10.3f} "
            f"{s['std']:8.3f} "
            f"{s['min']:8.3f} "
            f"{s['max']:8.3f} "
            f"{s.get('p5',0):8.3f} "
            f"{s.get('p50',0):8.3f} "
            f"{s.get('p95',0):8.3f} "
            f"{s['skewness']:7.2f}"
        )
    print(sep)


# ══════════════════════════════════════════════════════════════════════════════
# Histograms
# ══════════════════════════════════════════════════════════════════════════════
def plot_all_histograms(data: np.ndarray, outdir: Path, sample_for_plot: int = 200000):
    """Generate histogram panel + individual histograms."""
    outdir.mkdir(parents=True, exist_ok=True)
    n_total = len(data)
    if n_total > sample_for_plot:
        rng = np.random.default_rng(42)
        idx = rng.choice(n_total, sample_for_plot, replace=False)
        plot_data = data[idx]
        title_suffix = f" (sampled {sample_for_plot:,} / {n_total:,})"
    else:
        plot_data = data
        title_suffix = ""

    # ── Panel: all descriptors ──
    n_descriptors = len(ORDERED_DESCRIPTORS)
    n_cols = 6
    n_rows = int(np.ceil(n_descriptors / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.2, n_rows * 2.8))
    axes_flat = axes.flatten()

    for i, name in enumerate(ORDERED_DESCRIPTORS):
        ax = axes_flat[i]
        arr = plot_data[name]
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha="center", va="center")
            ax.set_title(name, fontsize=8)
            continue
        # Auto bins
        n_bins = min(80, max(20, int(np.sqrt(len(arr)))))
        ax.hist(arr, bins=n_bins, color="#3498DB", edgecolor="#2980B9", linewidth=0.3, alpha=0.85)
        # Vertical lines for P5, P50, P95
        for p, color, ls in [(5, "#E74C3C", "--"), (50, "#2C3E50", "-"), (95, "#27AE60", "--")]:
            pval = np.percentile(arr, p)
            ax.axvline(pval, color=color, linestyle=ls, linewidth=1.0, alpha=0.7)
        ax.set_title(DESCRIPTOR_LABELS.get(name, name), fontsize=7.5, fontweight="bold")
        ax.tick_params(labelsize=6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Hide empty subplots
    for j in range(n_descriptors, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f"ChEMBL 1.55M Molecular Descriptor Distributions{title_suffix}",
                 fontsize=14, fontweight="bold", y=1.005)
    plt.tight_layout()
    panel_path = outdir / "histogram_all.png"
    fig.savefig(panel_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Panel histogram saved: {panel_path}")

    # ── Individual high-res histograms per descriptor ──
    for name in ORDERED_DESCRIPTORS:
        arr = data[name]
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        n_bins = min(100, max(30, int(np.sqrt(len(arr)))))
        ax.hist(arr, bins=n_bins, color="#3498DB", edgecolor="#2980B9", linewidth=0.3, alpha=0.85)
        for p, color, ls, lbl in [
            (5, "#E74C3C", "--", "P5"),
            (25, "#F39C12", ":", "P25"),
            (50, "#2C3E50", "-", "P50"),
            (75, "#F39C12", ":", "P75"),
            (95, "#27AE60", "--", "P95"),
        ]:
            pval = np.percentile(arr, p)
            ax.axvline(pval, color=color, linestyle=ls, linewidth=1.5, alpha=0.7, label=f"{lbl}={pval:.2f}")
        mean_val = np.mean(arr)
        ax.axvline(mean_val, color="#9B59B6", linestyle="-", linewidth=2.0, alpha=0.9,
                   label=f"Mean={mean_val:.2f}")
        ax.legend(fontsize=8, loc="upper right")
        ax.set_title(f"{DESCRIPTOR_LABELS.get(name, name)} | ChEMBL (N={len(arr):,})",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel(DESCRIPTOR_LABELS.get(name, name), fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        single_path = outdir / f"histogram_{name}.png"
        fig.savefig(single_path, dpi=150, bbox_inches="tight")
        plt.close()
    print(f"[OK] {n_descriptors} individual histograms saved to {outdir}/histogram_*.png")


# ══════════════════════════════════════════════════════════════════════════════
# Correlation Heatmap
# ══════════════════════════════════════════════════════════════════════════════
def plot_correlation_heatmap(data: np.ndarray, outdir: Path, sample_for_corr: int = 100000):
    """Correlation heatmap of all descriptors."""
    n_total = len(data)
    if n_total > sample_for_corr:
        rng = np.random.default_rng(42)
        idx = rng.choice(n_total, sample_for_corr, replace=False)
        corr_data = data[idx]
    else:
        corr_data = data

    # Build 2D array: [n_samples, n_descriptors]
    n_desc = len(ORDERED_DESCRIPTORS)
    matrix = np.column_stack([corr_data[name] for name in ORDERED_DESCRIPTORS])
    # Drop rows with any NaN
    valid_mask = ~np.isnan(matrix).any(axis=1)
    matrix = matrix[valid_mask]

    # Correlation
    corr = np.corrcoef(matrix, rowvar=False)

    # Labels
    labels = [DESCRIPTOR_LABELS.get(n, n) for n in ORDERED_DESCRIPTORS]

    fig, ax = plt.subplots(figsize=(18, 15))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(n_desc))
    ax.set_yticks(range(n_desc))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)

    # Annotate high correlations (|r| > 0.5)
    for i in range(n_desc):
        for j in range(n_desc):
            if i == j:
                continue
            val = corr[i, j]
            if abs(val) > 0.5:
                text_color = "white" if abs(val) > 0.75 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=5.5, color=text_color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Pearson Correlation", fontsize=10)

    ax.set_title("ChEMBL 1.55M — Descriptor Correlation Matrix\n(|r|>0.5 annotated)",
                 fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    heatmap_path = outdir / "correlation_heatmap.png"
    fig.savefig(heatmap_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Correlation heatmap saved: {heatmap_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Scatter Matrix
# ══════════════════════════════════════════════════════════════════════════════
SCATTER_PAIRS = [
    ("mol_log_p", "mol_wt", "LogP", "MW"),
    ("mol_log_p", "tpsa", "LogP", "TPSA"),
    ("mol_wt", "tpsa", "MW", "TPSA"),
    ("mol_log_p", "num_rotatable_bonds", "LogP", "RotBonds"),
    ("mol_wt", "num_h_acceptors", "MW", "HBA"),
    ("num_aromatic_rings", "num_heteroatoms", "Aromatic Rings", "Heteroatoms"),
    ("carbon_fraction", "mol_log_p", "Carbon Fraction", "LogP"),
    ("fraction_csp3", "mol_log_p", "Fraction Csp3", "LogP"),
    ("num_h_acceptors", "num_h_donors", "HBA", "HBD"),
]


def plot_scatter_matrix(data: np.ndarray, outdir: Path, sample_for_scatter: int = 50000):
    """Scatter matrix for key property pairs."""
    n_total = len(data)
    if n_total > sample_for_scatter:
        rng = np.random.default_rng(42)
        idx = rng.choice(n_total, sample_for_scatter, replace=False)
        scatter_data = data[idx]
    else:
        scatter_data = data

    n_pairs = len(SCATTER_PAIRS)
    n_cols = 3
    n_rows = int(np.ceil(n_pairs / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5.5, n_rows * 4.5))
    axes_flat = axes.flatten()

    for i, (x_col, y_col, x_label, y_label) in enumerate(SCATTER_PAIRS):
        ax = axes_flat[i]
        x_arr = scatter_data[x_col]
        y_arr = scatter_data[y_col]
        valid = ~np.isnan(x_arr) & ~np.isnan(y_arr)
        x_valid = x_arr[valid]
        y_valid = y_arr[valid]

        # Hexbin for density (better for large datasets)
        hb = ax.hexbin(x_valid, y_valid, gridsize=40, cmap="YlOrRd",
                        mincnt=1, bins="log", linewidths=0.1)
        plt.colorbar(hb, ax=ax, label="log(count)")
        ax.set_xlabel(x_label, fontsize=10)
        ax.set_ylabel(y_label, fontsize=10)
        # Annotate correlation
        if len(x_valid) > 2:
            r = np.corrcoef(x_valid, y_valid)[0, 1]
            ax.set_title(f"{y_label} vs {x_label}  (r={r:.3f})", fontsize=10, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Hide unused
    for j in range(n_pairs, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f"ChEMBL 1.55M — Key Property Scatter Matrix (hexbin, sampled {sample_for_scatter:,})",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    scatter_path = outdir / "scatter_matrix.png"
    fig.savefig(scatter_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Scatter matrix saved: {scatter_path}")

    # ── Bonus: LogP distribution density curve ──
    fig, ax = plt.subplots(figsize=(9, 5))
    logp_arr = scatter_data["mol_log_p"]
    logp_arr = logp_arr[~np.isnan(logp_arr)]
    ax.hist(logp_arr, bins=100, density=True, color="#3498DB", edgecolor="#2980B9",
            linewidth=0.2, alpha=0.8, label="Histogram (density)")
    # KDE overlay if scipy available
    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(logp_arr)
        x_curve = np.linspace(logp_arr.min(), logp_arr.max(), 300)
        ax.plot(x_curve, kde(x_curve), color="#E74C3C", linewidth=2.5, label="KDE")
    except ImportError:
        pass
    for p, color, ls, lbl in [(5, "#F39C12", "--", "P5"), (50, "#2C3E50", "-", "P50"), (95, "#27AE60", "--", "P95")]:
        pval = np.percentile(logp_arr, p)
        ax.axvline(pval, color=color, linestyle=ls, linewidth=1.5, label=f"{lbl}={pval:.2f}")
    ax.legend(fontsize=9)
    ax.set_xlabel("LogP (Crippen)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("ChEMBL 1.55M — LogP Distribution (Density)", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    logp_density_path = outdir / "logp_density.png"
    fig.savefig(logp_density_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] LogP density curve saved: {logp_density_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Boxplot summary
# ══════════════════════════════════════════════════════════════════════════════
def plot_boxplot_summary(data: np.ndarray, outdir: Path):
    """Box plot summary of key descriptors (normalized to visualize together)."""
    key_descriptors = [
        "mol_log_p", "mol_wt", "tpsa", "heavy_atoms", "num_heteroatoms",
        "num_h_acceptors", "num_h_donors", "num_rotatable_bonds",
        "num_rings", "num_aromatic_rings", "fraction_csp3", "carbon_fraction",
    ]
    # Normalize each to [0, 1] for visual comparison
    box_data = []
    labels = []
    for name in key_descriptors:
        arr = data[name]
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            continue
        p1, p99 = np.percentile(arr, [1, 99])
        if p99 > p1:
            arr_norm = (arr - p1) / (p99 - p1)
        else:
            arr_norm = arr - p1
        # Clip outliers for cleaner plot
        arr_norm = np.clip(arr_norm, -0.1, 1.1)
        box_data.append(arr_norm)
        labels.append(DESCRIPTOR_LABELS.get(name, name))

    fig, ax = plt.subplots(figsize=(14, 6))
    bp = ax.boxplot(box_data, vert=True, patch_artist=True, labels=labels,
                     showfliers=False, widths=0.6)
    colors = plt.cm.Set2(np.linspace(0, 1, len(box_data)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax.set_title("ChEMBL 1.55M — Key Descriptors Normalized (P1-P99 → 0-1)",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Normalized Value (P1-P99 range)", fontsize=10)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    box_path = outdir / "boxplot_summary.png"
    fig.savefig(box_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Boxplot summary saved: {box_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Lipinski analysis
# ══════════════════════════════════════════════════════════════════════════════
def plot_lipinski_analysis(data: np.ndarray, outdir: Path):
    """Lipinski rule-of-5 analysis: violation distribution + radar."""
    lip_viol = data["lipinski_violations"]
    lip_viol = lip_viol[~np.isnan(lip_viol)]
    heavy_viol = data["lipinski_heavy_violations"]
    heavy_viol = heavy_viol[~np.isnan(heavy_viol)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # (1) Violation count bar chart
    ax = axes[0]
    unique_v, counts_v = np.unique(lip_viol.astype(int), return_counts=True)
    pct_v = counts_v / counts_v.sum() * 100
    bars = ax.bar(unique_v, pct_v, color="#3498DB", edgecolor="#2980B9", width=0.6)
    for bar, pct in zip(bars, pct_v):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{pct:.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_title("Lipinski Violations", fontsize=11, fontweight="bold")
    ax.set_xlabel("Number of Violations", fontsize=10)
    ax.set_ylabel("% of Molecules", fontsize=10)
    ax.set_xticks([0, 1, 2])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # (2) Individual rule compliance
    ax = axes[1]
    # Recompute rule compliance from raw descriptors
    mw_arr = data["mol_wt"]
    logp_arr = data["mol_log_p"]
    hba_arr = data["num_h_acceptors"]
    hbd_arr = data["num_h_donors"]
    valid = ~np.isnan(mw_arr) & ~np.isnan(logp_arr) & ~np.isnan(hba_arr) & ~np.isnan(hbd_arr)
    n_valid = valid.sum()
    pass_mw = (mw_arr[valid] <= 500).sum()
    pass_logp = (logp_arr[valid] <= 5).sum()
    pass_hba = (hba_arr[valid] <= 10).sum()
    pass_hbd = (hbd_arr[valid] <= 5).sum()
    rules = ["MW ≤ 500", "LogP ≤ 5", "HBA ≤ 10", "HBD ≤ 5"]
    compliance = [pass_mw / n_valid * 100, pass_logp / n_valid * 100,
                  pass_hba / n_valid * 100, pass_hbd / n_valid * 100]
    bars = ax.barh(rules, compliance, color=["#27AE60", "#3498DB", "#F39C12", "#E74C3C"],
                    edgecolor="white", height=0.5)
    for bar, pct in zip(bars, compliance):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%", va="center", fontsize=10, fontweight="bold")
    ax.set_title("Rule Compliance Rate", fontsize=11, fontweight="bold")
    ax.set_xlabel("% Compliant", fontsize=10)
    ax.set_xlim(0, 105)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("ChEMBL 1.55M — Lipinski Rule-of-5 Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    lip_path = outdir / "lipinski_analysis.png"
    fig.savefig(lip_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Lipinski analysis saved: {lip_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="ChEMBL 1.55M Full Property Analysis")
    parser.add_argument("--sample", type=int, default=None,
                        help="Sample N molecules (default: all 1.55M)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip plots, only compute statistics")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTDIR),
                        help=f"Output directory (default: {DEFAULT_OUTDIR})")
    parser.add_argument("--chembl-csv", type=str, default=str(DEFAULT_CHEMBL),
                        help=f"ChEMBL CSV path (default: {DEFAULT_CHEMBL})")
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    max_rows = int(args.sample) if args.sample else None

    print("=" * 70)
    print("ChEMBL Full Property Analysis")
    print(f"  Input:  {args.chembl_csv}")
    print(f"  Output: {outdir}")
    print(f"  Sample: {max_rows or 'ALL'}")
    print("=" * 70)

    # 1. Load
    data = load_data(args.chembl_csv, max_rows)
    print(f"  Memory: {data.nbytes / 1e6:.1f} MB\n")

    # 2. Statistics
    print("Computing statistics...")
    stats = compute_statistics(data)
    save_statistics(stats, outdir)
    print_statistics_summary(stats)

    if args.no_plots:
        print("\n[OK] Done (plots skipped).")
        return

    # 3. Histograms
    print("\nGenerating histograms...")
    plot_all_histograms(data, outdir)

    # 4. Correlation
    print("\nGenerating correlation heatmap...")
    plot_correlation_heatmap(data, outdir)

    # 5. Scatter
    print("\nGenerating scatter matrix...")
    plot_scatter_matrix(data, outdir)

    # 6. Boxplot
    print("\nGenerating boxplot summary...")
    plot_boxplot_summary(data, outdir)

    # 7. Lipinski
    print("\nGenerating Lipinski analysis...")
    plot_lipinski_analysis(data, outdir)

    print(f"\n{'=' * 70}")
    print(f"All outputs saved to: {outdir}")
    print(f"Files:")
    for f in sorted(outdir.glob("*.png")):
        print(f"  {f.name}")
    for f in sorted(outdir.glob("*.csv")):
        print(f"  {f.name}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
