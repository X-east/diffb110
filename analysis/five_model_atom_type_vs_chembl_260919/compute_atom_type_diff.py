#!/usr/bin/env python
"""五模型生成分子的原子类型分布 vs ChEMBL 参照。

口径
----
生成侧：``analysis/benchmark/benchmark5/five_model_repeats_500steps_260826/rep{4,5,6}_*/
filtered_sanitize_valid/<model>.sanitize_valid.sdf``（500 步、3 次重复、sanitize 合法分子）。
参照侧：``data/chembl/chembl_36_molecules.csv.gz``
（chembl_structure_pipeline 标准化 + mol_wt 150–500 + heavy_atoms 10–30）。

两侧使用同一实现：RDKit 解析 → largest-fragment 归约 → 统计每个元素的重原子数量。

两套差异口径
------------
1. ``tvd_vs_chembl``         ：全部生成分子直接与 ChEMBL 参照比（元素原子占比）。
2. ``tvd_vs_chembl_size_matched``：**先限制在双方重叠的重原子数窗口 [10, 30] 内**，
   再把 ChEMBL 按该子集的重原子数直方图分层重加权，消除分子大小差异。
   同时输出被排除分子的比例（``excluded_out_of_window_pct``），因为倒推的
   本方法各模型有 6%–26% 的分子落在窗口之外，直接"匹配"会静默丢弃这些分子。

差异度量：元素占比上的总变差距离 TVD（= 0.5 × L1）与 Jensen–Shannon 散度（log2）。
"""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

REPO = Path("/home/md666/data/disk6/DiffB110")
OUT_DIR = Path(__file__).resolve().parent

BENCH_ROOT = REPO / "analysis/benchmark/benchmark5/five_model_repeats_500steps_260826"
REPS = ("rep4_seed22_29", "rep5_seed30_37", "rep6_seed38_45")
MODELS = {
    "DiffSBDD": "diffsbdd.sanitize_valid.sdf",
    "DiffGui": "diffgui.sanitize_valid.sdf",
    "PMDM": "pmdm.sanitize_valid.sdf",
    "GCDM-SBDD": "gcdm.sanitize_valid.sdf",
    "TargetDiff": "targetdiff.sanitize_valid.sdf",
}
CHEMBL_CSV = REPO / "data/chembl/chembl_36_molecules.csv.gz"
WORKERS = 18

ELEMENTS = ["C", "N", "O", "S", "F", "Cl", "Br", "P", "I"]
OTHER = "other"
HAC_WINDOW = (10, 30)  # ChEMBL reference window
LOGGER = logging.getLogger("atom-type-diff")

COLORS = {
    "DiffSBDD": "#2E86C1",
    "DiffGui": "#E74C3C",
    "PMDM": "#8E44AD",
    "GCDM-SBDD": "#16A085",
    "TargetDiff": "#F39C12",
    "ChEMBL": "#2ECC71",
}


def _largest_fragment(mol: Chem.Mol) -> Chem.Mol:
    try:
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    except Exception:
        return mol
    if not frags:
        return mol
    return max(frags, key=lambda m: m.GetNumAtoms())


def count_elements(mol: Chem.Mol) -> dict[str, int]:
    counts = Counter(atom.GetSymbol() for atom in mol.GetAtoms())
    out = {el: int(counts.get(el, 0)) for el in ELEMENTS}
    out[OTHER] = int(sum(v for k, v in counts.items() if k not in ELEMENTS))
    out["_hac"] = int(sum(counts.values()))
    return out


def elements_from_smiles(smiles: str) -> dict[str, int] | None:
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return count_elements(_largest_fragment(mol))


def collect_sdf(path: Path) -> list[dict[str, int]]:
    supplier = Chem.SDMolSupplier(str(path), removeHs=True, sanitize=True)
    return [count_elements(_largest_fragment(mol)) for mol in supplier if mol is not None]


def collect_chembl() -> list[dict[str, int]]:
    frame = pd.read_csv(CHEMBL_CSV, usecols=["canonical_smiles"])
    smiles_list = frame["canonical_smiles"].astype(str).tolist()
    LOGGER.info("ChEMBL rows=%d; evaluating with %d workers", len(smiles_list), WORKERS)
    with Pool(processes=WORKERS) as pool:
        results = pool.map(elements_from_smiles, smiles_list, chunksize=2048)
    return [r for r in results if r is not None]


def fractions(rows: list[dict[str, int]]) -> pd.Series:
    total = sum(sum(r[el] for el in ELEMENTS + [OTHER]) for r in rows)
    counts = {el: sum(r[el] for r in rows) for el in ELEMENTS + [OTHER]}
    return pd.Series({el: counts[el] / total for el in ELEMENTS + [OTHER]})


def tvd(p: pd.Series, q: pd.Series) -> float:
    return float(0.5 * np.abs(p - q).sum())


def js_divergence(p: pd.Series, q: pd.Series) -> float:
    pv = p.to_numpy(dtype=float)
    qv = q.to_numpy(dtype=float)
    m = 0.5 * (pv + qv)
    out = 0.0
    for v in (pv, qv):
        mask = (v > 0) & (m > 0)
        out += 0.5 * float(np.sum(v[mask] * np.log2(v[mask] / m[mask])))
    return out


def hac_reweighted_reference(
    chembl_rows: list[dict[str, int]], hac_hist: dict[int, float]
) -> pd.Series:
    """Reweight the ChEMBL element fractions onto a target HAC distribution."""
    frame = pd.DataFrame(chembl_rows)
    total = {el: 0.0 for el in ELEMENTS + [OTHER]}
    weight_sum = 0.0
    for hac, weight in hac_hist.items():
        subset = frame[frame["_hac"] == hac]
        if subset.empty or weight <= 0:
            continue
        weight_sum += weight
        per_mol = subset[ELEMENTS + [OTHER]].sum(axis=0)
        for el in ELEMENTS + [OTHER]:
            total[el] += weight * (per_mol[el] / float(len(subset)))
    series = pd.Series(total)
    if weight_sum <= 0:
        raise ValueError("no overlapping HAC bin between model and ChEMBL")
    return series / series.sum()


def hac_stats(rows: list[dict[str, int]]) -> dict:
    hacs = np.array([r["_hac"] for r in rows])
    lo, hi = HAC_WINDOW
    return {
        "hac_min": int(hacs.min()),
        "hac_max": int(hacs.max()),
        "hac_mean": float(hacs.mean()),
        "hac_median": float(np.median(hacs)),
        "in_window_pct": float(((hacs >= lo) & (hacs <= hi)).mean() * 100),
        "below_window_pct": float((hacs < lo).mean() * 100),
        "above_window_pct": float((hacs > hi).mean() * 100),
    }


def render_plots(
    fractions_df: pd.DataFrame, deviation_df: pd.DataFrame, metrics: pd.DataFrame, hac_box: pd.DataFrame
) -> None:
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    if Path(font_path).exists():
        font_manager.fontManager.addfont(font_path)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.25)
    plt.rcParams["axes.edgecolor"] = "#d4d4d4"
    plt.rcParams["axes.linewidth"] = 1.2

    cols = list(fractions_df.columns)
    x = np.arange(len(cols))
    width = 0.14

    # --- 1) element fraction grouped bars -----------------------------------
    fig, ax = plt.subplots(figsize=(16, 7), dpi=300)
    for i, condition in enumerate(fractions_df.index):
        offset = (i - (len(fractions_df.index) - 1) / 2) * width
        ax.bar(
            x + offset,
            fractions_df.loc[condition, cols].to_numpy(dtype=float),
            width=width,
            label=condition,
            color=COLORS.get(condition, "#888888"),
            edgecolor="white",
            linewidth=0.5,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(cols, fontsize=14)
    ax.set_ylabel("Atom fraction", fontsize=14)
    ax.set_title(
        "Atom-type distribution: five 500-step generators vs ChEMBL reference",
        fontsize=17,
        fontweight="bold",
    )
    ax.legend(fontsize=12, ncol=6, frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "atom_type_fractions.png", dpi=300, facecolor="white")
    fig.savefig(OUT_DIR / "atom_type_fractions.svg", facecolor="white")
    plt.close(fig)

    # --- 2) deviation from ChEMBL (pp) --------------------------------------
    fig, ax = plt.subplots(figsize=(16, 7), dpi=300)
    for i, condition in enumerate(deviation_df.index):
        offset = (i - (len(deviation_df.index) - 1) / 2) * width
        ax.bar(
            x + offset,
            deviation_df.loc[condition, cols].to_numpy(dtype=float) * 100,
            width=width,
            label=condition,
            color=COLORS.get(condition, "#888888"),
            edgecolor="white",
            linewidth=0.5,
        )
    ax.axhline(0.0, color="#E74C3C", linestyle="--", linewidth=2.0, alpha=0.85, zorder=4)
    for idx, condition in enumerate(metrics.index):
        ax.text(
            0.015,
            0.97 - 0.062 * idx,
            f"{condition}: TVD = {metrics.loc[condition, 'tvd_vs_chembl'] * 100:.1f}%"
            f"  (size-matched {metrics.loc[condition, 'tvd_vs_chembl_size_matched'] * 100:.1f}%)",
            transform=ax.transAxes,
            fontsize=11.5,
            color=COLORS.get(condition, "#888888"),
            fontweight="bold",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(cols, fontsize=14)
    ax.set_ylabel("Deviation from ChEMBL (pp)", fontsize=14)
    ax.set_title(
        "Atom-fraction deviation from ChEMBL (positive = over-represented)",
        fontsize=17,
        fontweight="bold",
    )
    ax.legend(fontsize=12, ncol=5, frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "atom_fraction_deviation.png", dpi=300, facecolor="white")
    fig.savefig(OUT_DIR / "atom_fraction_deviation.svg", facecolor="white")
    plt.close(fig)

    # --- 3) heavy-atom-count distribution -----------------------------------
    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
    order = list(MODELS) + ["ChEMBL"]
    sns.boxplot(data=hac_box, x="condition", y="hac", order=order, ax=ax,
                palette=[COLORS[c] for c in order], width=0.62, showfliers=False)
    sns.stripplot(data=hac_box.sample(min(4000, len(hac_box)), random_state=0),
                  x="condition", y="hac", order=order, ax=ax,
                  color="#333333", alpha=0.12, size=1.6, jitter=0.28)
    ax.axhspan(HAC_WINDOW[0], HAC_WINDOW[1], color="#2ECC71", alpha=0.07, zorder=0)
    for bound in HAC_WINDOW:
        ax.axhline(bound, color="#E74C3C", linestyle="--", linewidth=2.0, alpha=0.85, zorder=4)
    ax.text(0.01, HAC_WINDOW[1], " ChEMBL reference window [10, 30]", color="#E74C3C",
            fontsize=12, fontweight="bold", va="bottom")
    ax.set_xlabel("")
    ax.set_ylabel("Heavy atom count", fontsize=14)
    ax.set_title("Heavy-atom-count distribution per generator (vs ChEMBL window)",
                 fontsize=17, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "hac_distribution.png", dpi=300, facecolor="white")
    fig.savefig(OUT_DIR / "hac_distribution.svg", facecolor="white")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(OUT_DIR / "atom_type_diff.log", mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    model_rows: dict[str, list[dict[str, int]]] = {}
    for model, filename in MODELS.items():
        rows: list[dict[str, int]] = []
        for rep in REPS:
            sdf = BENCH_ROOT / rep / "filtered_sanitize_valid" / filename
            if not sdf.is_file():
                raise FileNotFoundError(sdf)
            rows.extend(collect_sdf(sdf))
        model_rows[model] = rows

    chembl_rows = collect_chembl()
    chembl_frame = pd.DataFrame(chembl_rows)
    chembl_frac = fractions(chembl_rows)

    frac_rows: dict[str, pd.Series] = {}
    matched_rows: dict[str, pd.Series] = {}
    hac_rows: dict[str, dict] = {}
    hac_for_plot: list[pd.DataFrame] = []
    lo, hi = HAC_WINDOW

    for model, rows in model_rows.items():
        frac_rows[model] = fractions(rows)
        stats = hac_stats(rows)
        hac_rows[model] = stats
        hac_for_plot.append(
            pd.DataFrame({"condition": model, "hac": [r["_hac"] for r in rows]})
        )
        in_window = [r for r in rows if lo <= r["_hac"] <= hi]
        counts = Counter(r["_hac"] for r in in_window)
        hist = {h: c / len(in_window) for h, c in counts.items()}
        matched_rows[model] = hac_reweighted_reference(chembl_rows, hist)
        LOGGER.info(
            "%-11s n=%4d HAC %.1f [%d,%d]  in-window=%.1f%%  size-matched n_used=%d",
            model, len(rows), stats["hac_mean"], stats["hac_min"], stats["hac_max"],
            stats["in_window_pct"], len(in_window),
        )

    hac_for_plot.append(pd.DataFrame({"condition": "ChEMBL", "hac": chembl_frame["_hac"]}))
    hac_box = pd.concat(hac_for_plot, ignore_index=True)

    hac_rows["ChEMBL"] = hac_stats(chembl_rows)
    hac_rows["ChEMBL"]["n_molecules"] = len(chembl_rows)

    fractions_df = pd.DataFrame(frac_rows).T
    fractions_df.loc["ChEMBL"] = chembl_frac
    fractions_df = fractions_df[ELEMENTS + [OTHER]]
    matched_df = pd.DataFrame(matched_rows).T[fractions_df.columns]
    deviation_df = fractions_df.drop(index="ChEMBL") - chembl_frac[fractions_df.columns]

    model_names = list(frac_rows)
    metrics = pd.DataFrame(
        {
            "n_molecules": [len(model_rows[m]) for m in model_names],
            "hac_mean": [hac_rows[m]["hac_mean"] for m in model_names],
            "hac_min": [hac_rows[m]["hac_min"] for m in model_names],
            "hac_max": [hac_rows[m]["hac_max"] for m in model_names],
            "in_window_pct": [hac_rows[m]["in_window_pct"] for m in model_names],
            "excluded_out_of_window_pct": [100 - hac_rows[m]["in_window_pct"] for m in model_names],
            "tvd_vs_chembl": [tvd(fractions_df.loc[m], chembl_frac) for m in model_names],
            "jsd_vs_chembl": [js_divergence(fractions_df.loc[m], chembl_frac) for m in model_names],
            "tvd_vs_chembl_size_matched": [tvd(matched_df.loc[m], chembl_frac) for m in model_names],
        },
        index=model_names,
    )

    ratio_rows = {}
    chembl_hetero = sum(chembl_frac[el] for el in ["N", "O", "S", "P"])
    for m in model_names:
        model_frac = fractions_df.loc[m]
        model_hetero = sum(model_frac[el] for el in ["N", "O", "S", "P"])
        row = {
            "hetero_per_C_model": float(model_hetero / model_frac["C"]),
            "hetero_per_C_chembl": float(chembl_hetero / chembl_frac["C"]),
            "hetero_per_C_ratio": float((model_hetero / model_frac["C"]) / (chembl_hetero / chembl_frac["C"])),
        }
        for el in ["N", "O", "S", "F", "Cl", "Br"]:
            row[f"{el}_ratio_model_over_chembl"] = float(
                model_frac[el] / chembl_frac[el] if chembl_frac[el] > 0 else float("nan")
            )
        ratio_rows[m] = row
    ratios = pd.DataFrame(ratio_rows).T

    fractions_df.to_csv(OUT_DIR / "atom_type_fractions.csv", encoding="utf-8-sig")
    matched_df.to_csv(OUT_DIR / "atom_type_fractions_size_matched.csv", encoding="utf-8-sig")
    deviation_df.to_csv(OUT_DIR / "atom_fraction_deviation_pp.csv", encoding="utf-8-sig")
    metrics.to_csv(OUT_DIR / "atom_type_diff_metrics.csv", encoding="utf-8-sig")
    ratios.to_csv(OUT_DIR / "element_ratios_vs_chembl.csv", encoding="utf-8-sig")
    (OUT_DIR / "hac_stats.json").write_text(
        json.dumps(hac_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    render_plots(fractions_df, deviation_df, metrics, hac_box)

    (OUT_DIR / "summary.json").write_text(
        json.dumps(
            {
                "convention": {
                    "generated": f"500-step, 3 reps pooled, sanitize-valid SDF, largest fragment",
                    "chembl": f"chembl_36_molecules.csv.gz (standardised, MW 150-500, "
                              f"HAC {lo}-{hi}), largest fragment",
                    "metric": "TVD = 0.5 * L1 over element atom fractions; JSD in bits",
                    "size_matched": f"model and ChEMBL both restricted to HAC in [{lo},{hi}]; "
                                    "ChEMBL reweighted onto the model's HAC histogram",
                },
                "chembl_reference_fractions": chembl_frac.to_dict(),
                "metrics": metrics.to_dict(orient="index"),
                "hac_stats": hac_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    LOGGER.info("=" * 78)
    for m in metrics.index:
        LOGGER.info(
            "%-11s n=%4d TVD=%.2f%% JSD=%.4f bit  size-matched TVD=%.2f%% (in-window %.0f%%)",
            m, int(metrics.loc[m, "n_molecules"]),
            metrics.loc[m, "tvd_vs_chembl"] * 100,
            metrics.loc[m, "jsd_vs_chembl"],
            metrics.loc[m, "tvd_vs_chembl_size_matched"] * 100,
            metrics.loc[m, "in_window_pct"],
        )
    LOGGER.info("wrote outputs to %s", OUT_DIR)


if __name__ == "__main__":
    main()
