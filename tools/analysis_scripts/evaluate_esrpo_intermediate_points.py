"""Evaluate ESRPO mainline intermediate checkpoints for the structure-vs-docking trade-off.

For every step directory ``step_<TAG>/molecules.sdf`` under GEN_ROOT this script
computes (using the same relaxed口径 as work_records/260902-02):

  * strict sanitize pass rate (sanitize as-is)
  * relaxed sanitize pass rate (keep largest fragment, then sanitize)  <- 主卖点
  * relaxed "unreasonable structure" rate = 1 - relaxed pass rate
  * relaxed docking score median (read from step_*/docking/docking_results.csv,
    negative = better binding; only finite negative scores counted)

Outputs:
  * intermediate_points_metrics.csv  (one row per step)
  * structure_vs_step_curve.png      (论文主图: 双轴)

Usage:
  python tools/analysis_scripts/evaluate_esrpo_intermediate_points.py \
      --gen-root OUT/diffsbdd_e2po_mainline_intermediate_gen \
      --out-dir  analysis/e2po_mainline_intermediate_eval
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, QED, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.edgecolor"] = "#d4d4d4"
plt.rcParams["axes.linewidth"] = 1.2

DIFFSBDD_RED = "#E74C3C"
NATIVE_GREEN = "#2ECC71"


def largest_fragment(mol: Chem.Mol) -> Chem.Mol:
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if not frags:
        return mol
    return max(frags, key=lambda m: m.GetNumAtoms())


def relaxed_metrics(sdf: Path) -> dict:
    sup = Chem.SDMolSupplier(str(sdf), sanitize=False, removeHs=False)
    n = 0
    strict_ok = 0
    relaxed_ok = 0
    for m in sup:
        n += 1
        if m is None:
            continue
        try:
            mm = Chem.Mol(m)
            Chem.SanitizeMol(mm)
            strict_ok += 1
        except Exception:
            pass
        try:
            rm = Chem.Mol(largest_fragment(m))
            Chem.SanitizeMol(rm)
            relaxed_ok += 1
        except Exception:
            pass
    if n == 0:
        return {"n": 0, "strict": float("nan"), "relaxed": float("nan")}
    return {
        "n": n,
        "strict": 100.0 * strict_ok / n,
        "relaxed": 100.0 * relaxed_ok / n,
    }


def docking_median_relaxed(dock_csv: Path) -> float:
    """Read docking_results.csv, return median of finite negative scores."""
    if not dock_csv.exists():
        return float("nan")
    scores = []
    with dock_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                v = float(row.get("score", row.get("docking_score", "nan")))
            except (ValueError, TypeError):
                continue
            if np.isfinite(v) and v < 0:
                scores.append(v)
    if not scores:
        return float("nan")
    return float(np.median(scores))


def parse_step_tag(name: str) -> tuple[int, str]:
    m = re.search(r"step_(?P<tag>\w+)", name)
    tag = m.group("tag") if m else name
    if tag == "final":
        order = 10**9
    else:
        try:
            order = int(tag)
        except ValueError:
            order = 10**9
    return order, tag


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-root", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    gen_root = Path(args.gen_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    step_dirs = sorted(
        [p for p in gen_root.iterdir() if p.is_dir() and p.name.startswith("step_")],
        key=lambda p: parse_step_tag(p.name),
    )
    if not step_dirs:
        raise SystemExit(f"no step_* dirs found under {gen_root}")

    rows = []
    for sd in step_dirs:
        sdf = sd / "molecules.sdf"
        if not sdf.exists():
            print(f"[warn] missing {sdf}, skip")
            continue
        order, tag = parse_step_tag(sd.name)
        met = relaxed_metrics(sdf)
        dock_csv = sd / "docking" / "docking_results.csv"
        dock_med = docking_median_relaxed(dock_csv)
        rows.append(
            {
                "step_tag": tag,
                "update_step": order if tag != "final" else -1,
                "n_molecules": met["n"],
                "strict_sanitize_pct": round(met["strict"], 2),
                "relaxed_sanitize_pct": round(met["relaxed"], 2),
                "unreasonable_pct": round(100.0 - met["relaxed"], 2),
                "relaxed_dock_median": round(dock_med, 3) if np.isfinite(dock_med) else "",
            }
        )
        print(
            f"{sd.name:20s} strict={met['strict']:5.1f}% relaxed={met['relaxed']:5.1f}% "
            f"unreasonable={100-met['relaxed']:5.1f}% dock_med={dock_med:.2f}"
        )

    csv_path = out_dir / "intermediate_points_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[ok] wrote {csv_path}")

    # ---- 论文主图: 双轴 ----
    rows_sorted = sorted(rows, key=lambda r: r["update_step"])
    x = [r["update_step"] for r in rows_sorted]
    xlabels = [r["step_tag"] for r in rows_sorted]
    unreasonable = [r["unreasonable_pct"] for r in rows_sorted]
    relaxed_pass = [r["relaxed_sanitize_pct"] for r in rows_sorted]

    fig, ax1 = plt.subplots(figsize=(11, 6.5))
    # 左轴: 结构合理性 (不合理率 下降, 宽松通过率 上升)
    ax1.plot(x, unreasonable, "o-", color=DIFFSBDD_RED, linewidth=2.2, markersize=8, label="Unreasonable structure % (lower=better)")
    ax1.plot(x, relaxed_pass, "s--", color="#F39C12", linewidth=2.0, markersize=7, label="Relaxed sanitize pass %")
    ax1.set_xlabel("DPO optimizer update step")
    ax1.set_ylabel("Structure validity (%)")
    ax1.set_ylim(0, 100)
    ax1.set_xticks(x)
    ax1.set_xticklabels(xlabels, rotation=45, ha="right")

    # 右轴: relaxed 对接中位 (代价, 越低越好)
    dock_vals = [r["relaxed_dock_median"] for r in rows_sorted]
    if any(isinstance(v, (int, float)) and v != "" for v in dock_vals):
        ax2 = ax1.twinx()
        ax2.plot(x, dock_vals, "^:", color="#3498DB", linewidth=1.8, markersize=7, label="Relaxed docking median (lower=better)")
        ax2.set_ylabel("Relaxed docking score median (kcal/mol)")
        ax2.axhline(-7.93, color=NATIVE_GREEN, linestyle="--", linewidth=1.6, alpha=0.8)
        ax2.text(x[0], -7.93, "ESRPO relaxed median ref −7.93", color=NATIVE_GREEN, fontsize=9)

    # 合并图例
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels() if "ax2" in dir() else ([], [])
    ax1.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)

    plt.title("ESRPO mainline: structural validity vs training step (docking as acceptable cost)")
    fig.tight_layout()
    fig_path = out_dir / "structure_vs_step_curve.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {fig_path}")


if __name__ == "__main__":
    main()
