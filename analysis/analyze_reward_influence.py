"""Quantify per-component reward influence from actual training data.

Compares declared weight vs effective influence (weight × component_std)
to reveal why small-weight components can dominate (or be dominated by)
larger-weight ones.

Key metric: effective_influence_i = w_i * sigma_i
Where sigma_i = std of per-molecule reward values for component i.

Uses 7gpu-dock-ecfp4-fcfp4 experiment (has docking+chembl+diversity).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_molecules(exp_dir: str, epoch: int) -> pd.DataFrame:
    """Load molecule-level CSV for a given epoch (merge all rank files)."""
    epoch_dir = Path(exp_dir) / "epochs" / f"epoch_{epoch:04d}"
    csvs = sorted(epoch_dir.glob("molecules_rank*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV in {epoch_dir}")
    frames = [pd.read_csv(f) for f in csvs]
    df = pd.concat(frames, ignore_index=True)
    df = df[df["rdkit_valid"] == True].copy()
    return df


def analyze_components(df: pd.DataFrame, weights: dict[str, float]) -> dict:
    """Compute per-component stats and effective influence.

    Args:
        df: molecule-level DataFrame with {name}_reward columns.
        weights: declared normalized weights from CompositeReward.

    Returns:
        dict with per-component mean/std/range/effective_influence.
    """
    total_raw = df["reward_total_raw"].values if "reward_total_raw" in df.columns else None
    actual_total_std = float(np.std(total_raw)) if total_raw is not None else None

    results = {}
    for name, w in weights.items():
        col = f"{name}_reward"
        if col not in df.columns:
            continue
        vals = df[col].dropna().values
        if len(vals) == 0:
            continue

        mu = float(np.mean(vals))
        sigma = float(np.std(vals))
        vmin = float(np.min(vals))
        vmax = float(np.max(vals))
        w_sigma = w * sigma  # effective influence (absolute)

        results[name] = {
            "mean": mu,
            "std": sigma,
            "min": vmin,
            "max": vmax,
            "range": vmax - vmin,
            "normalized_weight": w,
            "effective_influence_abs": w_sigma,
        }

    if actual_total_std and actual_total_std > 0:
        total_influence = sum(r["effective_influence_abs"] for r in results.values())
        for name, r in results.items():
            r["effective_influence_pct"] = (
                100.0 * r["effective_influence_abs"] / total_influence
            )

    return {
        "components": results,
        "actual_total_std": actual_total_std,
    }


def print_table(
    title: str,
    results: dict,
    weights: dict[str, float],
):
    """Pretty-print the analysis table."""
    comps = results["components"]
    total_std = results.get("actual_total_std")

    print()
    print("=" * 95)
    print(f"  {title}")
    print("=" * 95)

    header = (
        f"{'Component':<28s} {'Weight':>7s} {'Mean':>8s} {'Std':>8s} "
        f"{'[Min, Max]':>26s} {'Influence':>10s} {'Infl%':>7s}"
    )
    print(header)
    print("-" * 95)

    for name in weights:
        if name not in comps:
            continue
        r = comps[name]
        infl_pct = r.get("effective_influence_pct", 0.0)
        print(
            f"{name:<28s} {r['normalized_weight']:7.3f} "
            f"{r['mean']:8.3f} {r['std']:8.3f} "
            f"[{r['min']:7.3f}, {r['max']:7.3f}]"
            f"{'':>4s}"
            f"{r['effective_influence_abs']:10.4f} {infl_pct:6.1f}%"
        )

    total_infl = sum(r["effective_influence_abs"] for r in comps.values())
    print("-" * 95)
    print(f"{'Sum / Total Std':<28s} {'':7s} {'':8s} {'':8s} {'':26s} {total_infl:10.4f}")
    if total_std is not None:
        print(f"{'(actual reward_total_raw std)':<28s} {'':7s} {'':8s} {total_std:8.3f}")


def print_comparison(
    before: dict, after: dict, weights: dict[str, float]
):
    """Side-by-side: declared weight vs effective influence %."""
    print()
    print("=" * 85)
    print("  WEIGHT vs TRUE INFLUENCE (before normalization)")
    print("=" * 85)
    print(
        f"{'Component':<28s} {'Declared%':>9s} {'TrueInfl%':>9s} "
        f"{'Ratio':>8s} {'Bias':>8s}"
    )
    print("-" * 85)

    for name in weights:
        if name not in before["components"]:
            continue
        declared_pct = weights[name] * 100
        true_pct = before["components"][name].get("effective_influence_pct", 0.0)
        ratio = true_pct / declared_pct if declared_pct > 0 else float("inf")
        bias = "" if 0.5 < ratio < 2.0 else f"{'OVER' if ratio > 2 else 'UNDER'}"
        print(
            f"{name:<28s} {declared_pct:8.1f}% {true_pct:8.1f}% "
            f"{ratio:7.1f}x {bias:>8s}"
        )


def main():
    base = "/home/md666/data/disk6/DiffB110/OUT"

    # ── Experiment 1: 7gpu-dock-ecfp4-fcfp4 (docking+chembl+diversity) ──
    exp_7gpu = f"{base}/7gpu-dock-ecfp4-fcfp4_20260521_212927"

    # CompositeReward normalized weights (from config: 0.5+0.25+0.25+0.1+0.1=1.2→normalized):
    # docking:     0.5/1.2 = 0.417
    # chembl_ecfp4:0.25/1.2= 0.208
    # chembl_fcfp4:0.25/1.2= 0.208
    # div_ecfp4:   0.10/1.2= 0.083
    # div_fcfp4:   0.10/1.2= 0.083
    weights_7gpu = {
        "docking": 0.417,
        "chembl_prior_ecfp4": 0.208,
        "chembl_prior_fcfp4": 0.208,
        "diversity_ecfp4": 0.083,
        "diversity_fcfp4": 0.083,
    }

    declared_weights_7gpu = {
        "docking": 0.50,
        "chembl_prior_ecfp4": 0.25,
        "chembl_prior_fcfp4": 0.25,
        "diversity_ecfp4": 0.10,
        "diversity_fcfp4": 0.10,
    }

    # ── Healthy epoch: E10 ──
    df10 = load_molecules(exp_7gpu, 10)
    r10 = analyze_components(df10, weights_7gpu)
    print_table("7GPU Experiment — Epoch 10 (HEALTHY, before collapse)", r10, weights_7gpu)
    print_comparison(r10, None, declared_weights_7gpu)

    # ── Collapse epoch: E26 ──
    df26 = load_molecules(exp_7gpu, 26)
    r26 = analyze_components(df26, weights_7gpu)
    print_table("7GPU Experiment — Epoch 26 (COLLAPSE, chembl_prior penalizing hard)", r26, weights_7gpu)

    # ── Deep collapse: E40 ──
    try:
        df40 = load_molecules(exp_7gpu, 40)
        r40 = analyze_components(df40, weights_7gpu)
        print_table("7GPU Experiment — Epoch 40 (DEEP COLLAPSE)", r40, weights_7gpu)
    except FileNotFoundError:
        print("\n[Epoch 40 not available, using E35]")
        df35 = load_molecules(exp_7gpu, 35)
        r35 = analyze_components(df35, weights_7gpu)
        print_table("7GPU Experiment — Epoch 35 (DEEP COLLAPSE)", r35, weights_7gpu)

    # ── Experiment 2: 6gpu-chembl-only ──
    exp_6gpu = f"{base}/6gpu-chembl-only_20260521_154957"
    weights_6gpu = {
        "chembl_prior_ecfp4": 0.50,
        "chembl_prior_fcfp4": 0.50,
    }
    try:
        df_c6 = load_molecules(exp_6gpu, 10)
        r_c6 = analyze_components(df_c6, weights_6gpu)
        print_table("6GPU-chembl-only — Epoch 10 (NO diversity, NO docking)", r_c6, weights_6gpu)
    except FileNotFoundError:
        print(f"\n[6gpu-chembl-only Epoch 10 not found, skipping]")

    # ── WHAT PER-COMPONENT NORMALIZATION WOULD CHANGE ──
    print()
    print("=" * 85)
    print("  IMPACT OF PER-COMPONENT NORMALIZATION (hypothetical @ E10)")
    print("=" * 85)
    print(f"{'Component':<28s} {'RawStd':>8s} {'Weight':>7s} {'Before':>8s} {'After':>8s} {'Change':>8s}")
    print("-" * 85)

    for name in weights_7gpu:
        if name not in r10["components"]:
            continue
        raw_std = r10["components"][name]["std"]
        w = weights_7gpu[name]
        before = w * raw_std
        after = w * 1.0  # after normalization, sigma ≈ 1.0
        change = (after - before) / (before + 1e-8) * 100
        sig = "▲" if change > 20 else "▼" if change < -20 else " "
        print(
            f"{name:<28s} {raw_std:8.3f} {w:7.3f} "
            f"{before:8.4f} {after:8.4f} {change:+7.0f}% {sig}"
        )

    # ── Summary: the "Diversity was silenced" story ──
    print()
    print("=" * 85)
    print("  KEY INSIGHT: Why diversity was silenced")
    print("=" * 85)
    for name in ["diversity_ecfp4", "diversity_fcfp4", "chembl_prior_ecfp4"]:
        if name not in r10["components"]:
            continue
        r = r10["components"][name]
        w = weights_7gpu[name]
        print(
            f"  {name}: range=[{r['min']:.3f}, {r['max']:.3f}], "
            f"std={r['std']:.3f}, declared_weight={w:.3f}, "
            f"effective_influence={r.get('effective_influence_pct',0):.1f}%"
        )

    # ── Before/after per-component normalization comparison table ──
    print()
    print("=" * 105)
    print("  BEFORE vs AFTER per-component normalization: True Influence %")
    print("=" * 105)
    print(f"{'Component':<28s} {'Declared%':>9s} {'Before%':>9s} {'After%':>9s} {'Drift':>9s}")
    print("-" * 105)
    total_raw_std = sum(weights_7gpu[n] * r10["components"][n]["std"] for n in weights_7gpu if n in r10["components"])
    for name in weights_7gpu:
        if name not in r10["components"]:
            continue
        declared_pct = declared_weights_7gpu[name] * 100
        w = weights_7gpu[name]
        before_infl = w * r10["components"][name]["std"] / total_raw_std * 100
        after_infl = w * 1.0 / sum(weights_7gpu.values()) * 100  # after norm sigma≈1
        drift = after_infl - before_infl
        maker = "★" if abs(drift) > 8 else ""
        print(
            f"{name:<28s} {declared_pct:8.1f}% {before_infl:8.1f}% "
            f"{after_infl:8.1f}% {drift:+7.1f}% {maker}"
        )

    print()
    print("KEY:")
    print("  Declared% = raw weight from YAML config (before normalization)")
    print("  Before%   = true contribution to total variance (weight × component_std)")
    print("  After%    = contribution if per-component z-score is applied")
    print("  ★ marks components whose influence changes by >8% after normalization")


if __name__ == "__main__":
    main()
