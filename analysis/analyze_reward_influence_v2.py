"""Reward influence analysis v2 — includes raw similarity distribution.

Fix: diversity is per-molecule, but ecfp4 target=0.15 sits above the natural
similarity range of random CrossDocked molecules (0.04-0.14), so it never triggers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


BASE = Path("/home/md666/data/disk6/DiffB110/OUT")


def load_epoch(exp: str, ep: int) -> pd.DataFrame:
    d = BASE / exp / "epochs" / f"epoch_{ep:04d}"
    frames = [pd.read_csv(f) for f in sorted(d.glob("molecules_rank*.csv"))]
    df = pd.concat(frames, ignore_index=True)
    return df[df["rdkit_valid"] == True].copy()


def main():
    exp = "7gpu-dock-ecfp4-fcfp4_20260521_212927"

    # weights from config (docking=0.5, ecfp4=0.25, fcfp4=0.25, div_ecfp4=0.10, div_fcfp4=0.10)
    # normalized by CompositeReward: sum=1.35
    weights_norm = {
        "docking": 0.5 / 1.35,
        "chembl_prior_ecfp4": 0.25 / 1.35,
        "chembl_prior_fcfp4": 0.25 / 1.35,
        "diversity_ecfp4": 0.10 / 1.35,
        "diversity_fcfp4": 0.10 / 1.35,
    }

    print("=" * 120)
    print("  REAL REWARD INFLUENCE — With raw similarity analysis")
    print("=" * 120)

    for ep in [1, 10, 23, 26, 40]:
        df = load_epoch(exp, ep)
        n = len(df)
        print(f"\n{'─'*100}")
        print(f"  Epoch {ep} (n={n})")

        # ── Raw similarity analysis ──
        for prefix, target in [("diversity_ecfp4", 0.15), ("diversity_fcfp4", 0.20)]:
            col = f"{prefix}_similarity_raw"
            if col not in df.columns:
                continue
            v = df[col].dropna().values
            above = np.sum(v > target)
            print(f"  {prefix}: mean_sim={np.mean(v):.4f}  std={np.std(v):.4f}  "
                  f"[{np.min(v):.4f}, {np.max(v):.4f}]  "
                  f">target({target})={above}/{n} ({100*above/max(1,n):.1f}%)")

        # ── Per-component reward influence ──
        print(f"\n  {'Component':<28s} {'Weight':>7s} {'Mean':>8s} {'Std':>8s} "
              f"{'[Min, Max]':>26s} {'Infl':>10s} {'Infl%':>7s}")
        print(f"  {'─'*95}")

        total_infl = 0.0
        rows = []
        for name, w in weights_norm.items():
            col = f"{name}_reward"
            if col not in df.columns:
                continue
            v = df[col].dropna().values
            if len(v) == 0:
                continue
            mu, sigma = float(np.mean(v)), float(np.std(v))
            infl = w * sigma
            total_infl += infl
            rows.append((name, w, mu, sigma, float(np.min(v)), float(np.max(v)), infl))

        for name, w, mu, sigma, vmin, vmax, infl in rows:
            pct = 100 * infl / total_infl if total_infl > 0 else 0.0
            print(f"  {name:<28s} {w:7.3f} {mu:8.3f} {sigma:8.3f} "
                  f"[{vmin:7.3f}, {vmax:7.3f}]"
                  f"{'':>4s}{infl:10.4f} {pct:6.1f}%")

        # ── Effective influence vs declared weight ──
        print(f"\n  {'Declared vs True':─^60}")
        print(f"  {'Component':<28s} {'Declared':>8s} {'True':>8s} {'Ratio':>7s}")
        declared = {"docking":0.5,"chembl_prior_ecfp4":0.25,"chembl_prior_fcfp4":0.25,
                     "diversity_ecfp4":0.10,"diversity_fcfp4":0.10}
        for name, w, mu, sigma, vmin, vmax, infl in rows:
            pct_true = 100 * infl / total_infl if total_infl > 0 else 0
            pct_decl = declared.get(name, 0) * 100
            ratio = pct_true / pct_decl if pct_decl > 0 else float("inf")
            flag = "⚠ OVER" if ratio > 1.5 else "⚠ UNDER" if ratio < 0.5 else ""
            print(f"  {name:<28s} {pct_decl:7.1f}% {pct_true:7.1f}% {ratio:6.1f}x {flag}")

    # ── What after normalization ──
    print(f"\n{'='*100}")
    print("  AFTER per-component normalization (hypothetical, σ≈1 for all)")
    print(f"{'='*100}")
    print(f"  {'Component':<28s} {'Before%':>8s} {'After%':>8s} {'Change':>8s}")

    # Use E10 data for "before"
    df10 = load_epoch(exp, 10)
    total_before = 0.0
    infl_before = {}
    for name, w in weights_norm.items():
        col = f"{name}_reward"
        if col not in df10.columns:
            continue
        v = df10[col].dropna().values
        infl = w * float(np.std(v))
        infl_before[name] = infl
        total_before += infl

    total_after = sum(weights_norm.values()) * 1.0  # σ≈1
    for name, w in weights_norm.items():
        before_pct = 100 * infl_before.get(name, 0) / total_before if total_before > 0 else 0
        after_pct = 100 * w / total_after if total_after > 0 else 0
        change = after_pct - before_pct
        star = " ★" if abs(change) > 8 else ""
        print(f"  {name:<28s} {before_pct:7.1f}% {after_pct:7.1f}% {change:+7.1f}pp{star}")

    # ── Diagnosis ──
    print(f"\n{'='*100}")
    print("  DIAGNOSIS")
    print(f"{'='*100}")
    print("""
  1. diversity_ecfp4 target=0.15 is ABOVE the natural ECFP4 pairwise similarity
     range of CrossDocked-generated molecules (0.04-0.14). It NEVER triggers.
     → This component is effectively disabled. Not a weight problem.

  2. diversity_fcfp4 target=0.20 is SLIGHTLY above natural mean (~0.17-0.18),
     so ~30% of molecules trigger it in healthy epochs. Mild -0.05 mean penalty.

  3. chembl_prior_fcfp4 has the widest range (std=0.57 at E10) and dominates
     with 41% true influence vs 25% declared.

  4. After per-component normalization, weight == exact influence share.
     No component dominates or vanishes due to its own value range.
""")


if __name__ == "__main__":
    main()
