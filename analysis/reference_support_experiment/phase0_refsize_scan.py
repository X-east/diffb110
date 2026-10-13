#!/usr/bin/env python
"""Phase 0: pick the reference size |R| and the reward definition (A vs B).

Two decisions are settled here:

1. ``|R|`` — the number of ChEMBL molecules used to build the BRICS inventory.
   Larger |R| means a stricter definition of "novel", but it also pushes every
   novel_frac down and can compress the learnable contrast.

2. Reward definition:
   A (hard) : novel_frac  = share of BRICS fragments with freq == 0
   B (soft) : soft_novel  = share weighted by log-frequency, with a floor at 0
              penalty only once freq >= T (default 1000)

The null (real-real) distribution is the decisive criterion: a reward must have
*spread* among real molecules, otherwise there is no gradient to learn from.
Note that A is expected to pile up at exactly 0 (floor saturation).

Reference sub-samples are NESTED (50k ⊂ 200k ⊂ 500k) so the comparison is clean.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import multiprocessing as mp
import pickle
import random
from collections import Counter
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import BRICS

RDLogger.DisableLog("rdApp.*")

ROOT = Path("/home/md666/data/disk6/DiffB110")
HERE = ROOT / "analysis/reference_support_experiment"
FP_PATH = ROOT / "data/chembl/chembl_36_ecfp4_fingerprints.pkl.gz"

SEED = 20260914
HOLDOUT_FRAC = 0.10
REF_SIZES = (50_000, 200_000, 500_000)
NULL_SIZE = 5_000
SOFT_T = 1000

# a lean but representative slice of the generated sets (keeps the scan cheap)
E2PO_STEPS = ("step_002000", "step_004000", "step_016000", "step_final")


# --------------------------------------------------------------------------- #
def _brics_worker(smiles: list[str]) -> Counter:
    c: Counter = Counter()
    for smi in smiles:
        try:
            mol = Chem.MolFromSmiles(smi)
        except Exception:
            continue
        if mol is None:
            continue
        try:
            for f in BRICS.BRICSDecompose(mol):
                c[f] += 1
        except Exception:
            pass
    return c


def build_inventory(smiles: list[str], workers: int) -> Counter:
    chunks = [smiles[i::workers] for i in range(workers)]
    with mp.Pool(workers) as pool:
        parts = pool.map(_brics_worker, chunks)
    inv: Counter = Counter()
    for p in parts:
        inv.update(p)
    return inv


def soft_novel(freq: int, t: int = SOFT_T) -> float:
    if freq <= 0:
        return 1.0
    return max(0.0, 1.0 - math.log1p(freq) / math.log1p(t))


def score(mol, inv: Counter) -> tuple[float, float] | None:
    """Return (hard novel_frac, soft novelty) — both are 'badness' (higher = worse)."""
    if mol is None:
        return None
    try:
        frags = list(BRICS.BRICSDecompose(mol))
    except Exception:
        return None
    if not frags:
        return None
    freqs = [inv.get(f, 0) for f in frags]
    hard = sum(1 for f in freqs if f == 0) / len(freqs)
    soft = float(np.mean([soft_novel(f) for f in freqs]))
    return hard, soft


# --------------------------------------------------------------------------- #
def main() -> None:
    with gzip.open(FP_PATH, "rb") as fh:
        obj = pickle.load(fh)
    smiles_all = [r["canonical_smiles"] for r in obj["records"]]
    n = len(smiles_all)
    perm = np.random.default_rng(SEED).permutation(n)
    n_hold = int(round(n * HOLDOUT_FRAC))
    hold_idx = np.sort(perm[:n_hold])
    ref_idx = np.sort(perm[n_hold:])
    print(f"[split] ref_pool={len(ref_idx):,}  holdout={len(hold_idx):,}")

    # nested sub-samples: 50k ⊂ 200k ⊂ 500k
    rng = np.random.default_rng(SEED + 1)
    pick = rng.choice(ref_idx, REF_SIZES[-1], replace=False)
    ref_sets = {size: [smiles_all[i] for i in pick[:size]] for size in REF_SIZES}

    null_sel = random.Random(SEED).sample(list(hold_idx), NULL_SIZE)
    null_smi = [smiles_all[i] for i in null_sel]

    # generated sets (lean slice)
    bench = ROOT / "analysis/benchmark/benchmark5/five_model_repeats_500steps_260826"
    models = {"diffsbdd": "DiffSBDD", "diffgui": "DiffGui", "pmdm": "PMDM",
              "gcdm": "GCDM-SBDD", "targetdiff": "TargetDiff"}
    targets: list[tuple[str, Path]] = []
    rep4 = sorted(bench.glob("rep4_*"))
    for rep in rep4:
        for slug, name in models.items():
            sdf = rep / "filtered_sanitize_valid" / f"{slug}.sanitize_valid.sdf"
            if sdf.exists():
                targets.append((name, sdf))
    for step in E2PO_STEPS:
        sdf = ROOT / "OUT/diffsbdd_e2po_mainline_intermediate_gen" / step / "molecules_relaxed_largest.sdf"
        if sdf.exists():
            targets.append((step, sdf))
    print(f"[sets] {len(targets)} generated sets")

    rows: list[dict] = []
    for size in REF_SIZES:
        print(f"\n{'=' * 92}")
        print(f"|R| = {size:,}")
        print("=" * 92)
        inv = build_inventory(ref_sets[size], 8)
        print(f"[inv] {len(inv):,} fragments")

        # --- null ---
        null_h, null_s = [], []
        for smi in null_smi:
            r = score(Chem.MolFromSmiles(smi), inv)
            if r:
                null_h.append(r[0])
                null_s.append(r[1])
        null_h = np.asarray(null_h)
        null_s = np.asarray(null_s)

        print(f"[null] A(hard): median={np.median(null_h):.4f}  p95={np.percentile(null_h,95):.4f}  "
              f"share==0={float((null_h == 0).mean()):.3f}  std={null_h.std():.4f}")
        print(f"[null] B(soft): median={np.median(null_s):.4f}  p95={np.percentile(null_s,95):.4f}  "
              f"share==0={float((null_s == 0).mean()):.3f}  std={null_s.std():.4f}")

        p95_h, p95_s = np.percentile(null_h, 95), np.percentile(null_s, 95)

        # --- generated ---
        print(f"\n{'set':<16s}{'A_mean':>9s}{'A>p95':>9s}{'B_mean':>9s}{'B>p95':>9s}")
        for label, path in targets:
            hs, ss = [], []
            for mol in Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True):
                r = score(mol, inv)
                if r:
                    hs.append(r[0])
                    ss.append(r[1])
            hs = np.asarray(hs)
            ss = np.asarray(ss)
            print(f"{label:<16s}{hs.mean():>9.4f}{float((hs > p95_h).mean()):>9.3f}"
                  f"{ss.mean():>9.4f}{float((ss > p95_s).mean()):>9.3f}")
            rows.append({
                "ref_size": size,
                "n_fragments": len(inv),
                "set": label,
                "A_mean": round(float(hs.mean()), 4),
                "A_above_null_p95": round(float((hs > p95_h).mean()), 4),
                "B_mean": round(float(ss.mean()), 4),
                "B_above_null_p95": round(float((ss > p95_s).mean()), 4),
            })
        rows.append({
            "ref_size": size, "n_fragments": len(inv), "set": "__NULL__",
            "A_mean": round(float(null_h.mean()), 4), "A_above_null_p95": 0.05,
            "B_mean": round(float(null_s.mean()), 4), "B_above_null_p95": 0.05,
        })

    with (HERE / "phase0_refsize_scan.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[wrote] {HERE / 'phase0_refsize_scan.csv'}")


if __name__ == "__main__":
    main()
