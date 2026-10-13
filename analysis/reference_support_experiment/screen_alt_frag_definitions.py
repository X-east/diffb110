#!/usr/bin/env python
"""Screen ADDITIONAL fragment-based realism definitions with the same real-real null.

Motivation
----------
The original 7-candidate screen (``test_frag_definitions.py``) rejected 6 of 7
because the real-real null saturates at the ceiling: BRICS fragments of real
molecules are almost always present in the ChEMBL reference, so every
"coverage" definition pins at 1.0.

This script tests a second family of definitions that try to escape saturation
along four different axes:

    decomposition harder   -> RECAP instead of BRICS
    different unit         -> Murcko scaffold, fragment PAIRS
    different statistic    -> document frequency (molecules) vs raw occurrence
    size control           -> residual after regressing out heavy-atom count

Every candidate is scored with the SAME protocol:
    null  = 10,000 real ChEMBL holdout molecules vs the |R|=200k reference
    sets  = DiffSBDD / DiffGui 500-step benchmark (rep4)

Reported per candidate:
    null p05 / median / p95      -> is it saturated?  (need real spread)
    share == 0  /  share == 1    -> floor / ceiling pile-up
    set means + upper-tail share -> does it separate generated from real?
"""

from __future__ import annotations

import gzip
import json
import math
import multiprocessing as mp
import pickle
import random
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import BRICS, Recap
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

ROOT = Path("/home/md666/data/disk6/DiffB110")
HERE = ROOT / "analysis/reference_support_experiment"
FP_PATH = ROOT / "data/chembl/chembl_36_ecfp4_fingerprints.pkl.gz"
BENCH = ROOT / "analysis/benchmark/benchmark5/five_model_repeats_500steps_260826"
OUT = HERE / "alt_frag_definitions_screen.json"

SEED = 20260914
HOLDOUT_FRAC = 0.10
REF_SIZE = 200_000
NULL_SIZE = 10_000
SOFT_T = 1000

_INV: dict[str, Counter] = {}


# --------------------------------------------------------------------------- #
# decomposition primitives
# --------------------------------------------------------------------------- #
def brics_frags(mol) -> list[str]:
    try:
        return list(BRICS.BRICSDecompose(mol))
    except Exception:
        return []


def _recap_walk(node, acc: set[str]) -> None:
    for smi, child in node.GetAllChildren().items():
        if smi not in acc:
            acc.add(smi)
            _recap_walk(child, acc)


def recap_frags(mol) -> list[str]:
    try:
        acc: set[str] = set()
        _recap_walk(Recap.RecapDecompose(mol), acc)
        return sorted(acc)
    except Exception:
        return []


def scaff_smiles(mol) -> str:
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol) or ""
    except Exception:
        return ""


def frag_pairs(frags: list[str]) -> list[tuple[str, str]]:
    if len(frags) < 2:
        return []
    return [(a, b) for a, b in combinations(sorted(frags), 2)]


def soft(freq: int, t: int = SOFT_T) -> float:
    if freq <= 0:
        return 1.0
    return max(0.0, 1.0 - math.log1p(freq) / math.log1p(t))


# --------------------------------------------------------------------------- #
# inventory building (parallel over the reference set)
# --------------------------------------------------------------------------- #
def _inv_worker(chunk: list[str]) -> dict[str, Counter]:
    b: Counter = Counter()
    r: Counter = Counter()
    s: Counter = Counter()
    p: Counter = Counter()
    for smi in chunk:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        bf = brics_frags(mol)
        for f in bf:
            b[f] += 1
        for f in recap_frags(mol):
            r[f] += 1
        sc = scaff_smiles(mol)
        if sc:
            s[sc] += 1
        for pr in frag_pairs(bf):
            p[f"{pr[0]}||{pr[1]}"] += 1
    return {"brics": b, "recap": r, "scaff": s, "pair": p}


def build_inventories(smiles_ref: list[str], workers: int) -> dict[str, Counter]:
    chunks = [smiles_ref[i::workers] for i in range(workers)]
    with mp.Pool(workers) as pool:
        parts = pool.map(_inv_worker, chunks)
    out: dict[str, Counter] = {}
    for key in ("brics", "recap", "scaff", "pair"):
        c: Counter = Counter()
        for p in parts:
            c.update(p[key])
        out[key] = c
        print(f"[inv] {key}: {len(c):,} distinct")
    return out


# --------------------------------------------------------------------------- #
# per-molecule metrics
# --------------------------------------------------------------------------- #
def metrics(mol, inv: dict[str, Counter]) -> dict | None:
    if mol is None:
        return None
    bf = brics_frags(mol)
    rf = recap_frags(mol)
    sc = scaff_smiles(mol)
    pf = frag_pairs(bf)

    out: dict[str, float] = {}

    # --- baseline: BRICS soft novelty -------------------------------------- #
    if bf:
        out["brics_soft"] = float(np.mean([soft(inv["brics"].get(f, 0)) for f in bf]))
        out["brics_hard"] = sum(1 for f in bf if inv["brics"].get(f, 0) == 0) / len(bf)
    # --- RECAP (harder decomposition) -------------------------------------- #
    if rf:
        out["recap_soft"] = float(np.mean([soft(inv["recap"].get(f, 0)) for f in rf]))
        out["recap_hard"] = sum(1 for f in rf if inv["recap"].get(f, 0) == 0) / len(rf)
    # --- Murcko scaffold --------------------------------------------------- #
    if sc:
        freq = inv["scaff"].get(sc, 0)
        out["scaff_soft"] = soft(freq)
        out["scaff_hard"] = 1.0 if freq == 0 else 0.0
    # --- fragment PAIRS (rarer unit) --------------------------------------- #
    if pf:
        out["pair_soft"] = float(np.mean([soft(inv["pair"].get(f"{a}||{b}", 0)) for a, b in pf]))
        out["pair_hard"] = sum(1 for a, b in pf if inv["pair"].get(f"{a}||{b}", 0) == 0) / len(pf)
    # --- descriptors for the size confound --------------------------------- #
    out["n_brics"] = float(len(bf))
    out["hac"] = float(mol.GetNumHeavyAtoms())
    return out if out else None


def score_set(mols, inv) -> dict[str, np.ndarray]:
    acc: dict[str, list[float]] = {}
    for mol in mols:
        m = metrics(mol, inv)
        if m is None:
            continue
        for k, v in m.items():
            acc.setdefault(k, []).append(v)
    return {k: np.asarray(v) for k, v in acc.items()}


DEFS = ["brics_soft", "brics_hard", "recap_soft", "recap_hard",
        "scaff_soft", "scaff_hard", "pair_soft", "pair_hard"]


def main() -> None:
    global _INV
    print("[load] chembl fingerprints ...")
    with gzip.open(FP_PATH, "rb") as fh:
        obj = pickle.load(fh)
    smiles_all = [r["canonical_smiles"] for r in obj["records"]]
    n = len(smiles_all)
    perm = np.random.default_rng(SEED).permutation(n)
    n_hold = int(round(n * HOLDOUT_FRAC))
    hold_idx = np.sort(perm[:n_hold])
    ref_idx = np.sort(perm[n_hold:])
    pick = np.random.default_rng(SEED + 1).choice(ref_idx, REF_SIZE, replace=False)
    smiles_ref = [smiles_all[i] for i in pick]
    print(f"[split] ref={len(smiles_ref):,}  holdout={len(hold_idx):,}")

    _INV = build_inventories(smiles_ref, 8)

    # ---------------- real-real null --------------------------------------- #
    sel = random.Random(SEED).sample(list(hold_idx), NULL_SIZE)
    null_mols = [Chem.MolFromSmiles(smiles_all[i]) for i in sel]
    null = score_set(null_mols, _INV)
    print(f"\n[null] scored n={len(next(iter(null.values()))):,}")

    # ---------------- generated sets --------------------------------------- #
    sets: dict[str, dict[str, np.ndarray]] = {}
    for rep in sorted(BENCH.glob("rep*_seed*")):
        for slug in ("diffsbdd", "diffgui"):
            sdf = rep / "filtered_sanitize_valid" / f"{slug}.sanitize_valid.sdf"
            if not sdf.exists():
                continue
            label = f"{slug}_{rep.name.split('_')[0]}"
            mols = list(Chem.SDMolSupplier(str(sdf), removeHs=False, sanitize=True))
            sets[label] = score_set(mols, _INV)
            print(f"[set] {label}: n={len(mols)}")

    # ---------------- report ----------------------------------------------- #
    def col(name: str, arr: dict[str, np.ndarray]):
        return arr.get(name)

    print("\n" + "=" * 118)
    print("Null spread per candidate  (real ChEMBL holdout vs |R|=200k reference)")
    print("=" * 118)
    print(f"{'metric':<14s}{'p05':>9s}{'p25':>9s}{'median':>9s}{'p75':>9s}{'p95':>9s}"
          f"{'==0':>9s}{'==1':>9s}{'std':>9s}  verdict")
    null_summary: dict[str, dict] = {}
    for d in DEFS:
        a = col(d, null)
        if a is None or a.size == 0:
            continue
        p = {q: float(np.percentile(a, q)) for q in (5, 25, 50, 75, 95)}
        s0 = float((a == 0).mean())
        s1 = float((a >= 0.999999).mean())
        sd = float(a.std())
        saturated = s1 > 0.90 or (p[50] >= 0.95 and p[5] >= 0.80)
        floored = s0 > 0.90
        verdict = "CEILING-SATURATED" if saturated else ("FLOOR-PILEUP" if floored else "HAS SPREAD")
        print(f"{d:<14s}{p[5]:>9.4f}{p[25]:>9.4f}{p[50]:>9.4f}{p[75]:>9.4f}{p[95]:>9.4f}"
              f"{s0:>9.3f}{s1:>9.3f}{sd:>9.4f}  {verdict}")
        null_summary[d] = {**{f"p{q}": p[q] for q in (5, 25, 50, 75, 95)},
                           "share_0": s0, "share_1": s1, "std": sd, "verdict": verdict}

    print("\n" + "=" * 118)
    print("Separation: set mean, and share ABOVE the real p95 (upper tail = generated is worse)")
    print("=" * 118)
    hdr = f"{'set':<22s}" + "".join(f"{d.replace('_','')[:11]:>13s}" for d in DEFS if d in null)
    print(hdr)
    sep_summary: dict[str, dict] = {}
    for label in sorted(sets):
        row = f"{label:<22s}"
        rec: dict[str, float] = {}
        for d in DEFS:
            a = col(d, sets[label])
            nnull = col(d, null)
            if a is None or nnull is None or a.size == 0:
                row += f"{'-':>13s}"
                continue
            p95 = float(np.percentile(nnull, 95))
            p05 = float(np.percentile(nnull, 5))
            # upper-tail share is the meaningful anomaly side for novelty metrics
            up = float((a > p95).mean())
            rec[d] = up
            row += f"{up:>13.3f}"
        print(row)
        sep_summary[label] = rec

    # size confound
    hs = col("hac", null)
    print("\n" + "=" * 118)
    print("Size confound: corr(metric, heavy_atom_count) on real molecules")
    print("=" * 118)
    for d in DEFS:
        a = col(d, null)
        if a is None or hs is None:
            continue
        if a.size != hs.size or a.std() == 0:
            print(f"  {d:<14s}  n/a (constant or shape mismatch)")
            continue
        print(f"  {d:<14s}  r = {np.corrcoef(a, hs)[0,1]:+.4f}")

    OUT.write_text(json.dumps({"null": null_summary, "upper_tail_by_set": sep_summary},
                              indent=2), encoding="utf-8")
    print(f"\n[wrote] {OUT}")


if __name__ == "__main__":
    main()
