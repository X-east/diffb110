#!/usr/bin/env python
"""Screen SIMPLE fragment-realism definitions under the same real-real null.

Design constraint discovered earlier: a usable per-molecule realism metric must
  (1) not pile up at either end on REAL molecules   (else no resolution)
  (2) let GENERATED molecules exceed the real p95   (else no alarm)
  (3) be defined for every molecule                  (else silent selection bias)
  (4) be roughly orthogonal to docking               (else the preference axis fights docking)

Here we test four SIMPLE variants that only change ONE thing relative to the
incumbent ``brics_soft``:

    brics_soft       baseline: mean over BRICS fragments of 1 - log1p(freq)/log1p(1000)
    brics_min_soft   AGGREGATION -> worst fragment instead of the mean ("short-board")
    brics_df_soft    STATISTIC   -> document frequency (how many molecules) not raw count
    brics_hacb_soft  REFERENCE   -> frequency counted only among same-heavy-atom reference molecules

All four share one BRICS pass over the reference, so the run is cheap.
"""

from __future__ import annotations

import gzip
import json
import math
import multiprocessing as mp
import pickle
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import BRICS

RDLogger.DisableLog("rdApp.*")

ROOT = Path("/home/md666/data/disk6/DiffB110")
HERE = ROOT / "analysis/reference_support_experiment"
FP_PATH = ROOT / "data/chembl/chembl_36_ecfp4_fingerprints.pkl.gz"
BENCH = ROOT / "analysis/benchmark/benchmark5/five_model_repeats_500steps_260826"
OUT = HERE / "simple_frag_definitions_screen.json"

SEED = 20260914
HOLDOUT_FRAC = 0.10
REF_SIZE = 200_000
NULL_SIZE = 10_000
T_MAIN = 1000
T_DF = 100          # document frequency: "seen in >=100 of 200k molecules" == common

_INV: dict = {}


def soft(freq: int, t: int) -> float:
    if freq <= 0:
        return 1.0
    return max(0.0, 1.0 - math.log1p(freq) / math.log1p(t))


def brics_frags(mol) -> list[str]:
    try:
        return list(BRICS.BRICSDecompose(mol))
    except Exception:
        return []


# --------------------------------------------------------------------------- #
def _ref_worker(chunk: list[str]):
    count: Counter = Counter()          # raw occurrence
    docf: Counter = Counter()           # document frequency
    byhac: dict[int, Counter] = defaultdict(Counter)
    hacn: Counter = Counter()           # reference molecules per heavy-atom count
    for smi in chunk:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fr = brics_frags(mol)
        if not fr:
            continue
        hac = mol.GetNumHeavyAtoms()
        hacn[hac] += 1
        for f in fr:
            count[f] += 1
            byhac[hac][f] += 1
        for f in set(fr):
            docf[f] += 1
    return count, docf, byhac, hacn


def build_reference(smiles_ref: list[str], workers: int):
    chunks = [smiles_ref[i::workers] for i in range(workers)]
    with mp.Pool(workers) as pool:
        parts = pool.map(_ref_worker, chunks)
    count: Counter = Counter()
    docf: Counter = Counter()
    byhac: dict[int, Counter] = defaultdict(Counter)
    hacn: Counter = Counter()
    for c, d, b, n in parts:
        count.update(c)
        docf.update(d)
        for h, cc in b.items():
            byhac[h].update(cc)
        hacn.update(n)
    print(f"[inv] brics={len(count):,}  df={len(docf):,}  hac-bins={len(byhac):,}")
    return count, docf, byhac, hacn


# --------------------------------------------------------------------------- #
def metrics(mol, inv) -> dict | None:
    if mol is None:
        return None
    fr = brics_frags(mol)
    if not fr:
        return None
    count, docf, byhac, hacn = inv
    hac = mol.GetNumHeavyAtoms()
    out = {}
    # 1) baseline
    out["brics_soft"] = float(np.mean([soft(count.get(f, 0), T_MAIN) for f in fr]))
    # 2) aggregation: short-board
    out["brics_min_soft"] = float(min(soft(count.get(f, 0), T_MAIN) for f in fr))
    # 3) statistic: document frequency
    out["brics_df_soft"] = float(np.mean([soft(docf.get(f, 0), T_DF) for f in fr]))
    # 4) reference: same heavy-atom bin, threshold scaled to the bin size
    bin_c = byhac.get(hac)
    if bin_c is not None and hacn.get(hac, 0) >= 200:
        t_bin = max(10, int(hacn[hac]) // 2)
        out["brics_hacb_soft"] = float(np.mean([soft(bin_c.get(f, 0), t_bin) for f in fr]))
    out["hac"] = float(hac)
    return out


def score_set(mols, inv) -> dict[str, list]:
    acc: dict[str, list] = defaultdict(list)
    for mol in mols:
        m = metrics(mol, inv)
        if m is None:
            continue
        for k, v in m.items():
            acc[k].append(v)
    return {k: np.asarray(v) for k, v in acc.items()}


DEFS = ["brics_soft", "brics_min_soft", "brics_df_soft", "brics_hacb_soft"]


def main() -> None:
    global _INV
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

    _INV = build_reference(smiles_ref, 8)

    sel = random.Random(SEED).sample(list(hold_idx), NULL_SIZE)
    null = score_set([Chem.MolFromSmiles(smiles_all[i]) for i in sel], _INV)
    print(f"[null] n={null['brics_soft'].size:,}")

    sets: dict[str, dict] = {}
    for rep in sorted(BENCH.glob("rep*_seed*")):
        for slug in ("diffsbdd", "diffgui"):
            sdf = rep / "filtered_sanitize_valid" / f"{slug}.sanitize_valid.sdf"
            if sdf.exists():
                sets[f"{slug}_{rep.name.split('_')[0]}"] = score_set(
                    list(Chem.SDMolSupplier(str(sdf), removeHs=False, sanitize=True)), _INV
                )

    print("\n" + "=" * 112)
    print("Null spread on REAL molecules (10k ChEMBL holdout vs |R|=200k)")
    print("=" * 112)
    print(f"{'metric':<18s}{'p05':>9s}{'p25':>9s}{'median':>9s}{'p75':>9s}{'p95':>9s}"
          f"{'==0':>8s}{'==1':>8s}{'std':>8s}   flag")
    summary: dict = {}
    for d in DEFS:
        a = null.get(d)
        if a is None or a.size == 0:
            continue
        ps = {q: float(np.percentile(a, q)) for q in (5, 25, 50, 75, 95)}
        s0, s1 = float((a == 0).mean()), float((a >= 0.999999).mean())
        flags = []
        if s0 > 0.50:
            flags.append("floor-pileup")
        if s1 > 0.20 or ps[95] >= 0.999:
            flags.append("ceiling-pileup")
        if ps[95] - ps[5] < 0.25:
            flags.append("narrow-range")
        print(f"{d:<18s}{ps[5]:>9.4f}{ps[25]:>9.4f}{ps[50]:>9.4f}{ps[75]:>9.4f}{ps[95]:>9.4f}"
              f"{s0:>8.3f}{s1:>8.3f}{a.std():>8.4f}   {','.join(flags) if flags else 'OK'}")
        summary[d] = {**{f"p{q}": ps[q] for q in (5, 25, 50, 75, 95)},
                      "share_0": s0, "share_1": s1, "std": float(a.std()),
                      "flags": flags}

    print("\n" + "=" * 112)
    print("Alarm power: share of GENERATED molecules above the real p95")
    print("=" * 112)
    print(f"{'set':<20s}" + "".join(f"{d.replace('brics_',''):>16s}" for d in DEFS))
    sep: dict = {}
    for label in sorted(sets):
        row, rec = f"{label:<20s}", {}
        for d in DEFS:
            a = sets[label].get(d)
            base = null.get(d)
            if a is None or base is None or a.size == 0:
                row += f"{'-':>16s}"
                continue
            up = float((a > float(np.percentile(base, 95))).mean())
            rec[d] = up
            row += f"{up:>16.3f}"
        print(row)
        sep[label] = rec

    print("\n" + "=" * 112)
    print("Docking independence is NOT re-measured here (needs the docked pool join).")
    print("=" * 112)

    OUT.write_text(json.dumps({"null": summary, "alarm_power": sep}, indent=2), encoding="utf-8")
    print(f"[wrote] {OUT}")


if __name__ == "__main__":
    main()
