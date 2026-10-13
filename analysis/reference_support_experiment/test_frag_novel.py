#!/usr/bin/env python
"""Focused test: BRICS "novel fragment fraction" as a per-molecule realism signal.

This is the only fragment candidate whose real-real null is NOT saturated
(null: p05=p25=median=p75=0.0, p95=0.5), so it may retain resolution where
binary coverage collapses.  Direction is inverted: higher = worse.
"""

from __future__ import annotations

import csv
import gzip
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
FRAG_INV = HERE / "frag_inventory.csv.gz"

SEED = 20260914
HOLDOUT_FRAC = 0.10
NULL_SIZE = 10_000


def load_inventory() -> Counter:
    inv: Counter = Counter()
    with gzip.open(FRAG_INV, "rt", newline="") as fh:
        for row in csv.DictReader(fh):
            inv[row["fragment"]] = int(row["count"])
    return inv


def novel_frac(mol, inv: Counter):
    if mol is None:
        return None
    try:
        frags = list(BRICS.BRICSDecompose(mol))
    except Exception:
        return None
    if not frags:
        return None
    return sum(1 for f in frags if inv.get(f, 0) == 0) / len(frags)


def main() -> None:
    inv = load_inventory()
    with gzip.open(FP_PATH, "rb") as fh:
        obj = pickle.load(fh)
    smiles_all = [r["canonical_smiles"] for r in obj["records"]]
    n = len(smiles_all)
    perm = np.random.default_rng(SEED).permutation(n)
    n_hold = int(round(n * HOLDOUT_FRAC))
    hold_idx = np.sort(perm[:n_hold])

    null = []
    for i in random.Random(SEED).sample(list(hold_idx), NULL_SIZE):
        v = novel_frac(Chem.MolFromSmiles(smiles_all[i]), inv)
        if v is not None:
            null.append(v)
    null = np.asarray(null)
    p50, p95, p99 = (np.percentile(null, q) for q in (50, 95, 99))
    print(f"[null] n={len(null):,}  median={p50:.4f}  p95={p95:.4f}  p99={p99:.4f}")
    print(f"[null] share with novel_frac > 0 : {(null > 0).mean():.4f}")
    print(f"[null] share with novel_frac > 0.5: {(null > 0.5).mean():.4f}")

    # generated sets
    bench = ROOT / "analysis/benchmark/benchmark5/five_model_repeats_500steps_260826"
    models = {"diffsbdd": "DiffSBDD", "diffgui": "DiffGui", "pmdm": "PMDM",
              "gcdm": "GCDM-SBDD", "targetdiff": "TargetDiff"}
    targets = []
    for rep in sorted(bench.glob("rep*_seed*")):
        for slug, name in models.items():
            sdf = rep / "filtered_sanitize_valid" / f"{slug}.sanitize_valid.sdf"
            if sdf.exists():
                targets.append((f"{name}_{rep.name.split('_')[0]}", sdf))
    for step_dir in sorted((ROOT / "OUT/diffsbdd_e2po_mainline_intermediate_gen").glob("step_*")):
        sdf = step_dir / "molecules_relaxed_largest.sdf"
        if sdf.exists():
            targets.append((step_dir.name, sdf))

    print()
    print(f"{'set':<24s}{'n':>7s}{'mean':>9s}{'>0':>9s}{'>0.5':>9s}{'>null_p95':>12s}")
    for label, path in targets:
        vals = []
        for mol in Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True):
            v = novel_frac(mol, inv)
            if v is not None:
                vals.append(v)
        v = np.asarray(vals)
        print(
            f"{label:<24s}{len(v):>7d}{v.mean():>9.4f}{(v > 0).mean():>9.3f}"
            f"{(v > 0.5).mean():>9.3f}{(v > p95).mean():>12.3f}"
        )


if __name__ == "__main__":
    main()
