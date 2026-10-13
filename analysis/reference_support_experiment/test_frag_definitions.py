#!/usr/bin/env python
"""Is BRICS-fragment support usable as a per-molecule realism signal?

The first attempt (binary coverage with freq >= {1,10,100}) failed Q1 because
the real-real null saturates at 1.0. Here we test 6 alternative per-molecule
fragment definitions plus the *set-level* MOSES Frag (cosine of fragment
frequency vectors), and report for each:

    - null spread (p05 / median / p95)
    - Q1 gate: share of generated molecules inside the real-real normal range
      (support >= 0.05); PASS requires < 0.30

Skips SNN entirely, so the run is cheap.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
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
IN_RANGE_P = 0.05

CANDIDATES = [
    "frag_cov_thr1",
    "frag_cov_thr10",
    "frag_cov_thr100",
    "frag_cov_thr1000",
    "frag_typ_logmean",   # mean of log(1+freq) over fragments
    "frag_min_log",       # log(1+min freq) -> rarity of the rarest piece
    "frag_novel_frac",    # fraction of fragments absent from the reference
]


def load_inventory() -> Counter:
    inv: Counter = Counter()
    with gzip.open(FRAG_INV, "rt", newline="") as fh:
        for row in csv.DictReader(fh):
            inv[row["fragment"]] = int(row["count"])
    return inv


def fragment_metrics(mol, inv: Counter) -> dict | None:
    if mol is None:
        return None
    try:
        frags = list(BRICS.BRICSDecompose(mol))
    except Exception:
        frags = []
    if not frags:
        return None

    freqs = [inv.get(f, 0) for f in frags]
    n = len(freqs)
    logs = [math.log1p(f) for f in freqs]

    out = {}
    for thr in (1, 10, 100, 1000):
        out[f"frag_cov_thr{thr}"] = sum(1 for f in freqs if f >= thr) / n
    out["frag_typ_logmean"] = float(np.mean(logs))
    out["frag_min_log"] = math.log1p(min(freqs))
    out["frag_novel_frac"] = sum(1 for f in freqs if f == 0) / n
    return out


def support(null_sorted: np.ndarray, value: float) -> float:
    return float(np.searchsorted(null_sorted, value, side="right") / null_sorted.size)


def main() -> None:
    inv = load_inventory()
    print(f"[inv] {len(inv):,} fragments")

    with gzip.open(FP_PATH, "rb") as fh:
        obj = pickle.load(fh)
    smiles_all = [r["canonical_smiles"] for r in obj["records"]]
    n = len(smiles_all)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    n_hold = int(round(n * HOLDOUT_FRAC))
    hold_idx = np.sort(perm[:n_hold])
    print(f"[split] holdout={len(hold_idx):,}")

    # ---------------- null -------------------------------------------------- #
    sel = random.Random(SEED).sample(list(hold_idx), NULL_SIZE)
    null_vals: dict[str, list[float]] = {k: [] for k in CANDIDATES}
    for idx in sel:
        mol = Chem.MolFromSmiles(smiles_all[idx])
        m = fragment_metrics(mol, inv)
        if m is None:
            continue
        for k in CANDIDATES:
            null_vals[k].append(m[k])
    null_arr = {k: np.sort(np.asarray(v)) for k, v in null_vals.items()}

    print()
    print("=" * 100)
    print("Null spread per candidate definition")
    print("=" * 100)
    print(f"{'metric':<20s}{'p05':>10s}{'p25':>10s}{'median':>10s}{'p75':>10s}{'p95':>10s}")
    for k in CANDIDATES:
        a = null_arr[k]
        print(
            f"{k:<20s}{np.percentile(a,5):>10.4f}{np.percentile(a,25):>10.4f}"
            f"{np.percentile(a,50):>10.4f}{np.percentile(a,75):>10.4f}"
            f"{np.percentile(a,95):>10.4f}"
        )

    # ---------------- generated sets ---------------------------------------- #
    bench_root = ROOT / "analysis/benchmark/benchmark5/five_model_repeats_500steps_260826"
    models = {"diffsbdd": "DiffSBDD", "diffgui": "DiffGui", "pmdm": "PMDM",
              "gcdm": "GCDM-SBDD", "targetdiff": "TargetDiff"}
    targets = []
    for rep in sorted(bench_root.glob("rep*_seed*")):
        for slug, name in models.items():
            sdf = rep / "filtered_sanitize_valid" / f"{slug}.sanitize_valid.sdf"
            if sdf.exists():
                targets.append(("five_model", f"{name}_{rep.name.split('_')[0]}", sdf))
    e2po_root = ROOT / "OUT/diffsbdd_e2po_mainline_intermediate_gen"
    for step_dir in sorted(e2po_root.glob("step_*")):
        sdf = step_dir / "molecules_relaxed_largest.sdf"
        if sdf.exists():
            targets.append(("e2po", step_dir.name, sdf))

    results: dict[str, dict[str, np.ndarray]] = {}
    frag_freq_sets: dict[str, Counter] = {}
    for _, label, path in targets:
        vals: dict[str, list[float]] = {k: [] for k in CANDIDATES}
        freq_c: Counter = Counter()
        for mol in Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True):
            if mol is None:
                continue
            try:
                frags = list(BRICS.BRICSDecompose(mol))
            except Exception:
                frags = []
            freq_c.update(frags)
            m = fragment_metrics(mol, inv)
            if m is None:
                continue
            for k in CANDIDATES:
                vals[k].append(m[k])
        results[label] = {k: np.asarray(v) for k, v in vals.items()}
        frag_freq_sets[label] = freq_c

    # ---------------- Q1 gate per candidate --------------------------------- #
    print()
    print("=" * 100)
    print(f"Q1 gate — share with support >= {IN_RANGE_P} (PASS requires < 0.30)")
    print("=" * 100)
    header = f"{'set':<22s}" + "".join(f"{k.replace('frag_',''):>13s}" for k in CANDIDATES)
    print(header)
    for label in sorted(results):
        row = f"{label:<22s}"
        for k in CANDIDATES:
            v = results[label][k]
            frac = float((np.asarray([support(null_arr[k], x) for x in v]) >= IN_RANGE_P).mean())
            row += f"{frac:>13.3f}"
        print(row)

    # ---------------- set-level MOSES Frag (cosine) -------------------------- #
    print()
    print("=" * 100)
    print("Set-level MOSES Frag = cosine(reference fragment freq, set fragment freq)")
    print("=" * 100)
    vocab = sorted(set(inv) | {f for c in frag_freq_sets.values() for f in c})
    idx = {f: i for i, f in enumerate(vocab)}
    ref_vec = np.zeros(len(vocab))
    for f, c in inv.items():
        ref_vec[idx[f]] = c
    ref_norm = np.linalg.norm(ref_vec)
    print(f"{'set':<22s}{'MOSES_Frag':>14s}")
    for label in sorted(frag_freq_sets):
        vec = np.zeros(len(vocab))
        for f, c in frag_freq_sets[label].items():
            vec[idx[f]] = c
        denom = np.linalg.norm(vec) * ref_norm
        cos = float(vec @ ref_vec / denom) if denom > 0 else 0.0
        print(f"{label:<22s}{cos:>14.4f}")

    out = {
        "null_spread": {
            k: {
                "p05": float(np.percentile(null_arr[k], 5)),
                "median": float(np.percentile(null_arr[k], 50)),
                "p95": float(np.percentile(null_arr[k], 95)),
            }
            for k in CANDIDATES
        }
    }
    (HERE / "frag_definition_test.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[wrote] {HERE / 'frag_definition_test.json'}")


if __name__ == "__main__":
    main()
