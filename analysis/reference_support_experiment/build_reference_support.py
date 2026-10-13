#!/usr/bin/env python
"""Reference-support profiling: SNN / BRICS-Frag / Murcko-Scaffold.

Experimental protocol (see the four validation questions Q1-Q4):
    Q1 discrimination : can the metric separate generated from real?
    Q2 sensitivity    : does the metric improve along the E2PO trajectory?
    Q3 docking link   : are high-docking molecules systematically less supported?
    Q4 decoupling     : are there molecules with low SNN but high fragment support?

Everything is calibrated against a *real-real* null: ChEMBL holdout molecules
scored against the ChEMBL reference, so no hand-set threshold is needed.

Usage
-----
    python build_reference_support.py --smoke          # tiny run, sanity check
    python build_reference_support.py                  # full run
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import multiprocessing as mp
import pickle
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import BRICS, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

ROOT = Path("/home/md666/data/disk6/DiffB110")
FP_PATH = ROOT / "data/chembl/chembl_36_ecfp4_fingerprints.pkl.gz"
OUT_DIR = ROOT / "analysis/reference_support_experiment"

SEED = 20260914
HOLDOUT_FRAC = 0.10
NULL_SIZE = 10_000
FRAG_FREQ_THRESHOLDS = (1, 10, 100)

# populated in main(); inherited by forked workers (copy-on-write)
_FPS: list = []
_NBITS: int = 2048
_RADIUS: int = 2


# --------------------------------------------------------------------------- #
# ChEMBL load / split
# --------------------------------------------------------------------------- #
def load_chembl(limit: int | None = None):
    """Return (smiles_list, fingerprint_list, n_bits, radius)."""
    t0 = time.time()
    with gzip.open(FP_PATH, "rb") as handle:
        obj = pickle.load(handle)
    fps = obj["fingerprints"]
    records = obj["records"]
    n_bits = int(obj["n_bits"])
    radius = int(obj["radius"])
    if limit:
        fps = fps[:limit]
        records = records[:limit]
    smiles = [rec["canonical_smiles"] for rec in records]
    print(
        f"[chembl] n={len(smiles):,}  n_bits={n_bits}  radius={radius}  "
        f"({time.time() - t0:.1f}s)"
    )
    return smiles, fps, n_bits, radius


def split_indices(n: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    n_hold = int(round(n * HOLDOUT_FRAC))
    hold = np.sort(perm[:n_hold])
    ref = np.sort(perm[n_hold:])
    return ref, hold


# --------------------------------------------------------------------------- #
# inventories
# --------------------------------------------------------------------------- #
def _mol_from_smiles(smi: str):
    try:
        return Chem.MolFromSmiles(smi)
    except Exception:
        return None


def _inventory_worker(chunk: list[str]) -> tuple[Counter, Counter]:
    frag_c: Counter = Counter()
    scaff_c: Counter = Counter()
    for smi in chunk:
        mol = _mol_from_smiles(smi)
        if mol is None:
            continue
        try:
            for frag in BRICS.BRICSDecompose(mol):
                frag_c[frag] += 1
        except Exception:
            pass
        try:
            scaff = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
            if scaff:
                scaff_c[scaff] += 1
        except Exception:
            pass
    return frag_c, scaff_c


def build_inventories(smiles_ref: list[str], workers: int) -> tuple[Counter, Counter]:
    t0 = time.time()
    chunks = [smiles_ref[i::workers] for i in range(workers)]
    with mp.Pool(workers) as pool:
        results = pool.map(_inventory_worker, chunks)
    frag_c: Counter = Counter()
    scaff_c: Counter = Counter()
    for fc, sc in results:
        frag_c.update(fc)
        scaff_c.update(sc)
    print(
        f"[inv] fragments={len(frag_c):,}  scaffolds={len(scaff_c):,}  "
        f"({time.time() - t0:.1f}s)"
    )
    return frag_c, scaff_c


# --------------------------------------------------------------------------- #
# per-molecule metrics
# --------------------------------------------------------------------------- #
def _ecfp(mol):
    from rdkit.Chem import AllChem

    return AllChem.GetMorganFingerprintAsBitVect(mol, _RADIUS, nBits=_NBITS)


def _snn_worker(fp) -> float:
    sims = DataStructs.BulkTanimotoSimilarity(fp, _FPS)
    return float(max(sims)) if sims else 0.0


def mol_structural(mol, frag_c: Counter, scaff_c: Counter) -> dict | None:
    """Per-molecule quantities that do NOT depend on the reference size."""
    if mol is None:
        return None
    try:
        frags = list(BRICS.BRICSDecompose(mol))
    except Exception:
        frags = []
    cov: dict[str, float] = {}
    for thr in FRAG_FREQ_THRESHOLDS:
        num = 0.0
        den = 0.0
        for frag in frags:
            freq = frag_c.get(frag, 0)
            w = math.log1p(freq) if freq > 0 else 1.0
            den += w
            if freq >= thr:
                num += w
        cov[f"frag_cov_thr{thr}"] = (num / den) if den > 0 else 0.0

    try:
        scaff = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:
        scaff = ""
    scaff_freq = scaff_c.get(scaff, 0) if scaff else 0
    return {
        **cov,
        "scaff_log": math.log1p(scaff_freq),
        "scaff_in_ref": 1 if scaff_freq > 0 else 0,
        "n_frags": len(frags),
    }


def score_mol_list(mols: list, frag_c: Counter, scaff_c: Counter, workers: int) -> list[dict]:
    """Score a list of RDKit mols (None entries skipped); SNN is batched."""
    good = [m for m in mols if m is not None]
    if not good:
        return []
    try:
        fps = [_ecfp(m) for m in good]
    except Exception:
        fps = []
    snns = batch_snn(fps, workers) if fps else [0.0] * len(good)
    out: list[dict] = []
    for mol, snn in zip(good, snns):
        m = mol_structural(mol, frag_c, scaff_c)
        if m is None:
            continue
        m["snn"] = float(snn)
        out.append(m)
    return out


def batch_snn(fps: list, workers: int) -> list[float]:
    with mp.Pool(workers) as pool:
        return pool.map(_snn_worker, fps, chunksize=8)


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #
class Null:
    """Holds raw null arrays; converts raw values to percentile-in-null."""

    def __init__(self, arrays: dict[str, np.ndarray]):
        self.arrays = {k: np.sort(np.asarray(v, dtype=float)) for k, v in arrays.items()}

    def support(self, key: str, value: float) -> float:
        arr = self.arrays.get(key)
        if arr is None or arr.size == 0:
            return float("nan")
        return float(np.searchsorted(arr, value, side="right") / arr.size)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny run for sanity check")
    ap.add_argument("--chembl-limit", type=int, default=None)
    ap.add_argument("--null-size", type=int, default=NULL_SIZE)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-sets", action="store_true", help="only build inventory+null")
    # |R| controls SNN directly (max over R). 1.4M saturates near 1.0 and loses
    # discriminative power, so we fix a MOSES-scale reference sub-sample.
    ap.add_argument("--ref-size", type=int, default=200_000)
    args = ap.parse_args()

    global _FPS, _NBITS, _RADIUS

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    limit = args.chembl_limit or (5_000 if args.smoke else None)
    smiles, fps, n_bits, radius = load_chembl(limit)
    _NBITS, _RADIUS = n_bits, radius

    n = len(smiles)
    ref_idx, hold_idx = split_indices(n)
    if args.ref_size and len(ref_idx) > args.ref_size:
        rng2 = np.random.default_rng(SEED + 1)
        ref_idx = np.sort(rng2.choice(ref_idx, args.ref_size, replace=False))
    print(f"[split] ref={len(ref_idx):,}  holdout={len(hold_idx):,}")

    # reference fingerprints (global, forked to workers)
    _FPS = [fps[i] for i in ref_idx]
    print(f"[ref] fingerprints ready: {len(_FPS):,}")

    # ---------------- inventories (BRICS / Murcko) ------------------------- #
    inv_path = OUT_DIR / "chembl_reference_inventory.json"
    if args.smoke:
        smiles_ref = [smiles[i] for i in ref_idx[: 3_000]]
    else:
        smiles_ref = [smiles[i] for i in ref_idx]
    frag_c, scaff_c = build_inventories(smiles_ref, args.workers)

    if not args.smoke:
        inv = {
            "n_reference": len(smiles_ref),
            "fragment_vocab": len(frag_c),
            "scaffold_vocab": len(scaff_c),
            "frag_freq_thresholds": list(FRAG_FREQ_THRESHOLDS),
            "seed": SEED,
            "n_bits": n_bits,
            "radius": radius,
        }
        inv_path.write_text(json.dumps(inv, indent=2), encoding="utf-8")
        with gzip.open(OUT_DIR / "frag_inventory.csv.gz", "wt", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["fragment", "count"])
            for frag, cnt in frag_c.most_common():
                w.writerow([frag, cnt])
        with gzip.open(OUT_DIR / "scaffold_inventory.csv.gz", "wt", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["scaffold", "count"])
            for scaff, cnt in scaff_c.most_common():
                w.writerow([scaff, cnt])
        print(f"[inv] wrote {inv_path}")

    # ---------------- real-real null --------------------------------------- #
    rng = random.Random(SEED)
    null_size = min(args.null_size, len(hold_idx))
    null_sel = rng.sample(list(hold_idx), null_size)
    print(f"[null] scoring {null_size:,} holdout molecules ...")
    t0 = time.time()
    null_rows: list[dict] = []
    chunk = 2000
    for start in range(0, len(null_sel), chunk):
        mols = [_mol_from_smiles(smiles[i]) for i in null_sel[start : start + chunk]]
        null_rows.extend(score_mol_list(mols, frag_c, scaff_c, args.workers))
        print(
            f"   null {min(start + chunk, len(null_sel))}/{len(null_sel)}"
            f"  ({time.time() - t0:.0f}s)"
        )
    print(f"[null] done n={len(null_rows):,}  ({time.time() - t0:.0f}s)")

    keys = ["snn", "scaff_log"] + [f"frag_cov_thr{t}" for t in FRAG_FREQ_THRESHOLDS]
    null = Null({k: [r[k] for r in null_rows] for k in keys})

    null_summary = {
        k: {
            "n": len(null.arrays[k]),
            "p05": float(np.percentile(null.arrays[k], 5)),
            "p25": float(np.percentile(null.arrays[k], 25)),
            "median": float(np.percentile(null.arrays[k], 50)),
            "p75": float(np.percentile(null.arrays[k], 75)),
            "p95": float(np.percentile(null.arrays[k], 95)),
        }
        for k in keys
    }
    (OUT_DIR / "chembl_null_summary.json").write_text(
        json.dumps(null_summary, indent=2), encoding="utf-8"
    )
    print("[null] summary:")
    for k, v in null_summary.items():
        print(
            f"   {k:<18s} p05={v['p05']:.4f}  median={v['median']:.4f}  p95={v['p95']:.4f}"
        )

    with (OUT_DIR / "chembl_null_distributions.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for row in null_rows:
            w.writerow({k: row[k] for k in keys})

    if args.skip_sets or args.smoke:
        print("\n[done] inventory + null only")
        return

    # ---------------- score generated sets --------------------------------- #
    targets = collect_targets()
    print(f"\n[sets] {len(targets)} target SDFs")

    out_csv = OUT_DIR / "metric_scores_by_set.csv"
    with out_csv.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["group", "set_label", "smiles", *keys, "scaff_in_ref", "n_frags"]
            + [f"sup_{k}" for k in keys]
        )
        for group, label, path in targets:
            if not Path(path).exists():
                print(f"   [skip missing] {label}: {path}")
                continue
            t1 = time.time()
            mols = list(Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True))
            n_fail = sum(1 for m in mols if m is None)
            rows = score_mol_list(mols, frag_c, scaff_c, args.workers)
            n_ok = len(rows)
            supports = {k: [] for k in keys}
            good_mols = [m for m in mols if m is not None]
            for mol, m in zip(good_mols, rows):
                sup = {k: null.support(k, m[k]) for k in keys}
                for k in keys:
                    supports[k].append(sup[k])
                try:
                    smi = Chem.MolToSmiles(mol)
                except Exception:
                    smi = ""
                writer.writerow(
                    [group, label, smi, *[m[k] for k in keys],
                     m["scaff_in_ref"], m["n_frags"], *[sup[k] for k in keys]]
                )
            print(
                f"   {label:<26s} ok={n_ok:<6d} fail={n_fail:<4d} "
                f"snn_sup={np.mean(supports['snn']) if supports['snn'] else float('nan'):.3f}  "
                f"frag_sup={np.mean(supports['frag_cov_thr10']) if supports['frag_cov_thr10'] else float('nan'):.3f}  "
                f"scaff_sup={np.mean(supports['scaff_log']) if supports['scaff_log'] else float('nan'):.3f}  "
                f"({time.time() - t1:.0f}s)"
            )
    print(f"\n[wrote] {out_csv}")


def collect_targets() -> list[tuple[str, str, str]]:
    """(group, label, sdf_path) for the zero-generation-cost pilot."""
    targets: list[tuple[str, str, str]] = []

    # 1) five-model 500-step benchmark (relaxed / sanitize-valid)
    bench_root = (
        ROOT / "analysis/benchmark/benchmark5/five_model_repeats_500steps_260826"
    )
    models = {
        "diffsbdd": "DiffSBDD",
        "diffgui": "DiffGui",
        "pmdm": "PMDM",
        "gcdm": "GCDM-SBDD",
        "targetdiff": "TargetDiff",
    }
    for rep in sorted(bench_root.glob("rep*_seed*")):
        for slug, name in models.items():
            sdf = rep / "filtered_sanitize_valid" / f"{slug}.sanitize_valid.sdf"
            if sdf.exists():
                targets.append(("five_model_500steps", f"{name}_{rep.name}", str(sdf)))

    # 2) E2PO mainline intermediate generations
    e2po_root = ROOT / "OUT/diffsbdd_e2po_mainline_intermediate_gen"
    for step_dir in sorted(e2po_root.glob("step_*")):
        sdf = step_dir / "molecules_relaxed_largest.sdf"
        if sdf.exists():
            targets.append(("e2po_mainline", step_dir.name, str(sdf)))

    return targets


if __name__ == "__main__":
    main()
