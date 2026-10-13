#!/usr/bin/env python
"""Step 1: profile the existing 200k DiffSBDD candidate pool.

No generation is required — the pool already exists in ``runs/diffsbdd_collect``
(200,000 attempts, 135,533 RDKit-valid, checkpoint 07f86764... which was verified
to match ``DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt``, i.e. the same
base model E2PO starts from).

For every unique molecule we compute:

    reward      : soft_novel  (|R| = 500,000 reference, T = 1000)
    reference   : hard novel_frac
    matching    : element_composition, heavy_atom_count, tpsa, molwt, logp
    guard rails : murcko scaffold, n_fragments

The real-real null is recomputed at |R| = 500k (10,000 holdout molecules) so the
reward can be reported in calibrated form alongside the raw value.
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
from rdkit.Chem import BRICS, Crippen, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

ROOT = Path("/home/md666/data/disk6/DiffB110")
HERE = ROOT / "analysis/reference_support_experiment"
FP_PATH = ROOT / "data/chembl/chembl_36_ecfp4_fingerprints.pkl.gz"
COLLECT = ROOT / "runs/diffsbdd_collect"
INV_CACHE = HERE / "frag_inventory_500k.csv.gz"

SEED = 20260914
HOLDOUT_FRAC = 0.10
REF_SIZE = 500_000
NULL_SIZE = 10_000
SOFT_T = 1000

_INV: Counter | None = None


# --------------------------------------------------------------------------- #
def soft_novel(freq: int, t: int = SOFT_T) -> float:
    if freq <= 0:
        return 1.0
    return max(0.0, 1.0 - math.log1p(freq) / math.log1p(t))


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


def load_or_build_inventory(workers: int, smiles_ref: list[str]) -> Counter:
    if INV_CACHE.exists():
        inv: Counter = Counter()
        with gzip.open(INV_CACHE, "rt", newline="") as fh:
            for row in csv.DictReader(fh):
                inv[row["fragment"]] = int(row["count"])
        print(f"[inv] loaded cache: {len(inv):,} fragments")
        return inv
    print(f"[inv] building from |R|={len(smiles_ref):,} ...")
    chunks = [smiles_ref[i::workers] for i in range(workers)]
    with mp.Pool(workers) as pool:
        parts = pool.map(_brics_worker, chunks)
    inv = Counter()
    for p in parts:
        inv.update(p)
    with gzip.open(INV_CACHE, "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["fragment", "count"])
        for f, c in inv.most_common():
            w.writerow([f, c])
    print(f"[inv] built: {len(inv):,} fragments -> {INV_CACHE}")
    return inv


def _init_worker(inv: Counter) -> None:
    global _INV
    _INV = inv


def _profile_shard(path_str: str) -> list[dict]:
    inv = _INV
    out: list[dict] = []
    try:
        supplier = Chem.SDMolSupplier(path_str, removeHs=False, sanitize=True)
        mols = [m for m in supplier if m is not None]
    except Exception:
        return out
    for mol_idx, mol in enumerate(mols):
        try:
            frags = list(BRICS.BRICSDecompose(mol))
        except Exception:
            frags = []
        if not frags:
            continue
        freqs = [inv.get(f, 0) for f in frags]
        hard = sum(1 for f in freqs if f == 0) / len(freqs)
        soft = float(np.mean([soft_novel(f) for f in freqs]))
        try:
            smi = Chem.MolToSmiles(mol)
        except Exception:
            continue
        try:
            scaff = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        except Exception:
            scaff = ""
        comp = Counter(a.GetSymbol() for a in mol.GetAtoms())
        out.append({
            "smiles": smi,
            "shard": path_str,
            "shard_index": mol_idx,
            "soft_novel": round(soft, 5),
            "hard_novel_frac": round(hard, 5),
            "n_fragments": len(frags),
            "element_composition": "".join(f"{k}{v}" for k, v in sorted(comp.items())),
            "heavy_atom_count": mol.GetNumHeavyAtoms(),
            "tpsa": round(Descriptors.TPSA(mol), 3),
            "molwt": round(Descriptors.MolWt(mol), 3),
            "logp": round(Crippen.MolLogP(mol), 3),
            "murcko_scaffold": scaff,
        })
    return out


# --------------------------------------------------------------------------- #
def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)

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

    inv = load_or_build_inventory(8, smiles_ref)

    # ---------------- null at |R| = 500k ------------------------------------ #
    null_sel = random.Random(SEED).sample(list(hold_idx), NULL_SIZE)
    # holdout molecules live only as SMILES, so score the null inline
    null_soft, null_hard = [], []
    for i in null_sel:
        mol = Chem.MolFromSmiles(smiles_all[i])
        if mol is None:
            continue
        try:
            frags = list(BRICS.BRICSDecompose(mol))
        except Exception:
            continue
        if not frags:
            continue
        freqs = [inv.get(f, 0) for f in frags]
        null_hard.append(sum(1 for f in freqs if f == 0) / len(freqs))
        null_soft.append(float(np.mean([soft_novel(f) for f in freqs])))
    null_soft = np.asarray(null_soft)
    null_hard = np.asarray(null_hard)
    print(f"[null] n={len(null_soft):,}  soft mean={null_soft.mean():.4f} "
          f"median={np.median(null_soft):.4f} std={null_soft.std():.4f} p95={np.percentile(null_soft,95):.4f}")

    # ---------------- candidate pool ---------------------------------------- #
    shards = sorted(str(p) for p in COLLECT.rglob("molecules/**/*.sdf"))
    print(f"[pool] {len(shards):,} shard SDFs")

    rows: list[dict] = []
    with mp.Pool(8, initializer=_init_worker, initargs=(inv,)) as pool:
        for k, part in enumerate(pool.imap_unordered(_profile_shard, shards, chunksize=64)):
            rows.extend(part)
            if (k + 1) % 4000 == 0:
                print(f"   shards {k + 1}/{len(shards)}  molecules={len(rows):,}")
    print(f"[pool] raw molecules: {len(rows):,}")

    # dedup by canonical SMILES
    seen: set[str] = set()
    uniq: list[dict] = []
    for r in rows:
        if r["smiles"] in seen:
            continue
        seen.add(r["smiles"])
        uniq.append(r)
    print(f"[pool] unique molecules: {len(uniq):,} (dropped {len(rows) - len(uniq):,} duplicates)")

    # calibrated support: percentile inside the real-real null
    null_sorted = np.sort(null_soft)
    for r in uniq:
        r["null_percentile"] = round(
            float(np.searchsorted(null_sorted, r["soft_novel"], side="right") / null_sorted.size), 5
        )

    out_csv = HERE / "candidate_200k_profiles.csv"
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(uniq[0].keys()))
        w.writeheader()
        w.writerows(uniq)

    summary = {
        "source": str(COLLECT),
        "shards": len(shards),
        "raw_molecules": len(rows),
        "unique_molecules": len(uniq),
        "checkpoint_sha256_expected": "07f86764bf569aafbc40a9c15fc02de8e2550437dd0f17f657eab3abe66c372c",
        "ref_size": REF_SIZE,
        "soft_T": SOFT_T,
        "null": {
            "n": int(len(null_soft)),
            "mean": round(float(null_soft.mean()), 5),
            "median": round(float(np.median(null_soft)), 5),
            "std": round(float(null_soft.std()), 5),
            "p95": round(float(np.percentile(null_soft, 95)), 5),
        },
        "pool": {
            "soft_novel_mean": round(float(np.mean([r["soft_novel"] for r in uniq])), 5),
            "soft_novel_median": round(float(np.median([r["soft_novel"] for r in uniq])), 5),
            "hard_novel_frac_mean": round(float(np.mean([r["hard_novel_frac"] for r in uniq])), 5),
            "share_above_null_p95": round(
                float(np.mean([r["soft_novel"] > np.percentile(null_soft, 95) for r in uniq])), 5
            ),
            "unique_murcko_scaffolds": len({r["murcko_scaffold"] for r in uniq}),
        },
    }
    (HERE / "candidate_200k_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print()
    print(json.dumps(summary["pool"], indent=2))
    print(f"\n[wrote] {out_csv}")
    print(f"[wrote] {HERE / 'candidate_200k_summary.json'}")


if __name__ == "__main__":
    main()
