#!/usr/bin/env python
"""Which fragment partition is most "synthesis-anchored"?

If synthesis is the narrative anchor, the partition that best tracks synthetic
accessibility (SA score) is the one that belongs in the story.  We measure
corr(partition metric, sa_raw) on a sample of the DiffSBDD pool.

Partitions tested (all expressed as "novelty" so that higher = further from the
reference; a synthesis anchor predicts a POSITIVE correlation with SA score):

    brics_soft      fragment LIST           (drops connectivity)   <- incumbent
    brics_hard      same, binary
    pair_soft       ADJACENT FRAGMENT PAIRS (keeps co-occurrence)
    scaff_soft      Murcko scaffold
    ringsys_soft    ring-system components  (keeps ring fusion topology)
    brics_min_soft  worst-fragment only

brics/scaff inventories are reused at |R| = 200k (frag_inventory.csv.gz /
scaffold_inventory.csv.gz); pair and ring-system inventories are built here over
the same 200k reference so all metrics share one reference scale.
"""

from __future__ import annotations

import csv
import gzip
import math
import pickle
import random
import statistics
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import multiprocessing as mp
from rdkit import Chem, RDLogger
from rdkit.Chem import BRICS
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

ROOT = Path("/home/md666/data/disk6/DiffB110")
HERE = ROOT / "analysis/reference_support_experiment"
FP_PATH = ROOT / "data/chembl/chembl_36_ecfp4_fingerprints.pkl.gz"
SIDECAR = ROOT / "analysis/offline_iql_labels_alidiff_unidock_affinity_8gpu_full_260824/reward_sidecar.csv"
OUT = HERE / "synthesis_anchor_screen.json"

SEED = 20260914
HOLDOUT_FRAC = 0.10
REF_SIZE = 200_000
SAMPLE = 12_000
T = 1000

_INV: dict = {}


def soft(freq: int, t: int = T) -> float:
    if freq <= 0:
        return 1.0
    return max(0.0, 1.0 - math.log1p(freq) / math.log1p(t))


def brics_frags(mol) -> list[str]:
    try:
        return list(BRICS.BRICSDecompose(mol))
    except Exception:
        return []


def ring_systems(mol) -> list[str]:
    """Canonical SMILES of each fused ring system, keeping fusion topology."""
    ri = mol.GetRingInfo()
    rings = [set(r) for r in ri.AtomRings()]
    if not rings:
        return []
    parent = list(range(len(rings)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(rings)):
        for j in range(i + 1, len(rings)):
            if rings[i] & rings[j]:
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[rb] = ra
    groups: dict[int, set[int]] = defaultdict(set)
    for i, r in enumerate(rings):
        groups[find(i)] |= r
    out: list[str] = []
    for atoms in groups.values():
        idx = sorted(atoms)
        em = Chem.RWMol()
        amap = {}
        for a in idx:
            amap[a] = em.AddAtom(Chem.Atom(mol.GetAtomWithIdx(a).GetAtomicNum()))
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if i in amap and j in amap:
                em.AddBond(amap[i], amap[j], b.GetBondType())
        try:
            m2 = em.GetMol()
            Chem.SanitizeMol(m2)
            out.append(Chem.MolToSmiles(m2))
        except Exception:
            pass
    return out


def pair_keys(frags: list[str]) -> list[str]:
    if len(frags) < 2:
        return []
    return [f"{a}||{b}" for a, b in combinations(sorted(frags), 2)]


# --------------------------------------------------------------------------- #
def _ref_worker(chunk: list[str]):
    pair: Counter = Counter()
    ring: Counter = Counter()
    for smi in chunk:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        for k in pair_keys(brics_frags(mol)):
            pair[k] += 1
        for k in ring_systems(mol):
            ring[k] += 1
    return pair, ring


def build_new_inventories(smiles_ref: list[str], workers: int):
    chunks = [smiles_ref[i::workers] for i in range(workers)]
    with mp.Pool(workers) as pool:
        parts = pool.map(_ref_worker, chunks)
    pair: Counter = Counter()
    ring: Counter = Counter()
    for p, r in parts:
        pair.update(p)
        ring.update(r)
    print(f"[inv] pair={len(pair):,}  ringsys={len(ring):,}")
    return pair, ring


def load_gz_counter(path: Path) -> Counter:
    c: Counter = Counter()
    with gzip.open(path, "rt", newline="") as fh:
        for row in csv.DictReader(fh):
            c[row[list(row.keys())[0]]] = int(row["count"])
    return c


# --------------------------------------------------------------------------- #
def metrics(mol, inv) -> dict | None:
    if mol is None:
        return None
    bf = brics_frags(mol)
    if not bf:
        return None
    out: dict[str, float] = {}
    fq = [inv["brics"].get(f, 0) for f in bf]
    out["brics_soft"] = float(np.mean([soft(f) for f in fq]))
    out["brics_hard"] = sum(1 for f in fq if f == 0) / len(fq)
    out["brics_min_soft"] = float(min(soft(f) for f in fq))

    pk = pair_keys(bf)
    if pk:
        out["pair_soft"] = float(np.mean([soft(inv["pair"].get(k, 0)) for k in pk]))

    try:
        sc = MurckoScaffold.MurckoScaffoldSmiles(mol=mol) or ""
    except Exception:
        sc = ""
    if sc:
        out["scaff_soft"] = soft(inv["scaff"].get(sc, 0))

    rs = ring_systems(mol)
    if rs:
        out["ringsys_soft"] = float(np.mean([soft(inv["ring"].get(k, 0)) for k in rs]))

    out["n_brics"] = float(len(bf))
    out["n_rings"] = float(mol.GetRingInfo().NumRings())
    out["hac"] = float(mol.GetNumHeavyAtoms())
    return out


DEFS = ["brics_soft", "brics_hard", "brics_min_soft", "pair_soft", "scaff_soft", "ringsys_soft"]


def main() -> None:
    global _INV
    with gzip.open(FP_PATH, "rb") as fh:
        obj = pickle.load(fh)
    smiles_all = [r["canonical_smiles"] for r in obj["records"]]
    n = len(smiles_all)
    perm = np.random.default_rng(SEED).permutation(n)
    n_hold = int(round(n * HOLDOUT_FRAC))
    ref_idx = np.sort(perm[n_hold:])
    pick = np.random.default_rng(SEED + 1).choice(ref_idx, REF_SIZE, replace=False)
    smiles_ref = [smiles_all[i] for i in pick]
    print(f"[split] reference={len(smiles_ref):,}")

    brics_inv = load_gz_counter(HERE / "frag_inventory.csv.gz")
    scaff_inv = load_gz_counter(HERE / "scaffold_inventory.csv.gz")
    print(f"[inv] reused brics={len(brics_inv):,} scaff={len(scaff_inv):,}")
    pair_cache = HERE / "pair_inventory_200k.csv.gz"
    ring_cache = HERE / "ringsys_inventory_200k.csv.gz"
    if pair_cache.is_file() and ring_cache.is_file():
        pair_inv = load_gz_counter(pair_cache)
        ring_inv = load_gz_counter(ring_cache)
        print(f"[inv] loaded cache pair={len(pair_inv):,} ring={len(ring_inv):,}")
    else:
        pair_inv, ring_inv = build_new_inventories(smiles_ref, 8)
        for path, c in ((pair_cache, pair_inv), (ring_cache, ring_inv)):
            with gzip.open(path, "wt", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["key", "count"])
                for k, v in c.most_common():
                    w.writerow([k, v])
            print(f"[inv] cached -> {path.name}")
    _INV = {"brics": brics_inv, "scaff": scaff_inv, "pair": pair_inv, "ring": ring_inv}

    # ---- pool sample with SA scores --------------------------------------- #
    prof = pd.read_csv(HERE / "candidate_200k_profiles.csv",
                       usecols=["shard", "shard_index", "smiles"], low_memory=False)
    ep = pd.read_csv(HERE / "endpoint_index.csv",
                     usecols=["sample_id", "sdf_path", "sdf_index"], low_memory=False)
    sc = pd.read_csv(SIDECAR, usecols=["sample_id", "sa_raw", "qed", "docking_raw"], low_memory=False)
    prof["k"] = prof["shard"] + "#" + prof["shard_index"].astype(str)
    ep["k"] = ep["sdf_path"] + "#" + ep["sdf_index"].astype(str)
    pool = prof.merge(ep[["k", "sample_id"]], on="k", how="inner").merge(sc, on="sample_id", how="inner")
    pool = pool[np.isfinite(pool["sa_raw"])]
    print(f"[pool] joined with SA: {len(pool):,}")
    sel = pool.sample(min(SAMPLE, len(pool)), random_state=SEED)

    recs: list[dict] = []
    for smi, s in zip(sel["smiles"], sel["sa_raw"]):
        m = metrics(Chem.MolFromSmiles(smi), _INV)
        if m is None:
            continue
        m["sa_raw"] = float(s)
        recs.append(m)
    df = pd.DataFrame(recs)
    print(f"[sample] scored n={len(df):,}\n")

    print("=" * 100)
    print("与合成可及性 (sa_raw) 的相关性  —— 正相关越强 = 越贴近'合成'锚点")
    print("=" * 100)
    print(f"{'划分方式':<18}{'覆盖率':>9}{'Pearson':>10}{'Spearman':>10}"
          f"{'四分位跨度':>12}{'单调':>7}{'r2':>8}")
    res = {}
    for d in DEFS:
        sub = df[["sa_raw", d]].dropna()
        if len(sub) < 500:
            print(f"{d:<18}{len(sub):>9}  有效样本不足")
            continue
        a = sub[d].to_numpy()
        s = sub["sa_raw"].to_numpy()
        if np.std(a) == 0:
            print(f"{d:<18}{len(sub):>9}  指标为常数")
            continue
        r = float(np.corrcoef(a, s)[0, 1])
        rs = float(sub[d].corr(sub["sa_raw"], method="spearman"))
        q = np.quantile(s, [0.25, 0.5, 0.75])
        lab = np.digitize(s, q)
        means = [float(a[lab == i].mean()) for i in range(4)]
        span = means[-1] - means[0]
        mono = all(means[i + 1] >= means[i] for i in range(3))
        print(f"{d:<18}{len(sub) / len(df):>9.3f}{r:>+10.4f}{rs:>+10.4f}{span:>12.4f}"
              f"{('是' if mono else '否'):>7}{r * r:>8.3f}")
        res[d] = {"coverage": round(len(sub) / len(df), 4), "pearson": round(r, 4),
                  "spearman": round(rs, 4), "quartile_span": round(span, 4),
                  "monotone": mono, "quartile_means": [round(x, 4) for x in means]}

    print("\n" + "=" * 100)
    print("按 |Pearson| 排序")
    print("=" * 100)
    for k, v in sorted(res.items(), key=lambda x: -abs(x[1]["pearson"])):
        print(f"  {k:<18} r={v['pearson']:+.4f}  跨度={v['quartile_span']:.4f}  "
              f"单调={'是' if v['monotone'] else '否'}  四分位均值={v['quartile_means']}")

    import json as _json
    OUT.write_text(_json.dumps({"n": len(df), "results": res}, indent=2), encoding="utf-8")
    print(f"\n[wrote] {OUT}")


if __name__ == "__main__":
    main()
