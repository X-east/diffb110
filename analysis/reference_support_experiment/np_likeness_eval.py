#!/usr/bin/env python
"""Literature-grounded evaluation of "deviation from real chemical space".

Instead of a self-invented metric, this uses two *published* per-molecule scores
that are already implemented in the local RDKit install:

  NP-likeness  Ertl, Roggo, Schuffenhauer, J. Chem. Inf. Model. 48, 68-74 (2008)
               doi:10.1021/ci700286x
               impl: rdkit/Contrib/NP_Score/npscorer.py  (+ publicnp.model.gz)
               range -5..5; higher = closer to natural-product structural space

  SA score     Ertl & Schuffenhauer, J. Cheminform. 1:8 (2009)
               impl: rdkit/Contrib/SA_Score/sascorer.py
               range 1..10; lower = easier to synthesise

Sets scored (all already on disk, nothing regenerated):
  - real ChEMBL holdout                  (the reference distribution)
  - 5 baselines x 3 repeats              benchmark5 filtered_sanitize_valid
  - E2PO trajectory, 10 checkpoints      OUT/diffsbdd_e2po_mainline_intermediate_gen

The only aggregation reported is the mean percentile of a set inside the real
distribution (equivalently AUC = P(x_gen > x_real)).  0.5 is the mathematical
reference point, so no threshold has to be chosen by hand.

Read-only w.r.t. every input; writes one CSV and one JSON next to itself.
"""
from __future__ import annotations

import csv
import gzip
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = Path("/home/md666/data/disk6/DiffB110")
HERE = ROOT / "analysis/reference_support_experiment"
FP_PATH = ROOT / "data/chembl/chembl_36_ecfp4_fingerprints.pkl.gz"
BENCH = ROOT / "analysis/benchmark/benchmark5/five_model_repeats_500steps_260826"
E2PO = ROOT / "OUT/diffsbdd_e2po_mainline_intermediate_gen"
OUT_CSV = HERE / "np_likeness_eval.csv"
OUT_JSON = HERE / "np_likeness_eval.json"

RDKIT = Path("/home/md666/miniforge3/lib/python3.12/site-packages/rdkit")
sys.path.insert(0, str(RDKIT / "Contrib/NP_Score"))
sys.path.insert(0, str(RDKIT / "Contrib/SA_Score"))
import npscorer  # noqa: E402
import sascorer  # noqa: E402

SEED = 20260914
HOLDOUT_FRAC = 0.10
NULL_SIZE = 20_000
SLUGS = ("diffsbdd", "diffgui", "gcdm", "pmdm", "targetdiff")

_FS: dict = {}  # NP-likeness fragment model, loaded in main()


def score_mols(mols) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (np_likeness, np_confidence, sa_score) arrays for valid molecules."""
    np_s, conf, sa = [], [], []
    for m in mols:
        if m is None:
            continue
        r = npscorer.scoreMolWConfidence(m, _FS)
        np_s.append(float(r.nplikeness))
        conf.append(float(r.confidence))
        sa.append(float(sascorer.calculateScore(m)))
    return np.asarray(np_s), np.asarray(conf), np.asarray(sa)


def pct_above(null_sorted: np.ndarray, values: np.ndarray) -> float:
    """Mean percentile (== AUC == P(x_gen > x_real)) of a set inside the null."""
    idx = np.searchsorted(null_sorted, values, side="right")
    return float((idx / null_sorted.size).mean())


def load_sdf(path: Path):
    return list(Chem.SDMolSupplier(str(path), sanitize=True, removeHs=False))


def main() -> None:
    global _FS
    _FS = npscorer.readNPModel()

    # ---- reference distribution: real ChEMBL holdout ------------------------
    with gzip.open(FP_PATH, "rb") as fh:
        obj = pickle.load(fh)
    smiles_all = [r["canonical_smiles"] for r in obj["records"]]
    perm = np.random.default_rng(SEED).permutation(len(smiles_all))
    hold = np.sort(perm[: int(round(len(smiles_all) * HOLDOUT_FRAC))])
    pick = np.random.default_rng(SEED + 2).choice(hold, NULL_SIZE, replace=False)
    print(f"[real] scoring {NULL_SIZE:,} ChEMBL holdout molecules ...", flush=True)
    real_np, real_conf, real_sa = score_mols(
        [Chem.MolFromSmiles(smiles_all[i]) for i in pick]
    )
    null_np = np.sort(real_np)
    null_sa = np.sort(real_sa)

    sets: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    sets["REAL_chembl"] = (real_np, real_conf, real_sa)

    print("[bench] scoring 5 baselines x 3 repeats ...", flush=True)
    for rep in sorted(BENCH.glob("rep*_seed*")):
        for slug in SLUGS:
            sdf = rep / "filtered_sanitize_valid" / f"{slug}.sanitize_valid.sdf"
            if not sdf.exists():
                continue
            sets[f"{slug}_{rep.name.split('_')[0]}"] = score_mols(load_sdf(sdf))

    print("[e2po] scoring 10 trajectory checkpoints ...", flush=True)
    for step in sorted(E2PO.glob("step_*")):
        sdf = step / "molecules.sdf"
        if sdf.exists():
            sets[step.name] = score_mols(load_sdf(sdf))

    # ---- write per-molecule CSV --------------------------------------------
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["set", "i", "np_likeness", "np_confidence", "sa_score"])
        for name, (a, c, s) in sets.items():
            for i in range(a.size):
                w.writerow([name, i, f"{a[i]:.6f}", f"{c[i]:.6f}", f"{s[i]:.6f}"])

    # ---- summary -----------------------------------------------------------
    summary: dict = {}
    print("\n" + "=" * 104)
    print("NP-likeness (Ertl 2008) and SA score (Ertl 2009) on real vs generated sets")
    print("=" * 104)
    print(f"{'set':<20s}{'n':>6s}{'NP mean':>10s}{'NP med':>9s}{'NP<0':>8s}"
          f"{'conf':>7s}{'np_pct':>9s}{'SA med':>9s}{'sa_pct':>9s}")
    print("-" * 104)
    for name, (a, c, s) in sets.items():
        pct = pct_above(null_np, a)
        pct_sa = pct_above(null_sa, s)
        summary[name] = {
            "n": int(a.size),
            "np_mean": float(a.mean()), "np_median": float(np.median(a)),
            "np_share_neg": float((a < 0).mean()),
            "np_confidence_mean": float(c.mean()),
            "np_percentile": pct,
            "sa_median": float(np.median(s)), "sa_percentile": pct_sa,
        }
        print(f"{name:<20s}{a.size:>6d}{a.mean():>10.3f}{np.median(a):>9.3f}"
              f"{(a < 0).mean():>8.3f}{c.mean():>7.3f}{pct:>9.3f}"
              f"{np.median(s):>9.3f}{pct_sa:>9.3f}")
    print("-" * 104)
    print("np_pct = mean percentile inside the real distribution (== AUC == P(gen > real));")
    print("         0.500 == indistinguishable from real molecules.")

    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[wrote] {OUT_CSV}\n[wrote] {OUT_JSON}")


if __name__ == "__main__":
    main()
