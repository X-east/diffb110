#!/usr/bin/env python
"""Verify the claimed properties of ``brics_hacb_soft`` on the DiffSBDD collection pool.

I claimed earlier that brics_hacb_soft (a) removes the heavy-atom-count confound
"by construction" and (b) stays orthogonal to docking.  Neither was measured.
This script measures both, side by side with the incumbent soft_novel:

  - definition-domain coverage  (share of pool molecules that get a value)
  - corr(metric, heavy_atom_count)     -> is the size confound really gone?
  - corr(metric, docking)              -> is it still orthogonal to docking?
  - all on the SAME molecule subset, so the comparison is apples-to-apples

Read-only: writes one JSON next to itself.
"""
from __future__ import annotations

import csv
import gzip
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import BRICS

RDLogger.DisableLog("rdApp.*")

HERE = Path("/home/md666/data/disk6/DiffB110/analysis/reference_support_experiment")
ROOT = Path("/home/md666/data/disk6/DiffB110")
FP_PATH = ROOT / "data/chembl/chembl_36_ecfp4_fingerprints.pkl.gz"
MANIFEST = HERE / "docking_subset/subset_manifest.csv"
DOCK = HERE / "docking_subset/docking/docking_results_all.csv"
OUT = HERE / "hacb_property_verification.json"

SEED = 20260914
HOLDOUT_FRAC = 0.10
REF_SIZE = 200_000
T_MAIN = 1000
MIN_BIN = 200

sys.path.insert(0, str(HERE))
from screen_simple_frag_definitions import build_reference, soft  # noqa: E402


def main() -> None:
    # ---- 1. reference split (identical to the screening runs) ----------------
    with gzip.open(FP_PATH, "rb") as fh:
        obj = pickle.load(fh)
    smiles_all = [r["canonical_smiles"] for r in obj["records"]]
    n = len(smiles_all)
    perm = np.random.default_rng(SEED).permutation(n)
    n_hold = int(round(n * HOLDOUT_FRAC))
    ref_idx = np.sort(perm[n_hold:])
    pick = np.random.default_rng(SEED + 1).choice(ref_idx, REF_SIZE, replace=False)
    smiles_ref = [smiles_all[i] for i in pick]
    print(f"[split] ref={len(smiles_ref):,}", flush=True)
    count, docf, byhac, hacn = build_reference(smiles_ref, 8)
    print(f"[inv] brics={len(count):,}  bins_total={len(byhac):,}  "
          f"bins_usable(>= {MIN_BIN})={sum(1 for h, c in hacn.items() if c >= MIN_BIN):,}", flush=True)

    # ---- 2. docking score by mol_id -----------------------------------------
    per_part: dict[str, list] = defaultdict(list)
    rows: list[dict] = []
    with open(MANIFEST) as fh:
        for r in csv.DictReader(fh):
            rows.append(r)
            per_part[f"part_{int(r['part']):04d}"].append(r)
    for p in per_part:
        per_part[p].sort(key=lambda r: int(r["mol_id"]))
    mol2dock: dict[str, float] = {}
    with open(DOCK) as fh:
        for r in csv.DictReader(fh):
            grp = per_part.get(r["part"])
            if grp is None or r.get("success") != "True":
                continue
            i = int(r["benchmark_index"])
            if i >= len(grp):
                continue
            try:
                mol2dock[grp[i]["mol_id"]] = float(r["score"])
            except ValueError:
                pass
    print(f"[dock] manifest={len(rows):,}  docked={len(mol2dock):,}", flush=True)

    # ---- 3. score every pool molecule --------------------------------------
    rec: dict[str, list] = defaultdict(list)
    n_no_frag = n_undefined = n_used = 0
    for r in rows:
        mol = Chem.MolFromSmiles(r["smiles"])
        fr = list(BRICS.BRICSDecompose(mol)) if mol is not None else []
        if not fr:
            n_no_frag += 1
            continue
        hac = mol.GetNumHeavyAtoms()
        d = mol2dock.get(r["mol_id"])
        if d is None:
            continue
        bin_c = byhac.get(hac)
        if bin_c is None or hacn.get(hac, 0) < MIN_BIN:
            n_undefined += 1                       # hacb has NO value here
            continue
        t_bin = max(10, int(hacn[hac]) // 2)
        rec["hacb"].append(float(np.mean([soft(bin_c.get(f, 0), t_bin) for f in fr])))
        rec["soft"].append(float(r["soft_novel"]))
        rec["hac"].append(float(hac))
        rec["dock"].append(d)
        n_used += 1

    hacb = np.asarray(rec["hacb"])
    softl = np.asarray(rec["soft"])
    hacv = np.asarray(rec["hac"])
    dock = np.asarray(rec["dock"])

    def cc(a, b):
        return float(np.corrcoef(a, b)[0, 1])

    res = {
        "n_manifest": len(rows),
        "n_scored": n_used,
        "n_no_brics_frag": n_no_frag,
        "n_hacb_undefined": n_undefined,
        "coverage_of_scored": n_used / max(1, (n_used + n_undefined)),
        "bins_usable": int(sum(1 for h, c in hacn.items() if c >= MIN_BIN)),
        "corr_hacb_vs_hac": cc(hacb, hacv),
        "corr_soft_vs_hac": cc(softl, hacv),
        "corr_hacb_vs_dock": cc(hacb, dock),
        "corr_soft_vs_dock": cc(softl, dock),
        "hacb_mean": float(hacb.mean()), "hacb_std": float(hacb.std()),
        "soft_mean": float(softl.mean()), "soft_std": float(softl.std()),
        "dock_mean": float(dock.mean()),
        "corr_hacb_vs_soft": cc(hacb, softl),
    }

    print("\n" + "=" * 74)
    print("brics_hacb_soft vs soft_novel  --  measured, same molecule subset")
    print("=" * 74)
    print(f"  scored molecules            : {n_used:,}")
    print(f"  hacb undefined (bin < {MIN_BIN}) : {n_undefined:,}   "
          f"-> coverage {res['coverage_of_scored']:.4f}")
    print(f"  unusable bins               : {res['bins_usable']:d}")
    print("-" * 74)
    print(f"  {'metric':<12s}{'corr(.,HAC)':>14s}{'corr(.,docking)':>18s}"
          f"{'mean':>9s}{'std':>9s}")
    print(f"  {'soft_novel':<12s}{res['corr_soft_vs_hac']:>+14.4f}"
          f"{res['corr_soft_vs_dock']:>+18.4f}{res['soft_mean']:>9.3f}{res['soft_std']:>9.3f}")
    print(f"  {'hacb_soft':<12s}{res['corr_hacb_vs_hac']:>+14.4f}"
          f"{res['corr_hacb_vs_dock']:>+18.4f}{res['hacb_mean']:>9.3f}{res['hacb_std']:>9.3f}")
    print("-" * 74)
    print(f"  corr(hacb, soft_novel) = {res['corr_hacb_vs_soft']:+.4f}")
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"[wrote] {OUT}")


if __name__ == "__main__":
    main()
