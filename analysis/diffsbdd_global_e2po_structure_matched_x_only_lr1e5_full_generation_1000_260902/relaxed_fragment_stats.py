"""Relaxed structural validity: take largest fragment then sanitize.

For each molecule in the SDF:
1. strict: sanitize as-is
2. relaxed: keep largest fragment, then sanitize

Reports rates so we can compare "unreasonable structures" fairly and
decide whether fragment rescue helps docking/statistics.

Usage:
  python relaxed_fragment_stats.py --sdf <path> [--label NAME]
"""
import argparse
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Crippen, QED, Lipinski, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")


def largest_fragment(mol: Chem.Mol) -> Chem.Mol:
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if not frags:
        return mol
    return max(frags, key=lambda m: m.GetNumAtoms())


def analyze(sdf: str, label: str):
    sup = Chem.SDMolSupplier(sdf, sanitize=False, removeHs=False)
    n = 0
    strict_ok = 0
    relaxed_ok = 0
    relaxed_from_strict_fail = 0
    props = []
    for m in sup:
        n += 1
        if m is None:
            continue
        # strict
        strict_valid = False
        try:
            mm = Chem.Mol(m)
            Chem.SanitizeMol(mm)
            strict_valid = True
            strict_ok += 1
        except Exception:
            pass
        # relaxed: largest fragment then sanitize
        relaxed_valid = False
        rm = None
        try:
            lf = largest_fragment(m)
            rm = Chem.Mol(lf)
            Chem.SanitizeMol(rm)
            relaxed_valid = True
            relaxed_ok += 1
            if not strict_valid:
                relaxed_from_strict_fail += 1
        except Exception:
            pass
        if rm is not None and relaxed_valid:
            clean = Chem.RemoveHs(rm)
            try:
                props.append({
                    "mw": Descriptors.MolWt(clean),
                    "logp": Crippen.MolLogP(clean),
                    "tpsa": Descriptors.TPSA(clean),
                    "hac": clean.GetNumHeavyAtoms(),
                    "qed": QED.qed(clean),
                    "rings": rdMolDescriptors.CalcNumAromaticRings(clean),
                })
            except Exception:
                pass

    print(f"\n=== {label} ({Path(sdf).name}) ===")
    print(f"records                        : {n}")
    print(f"strict sanitize pass           : {strict_ok}/{n} = {100*strict_ok/n:.1f}%")
    print(f"relaxed (max frag) sanitize    : {relaxed_ok}/{n} = {100*relaxed_ok/n:.1f}%")
    print(f"  rescued from strict fail     : {relaxed_from_strict_fail}")
    print(f"relaxed 'unreasonable' (fail)  : {n-relaxed_ok}/{n} = {100*(n-relaxed_ok)/n:.1f}%")
    if props:
        import numpy as np
        arr = np.array([[p[k] for p in props] for k in
                        ["mw", "logp", "tpsa", "hac", "qed", "rings"]], dtype=float)
        keys = ["mw", "logp", "tpsa", "hac", "qed", "rings"]
        print("  relaxed-valid descriptor medians:")
        for i, k in enumerate(keys):
            print(f"    {k:5s} median={np.median(arr[i]):.2f} mean={arr[i].mean():.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sdf", action="append", required=True)
    ap.add_argument("--label", action="append", default=None)
    args = ap.parse_args()
    for i, sdf in enumerate(args.sdf):
        label = args.label[i] if args.label and i < len(args.label) else sdf
        analyze(sdf, label)


if __name__ == "__main__":
    main()
