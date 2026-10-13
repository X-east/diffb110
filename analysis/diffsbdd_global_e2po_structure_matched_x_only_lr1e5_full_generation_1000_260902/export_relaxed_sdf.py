"""Export largest-fragment + sanitize-valid molecules from a raw SDF.

Writes one SDF per input: only molecules that survive (largest fragment,
sanitize). Keeps original benchmark_index via molecule order so it can be
aligned with docking CSVs that use record order.

Usage:
  python export_relaxed_sdf.py --sdf IN --out OUT
"""
import argparse

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


def largest_fragment(mol: Chem.Mol) -> Chem.Mol:
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if not frags:
        return mol
    return max(frags, key=lambda m: m.GetNumAtoms())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sdf", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sup = Chem.SDMolSupplier(args.sdf, sanitize=False, removeHs=False)
    writer = Chem.SDWriter(args.out)
    kept = 0
    for i, m in enumerate(sup, 1):
        if m is None:
            continue
        try:
            rm = Chem.Mol(largest_fragment(m))
            Chem.SanitizeMol(rm)
        except Exception:
            continue
        rm.SetProp("benchmark_index", str(i - 1))
        writer.write(rm)
        kept += 1
    writer.close()
    print(f"wrote {kept} molecules -> {args.out}")


if __name__ == "__main__":
    main()
