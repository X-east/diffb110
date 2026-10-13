#!/usr/bin/env python
"""Fill RDKit properties and UniDock scores for the substituted molecules.

SYNTHETIC COUNTERFACTUAL helper -- see README.md.  Output is not a model result.

Three stages:

    --stage collect   scan the epoch_* files, gather the unique substitute
                      SMILES, write an SDF for docking and a property CSV
    --stage merge     read the docking results back and rewrite every
                      epoch_* file with complete property/docking fields

Docking itself is run separately with the project's standard script:

    conda run -n dif python \\
      analysis/benchmark/8gjl_c1101_benchmark_20260624_100mol_500steps/scripts/dock_sdf_unidock.py \\
      --model SUB --input-sdf <subs.sdf> --out-dir <dock_dir> \\
      --exhaustiveness 16 --gpu-id 3

Field formulas were verified against the real run before use:
    ligand_efficiency_raw = -docking_raw / num_atoms        (exact)
    qed_reward            = qed_raw                        (exact)
    docking_reward        = -docking_raw / 10              (exact)
    sa_reward             = round((10 - sa_raw) / 9, 2)
"""
from __future__ import annotations

import argparse
import csv
import glob
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, QED, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

ROOT = Path("/home/md666/data/disk6/DiffB110")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import rdkit  # noqa: E402

sys.path.insert(0, str(Path(rdkit.__file__).resolve().parent / "Contrib/SA_Score"))
try:
    import sascorer  # noqa: E402
except Exception:  # pragma: no cover
    sascorer = None

SUBS_SDF = HERE / "pools/substitutes_unique.sdf"
SUBS_PROPS = HERE / "pools/substitutes_properties.csv"
DOCK_DIR = HERE / "pools/docking"
EPOCH_GLOB = str(HERE / "epochs/epoch_*_molecules.csv")

FILL_COLS = [
    "rdkit_valid", "sanitize_status", "reconstruction_status", "reconstruction_success",
    "num_atoms", "smiles", "canonical_smiles", "inchikey", "murcko_scaffold",
    "qed_raw", "qed_reward", "qed_success",
    "sa_raw", "sa_reward", "sa_success",
    "docking_success", "docking_raw", "docking_reward", "ligand_efficiency_raw",
]


def _sa(mol) -> float:
    return float(sascorer.calculateScore(mol)) if sascorer else float("nan")


def rdkit_props(smi: str) -> dict | None:
    """Per-molecule RDKit descriptors matching the real molecules.csv fields."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    qed = float(QED.qed(mol))
    sa = _sa(mol)
    return {
        "num_atoms": int(mol.GetNumHeavyAtoms()),
        "qed_raw": qed,
        "qed_reward": qed,
        "qed_success": 1,
        "sa_raw": sa,
        "sa_reward": round((10.0 - sa) / 9.0, 2),
        "sa_success": 1,
        "inchikey": Chem.MolToInchiKey(mol),
        "murcko_scaffold": MurckoScaffold.MurckoScaffoldSmiles(mol=mol),
        "molwt": float(Descriptors.MolWt(mol)),
        "logp": float(Crippen.MolLogP(mol)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "hba": int(rdMolDescriptors.CalcNumHBA(mol)),
        "hbd": int(rdMolDescriptors.CalcNumHBD(mol)),
        "ring_count": int(rdMolDescriptors.CalcNumRings(mol)),
        "aromatic_ring_count": int(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "carbon_fraction": (
            sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6)
            / max(1, mol.GetNumHeavyAtoms())
        ),
    }


def stage_collect() -> None:
    (HERE / "pools").mkdir(exist_ok=True)
    uniq: dict[str, int] = {}
    for p in sorted(glob.glob(EPOCH_GLOB)):
        for _, r in pd.read_csv(p, low_memory=False).iterrows():
            if not str(r.get("original_name", "")).startswith("sub_"):
                continue
            smi = str(r["canonical_smiles"]).strip()
            if smi and smi not in uniq:
                uniq[smi] = len(uniq)
    print(f"unique substitute SMILES: {len(uniq)}")

    rows, sdf_mols = [], []
    for smi, i in sorted(uniq.items(), key=lambda kv: kv[1]):
        pr = rdkit_props(smi)
        if pr is None:
            print(f"  [warn] unparseable, skipped: {smi}")
            continue
        mol = Chem.MolFromSmiles(smi)
        mol.SetProp("_Name", f"SUB_{i:04d}")
        sdf_mols.append(mol)
        rows.append({"sub_index": i, "smiles": smi, **pr})

    with Chem.SDWriter(str(SUBS_SDF)) as w:
        for m in sdf_mols:
            w.write(m)
    pd.DataFrame(rows).to_csv(SUBS_PROPS, index=False)
    print(f"[wrote] {SUBS_SDF}  ({len(sdf_mols)} mols)")
    print(f"[wrote] {SUBS_PROPS}")


def stage_merge() -> None:
    props = pd.read_csv(SUBS_PROPS)
    dock_csv = DOCK_DIR / "docking_results.csv"
    if not dock_csv.exists():
        raise SystemExit(f"docking results not found: {dock_csv}")
    dk = pd.read_csv(dock_csv)
    score = {}
    n_ok = 0
    for _, r in dk.iterrows():
        i = int(r["benchmark_index"])
        if str(r["success"]) != "True":
            continue
        score[i] = float(r["score"])
        n_ok += 1
    print(f"docking: {n_ok} success / {len(dk)} submitted")

    smi2dock = {}
    for _, r in props.iterrows():
        if int(r["sub_index"]) in score:
            smi2dock[r["smiles"]] = score[int(r["sub_index"])]
    print(f"substitutes with a score: {len(smi2dock)} / {len(props)}")

    n_total = 0
    for p in sorted(glob.glob(EPOCH_GLOB)):
        df = pd.read_csv(p, low_memory=False)
        for i in df.index[df["original_name"].astype(str).str.startswith("sub_")]:
            smi = str(df.at[i, "canonical_smiles"]).strip()
            pr = rdkit_props(smi)
            if pr is None:
                continue
            for c in FILL_COLS:
                if c in df.columns and c in pr:
                    df.at[i, c] = pr[c]
            s = smi2dock.get(smi)
            if s is not None:
                nac = int(pr["num_atoms"])
                df.at[i, "docking_success"] = 1
                df.at[i, "docking_raw"] = s
                df.at[i, "docking_reward"] = -s / 10.0
                df.at[i, "ligand_efficiency_raw"] = -s / max(1, nac)
            n_total += 1
        df.to_csv(p, index=False)
    print(f"filled {n_total} substitute rows across {len(glob.glob(EPOCH_GLOB))} epoch files")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["collect", "merge"], required=True)
    a = ap.parse_args()
    print("SYNTHETIC COUNTERFACTUAL -- NOT A MODEL RESULT")
    stage_collect() if a.stage == "collect" else stage_merge()


if __name__ == "__main__":
    main()
