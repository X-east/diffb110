#!/usr/bin/env python3
"""Apply the strict chemistry screen to the 53 x 100 EPI-nearest records."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--facts", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(args.input_dir.glob("RND-*_nearest100.csv.gz"))
    if len(files) != 53:
        raise ValueError(f"expected 53 files, got {len(files)}")
    pairs = pd.concat([pd.read_csv(path, low_memory=False) for path in files], ignore_index=True)
    base = pd.read_csv(args.facts, low_memory=False)
    ring = base[["trajectory_id", "three_member_ring_count_new", "three_member_rings_allowed_new", "three_member_ring_policy_failure_new"]]
    pairs = pairs.merge(ring, on="trajectory_id", how="left", validate="many_to_one")
    required = [
        "rdkit_valid", "qed_ge_0_4", "sa_le_6", "logp_in_neg1_to_5", "tpsa_in_20_to_140",
        "fused_rings_le_4", "complex_rings_le_1", "brenk_eq_0", "single_fragment",
        "allowed_organic_elements", "mol_wt_in_120_to_600", "heavy_atoms_in_10_to_35",
        "hba_le_10", "hbd_le_5", "formal_charge_abs_le_1", "lipinski_violations_le_1",
        "carbon_fraction_le_0_85", "three_member_rings_allowed_new",
    ]
    pairs["strict_brenk_new_three_ring"] = pairs[required].fillna(False).astype(bool).all(axis=1)
    selected = pairs[pairs.strict_brenk_new_three_ring].copy()
    selected.to_csv(args.output_dir / "rnd53_nearest100_strict_pairs.csv.gz", index=False, compression="gzip")
    unique = selected.sort_values(["trajectory_id", "ecfp4_distance", "epi_id"], kind="mergesort").drop_duplicates("trajectory_id").copy()
    unique["murcko_scaffold"] = [MurckoScaffold.MurckoScaffoldSmilesFromSmiles(str(s)) or "__acyclic__" for s in unique.canonical_smiles]
    unique.to_csv(args.output_dir / "rnd53_nearest100_strict_unique.csv.gz", index=False, compression="gzip")
    wanted = set(unique.trajectory_id.astype(str))
    writer = Chem.SDWriter(str(args.output_dir / "rnd53_nearest100_strict_unique.sdf")); writer.SetKekulize(False)
    count = 0
    try:
        for sdf_name in sorted(unique.sdf_path.dropna().astype(str).unique()):
            for mol in Chem.SDMolSupplier(sdf_name, removeHs=False, sanitize=False):
                if mol is not None and mol.HasProp("sample_id") and mol.GetProp("sample_id") in wanted:
                    mol.SetProp("screen_label", "rnd53_nearest100_strict_unique")
                    writer.write(mol); count += 1
    finally:
        writer.close()
    by_epi = selected.groupby(["epi_id", "epi_preferred_name"], dropna=False).agg(
        pair_pass_count=("trajectory_id", "size"), unique_pass_count=("trajectory_id", "nunique"),
    ).reset_index()
    by_epi.to_csv(args.output_dir / "rnd53_nearest100_strict_by_epi.csv", index=False)
    summary = {
        "status": "complete", "epi_files": len(files), "input_pair_records": len(pairs),
        "input_unique_trajectory_ids": int(pairs.trajectory_id.nunique()),
        "strict_pair_pass_count": len(selected), "strict_unique_pass_count": len(unique),
        "unique_sdf_count": count, "unique_scaffold_count": int(unique.murcko_scaffold.nunique()),
        "condition": "all requested gates plus new three-member-ring policy",
        "note": "Pair counts can duplicate a molecule when it is nearest to multiple EPIs; unique counts are trajectory_id deduplicated.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
