#!/usr/bin/env python3
"""Recompute a permissive but constrained three-member-ring policy."""
from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=28)
    return parser.parse_args()


def task(item: tuple[str, str]) -> dict[str, object]:
    trajectory_id, smiles = item
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(trajectory_id)
    rings = [set(r) for r in mol.GetRingInfo().AtomRings()]
    three = [ring for ring in rings if len(ring) == 3]
    invalid_reason: list[str] = []
    for ring in three:
        if any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in ring):
            invalid_reason.append("heteroatom")
        if any(other != ring and len(ring & other) >= 1 for other in rings):
            invalid_reason.append("fused_or_ring_shared_atom")
        external_neighbors = {neighbor.GetIdx() for i in ring for neighbor in mol.GetAtomWithIdx(i).GetNeighbors() if neighbor.GetIdx() not in ring}
        if len(external_neighbors) >= 2:
            invalid_reason.append("two_or_more_external_connections")
    return {
        "trajectory_id": trajectory_id,
        "three_member_ring_count_new": len(three),
        "three_member_rings_allowed_new": len(three) <= 2 and not invalid_reason,
        "three_member_ring_policy_failure_new": ";".join(sorted(set(invalid_reason))),
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    facts = pd.read_csv(args.facts, low_memory=False)
    valid = facts[facts.rdkit_valid.eq(1)].copy()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        extra = pd.DataFrame(pool.map(task, list(zip(valid.trajectory_id.astype(str), valid.canonical_smiles.astype(str))), chunksize=128))
    out = facts.merge(extra, on="trajectory_id", how="left", validate="one_to_one")
    out["normality_pre_chembl_new_three_ring_policy"] = out.normality_pre_chembl & out.three_member_rings_allowed_new.fillna(False)
    out.to_csv(args.output_dir / "facts_with_new_three_ring_policy.csv.gz", index=False, compression="gzip")
    rows = []
    for name, mask in {
        "rdkit_valid": out.rdkit_valid.eq(1),
        "old_normality_pre_chembl": out.normality_pre_chembl.eq(True),
        "new_three_ring_policy": out.three_member_rings_allowed_new.eq(True),
        "new_normality_pre_chembl": out.normality_pre_chembl_new_three_ring_policy.eq(True),
    }.items():
        rows.append({"scheme": name, "pass_count": int(mask.sum()), "rate_attempts": float(mask.sum() / 100000)})
    failure = valid.three_member_ring_count if "three_member_ring_count" in valid else pd.Series(dtype=int)
    pd.DataFrame(rows).to_csv(args.output_dir / "summary_counts.csv", index=False)
    out.three_member_ring_policy_failure_new.value_counts(dropna=False).rename_axis("failure").reset_index(name="count").to_csv(args.output_dir / "new_three_ring_failure_reasons.csv", index=False)
    (args.output_dir / "policy.txt").write_text("Allow 0-2 three-member rings; each must be all-carbon, share no atom with another ring, and have at most one external neighboring atom. All other existing filters unchanged.\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
