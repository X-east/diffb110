#!/usr/bin/env python3
"""Find nearest valid DiffSBDD molecules for each of 53 RND/EPI molecules."""
from __future__ import annotations

import argparse
import gzip
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epi", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument(
        "--strict-facts",
        type=Path,
        default=None,
        help="Optional facts table used for the existing strict chemistry screen.",
    )
    return parser.parse_args()


STRICT_COLUMNS = [
    "rdkit_valid", "qed_ge_0_4", "sa_le_6", "logp_in_neg1_to_5", "tpsa_in_20_to_140",
    "fused_rings_le_4", "complex_rings_le_1", "brenk_eq_0", "single_fragment",
    "allowed_organic_elements", "mol_wt_in_120_to_600", "heavy_atoms_in_10_to_35",
    "hba_le_10", "hbd_le_5", "formal_charge_abs_le_1", "lipinski_violations_le_1",
    "carbon_fraction_le_0_85", "three_member_rings_allowed_new",
]


def aggregate_by_trajectory(pairs: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate pair rows while retaining all reference provenance."""
    rows = []
    for trajectory_id, group in pairs.groupby("trajectory_id", sort=False):
        best = group.sort_values(
            ["ecfp4_distance", "epi_id", "epi_rank"], kind="mergesort"
        ).iloc[0].copy()
        best["source_epi_ids"] = ";".join(sorted(group["epi_id"].astype(str).unique()))
        best["source_epi_count"] = int(group["epi_id"].nunique())
        best["source_pair_count"] = int(len(group))
        best["best_epi_rank"] = int(group["epi_rank"].min())
        best["trajectory_id"] = trajectory_id
        rows.append(best)
    return pd.DataFrame(rows).reset_index(drop=True)


def strict_filter(pairs: pd.DataFrame, facts_path: Path) -> pd.DataFrame:
    """Apply the established strict chemistry gates after nearest-neighbor selection."""
    facts = pd.read_csv(facts_path, low_memory=False)
    missing = set(STRICT_COLUMNS) - set(facts.columns)
    if missing:
        raise ValueError(f"strict facts table lacks columns: {sorted(missing)}")
    if facts["trajectory_id"].duplicated().any():
        raise ValueError("strict facts trajectory_id is not unique")
    missing = [column for column in STRICT_COLUMNS if column not in pairs.columns]
    if missing:
        flags = facts[["trajectory_id", *missing]].copy()
        pairs = pairs.merge(flags, on="trajectory_id", how="left", validate="many_to_one")
    if pairs[STRICT_COLUMNS].isna().any().any():
        raise ValueError("some selected trajectories lack strict chemistry facts")
    mask = pairs[STRICT_COLUMNS].astype(bool).all(axis=1)
    selected = pairs.loc[mask].copy()
    selected["strict_screen_pass"] = True
    return selected


def write_unique_sdf(
    unique: pd.DataFrame,
    output_path: Path,
    label: str,
) -> int:
    """Recover unique selected molecules from their source shard SDFs."""
    wanted = set(unique["trajectory_id"].astype(str))
    written = 0
    writer = Chem.SDWriter(str(output_path))
    writer.SetKekulize(False)
    try:
        for source_sdf in sorted(unique.sdf_path.dropna().astype(str).unique()):
            for molecule in Chem.SDMolSupplier(source_sdf, removeHs=False, sanitize=False):
                if molecule is None or not molecule.HasProp("sample_id"):
                    continue
                trajectory_id = molecule.GetProp("sample_id")
                if trajectory_id not in wanted:
                    continue
                row = unique.loc[unique["trajectory_id"].eq(trajectory_id)].iloc[0]
                molecule.SetProp("screen_label", label)
                molecule.SetProp("source_epi_ids", str(row.get("source_epi_ids", "")))
                molecule.SetProp("source_epi_count", str(row.get("source_epi_count", "")))
                molecule.SetProp("source_pair_count", str(row.get("source_pair_count", "")))
                writer.write(molecule)
                written += 1
    finally:
        writer.close()
    return written


def main() -> int:
    args = parse_args()
    if args.top_k <= 0 or args.top_k > 67_694:
        raise ValueError("--top-k must be between 1 and 67694")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    epi = pd.read_csv(args.epi, low_memory=False)
    generated = pd.read_csv(args.generated, low_memory=False)
    generated = generated[generated.rdkit_valid.eq(1)].copy().reset_index(drop=True)
    if len(epi) != 53 or len(generated) != 67_694:
        raise ValueError(f"expected epi=53 and valid generated=67694, got {len(epi)}, {len(generated)}")
    generated_fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, nBits=2048) for s in generated.canonical_smiles]
    file_tag = f"nearest{args.top_k}"
    all_selected: dict[str, set[str]] = {}
    selected_frames: list[pd.DataFrame] = []
    summary = []
    for _, reference in epi.iterrows():
        ref_mol = Chem.MolFromSmiles(str(reference.canonical_smiles))
        if ref_mol is None:
            raise ValueError(f"cannot parse {reference.rnd_id}")
        ref_fp = AllChem.GetMorganFingerprintAsBitVect(ref_mol, 2, nBits=2048)
        similarity = np.asarray(DataStructs.BulkTanimotoSimilarity(ref_fp, generated_fps), dtype=np.float32)
        order = np.lexsort((generated.trajectory_id.to_numpy(dtype=str), -similarity))[: args.top_k]
        selected = generated.iloc[order].copy()
        selected.insert(0, "epi_rank", np.arange(1, len(selected) + 1))
        selected.insert(1, "epi_id", str(reference.rnd_id))
        selected.insert(2, "epi_preferred_name", str(reference.preferred_name))
        selected["ecfp4_tanimoto"] = similarity[order]
        selected["ecfp4_distance"] = 1.0 - similarity[order]
        selected["epi_canonical_smiles"] = str(reference.canonical_smiles)
        selected_frames.append(selected)
        safe_id = str(reference.rnd_id).replace("/", "_")
        selected.to_csv(args.output_dir / f"{safe_id}_{file_tag}.csv.gz", index=False, compression="gzip")
        ids = set(selected.trajectory_id.astype(str))
        all_selected[safe_id] = ids
        summary.append({"epi_id": safe_id, "preferred_name": str(reference.preferred_name), "top_k": len(selected), "max_tanimoto": float(selected.ecfp4_tanimoto.max()), "median_tanimoto": float(selected.ecfp4_tanimoto.median()), "min_distance": float(selected.ecfp4_distance.min()), "max_distance": float(selected.ecfp4_distance.max()), "sdf": str(args.output_dir / f"{safe_id}_{file_tag}.sdf"), "ranking_csv": str(args.output_dir / f"{safe_id}_{file_tag}.csv.gz")})

    wanted: dict[str, set[str]] = {}
    for epi_id, ids in all_selected.items():
        wanted.update({f"{epi_id}": ids})
    writers = {epi_id: Chem.SDWriter(str(args.output_dir / f"{epi_id}_{file_tag}.sdf")) for epi_id in wanted}
    for writer in writers.values():
        writer.SetKekulize(False)
    written = dict.fromkeys(wanted, 0)
    try:
        for source_sdf in sorted(generated.sdf_path.dropna().astype(str).unique()):
            for molecule in Chem.SDMolSupplier(source_sdf, removeHs=False, sanitize=False):
                if molecule is None or not molecule.HasProp("sample_id"):
                    continue
                molecule_id = molecule.GetProp("sample_id")
                for epi_id, ids in wanted.items():
                    if molecule_id in ids:
                        molecule.SetProp("nearest_epi_id", epi_id)
                        writers[epi_id].write(molecule)
                        written[epi_id] += 1
    finally:
        for writer in writers.values():
            writer.close()
    for row in summary:
        row["sdf_molecule_count"] = written[row["epi_id"]]
        if row["sdf_molecule_count"] != args.top_k:
            raise RuntimeError(f"SDF count mismatch for {row['epi_id']}: {row['sdf_molecule_count']}")

    pairs = pd.concat(selected_frames, ignore_index=True)
    unique = aggregate_by_trajectory(pairs)
    pairs.to_csv(args.output_dir / f"rnd53_{file_tag}_all_pairs.csv.gz", index=False, compression="gzip")
    unique.to_csv(args.output_dir / f"rnd53_{file_tag}_unique.csv.gz", index=False, compression="gzip")
    aggregate_sdf = args.output_dir / f"rnd53_{file_tag}_unique.sdf"
    aggregate_sdf_count = write_unique_sdf(unique, aggregate_sdf, f"rnd53_{file_tag}_unique")

    strict_summary = None
    if args.strict_facts is not None:
        strict_pairs = strict_filter(pairs, args.strict_facts)
        strict_unique = aggregate_by_trajectory(strict_pairs)
        strict_pairs.to_csv(args.output_dir / f"rnd53_{file_tag}_strict_pairs.csv.gz", index=False, compression="gzip")
        strict_unique.to_csv(args.output_dir / f"rnd53_{file_tag}_strict_unique.csv.gz", index=False, compression="gzip")
        strict_sdf = args.output_dir / f"rnd53_{file_tag}_strict_unique.sdf"
        strict_sdf_count = write_unique_sdf(strict_unique, strict_sdf, f"rnd53_{file_tag}_strict_unique")
        strict_by_epi = strict_pairs.groupby(["epi_id", "epi_preferred_name"], dropna=False).agg(
            pair_pass_count=("trajectory_id", "size"), unique_pass_count=("trajectory_id", "nunique")
        ).reset_index()
        strict_by_epi.to_csv(args.output_dir / f"rnd53_{file_tag}_strict_by_epi.csv", index=False)
        strict_summary = {
            "strict_pair_pass_count": int(len(strict_pairs)),
            "strict_unique_trajectory_count": int(len(strict_unique)),
            "strict_unique_canonical_smiles": int(strict_unique["canonical_smiles"].nunique()),
            "strict_unique_sdf_count": int(strict_sdf_count),
            "strict_unique_epi_count": int(strict_pairs["epi_id"].nunique()),
            "strict_facts": str(args.strict_facts.resolve()),
        }

    output_summary = {
        "status": "complete",
        "distance": "1 - ECFP4 Morgan radius=2 Tanimoto",
        "generated_valid": len(generated),
        "epi_count": len(epi),
        "top_k": args.top_k,
        "pair_rows": int(len(pairs)),
        "unique_trajectory_count": int(len(unique)),
        "unique_canonical_smiles": int(unique["canonical_smiles"].nunique()),
        "unique_sdf_count": int(aggregate_sdf_count),
        "unique_epi_count": int(pairs["epi_id"].nunique()),
        "strict": strict_summary,
        "summaries": summary,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(output_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
