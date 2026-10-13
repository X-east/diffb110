#!/usr/bin/env python
"""Per-molecule R1-R5 structural violation counts for docked ChEMBL winners.

Reads the small-molecule reference set (chembl_id -> SMILES) and the docking
results of the ChEMBL docking-reference run, keeps molecules that docked
successfully, and computes structure-pathology facts for each one.

Output: chembl_docked_per_mol_rules.csv with columns
  chembl_id, heavy_atoms, docking_raw, structural_violation_count, r1..r5 flags
"""

from __future__ import annotations

import argparse
import glob
import logging
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "offline_rl"))

from structure_pathology_rules import (  # noqa: E402
    compute_structure_pathology_from_smiles,
)

RDLogger.DisableLog("rdApp.*")

LOGGER = logging.getLogger("chembl_docked_rules")

FLAG_COLUMNS = [
    "r1_large_ring_violation",
    "r2_ring_system_violation",
    "r3_four_ring_violation",
    "r4a_three_ring_hetero_violation",
    "r4b_three_ring_over_attached_violation",
    "r4c_three_ring_spiro_or_fused_violation",
    "r5_carbon_fraction_violation",
]


def _load_docked_scores(docking_dir: Path) -> pd.DataFrame:
    """Concatenate every shard's docking_results.csv, keep successes."""
    frames = [
        pd.read_csv(path, usecols=["chembl_id", "success", "score"])
        for path in sorted(glob.glob(str(docking_dir / "shard_*" / "docking_results.csv")))
    ]
    if not frames:
        raise FileNotFoundError(f"no docking_results.csv under {docking_dir}")
    df = pd.concat(frames, ignore_index=True)
    df = df[df["success"] & df["score"].notna()]
    df = df.rename(columns={"score": "docking_raw"})
    return df[["chembl_id", "docking_raw"]]


def _worker(item: tuple[str, str, float, int]) -> dict[str, object] | None:
    chembl_id, smiles, docking_raw, heavy_atoms = item
    facts = compute_structure_pathology_from_smiles(smiles)
    if facts is None:
        return None
    row: dict[str, object] = {
        "chembl_id": chembl_id,
        "heavy_atoms": int(heavy_atoms),
        "ring_count": facts.ring_count,
        "docking_raw": float(docking_raw),
        "structural_violation_count": facts.structural_violation_count,
    }
    for col in FLAG_COLUMNS:
        row[col] = int(getattr(facts, col))
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=Path("data/chembl_smallmol_reference/reference_smallmol.csv.gz"),
    )
    parser.add_argument(
        "--docking-dir",
        type=Path,
        default=Path("analysis/chembl_docking_reference_260928/full/docking"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/crosssource_pairs_260929"))
    parser.add_argument("--chunk-size", type=int, default=50_000)
    parser.add_argument("--workers", type=int, default=18)
    parser.add_argument("--limit-chunks", type=int, default=0, help=">0: stop after N chunks (smoke test)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("loading docked chembl_ids ...")
    scores = _load_docked_scores(args.docking_dir)
    score_map = dict(zip(scores["chembl_id"], scores["docking_raw"]))
    LOGGER.info("docked successes: %d unique chembl_ids", len(score_map))

    out_path = args.output_dir / "chembl_docked_per_mol_rules.csv"
    n_seen = n_kept = n_written = 0
    t0 = time.time()
    reader = pd.read_csv(
        args.reference_csv,
        usecols=["chembl_id", "canonical_smiles", "heavy_atoms"],
        chunksize=args.chunk_size,
    )
    pool = Pool(args.workers)
    header_written = False
    with out_path.open("w") as handle:
        for chunk_index, chunk in enumerate(reader):
            if args.limit_chunks and chunk_index >= args.limit_chunks:
                break
            n_seen += len(chunk)
            keep = chunk[chunk["chembl_id"].isin(score_map)]
            n_kept += len(keep)
            items = [
                (cid, smiles, float(score_map[cid]), int(ha))
                for cid, smiles, ha in zip(
                    keep["chembl_id"], keep["canonical_smiles"], keep["heavy_atoms"]
                )
            ]
            for row in pool.imap(_worker, items, chunksize=512):
                if row is None:
                    continue
                if not header_written:
                    handle.write(",".join(row.keys()) + "\n")
                    header_written = True
                handle.write(
                    ",".join(str(row[k]) for k in row)
                    + "\n"
                )
                n_written += 1
            rate = n_written / max(time.time() - t0, 1.0)
            LOGGER.info(
                "seen=%d kept=%d written=%d rate=%.0f/s",
                n_seen, n_kept, n_written, rate,
            )
    pool.close()
    pool.join()
    LOGGER.info("completed written=%d elapsed=%.0fs -> %s", n_written, time.time() - t0, out_path)


if __name__ == "__main__":
    main()
