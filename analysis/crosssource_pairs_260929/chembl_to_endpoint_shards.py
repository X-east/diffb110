#!/usr/bin/env python
"""Convert ChEMBL dock poses into DiffSBDD endpoint-store shards.

For every winner (docking success + R1-R5 clean + HA 10-30) we read its
unidock pose SDF (pocket-frame coordinates, same frame as DiffSBDD endpoints),
keep heavy atoms only, map elements onto the DiffSBDD 10-type vocabulary, and
serialise payloads compatible with ``ddpo.global_dpo.endpoint_data._load_endpoint``:

  {"sample_ids": [str], "x": [B, N_MAX, 3], "categorical": [B, N_MAX, 10],
   "nodesxsample": [B]}

plus an index CSV (sample_id, shard, index, n_nodes, docking_raw,
structural_violation_count) used by the cross-source pair builder.
"""

from __future__ import annotations

import argparse
import glob
import logging
import time
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

LOGGER = logging.getLogger("chembl_to_endpoint_shards")

ATOM_TYPES = ["C", "N", "O", "S", "B", "Br", "Cl", "P", "I", "F"]
TYPE_INDEX = {sym: i for i, sym in enumerate(ATOM_TYPES)}
N_MAX = 30
import torch  # noqa: E402


def _pose_to_arrays(mol: Chem.Mol) -> tuple[int, list[float], list[int]] | None:
    """Return (n_nodes, flattened heavy-atom coords, type ids) or None."""
    conf = mol.GetConformer() if mol.GetNumConformers() else None
    if conf is None:
        return None
    coords: list[float] = []
    types: list[int] = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if atom.GetAtomicNum() <= 1:  # skip hydrogens from the dock output
            continue
        type_id = TYPE_INDEX.get(symbol)
        if type_id is None:
            return None  # element outside the DiffSBDD vocabulary
        pos = conf.GetAtomPosition(atom.GetIdx())
        coords.extend([float(pos.x), float(pos.y), float(pos.z)])
        types.append(type_id)
    n_nodes = len(types)
    if not 1 <= n_nodes <= N_MAX:
        return None
    return n_nodes, coords, types


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rules-csv",
        type=Path,
        default=Path("analysis/crosssource_pairs_260929/chembl_docked_per_mol_rules.csv"),
    )
    parser.add_argument(
        "--docking-dir",
        type=Path,
        default=Path("analysis/chembl_docking_reference_260928/full/docking"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/crosssource/chembl_endpoint_shards_260929"),
    )
    parser.add_argument("--batch-size", type=int, default=1000, help="samples per shard")
    parser.add_argument("--max-winners", type=int, default=0, help=">0: cap winners (smoke test)")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rules = pd.read_csv(args.rules_csv)
    winners = rules[
        (rules["structural_violation_count"] == 0) & rules["heavy_atoms"].between(10, N_MAX)
    ].copy()
    LOGGER.info("winner pool before pose conversion: %d", len(winners))
    if args.max_winners:
        winners = winners.sample(n=args.max_winners, random_state=args.seed)
    chembl_ids = winners.set_index("chembl_id")

    index_rows: list[dict[str, object]] = []
    batch_ids: list[str] = []
    batch_x: list[list[float]] = []
    batch_v: list[list[int]] = []
    batch_n: list[int] = []
    shard_idx = 0
    n_pose_fail = 0
    t0 = time.time()

    def _flush() -> None:
        nonlocal shard_idx, batch_ids, batch_x, batch_v, batch_n
        if not batch_ids:
            return
        b = len(batch_ids)
        x = torch.zeros(b, N_MAX, 3, dtype=torch.float32)
        v = torch.zeros(b, N_MAX, len(ATOM_TYPES), dtype=torch.float32)
        for i, (coords, types) in enumerate(zip(batch_x, batch_v)):
            n = len(types)
            x[i, :n] = torch.tensor(coords, dtype=torch.float32).view(n, 3)
            v[i, torch.arange(n), torch.tensor(types)] = 1.0
        payload = {
            "format": "chembl_endpoint_v1",
            "sample_ids": batch_ids,
            "x": x,
            "categorical": v,
            "nodesxsample": torch.tensor(batch_n, dtype=torch.int64),
        }
        shard_path = args.output_dir / f"shard_{shard_idx:04d}.pt"
        torch.save(payload, shard_path)
        index_rows.extend(
            {
                "sample_id": sid,
                "shard": str(shard_path),
                "index": i,
                "n_nodes": n,
                "docking_raw": chembl_ids.loc[sid, "docking_raw"],
                "structural_violation_count": 0,
            }
            for i, (sid, n) in enumerate(zip(batch_ids, batch_n))
        )
        shard_idx += 1
        batch_ids, batch_x, batch_v, batch_n = [], [], [], []

    # chembl_id -> shard dir, recovered from the docking results themselves
    # (avoids glob-scanning the 1.1M-file outputs tree per molecule). Shards
    # that went through the segfault fallback keep their scores but have no
    # pose files, so their molecules are simply skipped (pose availability is
    # driven by where a poison molecule sits, i.e. ~random w.r.t. chemistry).
    id_to_shard: dict[str, str] = {}
    for results_path in sorted(
        glob.glob(str(args.docking_dir / "shard_*" / "docking_results.csv"))
    ):
        shard_dir = str(Path(results_path).parent)
        for cid in pd.read_csv(results_path, usecols=["chembl_id"])["chembl_id"]:
            id_to_shard[str(cid)] = shard_dir
    LOGGER.info(
        "pose index: %d chembl_ids -> %d shards",
        len(id_to_shard), len(set(id_to_shard.values())),
    )

    done = 0
    for chembl_id in winners["chembl_id"]:
        pose_path = (
            f"{id_to_shard[str(chembl_id)]}/outputs/{chembl_id}_out.sdf"
            if str(chembl_id) in id_to_shard
            else None
        )
        mol = (
            Chem.MolFromMolFile(pose_path, sanitize=False, removeHs=True, strictParsing=False)
            if pose_path and Path(pose_path).exists()
            else None
        )
        arrays = _pose_to_arrays(mol) if mol is not None and mol.GetNumConformers() else None
        if arrays is None:
            n_pose_fail += 1
            continue
        n_nodes, coords, types = arrays
        batch_ids.append(str(chembl_id))
        batch_x.append(coords)
        batch_v.append(types)
        batch_n.append(n_nodes)
        done += 1
        if len(batch_ids) >= args.batch_size:
            _flush()
        if done % 50_000 == 0:
            LOGGER.info(
                "converted=%d shards=%d pose_fail=%d rate=%.0f/s",
                done, shard_idx, n_pose_fail, done / max(time.time() - t0, 1.0),
            )
    _flush()

    index_path = args.output_dir / "winner_index.csv"
    pd.DataFrame(index_rows).to_csv(index_path, index=False)
    LOGGER.info(
        "completed winners=%d shards=%d pose_fail=%d elapsed=%.0fs index=%s",
        done, shard_idx, n_pose_fail, time.time() - t0, index_path,
    )


if __name__ == "__main__":
    main()
