#!/usr/bin/env python
"""Build cross-source preference pairs: ChEMBL winners vs DiffSBDD endpoints.

Winners come from ``winner_index.csv`` (ChEMBL dock poses converted into
endpoint-store shards, R1-R5 clean by construction). Losers are DiffSBDD
endpoints read directly from the trajectory store.

Reward (joint structure + docking, ESRPO-compatible):

    reward = -(structural violations) - dock_weight * docking_raw

DiffSBDD endpoints that failed docking get ``docking_raw = 0`` (vina scores
are negative-is-better, so 0 is the worst possible score). Endpoints whose
reconstruction failed (``rdkit_valid = 0``) cannot be scored by the structure
rules and are excluded, mirroring the original pair builder.

Pairs are matched on exact heavy-atom count (n_nodes) within
[MIN_NODES, MAX_NODES], require reward_gap >= --min-gap, and are split
train/validation/test without endpoint leakage (winner ids are ChEMBL ids,
loser ids are DiffSBDD sample ids, so leakage is impossible by construction
but we still run the standard check).
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "offline_rl"))

from structure_pathology_rules import (  # noqa: E402
    compute_structure_pathology_from_smiles,
)

LOGGER = logging.getLogger("build_crosssource_pairs")

MIN_NODES, MAX_NODES = 10, 30
COLUMNS = [
    "pair_index", "split", "winner_id", "loser_id",
    "winner_docking_raw", "loser_docking_raw",
    "reward_winner", "reward_loser", "reward_gap",
    "winner_shard", "loser_shard", "winner_index", "loser_index",
    "winner_n_nodes", "loser_n_nodes", "n_nodes", "ring_count", "loser_category",
]


def _scan_store(store_glob: str) -> pd.DataFrame:
    """Index every endpoint in the DiffSBDD trajectory store."""
    rows = []
    shard_paths = sorted(glob.glob(store_glob))
    LOGGER.info("scanning store: %d shards", len(shard_paths))
    for path in shard_paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        sample_ids = payload["sample_ids"]
        nodes = payload["nodesxsample"]
        recon = {r["sample_id"]: r for r in payload.get("reconstruction_rows", [])}
        for i, sid in enumerate(sample_ids):
            r = recon.get(str(sid), {})
            rows.append(
                {
                    "sample_id": str(sid),
                    "shard": path,
                    "index": i,
                    "n_nodes": int(nodes[i]),
                    "smiles": r.get("smiles", ""),
                    "rdkit_valid": int(r.get("rdkit_valid", 0)),
                }
            )
    return pd.DataFrame(rows)


def _violations_worker(item: tuple[str, str]) -> dict[str, object] | None:
    sid, smiles = item
    facts = compute_structure_pathology_from_smiles(smiles)
    if facts is None:
        return None
    return {"sample_id": sid, "structural_violation_count": facts.structural_violation_count}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--winner-index",
        type=Path,
        default=Path("data/crosssource/chembl_endpoint_shards_260929/winner_index.csv"),
    )
    parser.add_argument(
        "--store-glob",
        type=str,
        default=(
            "/data/disk6/DiffB110/runs/diffsbdd_collect/"
            "diffsbdd-original-t1000-8gpu-100k-trajectory_20260730_194643/"
            "shards/rank_*/shard_*.pt"
        ),
    )
    parser.add_argument(
        "--reward-sidecar",
        type=Path,
        default=Path(
            "analysis/offline_iql_labels_alidiff_unidock_affinity_8gpu_full_260824/"
            "reward_sidecar.csv"
        ),
    )
    parser.add_argument("--dock-weight", type=float, default=0.5)
    parser.add_argument("--min-gap", type=float, default=2.0)
    parser.add_argument(
        "--winner-dock-max",
        type=float,
        default=-9.0,
        help="winners must dock at least this well (score <= threshold)",
    )
    parser.add_argument(
        "--loser-dock-min",
        type=float,
        default=-6.5,
        help="losers must dock this poorly or worse (score >= threshold); "
        "endpoints whose docking failed get score 0 and qualify",
    )
    parser.add_argument(
        "--loser-min-viol",
        type=int,
        default=1,
        help="losers must violate at least this many R1-R5 rules",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=50_000,
        help="cap on total pairs (a few tens of thousands suffice)",
    )
    parser.add_argument("--workers", type=int, default=18)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("analysis/crosssource_pairs_260929/pairs")
    )
    parser.add_argument(
        "--loser-manifest-glob",
        type=str,
        default=None,
        help="fast path: derive losers from existing ESRPO pair manifests "
        "(reward_loser == -violations because dock_weight was 0)",
    )
    parser.add_argument(
        "--match-key",
        default="size_ring",
        choices=["size_ring", "size", "none"],
        help="pairing key: 'size_ring' matches (HA, ring count); "
        "'size' matches heavy-atom count only (size is a confound that must "
        "never be a free signal); 'none' pairs freely",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    # --- winners (R1-R5 clean by construction, well-docked) ---------------
    winners = pd.read_csv(args.winner_index)
    winners = winners[
        winners["n_nodes"].between(MIN_NODES, MAX_NODES)
        & (winners["docking_raw"] <= args.winner_dock_max)
    ].copy()
    winners["reward"] = -args.dock_weight * winners["docking_raw"]  # violations == 0
    LOGGER.info("winners usable: %d", len(winners))

    # --- losers ----------------------------------------------------------
    if args.loser_manifest_glob:
        # fast path: reuse the original ESRPO pair manifests. reward_loser was
        # computed with dock_weight=0, i.e. reward_loser == -structural
        # violations, so violations come for free without scanning the store.
        frames = [pd.read_csv(p) for p in sorted(glob.glob(args.loser_manifest_glob))]
        allp = pd.concat(frames, ignore_index=True)
        uniq = allp.drop_duplicates("loser_id")
        losers = pd.DataFrame(
            {
                "sample_id": uniq["loser_id"],
                "shard": uniq["loser_shard"],
                "index": uniq["loser_index"],
                "n_nodes": uniq["loser_n_nodes"],
                "structural_violation_count": (-uniq["reward_loser"]).round().astype(int),
            }
        )
        LOGGER.info("losers from manifest glob: %d unique", len(losers))
    else:
        losers = _scan_store(args.store_glob)
        LOGGER.info("store endpoints: %d", len(losers))
        losers = losers[
            losers["n_nodes"].between(MIN_NODES, MAX_NODES) & losers["rdkit_valid"].eq(1)
        ]
        LOGGER.info("losers reconstructable in node range: %d", len(losers))

        with Pool(args.workers) as pool:
            viol_rows = [
                r
                for r in pool.imap(
                    _violations_worker,
                    list(zip(losers["sample_id"], losers["smiles"])),
                    chunksize=1024,
                )
                if r is not None
            ]
        viol = pd.DataFrame(viol_rows)
        losers = losers.merge(viol, on="sample_id", how="inner")
        LOGGER.info("losers with rule scores: %d", len(losers))

    sidecar = pd.read_csv(
        args.reward_sidecar, usecols=["sample_id", "docking_raw", "docking_success", "ring_count"]
    )
    losers = losers.merge(sidecar, on="sample_id", how="left")
    losers["docking_raw"] = losers["docking_raw"].fillna(0.0)  # failed dock -> worst score
    # ring counts are required for the ring-matched pairing; endpoints missing
    # from the sidecar cannot be ring-bucketed and are dropped
    losers = losers[losers["ring_count"].notna()]
    losers["ring_count"] = losers["ring_count"].astype(int)
    losers["reward"] = -losers["structural_violation_count"] - args.dock_weight * losers["docking_raw"]

    # extreme-loser screen: structural violations + poor docking
    n_before = len(losers)
    losers = losers[
        (losers["structural_violation_count"] >= args.loser_min_viol)
        & (losers["docking_raw"] >= args.loser_dock_min)
    ]
    LOGGER.info(
        "extreme losers (viol>=%d, dock>=%.1f): %d (from %d reconstructable)",
        args.loser_min_viol, args.loser_dock_min, len(losers), n_before,
    )
    LOGGER.info(
        "loser rewards: mean=%.2f  winner rewards: mean=%.2f",
        losers["reward"].mean(), winners["reward"].mean(),
    )

    # --- pairing on (n_nodes, ring_count) --------------------------------
    # Ring-matched pairing closes the "acyclic molecules clear R1-R5 for free"
    # reward-hacking shortcut: within a (size, ring) bucket the winner/loser
    # differ in pathology and docking quality, not in ring count, so the model
    # cannot lower its structural reward by abandoning rings altogether.
    winners["ring_count"] = winners["ring_count"].astype(int)
    if args.match_key == "none":
        # one global pool: no size/ring matching, so the contrast carries every
        # difference between the two sources (pure distribution-level signal)
        winner_buckets = {"all": winners.reset_index(drop=True)}
        loser_keys = {"all"}
    elif args.match_key == "size":
        # heavy-atom count is a confound that must never be a free signal
        # (random pairing left a +11.4 heavy-atom gap); everything else is
        # left free so ChEMBL's chemical complexity can transfer
        winner_buckets = {n: g.reset_index(drop=True) for n, g in winners.groupby("n_nodes")}
        loser_keys = set(losers["n_nodes"])
    else:
        winner_buckets = {
            k: g.reset_index(drop=True) for k, g in winners.groupby(["n_nodes", "ring_count"])
        }
        loser_keys = set(zip(losers["n_nodes"], losers["ring_count"]))
    eligible = (
        sorted(winner_buckets)
        if args.match_key == "none"
        else sorted(set(winner_buckets) & loser_keys)
    )
    per_node_cap = max(1, args.max_pairs // max(len(eligible), 1))
    LOGGER.info(
        "match_key=%s eligible buckets: %d -> per-bucket cap %d",
        args.match_key, len(eligible), per_node_cap,
    )
    pairs: list[dict[str, object]] = []
    used_winners: set[str] = set()  # allow_endpoint_reuse=false: one pair per winner
    n_gap_reject = 0
    stop = False
    for key in eligible:
        if stop:
            break
        if args.match_key == "none":
            n_nodes, ring_count = 0, -1
            bucket = losers
        elif args.match_key == "size":
            n_nodes, ring_count = key, -1
            bucket = losers[losers["n_nodes"] == n_nodes]
        else:
            n_nodes, ring_count = key
            bucket = losers[(losers["n_nodes"] == n_nodes) & (losers["ring_count"] == ring_count)]
        cand = winner_buckets[key]
        cand_rewards = cand["reward"].to_numpy()
        made = 0
        for loser in bucket.itertuples():
            if made >= per_node_cap or len(pairs) >= args.max_pairs:
                stop = len(pairs) >= args.max_pairs
                break
            # draw winners until unique + reward gap clears the threshold
            # (bounded attempts to keep per-loser cost constant)
            for _ in range(8):
                j = rng.randrange(len(cand))
                w_id = str(cand.iloc[j]["sample_id"])
                if w_id in used_winners:
                    n_gap_reject += 1
                    continue
                gap = float(cand_rewards[j]) - float(loser.reward)
                if gap >= args.min_gap:
                    w = cand.iloc[j]
                    pairs.append(
                        {
                            "winner_id": w_id,
                            "loser_id": loser.sample_id,
                            "winner_docking_raw": float(w["docking_raw"]),
                            "loser_docking_raw": float(loser.docking_raw),
                            "reward_winner": float(cand_rewards[j]),
                            "reward_loser": float(loser.reward),
                            "reward_gap": gap,
                            "winner_shard": w["shard"],
                            "loser_shard": loser.shard,
                            "winner_index": int(w["index"]),
                            "loser_index": int(loser.index),
                            "winner_n_nodes": int(w["n_nodes"]),
                            "loser_n_nodes": int(loser.n_nodes),
                            "n_nodes": int(n_nodes),
                            "ring_count": int(loser.ring_count),
                            "loser_category": {
                                "size_ring": "crosssource_ring_matched",
                                "size": "crosssource_size_matched",
                                "none": "crosssource_random",
                            }[args.match_key],
                        }
                    )
                    used_winners.add(w_id)
                    made += 1
                    break
                n_gap_reject += 1

    LOGGER.info("total pairs: %d (gap rejects: %d)", len(pairs), n_gap_reject)
    rng.shuffle(pairs)
    n = len(pairs)
    n_val, n_test = max(1, int(n * 0.01)), max(1, int(n * 0.01))
    for i, p in enumerate(pairs):
        p["pair_index"] = i
        p["split"] = "test" if i < n_test else "validation" if i < n_test + n_val else "train"

    frame = pd.DataFrame(pairs, columns=COLUMNS)
    for split in ("train", "validation", "test"):
        out = args.output_dir / f"{split}_pairs.csv"
        frame[frame["split"] == split].to_csv(out, index=False)
        LOGGER.info("wrote %s: %d", out, (frame["split"] == split).sum())

    audit = {
        "dock_weight": args.dock_weight,
        "min_gap": args.min_gap,
        "seed": args.seed,
        "node_range": [MIN_NODES, MAX_NODES],
        "winner_dock_max": args.winner_dock_max,
        "loser_dock_min": args.loser_dock_min,
        "loser_min_viol": args.loser_min_viol,
        "max_pairs": args.max_pairs,
        "n_winners": int(len(winners)),
        "n_losers_scored": int(len(losers)),
        "n_pairs": int(n),
        "n_gap_rejects": int(n_gap_reject),
        "reward_gap_stats": {
            "mean": float(frame["reward_gap"].mean()),
            "p5": float(frame["reward_gap"].quantile(0.05)),
            "p95": float(frame["reward_gap"].quantile(0.95)),
        },
        "loser_structural_violation_count": {
            "mean": float(losers["structural_violation_count"].mean()),
            "p95": float(losers["structural_violation_count"].quantile(0.95)),
        },
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (args.output_dir / "pair_audit.json").write_text(json.dumps(audit, indent=2))
    LOGGER.info("audit: %s", json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
