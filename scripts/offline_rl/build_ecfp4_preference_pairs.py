#!/usr/bin/env python3
"""Build split-isolated ECFP4 preference pairs from a frozen sidecar.

The preferred endpoint is selected from the upper reward quantile and the
rejected endpoint from the lower reward quantile within the same frozen split.
Technical failures are excluded from both pools and therefore are not treated
as chemical negatives.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


PAIR_FIELDS = [
    "pair_index", "split", "winner_id", "loser_id", "reward_winner", "reward_loser",
    "reward_gap", "winner_shard", "loser_shard", "winner_index", "loser_index",
    "winner_n_nodes", "loser_n_nodes", "n_nodes", "loser_category", "source_rule",
]
SAMPLE_ID_RE = re.compile(r"^r\d+_s\d+_m(?P<mol_index>\d+)$")


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def molecule_index(sample_id: str) -> int:
    """Return the endpoint's molecule index within its trajectory shard."""
    match = SAMPLE_ID_RE.fullmatch(str(sample_id))
    if match is None:
        raise ValueError(f"sample_id is not in rXX_sXXXX_mYYYY format: {sample_id}")
    return int(match.group("mol_index"))


def load_jsonl(path: Path) -> pd.DataFrame:
    """Load JSONL endpoint metadata into a dataframe."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    frame = pd.DataFrame(rows)
    required = {"sample_id", "num_nodes", "shard_path", "shard_index", "failure_type"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"episode index missing columns: {missing}")
    if frame["sample_id"].duplicated().any():
        raise ValueError("episode index contains duplicate sample_id")
    return frame


def build_pairs(
    sidecar: pd.DataFrame,
    episode_index: pd.DataFrame,
    source_run_dir: Path,
    *,
    low_quantile: float,
    high_quantile: float,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Build deterministic upper-vs-lower reward pairs separately per split."""
    if not 0.0 <= low_quantile < high_quantile <= 1.0:
        raise ValueError("quantiles must satisfy 0 <= low < high <= 1")
    required = {"sample_id", "split", "ecfp4_success", "ecfp4_reward", "rdkit_valid"}
    missing = sorted(required.difference(sidecar.columns))
    if missing:
        raise ValueError(f"sidecar missing columns: {missing}")
    index_metadata = episode_index[["sample_id", "num_nodes", "shard_path", "shard_index"]].rename(
        columns={
            "num_nodes": "num_nodes_index",
            "shard_path": "shard_path_index",
            "shard_index": "shard_index_index",
        }
    )
    merged = sidecar.merge(
        index_metadata,
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    if merged[["num_nodes_index", "shard_path_index", "shard_index_index"]].isna().any().any():
        raise ValueError("sidecar contains endpoints absent from episode index")
    for field in ("num_nodes", "shard_path", "shard_index"):
        index_field = f"{field}_index"
        if field in merged and not merged[field].astype(str).equals(merged[index_field].astype(str)):
            raise ValueError(f"sidecar and episode index disagree for {field}")
        merged[field] = merged[index_field]
    valid = merged.loc[
        merged["ecfp4_success"].eq(1)
        & merged["rdkit_valid"].eq(1)
        & merged["ecfp4_reward"].map(lambda value: isinstance(value, (float, int)) and math.isfinite(float(value)))
    ].copy()
    if valid.empty:
        raise ValueError("no valid ECFP4 endpoints available")
    outputs: dict[str, list[dict[str, Any]]] = {}
    summary: dict[str, Any] = {
        "status": "completed",
        "source_rule": "ecfp4_upper_lower_quantiles_within_frozen_split_v1",
        "low_quantile": low_quantile,
        "high_quantile": high_quantile,
        "sidecar_rows": int(len(sidecar)),
        "valid_ecfp4_endpoints": int(len(valid)),
        "splits": {},
    }
    for split_name in ("train", "validation", "test"):
        group = valid.loc[valid["split"].eq(split_name)].copy()
        if group.empty:
            raise ValueError(f"split={split_name} has no valid ECFP4 endpoints")
        low_cut = float(group["ecfp4_reward"].quantile(low_quantile, interpolation="lower"))
        high_cut = float(group["ecfp4_reward"].quantile(high_quantile, interpolation="higher"))
        losers = group.loc[group["ecfp4_reward"].le(low_cut)].sort_values(
            ["ecfp4_reward", "sample_id"], kind="mergesort"
        )
        winners = group.loc[group["ecfp4_reward"].ge(high_cut)].sort_values(
            ["ecfp4_reward", "sample_id"], ascending=[False, True], kind="mergesort"
        )
        pair_count = min(len(winners), len(losers))
        if pair_count <= 0:
            raise ValueError(f"split={split_name} produced no pairs")
        pairs: list[dict[str, Any]] = []
        for pair_index in range(pair_count):
            winner = winners.iloc[pair_index]
            loser = losers.iloc[pair_index % len(losers)]
            winner_reward = float(winner["ecfp4_reward"])
            loser_reward = float(loser["ecfp4_reward"])
            if not winner_reward > loser_reward:
                raise ValueError(f"non-preferred pair at split={split_name}, index={pair_index}")
            pairs.append(
                {
                    "pair_index": pair_index,
                    "split": split_name,
                    "winner_id": str(winner["sample_id"]),
                    "loser_id": str(loser["sample_id"]),
                    "reward_winner": winner_reward,
                    "reward_loser": loser_reward,
                    "reward_gap": winner_reward - loser_reward,
                    "winner_shard": str((source_run_dir / str(winner["shard_path"])).resolve()),
                    "loser_shard": str((source_run_dir / str(loser["shard_path"])).resolve()),
                    "winner_index": molecule_index(str(winner["sample_id"])),
                    "loser_index": molecule_index(str(loser["sample_id"])),
                    "winner_n_nodes": int(winner["num_nodes"]),
                    "loser_n_nodes": int(loser["num_nodes"]),
                    "n_nodes": int(winner["num_nodes"]),
                    "loser_category": "ecfp4_lower_quantile",
                    "source_rule": "ecfp4_upper_lower_quantiles_within_frozen_split_v1",
                }
            )
        outputs[split_name] = pairs
        gaps = [float(pair["reward_gap"]) for pair in pairs]
        summary["splits"][split_name] = {
            "valid_endpoints": int(len(group)),
            "low_cut": low_cut,
            "high_cut": high_cut,
            "winner_pool": int(len(winners)),
            "loser_pool": int(len(losers)),
            "pairs": int(len(pairs)),
            "unique_winners": int(len({pair["winner_id"] for pair in pairs})),
            "unique_losers": int(len({pair["loser_id"] for pair in pairs})),
            "reward_gap_min": min(gaps),
            "reward_gap_max": max(gaps),
            "reward_gap_mean": sum(gaps) / len(gaps),
        }
    return outputs, summary


def main() -> None:
    """Build train, validation, and test ECFP4 pair manifests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-index", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--low-quantile", type=float, default=0.20)
    parser.add_argument("--high-quantile", type=float, default=0.80)
    args = parser.parse_args()
    for path in (args.episode_index, args.sidecar, args.source_run_dir):
        if not path.exists():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    episode_index = load_jsonl(args.episode_index)
    sidecar = pd.read_csv(args.sidecar, low_memory=False)
    pairs, summary = build_pairs(
        sidecar,
        episode_index,
        args.source_run_dir,
        low_quantile=args.low_quantile,
        high_quantile=args.high_quantile,
    )
    summary["episode_index"] = str(args.episode_index.resolve())
    summary["episode_index_sha256"] = sha256_file(args.episode_index)
    summary["sidecar"] = str(args.sidecar.resolve())
    summary["sidecar_sha256"] = sha256_file(args.sidecar)
    for split_name, rows in pairs.items():
        with (args.output_dir / f"{split_name}_pairs.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=PAIR_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        summary["splits"][split_name]["manifest_sha256"] = sha256_file(args.output_dir / f"{split_name}_pairs.csv")
    (args.output_dir / "pair_manifest_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
