"""Frozen preference-pair construction for DiffSBDD endpoints."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class PreferencePair:
    """One winner/loser pair and its source rows."""

    winner_id: str
    loser_id: str
    winner_shard: str
    loser_shard: str
    winner_index: int
    loser_index: int
    n_nodes: int
    loser_category: str


def load_selected_manifest(path: str | Path) -> pd.DataFrame:
    """Load and validate the frozen good/hard-bad endpoint manifest."""
    frame = pd.read_csv(path, compression="gzip", low_memory=False)
    required = {
        "trajectory_id",
        "trajectory_shard",
        "mol_index",
        "heavy_atom_count",
        "split",
        "extreme_label",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"manifest missing required columns: {missing}")
    if frame["trajectory_id"].duplicated().any():
        raise ValueError("manifest contains duplicated trajectory_id")
    frame = frame[frame["extreme_label"].isin(["good", "hard_bad"])].copy()
    frame["heavy_atom_count"] = frame["heavy_atom_count"].astype(int)
    return frame.reset_index(drop=True)


def build_exact_atom_count_pairs(
    frame: pd.DataFrame,
    split: str,
    *,
    count: int | None = None,
    seed: int = 1,
) -> list[PreferencePair]:
    """Pair good and hard-bad rows within each split and atom count."""
    subset = frame[frame["split"].eq(split)]
    rng = random.Random(seed)
    pairs: list[PreferencePair] = []
    for n_nodes, group in subset.groupby("heavy_atom_count", sort=True):
        good = group[group["extreme_label"].eq("good")].to_dict("records")
        bad = group[group["extreme_label"].eq("hard_bad")].to_dict("records")
        rng.shuffle(good)
        rng.shuffle(bad)
        for winner, loser in zip(good, bad):
            pairs.append(
                PreferencePair(
                    winner_id=str(winner["trajectory_id"]),
                    loser_id=str(loser["trajectory_id"]),
                    winner_shard=str(winner["trajectory_shard"]),
                    loser_shard=str(loser["trajectory_shard"]),
                    winner_index=int(winner["mol_index"]),
                    loser_index=int(loser["mol_index"]),
                    n_nodes=int(n_nodes),
                    loser_category=str(loser.get("hard_bad_category", "unknown")),
                )
            )
    rng.shuffle(pairs)
    return pairs if count is None else pairs[: min(count, len(pairs))]


def pairs_to_frame(pairs: list[PreferencePair]) -> pd.DataFrame:
    """Serialize preference pairs to a stable tabular manifest."""
    return pd.DataFrame(
        [
            {
                "pair_index": index,
                "winner_id": pair.winner_id,
                "loser_id": pair.loser_id,
                "winner_shard": pair.winner_shard,
                "loser_shard": pair.loser_shard,
                "winner_index": pair.winner_index,
                "loser_index": pair.loser_index,
                "n_nodes": pair.n_nodes,
                "loser_category": pair.loser_category,
            }
            for index, pair in enumerate(pairs)
        ]
    )


def load_pair_manifest(path: str | Path) -> list[PreferencePair]:
    """Load a previously frozen pair manifest without rebuilding its pairs."""
    frame = pd.read_csv(path)
    required = {
        "winner_id",
        "loser_id",
        "winner_shard",
        "loser_shard",
        "winner_index",
        "loser_index",
        "n_nodes",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"pair manifest missing required columns: {missing}")
    pairs = [
        PreferencePair(
            winner_id=str(row.winner_id),
            loser_id=str(row.loser_id),
            winner_shard=str(row.winner_shard),
            loser_shard=str(row.loser_shard),
            winner_index=int(row.winner_index),
            loser_index=int(row.loser_index),
            n_nodes=int(row.n_nodes),
            loser_category=str(getattr(row, "loser_category", "unknown")),
        )
        for row in frame.itertuples(index=False)
    ]
    validate_pair_manifests(pairs, [], source=path)
    return pairs


def validate_pair_manifests(
    train_pairs: list[PreferencePair],
    validation_pairs: list[PreferencePair],
    *,
    source: str | Path = "pair manifests",
) -> None:
    """Validate pair identity, split leakage, and basic endpoint metadata."""
    all_pairs = train_pairs + validation_pairs
    pair_keys = [(pair.winner_id, pair.loser_id) for pair in all_pairs]
    if len(pair_keys) != len(set(pair_keys)):
        raise ValueError(f"duplicate winner/loser pair found in {source}")
    endpoint_ids = [item for pair in all_pairs for item in (pair.winner_id, pair.loser_id)]
    if any(pair.winner_id == pair.loser_id for pair in all_pairs):
        raise ValueError(f"a pair contains identical winner and loser in {source}")
    train_ids = {item for pair in train_pairs for item in (pair.winner_id, pair.loser_id)}
    validation_ids = {item for pair in validation_pairs for item in (pair.winner_id, pair.loser_id)}
    overlap = train_ids.intersection(validation_ids)
    if overlap:
        raise ValueError(f"train/validation endpoint leakage in {source}: {sorted(overlap)[:3]}")
    if any(pair.n_nodes <= 0 for pair in all_pairs):
        raise ValueError(f"non-positive n_nodes found in {source}")
    if len(endpoint_ids) != len(set(endpoint_ids)):
        raise ValueError(f"an endpoint is reused across frozen pairs in {source}")
