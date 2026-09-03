"""Frozen endpoint data access for DiffSBDD Global DPO."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd
import torch


@dataclass(frozen=True)
class GlobalPreferencePair:
    """One good/bad endpoint pair used by Global DPO."""

    winner_id: str
    loser_id: str
    winner_shard: str
    loser_shard: str
    winner_index: int
    loser_index: int
    n_nodes: int
    loser_category: str = "unknown"
    reward_winner: float | None = None
    reward_loser: float | None = None
    winner_n_nodes: int | None = None
    loser_n_nodes: int | None = None


@dataclass(frozen=True)
class EndpointSample:
    """One unpadded DiffSBDD endpoint with coordinates and atom features."""

    sample_id: str
    n_nodes: int
    x: torch.Tensor
    v: torch.Tensor


def _optional_finite_float(value: object, *, field: str, source: str | Path) -> float | None:
    """Convert an optional manifest value to a finite float."""
    if value is None or pd.isna(value):
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source}: {field} must be numeric, got {value!r}") from exc
    if not math.isfinite(converted):
        raise ValueError(f"{source}: {field} must be finite, got {converted}")
    return converted


def _load_reward_sidecar(
    path: str | Path,
    *,
    reward_key: str,
) -> dict[str, float]:
    """Load finite per-sample rewards keyed by the frozen endpoint ID."""
    frame = pd.read_csv(path, low_memory=False)
    required = {"sample_id", reward_key}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"reward sidecar missing required columns: {missing}")
    if frame["sample_id"].duplicated().any():
        raise ValueError(f"reward sidecar contains duplicated sample_id: {path}")
    rewards: dict[str, float] = {}
    for row in frame[["sample_id", reward_key]].to_dict("records"):
        sample_id = str(row["sample_id"])
        reward = _optional_finite_float(row[reward_key], field=reward_key, source=path)
        if reward is not None:
            rewards[sample_id] = reward
    return rewards


def load_pair_manifest(
    path: str | Path,
    *,
    reward_sidecar: str | Path | None = None,
    reward_key: str = "reward_total_raw",
    reward_sign: float = 1.0,
    allow_endpoint_reuse: bool = False,
) -> list[GlobalPreferencePair]:
    """Load a frozen pair manifest and optionally attach endpoint rewards.

    Older binary-DPO manifests may omit reward columns.  In that case rewards
    can be joined from a sidecar using ``sample_id``.  ``reward_sign`` is
    useful for scores where lower is better, such as docking energies.
    """
    if not math.isfinite(float(reward_sign)) or float(reward_sign) == 0.0:
        raise ValueError(f"reward_sign must be finite and non-zero, got {reward_sign}")
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
    sidecar_rewards = (
        _load_reward_sidecar(reward_sidecar, reward_key=reward_key)
        if reward_sidecar is not None
        else {}
    )

    def resolve_reward(row: dict[str, object], field: str, sample_id_field: str) -> float | None:
        """Resolve one reward from manifest columns, then the optional sidecar."""
        manifest_value = _optional_finite_float(row.get(field), field=field, source=path)
        if manifest_value is not None:
            return manifest_value * float(reward_sign)
        sample_id = str(row[sample_id_field])
        sidecar_value = sidecar_rewards.get(sample_id)
        return None if sidecar_value is None else sidecar_value * float(reward_sign)

    pairs = [
        GlobalPreferencePair(
            winner_id=str(row["winner_id"]),
            loser_id=str(row["loser_id"]),
            winner_shard=str(row["winner_shard"]),
            loser_shard=str(row["loser_shard"]),
            winner_index=int(row["winner_index"]),
            loser_index=int(row["loser_index"]),
            n_nodes=int(row["n_nodes"]),
            loser_category=str(row.get("loser_category", "unknown")),
            reward_winner=resolve_reward(row, "reward_winner", "winner_id"),
            reward_loser=resolve_reward(row, "reward_loser", "loser_id"),
            winner_n_nodes=int(row.get("winner_n_nodes", row["n_nodes"])),
            loser_n_nodes=int(row.get("loser_n_nodes", row["n_nodes"])),
        )
        for row in frame.to_dict("records")
    ]
    validate_pairs(pairs, source=path, allow_endpoint_reuse=allow_endpoint_reuse)
    return pairs


def validate_pairs(
    pairs: list[GlobalPreferencePair],
    *,
    source: str | Path = "pairs",
    allow_endpoint_reuse: bool = False,
) -> None:
    """Validate identity, endpoint reuse, and basic endpoint metadata."""
    if not pairs:
        raise ValueError(f"no pairs found in {source}")
    if any(pair.winner_id == pair.loser_id for pair in pairs):
        raise ValueError(f"a pair has identical winner and loser in {source}")
    pair_keys = [(pair.winner_id, pair.loser_id) for pair in pairs]
    if len(pair_keys) != len(set(pair_keys)):
        raise ValueError(f"duplicate pair found in {source}")
    endpoint_ids = [item for pair in pairs for item in (pair.winner_id, pair.loser_id)]
    if not allow_endpoint_reuse and len(endpoint_ids) != len(set(endpoint_ids)):
        raise ValueError(f"endpoint reused across pairs in {source}")
    if any(pair.n_nodes <= 0 for pair in pairs):
        raise ValueError(f"non-positive n_nodes found in {source}")
    for pair in pairs:
        for node_count in (pair.n_nodes, pair.winner_n_nodes, pair.loser_n_nodes):
            if node_count is not None and node_count <= 0:
                raise ValueError(f"non-positive node count in {source}")
        reward_presence = (pair.reward_winner is not None, pair.reward_loser is not None)
        if reward_presence[0] != reward_presence[1]:
            raise ValueError(
                f"pair {pair.winner_id}/{pair.loser_id} has only one endpoint reward"
            )
        for reward in (pair.reward_winner, pair.reward_loser):
            if reward is not None and not math.isfinite(float(reward)):
                raise ValueError(f"pair {pair.winner_id}/{pair.loser_id} has non-finite reward")


def validate_split_pairs(
    train_pairs: list[GlobalPreferencePair],
    validation_pairs: list[GlobalPreferencePair],
    test_pairs: list[GlobalPreferencePair] | None = None,
) -> None:
    """Reject endpoint leakage across train, validation, and test manifests."""
    groups = [train_pairs, validation_pairs] + ([] if test_pairs is None else [test_pairs])
    endpoint_sets = [{item for pair in group for item in (pair.winner_id, pair.loser_id)} for group in groups]
    for left_index, left in enumerate(endpoint_sets):
        for right in endpoint_sets[left_index + 1 :]:
            overlap = left.intersection(right)
            if overlap:
                raise ValueError(f"endpoint leakage across pair splits: {sorted(overlap)[:3]}")


def _load_endpoint(shard_path: str, sample_index: int) -> EndpointSample:
    """Load one endpoint from a trajectory shard using only endpoint fields."""
    payload = torch.load(shard_path, map_location="cpu", weights_only=False)
    sample_ids = payload.get("sample_ids")
    x = payload.get("x")
    v = payload.get("categorical")
    nodes = payload.get("nodesxsample")
    if not isinstance(sample_ids, list) or not isinstance(x, torch.Tensor):
        raise ValueError(f"invalid endpoint shard payload: {shard_path}")
    if not isinstance(v, torch.Tensor) or not isinstance(nodes, torch.Tensor):
        raise ValueError(f"endpoint shard missing categorical/nodesxsample: {shard_path}")
    if not 0 <= sample_index < len(sample_ids):
        raise IndexError(f"sample_index={sample_index} outside shard {shard_path}")
    n_nodes = int(nodes[sample_index].item())
    if x.ndim != 3 or v.ndim != 3 or not 1 <= n_nodes <= x.shape[1]:
        raise ValueError(f"invalid endpoint shapes in {shard_path}: x={x.shape}, v={v.shape}, n={n_nodes}")
    x_sample = x[sample_index, :n_nodes].float().contiguous()
    v_sample = v[sample_index, :n_nodes].float().contiguous()
    if x_sample.shape[-1] != 3 or v_sample.shape[0] != n_nodes:
        raise ValueError(f"unexpected endpoint dimensions in {shard_path}")
    if not torch.isfinite(x_sample).all() or not torch.isfinite(v_sample).all():
        raise ValueError(f"non-finite endpoint in {shard_path} index {sample_index}")
    return EndpointSample(
        sample_id=str(sample_ids[sample_index]),
        n_nodes=n_nodes,
        x=x_sample,
        v=v_sample,
    )


@lru_cache(maxsize=128)
def _cached_endpoint(shard_path: str, sample_index: int) -> EndpointSample:
    """Cache a bounded number of endpoint samples across batches."""
    return _load_endpoint(shard_path, sample_index)


class EndpointStore:
    """Bounded-cache endpoint access for Global DPO."""

    def sample(
        self,
        shard_path: str,
        sample_index: int,
        *,
        expected_sample_id: str | None = None,
    ) -> EndpointSample:
        """Return one validated endpoint sample and optionally verify identity.

        Pair manifests carry both an endpoint ID and a shard/index location.
        Checking the ID at every access prevents a stale or mismatched shard
        from silently supplying a different endpoint with the same node count.
        """
        sample = _cached_endpoint(str(shard_path), int(sample_index))
        if expected_sample_id is not None and sample.sample_id != str(expected_sample_id):
            raise ValueError(
                "endpoint identity mismatch: "
                f"shard={shard_path} index={sample_index} "
                f"manifest_id={expected_sample_id!r} actual_id={sample.sample_id!r}"
            )
        return sample
