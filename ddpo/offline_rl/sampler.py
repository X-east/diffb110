"""Shard-local, split-aware transition sampling for DiffSBDD offline RL."""

from __future__ import annotations

import hashlib
import math
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Collection, Sequence

import torch

from .dataset import DenoisingTrajectoryShard, load_trajectory_shard
from .index import EpisodeIndex, EpisodeIndexError, EpisodeIndexRow, SplitManifest
from .protocol import DenoisingTransitionBatch
from .reward_sidecar import RewardJoinedTrajectoryShard, RewardSidecar


@dataclass(frozen=True)
class DecisionTimeBin:
    """Inclusive decision-index interval and its sampling weight."""

    start: int
    end: int
    weight: float = 1.0

    def __post_init__(self) -> None:
        """Validate interval and weight."""
        if self.start < 0 or self.end < self.start:
            raise ValueError("time bin requires 0 <= start <= end")
        if not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("time bin weight must be finite and positive")


class SharedDecisionIndexScheduler:
    """Draw one resumable decision-index stream shared by all DDP ranks."""

    def __init__(
        self,
        time_bins: Sequence[DecisionTimeBin],
        *,
        seed: int = 1,
    ) -> None:
        """Initialize an independent CPU RNG for cross-rank timestep draws."""
        if seed <= 0:
            raise ValueError("seed must be positive")
        if not time_bins:
            raise ValueError("time_bins cannot be empty")
        self.time_bins = tuple(time_bins)
        self.seed = int(seed)
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(self.seed)
        self._time_weights = torch.tensor(
            [time_bin.weight for time_bin in self.time_bins],
            dtype=torch.float64,
        )
        self._draw_count = 0

    @property
    def draw_count(self) -> int:
        """Return the number of shared decisions already produced."""
        return self._draw_count

    def next_decision_index(self) -> int:
        """Draw one configured decision index and advance scheduler state."""
        position = int(
            torch.multinomial(
                self._time_weights,
                num_samples=1,
                replacement=True,
                generator=self._generator,
            ).item()
        )
        time_bin = self.time_bins[position]
        decision_index = int(
            torch.randint(
                time_bin.start,
                time_bin.end + 1,
                (1,),
                generator=self._generator,
            ).item()
        )
        self._draw_count += 1
        return decision_index

    def state_dict(self) -> dict[str, Any]:
        """Return exact RNG progress and immutable schedule identity."""
        return {
            "format": "diffsbdd_offline_shared_decision_scheduler_v1",
            "seed": self.seed,
            "time_bins": tuple(
                (time_bin.start, time_bin.end, time_bin.weight)
                for time_bin in self.time_bins
            ),
            "generator_state": self._generator.get_state().clone(),
            "draw_count": self._draw_count,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore RNG progress only when schedule identity is unchanged."""
        expected = {
            "format": "diffsbdd_offline_shared_decision_scheduler_v1",
            "seed": self.seed,
            "time_bins": tuple(
                (time_bin.start, time_bin.end, time_bin.weight)
                for time_bin in self.time_bins
            ),
        }
        mismatches = {
            key: (expected_value, state.get(key))
            for key, expected_value in expected.items()
            if state.get(key) != expected_value
        }
        if mismatches:
            raise ValueError(
                f"shared decision scheduler identity mismatch: {mismatches}"
            )
        generator_state = state.get("generator_state")
        if not isinstance(generator_state, torch.Tensor):
            raise TypeError("scheduler generator_state must be a tensor")
        draw_count = int(state.get("draw_count", -1))
        if draw_count < 0:
            raise ValueError("scheduler draw_count must be non-negative")
        self._generator.set_state(generator_state.detach().cpu())
        self._draw_count = draw_count


@dataclass(frozen=True)
class SamplerStats:
    """Point-in-time sampler cache and latency counters."""

    batches_sampled: int
    transitions_sampled: int
    cache_hits: int
    cache_misses: int
    shard_load_seconds: float
    sample_seconds: float

    @property
    def cache_hit_rate(self) -> float:
        """Return cache hits divided by all cache accesses."""
        accesses = self.cache_hits + self.cache_misses
        return 0.0 if accesses == 0 else self.cache_hits / accesses


class _ShardCache:
    """Small CPU-only LRU cache for validated trajectory shards."""

    def __init__(
        self,
        run_dir: Path,
        *,
        max_shards: int,
        reward_sidecar: RewardSidecar | None,
    ) -> None:
        """Initialize an empty cache."""
        if max_shards <= 0:
            raise ValueError("max_shards must be positive")
        self.run_dir = run_dir
        self.max_shards = max_shards
        self.reward_sidecar = reward_sidecar
        self._cache: OrderedDict[
            str,
            DenoisingTrajectoryShard | RewardJoinedTrajectoryShard,
        ] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.load_seconds = 0.0

    def get(
        self,
        shard_path: str,
        shard_sha256: str,
    ) -> DenoisingTrajectoryShard | RewardJoinedTrajectoryShard:
        """Return a cached read-only shard or load and provenance-check it."""
        key = f"{shard_path}|{shard_sha256.lower()}"
        cached = self._cache.pop(key, None)
        if cached is not None:
            self._cache[key] = cached
            self.hits += 1
            return cached

        started = time.perf_counter()
        shard = load_trajectory_shard(
            self.run_dir / shard_path,
            expected_sha256=shard_sha256,
        )
        loaded: DenoisingTrajectoryShard | RewardJoinedTrajectoryShard = shard
        if self.reward_sidecar is not None:
            loaded = self.reward_sidecar.join_shard(shard)
        self.load_seconds += time.perf_counter() - started
        self.misses += 1
        self._cache[key] = loaded
        while len(self._cache) > self.max_shards:
            self._cache.popitem(last=False)
        return loaded


class OfflineTransitionSampler:
    """Sample same-shard episode transitions from one frozen data split."""

    def __init__(
        self,
        *,
        run_dir: str | Path,
        episode_index: EpisodeIndex,
        split_manifest: SplitManifest,
        split: str,
        time_bins: Sequence[DecisionTimeBin],
        reward_sidecar: RewardSidecar | None = None,
        cache_size: int = 2,
        seed: int = 1,
        rank: int = 0,
        world_size: int = 1,
        same_decision_index_per_batch: bool = True,
        eligible_sample_ids: Collection[str] | None = None,
    ) -> None:
        """Initialize deterministic split, rank, time, and shard sampling."""
        if seed <= 0:
            raise ValueError("seed must be positive")
        if world_size <= 0 or rank < 0 or rank >= world_size:
            raise ValueError("rank/world_size must satisfy 0 <= rank < world_size")
        if not time_bins:
            raise ValueError("time_bins cannot be empty")
        split_manifest.validate_against(episode_index)
        self.run_dir = Path(run_dir).resolve()
        self.episode_index = episode_index
        self.split_manifest = split_manifest
        self.split = split
        self.time_bins = tuple(time_bins)
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.same_decision_index_per_batch = same_decision_index_per_batch
        self.reward_sidecar = reward_sidecar
        selected_ids = (
            None
            if eligible_sample_ids is None
            else frozenset(str(sample_id) for sample_id in eligible_sample_ids)
        )
        if selected_ids is not None:
            if not selected_ids:
                raise ValueError("eligible_sample_ids cannot be empty")
            known_ids = {row.sample_id for row in episode_index.rows}
            unknown_ids = sorted(selected_ids - known_ids)
            if unknown_ids:
                raise EpisodeIndexError(
                    "eligible_sample_ids contains unknown episodes: "
                    f"{unknown_ids[:5]}"
                )
        self.eligible_sample_ids = selected_ids
        self.eligible_sample_ids_sha256 = (
            None
            if selected_ids is None
            else hashlib.sha256(
                "".join(f"{sample_id}\n" for sample_id in sorted(selected_ids)).encode(
                    "utf-8"
                )
            ).hexdigest()
        )
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(seed)
        self._cache = _ShardCache(
            self.run_dir,
            max_shards=cache_size,
            reward_sidecar=reward_sidecar,
        )
        self._batches_sampled = 0
        self._transitions_sampled = 0
        self._sample_seconds = 0.0

        rows_by_shard: dict[tuple[str, str], list[EpisodeIndexRow]] = defaultdict(
            list
        )
        for row in episode_index.rows:
            if split_manifest.split_for(row.sample_id) != split:
                continue
            if selected_ids is not None and row.sample_id not in selected_ids:
                continue
            rank_owner = int(
                hashlib.sha256(row.shard_path.encode("utf-8")).hexdigest(),
                16,
            ) % world_size
            if rank_owner == rank:
                rows_by_shard[
                    (row.shard_path, row.source_shard_sha256.lower())
                ].append(row)
        if not rows_by_shard:
            raise EpisodeIndexError(
                f"split {split!r} has no episodes assigned to rank "
                f"{rank}/{world_size}"
            )
        self._shard_keys = tuple(sorted(rows_by_shard))
        self._rows_by_shard = {
            key: tuple(sorted(rows, key=lambda row: row.episode_index))
            for key, rows in rows_by_shard.items()
        }
        self._shard_weights = torch.tensor(
            [len(self._rows_by_shard[key]) for key in self._shard_keys],
            dtype=torch.float64,
        )
        self._time_weights = torch.tensor(
            [time_bin.weight for time_bin in self.time_bins],
            dtype=torch.float64,
        )
        self._max_decision_index = min(
            row.num_diffusion_steps for row in episode_index.rows
        )
        for time_bin in self.time_bins:
            if time_bin.end > self._max_decision_index:
                raise ValueError(
                    f"time bin end {time_bin.end} exceeds final decision index "
                    f"{self._max_decision_index}"
                )

    @property
    def num_eligible_episodes(self) -> int:
        """Return split episodes assigned to this rank."""
        return sum(len(rows) for rows in self._rows_by_shard.values())

    @property
    def num_eligible_shards(self) -> int:
        """Return shards assigned to this rank."""
        return len(self._shard_keys)

    @property
    def shard_paths(self) -> tuple[str, ...]:
        """Return rank-owned shard paths for DDP overlap diagnostics."""
        return tuple(key[0] for key in self._shard_keys)

    def sample_batch(
        self,
        batch_size: int,
        *,
        device: str | torch.device = "cpu",
        non_blocking: bool = False,
        decision_index: int | None = None,
    ) -> DenoisingTransitionBatch:
        """Sample one shard-local batch and optionally transfer it to a device."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        started = time.perf_counter()
        shard_position = int(
            torch.multinomial(
                self._shard_weights,
                num_samples=1,
                replacement=True,
                generator=self._generator,
            ).item()
        )
        shard_key = self._shard_keys[shard_position]
        eligible_rows = self._rows_by_shard[shard_key]
        episode_positions = torch.randint(
            0,
            len(eligible_rows),
            (batch_size,),
            generator=self._generator,
        )
        episode_indices = [
            eligible_rows[int(position)].episode_index
            for position in episode_positions
        ]
        if decision_index is None:
            decision_indices = self._sample_decision_indices(batch_size)
        else:
            self._validate_explicit_decision_index(decision_index)
            decision_indices = [decision_index] * batch_size
        loaded = self._cache.get(*shard_key)
        batch = loaded.select_transitions(
            episode_indices,
            decision_indices,
        )
        target_device = torch.device(device)
        if target_device.type != "cpu":
            batch = batch.to(target_device, non_blocking=non_blocking)
        self._batches_sampled += 1
        self._transitions_sampled += batch_size
        self._sample_seconds += time.perf_counter() - started
        return batch

    def stats(self) -> SamplerStats:
        """Return current immutable sampler counters."""
        return SamplerStats(
            batches_sampled=self._batches_sampled,
            transitions_sampled=self._transitions_sampled,
            cache_hits=self._cache.hits,
            cache_misses=self._cache.misses,
            shard_load_seconds=self._cache.load_seconds,
            sample_seconds=self._sample_seconds,
        )

    def state_dict(self) -> dict[str, Any]:
        """Return deterministic RNG/progress state for checkpointing."""
        return {
            "format": "diffsbdd_offline_transition_sampler_v2",
            "episode_index_sha256": self.episode_index.sha256,
            "split": self.split,
            "seed": self.seed,
            "rank": self.rank,
            "world_size": self.world_size,
            "same_decision_index_per_batch": self.same_decision_index_per_batch,
            "time_bins": tuple(
                (time_bin.start, time_bin.end, time_bin.weight)
                for time_bin in self.time_bins
            ),
            "shard_keys": self._shard_keys,
            "reward_version": (
                None
                if self.reward_sidecar is None
                else self.reward_sidecar.reward_version
            ),
            "eligible_sample_ids_sha256": self.eligible_sample_ids_sha256,
            "generator_state": self._generator.get_state().clone(),
            "batches_sampled": self._batches_sampled,
            "transitions_sampled": self._transitions_sampled,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore RNG/progress only when immutable sampler identity matches."""
        state_format = state.get("format")
        if state_format not in {
            "diffsbdd_offline_transition_sampler_v1",
            "diffsbdd_offline_transition_sampler_v2",
        }:
            raise ValueError(f"unsupported sampler checkpoint format {state_format!r}")
        if (
            state_format == "diffsbdd_offline_transition_sampler_v1"
            and self.eligible_sample_ids_sha256 is not None
        ):
            raise ValueError(
                "legacy sampler checkpoint cannot resume a curated selection"
            )
        expected = {
            "format": state_format,
            "episode_index_sha256": self.episode_index.sha256,
            "split": self.split,
            "seed": self.seed,
            "rank": self.rank,
            "world_size": self.world_size,
            "same_decision_index_per_batch": (
                self.same_decision_index_per_batch
            ),
            "time_bins": tuple(
                (time_bin.start, time_bin.end, time_bin.weight)
                for time_bin in self.time_bins
            ),
            "shard_keys": self._shard_keys,
            "reward_version": (
                None
                if self.reward_sidecar is None
                else self.reward_sidecar.reward_version
            ),
        }
        if state_format == "diffsbdd_offline_transition_sampler_v2":
            expected["eligible_sample_ids_sha256"] = (
                self.eligible_sample_ids_sha256
            )
        mismatches = {
            key: (expected_value, state.get(key))
            for key, expected_value in expected.items()
            if state.get(key) != expected_value
        }
        if mismatches:
            raise ValueError(f"sampler checkpoint identity mismatch: {mismatches}")
        generator_state = state.get("generator_state")
        if not isinstance(generator_state, torch.Tensor):
            raise TypeError("sampler generator_state must be a tensor")
        self._generator.set_state(generator_state.detach().cpu())
        self._batches_sampled = int(state.get("batches_sampled", 0))
        self._transitions_sampled = int(state.get("transitions_sampled", 0))

    def _sample_decision_indices(self, batch_size: int) -> list[int]:
        """Sample decision indices from configured weighted inclusive bins."""
        if self.same_decision_index_per_batch:
            time_bin = self._sample_time_bin()
            decision = int(
                torch.randint(
                    time_bin.start,
                    time_bin.end + 1,
                    (1,),
                    generator=self._generator,
                ).item()
            )
            return [decision] * batch_size
        decisions: list[int] = []
        for _ in range(batch_size):
            time_bin = self._sample_time_bin()
            decisions.append(
                int(
                    torch.randint(
                        time_bin.start,
                        time_bin.end + 1,
                        (1,),
                        generator=self._generator,
                    ).item()
                )
            )
        return decisions

    def _validate_explicit_decision_index(self, decision_index: int) -> None:
        """Require a DDP-supplied decision to belong to configured time bins."""
        if isinstance(decision_index, bool) or not isinstance(
            decision_index,
            int,
        ):
            raise TypeError("decision_index must be an integer")
        if not 0 <= decision_index <= self._max_decision_index:
            raise ValueError(
                f"decision_index must lie in [0,{self._max_decision_index}]"
            )
        if not any(
            time_bin.start <= decision_index <= time_bin.end
            for time_bin in self.time_bins
        ):
            raise ValueError(
                f"decision_index {decision_index} is outside configured time bins"
            )

    def _sample_time_bin(self) -> DecisionTimeBin:
        """Sample one configured time bin."""
        position = int(
            torch.multinomial(
                self._time_weights,
                num_samples=1,
                replacement=True,
                generator=self._generator,
            ).item()
        )
        return self.time_bins[position]


def default_diffsbdd_time_bins() -> tuple[DecisionTimeBin, ...]:
    """Return the pre-registered 1000-step DiffSBDD decision bins."""
    return (
        DecisionTimeBin(0, 49),
        DecisionTimeBin(50, 99),
        DecisionTimeBin(100, 249),
        DecisionTimeBin(250, 499),
        DecisionTimeBin(500, 749),
        DecisionTimeBin(750, 999),
        DecisionTimeBin(1000, 1000),
    )
