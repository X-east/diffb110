from __future__ import annotations

"""Disk-backed good-molecule and rollout-trajectory replay utilities.

The design follows the replay-buffer pattern used by SMILES-RL/REINVENT, but
keeps DiffSBDD-specific PPO data off GPU: selected rollout trajectories are
stored as CPU :class:`DataProto` shards and only streamed back through the
existing actor micro-batch path when replay is explicitly enabled.
"""

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ddpo.runtime.protocol import DataProto


REQUIRED_TRAINING_TENSOR_KEYS = (
    "latents",
    "logps",
    "nodesxsample",
    "timesteps",
    "advantages",
)
OPTIONAL_TRAINING_TENSOR_KEYS = ("context",)
DEFAULT_SCORE_KEY = "reward_total_clamped"


@dataclass
class GoodMoleculeSelection:
    """Selected molecule indices plus JSON-serializable per-molecule records."""

    indices: list[int] = field(default_factory=list)
    records: list[dict[str, Any]] = field(default_factory=list)

    def __len__(self) -> int:
        """Return the number of selected molecules."""
        return len(self.indices)


@dataclass
class ReplayStageResult:
    """Result of staging current rollout data with optional replay samples."""

    samples: DataProto
    metrics: dict[str, float]


def _as_list(value: Any, expected_len: int | None = None) -> list[Any]:
    """Convert tensor/list/array-like batch values to a Python list."""
    if value is None:
        return [] if expected_len is None else [None] * expected_len
    if isinstance(value, torch.Tensor):
        values = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, np.ndarray):
        values = value.reshape(-1).tolist()
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = list(np.asarray(value, dtype=object).reshape(-1).tolist())

    if expected_len is not None and len(values) < expected_len:
        values = values + [None] * (expected_len - len(values))
    return values


def _batch_value(data: DataProto, key: str) -> Any:
    """Read a value from tensor batch first, then non-tensor metadata."""
    if data.batch is not None and key in data.batch.keys():
        return data.batch[key]
    if data.non_tensor_batch is not None and key in data.non_tensor_batch:
        return data.non_tensor_batch[key]
    return None


def _float_or_none(value: Any) -> float | None:
    """Convert scalar-like values to finite float, otherwise ``None``."""
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _json_safe(value: Any) -> Any:
    """Convert common NumPy/Torch scalar values to JSON-safe Python objects."""
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return _json_safe(value.detach().cpu().item())
        return value.detach().cpu().reshape(-1).tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _select_tensor_training_view(data: DataProto) -> DataProto:
    """Return the tensor-only PPO training view needed by ``EDMActor``."""
    if data.batch is None:
        raise ValueError("Cannot build replay training view from DataProto without tensor batch")

    keys = set(str(key) for key in data.batch.keys())
    missing = [key for key in REQUIRED_TRAINING_TENSOR_KEYS if key not in keys]
    if missing:
        raise ValueError(f"Cannot build replay training view; missing tensor keys: {missing}")

    selected_keys = [
        key
        for key in (*REQUIRED_TRAINING_TENSOR_KEYS, *OPTIONAL_TRAINING_TENSOR_KEYS)
        if key in keys
    ]
    return data.select(batch_keys=selected_keys, non_tensor_batch_keys=[]).to("cpu")


def _record_smiles_key(record: dict[str, Any]) -> str | None:
    """Return a deduplication key for record SMILES."""
    smiles_value = record.get("smiles")
    if smiles_value in (None, ""):
        return None
    return str(smiles_value)


def _sort_and_dedupe_records(
    records: list[dict[str, Any]],
    *,
    deduplicate_smiles: bool,
) -> list[dict[str, Any]]:
    """Sort records by score descending and optionally keep one per SMILES."""
    sorted_records = sorted(records, key=lambda record: float(record.get("score", 0.0)), reverse=True)
    if not deduplicate_smiles:
        return sorted_records

    retained: list[dict[str, Any]] = []
    seen_smiles: set[str] = set()
    for record in sorted_records:
        smiles_key = _record_smiles_key(record)
        if smiles_key is not None and smiles_key in seen_smiles:
            continue
        if smiles_key is not None:
            seen_smiles.add(smiles_key)
        retained.append(record)
    return retained


class GoodMoleculeSelector:
    """Select high-quality molecules from a rewarded DiffSBDD rollout batch."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self.score_key = str(cfg.get("score_key", DEFAULT_SCORE_KEY))
        self.top_k_per_epoch = max(0, int(cfg.get("top_k_per_epoch", cfg.get("store_top_k", 64))))
        self.min_score = cfg.get("min_score")
        self.require_valid = bool(cfg.get("require_valid", True))
        self.deduplicate_smiles = bool(cfg.get("deduplicate_smiles", True))
        self.quality_filters = dict(cfg.get("quality_filters", {}) or {})
        self.missing_quality_filter_policy = str(cfg.get("missing_quality_filter_policy", "ignore"))
        quality_filter_fields = tuple(str(field_name) for field_name in self.quality_filters.keys())
        self.extra_record_fields = tuple(
            dict.fromkeys(
                str(field_name)
                for field_name in (
                    *cfg.get(
                        "record_fields",
                        (
                            "reward_total_raw",
                            "reward_total",
                            "reward_total_clamped",
                            "qed_raw",
                            "sa_raw",
                            "docking_raw",
                        ),
                    ),
                    *quality_filter_fields,
                )
            )
        )

    def _passes_quality_filters(self, idx: int, values_by_field: dict[str, list[Any]]) -> bool:
        """Return whether a molecule passes configured numeric quality filters."""
        for field_name, limits in self.quality_filters.items():
            values = values_by_field.get(str(field_name), [])
            raw_value = values[idx] if idx < len(values) else None
            value = _float_or_none(raw_value)
            if value is None:
                if self.missing_quality_filter_policy == "reject":
                    return False
                continue
            if "min" in limits and value < float(limits["min"]):
                return False
            if "max" in limits and value > float(limits["max"]):
                return False
            if "equals" in limits and value != float(limits["equals"]):
                return False
        return True

    def select(self, data: DataProto, epoch: int, rank: int) -> GoodMoleculeSelection:
        """Return top valid molecules and compact records for manifest writing."""
        batch_size = len(data)
        if batch_size <= 0 or self.top_k_per_epoch <= 0:
            return GoodMoleculeSelection()

        scores = _as_list(_batch_value(data, self.score_key), expected_len=batch_size)
        smiles = _as_list(_batch_value(data, "smiles"), expected_len=batch_size)
        valid = _as_list(_batch_value(data, "rdkit_valid"), expected_len=batch_size)
        failure_type = _as_list(_batch_value(data, "failure_type"), expected_len=batch_size)

        extra_values = {
            field_name: _as_list(_batch_value(data, field_name), expected_len=batch_size)
            for field_name in self.extra_record_fields
        }

        candidates: list[tuple[int, float, dict[str, Any]]] = []
        for idx in range(batch_size):
            score = _float_or_none(scores[idx])
            if score is None:
                continue
            if self.min_score is not None and score < float(self.min_score):
                continue
            is_valid = bool(int(valid[idx])) if valid[idx] is not None else False
            if self.require_valid and not is_valid:
                continue
            if not self._passes_quality_filters(idx, extra_values):
                continue
            record = {
                "epoch": int(epoch),
                "rank": int(rank),
                "local_index": int(idx),
                "score_key": self.score_key,
                "score": float(score),
                "smiles": _json_safe(smiles[idx]),
                "rdkit_valid": int(is_valid),
                "failure_type": _json_safe(failure_type[idx]),
            }
            for field_name, values in extra_values.items():
                record[field_name] = _json_safe(values[idx])
            candidates.append((idx, score, record))

        candidates.sort(key=lambda item: item[1], reverse=True)
        selected_indices: list[int] = []
        selected_records: list[dict[str, Any]] = []
        seen_smiles: set[str] = set()
        for idx, _score, record in candidates:
            smiles_key = _record_smiles_key(record)
            if self.deduplicate_smiles and smiles_key is not None and smiles_key in seen_smiles:
                continue
            if smiles_key is not None:
                seen_smiles.add(smiles_key)
            selected_indices.append(idx)
            selected_records.append(record)
            if len(selected_indices) >= self.top_k_per_epoch:
                break

        return GoodMoleculeSelection(indices=selected_indices, records=selected_records)


class GoodMoleculePool:
    """Append-only JSONL pool plus rolling top-quality retained snapshot."""

    def __init__(
        self,
        root_dir: str | Path,
        selector: GoodMoleculeSelector,
        *,
        rank: int = 0,
        enabled: bool = False,
        retention_window_epochs: int = 10,
        retained_top_k: int | None = None,
        deduplicate_retained_smiles: bool = True,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.selector = selector
        self.rank = int(rank)
        self.enabled = bool(enabled)
        self.retention_window_epochs = max(1, int(retention_window_epochs))
        self.retained_top_k = (
            max(0, int(retained_top_k))
            if retained_top_k is not None
            else int(selector.top_k_per_epoch)
        )
        self.deduplicate_retained_smiles = bool(deduplicate_retained_smiles)
        self.manifest_path = self.root_dir / f"good_molecules_rank{self.rank:02d}.jsonl"
        self.retained_path = self.root_dir / f"good_molecules_retained_rank{self.rank:02d}.jsonl"
        self.last_retained_records: list[dict[str, Any]] = []
        if self.enabled:
            self.root_dir.mkdir(parents=True, exist_ok=True)

    def _load_manifest_records(self) -> list[dict[str, Any]]:
        """Load all append-only manifest records for this rank."""
        if not self.manifest_path.exists():
            return []
        records: list[dict[str, Any]] = []
        with open(self.manifest_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    def retained_records(self, epoch: int) -> list[dict[str, Any]]:
        """Return best records inside the rolling retention epoch window."""
        min_epoch = int(epoch) - self.retention_window_epochs + 1
        window_records = [
            record
            for record in self._load_manifest_records()
            if min_epoch <= int(record.get("epoch", -1)) <= int(epoch)
        ]
        retained = _sort_and_dedupe_records(
            window_records,
            deduplicate_smiles=self.deduplicate_retained_smiles,
        )
        if self.retained_top_k > 0:
            retained = retained[: self.retained_top_k]
        return retained

    def _write_retained_snapshot(self, epoch: int) -> list[dict[str, Any]]:
        """Rewrite the rolling retained top-molecule snapshot."""
        retained = self.retained_records(epoch)
        with open(self.retained_path, "w", encoding="utf-8") as handle:
            for record in retained:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        self.last_retained_records = retained
        return retained

    def add_batch(self, data: DataProto, epoch: int) -> GoodMoleculeSelection:
        """Select and optionally persist good-molecule records for one epoch."""
        selection = self.selector.select(data=data, epoch=epoch, rank=self.rank)
        if not self.enabled:
            return selection

        if len(selection) > 0:
            with open(self.manifest_path, "a", encoding="utf-8") as handle:
                for record in selection.records:
                    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        self._write_retained_snapshot(epoch)
        return selection


class DiskTrajectoryReplayBuffer:
    """CPU/disk-backed short-window replay buffer for PPO rollout trajectories."""

    def __init__(
        self,
        root_dir: str | Path,
        *,
        rank: int = 0,
        enabled: bool = False,
        replay_ratio: float = 0.0,
        sample_size: int | None = None,
        max_age_epochs: int = 10,
        sample_strategy: str = "top_score",
        seed: int = 0,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.rank = int(rank)
        self.enabled = bool(enabled)
        self.replay_ratio = max(0.0, float(replay_ratio))
        self.sample_size = None if sample_size is None else max(0, int(sample_size))
        self.max_age_epochs = max(1, int(max_age_epochs))
        self.sample_strategy = str(sample_strategy)
        self.rng = random.Random(int(seed) + self.rank)
        self.shards_dir = self.root_dir / f"rank_{self.rank:02d}" / "shards"
        self.manifest_path = self.root_dir / f"trajectory_manifest_rank{self.rank:02d}.jsonl"
        self._manifest: list[dict[str, Any]] = []
        if self.enabled:
            self.shards_dir.mkdir(parents=True, exist_ok=True)
            self.root_dir.mkdir(parents=True, exist_ok=True)
            self._manifest = self._load_manifest()

    def _load_manifest(self) -> list[dict[str, Any]]:
        """Load existing shard manifest if present."""
        if not self.manifest_path.exists():
            return []
        records = []
        with open(self.manifest_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    def _append_manifest(self, record: dict[str, Any]) -> None:
        """Append one shard record to the manifest."""
        with open(self.manifest_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        self._manifest.append(record)

    def add_batch(self, data: DataProto, selection: GoodMoleculeSelection, epoch: int) -> dict[str, float]:
        """Store selected current-epoch trajectories as a CPU shard."""
        if not self.enabled or len(selection) == 0:
            return {"replay_store_count": 0.0}

        selected = data.select_idxs(selection.indices)
        training_view = _select_tensor_training_view(selected)
        shard_path = self.shards_dir / f"epoch_{int(epoch):04d}_rank_{self.rank:02d}.pkl"
        training_view.save_to_disk(shard_path)

        scores = [
            float(record["score"])
            for record in selection.records
            if _float_or_none(record.get("score")) is not None
        ]
        manifest_record = {
            "epoch": int(epoch),
            "rank": self.rank,
            "path": str(shard_path.relative_to(self.root_dir)),
            "count": len(selection),
            "score_mean": float(sum(scores) / len(scores)) if scores else 0.0,
            "score_max": float(max(scores)) if scores else 0.0,
        }
        self._append_manifest(manifest_record)
        return {
            "replay_store_count": float(len(selection)),
            "replay_store_score_mean": manifest_record["score_mean"],
            "replay_store_score_max": manifest_record["score_max"],
        }

    def _target_sample_count(self, current_batch_size: int) -> int:
        """Resolve how many replay samples should be added this epoch."""
        if self.sample_size is not None:
            return int(self.sample_size)
        return int(math.floor(max(0, current_batch_size) * self.replay_ratio))

    def _eligible_shards(self, epoch: int) -> list[dict[str, Any]]:
        """Return manifest entries from previous epochs within the age window."""
        min_epoch = int(epoch) - self.max_age_epochs
        return [
            record
            for record in self._manifest
            if min_epoch <= int(record.get("epoch", -1)) < int(epoch)
            and int(record.get("count", 0)) > 0
        ]

    def sample(self, epoch: int, current_batch_size: int) -> tuple[DataProto | None, dict[str, float]]:
        """Sample historical trajectory shards for this epoch."""
        if not self.enabled:
            return None, {}

        target_count = self._target_sample_count(current_batch_size)
        if target_count <= 0:
            return None, {
                "replay_sample_count": 0.0,
                "replay_eligible_shards": float(len(self._eligible_shards(epoch))),
            }

        eligible = self._eligible_shards(epoch)
        if not eligible:
            return None, {"replay_sample_count": 0.0, "replay_eligible_shards": 0.0}

        if self.sample_strategy == "random":
            self.rng.shuffle(eligible)
        elif self.sample_strategy == "recent":
            eligible.sort(key=lambda record: int(record.get("epoch", -1)), reverse=True)
        else:
            eligible.sort(
                key=lambda record: (
                    float(record.get("score_max", 0.0)),
                    int(record.get("epoch", -1)),
                ),
                reverse=True,
            )
        loaded: list[DataProto] = []
        loaded_count = 0
        for record in eligible:
            shard_path = self.root_dir / str(record["path"])
            if not shard_path.exists():
                continue
            shard = DataProto.load_from_disk(shard_path).to("cpu")
            loaded.append(shard)
            loaded_count += len(shard)
            if loaded_count >= target_count:
                break

        if not loaded:
            return None, {"replay_sample_count": 0.0, "replay_eligible_shards": float(len(eligible))}

        combined = DataProto.concat(loaded)
        if len(combined) > target_count:
            indices = self.rng.sample(range(len(combined)), target_count)
            combined = combined.select_idxs(indices)
        combined = combined.to("cpu")
        return combined, {
            "replay_sample_count": float(len(combined)),
            "replay_eligible_shards": float(len(eligible)),
        }


class ReplayCoordinator:
    """Coordinate good-molecule archiving and trajectory replay staging."""

    def __init__(
        self,
        good_molecule_pool: GoodMoleculePool | None = None,
        trajectory_replay: DiskTrajectoryReplayBuffer | None = None,
    ) -> None:
        self.good_molecule_pool = good_molecule_pool
        self.trajectory_replay = trajectory_replay

    @property
    def enabled(self) -> bool:
        """Return whether either replay subsystem is active."""
        pool_active = self.good_molecule_pool is not None and self.good_molecule_pool.enabled
        replay_active = self.trajectory_replay is not None and self.trajectory_replay.enabled
        return pool_active or replay_active

    def stage_for_training(self, current_samples: DataProto, epoch: int) -> ReplayStageResult:
        """Sample historical replay, then archive the current high-score rollout."""
        metrics: dict[str, float] = {}
        train_samples = current_samples
        replay_samples = None

        if self.trajectory_replay is not None:
            replay_samples, sample_metrics = self.trajectory_replay.sample(
                epoch=epoch,
                current_batch_size=len(current_samples),
            )
            metrics.update(sample_metrics)

        if replay_samples is not None and len(replay_samples) > 0:
            current_training_view = _select_tensor_training_view(current_samples)
            train_samples = DataProto.concat([current_training_view, replay_samples])
            metrics["replay_train_batch_size"] = float(len(train_samples))
        else:
            metrics.setdefault("replay_sample_count", 0.0)
            metrics["replay_train_batch_size"] = float(len(train_samples))

        selection = GoodMoleculeSelection()
        if self.good_molecule_pool is not None:
            selection = self.good_molecule_pool.add_batch(current_samples, epoch=epoch)
            metrics["good_pool_selected_count"] = float(len(selection))
            retained_records = self.good_molecule_pool.last_retained_records
            retained_scores = [
                float(record["score"])
                for record in retained_records
                if _float_or_none(record.get("score")) is not None
            ]
            metrics["good_pool_retained_count"] = float(len(retained_records))
            if retained_scores:
                metrics["good_pool_retained_score_mean"] = float(sum(retained_scores) / len(retained_scores))
                metrics["good_pool_retained_score_max"] = float(max(retained_scores))

        if self.trajectory_replay is not None:
            store_metrics = self.trajectory_replay.add_batch(
                current_samples,
                selection=selection,
                epoch=epoch,
            )
            metrics.update(store_metrics)

        return ReplayStageResult(samples=train_samples, metrics=metrics)


def build_replay_coordinator(
    *,
    config: dict[str, Any],
    output_dir: str,
    rank: int,
    logger: Any | None = None,
) -> ReplayCoordinator | None:
    """Build optional replay coordinator from config.

    Both subsystems are disabled by default.  Supported top-level config blocks:
    ``good_molecule_pool`` and ``trajectory_replay``.
    """
    pool_cfg = dict(config.get("good_molecule_pool", {}) or {})
    replay_cfg = dict(config.get("trajectory_replay", {}) or {})
    pool_enabled = bool(pool_cfg.get("enabled", False))
    replay_enabled = bool(replay_cfg.get("enabled", False))
    if not pool_enabled and not replay_enabled:
        return None

    root = Path(output_dir)
    selector_cfg = dict(pool_cfg)
    selector_cfg.update({
        k: v
        for k, v in replay_cfg.items()
        if k in {
            "score_key",
            "top_k_per_epoch",
            "store_top_k",
            "min_score",
            "require_valid",
            "deduplicate_smiles",
            "quality_filters",
            "missing_quality_filter_policy",
            "record_fields",
        }
    })
    selector = GoodMoleculeSelector(selector_cfg)
    retention_window_epochs = int(pool_cfg.get("retention_window_epochs", 10))
    pool = GoodMoleculePool(
        root_dir=root / str(pool_cfg.get("subdir", "good_molecule_pool")),
        selector=selector,
        rank=rank,
        enabled=pool_enabled or replay_enabled,
        retention_window_epochs=retention_window_epochs,
        retained_top_k=pool_cfg.get("retained_top_k"),
        deduplicate_retained_smiles=bool(pool_cfg.get("deduplicate_retained_smiles", True)),
    )
    replay = DiskTrajectoryReplayBuffer(
        root_dir=root / str(replay_cfg.get("subdir", "trajectory_replay")),
        rank=rank,
        enabled=replay_enabled,
        replay_ratio=float(replay_cfg.get("replay_ratio", 0.0)),
        sample_size=replay_cfg.get("sample_size"),
        max_age_epochs=int(replay_cfg.get("max_age_epochs", retention_window_epochs)),
        sample_strategy=str(replay_cfg.get("sample_strategy", "top_score")),
        seed=int(config.get("train", {}).get("seed", 42)),
    )

    if logger is not None:
        logger.info(
            "Initialized replay coordinator | good_pool_enabled=%s trajectory_replay_enabled=%s",
            pool_enabled,
            replay_enabled,
        )
    return ReplayCoordinator(good_molecule_pool=pool, trajectory_replay=replay)
