"""Dynamic recording fields for reward-produced batch values.

Reward modules may write arbitrary tensors/lists into ``DataProto.batch``.
This module lets those modules declare how such values should be recorded at
the molecule level and reduced at the epoch level, keeping CSV schemas and DDP
aggregation out of individual training loops.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch


@dataclass(frozen=True)
class FieldSpec:
    """Declaration for a value produced by a reward or runtime component."""

    name: str
    scope: str = "molecule"  # "molecule" | "epoch"
    source: str = "batch"  # "batch" | "non_tensor_batch" | "metrics"
    source_key: str | None = None
    reduce: str = "none"  # "none" | "valid_mean" | "generated_mean" | "rank_mean" | "sum" | "min" | "max"
    default: float = 0.0
    dtype: str = "float"
    order: int = 1000

    @property
    def key(self) -> str:
        return self.source_key or self.name


@dataclass
class FieldAggregate:
    """Local sufficient statistics for one epoch-level field."""

    reduce: str
    value: float = 0.0
    count: float = 0.0


def collect_field_specs(rewarder: Any) -> list[FieldSpec]:
    """Recursively collect FieldSpec declarations from a reward tree."""
    specs: list[FieldSpec] = []
    seen_rewards: set[int] = set()

    def _walk(candidate: Any) -> None:
        if candidate is None:
            return
        obj_id = id(candidate)
        if obj_id in seen_rewards:
            return
        seen_rewards.add(obj_id)

        getter = getattr(candidate, "field_specs", None)
        if callable(getter):
            specs.extend(getter() or [])

        base_reward = getattr(candidate, "base_reward", None)
        if base_reward is not None:
            _walk(base_reward)

        rewards = getattr(candidate, "rewards", None)
        if isinstance(rewards, dict):
            for child in rewards.values():
                _walk(child)

    _walk(rewarder)
    return _dedupe_specs(specs)


def molecule_field_specs(specs: Iterable[FieldSpec]) -> list[FieldSpec]:
    return sorted((s for s in specs if s.scope == "molecule"), key=lambda s: (s.order, s.name))


def epoch_field_specs(specs: Iterable[FieldSpec]) -> list[FieldSpec]:
    return sorted((s for s in specs if s.scope == "epoch"), key=lambda s: (s.order, s.name))


def append_registered_molecule_fields(row: dict[str, Any], samples: Any, specs: Iterable[FieldSpec], index: int) -> None:
    """Append all molecule-scope registered fields for one molecule row."""
    for spec in molecule_field_specs(specs):
        values = _source_values(samples, spec)
        if values is None:
            row[spec.name] = spec.default
            continue
        row[spec.name] = _value_at(values, index, spec.default)


def compute_registered_epoch_aggregates(samples: Any, specs: Iterable[FieldSpec]) -> dict[str, FieldAggregate]:
    """Compute local sufficient stats for epoch-scope registered fields."""
    valid_flags = _valid_flags(samples)
    generated_count = _batch_size(samples, valid_flags)
    aggregates: dict[str, FieldAggregate] = {}

    for spec in epoch_field_specs(specs):
        values = _source_values(samples, spec)
        reduce = spec.reduce
        if values is None:
            aggregates[spec.name] = FieldAggregate(reduce=reduce, value=float(spec.default), count=0.0)
            continue

        vals = _to_float_list(values)
        if reduce == "valid_mean":
            selected = [vals[i] for i, flag in enumerate(valid_flags) if flag and i < len(vals)]
            aggregates[spec.name] = FieldAggregate(
                reduce=reduce,
                value=float(sum(selected)),
                count=float(len(selected)),
            )
        elif reduce in {"generated_mean", "mean"}:
            aggregates[spec.name] = FieldAggregate(
                reduce="generated_mean",
                value=float(sum(vals)),
                count=float(len(vals) if vals else generated_count),
            )
        elif reduce == "rank_mean":
            aggregates[spec.name] = FieldAggregate(
                reduce=reduce,
                value=float(sum(vals) / len(vals)) if vals else float(spec.default),
                count=1.0 if vals else 0.0,
            )
        elif reduce == "sum":
            aggregates[spec.name] = FieldAggregate(reduce=reduce, value=float(sum(vals)), count=1.0)
        elif reduce == "min":
            aggregates[spec.name] = FieldAggregate(
                reduce=reduce,
                value=float(min(vals)) if vals else float("inf"),
                count=1.0 if vals else 0.0,
            )
        elif reduce == "max":
            aggregates[spec.name] = FieldAggregate(
                reduce=reduce,
                value=float(max(vals)) if vals else float("-inf"),
                count=1.0 if vals else 0.0,
            )
        elif reduce == "none":
            continue
        else:
            raise ValueError(f"Unknown FieldSpec reduce rule for {spec.name!r}: {reduce!r}")

    return aggregates


def reduce_registered_epoch_aggregates(
    local: dict[str, FieldAggregate],
    dist_ctx: Any,
    *,
    reduce_sum_scalar,
    reduce_min_scalar,
    reduce_max_scalar,
) -> dict[str, float]:
    """Reduce registered local aggregates across DDP ranks."""
    reduced: dict[str, float] = {}
    for name, agg in local.items():
        rule = agg.reduce
        if rule in {"valid_mean", "generated_mean", "mean"}:
            total = reduce_sum_scalar(float(agg.value), dist_ctx)
            count = reduce_sum_scalar(float(agg.count), dist_ctx)
            reduced[name] = (total / count) if count > 0 else 0.0
        elif rule == "rank_mean":
            total = reduce_sum_scalar(float(agg.value), dist_ctx)
            count = reduce_sum_scalar(float(agg.count), dist_ctx)
            reduced[name] = (total / count) if count > 0 else 0.0
        elif rule == "sum":
            reduced[name] = reduce_sum_scalar(float(agg.value), dist_ctx)
        elif rule == "min":
            count = reduce_sum_scalar(float(agg.count), dist_ctx)
            value = reduce_min_scalar(float(agg.value), dist_ctx)
            reduced[name] = value if count > 0 else 0.0
        elif rule == "max":
            count = reduce_sum_scalar(float(agg.count), dist_ctx)
            value = reduce_max_scalar(float(agg.value), dist_ctx)
            reduced[name] = value if count > 0 else 0.0
    return reduced


def _dedupe_specs(specs: Iterable[FieldSpec]) -> list[FieldSpec]:
    by_key: dict[tuple[str, str], FieldSpec] = {}
    for spec in specs:
        key = (spec.scope, spec.name)
        existing = by_key.get(key)
        if existing is not None and existing != spec:
            raise ValueError(f"Conflicting FieldSpec declarations for {spec.scope}:{spec.name}")
        by_key[key] = spec
    return sorted(by_key.values(), key=lambda s: (s.order, s.scope, s.name))


def _source_values(samples: Any, spec: FieldSpec) -> Any:
    if spec.source == "batch":
        return samples.batch.get(spec.key)
    if spec.source == "non_tensor_batch":
        ntb = getattr(samples, "non_tensor_batch", None)
        return ntb.get(spec.key) if isinstance(ntb, dict) else None
    if spec.source == "metrics":
        return None
    raise ValueError(f"Unknown FieldSpec source for {spec.name!r}: {spec.source!r}")


def _valid_flags(samples: Any) -> list[int]:
    ntb = getattr(samples, "non_tensor_batch", None)
    src = ntb if isinstance(ntb, dict) else samples.batch
    return [int(v) for v in _as_list(src.get("rdkit_valid"))]


def _batch_size(samples: Any, valid_flags: list[int]) -> int:
    if valid_flags:
        return len(valid_flags)
    batch = samples.batch
    batch_size = getattr(batch, "batch_size", None)
    if batch_size:
        return int(batch_size[0])
    if "nodesxsample" in batch:
        return int(batch["nodesxsample"].shape[0])
    if "x" in batch:
        return int(batch["x"].shape[0])
    return 0


def _as_list(values: Any) -> list[Any]:
    if values is None:
        return []
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().reshape(-1).tolist()
    return list(values)


def _to_float_list(values: Any) -> list[float]:
    return [float(v) for v in _as_list(values)]


def _value_at(values: Any, index: int, default: Any) -> Any:
    seq = _as_list(values)
    if index >= len(seq):
        return default
    value = seq[index]
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().item()
    return value
