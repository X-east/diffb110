"""Training-participation filtering for DiffGui PPO batches."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist


DEFAULT_SCORE_KEY = "reward_for_advantage"


@dataclass(frozen=True)
class FilterDecision:
    """One graph-level training filter decision."""

    graph_index: int
    keep: bool
    reason: str
    score: float | None = None

    def to_row(self) -> dict[str, Any]:
        """Return a CSV/JSON friendly row."""
        return {
            "local_mol_index": int(self.graph_index),
            "training_keep": int(self.keep),
            "training_filter_reason": self.reason,
            "training_filter_score": "" if self.score is None else float(self.score),
        }


@dataclass
class TrainingFilterSelection:
    """Selected graph indices plus filter decisions and metrics."""

    train_indices: list[int] = field(default_factory=list)
    decisions: list[FilterDecision] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    def decision_rows(self) -> list[dict[str, Any]]:
        """Return flat decision rows."""
        return [decision.to_row() for decision in self.decisions]

    def decision_by_index(self) -> dict[int, FilterDecision]:
        """Return decisions keyed by local graph index."""
        return {int(decision.graph_index): decision for decision in self.decisions}


class DiffGuiTrainingFilter:
    """Filter failed or extreme DiffGui molecules out of PPO training."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize from ``diffgui_training_filter`` YAML config."""
        cfg = config or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.require_reconstruction_success = bool(cfg.get("require_reconstruction_success", True))
        self.require_rdkit_valid = bool(cfg.get("require_rdkit_valid", True))
        self.require_reward_found = bool(cfg.get("require_reward_found", False))
        self.reject_non_finite_score = bool(cfg.get("reject_non_finite_score", True))
        self.score_key = str(cfg.get("score_key", DEFAULT_SCORE_KEY))
        self.extreme_reward_quantile_drop = max(0.0, float(cfg.get("extreme_reward_quantile_drop", 0.0)))
        self.min_train_graphs_per_rank = max(0, int(cfg.get("min_train_graphs_per_rank", 1)))
        self.ddp_crop_strategy = str(cfg.get("ddp_crop_strategy", "top_score"))
        self.quality_filters = dict(cfg.get("quality_filters", {}) or {})
        self.missing_quality_filter_policy = str(cfg.get("missing_quality_filter_policy", "ignore"))

    def select(self, molecule_rows: list[dict[str, Any]], *, batch_size: int) -> TrainingFilterSelection:
        """Return local graph indices that should participate in PPO."""
        total = max(0, int(batch_size))
        if total == 0:
            return TrainingFilterSelection(metrics=self._metrics([], total=0))
        if not self.enabled:
            decisions = [
                FilterDecision(graph_index=index, keep=True, reason="kept_filter_disabled", score=None)
                for index in range(total)
            ]
            return TrainingFilterSelection(
                train_indices=list(range(total)),
                decisions=decisions,
                metrics=self._metrics(decisions, total=total),
            )

        rows_by_index = {_row_index(row, fallback=index): row for index, row in enumerate(molecule_rows)}
        base_decisions: list[FilterDecision] = []
        candidates: list[tuple[int, float | None]] = []
        for graph_index in range(total):
            row = rows_by_index.get(graph_index, {})
            keep, reason, score = self._base_decision(row)
            if keep:
                candidates.append((graph_index, score))
            base_decisions.append(FilterDecision(graph_index=graph_index, keep=keep, reason=reason, score=score))

        extreme_indices = self._extreme_low_score_indices(candidates)
        decisions: list[FilterDecision] = []
        for decision in base_decisions:
            if decision.keep and decision.graph_index in extreme_indices:
                decisions.append(
                    FilterDecision(
                        graph_index=decision.graph_index,
                        keep=False,
                        reason="extreme_low_score",
                        score=decision.score,
                    )
                )
            else:
                decisions.append(decision)

        train_indices = [decision.graph_index for decision in decisions if decision.keep]
        return TrainingFilterSelection(
            train_indices=train_indices,
            decisions=decisions,
            metrics=self._metrics(decisions, total=total),
        )

    def crop_for_ddp(self, selection: TrainingFilterSelection, *, device: torch.device | str) -> TrainingFilterSelection:
        """Crop local selections to the global minimum graph count for DDP safety."""
        local_count = len(selection.train_indices)
        global_min = ddp_min_int(local_count, device=device)
        target_count = int(global_min)
        skipped = 0
        if target_count < self.min_train_graphs_per_rank:
            target_count = 0
            skipped = 1
        cropped_indices = self._crop_indices(selection, target_count)
        cropped_set = set(cropped_indices)
        decisions: list[FilterDecision] = []
        for decision in selection.decisions:
            if decision.keep and decision.graph_index not in cropped_set:
                reason = "ddp_cropped" if skipped == 0 else "ddp_min_train_graphs_not_met"
                decisions.append(
                    FilterDecision(
                        graph_index=decision.graph_index,
                        keep=False,
                        reason=reason,
                        score=decision.score,
                    )
                )
            else:
                decisions.append(decision)
        metrics = dict(selection.metrics)
        metrics.update(
            {
                "train_pre_ddp_selected_count": float(local_count),
                "train_ddp_min_count": float(global_min),
                "train_selected_count": float(len(cropped_indices)),
                "ppo_skipped_by_training_filter": float(skipped),
            }
        )
        metrics.update(_reason_metrics(decisions))
        return TrainingFilterSelection(train_indices=cropped_indices, decisions=decisions, metrics=metrics)

    def _base_decision(self, row: dict[str, Any]) -> tuple[bool, str, float | None]:
        """Return the non-quantile filter decision for one molecule row."""
        score = _float_or_none(row.get(self.score_key))
        if score is None and self.score_key != DEFAULT_SCORE_KEY:
            score = _float_or_none(row.get(DEFAULT_SCORE_KEY))
        if self.require_reconstruction_success and _row_int(row, "reconstruction_success") != 1:
            return False, "reconstruction_failed", score
        if self.require_rdkit_valid and _row_int(row, "rdkit_valid") != 1:
            return False, "rdkit_invalid", score
        if self.require_reward_found and _row_int(row, "reward_found") != 1:
            return False, "reward_missing", score
        if self.reject_non_finite_score and score is None:
            return False, "score_non_finite", score
        quality_reason = self._quality_failure_reason(row)
        if quality_reason is not None:
            return False, quality_reason, score
        return True, "kept", score

    def _quality_failure_reason(self, row: dict[str, Any]) -> str | None:
        """Return a quality-filter failure reason, or ``None`` if it passes."""
        for field_name, limits in self.quality_filters.items():
            value = _float_or_none(row.get(str(field_name)))
            if value is None:
                if self.missing_quality_filter_policy == "reject":
                    return f"quality_missing:{field_name}"
                continue
            if "min" in limits and value < float(limits["min"]):
                return f"quality_min:{field_name}"
            if "max" in limits and value > float(limits["max"]):
                return f"quality_max:{field_name}"
            if "equals" in limits and value != float(limits["equals"]):
                return f"quality_equals:{field_name}"
        return None

    def _extreme_low_score_indices(self, candidates: list[tuple[int, float | None]]) -> set[int]:
        """Return candidate indices in the configured bottom score quantile."""
        if self.extreme_reward_quantile_drop <= 0.0:
            return set()
        scored = [(index, score) for index, score in candidates if score is not None]
        if len(scored) <= 1:
            return set()
        drop_count = int(math.floor(len(scored) * self.extreme_reward_quantile_drop))
        if drop_count <= 0 and self.extreme_reward_quantile_drop > 0.0:
            drop_count = 1
        drop_count = min(drop_count, len(scored) - 1)
        scored.sort(key=lambda item: float(item[1]))
        return {index for index, _score in scored[:drop_count]}

    def _crop_indices(self, selection: TrainingFilterSelection, target_count: int) -> list[int]:
        """Crop selected graph indices by configured DDP strategy."""
        target = max(0, int(target_count))
        if target == 0:
            return []
        indices = list(selection.train_indices)
        if len(indices) <= target:
            return indices
        if self.ddp_crop_strategy == "top_score":
            by_index = selection.decision_by_index()
            indices.sort(
                key=lambda index: (
                    float("-inf") if by_index[index].score is None else float(by_index[index].score),
                    -int(index),
                ),
                reverse=True,
            )
            return sorted(indices[:target])
        return indices[:target]

    def _metrics(self, decisions: list[FilterDecision], *, total: int) -> dict[str, float]:
        """Return numeric filter metrics."""
        kept = sum(1 for decision in decisions if decision.keep)
        metrics = {
            "training_filter_enabled": float(self.enabled),
            "train_candidate_count": float(total),
            "train_selected_count": float(kept),
            "train_discarded_count": float(max(0, total - kept)),
        }
        metrics.update(_reason_metrics(decisions))
        return metrics


def annotate_molecule_rows(
    molecule_rows: list[dict[str, Any]],
    selection: TrainingFilterSelection,
    *,
    good_indices: set[int] | None = None,
) -> list[dict[str, Any]]:
    """Attach training-filter and good-pool flags to molecule records."""
    decision_by_index = selection.decision_by_index()
    good_set = set(good_indices or set())
    annotated: list[dict[str, Any]] = []
    for row_index, row in enumerate(molecule_rows):
        graph_index = _row_index(row, fallback=row_index)
        decision = decision_by_index.get(graph_index)
        output = dict(row)
        if decision is not None:
            output.update(decision.to_row())
        else:
            output.update(
                {
                    "training_keep": 0,
                    "training_filter_reason": "missing_filter_decision",
                    "training_filter_score": "",
                }
            )
        output["good_pool_selected"] = int(graph_index in good_set)
        annotated.append(output)
    return annotated


def ddp_min_int(value: int, *, device: torch.device | str) -> int:
    """Return the global minimum integer across ranks when DDP is initialized."""
    if not (dist.is_available() and dist.is_initialized()):
        return int(value)
    tensor = torch.tensor([int(value)], dtype=torch.long, device=torch.device(device))
    dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    return int(tensor.item())


def ddp_max_int(value: int, *, device: torch.device | str) -> int:
    """Return the global maximum integer across ranks when DDP is initialized."""
    if not (dist.is_available() and dist.is_initialized()):
        return int(value)
    tensor = torch.tensor([int(value)], dtype=torch.long, device=torch.device(device))
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return int(tensor.item())


def _row_index(row: dict[str, Any], *, fallback: int) -> int:
    """Read the local molecule index from a record row."""
    for key in ("local_mol_index", "mol_index", "graph_index", "local_index"):
        try:
            return int(float(row.get(key, "")))
        except (TypeError, ValueError):
            continue
    return int(fallback)


def _row_int(row: dict[str, Any], key: str, default: int = 0) -> int:
    """Read an int-like field from a molecule row."""
    try:
        return int(float(row.get(key, default) or default))
    except (TypeError, ValueError):
        return int(default)


def _float_or_none(value: Any) -> float | None:
    """Convert scalar-like values to finite float, otherwise ``None``."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _reason_metrics(decisions: list[FilterDecision]) -> dict[str, float]:
    """Return count metrics for filter reasons."""
    counts: Counter[str] = Counter(decision.reason for decision in decisions)
    return {
        f"train_filter_reason_{_metric_safe(reason)}": float(count)
        for reason, count in sorted(counts.items())
    }


def _metric_safe(value: str) -> str:
    """Convert a reason string into a metric-safe suffix."""
    return "".join(ch if ch.isalnum() else "_" for ch in str(value)).strip("_")
