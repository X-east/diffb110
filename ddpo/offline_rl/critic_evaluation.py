"""Pure held-out metrics and baselines for offline-IQL critics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .index import EpisodeIndex, SplitManifest
from .reward_sidecar import RewardSidecar


@dataclass(frozen=True)
class CriticBaselineFacts:
    """Train-split reward baselines evaluated beside learned critics."""

    reward_mean: float
    node_intercept: float
    node_slope: float
    invalid_mean: float
    valid_mean: float


@dataclass(frozen=True)
class CriticPredictionFacts:
    """One held-out transition prediction with immutable label facts."""

    sample_id: str
    time_bin: str
    decision_index: int
    num_nodes: int
    rdkit_valid: bool
    pb_evaluated: bool
    pb_valid: bool
    docking_success: bool
    actual_good: bool
    logp: float | None
    reward_target: float
    td_target: float
    q1: float
    q2: float
    value: float

    @property
    def q_min(self) -> float:
        """Return the conservative twin-Q prediction."""
        return min(self.q1, self.q2)

    @property
    def bellman_residual(self) -> float:
        """Return conservative Q minus the one-step held-out TD target."""
        return self.q_min - self.td_target

    @property
    def quality_stratum(self) -> str:
        """Return an explicit molecule-quality layer for ordering audits."""
        if not self.rdkit_valid:
            return "invalid"
        if self.actual_good:
            return "actual_good"
        if self.pb_valid and self.docking_success:
            return "scored_not_actual_good"
        if not self.pb_evaluated:
            return "pb_not_evaluated"
        return "pb_invalid"


def fit_critic_baselines(
    episode_index: EpisodeIndex,
    split_manifest: SplitManifest,
    reward_sidecar: RewardSidecar,
    *,
    split: str = "train",
) -> CriticBaselineFacts:
    """Fit mean, node-count linear, and validity-group train baselines."""
    rows = [
        row
        for row in episode_index.rows
        if split_manifest.split_for(row.sample_id) == split
    ]
    if not rows:
        raise ValueError(f"split {split!r} contains no episodes")
    nodes = np.asarray([row.num_nodes for row in rows], dtype=np.float64)
    labels = [reward_sidecar.get(row.sample_id) for row in rows]
    for index_row, label in zip(rows, labels):
        if index_row.rdkit_valid != label.rdkit_valid:
            raise ValueError(
                f"index/sidecar validity mismatch for {index_row.sample_id!r}"
            )
    rewards = np.asarray(
        [label.reward_total_train for label in labels],
        dtype=np.float64,
    )
    valid = np.asarray([row.rdkit_valid for row in rows], dtype=bool)
    if not bool(np.isfinite(rewards).all()):
        raise ValueError("baseline rewards must be finite")
    design = np.column_stack((np.ones_like(nodes), nodes))
    intercept, slope = np.linalg.lstsq(design, rewards, rcond=None)[0]
    reward_mean = float(rewards.mean())
    invalid_mean = float(rewards[~valid].mean()) if bool((~valid).any()) else reward_mean
    valid_mean = float(rewards[valid].mean()) if bool(valid.any()) else reward_mean
    return CriticBaselineFacts(
        reward_mean=reward_mean,
        node_intercept=float(intercept),
        node_slope=float(slope),
        invalid_mean=invalid_mean,
        valid_mean=valid_mean,
    )


def regression_metrics(
    targets: Sequence[float],
    predictions: Sequence[float],
) -> dict[str, float | int]:
    """Return finite ranking and regression metrics for paired vectors."""
    target = np.asarray(targets, dtype=np.float64)
    prediction = np.asarray(predictions, dtype=np.float64)
    if target.ndim != 1 or prediction.ndim != 1 or target.shape != prediction.shape:
        raise ValueError("targets and predictions must be equal-length vectors")
    if target.size == 0:
        raise ValueError("metric vectors cannot be empty")
    if not bool(np.isfinite(target).all() and np.isfinite(prediction).all()):
        raise ValueError("metric vectors must contain only finite values")
    residual = prediction - target
    return {
        "count": int(target.size),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "pearson": _correlation(target, prediction),
        "spearman": _correlation(_average_ranks(target), _average_ranks(prediction)),
    }


def build_critic_metric_rows(
    rows: Sequence[CriticPredictionFacts],
    baselines: CriticBaselineFacts,
) -> tuple[dict[str, float | int | str], ...]:
    """Compare learned critics and registered baselines on identical rows."""
    if not rows:
        raise ValueError("critic prediction rows cannot be empty")
    targets = [row.reward_target for row in rows]
    predictions: Mapping[str, Sequence[float]] = {
        "reward_mean_baseline": [baselines.reward_mean for _ in rows],
        "node_count_baseline": [
            baselines.node_intercept + baselines.node_slope * row.num_nodes
            for row in rows
        ],
        "validity_baseline": [
            baselines.valid_mean if row.rdkit_valid else baselines.invalid_mean
            for row in rows
        ],
        "q1": [row.q1 for row in rows],
        "q2": [row.q2 for row in rows],
        "q_min": [row.q_min for row in rows],
        "value": [row.value for row in rows],
    }
    return tuple(
        {"model": name, **regression_metrics(targets, values)}
        for name, values in predictions.items()
    )


def build_stratified_metric_rows(
    rows: Sequence[CriticPredictionFacts],
) -> tuple[dict[str, float | int | str], ...]:
    """Report min-Q metrics within every represented timestep bin."""
    grouped: dict[str, list[CriticPredictionFacts]] = {}
    for row in rows:
        grouped.setdefault(row.time_bin, []).append(row)
    output: list[dict[str, float | int | str]] = []
    for time_bin, group in sorted(grouped.items()):
        terminal_metrics = regression_metrics(
            [row.reward_target for row in group],
            [row.q_min for row in group],
        )
        bellman_metrics = regression_metrics(
            [row.td_target for row in group],
            [row.q_min for row in group],
        )
        output.append(
            {
                "time_bin": time_bin,
                **terminal_metrics,
                "bellman_mae": bellman_metrics["mae"],
                "bellman_rmse": bellman_metrics["rmse"],
                "q_disagreement_mean": float(
                    np.mean([abs(row.q1 - row.q2) for row in group])
                ),
            }
        )
    return tuple(output)


def build_quality_order_rows(
    rows: Sequence[CriticPredictionFacts],
) -> tuple[dict[str, float | int | str], ...]:
    """Summarize target and critic means across explicit quality strata."""
    grouped: dict[str, list[CriticPredictionFacts]] = {}
    for row in rows:
        grouped.setdefault(row.quality_stratum, []).append(row)
    return tuple(
        {
            "quality_stratum": stratum,
            "count": len(group),
            "reward_target_mean": float(np.mean([row.reward_target for row in group])),
            "q_min_mean": float(np.mean([row.q_min for row in group])),
            "value_mean": float(np.mean([row.value for row in group])),
        }
        for stratum, group in sorted(grouped.items())
    )


def build_calibration_rows(
    rows: Sequence[CriticPredictionFacts],
    *,
    bins: int = 10,
) -> tuple[dict[str, float | int], ...]:
    """Build equal-count min-Q calibration bins without dropping tied values."""
    if bins <= 0:
        raise ValueError("calibration bins must be positive")
    if not rows:
        raise ValueError("calibration rows cannot be empty")
    ordered = sorted(rows, key=lambda row: (row.q_min, row.sample_id, row.decision_index))
    groups = np.array_split(np.arange(len(ordered)), min(bins, len(ordered)))
    output: list[dict[str, float | int]] = []
    for bin_index, positions in enumerate(groups):
        group = [ordered[int(position)] for position in positions]
        output.append(
            {
                "calibration_bin": bin_index,
                "count": len(group),
                "q_min_mean": float(np.mean([row.q_min for row in group])),
                "reward_target_mean": float(
                    np.mean([row.reward_target for row in group])
                ),
                "q_min_min": min(row.q_min for row in group),
                "q_min_max": max(row.q_min for row in group),
            }
        )
    return tuple(output)


def build_confounding_metrics(
    rows: Sequence[CriticPredictionFacts],
) -> dict[str, float | int]:
    """Measure min-Q residual association with simple molecular covariates."""
    if not rows:
        raise ValueError("confounding rows cannot be empty")
    residual = np.asarray(
        [row.q_min - row.reward_target for row in rows],
        dtype=np.float64,
    )
    nodes = np.asarray([row.num_nodes for row in rows], dtype=np.float64)
    valid = np.asarray([row.rdkit_valid for row in rows], dtype=bool)
    logp_rows = [row for row in rows if row.logp is not None]
    logp_correlation = 0.0
    if logp_rows:
        logp_correlation = _correlation(
            np.asarray([float(row.logp) for row in logp_rows], dtype=np.float64),
            np.asarray(
                [row.q_min - row.reward_target for row in logp_rows],
                dtype=np.float64,
            ),
        )
    return {
        "count": len(rows),
        "residual_vs_num_nodes_pearson": _correlation(residual, nodes),
        "residual_vs_logp_pearson": logp_correlation,
        "valid_count": int(valid.sum()),
        "invalid_count": int((~valid).sum()),
        "valid_residual_mean": (
            float(residual[valid].mean()) if bool(valid.any()) else 0.0
        ),
        "invalid_residual_mean": (
            float(residual[~valid].mean()) if bool((~valid).any()) else 0.0
        ),
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return deterministic one-based average ranks with tie handling."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    position = 0
    while position < values.size:
        end = position + 1
        while end < values.size and values[order[end]] == values[order[position]]:
            end += 1
        average_rank = (position + 1 + end) / 2.0
        ranks[order[position:end]] = average_rank
        position = end
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Return Pearson correlation or zero when either vector is constant."""
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = math.sqrt(
        float(np.square(left_centered).sum() * np.square(right_centered).sum())
    )
    if denominator == 0.0:
        return 0.0
    return float(np.dot(left_centered, right_centered) / denominator)
