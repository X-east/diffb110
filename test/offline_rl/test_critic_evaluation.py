"""Tests for held-out critic baselines, ranking metrics, and strata."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from ddpo.offline_rl.critic_evaluation import (
    CriticPredictionFacts,
    build_calibration_rows,
    build_confounding_metrics,
    build_critic_metric_rows,
    build_quality_order_rows,
    build_stratified_metric_rows,
    fit_critic_baselines,
    regression_metrics,
)
from ddpo.offline_rl.critic_evaluation import CriticBaselineFacts


def _row(index: int) -> CriticPredictionFacts:
    """Build one deterministic prediction row spanning reward strata."""
    return CriticPredictionFacts(
        sample_id=f"sample-{index}",
        time_bin="early" if index < 2 else "late",
        decision_index=index,
        num_nodes=10 + index,
        rdkit_valid=index > 0,
        pb_evaluated=index > 1,
        pb_valid=index > 2,
        docking_success=index > 2,
        actual_good=index == 4,
        logp=None if index == 0 else float(index),
        reward_target=float(index - 2),
        td_target=float(index - 2) + 0.2,
        q1=float(index - 2) + 0.1,
        q2=float(index - 2) - 0.1,
        value=float(index - 2) * 0.5,
    )


def test_regression_metrics_handle_ranking_ties_and_constants() -> None:
    """Ranking metrics should be finite and use average ranks for ties."""
    perfect = regression_metrics([0.0, 1.0, 1.0, 3.0], [0.0, 1.0, 1.0, 3.0])
    reverse = regression_metrics([0.0, 1.0, 2.0], [2.0, 1.0, 0.0])
    constant = regression_metrics([0.0, 1.0, 2.0], [1.0, 1.0, 1.0])

    assert perfect["pearson"] == pytest.approx(1.0)
    assert perfect["spearman"] == pytest.approx(1.0)
    assert reverse["spearman"] == pytest.approx(-1.0)
    assert constant["pearson"] == 0.0
    assert constant["spearman"] == 0.0


def test_metric_rows_compare_registered_baselines_and_critics() -> None:
    """Every model should be scored on exactly the same held-out targets."""
    rows = tuple(_row(index) for index in range(5))
    baselines = CriticBaselineFacts(
        reward_mean=0.0,
        node_intercept=-10.0,
        node_slope=1.0,
        invalid_mean=-2.0,
        valid_mean=0.5,
    )

    metrics = build_critic_metric_rows(rows, baselines)

    assert [row["model"] for row in metrics] == [
        "reward_mean_baseline",
        "node_count_baseline",
        "validity_baseline",
        "q1",
        "q2",
        "q_min",
        "value",
    ]
    assert all(row["count"] == 5 for row in metrics)
    assert next(row for row in metrics if row["model"] == "q_min")[
        "spearman"
    ] == pytest.approx(1.0)


def test_strata_and_calibration_preserve_all_prediction_rows() -> None:
    """Timestep, quality, and calibration summaries must retain denominators."""
    rows = tuple(_row(index) for index in range(5))
    time_rows = build_stratified_metric_rows(rows)
    quality_rows = build_quality_order_rows(rows)
    calibration_rows = build_calibration_rows(rows, bins=3)

    assert sum(int(row["count"]) for row in time_rows) == 5
    assert sum(int(row["count"]) for row in quality_rows) == 5
    assert sum(int(row["count"]) for row in calibration_rows) == 5
    assert {row["quality_stratum"] for row in quality_rows} == {
        "invalid",
        "pb_not_evaluated",
        "pb_invalid",
        "scored_not_actual_good",
        "actual_good",
    }
    assert calibration_rows[-1]["q_min_max"] == pytest.approx(rows[-1].q_min)


def test_prediction_quality_stratum_prioritizes_actual_good() -> None:
    """Actual-good should remain distinct from other PB-valid molecules."""
    row = replace(_row(3), actual_good=True)

    assert row.quality_stratum == "actual_good"


def test_baselines_and_confounding_metrics_are_train_split_bound() -> None:
    """Baselines should fit train rows and expose finite confounding facts."""
    index_rows = [
        SimpleNamespace(sample_id=f"s{index}", num_nodes=10 + index, rdkit_valid=index > 0)
        for index in range(4)
    ]
    reward_rows = {
        f"s{index}": SimpleNamespace(
            reward_total_train=float(index - 1),
            rdkit_valid=index > 0,
        )
        for index in range(4)
    }
    episode_index = SimpleNamespace(rows=index_rows)
    split_manifest = SimpleNamespace(
        split_for=lambda sample_id: "train" if sample_id != "s3" else "validation"
    )
    reward_sidecar = SimpleNamespace(get=reward_rows.__getitem__)

    baselines = fit_critic_baselines(
        episode_index,
        split_manifest,
        reward_sidecar,
    )
    confounding = build_confounding_metrics(tuple(_row(index) for index in range(5)))

    assert baselines.reward_mean == pytest.approx(0.0)
    assert baselines.invalid_mean == pytest.approx(-1.0)
    assert baselines.valid_mean == pytest.approx(0.5)
    assert confounding["count"] == 5
    assert confounding["valid_count"] == 4
    assert confounding["invalid_count"] == 1
