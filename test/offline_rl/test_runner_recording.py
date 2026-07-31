"""Tests for durable offline-IQL epoch execution and exact resume."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from ddpo.offline_rl.recording import (
    EPOCH_METRIC_FIELDS,
    STEP_METRIC_FIELDS,
    LocalIQLRecorder,
)
from ddpo.offline_rl.runner import IQLRunnerConfig, OfflineIQLRunner
from ddpo.offline_rl.sampler import SamplerStats
from ddpo.offline_rl.trainer import ActorStepMetrics, IQLStepMetrics


class _FakeSampler:
    """Deterministic sampler double with resumable progress."""

    def __init__(self) -> None:
        """Initialize a zero sample count."""
        self.samples = 0

    def sample_batch(
        self,
        batch_size: int,
        *,
        device: str,
        non_blocking: bool,
    ) -> tuple[int, str, bool]:
        """Return a small marker while advancing sample progress."""
        self.samples += batch_size
        return self.samples, device, non_blocking

    def stats(self) -> SamplerStats:
        """Return predictable cumulative counters."""
        return SamplerStats(
            batches_sampled=self.samples // 2,
            transitions_sampled=self.samples,
            cache_hits=max(0, self.samples // 2 - 1),
            cache_misses=min(1, self.samples // 2),
            shard_load_seconds=0.25,
            sample_seconds=0.5,
        )

    def state_dict(self) -> dict[str, Any]:
        """Serialize fake progress."""
        return {"format": "fake_sampler_v1", "samples": self.samples}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore fake progress."""
        if state.get("format") != "fake_sampler_v1":
            raise ValueError("bad fake sampler state")
        self.samples = int(state["samples"])


class _FakeTrainer:
    """Trainer double that emits complete IQL metrics and checkpoints."""

    def __init__(self, *, fail_at_step: int | None = None) -> None:
        """Initialize update count and optional injected failure."""
        self.global_step = 0
        self.fail_at_step = fail_at_step

    def step(self, transitions: object) -> IQLStepMetrics:
        """Return deterministic metrics and advance global progress."""
        del transitions
        if self.fail_at_step == self.global_step:
            raise RuntimeError("injected trainer failure")
        step = self.global_step
        actor = None
        if step % 2 == 1:
            actor = ActorStepMetrics(
                loss=0.3 + step,
                log_prob_mean=-1.0,
                weight_mean=1.2,
                weight_max=2.0,
                effective_sample_size=1.8,
                clipped_fraction=0.1,
                grad_norm=0.4,
            )
        metrics = IQLStepMetrics(
            global_step=step,
            value_loss=0.1 + step,
            q_loss=0.2 + step,
            q1_loss=0.21 + step,
            q2_loss=0.22 + step,
            value_grad_norm=0.5,
            q_grad_norm=0.6,
            q1_mean=0.7,
            q2_mean=0.8,
            value_mean=0.9,
            td_target_mean=1.0,
            td_target_std=1.1,
            td_target_min=-1.0,
            td_target_max=2.0,
            advantage_mean=0.2,
            advantage_std=0.3,
            q_disagreement_mean=0.05,
            actor_updated=actor is not None,
            actor=actor,
        )
        self.global_step += 1
        return metrics

    def state_dict(self) -> dict[str, Any]:
        """Serialize fake progress."""
        return {"format": "fake_trainer_v1", "global_step": self.global_step}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore fake progress."""
        if state.get("format") != "fake_trainer_v1":
            raise ValueError("bad fake trainer state")
        self.global_step = int(state["global_step"])

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        epoch: int,
        sampler_state: dict[str, Any],
        extra: dict[str, Any] | None = None,
    ) -> Path:
        """Write the same outer payload contract as the real trainer."""
        output_path = Path(path)
        torch.save(
            {
                "trainer": self.state_dict(),
                "sampler": sampler_state,
                "epoch": epoch,
                "extra": extra or {},
            },
            output_path,
        )
        return output_path


def _read_csv(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    """Read a CSV header and all rows."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return tuple(reader.fieldnames or ()), list(reader)


def test_runner_records_every_step_epoch_and_checkpoint(tmp_path: Path) -> None:
    """A two-epoch run should leave complete local evidence."""
    recorder = LocalIQLRecorder(tmp_path, rank=0)
    trainer = _FakeTrainer()
    sampler = _FakeSampler()
    runner = OfflineIQLRunner(
        trainer=trainer,  # type: ignore[arg-type]
        sampler=sampler,  # type: ignore[arg-type]
        recorder=recorder,
        config=IQLRunnerConfig(
            epochs=2,
            steps_per_epoch=3,
            batch_size=2,
            log_every_steps=2,
        ),
    )
    try:
        summaries = runner.run()
    finally:
        recorder.close()

    step_header, step_rows = _read_csv(tmp_path / "metrics" / "metrics.csv")
    epoch_header, epoch_rows = _read_csv(tmp_path / "metrics" / "epochs.csv")
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    checkpoints = sorted((tmp_path / "checkpoints").glob("*.pt"))

    assert len(summaries) == 2
    assert step_header == STEP_METRIC_FIELDS
    assert epoch_header == EPOCH_METRIC_FIELDS
    assert len(step_rows) == 6
    assert [int(row["global_step"]) for row in step_rows] == list(range(6))
    assert step_rows[0]["actor_updated"] == "0"
    assert step_rows[0]["actor_loss"] == ""
    assert step_rows[1]["actor_updated"] == "1"
    assert float(step_rows[1]["actor_loss"]) == pytest.approx(1.3)
    assert len(epoch_rows) == 2
    assert len(checkpoints) == 2
    assert status["status"] == "completed"
    assert status["epoch"] == 1
    assert status["global_step"] == 6
    assert (tmp_path / "train.log").is_file()


def test_runner_resume_aligns_uncheckpointed_csv_tail(tmp_path: Path) -> None:
    """Resume should remove an uncheckpointed tail and preserve exact counts."""
    first_recorder = LocalIQLRecorder(tmp_path, rank=0)
    first_runner = OfflineIQLRunner(
        trainer=_FakeTrainer(),  # type: ignore[arg-type]
        sampler=_FakeSampler(),  # type: ignore[arg-type]
        recorder=first_recorder,
        config=IQLRunnerConfig(
            epochs=1,
            steps_per_epoch=2,
            batch_size=2,
        ),
    )
    try:
        first_runner.run()
    finally:
        first_recorder.close()

    step_path = tmp_path / "metrics" / "metrics.csv"
    _, rows = _read_csv(step_path)
    uncheckpointed = dict(rows[-1])
    uncheckpointed["epoch"] = "1"
    uncheckpointed["epoch_step"] = "0"
    uncheckpointed["global_step"] = "2"
    with step_path.open("a", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=STEP_METRIC_FIELDS).writerow(
            uncheckpointed
        )

    checkpoint = tmp_path / "checkpoints" / "epoch_0000_rank_0000.pt"
    resumed_recorder = LocalIQLRecorder(tmp_path, rank=0)
    resumed_trainer = _FakeTrainer()
    resumed_sampler = _FakeSampler()
    resumed_runner = OfflineIQLRunner(
        trainer=resumed_trainer,  # type: ignore[arg-type]
        sampler=resumed_sampler,  # type: ignore[arg-type]
        recorder=resumed_recorder,
        config=IQLRunnerConfig(
            epochs=3,
            steps_per_epoch=2,
            batch_size=2,
        ),
    )
    try:
        summaries = resumed_runner.run(resume_checkpoint=checkpoint)
    finally:
        resumed_recorder.close()

    _, final_steps = _read_csv(step_path)
    _, final_epochs = _read_csv(tmp_path / "metrics" / "epochs.csv")
    assert [summary.epoch for summary in summaries] == [1, 2]
    assert [int(row["global_step"]) for row in final_steps] == list(range(6))
    assert [int(row["epoch"]) for row in final_epochs] == [0, 1, 2]
    assert resumed_trainer.global_step == 6
    assert resumed_sampler.samples == 12


def test_runner_failure_writes_failed_status(tmp_path: Path) -> None:
    """An update exception should be logged and reflected in status JSON."""
    recorder = LocalIQLRecorder(tmp_path, rank=0)
    runner = OfflineIQLRunner(
        trainer=_FakeTrainer(fail_at_step=1),  # type: ignore[arg-type]
        sampler=_FakeSampler(),  # type: ignore[arg-type]
        recorder=recorder,
        config=IQLRunnerConfig(
            epochs=1,
            steps_per_epoch=2,
            batch_size=2,
        ),
    )
    try:
        with pytest.raises(RuntimeError, match="injected trainer failure"):
            runner.run()
    finally:
        recorder.close()

    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert status["epoch"] == 0
    assert status["global_step"] == 1
    assert "injected trainer failure" in status["error"]


def test_runner_config_requires_every_epoch_checkpoint() -> None:
    """Formal offline IQL cannot skip completed-epoch checkpoints."""
    with pytest.raises(ValueError, match="checkpoint_every_epochs == 1"):
        IQLRunnerConfig(
            epochs=2,
            steps_per_epoch=3,
            batch_size=4,
            checkpoint_every_epochs=2,
        )


def test_fresh_run_rejects_existing_evidence_without_overwriting_status(
    tmp_path: Path,
) -> None:
    """Fresh mode must not mutate an already completed run directory."""
    status_path = tmp_path / "status.json"
    status_path.write_text('{"status": "completed"}\n', encoding="utf-8")
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    (metrics_dir / "metrics.csv").write_text(
        ",".join(STEP_METRIC_FIELDS) + "\n"
        + ",".join(["0"] * len(STEP_METRIC_FIELDS))
        + "\n",
        encoding="utf-8",
    )
    recorder = LocalIQLRecorder(tmp_path, rank=0)
    runner = OfflineIQLRunner(
        trainer=_FakeTrainer(),  # type: ignore[arg-type]
        sampler=_FakeSampler(),  # type: ignore[arg-type]
        recorder=recorder,
        config=IQLRunnerConfig(
            epochs=1,
            steps_per_epoch=1,
            batch_size=2,
        ),
    )
    try:
        with pytest.raises(FileExistsError, match="training evidence"):
            runner.run()
    finally:
        recorder.close()

    assert json.loads(status_path.read_text(encoding="utf-8")) == {
        "status": "completed"
    }
    assert not (tmp_path / "train.log").exists()


def test_resume_rejects_changed_epoch_work_shape(tmp_path: Path) -> None:
    """Resume cannot silently change the saved steps-per-epoch contract."""
    first_recorder = LocalIQLRecorder(tmp_path, rank=0)
    first_runner = OfflineIQLRunner(
        trainer=_FakeTrainer(),  # type: ignore[arg-type]
        sampler=_FakeSampler(),  # type: ignore[arg-type]
        recorder=first_recorder,
        config=IQLRunnerConfig(
            epochs=1,
            steps_per_epoch=2,
            batch_size=2,
        ),
    )
    try:
        first_runner.run()
    finally:
        first_recorder.close()

    checkpoint = tmp_path / "checkpoints" / "epoch_0000_rank_0000.pt"
    recorder = LocalIQLRecorder(tmp_path, rank=0)
    changed_runner = OfflineIQLRunner(
        trainer=_FakeTrainer(),  # type: ignore[arg-type]
        sampler=_FakeSampler(),  # type: ignore[arg-type]
        recorder=recorder,
        config=IQLRunnerConfig(
            epochs=2,
            steps_per_epoch=3,
            batch_size=2,
        ),
    )
    try:
        with pytest.raises(ValueError, match="steps_per_epoch"):
            changed_runner.run(resume_checkpoint=checkpoint)
    finally:
        recorder.close()
