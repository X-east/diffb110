"""MetricsWriter and ConsoleReporter: training metrics persistence.

Responsibility:
    - MetricsWriter: write epoch summary CSV, per-epoch metrics JSON,
      epoch status JSON, checkpoint status JSON.
    - ConsoleReporter: thin wrapper over logger for console-tagged messages.

Separated from run_recorder.py so that RunRecorder is a pure facade.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from ddpo.runtime.recording.run_context import RunContext
from ddpo.runtime.recording.schemas import EpochFacts


class ConsoleReporter:
    """Thin wrapper that tags log messages as console-visible."""

    def __init__(self, logger) -> None:
        self.logger = logger

    def info(self, message: str) -> None:
        self.logger.info(message, extra={"is_console": True})


class MetricsWriter:
    """Write epoch-level training metrics to CSV and JSON."""

    SUMMARY_HEADER = [
        "epoch", "timestamp", "avg_loss",
        "generated_total", "rdkit_valid", "rdkit_valid_ratio",
        "reward_mean", "reward_min", "reward_max",
        "avg_qed", "avg_sa", "avg_docking", "avg_similarity", "avg_diversity",
        "avg_qed_norm", "avg_sa_norm", "avg_docking_norm", "avg_diversity_norm",
        "filtering_ratio_mean", "novelty_penalty_ratio_mean",
        "failure_counts", "checkpoint_due", "checkpoint_path",
        "avg_kl", "avg_ppo_loss",
    ]

    def __init__(self, run_context: RunContext) -> None:
        self.run_context = run_context
        self.summary_path = Path(run_context.training_summary_csv_path)
        if not self.summary_path.exists():
            with open(self.summary_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.SUMMARY_HEADER)
                writer.writeheader()

    def _epoch_dir(self, epoch: int) -> Path:
        epoch_dir = Path(self.run_context.epochs_dir) / f"epoch_{epoch:04d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        return epoch_dir

    def _read_rows(self) -> list:
        with open(self.summary_path, "r", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def _write_rows(self, rows: list) -> None:
        with open(self.summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.SUMMARY_HEADER)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def write_epoch_summary(self, facts: EpochFacts, checkpoint_path: Optional[str] = None) -> None:
        row = {
            "epoch": int(facts.epoch),
            "timestamp": facts.timestamp,
            "avg_loss": f"{facts.avg_loss:.6f}",
            "generated_total": int(facts.generated_total),
            "rdkit_valid": int(facts.rdkit_valid),
            "rdkit_valid_ratio": f"{facts.rdkit_valid_ratio:.6f}",
            "reward_mean": f"{facts.reward_mean:.6f}",
            "reward_min": f"{facts.reward_min:.6f}",
            "reward_max": f"{facts.reward_max:.6f}",
            "avg_qed": f"{facts.avg_qed:.6f}",
            "avg_sa": f"{facts.avg_sa:.6f}",
            "avg_docking": f"{facts.avg_docking:.6f}",
            "avg_similarity": f"{facts.avg_similarity:.6f}",
            "avg_diversity": f"{facts.avg_diversity:.6f}",
            "avg_qed_norm": f"{facts.extra_metrics.get('avg_qed_norm', 0.0):.6f}",
            "avg_sa_norm": f"{facts.extra_metrics.get('avg_sa_norm', 0.0):.6f}",
            "avg_docking_norm": f"{facts.extra_metrics.get('avg_docking_norm', 0.0):.6f}",
            "avg_diversity_norm": f"{facts.extra_metrics.get('avg_diversity_norm', 0.0):.6f}",
            "filtering_ratio_mean": f"{facts.filtering_ratio_mean:.6f}",
            "novelty_penalty_ratio_mean": f"{facts.novelty_penalty_ratio_mean:.6f}",
            "failure_counts": json.dumps(facts.failure_counts, ensure_ascii=False, sort_keys=True),
            "checkpoint_due": int(facts.checkpoint_due),
            "checkpoint_path": checkpoint_path or "",
            "avg_kl": (
                f"{facts.extra_metrics.get('KL_Penalty', 0.0):.6f}"
                if "KL_Penalty" in facts.extra_metrics else ""
            ),
            "avg_ppo_loss": f"{facts.extra_metrics.get('PPO_Loss', 0.0):.6f}",
        }
        with open(self.summary_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.SUMMARY_HEADER)
            writer.writerow(row)

    def update_epoch_checkpoint_path(self, epoch: int, checkpoint_path: str) -> None:
        rows = self._read_rows()
        updated = [
            {**row, "checkpoint_path": checkpoint_path}
            if int(row.get("epoch", -1)) == epoch else row
            for row in rows
        ]
        self._write_rows(updated)

    def write_epoch_metrics_json(self, facts: EpochFacts) -> None:
        epoch_dir = self._epoch_dir(facts.epoch)
        metrics_path = epoch_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(asdict(facts), f, default=str)

    def write_epoch_status(
        self,
        facts: EpochFacts,
        molecules_path: Optional[str],
        checkpoint_path: Optional[str],
    ) -> None:
        epoch_dir = self._epoch_dir(facts.epoch)
        status_path = epoch_dir / "epoch_status.json"
        payload = {
            "epoch": facts.epoch,
            "timestamp": facts.timestamp,
            "generated_total": facts.generated_total,
            "rdkit_valid": facts.rdkit_valid,
            "rdkit_valid_ratio": facts.rdkit_valid_ratio,
            "avg_loss": facts.avg_loss,
            "reward_mean": facts.reward_mean,
            "reward_min": facts.reward_min,
            "reward_max": facts.reward_max,
            "avg_qed": facts.avg_qed,
            "avg_sa": facts.avg_sa,
            "avg_docking": facts.avg_docking,
            "avg_similarity": facts.avg_similarity,
            "avg_diversity": facts.avg_diversity,
            "filtering_ratio_mean": facts.filtering_ratio_mean,
            "novelty_penalty_ratio_mean": facts.novelty_penalty_ratio_mean,
            "failure_counts": facts.failure_counts,
            "checkpoint_due": facts.checkpoint_due,
            "extra_metrics": facts.extra_metrics,
            "molecules_csv": str(molecules_path),
            "checkpoint_path": str(checkpoint_path),
        }
        with open(status_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, default=str)

    def write_checkpoint_status(self, epoch: int, payload: dict) -> None:
        epoch_dir = self._epoch_dir(epoch)
        status_path = epoch_dir / "checkpoint_status.json"
        with open(status_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, default=str)
