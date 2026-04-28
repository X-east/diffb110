"""RunRecorder: facade coordinating all recording subsystems for one training run.

This module is intentionally thin — all persistence logic lives in writers/:
    - MetricsWriter:      epoch summary CSV + per-epoch JSON
    - MoleculeWriter:     per-rank molecule CSV + epoch merge
    - CheckpointTracker:  checkpoint manifest CSV
    - ConsoleReporter:    logger wrapper for console-tagged messages
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from .run_context import RunContext
from .schemas import BatchFacts, CheckpointFacts, EpochFacts, RunLifecycleFacts, RunStatus
from .writers import CheckpointTracker, ConsoleReporter, MetricsWriter, MoleculeWriter


class RunRecorder:
    """Facade coordinating all recording subsystems for one training run.

    Owns no persistence logic — delegates entirely to the writer classes.
    Callers should interact only with the on_* event methods.
    """

    def __init__(self, run_context: RunContext, logger, world_size: int = 1) -> None:
        self.run_context = run_context
        self.logger = logger
        self.world_size = int(world_size)
        self.console = ConsoleReporter(logger)
        self.metrics = MetricsWriter(run_context)
        self.molecules = MoleculeWriter(run_context)
        self.checkpoints = CheckpointTracker(run_context)
        self.last_checkpoint_path: Optional[str] = None
        self.last_epoch_facts: Optional[EpochFacts] = None

    # ------------------------------------------------------------------
    # Run lifecycle events
    # ------------------------------------------------------------------

    def write_run_manifest(self, payload: dict) -> None:
        with open(self.run_context.run_manifest_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    def on_run_initialized(self, payload: dict) -> None:
        self.write_run_manifest(payload)

    def on_run_started(self) -> None:
        manifest = {
            "run_id": self.run_context.run_id,
            "run_name": self.run_context.run_name,
            "output_dir": self.run_context.output_dir,
            "status": RunStatus.RUNNING.value,
            "started_at": self.run_context.started_at,
        }
        self.write_run_manifest(manifest)
        for line in [
            f"Run ID    : {self.run_context.run_id}",
            f"Run Name  : {self.run_context.run_name}",
            f"Output Dir: {self.run_context.output_dir}",
        ]:
            self.console.info(line)

    def on_run_finished(self, lifecycle: RunLifecycleFacts, extra_lines: List[str] = None) -> None:
        payload = {
            "run_id": lifecycle.run_id,
            "run_name": lifecycle.run_name,
            "output_dir": lifecycle.output_dir,
            "status": lifecycle.status.value,
            "started_at": lifecycle.started_at,
            "ended_at": lifecycle.ended_at,
            "message": lifecycle.message,
        }
        self.write_run_manifest(payload)
        title = f"Training {lifecycle.status.value.capitalize()}"
        self.console.info("=" * 70)
        self.console.info(title)
        if extra_lines:
            for line in extra_lines:
                self.console.info(line)
        self.console.info("=" * 70)

    # ------------------------------------------------------------------
    # Training events
    # ------------------------------------------------------------------

    def on_batch_end(self, facts: BatchFacts) -> None:
        self.molecules.write_rank_batch(facts)

    def on_epoch_end(self, facts: EpochFacts) -> None:
        merged_path = self.molecules.merge_epoch(facts.epoch, self.world_size)
        self.metrics.write_epoch_metrics_json(facts)
        self.metrics.write_epoch_summary(facts, self.last_checkpoint_path)
        self.metrics.write_epoch_status(facts, merged_path, self.last_checkpoint_path)
        self.last_epoch_facts = facts

        max_epochs = facts.extra_metrics.get("max_epochs", "?")
        self.console.info(
            f"[Epoch {facts.epoch}/{max_epochs}]\n"
            f"  Generated Total  : {facts.generated_total}\n"
            f"  RDKit Validity   : {facts.rdkit_valid} / {facts.generated_total} "
            f"({facts.rdkit_valid_ratio:.2%})\n"
            f"  Loss             : {facts.avg_loss: .4f}\n"
            f"  Reward           : {facts.reward_mean: .4f}\n"
            f"  Avg QED          : {facts.avg_qed: .4f}\n"
            f"  Avg SA           : {facts.avg_sa: .4f}\n"
            f"  Avg Docking      : {facts.avg_docking: .4f}"
        )

    def on_checkpoint_saved(self, facts: CheckpointFacts) -> None:
        self.checkpoints.append(facts)
        self.last_checkpoint_path = facts.path
        epoch = facts.epoch
        checkpoint_status = {
            "generated_total": self.last_epoch_facts.generated_total if self.last_epoch_facts else 0,
            "rdkit_valid": self.last_epoch_facts.rdkit_valid if self.last_epoch_facts else 0,
            "reward_mean": self.last_epoch_facts.reward_mean if self.last_epoch_facts else 0.0,
        }
        molecules_path = (
            Path(self.run_context.epochs_dir) / f"epoch_{epoch:04d}" / "molecules.csv"
        )
        self.metrics.write_checkpoint_status(epoch, checkpoint_status)
        if self.last_epoch_facts:
            self.metrics.update_epoch_checkpoint_path(epoch, facts.path)
            self.metrics.write_epoch_status(
                self.last_epoch_facts,
                str(molecules_path) if molecules_path.exists() else None,
                facts.path,
            )
        size_mb = facts.size_bytes / (1024 * 1024)
        if facts.confirmed:
            self.console.info(
                f"[CHECKPOINT] Epoch {epoch}: saved {Path(facts.path).name} ({size_mb:.1f}MB)"
            )
        else:
            self.console.info(f"[CHECKPOINT] Epoch {epoch}: not confirmed")
