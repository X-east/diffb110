"""RunRecorder: facade coordinating all recording subsystems for one training run.

This module is intentionally thin — all persistence logic lives in writers/:
    - MetricsWriter:      epoch summary CSV + per-epoch JSON
    - MoleculeWriter:     per-rank molecule CSV + epoch merge
    - CheckpointTracker:  checkpoint manifest CSV
    - ConsoleReporter:    logger wrapper for console-tagged messages

Reward display is fully config-driven: the terminal output for each sub-reward
is derived from ``extra_metrics["reward_enabled"]`` (populated by
:func:`build_epoch_facts`) and the ``_REWARD_DISPLAY_REGISTRY`` mapping below.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from .run_context import RunContext
from .schemas import BatchFacts, CheckpointFacts, EpochFacts, RunLifecycleFacts, RunStatus
from .writers import CheckpointTracker, ConsoleReporter, MetricsWriter, MoleculeWriter

# ---------------------------------------------------------------------------
# Reward display registry
#   key       = name in reward.weights / reward_enabled
#   label     = terminal column header
#   field     = attribute name on EpochFacts
#   fmt       = format specifier for f-string
#
# Adding a new reward component requires only one new entry here.
# ---------------------------------------------------------------------------
_REWARD_DISPLAY_REGISTRY: dict[str, dict[str, str]] = {
    "qed":         {"label": "Avg QED",         "field": "avg_qed",           "fmt": ".4f"},
    "sa":          {"label": "Avg SA",          "field": "avg_sa",            "fmt": ".4f"},
    "docking":     {"label": "Avg Docking",     "field": "avg_docking",       "fmt": ".4f"},
    "chembl_prior": {"label": "ChEMBL Sim TopK", "field": "avg_chembl_similarity", "fmt": ".4f"},
    "chembl_prior_ecfp4": {"label": "ECFP4 Sim", "field": "avg_chembl_prior_ecfp4_similarity", "fmt": ".4f"},
    "chembl_prior_fcfp4": {"label": "FCFP4 Sim", "field": "avg_chembl_prior_fcfp4_similarity", "fmt": ".4f"},
    "diversity_ecfp4": {"label": "ECFP4 Div",   "field": "avg_diversity_ecfp4_diversity", "fmt": ".4f"},
    "diversity_fcfp4": {"label": "FCFP4 Div",   "field": "avg_diversity_fcfp4_diversity", "fmt": ".4f"},
}


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
        resume_info = getattr(self, "resume_info", None)
        if resume_info:
            manifest["resume"] = resume_info
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
        resume_info = getattr(self, "resume_info", None)
        if resume_info:
            payload["resume"] = resume_info
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

    def on_batch_sdf(self, facts: BatchFacts, rdmols: list) -> None:
        self.molecules.write_rank_sdf(facts, rdmols)

    def on_epoch_end(self, facts: EpochFacts) -> None:
        merged_path = self.molecules.merge_epoch(facts.epoch, self.world_size)
        self.metrics.write_epoch_metrics_json(facts)
        self.metrics.write_epoch_summary(facts, self.last_checkpoint_path)
        self.metrics.write_epoch_status(facts, merged_path, self.last_checkpoint_path)
        self.last_epoch_facts = facts

        max_epochs = facts.extra_metrics.get("max_epochs", "?")
        avg_ppo_loss = facts.extra_metrics.get("PPO_Loss", 0.0)
        # 使用原始总奖励（归一化前）展示给用户，而非 z-score 后恒 ≈0 的值
        avg_reward_raw = facts.extra_metrics.get(
            "avg_reward_raw", facts.reward_mean
        )

        lines = [
            f"[Epoch {facts.epoch}/{max_epochs}]",
            f"  Generated Total  : {facts.generated_total}",
            f"  RDKit Validity   : {facts.rdkit_valid} / {facts.generated_total} "
            f"({facts.rdkit_valid_ratio:.2%})",
            f"  Loss             : {facts.avg_loss: .4f}",
            f"  PPO Loss         : {avg_ppo_loss: .4f}",
            f"  Reward           : {avg_reward_raw: .4f}",
        ]

        # Dynamic: KL line appears only when Actor produced KL_Penalty
        if "KL_Penalty" in facts.extra_metrics:
            lines.insert(-1, f"  KL Penalty       : {facts.extra_metrics['KL_Penalty']: .6f}")

        # 从 extra_metrics 读取配置驱动的奖励列表，动态构建终端行
        reward_enabled = facts.extra_metrics.get(
            "reward_enabled", ["qed", "sa"]
        )
        for reward_name in reward_enabled:
            entry = _REWARD_DISPLAY_REGISTRY.get(reward_name)
            if entry is None and str(reward_name).startswith("chembl_prior_"):
                entry = {
                    "label": str(reward_name).replace("chembl_prior_", "").upper() + " Sim",
                    "field": f"avg_{reward_name}_similarity",
                    "fmt": ".4f",
                }
            if entry is None:
                continue
            value = getattr(facts, entry["field"], None)
            if value is None:
                value = facts.extra_metrics.get(entry["field"])
            if value is None:
                continue
            lines.append(f"  {entry['label']:15s}: {value:{entry['fmt']}}")

        # ── Timing summary ──
        timing_keys = sorted(
            k for k in facts.extra_metrics if k.startswith("time_")
        )
        if timing_keys:
            lines.append(f"  ── Timings (sec) ──")
            for key in timing_keys:
                step_name = key.removeprefix("time_")
                value = float(facts.extra_metrics[key])
                lines.append(f"  {step_name:25s}: {value:7.3f}")
            # Total from the sum of all time_* entries
            total = sum(float(facts.extra_metrics[k]) for k in timing_keys)
            lines.append(f"  {'─' * 36}")
            lines.append(f"  {'epoch_total':25s}: {total:7.3f}")

        print("\n".join(lines), flush=True)

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
