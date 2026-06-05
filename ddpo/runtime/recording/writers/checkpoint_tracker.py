"""CheckpointTracker: append-only checkpoint manifest writer.

Responsibility:
    Maintain a CSV manifest of all checkpoints saved during training.
    One row per checkpoint event (regular interval or best-model).
"""
from __future__ import annotations

import csv
from pathlib import Path

from ddpo.runtime.recording.run_context import RunContext
from ddpo.runtime.recording.schemas import CheckpointFacts


class CheckpointTracker:
    """Append-only writer for the checkpoint manifest CSV."""

    FIELDNAMES = [
        "epoch", "kind", "path", "exists", "size_bytes",
        "monitor", "monitor_value", "confirmed", "saved_at",
    ]

    def __init__(self, run_context: RunContext) -> None:
        self.run_context = run_context
        manifest_path = Path(run_context.checkpoint_manifest_path)
        if not manifest_path.exists():
            with open(manifest_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                writer.writeheader()

    def append(self, facts: CheckpointFacts) -> None:
        """Append one checkpoint record to the manifest."""
        with open(self.run_context.checkpoint_manifest_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writerow({
                "epoch": facts.epoch,
                "kind": facts.kind,
                "path": facts.path,
                "exists": int(facts.exists),
                "size_bytes": int(facts.size_bytes),
                "monitor": facts.monitor,
                "monitor_value": (
                    f"{facts.monitor_value:.6f}" if facts.monitor_value is not None else ""
                ),
                "confirmed": int(facts.confirmed),
                "saved_at": facts.saved_at,
            })
