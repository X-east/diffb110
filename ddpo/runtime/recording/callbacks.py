from __future__ import annotations

import os
from datetime import datetime

import pytorch_lightning as pl

from .schemas import CheckpointFacts


class RecordingCheckpointCallback(pl.Callback):
    """Finalize checkpoint recording on the recording mainline.

    This callback runs after the Lightning epoch lifecycle and converts the
    current `ModelCheckpoint` state into structured `CheckpointFacts` consumed
    by `RunRecorder`.
    """

    def __init__(self, run_recorder):
        super().__init__()
        self.run_recorder = run_recorder

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not trainer.is_global_zero:
            return

        current_epoch = int(trainer.current_epoch) + 1
        checkpoint_callback = getattr(trainer, "checkpoint_callback", None)
        ckpt_path = getattr(checkpoint_callback, "last_model_path", "") or ""
        monitor = getattr(checkpoint_callback, "monitor", None)

        ckpt_exists = bool(ckpt_path and os.path.exists(ckpt_path))
        ckpt_size = os.path.getsize(ckpt_path) if ckpt_exists else 0

        facts = CheckpointFacts(
            epoch=current_epoch,
            path=ckpt_path,
            kind="last",
            monitor=monitor,
            monitor_value=None,
            exists=ckpt_exists,
            size_bytes=ckpt_size,
            confirmed=ckpt_exists,
            saved_at=datetime.now().isoformat(timespec="seconds"),
        )
        self.run_recorder.on_checkpoint_saved(facts)
