"""Regression tests for bug6: RunContext/RunRecorder must not enter Lightning hparams.

Bug6 root cause:
    save_hyperparameters() without ignoring run_context/run_recorder causes
    FrozenInstanceError on the frozen=True RunContext dataclass, leading to
    DeadlockDetectedException in DDP.
"""
from __future__ import annotations

import inspect
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
for extra in [
    str(PROJECT_ROOT),
    str(PROJECT_ROOT / "DiffSBDD"),
]:
    if extra not in sys.path:
        sys.path.insert(0, extra)

pytestmark = pytest.mark.skip(
    reason=(
        "Legacy Lightning mainline test. The active training entrypoint is "
        "ddpo/main_diffsbdd_ddp.py."
    )
)


def _make_minimal_module():
    from ddpo.trainer.lightning_ddpo_module import LightningDDPOModule
    config = {
        "train": {"epochs": 1, "batch_size": 2, "checkpoint_interval": 5},
        "reward": {"enabled": ["druglikeness"]},
    }
    return LightningDDPOModule(
        model=MagicMock(),
        rollout=MagicMock(),
        rewarder=MagicMock(),
        filters=MagicMock(),
        actor=MagicMock(),
        config=config,
    )


class TestBug6HparamsIsolation:

    def test_run_context_not_in_hparams(self):
        module = _make_minimal_module()
        assert "run_context" not in module.hparams, (
            "run_context must be excluded from hparams — frozen dataclass "
            "causes FrozenInstanceError in Lightning hparam serialization."
        )

    def test_run_recorder_not_in_hparams(self):
        module = _make_minimal_module()
        assert "run_recorder" not in module.hparams, (
            "run_recorder must be excluded from hparams — runtime object, not a hyperparameter."
        )

    def test_config_is_in_hparams(self):
        module = _make_minimal_module()
        assert "config" in module.hparams

    def test_save_hyperparameters_ignore_list_covers_runtime_objects(self):
        """Structural guard: ignore list must explicitly name run_context and run_recorder."""
        from ddpo.trainer.lightning_ddpo_module import LightningDDPOModule
        source = inspect.getsource(LightningDDPOModule.__init__)
        assert "run_context" in source, (
            "'run_context' must appear in save_hyperparameters ignore list"
        )
        assert "run_recorder" in source, (
            "'run_recorder' must appear in save_hyperparameters ignore list"
        )
