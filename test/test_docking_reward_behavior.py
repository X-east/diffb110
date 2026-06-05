"""Regression coverage for DockingReward reward/failure writeback semantics."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from ddpo.rewards.docking import DockingReward
from ddpo.rewards.docking_backends import DockingResult


class _BackendBatchStub:
    support_level = "supported"

    def dock_batch_results(self, ligand_files, receptor_file, **kwargs):
        return [
            DockingResult(None, "autodock_gpu", True, -8.0, None),
            DockingResult(None, "autodock_gpu", False, None, "timeout"),
        ]


class _BackendSingleStub:
    support_level = "supported"

    def __init__(self):
        self.calls = 0

    def dock_single_result(self, ligand_file, receptor_file, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return DockingResult(None, "smina", True, -6.0, None)
        return DockingResult(None, "smina", False, None, "score_parse_failed")


class _PreparedBatch:
    def __init__(self):
        self.ligand_files = ["a.sdf", "b.sdf"]
        self.sample_indices = [0, 1]
        self.failures = {}
        self.cleaned = False

    def cleanup(self):
        self.cleaned = True


class _PreparedSingle:
    def __init__(self, path, failure_type=None):
        self.path = path
        self.failure_type = failure_type
        self.cleaned = False

    def cleanup(self):
        self.cleaned = True


def _make_data(batch_size: int = 2):
    return SimpleNamespace(
        batch={
            "x": torch.zeros(batch_size, 1, 3),
            "categorical": torch.zeros(batch_size, 1, 1),
            "nodesxsample": torch.ones(batch_size, dtype=torch.long),
            "rdkit_valid": [1] * batch_size,
            "failure_type": ["success"] * batch_size,
        }
    )


def _make_reward(backend_impl, execution_mode: str, tmp_path: Path) -> DockingReward:
    receptor = tmp_path / "receptor.pdbqt"
    receptor.write_text("receptor\n", encoding="utf-8")
    return DockingReward(
        backend_impl=backend_impl,
        backend_name=getattr(backend_impl, "backend_name", "stub"),
        execution_mode=execution_mode,
        receptor_file=str(receptor),
        normalization_factor=-10.0,
        failure_penalty=-1.0,
        dataset_info={},
    )


def test_docking_reward_batch_mode_writes_rewards_and_failure_reason(monkeypatch, tmp_path):
    prepared = _PreparedBatch()
    monkeypatch.setattr(
        "ddpo.rewards.docking.prepare_batch_sdf_inputs",
        lambda **kwargs: prepared,
    )

    reward = _make_reward(_BackendBatchStub(), "batch", tmp_path)
    data = _make_data(batch_size=2)

    out = reward.calculate_rewards(data)

    assert torch.allclose(out.batch["rewards"], torch.tensor([0.8, -1.0]))
    assert out.batch["failure_type"][0] == "success"
    assert out.batch["failure_type"][1] == "docking_backend:timeout"
    assert prepared.cleaned is True



def test_docking_reward_sequential_mode_respects_rdkit_gate_and_failure_writeback(monkeypatch, tmp_path):
    prepared_items = [
        _PreparedSingle("/tmp/mol0.sdf"),
        _PreparedSingle(None, "docking_input_failed"),
    ]

    def fake_prepare_single_sdf_input(**kwargs):
        sample_idx = kwargs["sample_idx"]
        return prepared_items[sample_idx]

    monkeypatch.setattr(
        "ddpo.rewards.docking.prepare_single_sdf_input",
        fake_prepare_single_sdf_input,
    )

    backend = _BackendSingleStub()
    reward = _make_reward(backend, "sequential", tmp_path)
    data = _make_data(batch_size=3)
    data.batch["rdkit_valid"] = [1, 1, 0]
    data.batch["failure_type"] = ["success", "success", "success"]

    out = reward.calculate_rewards(data)

    assert torch.allclose(out.batch["rewards"], torch.tensor([0.6, -1.0, -1.0]))
    assert out.batch["failure_type"][0] == "success"
    assert out.batch["failure_type"][1] == "docking_input_failed"
    assert out.batch["failure_type"][2] == "success"
    assert prepared_items[0].cleaned is True
    assert prepared_items[1].cleaned is True
