"""Tests for DiffSBDD DDPO transition log-probability aggregation modes."""

from __future__ import annotations

import math
import inspect

import pytest
import torch

from ddpo.integration.diffsbdd_model import DEFAULT_LOGPROB_MODE, DiffSBDDModel
from ddpo.runtime import build as runtime_build


def _make_model(logprob_mode: str) -> DiffSBDDModel:
    """Create a minimal DiffSBDDModel instance for pure logprob tests."""
    model = DiffSBDDModel.__new__(DiffSBDDModel)
    model.logprob_mode = logprob_mode
    model.logprob_clamp_min = -100.0
    model.logprob_clamp_max = 0.0
    return model


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("mean_per_element", -0.5),
        ("mean_per_atom", -1.5),
        ("full_mean_per_element", -0.5 - 0.5 * math.log(2.0 * math.pi)),
        ("full_mean_per_atom", 3.0 * (-0.5 - 0.5 * math.log(2.0 * math.pi))),
    ],
)
def test_logprob_modes_have_expected_scale(mode: str, expected: float) -> None:
    """Each mode should aggregate the same Gaussian terms at the intended scale."""
    model = _make_model(mode)
    zs = torch.ones(1, 2, 3)
    mu = torch.zeros_like(zs)
    sigma = torch.ones(1, 1)
    node_mask = torch.ones(1, 2, 1)

    log_prob = model.compute_log_p_zs_given_zt(zs, mu, sigma, node_mask)

    assert log_prob.shape == (1,)
    assert log_prob.item() == pytest.approx(expected)


def test_batch_sigma_broadcast_and_padding_mask() -> None:
    """[B, 1] sigma should broadcast correctly with padded node masks."""
    model = _make_model("mean_per_atom")
    zs = torch.ones(2, 3, 2)
    mu = torch.zeros_like(zs)
    sigma = torch.ones(2, 1)
    node_mask = torch.tensor(
        [
            [[1.0], [1.0], [0.0]],
            [[1.0], [0.0], [0.0]],
        ]
    )

    log_prob = model.compute_log_p_zs_given_zt(zs, mu, sigma, node_mask)

    assert log_prob.shape == (2,)
    assert log_prob.tolist() == pytest.approx([-1.0, -1.0])


def test_mean_per_atom_is_default_logprob_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Implicit model configs should use atom-normalized transition logprobs."""
    signature = inspect.signature(DiffSBDDModel)
    assert signature.parameters["logprob_mode"].default == DEFAULT_LOGPROB_MODE
    assert DEFAULT_LOGPROB_MODE == "mean_per_atom"

    captured_kwargs = {}

    def _fake_model(**kwargs):
        captured_kwargs.update(kwargs)
        return object()

    monkeypatch.setattr(runtime_build, "DiffSBDDModel", _fake_model)
    config = {
        "model": {"checkpoint": "checkpoint.ckpt"},
        "pocket": {
            "pdb_path": "protein.pdb",
            "ref_ligand": "ligand.sdf",
        },
    }

    runtime_build.build_model(config, device="cuda", dataset_name="crossdock")

    assert captured_kwargs["logprob_mode"] == DEFAULT_LOGPROB_MODE
