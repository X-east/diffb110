"""Unit tests for offline-RL target-network updates."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from ddpo.offline_rl.target import polyak_update


class _BufferedLinear(nn.Module):
    """Small module carrying both parameters and non-trainable state."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1, bias=True)
        self.register_buffer("updates", torch.tensor(0, dtype=torch.long))


def test_polyak_update_interpolates_parameters_and_copies_buffers() -> None:
    """Target parameters should move by tau while buffers synchronize exactly."""
    source = _BufferedLinear()
    target = _BufferedLinear()
    with torch.no_grad():
        source.linear.weight.fill_(4.0)
        source.linear.bias.fill_(2.0)
        source.updates.fill_(7)
        target.linear.weight.fill_(0.0)
        target.linear.bias.fill_(-2.0)
        target.updates.fill_(1)

    polyak_update(source, target, tau=0.25)

    torch.testing.assert_close(target.linear.weight, torch.tensor([[1.0, 1.0]]))
    torch.testing.assert_close(target.linear.bias, torch.tensor([-1.0]))
    assert target.updates.item() == 7


def test_polyak_update_tau_one_is_hard_parameter_update() -> None:
    """Tau one should exactly initialize target parameters from the source."""
    source = nn.Linear(2, 1)
    target = nn.Linear(2, 1)
    with torch.no_grad():
        source.weight.fill_(3.0)
        source.bias.fill_(4.0)
        target.weight.zero_()
        target.bias.zero_()

    polyak_update(source, target, tau=1.0)

    assert torch.equal(target.weight, source.weight)
    assert torch.equal(target.bias, source.bias)


@pytest.mark.parametrize("tau", [-0.1, 1.1, float("nan")])
def test_polyak_update_rejects_invalid_tau(tau: float) -> None:
    """Polyak interpolation requires a finite closed-interval coefficient."""
    with pytest.raises(ValueError, match="tau"):
        polyak_update(nn.Linear(1, 1), nn.Linear(1, 1), tau=tau)


def test_polyak_update_rejects_different_module_states() -> None:
    """Different parameter structures must fail instead of being zipped."""
    with pytest.raises(ValueError, match="parameter names"):
        polyak_update(nn.Linear(2, 1), nn.Sequential(nn.Linear(2, 1)), tau=0.5)
