"""Target-network updates for decoupled offline reinforcement learning."""

from __future__ import annotations

import math

import torch
from torch import nn


def _named_parameters(module: nn.Module) -> dict[str, nn.Parameter]:
    """Return a stable name-to-parameter mapping."""
    return dict(module.named_parameters())


def _named_buffers(module: nn.Module) -> dict[str, torch.Tensor]:
    """Return a stable name-to-buffer mapping."""
    return dict(module.named_buffers())


def _require_matching_state(source: nn.Module, target: nn.Module) -> None:
    """Ensure an update cannot silently zip unrelated module states."""
    source_parameters = _named_parameters(source)
    target_parameters = _named_parameters(target)
    if source_parameters.keys() != target_parameters.keys():
        raise ValueError("source and target parameter names do not match")

    for name, source_parameter in source_parameters.items():
        target_parameter = target_parameters[name]
        if source_parameter.shape != target_parameter.shape:
            raise ValueError(
                f"parameter {name!r} shape mismatch: "
                f"source={tuple(source_parameter.shape)} "
                f"target={tuple(target_parameter.shape)}"
            )
        if source_parameter.device != target_parameter.device:
            raise ValueError(
                f"parameter {name!r} device mismatch: "
                f"source={source_parameter.device} target={target_parameter.device}"
            )
        if source_parameter.dtype != target_parameter.dtype:
            raise ValueError(
                f"parameter {name!r} dtype mismatch: "
                f"source={source_parameter.dtype} target={target_parameter.dtype}"
            )

    source_buffers = _named_buffers(source)
    target_buffers = _named_buffers(target)
    if source_buffers.keys() != target_buffers.keys():
        raise ValueError("source and target buffer names do not match")
    for name, source_buffer in source_buffers.items():
        target_buffer = target_buffers[name]
        if source_buffer.shape != target_buffer.shape:
            raise ValueError(
                f"buffer {name!r} shape mismatch: "
                f"source={tuple(source_buffer.shape)} target={tuple(target_buffer.shape)}"
            )
        if source_buffer.device != target_buffer.device:
            raise ValueError(
                f"buffer {name!r} device mismatch: "
                f"source={source_buffer.device} target={target_buffer.device}"
            )
        if source_buffer.dtype != target_buffer.dtype:
            raise ValueError(
                f"buffer {name!r} dtype mismatch: "
                f"source={source_buffer.dtype} target={target_buffer.dtype}"
            )


def polyak_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    """Update target parameters toward source parameters in place.

    The update is ``target = (1 - tau) * target + tau * source``. Floating
    parameters are interpolated and all buffers are copied exactly from the
    source, which keeps counters and other non-trainable state synchronized.

    Args:
        source: Online module providing the newest parameters.
        target: Target module to update in place.
        tau: Interpolation coefficient in ``[0, 1]``. ``tau=1`` is a hard
            update and ``tau=0`` only synchronizes buffers.

    Raises:
        ValueError: If ``tau`` is invalid or module states are incompatible.
    """
    if not math.isfinite(tau) or not 0.0 <= tau <= 1.0:
        raise ValueError(f"tau must be finite and in [0, 1], got {tau!r}")
    _require_matching_state(source, target)

    source_parameters = _named_parameters(source)
    target_parameters = _named_parameters(target)
    source_buffers = _named_buffers(source)
    target_buffers = _named_buffers(target)

    with torch.no_grad():
        for name, source_parameter in source_parameters.items():
            target_parameters[name].lerp_(source_parameter, tau)
        for name, source_buffer in source_buffers.items():
            target_buffers[name].copy_(source_buffer)
