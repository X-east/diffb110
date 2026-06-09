"""Timestep configuration helpers for the decoupled DiffGui runtime."""

from __future__ import annotations

import math
from typing import Any

import torch


TIMESTEP_DERIVED_STATE_PREFIXES = (
    "pos_transition.",
    "node_transition.",
    "edge_transition.",
    "time_emb.",
)


def configure_diffgui_model_timesteps(model: torch.nn.Module, num_timesteps: int) -> dict[str, Any]:
    """Regenerate DiffGui timestep-dependent buffers for a target trajectory length."""
    target_timesteps = int(num_timesteps)
    if target_timesteps <= 0:
        raise ValueError("num_timesteps must be positive.")
    original_timesteps = int(getattr(model, "num_timesteps", 0))
    if original_timesteps == target_timesteps:
        return {
            "changed": False,
            "num_timesteps": target_timesteps,
            "reason": "already_configured",
        }

    model_config = getattr(model, "config", None)
    diff_config = getattr(model_config, "diff", None)
    if diff_config is None:
        raise ValueError("DiffGui model.config.diff is required to auto-configure timesteps.")
    if not hasattr(model, "define_betas_alphas"):
        raise ValueError("DiffGui model must expose define_betas_alphas(config.diff).")

    device = infer_module_device(model)
    was_training = bool(model.training)
    bond_segments_before = get_bond_time_segments(diff_config)
    force_diff_config_num_timesteps(diff_config, target_timesteps)
    model.define_betas_alphas(diff_config)
    rebuilt_time_embedding = rebuild_time_embedding(model, target_timesteps, device=device)
    model.to(device)
    model.train(was_training)
    return {
        "changed": True,
        "from_num_timesteps": original_timesteps,
        "to_num_timesteps": target_timesteps,
        "bond_time_segment_before": bond_segments_before,
        "bond_time_segment_after": get_bond_time_segments(diff_config),
        "rebuilt_time_embedding": rebuilt_time_embedding,
    }


def load_diffgui_network_state(
    model: torch.nn.Module,
    checkpoint_state: dict[str, torch.Tensor],
    *,
    skip_timestep_derived_buffers: bool = True,
) -> dict[str, Any]:
    """Load checkpoint weights while preserving timestep-derived buffers from the active config."""
    model_state = model.state_dict()
    filtered: dict[str, torch.Tensor] = {}
    skipped: list[dict[str, Any]] = []
    for key, value in checkpoint_state.items():
        if skip_timestep_derived_buffers and key.startswith(TIMESTEP_DERIVED_STATE_PREFIXES):
            skipped.append(
                {
                    "key": key,
                    "checkpoint_shape": list(value.shape),
                    "model_shape": list(model_state[key].shape) if key in model_state else None,
                    "reason": "timestep_derived_buffer_regenerated_from_active_config",
                }
            )
            continue
        model_value = model_state.get(key)
        if model_value is not None and tuple(model_value.shape) == tuple(value.shape):
            filtered[key] = value
            continue
        skipped.append(
            {
                "key": key,
                "checkpoint_shape": None if not hasattr(value, "shape") else list(value.shape),
                "model_shape": None if model_value is None else list(model_value.shape),
                "reason": "shape_mismatch_or_missing_key",
            }
        )

    unexpected_skipped = [
        item["key"]
        for item in skipped
        if not str(item["key"]).startswith(TIMESTEP_DERIVED_STATE_PREFIXES)
    ]
    if unexpected_skipped:
        raise RuntimeError(f"Unexpected checkpoint shape mismatches: {unexpected_skipped[:10]}")
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    return {
        "loaded_keys": len(filtered),
        "skipped_timestep_derived_keys": sum(
            1 for item in skipped if str(item["key"]).startswith(TIMESTEP_DERIVED_STATE_PREFIXES)
        ),
        "skipped_total_keys": len(skipped),
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
    }


def force_diff_config_num_timesteps(diff_config: Any, num_timesteps: int) -> None:
    """Update DiffGui diffusion config and scale segmented bond schedule."""
    diff_config.num_timesteps = int(num_timesteps)
    diff_bond = getattr(diff_config, "diff_bond", None)
    if diff_bond is None or str(getattr(diff_bond, "beta_schedule", "")) != "segment":
        return
    time_segment = [int(value) for value in getattr(diff_bond, "time_segment")]
    diff_bond.time_segment = scale_segments(time_segment, int(num_timesteps))


def scale_segments(segments: list[int], total: int) -> list[int]:
    """Scale positive segment lengths to a new total while preserving proportions."""
    if total <= 0:
        raise ValueError("total must be positive.")
    if not segments or any(segment <= 0 for segment in segments):
        raise ValueError("segments must contain positive integers.")
    segment_sum = float(sum(segments))
    scaled = [float(segment) / segment_sum * float(total) for segment in segments]
    floored = [max(1, int(math.floor(value))) for value in scaled]
    while sum(floored) < total:
        deltas = [value - floored[idx] for idx, value in enumerate(scaled)]
        floored[int(max(range(len(deltas)), key=deltas.__getitem__))] += 1
    while sum(floored) > total:
        candidates = [idx for idx, value in enumerate(floored) if value > 1]
        if not candidates:
            break
        idx_to_reduce = max(candidates, key=lambda idx: floored[idx] - scaled[idx])
        floored[idx_to_reduce] -= 1
    return [int(value) for value in floored]


def rebuild_time_embedding(model: torch.nn.Module, num_timesteps: int, *, device: torch.device) -> bool:
    """Rebuild DiffGui Gaussian time embedding on the current model device."""
    time_emb = getattr(model, "time_emb", None)
    if time_emb is None or len(time_emb) == 0:
        return False
    smearing = time_emb[0]
    offset = getattr(smearing, "offset", None)
    if offset is None:
        return False
    gaussian_smearing_cls = smearing.__class__
    num_gaussians = int(offset.numel())
    rebuilt = gaussian_smearing_cls(stop=int(num_timesteps), num_gaussians=num_gaussians, type_="linear")
    model.time_emb = torch.nn.Sequential(rebuilt).to(device)
    return True


def infer_module_device(module: torch.nn.Module) -> torch.device:
    """Infer module device from parameters or buffers."""
    try:
        return next(module.parameters()).device
    except StopIteration:
        pass
    try:
        return next(module.buffers()).device
    except StopIteration:
        return torch.device("cpu")


def get_bond_time_segments(diff_config: Any) -> list[int] | None:
    """Return bond segmented schedule lengths when present."""
    diff_bond = getattr(diff_config, "diff_bond", None)
    if diff_bond is None or not hasattr(diff_bond, "time_segment"):
        return None
    return [int(value) for value in getattr(diff_bond, "time_segment")]
