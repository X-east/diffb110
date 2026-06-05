"""Runtime-only MolSnapper timestep override.

This mirrors the DiffGui benchmark patch style and leaves upstream source files
unchanged. Set MOLSNAPPER_NUM_TIMESTEPS=500 before importing model modules.
"""

from __future__ import annotations

import os
import sys


def _get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _set(obj, key, value):
    if isinstance(obj, dict):
        obj[key] = value
    else:
        setattr(obj, key, value)


def _scale_segments(old_segments, total):
    old_segments = list(old_segments or [])
    if not old_segments:
        return [total]
    old_sum = sum(int(item) for item in old_segments)
    if old_sum <= 0:
        return [total]
    new_segments = []
    remaining = total
    for index, old_value in enumerate(old_segments):
        slots_left = len(old_segments) - index
        if index == len(old_segments) - 1:
            value = remaining
        else:
            value = int(round(total * int(old_value) / old_sum))
            value = max(1, min(value, remaining - (slots_left - 1)))
        new_segments.append(value)
        remaining -= value
    return new_segments


def _patch_diff_config(diff_config, steps):
    _set(diff_config, "num_timesteps", int(steps))
    section = _get(diff_config, "diff_bond")
    if section is not None and _get(section, "beta_schedule") == "segment":
        _set(section, "time_segment", _scale_segments(_get(section, "time_segment", []), int(steps)))


def _apply_steps(module, steps):
    class_name = module.__class__.__name__
    if class_name not in {"MolDiff", "BondPredictor"}:
        return
    diff_config = module.config.diff
    _patch_diff_config(diff_config, int(steps))
    module.define_betas_alphas(diff_config)
    if class_name == "MolDiff":
        from models.common import GaussianSmearing
        import torch.nn as nn

        time_dim = int(_get(diff_config, "time_dim", 10))
        module.time_emb = nn.Sequential(
            GaussianSmearing(stop=module.num_timesteps, num_gaussians=time_dim, type_="linear")
        )
    elif module.num_timesteps != 0:
        from models.common import GaussianSmearing

        time_dim = int(_get(diff_config, "time_dim", 10))
        module.time_emb = GaussianSmearing(stop=module.num_timesteps, num_gaussians=time_dim, type_="linear")
    try:
        device = next(module.parameters()).device
        module.to(device)
    except StopIteration:
        pass
    print(f"[molsnapper runtime patch] {class_name} num_timesteps={module.num_timesteps}", file=sys.stderr)


_steps = os.environ.get("MOLSNAPPER_NUM_TIMESTEPS")
if _steps:
    try:
        import torch.nn as _nn
    except ModuleNotFoundError:
        _nn = None
    if _nn is not None:
        _orig_load_state_dict = _nn.Module.load_state_dict

        def _patched_load_state_dict(self, state_dict, strict=True):
            result = _orig_load_state_dict(self, state_dict, strict=strict)
            _apply_steps(self, int(_steps))
            return result

        _nn.Module.load_state_dict = _patched_load_state_dict
