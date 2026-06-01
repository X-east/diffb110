"""Runtime-only DiffGui sampling step override.

This patch is loaded through PYTHONPATH. It leaves the reference project source
untouched and rebuilds diffusion schedules after checkpoint weights are loaded.
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
    old_sum = sum(int(x) for x in old_segments)
    if old_sum <= 0:
        first = max(1, total // 2)
        return [first, total - first]

    new_segments = []
    remaining = total
    for idx, old_value in enumerate(old_segments):
        slots_left = len(old_segments) - idx
        if idx == len(old_segments) - 1:
            value = remaining
        else:
            value = int(round(total * int(old_value) / old_sum))
            value = max(1, min(value, remaining - (slots_left - 1)))
        new_segments.append(value)
        remaining -= value
    return new_segments


def _patch_diff_config(diff_config, steps):
    _set(diff_config, "num_timesteps", int(steps))
    for section_name in ("diff_bond",):
        section = _get(diff_config, section_name)
        if section is None:
            continue
        if _get(section, "beta_schedule") == "segment":
            old_segments = _get(section, "time_segment", [])
            _set(section, "time_segment", _scale_segments(old_segments, int(steps)))


def _apply_steps(module, steps):
    class_name = module.__class__.__name__
    if class_name not in {"DiffGui", "BondPredictor"}:
        return
    try:
        try:
            device = next(module.parameters()).device
        except StopIteration:
            device = None
        diff_config = module.config.diff
        _patch_diff_config(diff_config, int(steps))
        module.define_betas_alphas(diff_config)

        from models.common import GaussianSmearing
        import torch.nn as nn

        time_dim = int(_get(diff_config, "time_dim", 10))
        if class_name == "DiffGui":
            module.time_emb = nn.Sequential(
                GaussianSmearing(stop=module.num_timesteps, num_gaussians=time_dim, type_="linear")
            )
        elif module.num_timesteps != 0:
            module.time_emb = GaussianSmearing(
                stop=module.num_timesteps, num_gaussians=time_dim, type_="linear"
            )

        if device is not None:
            module.to(device)
        print(f"[diffgui runtime patch] {class_name} num_timesteps={module.num_timesteps}", file=sys.stderr)
    except Exception as exc:
        print(f"[diffgui runtime patch] failed for {class_name}: {exc}", file=sys.stderr)
        raise


_steps = os.environ.get("DIFFGUI_NUM_TIMESTEPS")
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
