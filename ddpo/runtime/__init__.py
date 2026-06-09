"""Runtime public API with lazy imports for heavy DiffSBDD components."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .distributed import (
    DistributedContext,
    all_gather_objects,
    barrier_if_needed,
    broadcast_object,
    destroy_distributed,
    init_distributed_from_env,
    is_distributed_enabled,
    is_main_process,
    reduce_max_scalar,
    reduce_mean_scalar,
    reduce_min_scalar,
    reduce_sum_scalar,
)
from .logging_utils import log_cuda_memory, setup_logging
from .protocol import DataProto, TensorDict
from .seed import set_seed, set_seed_for_rank


_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "apply_cli_overrides": ("ddpo.runtime.build", "apply_cli_overrides"),
    "build_actor": ("ddpo.runtime.build", "build_actor"),
    "build_dataloader": ("ddpo.runtime.build", "build_dataloader"),
    "build_filter": ("ddpo.runtime.build", "build_filter"),
    "build_model": ("ddpo.runtime.build", "build_model"),
    "build_reward_normalizer": ("ddpo.runtime.build", "build_reward_normalizer"),
    "load_config": ("ddpo.runtime.build", "load_config"),
    "resolve_device": ("ddpo.runtime.build", "resolve_device"),
    "save_config_snapshot": ("ddpo.runtime.build", "save_config_snapshot"),
    "parse_args": ("ddpo.runtime.cli", "parse_args"),
    "DiffSBDDDataLoader": ("ddpo.runtime.data_loader", "DiffSBDDDataLoader"),
    "StabilityAwareReward": ("ddpo.runtime.reward_wrappers", "StabilityAwareReward"),
    "DiffSBDDRollout": ("ddpo.runtime.rollout", "DiffSBDDRollout"),
    "BatchStats": ("ddpo.runtime.train_helpers", "BatchStats"),
    "EpochStepResult": ("ddpo.runtime.train_helpers", "EpochStepResult"),
    "PreparedSamples": ("ddpo.runtime.train_helpers", "PreparedSamples"),
    "build_batch_facts": ("ddpo.runtime.train_helpers", "build_batch_facts"),
    "build_epoch_facts": ("ddpo.runtime.train_helpers", "build_epoch_facts"),
    "compute_batch_stats": ("ddpo.runtime.train_helpers", "compute_batch_stats"),
    "finish_run": ("ddpo.runtime.train_helpers", "finish_run"),
    "prepare_training_samples": ("ddpo.runtime.train_helpers", "prepare_training_samples"),
    "update_policy": ("ddpo.runtime.train_helpers", "update_policy"),
}


def __getattr__(name: str) -> Any:
    """Lazily import heavy runtime objects only when requested."""
    if name == "build":
        module = import_module("ddpo.runtime.build")
        globals()[name] = module
        return module
    if name not in _LAZY_ATTRS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_ATTRS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


__all__ = [
    "build_model",
    "build_dataloader",
    "build_filter",
    "build_actor",
    "build_reward_normalizer",
    "load_config",
    "resolve_device",
    "save_config_snapshot",
    "parse_args",
    "DataProto",
    "DiffSBDDDataLoader",
    "DistributedContext",
    "all_gather_objects",
    "barrier_if_needed",
    "broadcast_object",
    "destroy_distributed",
    "init_distributed_from_env",
    "is_distributed_enabled",
    "is_main_process",
    "reduce_max_scalar",
    "reduce_mean_scalar",
    "reduce_min_scalar",
    "reduce_sum_scalar",
    "setup_logging",
    "log_cuda_memory",
    "StabilityAwareReward",
    "DiffSBDDRollout",
    "set_seed",
    "set_seed_for_rank",
    "PreparedSamples",
    "EpochStepResult",
    "BatchStats",
    "prepare_training_samples",
    "update_policy",
    "compute_batch_stats",
    "build_batch_facts",
    "build_epoch_facts",
    "finish_run",
    "TensorDict",
    "build",
]
