"""Load frozen-policy DiffSBDD trajectory shards back into the RL data format.

Collected shards (see ``ddpo/main_diffsbdd_collect_ddp.py``) are plain
``torch.save`` payloads.  This module is the single place that converts such a
payload back into the exact :class:`~ddpo.runtime.protocol.DataProto` layout
that the online RL path produces, so an offline-collected trajectory can be fed
to ``EDMActor.update_policy`` / ``ddpo.runtime.train_helpers.update_policy``
without any bespoke glue.

Format constraints come from:
  * ``ddpo/runtime/actor.py::EDMActor.update_policy``  (tensor + meta_info keys)
  * ``ddpo/runtime/replay.py::REQUIRED_TRAINING_TENSOR_KEYS`` (replay buffer)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch

from ddpo.runtime.protocol import DataProto, TensorDict
from ddpo.runtime.replay import REQUIRED_TRAINING_TENSOR_KEYS


TRAJECTORY_SHARD_FORMAT = "diffsbdd_frozen_policy_trajectory_v1"

#: Tensors persisted by the collector that ``EDMActor`` consumes directly.
TRAJECTORY_TENSOR_KEYS = ("latents", "logps", "timesteps", "nodesxsample")

#: ``meta_info`` keys that ``EDMActor.update_policy`` reads.
REQUIRED_META_INFO_KEYS = ("max_n_nodes", "condition")


class TrajectoryShardError(RuntimeError):
    """Raised when a shard cannot be turned into a valid RL training batch."""


def load_shard_payload(path: str | Path) -> dict[str, Any]:
    """Load and format-check one collected trajectory shard.

    Args:
        path: Path to a ``shard_XXXX.pt`` file written by the collector.

    Returns:
        The raw shard payload dictionary.

    Raises:
        TrajectoryShardError: If the payload is not a supported shard format.
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TrajectoryShardError(f"Shard payload is not a dict: {type(payload)!r}")
    shard_format = payload.get("format")
    if shard_format != TRAJECTORY_SHARD_FORMAT:
        raise TrajectoryShardError(
            f"Unsupported shard format {shard_format!r}; expected {TRAJECTORY_SHARD_FORMAT!r}"
        )
    missing = [key for key in TRAJECTORY_TENSOR_KEYS if payload.get(key) is None]
    if missing:
        raise TrajectoryShardError(f"Shard is missing trajectory tensors: {missing}")
    if "metadata" not in payload:
        raise TrajectoryShardError("Shard payload has no 'metadata' block")
    return payload


def shard_advantages(payload: dict[str, Any]) -> torch.Tensor | None:
    """Return stored advantages, or ``None`` when offline scoring has not run."""
    advantages = payload.get("advantages")
    if advantages is None:
        return None
    return torch.as_tensor(advantages, dtype=torch.float32).reshape(-1)


def build_training_dataproto(
    payload: dict[str, Any],
    *,
    advantages: torch.Tensor | Sequence[float] | None = None,
    indices: Sequence[int] | None = None,
    device: str | torch.device = "cpu",
) -> DataProto:
    """Rebuild the PPO training ``DataProto`` from a shard payload.

    Args:
        payload: Shard payload from :func:`load_shard_payload`.
        advantages: Per-molecule advantages overriding the stored ones. Required
            when the shard was collected before the offline scoring pass.
        indices: Optional subset of molecule indices to keep.
        device: Device the returned tensors are placed on.

    Returns:
        A ``DataProto`` carrying ``latents`` / ``logps`` / ``timesteps`` /
        ``nodesxsample`` / ``advantages`` plus the ``meta_info`` that
        ``EDMActor.update_policy`` requires.

    Raises:
        TrajectoryShardError: If advantages are unavailable or mis-shaped.
    """
    tensors = {key: payload[key] for key in TRAJECTORY_TENSOR_KEYS}
    batch_size = int(tensors["latents"].shape[0])

    if indices is not None:
        index_list = [int(i) for i in indices]
        if any(i < 0 or i >= batch_size for i in index_list):
            raise TrajectoryShardError(
                f"Molecule indices out of range for shard of size {batch_size}: {index_list}"
            )
        tensors = {key: value[index_list] for key, value in tensors.items()}
        batch_size = len(index_list)

    resolved = shard_advantages(payload) if advantages is None else torch.as_tensor(
        advantages, dtype=torch.float32
    ).reshape(-1)
    if resolved is None:
        raise TrajectoryShardError(
            "Shard has no advantages yet (advantages_status="
            f"{payload.get('metadata', {}).get('advantages_status', 'unknown')!r}). "
            "Run the offline scoring pass or pass advantages explicitly."
        )
    if advantages is None and indices is not None:
        resolved = resolved[[int(i) for i in indices]]
    if int(resolved.numel()) != batch_size:
        raise TrajectoryShardError(
            f"Advantages length {int(resolved.numel())} does not match batch size {batch_size}"
        )

    target = torch.device(device)
    batch = TensorDict(
        {
            **{key: value.to(target) for key, value in tensors.items()},
            "advantages": resolved.to(target),
        },
        batch_size=[batch_size],
    )
    metadata = payload["metadata"]
    return DataProto(
        batch=batch,
        meta_info={
            "max_n_nodes": int(metadata["max_n_nodes"]),
            "condition": bool(metadata.get("condition", False)),
        },
    )


def assert_rl_pipeline_compatible(data: DataProto) -> dict[str, Any]:
    """Check a rebuilt batch against the online RL / replay format contract.

    Args:
        data: Batch produced by :func:`build_training_dataproto`.

    Returns:
        A report describing which keys were found and the trajectory shapes.

    Raises:
        TrajectoryShardError: If any required tensor or meta_info key is absent.
    """
    if data.batch is None:
        raise TrajectoryShardError("Training batch has no tensor batch")
    keys = {str(key) for key in data.batch.keys()}
    missing_tensors = [key for key in REQUIRED_TRAINING_TENSOR_KEYS if key not in keys]
    if missing_tensors:
        raise TrajectoryShardError(
            f"Batch is missing replay-required tensors: {missing_tensors}"
        )
    missing_meta = [key for key in REQUIRED_META_INFO_KEYS if key not in data.meta_info]
    if missing_meta:
        raise TrajectoryShardError(f"Batch is missing required meta_info: {missing_meta}")

    latents = data.batch["latents"]
    logps = data.batch["logps"]
    if int(latents.shape[1]) != int(logps.shape[1]) + 1:
        raise TrajectoryShardError(
            "Trajectory length mismatch: expected latents length == logps length + 1, "
            f"got latents={int(latents.shape[1])} logps={int(logps.shape[1])}"
        )
    return {
        "batch_size": len(data),
        "tensor_keys": sorted(keys),
        "meta_info": {key: data.meta_info[key] for key in REQUIRED_META_INFO_KEYS},
        "latents_shape": list(latents.shape),
        "logps_shape": list(logps.shape),
        "replay_required_keys_present": True,
    }
