from __future__ import annotations

"""Checkpoint save/restore helpers for native DDP training."""

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ddpo.runtime.distributed import all_gather_objects, barrier_if_needed, is_main_process
from ddpo.runtime.recording import CheckpointFacts


CHECKPOINT_VERSION = 2


def resolve_checkpoint_path(args: Any) -> str | None:
    """Resolve the checkpoint requested by ``--checkpoint_path`` or ``--resume``.

    ``--checkpoint_path`` is explicit and preferred. ``--resume`` without a
    checkpoint path looks for the newest epoch checkpoint under
    ``--save_path/checkpoints``. This keeps the rule predictable and avoids
    guessing from templated output directories.
    """
    checkpoint_path = getattr(args, "checkpoint_path", None)
    if checkpoint_path:
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return str(path)

    if not bool(getattr(args, "resume", False)):
        return None

    save_path = getattr(args, "save_path", None)
    if not save_path:
        raise ValueError("--resume requires --checkpoint_path or --save_path pointing to an existing run directory.")

    checkpoints_dir = Path(save_path) / "checkpoints"
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"Cannot resume: checkpoints directory not found: {checkpoints_dir}")

    candidates = sorted(
        checkpoints_dir.glob("diffsbdd-ddpo-epoch=*.pt"),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"Cannot resume: no epoch checkpoints found in {checkpoints_dir}")
    return str(candidates[-1])


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state()
        state["cuda_all"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if torch.cuda.is_available():
        if "cuda" in state:
            torch.cuda.set_rng_state(state["cuda"])
        elif "cuda_all" in state:
            torch.cuda.set_rng_state_all(state["cuda_all"])


def _collect_rng_states(runtime: Any) -> dict[str, dict[str, Any]]:
    local_state = {
        "rank": int(runtime.dist_ctx.rank),
        "local_rank": int(runtime.dist_ctx.local_rank),
        "rng_state": _capture_rng_state(),
    }
    gathered = all_gather_objects(local_state, runtime.dist_ctx)
    return {str(item["rank"]): item for item in gathered}


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _load_model_state(runtime: Any, state_dict: dict[str, Any]) -> None:
    try:
        runtime.model_impl.model.load_state_dict(state_dict, strict=True)
        return
    except RuntimeError:
        stripped = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }
        runtime.model_impl.model.load_state_dict(stripped, strict=True)


def _torch_load_checkpoint(checkpoint_path: str, map_location: torch.device) -> dict[str, Any]:
    try:
        return torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=map_location)


def _restore_rank_rng(runtime: Any, checkpoint: dict[str, Any]) -> None:
    rng_by_rank = checkpoint.get("rng_state_by_rank")
    if isinstance(rng_by_rank, dict):
        rank_state = rng_by_rank.get(str(runtime.dist_ctx.rank))
        if isinstance(rank_state, dict):
            _restore_rng_state(rank_state.get("rng_state"))
            return

    # Backward/forward compatibility for single-process checkpoints.
    _restore_rng_state(checkpoint.get("rng_state"))


def restore_training_state(runtime: Any, checkpoint_path: str) -> dict[str, Any]:
    """Restore model, optimizer, epoch, and RNG state into an initialized runtime."""
    checkpoint = _torch_load_checkpoint(checkpoint_path, map_location=runtime.dist_ctx.device)

    model_state = checkpoint.get("model_state_dict")
    if model_state is None:
        raise KeyError(f"Checkpoint is missing model_state_dict: {checkpoint_path}")
    _load_model_state(runtime, model_state)

    optimizer_state = checkpoint.get("optimizer_state_dict")
    if optimizer_state is not None:
        runtime.actor.optimizer.load_state_dict(optimizer_state)
        _move_optimizer_state_to_device(runtime.actor.optimizer, runtime.dist_ctx.device)
    else:
        runtime.logger.warning("Checkpoint has no optimizer_state_dict; continuing with a fresh optimizer.")

    _restore_rank_rng(runtime, checkpoint)

    runtime.start_epoch = int(checkpoint.get("epoch", 0))
    runtime.resumed_from_checkpoint = str(checkpoint_path)
    runtime.resume_checkpoint_version = int(checkpoint.get("checkpoint_version", 1))
    runtime.run_recorder.last_checkpoint_path = str(checkpoint_path)
    runtime.run_recorder.resume_info = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": runtime.start_epoch,
        "checkpoint_version": runtime.resume_checkpoint_version,
    }
    runtime.logger.info(
        "Restored checkpoint | path=%s epoch=%d version=%s",
        checkpoint_path,
        runtime.start_epoch,
        checkpoint.get("checkpoint_version", "legacy"),
    )
    return checkpoint


def restore_checkpoint_if_requested(runtime: Any, checkpoint_path: str | None) -> dict[str, Any] | None:
    if checkpoint_path is None:
        return None
    checkpoint = restore_training_state(runtime, checkpoint_path)
    barrier_if_needed(runtime.dist_ctx)
    return checkpoint


def _checkpoint_metadata(runtime: Any, epoch: int, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "epoch": int(epoch),
        "next_epoch": int(epoch) + 1,
        "config": runtime.config,
        "seed": int(runtime.seed),
        "metrics": dict(metrics),
        "runtime": {
            "dataset_name": runtime.dataset_name,
            "device": str(runtime.device),
            "world_size": int(runtime.dist_ctx.world_size),
            "rank_saved_by": int(runtime.dist_ctx.rank),
            "per_rank_batch_size": int(runtime.per_rank_batch_size),
            "global_batch_size": int(runtime.global_batch_size),
            "max_epochs": int(runtime.max_epochs),
            "checkpoint_interval": int(runtime.checkpoint_interval),
        },
        "paths": {
            "output_dir": runtime.run_context.output_dir,
            "checkpoints_dir": runtime.run_context.checkpoints_dir,
            "config_snapshot_path": runtime.run_context.config_snapshot_path,
        },
        "resume": {
            "resumed_from_checkpoint": getattr(runtime, "resumed_from_checkpoint", None),
            "resume_start_epoch": int(getattr(runtime, "start_epoch", 0)),
            "resume_checkpoint_version": getattr(runtime, "resume_checkpoint_version", None),
        },
    }


def build_checkpoint_payload(
    runtime: Any,
    epoch: int,
    metrics: dict[str, Any],
    rng_state_by_rank: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    payload = _checkpoint_metadata(runtime, epoch, metrics)
    payload.update(
        {
            "model_state_dict": runtime.model_impl.model.state_dict(),
            "optimizer_state_dict": runtime.actor.optimizer.state_dict(),
            "rng_state_by_rank": rng_state_by_rank,
        }
    )
    return payload


def _write_latest_pointer(checkpoints_dir: str, checkpoint_path: str, epoch: int) -> None:
    pointer_path = Path(checkpoints_dir) / "checkpoint_latest.json"
    payload = {
        "epoch": int(epoch),
        "path": str(checkpoint_path),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(pointer_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_checkpoint_ddp(runtime: Any, epoch: int, metrics: dict[str, Any]) -> CheckpointFacts | None:
    """Save a full DDP training checkpoint on rank 0.

    All ranks participate in RNG collection so resuming can restore rank-local
    random streams. Only rank 0 writes the actual checkpoint file.
    """
    barrier_if_needed(runtime.dist_ctx)
    if (epoch % runtime.checkpoint_interval) != 0:
        barrier_if_needed(runtime.dist_ctx)
        return None

    rng_state_by_rank = _collect_rng_states(runtime)
    checkpoint_name = f"diffsbdd-ddpo-epoch={epoch:04d}.pt"
    checkpoint_path = str(Path(runtime.run_context.checkpoints_dir) / checkpoint_name)

    facts: CheckpointFacts | None = None
    if is_main_process(runtime.dist_ctx):
        payload = build_checkpoint_payload(
            runtime=runtime,
            epoch=epoch,
            metrics=metrics,
            rng_state_by_rank=rng_state_by_rank,
        )
        torch.save(payload, checkpoint_path)
        _write_latest_pointer(runtime.run_context.checkpoints_dir, checkpoint_path, epoch)

        checkpoint_file = Path(checkpoint_path)
        exists = checkpoint_file.exists()
        size_bytes = checkpoint_file.stat().st_size if exists else 0
        avg_loss = float(metrics.get("Loss", metrics.get("loss", metrics.get("ppo_loss", 0.0))))
        facts = CheckpointFacts(
            epoch=epoch,
            path=checkpoint_path,
            kind="interval",
            monitor="avg_loss",
            monitor_value=avg_loss,
            exists=exists,
            size_bytes=size_bytes,
            confirmed=exists and size_bytes > 0,
            saved_at=datetime.now().isoformat(timespec="seconds"),
        )
        runtime.run_recorder.on_checkpoint_saved(facts)

    barrier_if_needed(runtime.dist_ctx)
    return facts
