"""Collect frozen-policy DiffGui trajectories with PyTorch distributed workers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import random
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIFFGUI_ROOT = PROJECT_ROOT / "reference project" / "代码" / "DiffGui"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(DIFFGUI_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFGUI_ROOT))

from ddpo.diffgui.runtime.rollout import DiffGuiRollout  # noqa: E402
from scripts.diffgui_docking_batch_sweep import (  # noqa: E402
    _build_prompt,
    _git_snapshot,
    _load_diffgui_model,
    _load_yaml,
    _resolve_path,
    _seed_diffgui,
    _write_reconstructed_outputs,
)


LOGGER = logging.getLogger("DiffGuiTrajectoryCollector")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Collection YAML config.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Exact shared run directory.")
    parser.add_argument("--resume", action="store_true", help="Resume from completed rank-local shards.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    """Collect frozen-policy trajectories without rewards or optimizer updates."""
    args = parse_args()
    config = _load_yaml(args.config)
    dist_info = _init_distributed()
    run_dir = _resolve_run_dir(config, args.output_dir, dist_info)
    _prepare_dirs(run_dir, dist_info)
    _setup_logging(run_dir / "logs" / f"rank_{dist_info['rank']:02d}" / "collect.log", args.log_level, dist_info)
    attempt_id = _broadcast_attempt_id(dist_info)

    completed = False
    try:
        summary = _collect(
            config=config,
            run_dir=run_dir,
            dist_info=dist_info,
            resume=args.resume,
            attempt_id=attempt_id,
        )
        if dist_info["rank"] == 0:
            (run_dir / "summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            _write_run_manifest(args, config, run_dir, dist_info, status="completed", summary=summary)
        completed = True
    finally:
        if dist_info["rank"] == 0 and not completed:
            _write_run_manifest(args, config, run_dir, dist_info, status="interrupted")
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _collect(
    *,
    config: dict[str, Any],
    run_dir: Path,
    dist_info: dict[str, int],
    resume: bool,
    attempt_id: str,
) -> dict[str, Any]:
    """Run rank-local collection and aggregate a global summary."""
    collection_cfg = config.get("diffgui_collection", {}) or {}
    model_cfg = config.get("diffgui_model", {}) or {}
    sweep_cfg = config.get("diffgui_docking_batch_sweep", {}) or {}
    batch_size = int(collection_cfg.get("batch_size_per_rank", 1))
    num_shards = int(collection_cfg.get("num_shards_per_rank", 1))
    if batch_size <= 0 or num_shards <= 0:
        raise ValueError("Collection batch_size_per_rank and num_shards_per_rank must be positive.")

    rank = int(dist_info["rank"])
    local_rank = int(dist_info["local_rank"])
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("DiffGui trajectory collection requires CUDA.")
    torch.cuda.set_device(device)

    checkpoint = _resolve_path(model_cfg.get("checkpoint", DIFFGUI_ROOT / "ckpt/trained.pt"))
    checkpoint_sha256 = _sha256_file(checkpoint)
    sample_config = _resolve_path(
        model_cfg.get("sample_config", DIFFGUI_ROOT / "configs/sample/sample.yml")
    )
    num_timesteps = int(sweep_cfg.get("num_timesteps", 200))
    model, featurizer, sample_cfg = _load_diffgui_model(
        sample_config=sample_config,
        checkpoint=checkpoint,
        num_timesteps=num_timesteps,
        device=device,
    )
    model.eval()
    model_digest_before = _model_digest(model)
    seed = local_rank + 1
    start_shard = _restore_or_seed(run_dir, rank=rank, seed=seed, resume=resume, device=device)
    LOGGER.info(
        "collection start checkpoint=%s sha256=%s steps=%d batch=%d shards=%d seed=%d start_shard=%d",
        checkpoint,
        checkpoint_sha256,
        num_timesteps,
        batch_size,
        num_shards,
        seed,
        start_shard,
    )

    manifest_path = run_dir / "manifests" / f"rank_{rank:02d}.jsonl"
    rank_started = time.perf_counter()
    new_attempts = 0
    new_reconstructed = 0
    new_bytes = 0
    for shard_index in range(start_shard, num_shards):
        shard_started = time.perf_counter()
        output_dir = run_dir / "molecules" / f"rank_{rank:02d}" / f"shard_{shard_index:04d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        prompt, batch_node, halfedge_index, batch_halfedge = _build_prompt(
            sample_cfg=sample_cfg,
            featurizer=featurizer,
            pocket_pdb=_resolve_path(model_cfg["pocket_pdb"]),
            ligand_sdf=str(model_cfg.get("ligand_sdf", "None")),
            batch_size=batch_size,
            ligand_atoms=int(sweep_cfg.get("ligand_atoms", 24)),
            sweep_cfg=sweep_cfg,
            device=device,
        )
        rollout = DiffGuiRollout(
            model,
            {
                "diffgui_rollout": {
                    "num_timesteps": num_timesteps,
                    "auto_configure_timesteps": bool(sweep_cfg.get("auto_configure_timesteps", True)),
                    "gui_strength": float(sweep_cfg.get("gui_strength", 0.0)),
                    "cpu_offload": True,
                    "synthetic_advantage_mode": "zeros",
                }
            },
        )
        policy_batch = rollout.generate_policy_batch(prompt)
        policy_batch.validate()
        rows, reconstructed = _write_reconstructed_outputs(
            policy_batch=policy_batch,
            featurizer=featurizer,
            n_graphs=batch_size,
            batch_node=batch_node,
            halfedge_index=halfedge_index,
            batch_halfedge=batch_halfedge,
            output_dir=output_dir,
            name_prefix=f"diffgui_collect_r{rank:02d}_s{shard_index:04d}",
        )
        sample_ids = [f"r{rank:02d}_s{shard_index:04d}_m{index:04d}" for index in range(batch_size)]
        _write_collection_rows(
            output_dir / "reconstruction.csv",
            rows,
            sample_ids=sample_ids,
            rank=rank,
            shard_index=shard_index,
            seed=seed,
        )

        shard_path = run_dir / "shards" / f"rank_{rank:02d}" / f"shard_{shard_index:04d}.pt"
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        rng_state_after = _capture_rng_state(device)
        payload = {
            "format": "diffgui_frozen_policy_trajectory_v1",
            "policy_batch": policy_batch,
            "sample_ids": sample_ids,
            "metadata": {
                "rank": rank,
                "local_rank": local_rank,
                "shard_index": shard_index,
                "seed": seed,
                "num_timesteps": num_timesteps,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_sha256,
                "advantages_status": "placeholder_zeros_replace_after_offline_scoring",
            },
            "reconstruction_rows": rows,
            "rng_state_after": rng_state_after,
        }
        temporary_path = shard_path.with_suffix(".pt.tmp")
        torch.save(payload, temporary_path)
        temporary_path.replace(shard_path)
        shard_sha256 = _sha256_file(shard_path)
        shard_bytes = shard_path.stat().st_size
        ligand_counts = torch.bincount(policy_batch.ligand_batch.long(), minlength=batch_size)
        manifest_row = {
            "rank": rank,
            "shard_index": shard_index,
            "seed": seed,
            "attempts": batch_size,
            "reconstructed": reconstructed,
            "num_timesteps": num_timesteps,
            "trajectory_steps": int(policy_batch.old_logprob.shape[0]),
            "state_steps": int(policy_batch.node_traj.shape[0]),
            "ligand_atoms_min": int(ligand_counts.min().item()),
            "ligand_atoms_mean": float(ligand_counts.float().mean().item()),
            "ligand_atoms_max": int(ligand_counts.max().item()),
            "shard_path": str(shard_path.relative_to(run_dir)),
            "shard_bytes": shard_bytes,
            "shard_sha256": shard_sha256,
            "elapsed_seconds": time.perf_counter() - shard_started,
            "torch_peak_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024.0**2),
            "status": "completed",
        }
        _append_jsonl(manifest_path, manifest_row)
        new_attempts += batch_size
        new_reconstructed += reconstructed
        new_bytes += shard_bytes
        LOGGER.info("shard complete %s", json.dumps(manifest_row, ensure_ascii=False))

    model_digest_after = _model_digest(model)
    manifest_records = _read_jsonl(manifest_path)
    local_summary = {
        "attempt_id": attempt_id,
        "rank": rank,
        "seed": seed,
        "new_attempts": new_attempts,
        "new_reconstructed": new_reconstructed,
        "new_shard_bytes": new_bytes,
        "attempts_total": sum(int(row.get("attempts", 0)) for row in manifest_records),
        "reconstructed_total": sum(int(row.get("reconstructed", 0)) for row in manifest_records),
        "shard_bytes_total": sum(int(row.get("shard_bytes", 0)) for row in manifest_records),
        "completed_shards": sum(1 for row in manifest_records if row.get("status") == "completed"),
        "model_parameters_unchanged": model_digest_before == model_digest_after,
        "elapsed_seconds": time.perf_counter() - rank_started,
    }
    rank_summary_path = run_dir / "rank_summaries" / attempt_id / f"rank_{rank:02d}.json"
    _write_json_atomic(rank_summary_path, local_summary)
    LOGGER.info("rank collection complete summary=%s", json.dumps(local_summary, ensure_ascii=False))
    if rank != 0:
        return local_summary

    gathered = _wait_for_rank_summaries(
        run_dir=run_dir,
        attempt_id=attempt_id,
        world_size=int(dist_info["world_size"]),
        timeout_seconds=float(collection_cfg.get("finalize_timeout_seconds", 7200.0)),
    )
    summary = {
        "stage": "diffgui_frozen_policy_trajectory_collection",
        "status": "completed",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "world_size": int(dist_info["world_size"]),
        "num_timesteps": num_timesteps,
        "batch_size_per_rank": batch_size,
        "num_shards_per_rank": num_shards,
        "attempts_new": sum(int(row["new_attempts"]) for row in gathered),
        "attempts_total": sum(int(row["attempts_total"]) for row in gathered),
        "reconstructed_new": sum(int(row["new_reconstructed"]) for row in gathered),
        "reconstructed_total": sum(int(row["reconstructed_total"]) for row in gathered),
        "shard_bytes_new": sum(int(row["new_shard_bytes"]) for row in gathered),
        "shard_bytes_total": sum(int(row["shard_bytes_total"]) for row in gathered),
        "all_models_unchanged": all(bool(row["model_parameters_unchanged"]) for row in gathered),
        "rank_summaries": gathered,
    }
    return summary


def _wait_for_rank_summaries(
    *, run_dir: Path, attempt_id: str, world_size: int, timeout_seconds: float
) -> list[dict[str, Any]]:
    """Wait for rank-local filesystem summaries without a terminal NCCL collective."""
    summary_dir = run_dir / "rank_summaries" / attempt_id
    deadline = time.monotonic() + max(1.0, float(timeout_seconds))
    last_reported = -1
    while True:
        gathered: list[dict[str, Any]] = []
        for rank in range(int(world_size)):
            path = summary_dir / f"rank_{rank:02d}.json"
            if not path.exists():
                continue
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if row.get("attempt_id") == attempt_id and int(row.get("rank", -1)) == rank:
                gathered.append(row)
        if len(gathered) == int(world_size):
            return sorted(gathered, key=lambda row: int(row["rank"]))
        if len(gathered) != last_reported:
            LOGGER.info(
                "waiting for rank summaries attempt_id=%s ready=%d/%d",
                attempt_id,
                len(gathered),
                world_size,
            )
            last_reported = len(gathered)
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Timed out waiting for rank summaries: attempt_id={attempt_id} "
                f"ready={len(gathered)}/{world_size}"
            )
        time.sleep(5.0)


def _restore_or_seed(
    run_dir: Path, *, rank: int, seed: int, resume: bool, device: torch.device
) -> int:
    """Restore the last completed shard RNG state or initialize the rank seed once."""
    manifest_path = run_dir / "manifests" / f"rank_{rank:02d}.jsonl"
    records = _read_jsonl(manifest_path)
    if not resume or not records:
        if records and not resume:
            raise FileExistsError(f"Existing manifest requires --resume: {manifest_path}")
        _seed_diffgui(seed)
        return 0
    last = max(records, key=lambda row: int(row["shard_index"]))
    shard_path = run_dir / str(last["shard_path"])
    payload = torch.load(shard_path, map_location="cpu", weights_only=False)
    _restore_rng_state(payload["rng_state_after"], device)
    return int(last["shard_index"]) + 1


def _capture_rng_state(device: torch.device) -> dict[str, Any]:
    """Capture all RNG states needed for duplicate-free continuation."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(device),
    }


def _restore_rng_state(state: dict[str, Any], device: torch.device) -> None:
    """Restore Python, NumPy, CPU Torch, and rank-local CUDA RNG states."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state(state["torch_cuda"], device)


def _write_collection_rows(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    sample_ids: list[str],
    rank: int,
    shard_index: int,
    seed: int,
) -> None:
    """Write reconstruction rows with stable trajectory identifiers."""
    enriched: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        mol_index = int(item["mol_index"])
        item.update(
            {
                "sample_id": sample_ids[mol_index],
                "rank": rank,
                "shard_index": shard_index,
                "rank_seed": seed,
            }
        )
        enriched.append(item)
    fieldnames = [
        "sample_id",
        "rank",
        "shard_index",
        "rank_seed",
        "mol_index",
        "reconstruction_status",
        "smiles",
        "num_atoms",
        "old_logprob_mean",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(enriched)


def _model_digest(model: torch.nn.Module) -> str:
    """Return a deterministic digest of model parameters."""
    digest = hashlib.sha256()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            digest.update(name.encode("utf-8"))
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    """Append one durable JSONL record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write one JSON object."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary_path.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read existing non-empty JSONL records."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _resolve_run_dir(
    config: dict[str, Any], output_dir: Path | None, dist_info: dict[str, int]
) -> Path:
    """Resolve and broadcast one shared run directory."""
    if dist_info["rank"] == 0:
        if output_dir is not None:
            value = str(output_dir.resolve())
        else:
            cfg = config.get("diffgui_collection", {}) or {}
            root = _resolve_path(cfg.get("output_root", "runs/diffgui_collection"))
            name = str(config.get("name", "diffgui-trajectory-collection"))
            value = str(root / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    else:
        value = ""
    values = [value]
    if dist.is_available() and dist.is_initialized():
        dist.broadcast_object_list(values, src=0)
    return Path(values[0]).resolve()


def _prepare_dirs(run_dir: Path, dist_info: dict[str, int]) -> None:
    """Prepare shared and rank-local collection directories."""
    if dist_info["rank"] == 0:
        for subdir in ("config", "logs", "manifests", "molecules", "shards"):
            (run_dir / subdir).mkdir(parents=True, exist_ok=True)
    _barrier()
    (run_dir / "logs" / f"rank_{dist_info['rank']:02d}").mkdir(parents=True, exist_ok=True)


def _setup_logging(log_path: Path, level: str, dist_info: dict[str, int]) -> None:
    """Configure durable rank logs and rank-0 terminal progress."""
    LOGGER.setLevel(getattr(logging, level))
    LOGGER.handlers.clear()
    formatter = logging.Formatter(
        f"%(asctime)s %(levelname)s %(name)s rank={dist_info['rank']}: %(message)s"
    )
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    if dist_info["rank"] == 0:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        LOGGER.addHandler(stream_handler)


def _write_run_manifest(
    args: argparse.Namespace,
    config: dict[str, Any],
    run_dir: Path,
    dist_info: dict[str, int],
    *,
    status: str,
    summary: dict[str, Any] | None = None,
) -> None:
    """Write the durable collection manifest and resolved config."""
    if dist_info["rank"] != 0:
        return
    resolved_config = run_dir / "config" / "resolved_config.yaml"
    resolved_config.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    payload = {
        "stage": "diffgui_frozen_policy_trajectory_collection",
        "status": status,
        "run_dir": str(run_dir),
        "config_path": str(args.config.resolve()),
        "world_size": int(dist_info["world_size"]),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "conda_prefix": os.environ.get("CONDA_PREFIX", ""),
        "python": sys.executable,
        "git": _git_snapshot(),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "summary": summary or {},
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _init_distributed() -> dict[str, int]:
    """Initialize torch.distributed from torchrun environment variables."""
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    return {"rank": rank, "local_rank": local_rank, "world_size": world_size}


def _broadcast_attempt_id(dist_info: dict[str, int]) -> str:
    """Create one identifier shared by all ranks for filesystem finalization."""
    value = uuid.uuid4().hex if int(dist_info["rank"]) == 0 else ""
    values = [value]
    if dist.is_available() and dist.is_initialized():
        dist.broadcast_object_list(values, src=0)
    return str(values[0])


def _barrier() -> None:
    """Synchronize distributed workers when initialized."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    main()
