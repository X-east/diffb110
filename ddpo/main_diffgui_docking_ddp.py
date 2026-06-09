"""DDP entrypoint for decoupled DiffGui docking PPO training."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIFFGUI_ROOT = PROJECT_ROOT / "reference project" / "DiffGui"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(DIFFGUI_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFGUI_ROOT))

from ddpo.diffgui.runtime.actor import DiffGuiActor  # noqa: E402
from ddpo.diffgui.runtime.rollout import DiffGuiRollout  # noqa: E402
from ddpo.main_diffgui_docking_train import (  # noqa: E402
    _build_actor_config,
    _prepare_run_dir,
    _should_save_checkpoint,
    _write_metrics_csv,
)
from scripts.diffgui_docking_batch_sweep import (  # noqa: E402
    _build_batch_summary_row,
    _build_prompt,
    _git_snapshot,
    _load_diffgui_model,
    _load_yaml,
    _physical_gpu_id_from_config,
    _replace_policy_advantages,
    _resolve_path,
    _rewards_by_graph_index,
    _score_reconstructed_sdf,
    _seed_diffgui,
    _start_gpu_memory_monitor,
    _stop_gpu_memory_monitor,
    _torch_peak_memory_mb,
    _write_reconstructed_outputs,
    _write_summary_plot,
)


LOGGER = logging.getLogger("DiffGuiDockingDDP")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="YAML config path.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Exact shared run directory override.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override per-rank graph batch size.")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    """Run DiffGui docking training with PyTorch DDP."""
    args = parse_args()
    dist_info = _init_distributed()
    completed = False
    try:
        config = _apply_cli_overrides(_load_yaml(args.config), args)
        run_dir = _resolve_shared_run_dir(config, args.output_dir, dist_info)
        _prepare_rank_dirs(run_dir, dist_info)
        _setup_logging(run_dir / "logs" / f"rank_{dist_info['rank']:02d}" / "train.log", args.log_level, dist_info)
        _log_startup(args, config, run_dir, dist_info)

        if dist_info["rank"] == 0:
            (run_dir / "config" / "resolved_config.yaml").write_text(
                yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
        _barrier()

        rows = _run_training(args=args, config=config, run_dir=run_dir, dist_info=dist_info)
        if dist_info["rank"] == 0:
            summary = {
                "stage": "diffgui_docking_ddp_train",
                "run_dir": str(run_dir),
                "world_size": int(dist_info["world_size"]),
                "epochs": int(config.get("diffgui_training", {}).get("epochs", 1)),
                "per_rank_batch_size": int(config.get("diffgui_training", {}).get("batch_size", 1)),
                "global_batch_size": int(config.get("diffgui_training", {}).get("batch_size", 1))
                * int(dist_info["world_size"]),
                "metrics_csv": str(run_dir / "metrics" / "metrics.csv"),
                "final_metrics": rows[-1] if rows else {},
            }
            (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
            LOGGER.info("training complete summary=%s", json.dumps(summary, ensure_ascii=False))
        completed = True
    finally:
        if completed:
            _barrier()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _run_training(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    run_dir: Path,
    dist_info: dict[str, int],
) -> list[dict[str, Any]]:
    """Run the DDP training loop."""
    train_cfg = config.get("diffgui_training", {})
    sweep_cfg = config.get("diffgui_docking_batch_sweep", {})
    model_cfg = config.get("diffgui_model", {})
    epochs = int(train_cfg.get("epochs", 1))
    per_rank_batch = int(train_cfg.get("batch_size", 1))
    if epochs <= 0 or per_rank_batch <= 0:
        raise ValueError("diffgui_training.epochs and batch_size must be positive.")

    device = torch.device(f"cuda:{dist_info['local_rank']}" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("DiffGui DDP currently requires CUDA.")
    torch.cuda.set_device(device)
    rank_config = _rank_config(config, dist_info, device)
    model, featurizer, sample_cfg = _load_diffgui_model(
        sample_config=_resolve_path(model_cfg.get("sample_config", DIFFGUI_ROOT / "configs/sample/sample.yml")),
        checkpoint=_resolve_path(model_cfg.get("checkpoint", DIFFGUI_ROOT / "ckpt/trained.pt")),
        num_timesteps=int(sweep_cfg.get("num_timesteps", 100)),
        device=device,
    )
    DiffGuiActor._apply_freeze_policy(model, config.get("train", {}).get("freeze", {}))
    ddp_model = DDP(
        model,
        device_ids=[dist_info["local_rank"]],
        output_device=dist_info["local_rank"],
        find_unused_parameters=bool(config.get("train", {}).get("ddp_find_unused_parameters", False)),
    )
    actor = DiffGuiActor(ddp_model, _build_actor_config(config))
    LOGGER.info("actor_freeze_report=%s", actor.freeze_report)

    metrics_rows: list[dict[str, Any]] = []
    metrics_csv = run_dir / "metrics" / "metrics.csv"
    start_time = time.perf_counter()
    for epoch in range(1, epochs + 1):
        local_row = _run_epoch(
            config=rank_config,
            model=ddp_model,
            actor=actor,
            featurizer=featurizer,
            sample_cfg=sample_cfg,
            epoch=epoch,
            per_rank_batch=per_rank_batch,
            run_dir=run_dir,
            dist_info=dist_info,
        )
        gathered_rows = _all_gather_object(local_row, dist_info["world_size"])
        if dist_info["rank"] == 0:
            row = _aggregate_rank_rows(gathered_rows, epoch=epoch, elapsed=time.perf_counter() - start_time)
            metrics_rows.append(row)
            _write_metrics_csv(metrics_csv, metrics_rows)
            _write_summary_plot(run_dir / "metrics" / "training_summary.png", metrics_rows)
            if _should_save_checkpoint(config, epoch):
                checkpoint_path = _save_checkpoint(run_dir, epoch, ddp_model.module, actor, config, row, dist_info)
                row["checkpoint"] = str(checkpoint_path)
                _write_metrics_csv(metrics_csv, metrics_rows)
            LOGGER.info(
                "epoch=%d/%d global_batch=%d reward_mean=%.4f recon=%d/%d docking_total=%.3f ppo_loss=%.4f elapsed=%.1fs",
                epoch,
                epochs,
                int(row["global_batch_size"]),
                float(row.get("reward_total_mean", 0.0)),
                int(row.get("reconstruction_success", 0)),
                int(row.get("reconstruction_attempted", 0)),
                float(row.get("docking_success_ratio_total", row.get("docking_success_ratio", 0.0))),
                float(row.get("PPO_Loss", 0.0)),
                float(row["elapsed_train_seconds"]),
            )
        _barrier()
    return metrics_rows


def _run_epoch(
    *,
    config: dict[str, Any],
    model: DDP,
    actor: DiffGuiActor,
    featurizer: Any,
    sample_cfg: Any,
    epoch: int,
    per_rank_batch: int,
    run_dir: Path,
    dist_info: dict[str, int],
) -> dict[str, Any]:
    """Run one local rank epoch and return rank-local metrics."""
    sweep_cfg = config.get("diffgui_docking_batch_sweep", {})
    model_cfg = config.get("diffgui_model", {})
    rank = int(dist_info["rank"])
    epoch_dir = run_dir / "epochs" / f"epoch_{epoch:04d}" / f"rank_{rank:02d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    reward_batch_size = int(sweep_cfg.get("reward_batch_size") or per_rank_batch)
    seed = int(sweep_cfg.get("seed_base", 260609)) + epoch * int(dist_info["world_size"]) + rank
    _seed_diffgui(seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    monitor = _start_gpu_memory_monitor(
        _physical_gpu_id_from_config(config, device),
        epoch_dir,
        float(sweep_cfg.get("memory_poll_interval", 1.0)),
    )
    timing: dict[str, float] = {}
    started = time.perf_counter()
    try:
        model.eval()
        phase_start = time.perf_counter()
        prompt, batch_node, halfedge_index, batch_halfedge = _build_prompt(
            sample_cfg=sample_cfg,
            featurizer=featurizer,
            pocket_pdb=_resolve_path(model_cfg.get("pocket_pdb", DIFFGUI_ROOT / "sample/3ztx_pocket.pdb")),
            ligand_sdf=str(model_cfg.get("ligand_sdf", "None")),
            batch_size=per_rank_batch,
            ligand_atoms=int(sweep_cfg.get("ligand_atoms", 24)),
            sweep_cfg=sweep_cfg,
            device=device,
        )
        rollout = DiffGuiRollout(
            model,
            {
                "diffgui_rollout": {
                    "num_timesteps": int(sweep_cfg.get("num_timesteps", 100)),
                    "auto_configure_timesteps": bool(sweep_cfg.get("auto_configure_timesteps", True)),
                    "gui_strength": float(sweep_cfg.get("gui_strength", 0.0)),
                    "cpu_offload": True,
                    "synthetic_advantage_mode": "zeros",
                    "synthetic_advantage_scale": 1.0,
                }
            },
        )
        policy_batch = rollout.generate_policy_batch(prompt)
        policy_batch.validate()
        timing["rollout_seconds"] = time.perf_counter() - phase_start

        phase_start = time.perf_counter()
        reconstruction_rows, written_mols = _write_reconstructed_outputs(
            policy_batch=policy_batch,
            featurizer=featurizer,
            n_graphs=per_rank_batch,
            batch_node=batch_node,
            halfedge_index=halfedge_index,
            batch_halfedge=batch_halfedge,
            output_dir=epoch_dir,
            name_prefix=f"diffgui_ddp_e{epoch:04d}_r{rank:02d}",
        )
        timing["reconstruction_seconds"] = time.perf_counter() - phase_start

        phase_start = time.perf_counter()
        reward_rows = _score_reconstructed_sdf(
            config=config,
            input_sdf=epoch_dir / "generated.sdf",
            output_dir=epoch_dir,
            batch_size=reward_batch_size,
        )
        timing["reward_seconds"] = time.perf_counter() - phase_start

        rewards_by_graph = _rewards_by_graph_index(
            reconstruction_rows=reconstruction_rows,
            reward_rows=reward_rows,
            failure_penalty=float(sweep_cfg.get("failure_advantage", -1.0)),
        )
        advantages = _make_global_advantages(rewards_by_graph, sweep_cfg=sweep_cfg, dist_info=dist_info)
        policy_batch = _replace_policy_advantages(policy_batch, advantages)

        phase_start = time.perf_counter()
        actor_metrics = actor.update_policy(policy_batch)
        timing["ppo_seconds"] = time.perf_counter() - phase_start

        total_seconds = time.perf_counter() - started
        memory = _stop_gpu_memory_monitor(monitor)
        row = _build_batch_summary_row(
            config=config,
            batch_size=per_rank_batch,
            reward_batch_size=reward_batch_size,
            seed=seed,
            timing=timing,
            total_seconds=total_seconds,
            reconstruction_rows=reconstruction_rows,
            written_mols=written_mols,
            reward_rows=reward_rows,
            rewards_by_graph=rewards_by_graph,
            advantages=advantages,
            policy_batch=policy_batch,
            actor_metrics=actor_metrics,
            torch_peak_mb=_torch_peak_memory_mb(device),
            gpu_memory=memory,
            output_dir=epoch_dir,
        )
        row.update(
            {
                "rank": rank,
                "local_rank": int(dist_info["local_rank"]),
                "world_size": int(dist_info["world_size"]),
                "global_batch_size": int(per_rank_batch) * int(dist_info["world_size"]),
                "epoch": int(epoch),
                "global_step": int(epoch),
                "epoch_dir": str(epoch_dir),
            }
        )
        (epoch_dir / "summary.json").write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
        return row
    finally:
        if monitor is not None and not monitor.stop_event.is_set():
            _stop_gpu_memory_monitor(monitor)


def _make_global_advantages(
    rewards: list[float],
    *,
    sweep_cfg: dict[str, Any],
    dist_info: dict[str, int],
) -> torch.Tensor:
    """Convert rank-local rewards into advantages using global DDP statistics."""
    local_rewards = [float(value) for value in rewards]
    gathered = _all_gather_object(local_rewards, int(dist_info["world_size"]))
    flat_rewards = [float(value) for row in gathered for value in row]
    global_values = torch.tensor(flat_rewards, dtype=torch.float32)
    local_values = torch.tensor(local_rewards, dtype=torch.float32)
    if bool(sweep_cfg.get("normalize_advantages", True)) and global_values.numel() > 1:
        epsilon = float(sweep_cfg.get("advantage_epsilon", 1.0e-6))
        std = global_values.std(unbiased=False)
        if float(std.item()) > epsilon:
            local_values = (local_values - global_values.mean()) / (std + epsilon)
        else:
            local_values = local_values - global_values.mean()
    clamp_min = sweep_cfg.get("advantage_clip_min", None)
    clamp_max = sweep_cfg.get("advantage_clip_max", None)
    if clamp_min is not None or clamp_max is not None:
        min_value = -float("inf") if clamp_min is None else float(clamp_min)
        max_value = float("inf") if clamp_max is None else float(clamp_max)
        local_values = local_values.clamp(min=min_value, max=max_value)
    return local_values


def _aggregate_rank_rows(rows: list[dict[str, Any]], *, epoch: int, elapsed: float) -> dict[str, Any]:
    """Aggregate rank-local metric rows into one rank-0 metrics row."""
    global_batch = sum(int(row.get("batch_size", 0)) for row in rows)
    attempted = sum(int(row.get("reconstruction_attempted", 0)) for row in rows)
    recon = sum(int(row.get("reconstruction_success", 0)) for row in rows)
    docking = sum(int(row.get("docking_success", 0)) for row in rows)
    reward_records = sum(int(row.get("reward_records", 0)) for row in rows)
    reward_valid = sum(int(row.get("reward_rdkit_valid", 0)) for row in rows)
    weighted_reward = sum(float(row.get("reward_total_mean", 0.0)) * int(row.get("batch_size", 0)) for row in rows)
    row: dict[str, Any] = {
        "status": "ok",
        "epoch": int(epoch),
        "global_step": int(epoch),
        "world_size": len(rows),
        "batch_size": int(global_batch),
        "global_batch_size": int(global_batch),
        "per_rank_batch_size": int(rows[0].get("batch_size", 0)) if rows else 0,
        "num_timesteps": int(rows[0].get("num_timesteps", 0)) if rows else 0,
        "ligand_size_mode": str(rows[0].get("ligand_size_mode", "")) if rows else "",
        "ligand_atoms_min": min(int(row.get("ligand_atoms_min", 0)) for row in rows) if rows else 0,
        "ligand_atoms_mean": _mean(rows, "ligand_atoms_mean"),
        "ligand_atoms_max": max(int(row.get("ligand_atoms_max", 0)) for row in rows) if rows else 0,
        "total_seconds": max(float(row.get("total_seconds", 0.0)) for row in rows) if rows else 0.0,
        "rollout_seconds": max(float(row.get("rollout_seconds", 0.0)) for row in rows) if rows else 0.0,
        "reward_seconds": max(float(row.get("reward_seconds", 0.0)) for row in rows) if rows else 0.0,
        "ppo_seconds": max(float(row.get("ppo_seconds", 0.0)) for row in rows) if rows else 0.0,
        "seconds_per_graph": max(float(row.get("total_seconds", 0.0)) for row in rows) / float(max(global_batch, 1)),
        "reconstruction_attempted": int(attempted),
        "reconstruction_success": int(recon),
        "reconstruction_success_ratio": float(recon / attempted) if attempted else 0.0,
        "reward_records": int(reward_records),
        "reward_rdkit_valid": int(reward_valid),
        "docking_success": int(docking),
        "docking_success_ratio": float(docking / attempted) if attempted else 0.0,
        "docking_success_ratio_total": float(docking / attempted) if attempted else 0.0,
        "docking_success_ratio_scored": float(docking / reward_records) if reward_records else 0.0,
        "reward_total_mean": float(weighted_reward / global_batch) if global_batch else 0.0,
        "reward_total_min": min(float(row.get("reward_total_min", 0.0)) for row in rows) if rows else 0.0,
        "reward_total_max": max(float(row.get("reward_total_max", 0.0)) for row in rows) if rows else 0.0,
        "advantage_mean": _mean(rows, "advantage_mean"),
        "advantage_std": _mean(rows, "advantage_std"),
        "PPO_Loss": _mean(rows, "PPO_Loss"),
        "PPO_ApproxKL": _mean(rows, "PPO_ApproxKL"),
        "ClipFrac": _mean(rows, "ClipFrac"),
        "RatioMean": _mean(rows, "RatioMean"),
        "GradNorm": _mean(rows, "GradNorm"),
        "torch_peak_allocated_mb": max(float(row.get("torch_peak_allocated_mb", 0.0)) for row in rows) if rows else 0.0,
        "gpu_monitor_peak_used_mb": max(float(row.get("gpu_monitor_peak_used_mb", 0.0)) for row in rows) if rows else 0.0,
        "elapsed_train_seconds": float(elapsed),
    }
    row["rank_rows_json"] = json.dumps(rows, ensure_ascii=False)
    return row


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if key in row and row[key] != ""]
    return float(sum(values) / len(values)) if values else 0.0


def _save_checkpoint(
    run_dir: Path,
    epoch: int,
    model: torch.nn.Module,
    actor: DiffGuiActor,
    config: dict[str, Any],
    metrics: dict[str, Any],
    dist_info: dict[str, int],
) -> Path:
    """Save one rank-0 DDP checkpoint."""
    checkpoint_path = run_dir / "checkpoints" / f"diffgui-docking-ddp-epoch={epoch:04d}.pt"
    payload = {
        "stage": "diffgui_docking_ddp_train",
        "epoch": int(epoch),
        "world_size": int(dist_info["world_size"]),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": actor.optimizer.state_dict(),
        "config": config,
        "metrics": metrics,
        "freeze_report": actor.freeze_report,
    }
    torch.save(payload, checkpoint_path)
    torch.save(payload, run_dir / "checkpoints" / "latest.pt")
    LOGGER.info("checkpoint saved epoch=%d path=%s", epoch, checkpoint_path)
    return checkpoint_path


def _apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Apply DDP-safe CLI overrides to a loaded config."""
    training_cfg = config.setdefault("diffgui_training", {})
    if args.epochs is not None:
        training_cfg["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        training_cfg["batch_size"] = int(args.batch_size)
    return config


def _rank_config(config: dict[str, Any], dist_info: dict[str, int], device: torch.device) -> dict[str, Any]:
    """Return a shallow rank-local config with device and docking GPU bound."""
    rank_config = dict(config)
    sweep_cfg = dict(config.get("diffgui_docking_batch_sweep", {}))
    reward_cfg = dict(config.get("reward", {}))
    docking_cfg = dict(reward_cfg.get("docking", {}))
    sweep_cfg["device"] = str(device)
    docking_cfg["gpu_id"] = int(_physical_gpu_id(dist_info["local_rank"]))
    reward_cfg["docking"] = docking_cfg
    rank_config["diffgui_docking_batch_sweep"] = sweep_cfg
    rank_config["reward"] = reward_cfg
    return rank_config


def _physical_gpu_id(local_rank: int) -> int:
    """Resolve physical GPU id from CUDA_VISIBLE_DEVICES and local rank."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        parts = [part.strip() for part in visible.split(",") if part.strip()]
        if 0 <= int(local_rank) < len(parts):
            return int(parts[int(local_rank)])
    return int(local_rank)


def _resolve_shared_run_dir(config: dict[str, Any], output_dir: Path | None, dist_info: dict[str, int]) -> Path:
    """Resolve and broadcast one shared run directory."""
    if dist_info["rank"] == 0:
        if output_dir is not None:
            value = str(output_dir.resolve())
        else:
            train_cfg = config.get("diffgui_training", {})
            output_root = _resolve_path(train_cfg.get("output_root", "OUT"))
            run_name = str(config.get("name", "diffgui-docking-ddp-train")).replace("/", "_")
            value = str(output_root / f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    else:
        value = ""
    values = [value]
    if dist.is_available() and dist.is_initialized():
        dist.broadcast_object_list(values, src=0)
    return Path(values[0]).resolve()


def _prepare_rank_dirs(run_dir: Path, dist_info: dict[str, int]) -> None:
    """Create shared and rank-local output directories."""
    if dist_info["rank"] == 0:
        _prepare_run_dir(run_dir)
    _barrier()
    (run_dir / "logs" / f"rank_{dist_info['rank']:02d}").mkdir(parents=True, exist_ok=True)


def _setup_logging(log_path: Path, log_level: str, dist_info: dict[str, int]) -> None:
    """Configure rank-local file logging and rank-0 terminal logging."""
    LOGGER.setLevel(getattr(logging, log_level))
    LOGGER.handlers.clear()
    formatter = logging.Formatter(f"%(asctime)s %(levelname)s %(name)s rank={dist_info['rank']}: %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    if dist_info["rank"] == 0:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        LOGGER.addHandler(stream_handler)


def _log_startup(args: argparse.Namespace, config: dict[str, Any], run_dir: Path, dist_info: dict[str, int]) -> None:
    """Log startup metadata."""
    LOGGER.info("starting DDP DiffGui docking training")
    LOGGER.info("config_path=%s", args.config.resolve())
    LOGGER.info("run_dir=%s", run_dir)
    LOGGER.info("dist=%s", dist_info)
    LOGGER.info("git=%s", _git_snapshot())
    LOGGER.info("conda_prefix=%s python=%s", os.environ.get("CONDA_PREFIX", ""), sys.executable)
    LOGGER.info("cuda_visible_devices=%s", os.environ.get("CUDA_VISIBLE_DEVICES", ""))


def _init_distributed() -> dict[str, int]:
    """Initialize torch.distributed from torchrun environment."""
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    return {"rank": rank, "local_rank": local_rank, "world_size": world_size}


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _all_gather_object(value: Any, world_size: int) -> list[Any]:
    """Gather picklable objects from every rank."""
    if dist.is_available() and dist.is_initialized():
        gathered: list[Any] = [None for _ in range(int(world_size))]
        dist.all_gather_object(gathered, value)
        return gathered
    return [value]


if __name__ == "__main__":
    main()
