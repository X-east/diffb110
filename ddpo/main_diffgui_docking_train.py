"""Formal single-process DiffGui docking DDPO/PPO training entrypoint.

This entrypoint keeps DiffGui training decoupled from the existing DiffSBDD
DDPO mainline.  DiffGui rollout and actor update run in the ``diffgui`` conda
environment, while molecular rewards are evaluated through the existing
reward-only SDF adapter in the ``dif`` environment.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIFFGUI_ROOT = PROJECT_ROOT / "reference project" / "DiffGui"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(DIFFGUI_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFGUI_ROOT))

from ddpo.diffgui.runtime.actor import DiffGuiActor  # noqa: E402
from ddpo.diffgui.runtime.rollout import DiffGuiRollout  # noqa: E402
from scripts.diffgui_docking_batch_sweep import (  # noqa: E402
    _build_batch_summary_row,
    _build_prompt,
    _git_snapshot,
    _load_diffgui_model,
    _load_yaml,
    _make_advantages,
    _physical_gpu_id_from_config,
    _replace_policy_advantages,
    _resolve_path,
    _rewards_by_graph_index,
    _score_reconstructed_sdf,
    _seed_diffgui,
    _start_gpu_memory_monitor,
    _stop_gpu_memory_monitor,
    _torch_peak_memory_mb,
    _union_fieldnames,
    _write_reconstructed_outputs,
    _write_summary_plot,
)


LOGGER = logging.getLogger("DiffGuiDockingTrain")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="YAML config path.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Exact run directory override.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override graph batch size.")
    parser.add_argument("--device", type=str, default=None, help="Override DiffGui torch device, e.g. cuda:7.")
    parser.add_argument("--docking-gpu-id", type=int, default=None, help="Override UniDock physical GPU id.")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    """Run formal DiffGui docking training."""
    args = parse_args()
    config = _apply_cli_overrides(_load_yaml(args.config), args)
    run_dir = _resolve_run_dir(config, args.output_dir)
    _prepare_run_dir(run_dir)
    _setup_logging(run_dir / "logs" / "train.log", args.log_level)

    LOGGER.info("starting formal DiffGui docking training")
    LOGGER.info("config_path=%s", args.config.resolve())
    LOGGER.info("run_dir=%s", run_dir)
    LOGGER.info("git=%s", _git_snapshot())
    LOGGER.info("conda_prefix=%s python=%s", os.environ.get("CONDA_PREFIX", ""), sys.executable)
    LOGGER.info("cuda_visible_devices=%s", os.environ.get("CUDA_VISIBLE_DEVICES", ""))

    (run_dir / "config" / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    train_cfg = config.get("diffgui_training", {})
    sweep_cfg = config.get("diffgui_docking_batch_sweep", {})
    model_cfg = config.get("diffgui_model", {})
    epochs = int(train_cfg.get("epochs", 1))
    batch_size = int(train_cfg.get("batch_size", 4))
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("diffgui_training.epochs and batch_size must be positive.")

    device = torch.device(str(sweep_cfg.get("device", "cuda:0")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false in this process.")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    model, featurizer, sample_cfg = _load_diffgui_model(
        sample_config=_resolve_path(model_cfg.get("sample_config", DIFFGUI_ROOT / "configs/sample/sample.yml")),
        checkpoint=_resolve_path(model_cfg.get("checkpoint", DIFFGUI_ROOT / "ckpt/trained.pt")),
        num_timesteps=int(sweep_cfg.get("num_timesteps", 100)),
        device=device,
    )
    actor = DiffGuiActor(model, _build_actor_config(config))
    LOGGER.info("actor_freeze_report=%s", actor.freeze_report)

    metrics_rows: list[dict[str, Any]] = []
    metrics_csv = run_dir / "metrics" / "metrics.csv"
    start_time = time.perf_counter()
    for epoch in range(1, epochs + 1):
        row = _run_epoch(
            config=config,
            model=model,
            actor=actor,
            featurizer=featurizer,
            sample_cfg=sample_cfg,
            epoch=epoch,
            batch_size=batch_size,
            run_dir=run_dir,
        )
        row["elapsed_train_seconds"] = float(time.perf_counter() - start_time)
        metrics_rows.append(row)
        _write_metrics_csv(metrics_csv, metrics_rows)
        _write_summary_plot(run_dir / "metrics" / "training_summary.png", metrics_rows)
        if _should_save_checkpoint(config, epoch):
            checkpoint_path = _save_checkpoint(run_dir, epoch, model, actor, config, row)
            row["checkpoint"] = str(checkpoint_path)
            _write_metrics_csv(metrics_csv, metrics_rows)
        LOGGER.info(
            "epoch=%d/%d step=%d reward_mean=%.4f recon=%d/%d docking_total=%.3f ppo_loss=%.4f elapsed=%.1fs",
            epoch,
            epochs,
            int(row.get("global_step", epoch)),
            float(row.get("reward_total_mean", 0.0)),
            int(row.get("reconstruction_success", 0)),
            int(row.get("reconstruction_attempted", 0)),
            float(row.get("docking_success_ratio_total", row.get("docking_success_ratio", 0.0))),
            float(row.get("PPO_Loss", 0.0)),
            float(row["elapsed_train_seconds"]),
        )

    summary = {
        "stage": "diffgui_docking_formal_train",
        "run_dir": str(run_dir),
        "epochs": epochs,
        "batch_size": batch_size,
        "metrics_csv": str(metrics_csv),
        "final_metrics": metrics_rows[-1] if metrics_rows else {},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("training complete summary=%s", json.dumps(summary, ensure_ascii=False))


def _apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Apply CLI overrides to a loaded config."""
    training_cfg = config.setdefault("diffgui_training", {})
    sweep_cfg = config.setdefault("diffgui_docking_batch_sweep", {})
    reward_cfg = config.setdefault("reward", {})
    docking_cfg = reward_cfg.setdefault("docking", {})
    if args.epochs is not None:
        training_cfg["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        training_cfg["batch_size"] = int(args.batch_size)
    if args.device is not None:
        sweep_cfg["device"] = str(args.device)
        device_index = _cuda_index_from_device(args.device)
        if args.docking_gpu_id is None and device_index is not None:
            docking_cfg["gpu_id"] = int(device_index)
    if args.docking_gpu_id is not None:
        docking_cfg["gpu_id"] = int(args.docking_gpu_id)
    return config


def _cuda_index_from_device(value: str) -> int | None:
    """Parse a physical CUDA index from a device string."""
    device = str(value).strip().lower()
    if device == "cuda":
        return 0
    if not device.startswith("cuda:"):
        return None
    try:
        return int(device.split(":", 1)[1])
    except ValueError:
        return None


def _resolve_run_dir(config: dict[str, Any], output_dir: Path | None) -> Path:
    """Resolve the run directory."""
    if output_dir is not None:
        return output_dir.resolve()
    train_cfg = config.get("diffgui_training", {})
    output_root = _resolve_path(train_cfg.get("output_root", "OUT"))
    run_name = str(config.get("name", "diffgui-docking-train")).replace("/", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_root / f"{run_name}_{timestamp}"


def _prepare_run_dir(run_dir: Path) -> None:
    """Create standard run subdirectories."""
    for subdir in ["logs", "metrics", "checkpoints", "epochs", "config"]:
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)


def _setup_logging(log_path: Path, log_level: str) -> None:
    """Configure readable terminal and durable file logging."""
    LOGGER.setLevel(getattr(logging, log_level))
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(stream_handler)
    LOGGER.addHandler(file_handler)


def _build_actor_config(config: dict[str, Any]) -> dict[str, Any]:
    """Build the DiffGuiActor config from the formal YAML."""
    train_cfg = dict(config.get("train", {}))
    sweep_cfg = config.get("diffgui_docking_batch_sweep", {})
    train_cfg.setdefault("precision", sweep_cfg.get("precision", "bf16"))
    train_cfg.setdefault("graph_micro_batch_size", sweep_cfg.get("graph_micro_batch_size", 2))
    return {"train": train_cfg}


def _run_epoch(
    *,
    config: dict[str, Any],
    model: torch.nn.Module,
    actor: DiffGuiActor,
    featurizer: Any,
    sample_cfg: Any,
    epoch: int,
    batch_size: int,
    run_dir: Path,
) -> dict[str, Any]:
    """Run one on-policy DiffGui docking PPO epoch."""
    sweep_cfg = config.get("diffgui_docking_batch_sweep", {})
    model_cfg = config.get("diffgui_model", {})
    epoch_dir = run_dir / "epochs" / f"epoch_{epoch:04d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    reward_batch_size = int(sweep_cfg.get("reward_batch_size") or batch_size)
    seed = int(sweep_cfg.get("seed_base", 260609)) + epoch
    _seed_diffgui(seed)
    if device.type == "cuda":
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
            batch_size=batch_size,
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
            n_graphs=batch_size,
            batch_node=batch_node,
            halfedge_index=halfedge_index,
            batch_halfedge=batch_halfedge,
            output_dir=epoch_dir,
            name_prefix=f"diffgui_train_e{epoch:04d}",
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
        advantages = _make_advantages(
            rewards_by_graph,
            normalize=bool(sweep_cfg.get("normalize_advantages", True)),
            epsilon=float(sweep_cfg.get("advantage_epsilon", 1.0e-6)),
            clamp_min=sweep_cfg.get("advantage_clip_min", None),
            clamp_max=sweep_cfg.get("advantage_clip_max", None),
        )
        policy_batch = _replace_policy_advantages(policy_batch, advantages)

        phase_start = time.perf_counter()
        actor_metrics = actor.update_policy(policy_batch)
        timing["ppo_seconds"] = time.perf_counter() - phase_start

        total_seconds = time.perf_counter() - started
        memory = _stop_gpu_memory_monitor(monitor)
        row = _build_batch_summary_row(
            config=config,
            batch_size=batch_size,
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
        row["epoch"] = int(epoch)
        row["global_step"] = int(epoch)
        row["epoch_dir"] = str(epoch_dir)
        (epoch_dir / "summary.json").write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
        return row
    finally:
        if monitor is not None and not monitor.stop_event.is_set():
            _stop_gpu_memory_monitor(monitor)


def _write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a stable union-schema metrics CSV."""
    fieldnames = ["epoch", "global_step", *_union_fieldnames(rows), "checkpoint", "elapsed_train_seconds"]
    deduped: list[str] = []
    for field in fieldnames:
        if field not in deduped:
            deduped.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=deduped)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in deduped})


def _should_save_checkpoint(config: dict[str, Any], epoch: int) -> bool:
    """Return whether a checkpoint should be saved after this epoch."""
    every = int(config.get("diffgui_training", {}).get("checkpoint_every_epochs", 1))
    return every > 0 and epoch % every == 0


def _save_checkpoint(
    run_dir: Path,
    epoch: int,
    model: torch.nn.Module,
    actor: DiffGuiActor,
    config: dict[str, Any],
    metrics: dict[str, Any],
) -> Path:
    """Save a formal DiffGui training checkpoint."""
    checkpoint_path = run_dir / "checkpoints" / f"diffgui-docking-epoch={epoch:04d}.pt"
    payload = {
        "stage": "diffgui_docking_formal_train",
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": actor.optimizer.state_dict(),
        "config": config,
        "metrics": metrics,
        "freeze_report": actor.freeze_report,
    }
    torch.save(payload, checkpoint_path)
    latest_path = run_dir / "checkpoints" / "latest.pt"
    torch.save(payload, latest_path)
    LOGGER.info("checkpoint saved epoch=%d path=%s", epoch, checkpoint_path)
    return checkpoint_path


if __name__ == "__main__":
    main()
