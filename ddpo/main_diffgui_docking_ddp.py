"""DDP entrypoint for decoupled DiffGui docking PPO training."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import statistics
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import numpy as np
import yaml
from rdkit import Chem
from torch.nn.parallel import DistributedDataParallel as DDP


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIFFGUI_ROOT = PROJECT_ROOT / "reference project" / "DiffGui"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(DIFFGUI_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFGUI_ROOT))

from ddpo.diffgui.deppa import (  # noqa: E402
    build_local_deppa_records,
    compute_deppa_advantages,
    is_deppa_mode_enabled,
)
from ddpo.diffgui.runtime.actor import DiffGuiActor  # noqa: E402
from ddpo.diffgui.runtime.good_pool import DiffGuiGoodMoleculePool, GoodPoolSelection  # noqa: E402
from ddpo.diffgui.runtime.per_timestep_diagnostics import (  # noqa: E402
    export_per_timestep,
    merge_per_timestep,
)
from ddpo.diffgui.runtime.replay import (  # noqa: E402
    DiffGuiTrajectoryReplayBuffer,
    build_train_batch_with_replay,
)
from ddpo.diffgui.runtime.rollout import DiffGuiRollout  # noqa: E402
from ddpo.diffgui.runtime.structural_stats import compute_structural_metrics  # noqa: E402
from ddpo.diffgui.runtime.training_filter import (  # noqa: E402
    DiffGuiTrainingFilter,
    annotate_molecule_rows,
)
from ddpo.main_diffgui_docking_train import (  # noqa: E402
    _build_actor_config,
    _prepare_run_dir,
    _should_save_checkpoint,
    _write_metrics_csv,
)
from ddpo.utils.molecular_descriptors import (  # noqa: E402
    compute_molecular_descriptors,
    load_chembl_6d_vectors,
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
    _torch_load_checkpoint,
    _torch_peak_memory_mb,
    _write_reconstructed_outputs,
    _write_summary_plot,
    write_molecule_records_csv,
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
            _write_run_manifest(args, config, run_dir, dist_info, status="running")
        _barrier()

        rows = _run_training(args=args, config=config, run_dir=run_dir, dist_info=dist_info)
        if dist_info["rank"] == 0:
            train_cfg = config.get("diffgui_training", {})
            summary = {
                "stage": "diffgui_docking_ddp_train",
                "run_dir": str(run_dir),
                "world_size": int(dist_info["world_size"]),
                "epochs": int(train_cfg.get("epochs", 1)),
                "epoch_offset": int(train_cfg.get("epoch_offset", 0)),
                "first_epoch": int(rows[0].get("epoch", 0)) if rows else 0,
                "final_epoch": int(rows[-1].get("epoch", 0)) if rows else 0,
                "per_rank_batch_size": int(train_cfg.get("batch_size", 1)),
                "global_batch_size": int(train_cfg.get("batch_size", 1))
                * int(dist_info["world_size"]),
                "metrics_csv": str(run_dir / "metrics" / "metrics.csv"),
                "final_metrics": rows[-1] if rows else {},
            }
            (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
            _write_run_manifest(args, config, run_dir, dist_info, status="completed", final_summary=summary)
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
    epoch_offset = int(train_cfg.get("epoch_offset", 0))
    per_rank_batch = int(train_cfg.get("batch_size", 1))
    if epochs <= 0 or per_rank_batch <= 0:
        raise ValueError("diffgui_training.epochs and batch_size must be positive.")
    if epoch_offset < 0:
        raise ValueError("diffgui_training.epoch_offset must be non-negative.")

    device = torch.device(f"cuda:{dist_info['local_rank']}" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("DiffGui DDP currently requires CUDA.")
    torch.cuda.set_device(device)
    rank_config = _rank_config(config, dist_info, device)
    checkpoint_path = _resolve_path(model_cfg.get("checkpoint", DIFFGUI_ROOT / "ckpt/trained.pt"))
    model, featurizer, sample_cfg = _load_diffgui_model(
        sample_config=_resolve_path(model_cfg.get("sample_config", DIFFGUI_ROOT / "configs/sample/sample.yml")),
        checkpoint=checkpoint_path,
        num_timesteps=int(sweep_cfg.get("num_timesteps", 100)),
        device=device,
    )
    DiffGuiActor._apply_freeze_policy(model, config.get("train", {}).get("freeze", {}))
    if dist_info["world_size"] > 1:
        training_model: torch.nn.Module = DDP(
            model,
            device_ids=[dist_info["local_rank"]],
            output_device=dist_info["local_rank"],
            find_unused_parameters=bool(config.get("train", {}).get("ddp_find_unused_parameters", False)),
        )
    else:
        training_model = model
    actor = DiffGuiActor(training_model, _build_actor_config(config))
    _restore_actor_optimizer_state(actor=actor, checkpoint=checkpoint_path, device=device)
    training_filter = DiffGuiTrainingFilter(config.get("diffgui_training_filter", {}) or {})
    good_pool = DiffGuiGoodMoleculePool(
        run_dir / str((config.get("diffgui_good_molecule_pool", {}) or {}).get("subdir", "good_molecule_pool")),
        config.get("diffgui_good_molecule_pool", {}) or {},
        rank=int(dist_info["rank"]),
    )
    replay_buffer = DiffGuiTrajectoryReplayBuffer(
        run_dir / str((config.get("diffgui_trajectory_replay", {}) or {}).get("subdir", "trajectory_replay")),
        {
            **(config.get("diffgui_trajectory_replay", {}) or {}),
            "seed": int(config.get("diffgui_docking_batch_sweep", {}).get("seed_base", 260609)),
        },
        rank=int(dist_info["rank"]),
    )
    LOGGER.info("actor_freeze_report=%s", actor.freeze_report)
    LOGGER.info(
        "diffgui_training_filter_enabled=%s good_pool_enabled=%s trajectory_replay_enabled=%s store_enabled=%s",
        training_filter.enabled,
        good_pool.enabled,
        replay_buffer.enabled,
        replay_buffer.store_enabled,
    )

    metrics_rows: list[dict[str, Any]] = []
    metrics_csv = run_dir / "metrics" / "metrics.csv"
    start_time = time.perf_counter()
    for local_epoch in range(1, epochs + 1):
        epoch = epoch_offset + local_epoch
        local_row = _run_epoch(
            config=rank_config,
            model=training_model,
            actor=actor,
            training_filter=training_filter,
            good_pool=good_pool,
            replay_buffer=replay_buffer,
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
            row.update(_merge_epoch_artifacts(run_dir, epoch=epoch, world_size=dist_info["world_size"]))
            metrics_rows.append(row)
            _write_metrics_csv(metrics_csv, metrics_rows)
            _write_metrics_csv(run_dir / "training_summary.csv", metrics_rows)
            _write_summary_plot(run_dir / "metrics" / "training_summary.png", metrics_rows)
            if _should_save_checkpoint(config, epoch):
                checkpoint_model = training_model.module if hasattr(training_model, "module") else training_model
                checkpoint_path = _save_checkpoint(run_dir, epoch, checkpoint_model, actor, config, row, dist_info)
                row["checkpoint"] = str(checkpoint_path)
                _write_metrics_csv(metrics_csv, metrics_rows)
                _write_metrics_csv(run_dir / "training_summary.csv", metrics_rows)
            _write_epoch_status(run_dir, epoch, row)
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
    model: torch.nn.Module,
    actor: DiffGuiActor,
    training_filter: DiffGuiTrainingFilter,
    good_pool: DiffGuiGoodMoleculePool,
    replay_buffer: DiffGuiTrajectoryReplayBuffer,
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

        deppa_result = None
        if is_deppa_mode_enabled(config):
            local_deppa_records = build_local_deppa_records(
                reconstruction_rows=reconstruction_rows,
                reward_rows=reward_rows,
                rank=rank,
                config=config.get("diffgui_deppa_mode", {}) or {},
            )
            gathered_deppa_records = _all_gather_object(local_deppa_records, int(dist_info["world_size"]))
            deppa_result = compute_deppa_advantages(
                local_records=local_deppa_records,
                gathered_records=gathered_deppa_records,
                config=config.get("diffgui_deppa_mode", {}) or {},
            )
            rewards_by_graph = deppa_result.rewards_by_graph
            advantages = deppa_result.advantages
        else:
            rewards_by_graph = _rewards_by_graph_index(
                reconstruction_rows=reconstruction_rows,
                reward_rows=reward_rows,
                failure_penalty=float(sweep_cfg.get("failure_advantage", -1.0)),
            )
            advantages = _make_global_advantages(rewards_by_graph, sweep_cfg=sweep_cfg, dist_info=dist_info)
        policy_batch = _replace_policy_advantages(policy_batch, advantages)
        molecule_rows = write_molecule_records_csv(
            epoch_dir / "molecules_with_rewards.csv",
            reconstruction_rows=reconstruction_rows,
            reward_rows=reward_rows,
            rewards_by_graph=rewards_by_graph,
            advantages=advantages,
            epoch=epoch,
            rank=rank,
            seed=seed,
        )
        if deppa_result is not None:
            molecule_rows = _merge_molecule_deppa_fields(molecule_rows, deppa_result.fields_by_graph)
            _write_csv_rows(
                epoch_dir / "molecules_with_rewards.csv",
                molecule_rows,
                fieldnames=_ordered_fieldnames(molecule_rows),
            )

        filter_selection = training_filter.select(molecule_rows, batch_size=per_rank_batch)
        filter_selection = training_filter.crop_for_ddp(filter_selection, device=device)
        if good_pool.enabled or replay_buffer.store_enabled:
            good_selection = good_pool.add_epoch(
                molecule_rows=molecule_rows,
                training_selection=filter_selection,
                epoch=epoch,
                generated_sdf=epoch_dir / "generated.sdf",
                top_sdf_path=epoch_dir / "good_pool_selected.sdf",
            )
        else:
            good_selection = GoodPoolSelection(
                metrics={
                    "good_pool_enabled": 0.0,
                    "good_pool_selected_count": 0.0,
                    "good_pool_retained_count": 0.0,
                }
            )
        replay_store_metrics = replay_buffer.store(
            policy_batch=policy_batch,
            selection=good_selection,
            epoch=epoch,
        )
        molecule_rows = annotate_molecule_rows(
            molecule_rows,
            filter_selection,
            good_indices=set(good_selection.indices),
        )
        _write_csv_rows(
            epoch_dir / "molecules_with_rewards.csv",
            molecule_rows,
            fieldnames=_ordered_fieldnames(molecule_rows),
        )
        _write_csv_rows(
            epoch_dir / "training_filter_decisions.csv",
            filter_selection.decision_rows(),
            fieldnames=_ordered_fieldnames(filter_selection.decision_rows()),
        )
        if filter_selection.train_indices:
            train_batch = policy_batch.select_graphs(filter_selection.train_indices)
            train_batch, replay_train_metrics = build_train_batch_with_replay(
                current_batch=train_batch,
                replay_buffer=replay_buffer,
                epoch=epoch,
                ddp_device=device,
            )
        else:
            train_batch = None
            replay_train_metrics = {
                "trajectory_replay_enabled": float(replay_buffer.enabled),
                "trajectory_replay_current_count": 0.0,
                "trajectory_replay_sample_count": 0.0,
                "trajectory_replay_train_total_count": 0.0,
            }

        phase_start = time.perf_counter()
        if train_batch is None:
            actor_metrics = {
                "PPO_Loss": 0.0,
                "PPO_ApproxKL": 0.0,
                "ClipFrac": 0.0,
                "RatioMean": 0.0,
                "TrainBatchSize": 0.0,
                "PPO_Skipped": 1.0,
            }
            actor.last_step_metrics = []
        else:
            actor_metrics = actor.update_policy(train_batch)
            actor_metrics["PPO_Skipped"] = 0.0
        actor_metrics.update(filter_selection.metrics)
        actor_metrics.update(good_selection.metrics)
        actor_metrics.update(replay_store_metrics)
        actor_metrics.update(replay_train_metrics)
        actor_metrics.update(replay_buffer.observe_update_metrics(actor_metrics, epoch=epoch))
        timing["ppo_seconds"] = time.perf_counter() - phase_start
        step_metrics_path = _write_step_metrics_csv(
            epoch_dir / "ppo_step_metrics.csv",
            actor.last_step_metrics,
            epoch=epoch,
            rank=rank,
        )
        export_per_timestep(
            epoch_dir, actor.last_step_metrics, epoch=epoch, rank=rank,
        )

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
        if deppa_result is not None:
            row.update(deppa_result.summary)
        row.update(
            {
                "rank": rank,
                "local_rank": int(dist_info["local_rank"]),
                "world_size": int(dist_info["world_size"]),
                "global_batch_size": int(per_rank_batch) * int(dist_info["world_size"]),
                "epoch": int(epoch),
                "global_step": int(epoch),
                "epoch_dir": str(epoch_dir),
                "molecule_records": int(len(molecule_rows)),
                "outputs_molecule_csv": str(epoch_dir / "molecules_with_rewards.csv"),
                "outputs_training_filter_csv": str(epoch_dir / "training_filter_decisions.csv"),
                "outputs_ppo_step_metrics": str(step_metrics_path),
                "outputs_good_pool_manifest": str(good_pool.manifest_path),
                "outputs_good_pool_selected_sdf": str(epoch_dir / "good_pool_selected.sdf"),
                "outputs_trajectory_replay_manifest": str(replay_buffer.manifest_path),
            }
        )
        (epoch_dir / "summary.json").write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
        return row
    finally:
        if monitor is not None and not monitor.stop_event.is_set():
            _stop_gpu_memory_monitor(monitor)


def _write_step_metrics_csv(path: Path, rows: list[dict[str, float]], *, epoch: int, rank: int) -> Path:
    """Write per-timestep PPO diagnostics for one rank."""
    path.parent.mkdir(parents=True, exist_ok=True)
    enriched: list[dict[str, Any]] = []
    for row in rows:
        enriched.append({"epoch": int(epoch), "rank": int(rank), **row})
    fieldnames = _ordered_fieldnames(
        enriched,
        preferred=[
            "epoch",
            "rank",
            "ppo_epoch",
            "microbatch_index",
            "step_idx",
            "timestep_mean",
            "train_batch_size",
            "graph_loss_weight",
            "clipfrac",
            "approx_kl",
            "ratio_mean",
            "logprob_current_mean",
            "logprob_old_mean",
            "logprob_pos_mean",
            "logprob_node_mean",
            "logprob_halfedge_mean",
        ],
    )
    _write_csv_rows(path, enriched, fieldnames=fieldnames)
    return path


def _merge_epoch_artifacts(run_dir: Path, *, epoch: int, world_size: int) -> dict[str, Any]:
    """Merge rank-local molecule and PPO artifacts into epoch-level files."""
    epoch_dir = run_dir / "epochs" / f"epoch_{epoch:04d}"
    molecule_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    kl_rows: list[dict[str, Any]] = []
    clipfrac_rows: list[dict[str, Any]] = []
    ratio_rows: list[dict[str, Any]] = []
    generated_sdfs: list[Path] = []
    scored_sdfs: list[Path] = []
    for rank in range(int(world_size)):
        rank_dir = epoch_dir / f"rank_{rank:02d}"
        molecule_rows.extend(_read_csv_rows(rank_dir / "molecules_with_rewards.csv"))
        step_rows.extend(_read_csv_rows(rank_dir / "ppo_step_metrics.csv"))
        kl_rows.extend(_read_csv_rows(rank_dir / "kl_by_timestep.csv"))
        clipfrac_rows.extend(_read_csv_rows(rank_dir / "clipfrac_by_timestep.csv"))
        ratio_rows.extend(_read_csv_rows(rank_dir / "ratio_by_timestep.csv"))
        generated_sdf = rank_dir / "generated.sdf"
        if generated_sdf.exists():
            generated_sdfs.append(generated_sdf)
        scored_sdf = rank_dir / "docking_scored.sdf"
        if scored_sdf.exists():
            scored_sdfs.append(scored_sdf)

    molecule_rows = _annotate_duplicate_fields(molecule_rows)
    molecules_csv = epoch_dir / "molecules.csv"
    _write_csv_rows(molecules_csv, molecule_rows, fieldnames=_ordered_fieldnames(molecule_rows))

    failures = [row for row in molecule_rows if not _is_full_success(row)]
    failures_csv = epoch_dir / "failures.csv"
    _write_csv_rows(failures_csv, failures, fieldnames=_ordered_fieldnames(failures or molecule_rows))

    ppo_step_csv = epoch_dir / "ppo_step_metrics.csv"
    _write_csv_rows(ppo_step_csv, step_rows, fieldnames=_ordered_fieldnames(step_rows))

    # Merge per-timestep diagnostics across ranks
    kl_merged_csv = merge_per_timestep(kl_rows, epoch_dir / "kl_by_timestep.csv", world_size)
    cf_merged_csv = merge_per_timestep(clipfrac_rows, epoch_dir / "clipfrac_by_timestep.csv", world_size)
    ratio_merged_csv = merge_per_timestep(ratio_rows, epoch_dir / "ratio_by_timestep.csv", world_size)

    generated_merged = _merge_sdf_files(generated_sdfs, epoch_dir / "generated_merged.sdf")
    scored_merged = _merge_sdf_files(scored_sdfs, epoch_dir / "molecules.sdf")
    top_reward_sdf = _write_top_reward_sdf(scored_sdfs, epoch_dir / "top_reward.sdf", top_k=50)

    metrics = _compute_molecule_epoch_metrics(molecule_rows)
    metrics.update(
        {
            "epoch_molecules_csv": str(molecules_csv),
            "epoch_failures_csv": str(failures_csv),
            "epoch_ppo_step_metrics_csv": str(ppo_step_csv),
            "epoch_kl_by_timestep_csv": str(kl_merged_csv) if kl_merged_csv else "",
            "epoch_clipfrac_by_timestep_csv": str(cf_merged_csv) if cf_merged_csv else "",
            "epoch_ratio_by_timestep_csv": str(ratio_merged_csv) if ratio_merged_csv else "",
            "epoch_generated_merged_sdf": str(generated_merged) if generated_merged else "",
            "epoch_molecules_sdf": str(scored_merged) if scored_merged else "",
            "epoch_top_reward_sdf": str(top_reward_sdf) if top_reward_sdf else "",
        }
    )
    return metrics


def _annotate_duplicate_fields(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add duplicate/scaffold fields to merged molecule rows."""
    smiles_counts = Counter(row.get("canonical_smiles", "") for row in rows if row.get("canonical_smiles"))
    scaffold_counts = Counter(row.get("murcko_scaffold", "") for row in rows if row.get("murcko_scaffold"))
    annotated: list[dict[str, Any]] = []
    for row in rows:
        out = dict(row)
        smiles = str(out.get("canonical_smiles", ""))
        scaffold = str(out.get("murcko_scaffold", ""))
        out["duplicate_count"] = int(smiles_counts.get(smiles, 0)) if smiles else 0
        out["is_duplicate"] = int(bool(smiles) and smiles_counts.get(smiles, 0) > 1)
        out["scaffold_count"] = int(scaffold_counts.get(scaffold, 0)) if scaffold else 0
        annotated.append(out)
    return annotated


def _is_full_success(row: dict[str, Any]) -> bool:
    """Return whether a molecule passed reconstruction, sanitize, and docking."""
    return (
        _row_int(row, "reconstruction_success") == 1
        and _row_int(row, "rdkit_valid") == 1
        and _row_int(row, "docking_success") == 1
    )


def _merge_sdf_files(sources: list[Path], output_path: Path) -> Path | None:
    """Merge readable molecules from several SDF files."""
    mols: list[Chem.Mol] = []
    for source in sources:
        supplier = Chem.SDMolSupplier(str(source), sanitize=False, removeHs=False)
        mols.extend(mol for mol in supplier if mol is not None)
    if not mols:
        return None
    writer = Chem.SDWriter(str(output_path))
    writer.SetKekulize(False)
    try:
        for mol in mols:
            writer.write(mol)
    finally:
        writer.close()
    return output_path


def _write_top_reward_sdf(sources: list[Path], output_path: Path, *, top_k: int) -> Path | None:
    """Write top reward molecules from scored rank SDFs."""
    mols: list[tuple[float, Chem.Mol]] = []
    for source in sources:
        supplier = Chem.SDMolSupplier(str(source), sanitize=False, removeHs=False)
        for mol in supplier:
            if mol is None:
                continue
            reward = _row_float({key: mol.GetProp(key) for key in mol.GetPropNames()}, "reward_total", default=-float("inf"))
            mols.append((reward, mol))
    if not mols:
        return None
    mols.sort(key=lambda item: item[0], reverse=True)
    writer = Chem.SDWriter(str(output_path))
    writer.SetKekulize(False)
    try:
        for _, mol in mols[: max(1, int(top_k))]:
            writer.write(mol)
    finally:
        writer.close()
    return output_path


def _compute_molecule_epoch_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute epoch-level molecule/failure/descriptor metrics."""
    total = len(rows)
    reconstructed = sum(_row_int(row, "reconstruction_success") for row in rows)
    rdkit_valid = sum(_row_int(row, "rdkit_valid") for row in rows)
    docking_success = sum(_row_int(row, "docking_success") for row in rows)
    canonical = [str(row.get("canonical_smiles", "")) for row in rows if row.get("canonical_smiles")]
    unique_canonical = len(set(canonical))
    duplicate_ratio = float((len(canonical) - unique_canonical) / len(canonical)) if canonical else 0.0
    scaffolds = [str(row.get("murcko_scaffold", "")) for row in rows if row.get("murcko_scaffold")]
    scaffold_counts = Counter(scaffolds)
    failure_counts = _failure_counts(rows)
    metrics: dict[str, Any] = {
        "molecule_records_total": int(total),
        "molecule_reconstruction_success": int(reconstructed),
        "molecule_reconstruction_success_ratio": float(reconstructed / total) if total else 0.0,
        "molecule_rdkit_valid": int(rdkit_valid),
        "molecule_rdkit_valid_ratio": float(rdkit_valid / total) if total else 0.0,
        "molecule_docking_success": int(docking_success),
        "molecule_docking_success_ratio": float(docking_success / total) if total else 0.0,
        "unique_canonical_smiles": int(unique_canonical),
        "duplicate_ratio": float(duplicate_ratio),
        "unique_scaffolds": int(len(scaffold_counts)),
        "scaffold_top1_fraction": float(max(scaffold_counts.values()) / len(scaffolds)) if scaffolds else 0.0,
        "failure_counts": json.dumps(failure_counts, ensure_ascii=False, sort_keys=True),
    }
    metrics.update(_continuous_reward_metrics(rows))
    # Descriptor + structural stats via shared modules
    rdmols: list[Chem.Mol | None] = []
    valid_mask: list[bool] = []
    for row in rows:
        smiles = str(row.get("canonical_smiles") or row.get("smiles") or "")
        is_valid = _row_int(row, "rdkit_valid") == 1 and bool(smiles)
        valid_mask.append(is_valid)
        rdmols.append(Chem.MolFromSmiles(smiles) if is_valid else None)
    try:
        ref = load_chembl_6d_vectors(project_root=PROJECT_ROOT)
        if ref is not None and len(ref) == 0:
            ref = None
    except Exception:
        ref = None
    metrics.update(compute_molecular_descriptors(rdmols, valid_mask, reference_descriptor_vectors=ref))
    valid_mols = [m for m in rdmols if m is not None]
    metrics.update(compute_structural_metrics(valid_mols))
    return metrics


def _continuous_reward_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Summarize continuous reward fields that decide docking trends."""
    return {
        "reward_for_advantage_mean": _float_mean(rows, "reward_for_advantage"),
        "reward_total_scored_mean": _float_mean(rows, "reward_total"),
        "docking_raw_mean": _float_mean(rows, "docking_raw"),
        "docking_reward_mean": _float_mean(rows, "docking_reward"),
        "ligand_efficiency_raw_mean": _float_mean(rows, "ligand_efficiency_raw"),
    }


def _float_mean(rows: list[dict[str, Any]], key: str) -> float:
    """Return the mean of present finite float values in CSV rows."""
    values: list[float] = []
    for row in rows:
        raw = row.get(key, "")
        if raw == "":
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if value == value:
            values.append(value)
    return float(sum(values) / len(values)) if values else 0.0


def _failure_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Build clear failure counts from molecule rows."""
    counts: Counter[str] = Counter()
    for row in rows:
        if _row_int(row, "reconstruction_success") != 1:
            counts[f"reconstruction:{row.get('reconstruction_status', 'unknown')}"] += 1
            continue
        if _row_int(row, "rdkit_valid") != 1:
            counts[f"sanitize:{row.get('sanitize_status', 'unknown')}"] += 1
            continue
        if _row_int(row, "docking_success") != 1:
            counts["docking:failed_or_zero"] += 1
            continue
        counts["success"] += 1
    return dict(counts)



def _write_epoch_status(run_dir: Path, epoch: int, row: dict[str, Any]) -> None:
    """Write a per-epoch status JSON similar to the DiffSBDD recorder."""
    epoch_dir = run_dir / "epochs" / f"epoch_{epoch:04d}"
    payload = {
        "epoch": int(epoch),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "metrics": row,
        "molecules_csv": row.get("epoch_molecules_csv", ""),
        "failures_csv": row.get("epoch_failures_csv", ""),
        "ppo_step_metrics_csv": row.get("epoch_ppo_step_metrics_csv", ""),
        "checkpoint_path": row.get("checkpoint", ""),
    }
    (epoch_dir / "epoch_status.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    """Read CSV rows if the file exists."""
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv_rows(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str] | None = None) -> None:
    """Write CSV rows with a stable union schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fieldnames or _ordered_fieldnames(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fields})


def _merge_molecule_deppa_fields(
    molecule_rows: list[dict[str, Any]],
    fields_by_graph: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach DePPA audit fields to molecule rows by local graph index."""
    merged: list[dict[str, Any]] = []
    for row in molecule_rows:
        out = dict(row)
        graph_index = _row_int(out, "local_mol_index", default=-1)
        if graph_index in fields_by_graph:
            out.update(fields_by_graph[graph_index])
        merged.append(out)
    return merged


def _ordered_fieldnames(rows: list[dict[str, Any]], *, preferred: list[str] | None = None) -> list[str]:
    """Return preferred fields followed by dynamic fields."""
    ordered = list(preferred or [])
    for row in rows:
        for key in row:
            if key not in ordered:
                ordered.append(str(key))
    return ordered


def _csv_value(value: Any) -> Any:
    """Convert complex values into CSV-friendly scalars."""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _row_int(row: dict[str, Any], key: str, default: int = 0) -> int:
    """Read an int-ish CSV value."""
    try:
        return int(float(row.get(key, default) or default))
    except (TypeError, ValueError):
        return int(default)


def _row_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    """Read a finite float-ish CSV value."""
    try:
        value = float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return float(default)
    return value if value == value else float(default)


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
        "training_filter_enabled": _mean(rows, "training_filter_enabled"),
        "train_candidate_count": _sum(rows, "train_candidate_count"),
        "train_pre_ddp_selected_count": _sum(rows, "train_pre_ddp_selected_count"),
        "train_ddp_min_count": min(float(row.get("train_ddp_min_count", 0.0)) for row in rows) if rows else 0.0,
        "train_selected_count": _sum(rows, "train_selected_count"),
        "train_discarded_count": _sum(rows, "train_discarded_count"),
        "ppo_skipped_by_training_filter": max(float(row.get("ppo_skipped_by_training_filter", 0.0)) for row in rows)
        if rows
        else 0.0,
        "PPO_Skipped": max(float(row.get("PPO_Skipped", 0.0)) for row in rows) if rows else 0.0,
        "good_pool_enabled": _mean(rows, "good_pool_enabled"),
        "good_pool_selected_count": _sum(rows, "good_pool_selected_count"),
        "good_pool_selected_score_mean": _mean(rows, "good_pool_selected_score_mean"),
        "good_pool_selected_score_max": max(float(row.get("good_pool_selected_score_max", 0.0)) for row in rows)
        if rows
        else 0.0,
        "good_pool_retained_count": _sum(rows, "good_pool_retained_count"),
        "good_pool_retained_score_mean": _mean(rows, "good_pool_retained_score_mean"),
        "good_pool_retained_score_max": max(float(row.get("good_pool_retained_score_max", 0.0)) for row in rows)
        if rows
        else 0.0,
        "trajectory_replay_enabled": _mean(rows, "trajectory_replay_enabled"),
        "trajectory_replay_current_count": _sum(rows, "trajectory_replay_current_count"),
        "trajectory_replay_sample_count": _sum(rows, "trajectory_replay_sample_count"),
        "trajectory_replay_train_total_count": _sum(rows, "trajectory_replay_train_total_count"),
        "trajectory_replay_eligible_shards": _sum(rows, "trajectory_replay_eligible_shards"),
        "trajectory_replay_age_mean": _mean(rows, "trajectory_replay_age_mean"),
        "trajectory_replay_age_max": max(float(row.get("trajectory_replay_age_max", 0.0)) for row in rows)
        if rows
        else 0.0,
        "trajectory_replay_store_count": _sum(rows, "trajectory_replay_store_count"),
        "trajectory_replay_store_score_mean": _mean(rows, "trajectory_replay_store_score_mean"),
        "trajectory_replay_store_score_max": max(float(row.get("trajectory_replay_store_score_max", 0.0)) for row in rows)
        if rows
        else 0.0,
        "trajectory_replay_gate_enabled": _mean(rows, "trajectory_replay_gate_enabled"),
        "trajectory_replay_gate_observed": _mean(rows, "trajectory_replay_gate_observed"),
        "trajectory_replay_gate_violation": max(float(row.get("trajectory_replay_gate_violation", 0.0)) for row in rows)
        if rows
        else 0.0,
        "trajectory_replay_gate_reason_count": max(
            float(row.get("trajectory_replay_gate_reason_count", 0.0)) for row in rows
        )
        if rows
        else 0.0,
        "trajectory_replay_gate_ratio_before": _mean(rows, "trajectory_replay_gate_ratio_before"),
        "trajectory_replay_gate_ratio_after": _mean(rows, "trajectory_replay_gate_ratio_after"),
        "trajectory_replay_gate_violation_streak": max(
            float(row.get("trajectory_replay_gate_violation_streak", 0.0)) for row in rows
        )
        if rows
        else 0.0,
        "trajectory_replay_gate_recovery_streak": min(
            float(row.get("trajectory_replay_gate_recovery_streak", 0.0)) for row in rows
        )
        if rows
        else 0.0,
        "trajectory_replay_gate_approx_kl": _mean(rows, "trajectory_replay_gate_approx_kl"),
        "trajectory_replay_gate_clipfrac": _mean(rows, "trajectory_replay_gate_clipfrac"),
        "trajectory_replay_gate_ratio_mean": _mean(rows, "trajectory_replay_gate_ratio_mean"),
        "torch_peak_allocated_mb": max(float(row.get("torch_peak_allocated_mb", 0.0)) for row in rows) if rows else 0.0,
        "gpu_monitor_peak_used_mb": max(float(row.get("gpu_monitor_peak_used_mb", 0.0)) for row in rows) if rows else 0.0,
        "elapsed_train_seconds": float(elapsed),
    }
    row.update(_aggregate_prefixed_numeric_metrics(rows, "deppa_"))
    row["rank_rows_json"] = json.dumps(rows, ensure_ascii=False)
    return row


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if key in row and row[key] != ""]
    return float(sum(values) / len(values)) if values else 0.0


def _aggregate_prefixed_numeric_metrics(rows: list[dict[str, Any]], prefix: str) -> dict[str, float]:
    """Aggregate optional prefixed metrics that may be absent in legacy runs."""
    aggregated: dict[str, float] = {}
    keys = sorted({key for row in rows for key in row if str(key).startswith(prefix)})
    for key in keys:
        values: list[float] = []
        for row in rows:
            if key not in row or row[key] == "":
                continue
            try:
                values.append(float(row[key]))
            except (TypeError, ValueError):
                continue
        if not values:
            continue
        if key.endswith("_valid_count") or key.endswith("_invalid_count"):
            aggregated[key] = float(sum(values))
        elif key.endswith("_enabled"):
            aggregated[key] = float(max(values))
        elif key.endswith("_min"):
            aggregated[key] = float(min(values))
        elif key.endswith("_max"):
            aggregated[key] = float(max(values))
        else:
            aggregated[key] = float(sum(values) / len(values))
    return aggregated


def _restore_actor_optimizer_state(*, actor: DiffGuiActor, checkpoint: Path, device: torch.device) -> None:
    """Restore PPO optimizer state when the checkpoint was saved by this trainer."""
    ckpt = _torch_load_checkpoint(checkpoint, device)
    optimizer_state = ckpt.get("optimizer_state_dict")
    if optimizer_state is None:
        LOGGER.info("checkpoint has no optimizer_state_dict; using fresh optimizer checkpoint=%s", checkpoint)
        return
    try:
        actor.optimizer.load_state_dict(optimizer_state)
    except Exception as exc:
        raise RuntimeError(f"Failed to restore optimizer_state_dict from {checkpoint}") from exc
    _move_optimizer_state_to_device(actor.optimizer, device)
    LOGGER.info(
        "restored optimizer_state_dict checkpoint=%s checkpoint_epoch=%s",
        checkpoint,
        ckpt.get("epoch", ""),
    )


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """Move optimizer tensors to the rank-local training device."""
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _sum(rows: list[dict[str, Any]], key: str) -> float:
    """Return the numeric sum for a metric key across rank rows."""
    values = [float(row[key]) for row in rows if key in row and row[key] != ""]
    return float(sum(values)) if values else 0.0


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
    _append_checkpoint_manifest(run_dir, epoch, checkpoint_path, metrics)
    LOGGER.info("checkpoint saved epoch=%d path=%s", epoch, checkpoint_path)
    return checkpoint_path


def _append_checkpoint_manifest(run_dir: Path, epoch: int, checkpoint_path: Path, metrics: dict[str, Any]) -> None:
    """Append one checkpoint row to a manifest CSV."""
    manifest_path = run_dir / "checkpoints" / "checkpoint_manifest.csv"
    fieldnames = [
        "epoch",
        "path",
        "exists",
        "size_bytes",
        "saved_at",
        "reward_total_mean",
        "PPO_Loss",
        "PPO_ApproxKL",
        "reconstruction_success_ratio",
        "docking_success_ratio_total",
    ]
    exists = checkpoint_path.exists()
    row = {
        "epoch": int(epoch),
        "path": str(checkpoint_path),
        "exists": int(exists),
        "size_bytes": checkpoint_path.stat().st_size if exists else 0,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "reward_total_mean": metrics.get("reward_total_mean", ""),
        "PPO_Loss": metrics.get("PPO_Loss", ""),
        "PPO_ApproxKL": metrics.get("PPO_ApproxKL", ""),
        "reconstruction_success_ratio": metrics.get("reconstruction_success_ratio", ""),
        "docking_success_ratio_total": metrics.get("docking_success_ratio_total", ""),
    }
    file_exists = manifest_path.exists()
    with manifest_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _write_run_manifest(
    args: argparse.Namespace,
    config: dict[str, Any],
    run_dir: Path,
    dist_info: dict[str, int],
    *,
    status: str,
    final_summary: dict[str, Any] | None = None,
) -> None:
    """Write a durable run manifest for DiffGui DDP training."""
    payload: dict[str, Any] = {}
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
    payload.update(
        {
            "stage": "diffgui_docking_ddp_train",
            "status": str(status),
            "run_dir": str(run_dir),
            "config_path": str(args.config.resolve()),
            "world_size": int(dist_info["world_size"]),
            "per_rank_batch_size": int(config.get("diffgui_training", {}).get("batch_size", 1)),
            "global_batch_size": int(config.get("diffgui_training", {}).get("batch_size", 1))
            * int(dist_info["world_size"]),
            "seed_base": int(config.get("diffgui_docking_batch_sweep", {}).get("seed_base", 260609)),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "git": _git_snapshot(),
            "python": sys.executable,
            "conda_prefix": os.environ.get("CONDA_PREFIX", ""),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    payload.setdefault("created_at", payload["updated_at"])
    if final_summary is not None:
        payload["final_summary"] = final_summary
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


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
    local_rank = int(dist_info["local_rank"])
    phys_gpu = int(_physical_gpu_id(local_rank))
    rank_config = dict(config)
    sweep_cfg = dict(config.get("diffgui_docking_batch_sweep", {}))
    reward_cfg = dict(config.get("reward", {}))
    docking_cfg = dict(reward_cfg.get("docking", {}))
    sweep_cfg["device"] = str(device)
    sweep_cfg["reward_cuda_visible"] = str(phys_gpu)
    docking_cfg["gpu_id"] = int(phys_gpu)
    # Isolate classifier to the same GPU
    if "chembl_classifier" in reward_cfg:
        clf_cfg = dict(reward_cfg["chembl_classifier"])
        clf_cfg["device"] = f"cuda:{phys_gpu}"
        reward_cfg["chembl_classifier"] = clf_cfg
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
