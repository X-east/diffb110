"""Train the standalone x/h/e edge diffusion model with optional PyTorch DDP."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIFFSBDD_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_ROOT = Path(__file__).resolve().parent
for extra in (PROJECT_ROOT, DIFFSBDD_ROOT, ANALYSIS_ROOT):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from edge_diffusion import (  # noqa: E402
    EdgeBatchCollator,
    EdgeBundleDataset,
    EdgeCategoricalTransition,
    EdgeDiffusionLossConfig,
    EdgeDiffusionModel,
    EdgeDiffusionModelConfig,
    EdgeTransitionConfig,
    classification_metrics,
    make_edge_class_weight,
)
from train_edge_diffusion_overfit import (  # noqa: E402
    _count_atom_classes,
    _count_edge_classes,
    _evaluate,
    _infer_num_atom_types,
    _make_inverse_sqrt_weight,
    _plot_training_curves,
    _split_dataset,
    _train_one_epoch,
    _write_metrics_csv,
)


def run_training(args: argparse.Namespace) -> Path | None:
    """Run single-process or DDP edge diffusion training."""
    dist_state = _init_distributed(force_cpu=args.force_cpu)
    _set_seed(args.seed + dist_state.rank)
    if dist_state.is_main:
        args.outdir.mkdir(parents=True, exist_ok=True)
        _write_resolved_config(args.outdir / "resolved_config.json", args, dist_state)
    _barrier(dist_state)

    dataset = EdgeBundleDataset(
        edge_npz=args.edge_npz,
        semantic_npz=args.semantic_npz if args.semantic_npz.exists() else None,
        max_molecules=args.max_molecules,
    )
    train_dataset, eval_dataset, split_info = _split_dataset(dataset, args)
    train_sampler = _make_train_sampler(train_dataset, dist_state, args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=dist_state.device.type == "cuda",
        collate_fn=EdgeBatchCollator(),
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=dist_state.device.type == "cuda",
        collate_fn=EdgeBatchCollator(),
    )

    num_atom_types = _infer_num_atom_types(dataset)
    args.num_atom_types_inferred = num_atom_types
    edge_class_counts = _count_edge_classes(train_dataset, args.num_edge_types)
    eval_edge_class_counts = _count_edge_classes(eval_dataset, args.num_edge_types)
    atom_class_counts = _count_atom_classes(train_dataset, num_atom_types)
    edge_class_weight = make_edge_class_weight(
        edge_class_counts,
        mode=args.edge_class_weight_mode,
        no_bond_weight=args.no_bond_weight,
    ).to(dist_state.device)
    atom_class_weight = _make_inverse_sqrt_weight(atom_class_counts).to(dist_state.device)

    model = EdgeDiffusionModel(EdgeDiffusionModelConfig(
        num_atom_types=num_atom_types,
        num_edge_types=args.num_edge_types,
        num_timesteps=args.num_timesteps,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        distance_basis_dim=args.distance_basis_dim,
        dropout=args.dropout,
        coordinate_scale=args.coordinate_scale,
    )).to(dist_state.device)
    model_train: torch.nn.Module
    if dist_state.distributed:
        if dist_state.device.type == "cuda":
            model_train = DistributedDataParallel(
                model,
                device_ids=[dist_state.local_rank],
                output_device=dist_state.local_rank,
            )
        else:
            model_train = DistributedDataParallel(model)
    else:
        model_train = model

    optimizer = torch.optim.AdamW(model_train.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    transition = EdgeCategoricalTransition(EdgeTransitionConfig(
        num_classes=args.num_edge_types,
        num_timesteps=args.num_timesteps,
        prior=args.edge_prior,
    ))
    loss_config = EdgeDiffusionLossConfig(
        pos_weight=args.pos_weight,
        atom_weight=args.atom_weight,
        edge_weight=args.edge_weight,
        bond_length_weight=args.bond_length_weight,
    )

    metrics_rows: list[dict[str, float]] = []
    started_at = time.time()
    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_metrics = _train_one_epoch(
            model=model_train,
            dataloader=train_loader,
            optimizer=optimizer,
            transition=transition,
            loss_config=loss_config,
            atom_class_weight=atom_class_weight,
            edge_class_weight=edge_class_weight,
            args=args,
            device=dist_state.device,
            seed=args.seed + epoch + dist_state.rank * 100_000,
        )
        train_metrics = _reduce_metric_dict(train_metrics, dist_state)
        if dist_state.is_main:
            eval_metrics = _evaluate(
                model=model,
                dataloader=eval_loader,
                transition=transition,
                loss_config=loss_config,
                atom_class_weight=atom_class_weight,
                edge_class_weight=edge_class_weight,
                args=args,
                device=dist_state.device,
                seed=args.seed + 1000 + epoch,
            )
            row = {
                "epoch": float(epoch),
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"eval_{key}": value for key, value in eval_metrics.items()},
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
            metrics_rows.append(row)
            _write_metrics_csv(args.outdir / "edge_diffusion_ddp_metrics.csv", metrics_rows)
            if epoch % args.checkpoint_interval == 0 or epoch == args.epochs:
                _save_checkpoint(
                    path=args.outdir / f"edge_diffusion_ddp_epoch_{epoch:04d}.pt",
                    model=model,
                    args=args,
                    split_info=split_info,
                    metrics_rows=metrics_rows,
                    edge_class_counts=edge_class_counts,
                    eval_edge_class_counts=eval_edge_class_counts,
                    atom_class_counts=atom_class_counts,
                    elapsed_seconds=time.time() - started_at,
                    dist_state=dist_state,
                )
        _barrier(dist_state)

    report_path = None
    if dist_state.is_main:
        summary = _build_summary(
            args=args,
            split_info=split_info,
            metrics_rows=metrics_rows,
            edge_class_counts=edge_class_counts,
            eval_edge_class_counts=eval_edge_class_counts,
            atom_class_counts=atom_class_counts,
            elapsed_seconds=time.time() - started_at,
            dist_state=dist_state,
        )
        summary_path = args.outdir / "edge_diffusion_ddp_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        _plot_training_curves(args.outdir / "edge_diffusion_ddp_curves.png", metrics_rows)
        report_path = _write_report(args.outdir, summary)
    _cleanup_distributed(dist_state)
    return report_path


class DistributedState:
    """Small container for DDP process state."""

    def __init__(
        self,
        *,
        distributed: bool,
        rank: int,
        local_rank: int,
        world_size: int,
        device: torch.device,
    ) -> None:
        self.distributed = distributed
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = device

    @property
    def is_main(self) -> bool:
        """Return whether this process should write shared artifacts."""
        return self.rank == 0


def _init_distributed(force_cpu: bool) -> DistributedState:
    """Initialize torch.distributed from torchrun environment variables."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    cuda_available = torch.cuda.is_available() and not force_cpu
    if cuda_available:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    distributed = world_size > 1
    if distributed and not dist.is_initialized():
        dist.init_process_group(backend=backend)
    return DistributedState(
        distributed=distributed,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
    )


def _cleanup_distributed(dist_state: DistributedState) -> None:
    """Destroy the process group when this script created one."""
    if dist_state.distributed and dist.is_initialized():
        dist.destroy_process_group()


def _barrier(dist_state: DistributedState) -> None:
    """Synchronize DDP workers when distributed training is active."""
    if dist_state.distributed:
        dist.barrier()


def _make_train_sampler(
    train_dataset: torch.utils.data.Dataset,
    dist_state: DistributedState,
    seed: int,
) -> DistributedSampler | None:
    """Build a distributed train sampler only under DDP."""
    if not dist_state.distributed:
        return None
    return DistributedSampler(
        train_dataset,
        num_replicas=dist_state.world_size,
        rank=dist_state.rank,
        shuffle=True,
        seed=int(seed),
        drop_last=False,
    )


def _reduce_metric_dict(metrics: dict[str, float], dist_state: DistributedState) -> dict[str, float]:
    """Average scalar metric dictionaries across DDP ranks."""
    if not dist_state.distributed:
        return metrics
    keys = sorted(metrics)
    values = torch.tensor(
        [float(metrics[key]) for key in keys],
        dtype=torch.float64,
        device=dist_state.device,
    )
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values = values / float(dist_state.world_size)
    return {key: float(value) for key, value in zip(keys, values.detach().cpu().tolist())}


def _save_checkpoint(
    path: Path,
    model: EdgeDiffusionModel,
    args: argparse.Namespace,
    split_info: dict[str, object],
    metrics_rows: list[dict[str, float]],
    edge_class_counts: torch.Tensor,
    eval_edge_class_counts: torch.Tensor,
    atom_class_counts: torch.Tensor,
    elapsed_seconds: float,
    dist_state: DistributedState,
) -> None:
    """Write a rank-0 checkpoint compatible with the standalone sampler."""
    summary = _build_summary(
        args=args,
        split_info=split_info,
        metrics_rows=metrics_rows,
        edge_class_counts=edge_class_counts,
        eval_edge_class_counts=eval_edge_class_counts,
        atom_class_counts=atom_class_counts,
        elapsed_seconds=elapsed_seconds,
        dist_state=dist_state,
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(model.config),
            "summary": summary,
        },
        path,
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(model.config),
            "summary": summary,
        },
        path.parent / "edge_diffusion_ddp_latest.pt",
    )


def _build_summary(
    args: argparse.Namespace,
    split_info: dict[str, object],
    metrics_rows: list[dict[str, float]],
    edge_class_counts: torch.Tensor,
    eval_edge_class_counts: torch.Tensor,
    atom_class_counts: torch.Tensor,
    elapsed_seconds: float,
    dist_state: DistributedState,
) -> dict[str, object]:
    """Build a JSON summary for a DDP edge diffusion run."""
    final = metrics_rows[-1] if metrics_rows else {}
    best = max(metrics_rows, key=lambda row: row.get("eval_edge_real_bond_f1", float("-inf"))) if metrics_rows else {}
    return {
        "task": "edge_diffusion_ddp_train",
        "decoupled_from_rl": True,
        "ddpo_imported": False,
        "edge_npz": str(args.edge_npz),
        "semantic_npz": str(args.semantic_npz) if args.semantic_npz.exists() else "",
        "outdir": str(args.outdir),
        "device": str(dist_state.device),
        "distributed": dist_state.distributed,
        "world_size": int(dist_state.world_size),
        "max_molecules": int(args.max_molecules),
        "split": split_info,
        "epochs": int(args.epochs),
        "batch_size_per_rank": int(args.batch_size),
        "global_batch_size": int(args.batch_size * dist_state.world_size),
        "eval_batch_size": int(args.eval_batch_size),
        "num_workers_per_rank": int(args.num_workers),
        "elapsed_seconds": float(elapsed_seconds),
        "seconds_per_epoch": float(elapsed_seconds / max(1, args.epochs)),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "num_timesteps": int(args.num_timesteps),
        "eval_timestep": int(args.eval_timestep),
        "edge_prior": str(args.edge_prior),
        "pos_noise_std": float(args.pos_noise_std),
        "pos_noise_mode": str(args.pos_noise_mode),
        "atom_corruption_prob": float(args.atom_corruption_prob),
        "edge_class_weight_mode": str(args.edge_class_weight_mode),
        "no_bond_weight": float(args.no_bond_weight),
        "loss_config": asdict(EdgeDiffusionLossConfig(
            pos_weight=args.pos_weight,
            atom_weight=args.atom_weight,
            edge_weight=args.edge_weight,
            bond_length_weight=args.bond_length_weight,
        )),
        "model_config": asdict(EdgeDiffusionModelConfig(
            num_atom_types=int(args.num_atom_types_inferred),
            num_edge_types=int(args.num_edge_types),
            num_timesteps=int(args.num_timesteps),
            hidden_dim=int(args.hidden_dim),
            num_layers=int(args.num_layers),
            distance_basis_dim=int(args.distance_basis_dim),
            dropout=float(args.dropout),
            coordinate_scale=float(args.coordinate_scale),
        )),
        "edge_class_counts": _tensor_count_dict(edge_class_counts),
        "eval_edge_class_counts": _tensor_count_dict(eval_edge_class_counts),
        "atom_class_counts": {
            str(idx): int(value)
            for idx, value in enumerate(atom_class_counts.tolist())
        },
        "final_epoch": final,
        "best_real_bond_f1_epoch": best,
    }


def _tensor_count_dict(counts: torch.Tensor) -> dict[str, int]:
    """Return bond class counts keyed by class id."""
    return {str(idx): int(value) for idx, value in enumerate(counts.tolist())}


def _write_report(outdir: Path, summary: dict[str, object]) -> Path:
    """Write a compact Markdown report for DDP edge diffusion training."""
    report_path = outdir / "edge_diffusion_ddp_report.md"
    final = summary["final_epoch"]
    best = summary["best_real_bond_f1_epoch"]
    with report_path.open("w") as handle:
        handle.write("# Edge Diffusion DDP Training Report\n\n")
        handle.write("## Scope\n\n")
        handle.write(
            "Standalone x/h/e edge diffusion supervised training. "
            "This path is intentionally decoupled from DDPO/RL and does not import `ddpo`.\n\n"
        )
        handle.write("## Configuration\n\n")
        handle.write(f"- edge_npz: `{summary['edge_npz']}`\n")
        handle.write(f"- semantic_npz: `{summary['semantic_npz']}`\n")
        handle.write(f"- device: `{summary['device']}`\n")
        handle.write(f"- distributed: `{summary['distributed']}`\n")
        handle.write(f"- world_size: {summary['world_size']}\n")
        handle.write(f"- max_molecules: {summary['max_molecules']}\n")
        handle.write(f"- train_molecules: {summary['split']['train_molecules']}\n")
        handle.write(f"- eval_molecules: {summary['split']['eval_molecules']}\n")
        handle.write(f"- epochs: {summary['epochs']}\n")
        handle.write(f"- batch_size_per_rank: {summary['batch_size_per_rank']}\n")
        handle.write(f"- global_batch_size: {summary['global_batch_size']}\n")
        handle.write(f"- elapsed_seconds: {summary['elapsed_seconds']:.2f}\n")
        handle.write(f"- seconds_per_epoch: {summary['seconds_per_epoch']:.4f}\n\n")

        handle.write("## Final Metrics\n\n")
        handle.write("| metric | value |\n")
        handle.write("|---|---:|\n")
        for metric in [
            "train_total", "train_pos", "train_atom", "train_edge", "train_bond_length",
            "eval_total", "eval_pos", "eval_atom", "eval_edge", "eval_bond_length",
            "eval_edge_accuracy", "eval_edge_balanced_accuracy",
            "eval_edge_real_bond_precision", "eval_edge_real_bond_recall",
            "eval_edge_real_bond_f1", "eval_atom_accuracy",
            "eval_pos_rmse", "eval_bond_length_mae",
        ]:
            handle.write(f"| {metric} | {_format_float(final.get(metric))} |\n")

        handle.write("\n## Best Real-Bond F1 Epoch\n\n")
        handle.write(f"- epoch: {int(best.get('epoch', 0))}\n")
        handle.write(f"- eval_edge_real_bond_f1: {_format_float(best.get('eval_edge_real_bond_f1'))}\n")
        handle.write(f"- eval_edge_real_bond_recall: {_format_float(best.get('eval_edge_real_bond_recall'))}\n\n")

        handle.write("## Files\n\n")
        handle.write("- [edge_diffusion_ddp_metrics.csv](edge_diffusion_ddp_metrics.csv)\n")
        handle.write("- [edge_diffusion_ddp_summary.json](edge_diffusion_ddp_summary.json)\n")
        handle.write("- [edge_diffusion_ddp_latest.pt](edge_diffusion_ddp_latest.pt)\n")
        handle.write("- [edge_diffusion_ddp_curves.png](edge_diffusion_ddp_curves.png)\n\n")
        handle.write("![curves](edge_diffusion_ddp_curves.png)\n")
    return report_path


def _format_float(value: object) -> str:
    """Format numeric values for Markdown."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "nan"
    if math.isnan(number):
        return "nan"
    return f"{number:.6f}"


def _write_resolved_config(path: Path, args: argparse.Namespace, dist_state: DistributedState) -> None:
    """Write the flattened config used by the DDP run."""
    payload = vars(args).copy()
    for key, value in list(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
    payload["world_size"] = dist_state.world_size
    payload["device"] = str(dist_state.device)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def _args_from_config(config: dict[str, Any], config_path: Path | None = None) -> argparse.Namespace:
    """Flatten the YAML config into the argument namespace expected by edge helpers."""
    data = config.get("data", {}) or {}
    train = config.get("train", {}) or {}
    model = config.get("model", {}) or {}
    diffusion = config.get("diffusion", {}) or {}
    loss = config.get("loss", {}) or {}
    dataloader = config.get("dataloader", {}) or {}
    evaluation = config.get("evaluation", {}) or {}
    output = config.get("output", {}) or {}

    run_name = str(config.get("name", "edge_diffusion_ddp"))
    default_outdir = Path("analysis") / f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    args = argparse.Namespace(
        config_path=config_path,
        name=run_name,
        edge_npz=Path(data.get("edge_npz", "analysis/edge_data_chain_20260602_repair_n1000/train_edge.npz")),
        semantic_npz=Path(data.get("semantic_npz", "analysis/edge_semantic_bundle_20260602_n1000/train_edge_semantic.npz")),
        outdir=Path(output.get("outdir", default_outdir)),
        max_molecules=int(data.get("max_molecules", 1000)),
        val_molecules=int(data.get("val_molecules", 100)),
        split_seed=int(data.get("split_seed", train.get("seed", 260604))),
        batch_size=int(dataloader.get("batch_size", 8)),
        eval_batch_size=int(dataloader.get("eval_batch_size", dataloader.get("batch_size", 8))),
        num_workers=int(dataloader.get("num_workers", 2)),
        epochs=int(train.get("epochs", 30)),
        lr=float(train.get("lr", 2e-3)),
        weight_decay=float(train.get("weight_decay", 1e-4)),
        grad_clip=float(train.get("grad_clip", 5.0)),
        checkpoint_interval=int(train.get("checkpoint_interval", 5)),
        seed=int(train.get("seed", 260604)),
        hidden_dim=int(model.get("hidden_dim", 128)),
        num_layers=int(model.get("num_layers", 3)),
        distance_basis_dim=int(model.get("distance_basis_dim", 32)),
        dropout=float(model.get("dropout", 0.0)),
        coordinate_scale=float(model.get("coordinate_scale", 0.1)),
        num_edge_types=int(diffusion.get("num_edge_types", 5)),
        num_timesteps=int(diffusion.get("num_timesteps", 1000)),
        edge_prior=str(diffusion.get("edge_prior", "uniform")),
        pos_noise_std=float(diffusion.get("pos_noise_std", 0.05)),
        pos_noise_mode=str(diffusion.get("pos_noise_mode", "t_scaled")),
        atom_corruption_prob=float(diffusion.get("atom_corruption_prob", 0.05)),
        pos_weight=float(loss.get("pos_weight", 1.0)),
        atom_weight=float(loss.get("atom_weight", 1.0)),
        edge_weight=float(loss.get("edge_weight", 1.0)),
        bond_length_weight=float(loss.get("bond_length_weight", 0.5)),
        edge_class_weight_mode=str(loss.get("edge_class_weight_mode", "inverse_sqrt")),
        no_bond_weight=float(loss.get("no_bond_weight", 0.1)),
        eval_timestep=int(evaluation.get("eval_timestep", 500)),
        force_cpu=bool(train.get("force_cpu", False)),
        num_atom_types_inferred=1,
    )
    _validate_args(args)
    return args


def _validate_args(args: argparse.Namespace) -> None:
    """Validate flattened config values."""
    if args.max_molecules <= 0:
        raise ValueError("data.max_molecules must be positive")
    if args.val_molecules < 0:
        raise ValueError("data.val_molecules must be non-negative")
    if args.batch_size <= 0 or args.eval_batch_size <= 0:
        raise ValueError("dataloader batch sizes must be positive")
    if args.epochs <= 0:
        raise ValueError("train.epochs must be positive")
    if args.checkpoint_interval <= 0:
        raise ValueError("train.checkpoint_interval must be positive")
    if args.num_workers < 0:
        raise ValueError("dataloader.num_workers must be non-negative")
    if args.pos_noise_mode not in {"constant", "t_scaled"}:
        raise ValueError("diffusion.pos_noise_mode must be constant or t_scaled")
    if args.edge_prior not in {"uniform", "absorb", "tomask"}:
        raise ValueError("diffusion.edge_prior must be uniform, absorb, or tomask")
    if args.edge_class_weight_mode not in {"inverse_sqrt", "diffgui", "none"}:
        raise ValueError("loss.edge_class_weight_mode must be inverse_sqrt, diffgui, or none")


def _load_config(path: Path) -> dict[str, Any]:
    """Load a YAML config file."""
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _set_seed(seed: int) -> None:
    """Set Python, NumPy, and torch seeds for this rank."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "--config_path", dest="config_path", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--max-molecules", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--force-cpu", action="store_true")
    parsed = parser.parse_args()
    args = _args_from_config(_load_config(parsed.config_path), parsed.config_path)
    if parsed.outdir is not None:
        args.outdir = parsed.outdir
    if parsed.max_molecules is not None:
        args.max_molecules = int(parsed.max_molecules)
    if parsed.epochs is not None:
        args.epochs = int(parsed.epochs)
    if parsed.batch_size is not None:
        args.batch_size = int(parsed.batch_size)
    if parsed.force_cpu:
        args.force_cpu = True
    _validate_args(args)
    return args


if __name__ == "__main__":
    output_report = run_training(parse_args())
    if output_report is not None:
        print(f"Wrote edge diffusion DDP report to {output_report}")
