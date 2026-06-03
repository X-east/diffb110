"""Train the standalone x/h/e edge diffusion skeleton on small edge bundles."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from rdkit import Chem, RDLogger
from torch.utils.data import DataLoader, Dataset, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIFFSBDD_ROOT = Path(__file__).resolve().parents[1]
for extra in (PROJECT_ROOT, DIFFSBDD_ROOT):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from edge_diffusion import (  # noqa: E402
    EdgeBatch,
    EdgeBatchCollator,
    EdgeBundleDataset,
    EdgeCategoricalTransition,
    EdgeDiffusionInputs,
    EdgeDiffusionLossConfig,
    EdgeDiffusionModel,
    EdgeDiffusionModelConfig,
    EdgeTransitionConfig,
    batch_confusion,
    classification_metrics,
    compute_edge_diffusion_loss,
    make_edge_class_weight,
    reconstruction_metric_defaults,
    reconstruction_summary_for_training,
)
from edge_diffusion.evaluation import (  # noqa: E402
    build_reconstruction_row,
    build_semantic_result,
    reconstruction_fieldnames,
    summarize_reconstruction,
    write_result_mol,
)
from edge_utils import BOND_TYPE_NAMES  # noqa: E402

RDLogger.DisableLog("rdApp.*")


def run_overfit(args: argparse.Namespace) -> Path:
    """Train the x/h/e model and write reports."""
    _set_seed(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)
    device = _choose_device(args.force_cpu)
    started_at = time.time()

    dataset = EdgeBundleDataset(
        edge_npz=args.edge_npz,
        semantic_npz=args.semantic_npz,
        max_molecules=args.max_molecules,
    )
    train_dataset, eval_dataset, split_info = _split_dataset(dataset, args)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=EdgeBatchCollator(),
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
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
    ).to(device)
    atom_class_weight = _make_inverse_sqrt_weight(atom_class_counts).to(device)

    model = EdgeDiffusionModel(EdgeDiffusionModelConfig(
        num_atom_types=num_atom_types,
        num_edge_types=args.num_edge_types,
        num_timesteps=args.num_timesteps,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        distance_basis_dim=args.distance_basis_dim,
        dropout=args.dropout,
        coordinate_scale=args.coordinate_scale,
    )).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
    for epoch in range(1, args.epochs + 1):
        train_metrics = _train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            transition=transition,
            loss_config=loss_config,
            atom_class_weight=atom_class_weight,
            edge_class_weight=edge_class_weight,
            args=args,
            device=device,
            seed=args.seed + epoch,
        )
        eval_metrics = _evaluate(
            model=model,
            dataloader=eval_loader,
            transition=transition,
            loss_config=loss_config,
            atom_class_weight=atom_class_weight,
            edge_class_weight=edge_class_weight,
            args=args,
            device=device,
            seed=args.seed + 1000 + epoch,
        )
        row = {
            "epoch": float(epoch),
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"eval_{key}": value for key, value in eval_metrics.items()},
            **_maybe_evaluate_reconstruction(
                model=model,
                dataset=dataset,
                split_info=split_info,
                args=args,
                transition=transition,
                epoch=epoch,
                device=device,
            ),
        }
        metrics_rows.append(row)

    metrics_path = args.outdir / "edge_diffusion_overfit_metrics.csv"
    _write_metrics_csv(metrics_path, metrics_rows)
    elapsed_seconds = time.time() - started_at
    summary = _build_summary(
        args=args,
        device=device,
        dataset=dataset,
        split_info=split_info,
        edge_class_counts=edge_class_counts,
        eval_edge_class_counts=eval_edge_class_counts,
        atom_class_counts=atom_class_counts,
        edge_class_weight=edge_class_weight.detach().cpu(),
        atom_class_weight=atom_class_weight.detach().cpu(),
        model=model,
        metrics_rows=metrics_rows,
        elapsed_seconds=elapsed_seconds,
    )
    (args.outdir / "edge_diffusion_overfit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(model.config),
            "summary": summary,
        },
        args.outdir / "edge_diffusion_overfit.pt",
    )
    _plot_training_curves(args.outdir / "edge_diffusion_overfit_curves.png", metrics_rows)
    return _write_report(args.outdir, summary)


def _train_one_epoch(
    model: EdgeDiffusionModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    transition: EdgeCategoricalTransition,
    loss_config: EdgeDiffusionLossConfig,
    atom_class_weight: torch.Tensor,
    edge_class_weight: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    """Train one epoch and return averaged loss terms."""
    model.train()
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    totals = _empty_loss_totals()
    total_batches = 0
    for batch in dataloader:
        batch = batch.to(device)
        inputs = _make_noised_inputs(
            batch=batch,
            transition=transition,
            args=args,
            device=device,
            generator=generator,
            fixed_timestep=None,
        )
        output = model(batch, inputs)
        loss = compute_edge_diffusion_loss(
            output=output,
            batch=batch,
            config=loss_config,
            atom_class_weight=atom_class_weight,
            edge_class_weight=edge_class_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        _accumulate_loss_totals(totals, loss.as_log_dict(prefix=""))
        total_batches += 1
    return _normalize_loss_totals(totals, total_batches)


@torch.no_grad()
def _evaluate(
    model: EdgeDiffusionModel,
    dataloader: DataLoader,
    transition: EdgeCategoricalTransition,
    loss_config: EdgeDiffusionLossConfig,
    atom_class_weight: torch.Tensor,
    edge_class_weight: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    """Evaluate denoising losses and x/h/e metrics at a fixed timestep."""
    model.eval()
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    loss_totals = _empty_loss_totals()
    total_batches = 0
    edge_confusion = torch.zeros((args.num_edge_types, args.num_edge_types), dtype=torch.long, device=device)
    atom_correct = 0
    atom_total = 0
    pos_squared_error = 0.0
    pos_count = 0
    bond_length_absolute_error = 0.0
    bond_length_count = 0

    for batch in dataloader:
        batch = batch.to(device)
        inputs = _make_noised_inputs(
            batch=batch,
            transition=transition,
            args=args,
            device=device,
            generator=generator,
            fixed_timestep=args.eval_timestep,
        )
        output = model(batch, inputs)
        loss = compute_edge_diffusion_loss(
            output=output,
            batch=batch,
            config=loss_config,
            atom_class_weight=atom_class_weight,
            edge_class_weight=edge_class_weight,
        )
        _accumulate_loss_totals(loss_totals, loss.as_log_dict(prefix=""))
        total_batches += 1

        edge_pred = output.pred_edge_type
        edge_confusion += batch_confusion(
            target=batch.target_edge_type,
            pred=edge_pred,
            num_classes=args.num_edge_types,
        )
        atom_pred = output.pred_atom_type
        atom_correct += int((atom_pred == batch.ligand_atom_type).sum().item())
        atom_total += int(batch.ligand_atom_type.numel())
        pos_squared_error += float(((output.pred_pos - batch.ligand_pos) ** 2).sum().item())
        pos_count += int(batch.ligand_pos.numel())
        length_error, length_count = _bond_length_error(output.pred_pos, batch)
        bond_length_absolute_error += length_error
        bond_length_count += length_count

    loss_metrics = _normalize_loss_totals(loss_totals, total_batches)
    edge_metrics, edge_class_rows = classification_metrics(
        edge_confusion.detach().cpu(),
        loss_metrics["edge"],
    )
    prefixed_edge_metrics = {f"edge_{key}": float(value) for key, value in edge_metrics.items()}
    for row in edge_class_rows:
        class_name = str(row["class_name"])
        prefixed_edge_metrics[f"edge_precision_{class_name}"] = float(row["precision"])
        prefixed_edge_metrics[f"edge_recall_{class_name}"] = float(row["recall"])
        prefixed_edge_metrics[f"edge_f1_{class_name}"] = float(row["f1"])
    return {
        **loss_metrics,
        **prefixed_edge_metrics,
        "atom_accuracy": atom_correct / max(1, atom_total),
        "pos_rmse": math.sqrt(pos_squared_error / max(1, pos_count)),
        "bond_length_mae": bond_length_absolute_error / max(1, bond_length_count),
    }


def _make_noised_inputs(
    batch: EdgeBatch,
    transition: EdgeCategoricalTransition,
    args: argparse.Namespace,
    device: torch.device,
    generator: torch.Generator,
    fixed_timestep: int | None,
) -> EdgeDiffusionInputs:
    """Build noised x/h/e inputs for one batch."""
    if fixed_timestep is None:
        t = torch.randint(
            low=0,
            high=args.num_timesteps,
            size=(len(batch.names),),
            device=device,
            generator=generator,
        )
    else:
        t = torch.full((len(batch.names),), int(fixed_timestep), dtype=torch.long, device=device)
    noised_edge_type = transition.q_sample(
        batch.target_edge_type,
        t[batch.edge_batch],
        generator=generator,
    )
    ligand_pos_t = _noise_positions(batch.ligand_pos, t, batch.atom_batch, args, generator)
    ligand_atom_type_t = _noise_atom_types(batch.ligand_atom_type, t, batch.atom_batch, args, generator)
    return EdgeDiffusionInputs(
        ligand_pos_t=ligand_pos_t,
        ligand_atom_type_t=ligand_atom_type_t,
        ligand_edge_type_t=noised_edge_type,
        t=t,
    )


def _noise_positions(
    pos: torch.Tensor,
    t: torch.Tensor,
    atom_batch: torch.Tensor,
    args: argparse.Namespace,
    generator: torch.Generator,
) -> torch.Tensor:
    """Add simple Gaussian coordinate noise, optionally scaled by timestep."""
    if args.pos_noise_std <= 0:
        return pos
    noise = torch.randn(pos.shape, device=pos.device, dtype=pos.dtype, generator=generator)
    if args.pos_noise_mode == "constant":
        scale = torch.full((pos.shape[0], 1), float(args.pos_noise_std), device=pos.device, dtype=pos.dtype)
    else:
        t_per_atom = t[atom_batch.long()].float() / float(max(1, args.num_timesteps - 1))
        scale = float(args.pos_noise_std) * t_per_atom.unsqueeze(-1)
    return pos + noise * scale


def _noise_atom_types(
    atom_type: torch.Tensor,
    t: torch.Tensor,
    atom_batch: torch.Tensor,
    args: argparse.Namespace,
    generator: torch.Generator,
) -> torch.Tensor:
    """Randomly corrupt atom labels with timestep-scaled probability."""
    if args.atom_corruption_prob <= 0:
        return atom_type
    t_per_atom = t[atom_batch.long()].float() / float(max(1, args.num_timesteps - 1))
    corruption_prob = (float(args.atom_corruption_prob) * t_per_atom).clamp(max=1.0)
    random_values = torch.rand(atom_type.shape, device=atom_type.device, generator=generator)
    corrupt_mask = random_values < corruption_prob
    random_atom_type = torch.randint(
        low=0,
        high=args.num_atom_types_inferred,
        size=atom_type.shape,
        device=atom_type.device,
        generator=generator,
    )
    return torch.where(corrupt_mask, random_atom_type, atom_type)


@torch.no_grad()
def _maybe_evaluate_reconstruction(
    model: EdgeDiffusionModel,
    dataset: EdgeBundleDataset,
    split_info: dict[str, object],
    args: argparse.Namespace,
    transition: EdgeCategoricalTransition,
    epoch: int,
    device: torch.device,
) -> dict[str, float]:
    """Run optional reconstruction validation with x/h/e outputs."""
    defaults = reconstruction_metric_defaults()
    if args.reconstruction_every <= 0:
        return defaults
    if epoch % args.reconstruction_every != 0 and epoch != args.epochs:
        return defaults
    eval_indices = [int(idx) for idx in split_info["eval_indices"]]
    _, summary = _evaluate_reconstruction(
        model=model,
        dataset=dataset,
        eval_indices=eval_indices,
        edge_npz=args.edge_npz,
        semantic_npz=args.semantic_npz,
        transition=transition,
        args=args,
        device=device,
        outdir=args.outdir,
        write_sdf=epoch == args.epochs and args.write_sdf,
    )
    return reconstruction_summary_for_training(summary)


def _evaluate_reconstruction(
    model: EdgeDiffusionModel,
    dataset: EdgeBundleDataset,
    eval_indices: list[int],
    edge_npz: Path,
    semantic_npz: Path | None,
    transition: EdgeCategoricalTransition,
    args: argparse.Namespace,
    device: torch.device,
    outdir: Path,
    write_sdf: bool,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Predict x/e for validation molecules and reconstruct RDKit molecules."""
    import numpy as np

    edge_data = np.load(edge_npz, allow_pickle=True)
    semantic_data = None if semantic_npz is None else np.load(semantic_npz, allow_pickle=True)
    selected_indices = eval_indices if args.reconstruction_limit is None else eval_indices[:args.reconstruction_limit]
    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed) + 100_000 + int(args.reconstruction_timestep))
    plain_writer = None
    semantic_writer = None
    if write_sdf:
        plain_writer = Chem.SDWriter(str(outdir / f"edge_diffusion_pred_plain_t{args.reconstruction_timestep}.sdf"))
        semantic_writer = Chem.SDWriter(str(outdir / f"edge_diffusion_pred_semantic_t{args.reconstruction_timestep}.sdf"))

    rows: list[dict[str, object]] = []
    strategy_counts: Counter[str] = Counter()
    was_training = bool(model.training)
    model.eval()
    try:
        for global_idx in selected_indices:
            sample = dataset[global_idx]
            batch = EdgeBatchCollator()([sample]).to(device)
            inputs = _make_noised_inputs(
                batch=batch,
                transition=transition,
                args=args,
                device=device,
                generator=generator,
                fixed_timestep=args.reconstruction_timestep,
            )
            output = model(batch, inputs)
            pred_edge_type = output.pred_edge_type.detach().cpu()
            target_edge_type = batch.target_edge_type.detach().cpu()
            halfedge_index = batch.halfedge_index.detach().cpu()
            positions = output.pred_pos.detach().cpu()
            atom_symbols = [str(value) for value in edge_data["ligand_atom_symbol"][global_idx]]
            plain_result = build_semantic_result(
                semantic_data=None,
                global_idx=global_idx,
                atom_symbols=atom_symbols,
                positions=positions,
                halfedge_index=halfedge_index,
                halfedge_type=pred_edge_type,
            )
            semantic_result = build_semantic_result(
                semantic_data=semantic_data,
                global_idx=global_idx,
                atom_symbols=atom_symbols,
                positions=positions,
                halfedge_index=halfedge_index,
                halfedge_type=pred_edge_type,
            )
            row = build_reconstruction_row(
                global_idx=global_idx,
                name=str(sample["name"]),
                atom_symbols=atom_symbols,
                positions=positions,
                halfedge_index=halfedge_index,
                target_edge_type=target_edge_type,
                pred_edge_type=pred_edge_type,
                plain_result=plain_result,
                semantic_result=semantic_result,
                original_smiles=(
                    "" if semantic_data is None
                    else str(semantic_data["original_smiles_no_stereo"][global_idx])
                ),
                min_bond_length=args.min_bond_length,
                max_bond_length=args.max_bond_length,
            )
            rows.append(row)
            strategy_counts[f"plain:{plain_result.repair_strategy}"] += 1
            strategy_counts[f"semantic:{semantic_result.repair_strategy}"] += 1
            if plain_writer is not None:
                write_result_mol(plain_writer, plain_result.mol, row, prefix="plain")
            if semantic_writer is not None:
                write_result_mol(semantic_writer, semantic_result.mol, row, prefix="semantic")
    finally:
        if plain_writer is not None:
            plain_writer.close()
        if semantic_writer is not None:
            semantic_writer.close()
        if was_training:
            model.train()
        edge_data.close()
        if semantic_data is not None:
            semantic_data.close()
    if rows:
        _write_metrics_csv(outdir / "edge_diffusion_reconstruction_rows.csv", rows)
    else:
        _write_metrics_csv(outdir / "edge_diffusion_reconstruction_rows.csv", [])
    return rows, summarize_reconstruction(rows, strategy_counts)


def _split_dataset(
    dataset: EdgeBundleDataset,
    args: argparse.Namespace,
) -> tuple[Dataset, Dataset, dict[str, object]]:
    """Return train/eval datasets and split metadata."""
    num_molecules = len(dataset)
    val_molecules = int(args.val_molecules)
    if val_molecules <= 0:
        indices = list(range(num_molecules))
        return dataset, dataset, {
            "mode": "overfit",
            "train_molecules": num_molecules,
            "eval_molecules": num_molecules,
            "val_molecules": 0,
            "split_seed": None,
            "train_indices": indices,
            "eval_indices": indices,
        }
    if val_molecules >= num_molecules:
        raise ValueError("val_molecules must be smaller than max_molecules")

    generator = torch.Generator()
    generator.manual_seed(int(args.split_seed))
    permutation = torch.randperm(num_molecules, generator=generator).tolist()
    eval_indices = sorted(permutation[:val_molecules])
    train_indices = sorted(permutation[val_molecules:])
    return (
        Subset(dataset, train_indices),
        Subset(dataset, eval_indices),
        {
            "mode": "train_val",
            "train_molecules": len(train_indices),
            "eval_molecules": len(eval_indices),
            "val_molecules": val_molecules,
            "split_seed": int(args.split_seed),
            "train_indices": train_indices,
            "eval_indices": eval_indices,
        },
    )


def _count_edge_classes(dataset: Dataset, num_classes: int) -> torch.Tensor:
    """Count target edge classes in a dataset or subset."""
    counts = torch.zeros((num_classes,), dtype=torch.long)
    for idx in range(len(dataset)):
        sample = dataset[idx]
        counts += torch.bincount(sample["target_edge_type"], minlength=num_classes)
    return counts


def _count_atom_classes(dataset: Dataset, num_classes: int) -> torch.Tensor:
    """Count atom type classes in a dataset or subset."""
    counts = torch.zeros((num_classes,), dtype=torch.long)
    for idx in range(len(dataset)):
        sample = dataset[idx]
        counts += torch.bincount(sample["ligand_atom_type"], minlength=num_classes)
    return counts


def _infer_num_atom_types(dataset: EdgeBundleDataset) -> int:
    """Infer atom embedding size from atom type IDs."""
    max_atom_type = 0
    for idx in range(len(dataset)):
        sample = dataset[idx]
        if sample["ligand_atom_type"].numel() == 0:
            continue
        max_atom_type = max(max_atom_type, int(sample["ligand_atom_type"].max().item()))
    return max_atom_type + 1


def _make_inverse_sqrt_weight(counts: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Return inverse-sqrt class weights normalized to mean one."""
    weights = 1.0 / torch.sqrt(counts.float() + float(eps))
    return weights / weights.mean().clamp_min(1e-8)


def _bond_length_error(pred_pos: torch.Tensor, batch: EdgeBatch) -> tuple[float, int]:
    """Return summed absolute true-bond length error and count."""
    is_bond = batch.target_edge_type > 0
    if not bool(is_bond.any()):
        return 0.0, 0
    edge_index = batch.halfedge_index[:, is_bond]
    src, dst = edge_index
    pred_distance = torch.linalg.norm(pred_pos[src] - pred_pos[dst], dim=-1)
    target_distance = torch.linalg.norm(batch.ligand_pos[src] - batch.ligand_pos[dst], dim=-1)
    return float(torch.abs(pred_distance - target_distance).sum().item()), int(pred_distance.numel())


def _empty_loss_totals() -> dict[str, float]:
    """Return zeroed loss accumulators."""
    return {"total": 0.0, "pos": 0.0, "atom": 0.0, "edge": 0.0, "bond_length": 0.0}


def _accumulate_loss_totals(totals: dict[str, float], loss_dict: dict[str, float]) -> None:
    """Accumulate detached loss dictionary values."""
    for key in totals:
        totals[key] += float(loss_dict[f"/{key}"])


def _normalize_loss_totals(totals: dict[str, float], count: int) -> dict[str, float]:
    """Normalize loss accumulators by batch count."""
    return {key: value / max(1, count) for key, value in totals.items()}


def _write_metrics_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write rows to CSV, preserving row key order."""
    fieldnames = list(rows[0]) if rows else reconstruction_fieldnames()
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _build_summary(
    args: argparse.Namespace,
    device: torch.device,
    dataset: EdgeBundleDataset,
    split_info: dict[str, object],
    edge_class_counts: torch.Tensor,
    eval_edge_class_counts: torch.Tensor,
    atom_class_counts: torch.Tensor,
    edge_class_weight: torch.Tensor,
    atom_class_weight: torch.Tensor,
    model: EdgeDiffusionModel,
    metrics_rows: list[dict[str, float]],
    elapsed_seconds: float,
) -> dict[str, object]:
    """Build JSON summary for the x/h/e overfit run."""
    final = metrics_rows[-1]
    best = max(metrics_rows, key=lambda row: row["eval_edge_real_bond_f1"])
    return {
        "edge_npz": str(args.edge_npz),
        "semantic_npz": "" if args.semantic_npz is None else str(args.semantic_npz),
        "device": str(device),
        "max_molecules": len(dataset),
        "split": split_info,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "elapsed_seconds": elapsed_seconds,
        "seconds_per_epoch": elapsed_seconds / max(1, args.epochs),
        "num_timesteps": args.num_timesteps,
        "eval_timestep": args.eval_timestep,
        "edge_prior": args.edge_prior,
        "pos_noise_std": args.pos_noise_std,
        "pos_noise_mode": args.pos_noise_mode,
        "atom_corruption_prob": args.atom_corruption_prob,
        "edge_class_weight_mode": args.edge_class_weight_mode,
        "no_bond_weight": args.no_bond_weight,
        "loss_config": asdict(EdgeDiffusionLossConfig(
            pos_weight=args.pos_weight,
            atom_weight=args.atom_weight,
            edge_weight=args.edge_weight,
            bond_length_weight=args.bond_length_weight,
        )),
        "model_config": asdict(model.config),
        "reconstruction_validation": {
            "enabled": args.reconstruction_every > 0,
            "every": args.reconstruction_every,
            "timestep": args.reconstruction_timestep,
            "limit": args.reconstruction_limit,
            "write_sdf": args.write_sdf,
            "min_bond_length": args.min_bond_length,
            "max_bond_length": args.max_bond_length,
        },
        "edge_class_counts": {
            BOND_TYPE_NAMES[idx]: int(value)
            for idx, value in enumerate(edge_class_counts.tolist())
        },
        "eval_edge_class_counts": {
            BOND_TYPE_NAMES[idx]: int(value)
            for idx, value in enumerate(eval_edge_class_counts.tolist())
        },
        "edge_class_weight": {
            BOND_TYPE_NAMES[idx]: float(value)
            for idx, value in enumerate(edge_class_weight.tolist())
        },
        "atom_class_counts": {
            str(idx): int(value)
            for idx, value in enumerate(atom_class_counts.tolist())
        },
        "atom_class_weight": {
            str(idx): float(value)
            for idx, value in enumerate(atom_class_weight.tolist())
        },
        "final_epoch": final,
        "best_real_bond_f1_epoch": best,
    }


def _plot_training_curves(path: Path, rows: list[dict[str, float]]) -> None:
    """Plot loss and key x/h/e metrics."""
    epochs = [int(row["epoch"]) for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].plot(epochs, [row["train_total"] for row in rows], label="train total")
    axes[0, 0].plot(epochs, [row["eval_total"] for row in rows], label="eval total")
    axes[0, 0].plot(epochs, [row["eval_edge"] for row in rows], label="eval edge")
    axes[0, 0].set_title("Loss")
    axes[0, 0].legend()

    axes[0, 1].plot(epochs, [row["eval_edge_real_bond_precision"] for row in rows], label="real bond precision")
    axes[0, 1].plot(epochs, [row["eval_edge_real_bond_recall"] for row in rows], label="real bond recall")
    axes[0, 1].plot(epochs, [row["eval_edge_real_bond_f1"] for row in rows], label="real bond F1")
    axes[0, 1].set_title("Edge real-bond metrics")
    axes[0, 1].legend()

    axes[1, 0].plot(epochs, [row["eval_atom_accuracy"] for row in rows], label="atom accuracy")
    axes[1, 0].plot(epochs, [row["eval_pos_rmse"] for row in rows], label="pos RMSE")
    axes[1, 0].plot(epochs, [row["eval_bond_length_mae"] for row in rows], label="bond length MAE")
    axes[1, 0].set_title("x/h metrics")
    axes[1, 0].legend()

    if _has_finite_metric(rows, "recon_semantic_sanitize_success_rate"):
        axes[1, 1].plot(
            epochs,
            [row["recon_semantic_sanitize_success_rate"] for row in rows],
            marker="o",
            label="semantic sanitize",
        )
        axes[1, 1].plot(
            epochs,
            [row["recon_bond_type_map_match_rate"] for row in rows],
            marker="o",
            label="bond type map match",
        )
        axes[1, 1].plot(
            epochs,
            [row["recon_mean_bad_bond_length_fraction"] for row in rows],
            marker="o",
            label="bad bond length frac",
        )
    axes[1, 1].set_title("Reconstruction")
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _has_finite_metric(rows: list[dict[str, float]], key: str) -> bool:
    """Return whether any row contains a finite metric value."""
    return any(math.isfinite(float(row.get(key, float("nan")))) for row in rows)


def _write_report(outdir: Path, summary: dict[str, object]) -> Path:
    """Write Markdown report for the x/h/e overfit experiment."""
    report_path = outdir / "edge_diffusion_overfit_report.md"
    final = summary["final_epoch"]
    best = summary["best_real_bond_f1_epoch"]
    with report_path.open("w") as handle:
        handle.write("# Edge Diffusion x/h/e Overfit Report\n\n")
        handle.write(f"- edge_npz: `{summary['edge_npz']}`\n")
        handle.write(f"- semantic_npz: `{summary['semantic_npz']}`\n")
        handle.write(f"- device: `{summary['device']}`\n")
        handle.write(f"- max_molecules: {summary['max_molecules']}\n")
        handle.write(f"- split_mode: `{summary['split']['mode']}`\n")
        handle.write(f"- train_molecules: {summary['split']['train_molecules']}\n")
        handle.write(f"- eval_molecules: {summary['split']['eval_molecules']}\n")
        handle.write(f"- epochs: {summary['epochs']}\n")
        handle.write(f"- batch_size: {summary['batch_size']}\n")
        handle.write(f"- elapsed_seconds: {summary['elapsed_seconds']:.2f}\n")
        handle.write(f"- seconds_per_epoch: {summary['seconds_per_epoch']:.4f}\n")
        handle.write(f"- eval_timestep: {summary['eval_timestep']}\n")
        handle.write(f"- pos_noise_std: {summary['pos_noise_std']}\n")
        handle.write(f"- pos_noise_mode: `{summary['pos_noise_mode']}`\n")
        handle.write(f"- atom_corruption_prob: {summary['atom_corruption_prob']}\n")
        handle.write(f"- edge_prior: `{summary['edge_prior']}`\n")
        handle.write(f"- edge_class_weight_mode: `{summary['edge_class_weight_mode']}`\n")
        handle.write(f"- reconstruction_validation: `{summary['reconstruction_validation']['enabled']}`\n\n")

        handle.write("## Edge Class Counts And Weights\n\n")
        handle.write("| bond type | train count | eval count | weight |\n")
        handle.write("|---|---:|---:|---:|\n")
        for bond_name, count in summary["edge_class_counts"].items():
            handle.write(
                f"| {bond_name} | {count} | "
                f"{summary['eval_edge_class_counts'][bond_name]} | "
                f"{summary['edge_class_weight'][bond_name]:.6f} |\n"
            )

        handle.write("\n## Final Metrics\n\n")
        handle.write("| metric | value |\n")
        handle.write("|---|---:|\n")
        for metric in [
            "train_total", "train_pos", "train_atom", "train_edge", "train_bond_length",
            "eval_total", "eval_pos", "eval_atom", "eval_edge", "eval_bond_length",
            "eval_edge_accuracy", "eval_edge_balanced_accuracy",
            "eval_edge_real_bond_precision", "eval_edge_real_bond_recall",
            "eval_edge_real_bond_f1", "eval_atom_accuracy",
            "eval_pos_rmse", "eval_bond_length_mae",
            "eval_edge_recall_single", "eval_edge_recall_double",
            "eval_edge_recall_triple", "eval_edge_recall_aromatic",
        ]:
            handle.write(f"| {metric} | {float(final[metric]):.6f} |\n")

        if summary["reconstruction_validation"]["enabled"]:
            handle.write("\n## Final Reconstruction Metrics\n\n")
            handle.write("| metric | value |\n")
            handle.write("|---|---:|\n")
            for metric in [
                "recon_molecules", "recon_exact_halfedge_match_rate",
                "recon_bond_pair_set_match_rate", "recon_bond_type_map_match_rate",
                "recon_plain_sanitize_success_rate", "recon_semantic_sanitize_success_rate",
                "recon_plain_smiles_no_stereo_match_rate",
                "recon_semantic_smiles_no_stereo_match_rate",
                "recon_mean_bad_bond_length_fraction",
            ]:
                value = float(final[metric])
                value_text = "nan" if math.isnan(value) else f"{value:.6f}"
                handle.write(f"| {metric} | {value_text} |\n")

        handle.write("\n## Best Real-Bond F1 Epoch\n\n")
        handle.write(f"- epoch: {int(best['epoch'])}\n")
        handle.write(f"- eval_edge_real_bond_f1: {float(best['eval_edge_real_bond_f1']):.6f}\n")
        handle.write(f"- eval_edge_real_bond_recall: {float(best['eval_edge_real_bond_recall']):.6f}\n\n")

        handle.write("## Files\n\n")
        handle.write("- [edge_diffusion_overfit_metrics.csv](edge_diffusion_overfit_metrics.csv)\n")
        handle.write("- [edge_diffusion_overfit_summary.json](edge_diffusion_overfit_summary.json)\n")
        handle.write("- [edge_diffusion_overfit.pt](edge_diffusion_overfit.pt)\n")
        handle.write("- [edge_diffusion_reconstruction_rows.csv](edge_diffusion_reconstruction_rows.csv)\n")
        handle.write("\n![curves](edge_diffusion_overfit_curves.png)\n")
    return report_path


def _choose_device(force_cpu: bool) -> torch.device:
    """Choose CUDA when available unless CPU is forced."""
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _set_seed(seed: int) -> None:
    """Set Python and torch random seeds."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--edge-npz",
        type=Path,
        default=Path("analysis/edge_data_chain_20260602_repair_n1000/train_edge.npz"),
    )
    parser.add_argument(
        "--semantic-npz",
        type=Path,
        default=Path("analysis/edge_semantic_bundle_20260602_n1000/train_edge_semantic.npz"),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("analysis") / f"edge_diffusion_overfit_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument("--max-molecules", type=int, default=128)
    parser.add_argument("--val-molecules", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--distance-basis-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--coordinate-scale", type=float, default=0.1)
    parser.add_argument("--num-edge-types", type=int, default=5)
    parser.add_argument("--num-timesteps", type=int, default=1000)
    parser.add_argument("--edge-prior", choices=["uniform", "absorb", "tomask"], default="absorb")
    parser.add_argument("--eval-timestep", type=int, default=500)
    parser.add_argument("--pos-noise-std", type=float, default=0.05)
    parser.add_argument("--pos-noise-mode", choices=["constant", "t_scaled"], default="t_scaled")
    parser.add_argument("--atom-corruption-prob", type=float, default=0.05)
    parser.add_argument("--pos-weight", type=float, default=1.0)
    parser.add_argument("--atom-weight", type=float, default=1.0)
    parser.add_argument("--edge-weight", type=float, default=1.0)
    parser.add_argument("--bond-length-weight", type=float, default=0.5)
    parser.add_argument(
        "--edge-class-weight-mode",
        choices=["inverse_sqrt", "diffgui", "none"],
        default="inverse_sqrt",
    )
    parser.add_argument("--no-bond-weight", type=float, default=0.1)
    parser.add_argument("--reconstruction-every", type=int, default=20)
    parser.add_argument("--reconstruction-timestep", type=int, default=500)
    parser.add_argument("--reconstruction-limit", type=int, default=32)
    parser.add_argument("--write-sdf", action="store_true")
    parser.add_argument("--min-bond-length", type=float, default=0.7)
    parser.add_argument("--max-bond-length", type=float, default=2.1)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=260603)
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()
    args.num_atom_types_inferred = 1
    return args


if __name__ == "__main__":
    parsed_args = parse_args()
    parsed_args.num_atom_types_inferred = _infer_num_atom_types(EdgeBundleDataset(
        edge_npz=parsed_args.edge_npz,
        semantic_npz=parsed_args.semantic_npz,
        max_molecules=parsed_args.max_molecules,
    ))
    report = run_overfit(parsed_args)
    print(f"Wrote edge diffusion overfit report to {report}")
