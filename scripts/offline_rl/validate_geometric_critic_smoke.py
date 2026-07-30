"""Run a real-shard capacity smoke test for DiffSBDD geometric critics.

This is not a scientific offline-RL evaluation. It checks that the critic
boundary can consume a formal trajectory shard and the exact fixed pocket,
then overfit one final-transition batch to a finite terminal proxy target
(QED for valid molecules and an explicit penalty for invalid molecules).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import QED
from torch.nn import functional as F

from ddpo.integration.diffsbdd_model import DiffSBDDModel
from ddpo.offline_rl import (
    PocketContext,
    build_geometric_critic_input,
    load_trajectory_shard,
)
from ddpo.offline_rl.models import (
    GeometricCriticConfig,
    GeometricValueNetwork,
    TwinQCritic,
)


LOGGER = logging.getLogger("offline_rl.critic_smoke")


def _parse_args() -> argparse.Namespace:
    """Parse command-line paths and smoke-test hyperparameters."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pocket-pdb", type=Path, required=True)
    parser.add_argument("--ref-ligand", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--invalid-penalty", type=float, default=-1.0)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--message-layers", type=int, default=2)
    parser.add_argument("--max-neighbors", type=int, default=24)
    return parser.parse_args()


def _configure_logging(output_dir: Path) -> None:
    """Write grep-friendly progress to terminal and a durable log."""
    output_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    file_handler = logging.FileHandler(
        output_dir / "critic_smoke.log",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(stream_handler)


def _proxy_targets(
    reconstruction_rows: tuple[dict[str, Any] | Any, ...],
    *,
    invalid_penalty: float,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Compute finite QED/invalid targets for capacity testing."""
    targets: list[float] = []
    rows: list[dict[str, Any]] = []
    for episode_index, raw_row in enumerate(reconstruction_rows):
        row = raw_row if isinstance(raw_row, dict) else {}
        smiles = row.get("smiles")
        rdkit_valid = bool(int(row.get("rdkit_valid", 0)))
        molecule = Chem.MolFromSmiles(str(smiles)) if rdkit_valid and smiles else None
        if molecule is None:
            target = float(invalid_penalty)
            rdkit_valid = False
        else:
            target = float(QED.qed(molecule))
        targets.append(target)
        rows.append(
            {
                "episode_index": episode_index,
                "sample_id": str(row.get("sample_id", "")),
                "rdkit_valid": int(rdkit_valid),
                "proxy_target": target,
                "smiles": "" if smiles is None else str(smiles),
            }
        )
    target_tensor = torch.tensor(targets, dtype=torch.float32)
    if not bool(torch.isfinite(target_tensor).all()):
        raise ValueError("proxy targets must all be finite")
    return target_tensor, rows


def _pearson(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """Return a finite Pearson correlation or zero for a constant vector."""
    prediction_centered = prediction - prediction.mean()
    target_centered = target - target.mean()
    denominator = torch.sqrt(
        prediction_centered.square().sum()
        * target_centered.square().sum()
    )
    if float(denominator) == 0.0:
        return 0.0
    return float(
        (prediction_centered * target_centered).sum() / denominator
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a non-empty list of homogeneous dictionaries to CSV."""
    if not rows:
        raise ValueError(f"cannot write empty CSV {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_loss(metrics: list[dict[str, Any]], output_path: Path) -> None:
    """Plot Q1/Q2/V regression losses for the capacity smoke."""
    steps = [int(row["step"]) for row in metrics]
    figure, axis = plt.subplots(figsize=(10, 6))
    for field, label in (
        ("q1_mse", "Q1 MSE"),
        ("q2_mse", "Q2 MSE"),
        ("v_mse", "V MSE"),
    ):
        axis.plot(
            steps,
            [float(row[field]) for row in metrics],
            label=label,
            linewidth=2.2,
        )
    axis.set_yscale("log")
    axis.set_xlabel("Optimization step", fontsize=14)
    axis.set_ylabel("Batch MSE", fontsize=14)
    axis.set_title("Real-shard geometric critic capacity smoke", fontsize=18)
    axis.tick_params(labelsize=12)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=12)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    """Run real-input forward/backward and a bounded overfit experiment."""
    args = _parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    _configure_logging(args.output_dir)
    torch.manual_seed(1)
    np.random.seed(1)
    torch.set_num_threads(18)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    LOGGER.info(
        "starting shard=%s checkpoint=%s device=%s steps=%d",
        args.shard,
        args.checkpoint,
        device,
        args.steps,
    )

    shard = load_trajectory_shard(args.shard)
    proxy_target_cpu, target_rows = _proxy_targets(
        shard.reconstruction_rows,
        invalid_penalty=args.invalid_penalty,
    )
    valid_count = sum(int(row["rdkit_valid"]) for row in target_rows)
    if valid_count == 0 or valid_count == shard.batch_size:
        raise RuntimeError(
            "capacity smoke requires a shard containing both valid and invalid episodes"
        )
    decision_indices = torch.full(
        (shard.batch_size,),
        shard.final_decision_index,
        dtype=torch.long,
    )
    transitions = shard.select_transitions(
        torch.arange(shard.batch_size),
        decision_indices,
        episode_rewards=proxy_target_cpu,
    ).to(device)

    policy = DiffSBDDModel(
        checkpoint_path=str(args.checkpoint),
        pocket_pdb_path=str(args.pocket_pdb),
        ref_ligand_path=args.ref_ligand,
        device=str(device),
    )
    pocket = PocketContext.from_diffsbdd_mapping(
        policy.export_single_pocket_context()
    )
    del policy
    if device.type == "cuda":
        torch.cuda.empty_cache()

    inputs = build_geometric_critic_input(
        transitions,
        pocket,
        num_diffusion_steps=shard.num_diffusion_steps,
    )
    centered_pocket_com = (
        inputs.pocket_coordinates * inputs.pocket_mask
    ).sum(dim=1) / inputs.pocket_mask.sum(dim=1)
    pocket_com_max_abs = float(centered_pocket_com.abs().max())
    if pocket_com_max_abs > 1e-4:
        raise RuntimeError(
            f"centered pocket COM drift is too large: {pocket_com_max_abs}"
        )

    config = GeometricCriticConfig(
        ligand_feature_dim=int(inputs.state_features.shape[-1]),
        pocket_feature_dim=int(inputs.pocket_features.shape[-1]),
        hidden_dim=args.hidden_dim,
        num_message_layers=args.message_layers,
        num_rbf=8,
        time_embedding_dim=16,
        max_neighbors=args.max_neighbors,
    )
    twin_q = TwinQCritic(config).to(device)
    value = GeometricValueNetwork(config).to(device)
    optimizer = torch.optim.Adam(
        list(twin_q.parameters()) + list(value.parameters()),
        lr=args.learning_rate,
    )
    target = proxy_target_cpu.to(device)
    parameter_count = sum(
        parameter.numel()
        for parameter in list(twin_q.parameters()) + list(value.parameters())
    )
    LOGGER.info(
        "prepared episodes=%d valid=%d invalid=%d ligand_nodes=%d "
        "pocket_nodes=%d parameters=%d pocket_com_max_abs=%.3g",
        shard.batch_size,
        valid_count,
        shard.batch_size - valid_count,
        int(inputs.state_coordinates.shape[1]),
        int(inputs.pocket_coordinates.shape[1]),
        parameter_count,
        pocket_com_max_abs,
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    metrics: list[dict[str, Any]] = []
    started = time.perf_counter()
    for step in range(args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        q1, q2 = twin_q(inputs)
        v = value(inputs)
        q1_mse = F.mse_loss(q1, target)
        q2_mse = F.mse_loss(q2, target)
        v_mse = F.mse_loss(v, target)
        total_loss = q1_mse + q2_mse + v_mse
        metrics.append(
            {
                "step": step,
                "total_loss": float(total_loss.detach()),
                "q1_mse": float(q1_mse.detach()),
                "q2_mse": float(q2_mse.detach()),
                "v_mse": float(v_mse.detach()),
                "q1_pearson": _pearson(q1.detach(), target),
                "q2_pearson": _pearson(q2.detach(), target),
                "v_pearson": _pearson(v.detach(), target),
            }
        )
        if step == args.steps:
            break
        total_loss.backward()
        gradients = [
            parameter.grad
            for parameter in list(twin_q.parameters()) + list(value.parameters())
            if parameter.grad is not None
        ]
        if not gradients or not all(
            bool(torch.isfinite(gradient).all()) for gradient in gradients
        ):
            raise RuntimeError(f"non-finite or missing critic gradients at step {step}")
        optimizer.step()
        if step == 0 or (step + 1) % 20 == 0:
            LOGGER.info(
                "step=%d total_loss=%.6f q1=%.6f q2=%.6f v=%.6f",
                step + 1,
                float(total_loss.detach()),
                float(q1_mse.detach()),
                float(q2_mse.detach()),
                float(v_mse.detach()),
            )

    elapsed_seconds = time.perf_counter() - started
    final = metrics[-1]
    initial = metrics[0]
    final_predictions = {
        "q1": q1.detach().cpu(),
        "q2": q2.detach().cpu(),
        "v": v.detach().cpu(),
    }
    prediction_rows = []
    for episode_index, row in enumerate(target_rows):
        prediction_rows.append(
            {
                **row,
                "q1_prediction": float(final_predictions["q1"][episode_index]),
                "q2_prediction": float(final_predictions["q2"][episode_index]),
                "v_prediction": float(final_predictions["v"][episode_index]),
            }
        )
    final_ratio = float(final["total_loss"]) / max(
        float(initial["total_loss"]),
        1e-12,
    )
    min_final_correlation = min(
        float(final["q1_pearson"]),
        float(final["q2_pearson"]),
        float(final["v_pearson"]),
    )
    passed = final_ratio < 0.05 and min_final_correlation > 0.95
    peak_cuda_bytes = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0
    )
    summary = {
        "status": "passed" if passed else "failed",
        "scope": "capacity_smoke_not_scientific_reward_validation",
        "shard": str(args.shard.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "device": str(device),
        "seed": 1,
        "episodes": shard.batch_size,
        "valid_count": valid_count,
        "invalid_count": shard.batch_size - valid_count,
        "decision_index": shard.final_decision_index,
        "parameter_count": parameter_count,
        "pocket_nodes": int(inputs.pocket_coordinates.shape[1]),
        "ligand_max_nodes": int(inputs.state_coordinates.shape[1]),
        "pocket_com_max_abs": pocket_com_max_abs,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "initial_total_loss": float(initial["total_loss"]),
        "final_total_loss": float(final["total_loss"]),
        "final_to_initial_loss_ratio": final_ratio,
        "final_q1_pearson": float(final["q1_pearson"]),
        "final_q2_pearson": float(final["q2_pearson"]),
        "final_v_pearson": float(final["v_pearson"]),
        "elapsed_seconds": elapsed_seconds,
        "peak_cuda_allocated_bytes": peak_cuda_bytes,
    }
    _write_csv(args.output_dir / "metrics.csv", metrics)
    _write_csv(args.output_dir / "predictions.csv", prediction_rows)
    _plot_loss(metrics, args.output_dir / "critic_overfit_loss.png")
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    torch.save(
        {
            "twin_q": twin_q.state_dict(),
            "value": value.state_dict(),
            "config": config.__dict__,
            "summary": summary,
        },
        args.output_dir / "critic_smoke.pt",
    )
    LOGGER.info("completed summary=%s", json.dumps(summary, ensure_ascii=False))
    if not passed:
        raise RuntimeError(
            "critic capacity smoke did not meet the loss/correlation thresholds"
        )


if __name__ == "__main__":
    main()
