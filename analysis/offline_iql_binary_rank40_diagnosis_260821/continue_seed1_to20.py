"""Continue the full-data rank-40 binary critic from seed-1 epoch five.

This continuation is intentionally separate from the original three-seed
launcher.  It reuses the frozen terminal cache and the completed epoch-5
checkpoint, so only seed 1 is extended to a target epoch of 20.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_full_binary_geometric_critic as base  # noqa: E402


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse seed-1 continuation settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "analysis/offline_iql_binary_rank40_diagnosis_260821/"
            "binary_rank40_manifest.csv.gz"
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(
            "analysis/offline_iql_binary_rank40_diagnosis_260821/"
            "full_binary_geometric_critic/terminal_cache"
        ),
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=Path(
            "analysis/offline_iql_binary_rank40_diagnosis_260821/"
            "full_binary_geometric_critic/seed_1/checkpoints/epoch_005.pt"
        ),
    )
    parser.add_argument(
        "--old-metrics",
        type=Path,
        default=Path(
            "analysis/offline_iql_binary_rank40_diagnosis_260821/"
            "full_binary_geometric_critic/seed_1/metrics.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "analysis/offline_iql_binary_rank40_diagnosis_260821/"
            "full_binary_geometric_critic/seed_1_extended"
        ),
    )
    parser.add_argument("--target-epoch", type=int, default=20)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt"),
    )
    parser.add_argument("--pocket-pdb", type=Path, default=Path("data/8GJL.pdb"))
    parser.add_argument("--ref-ligand", default="C:1101")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    """Move loaded Adam state tensors onto the critic device."""
    for state in optimizer.state.values():
        for key, value in tuple(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device=device)


def load_previous_metrics(path: Path) -> list[dict[str, Any]]:
    """Load the completed seed-1 history for a continuous report."""
    if not path.is_file():
        return []
    return pd.read_csv(path).to_dict(orient="records")


def write_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write all continuation and inherited history rows."""
    if not rows:
        raise ValueError("cannot write an empty metric history")
    fieldnames: list[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_history(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    """Plot the continuous seed-1 train/validation ranking history."""
    frame = pd.DataFrame(rows)
    history = frame[frame["split"] == "validation"].copy()
    history = history[history["epoch_train_loss"].notna()]
    if history.empty:
        return
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    axes[0].plot(
        history["epoch"],
        history["pairwise_preferred_gt_non_preferred"],
        marker="o",
        linewidth=2.2,
        color="#2F6B9A",
    )
    axes[0].axhline(0.5, color="#555555", linestyle="--", linewidth=1.4)
    axes[0].set_title("Seed 1 validation pairwise by epoch", fontsize=16)
    axes[0].set_xlabel("Epoch", fontsize=13)
    axes[0].set_ylabel("preferred > non-preferred", fontsize=13)
    axes[0].set_ylim(0, 1.02)
    axes[1].plot(
        history["epoch"],
        history["q_loss"],
        marker="o",
        linewidth=2.2,
        color="#C96B3C",
    )
    axes[1].set_title("Seed 1 validation Q loss by epoch", fontsize=16)
    axes[1].set_xlabel("Epoch", fontsize=13)
    axes[1].set_ylabel("Q loss", fontsize=13)
    for axis in axes:
        axis.grid(axis="y", alpha=0.22)
        axis.tick_params(labelsize=11)
    figure.suptitle(
        "Rank-40 binary geometric critic: seed-1 continuation to epoch 20",
        fontsize=19,
    )
    figure.savefig(output_dir / "seed1_extended_training.png", dpi=190)
    figure.savefig(output_dir / "seed1_extended_training.svg")
    plt.close(figure)


def main() -> None:
    """Resume seed 1 and train through the requested target epoch."""
    args = parse_args()
    base.configure_logging(args.output_dir)
    if args.target_epoch <= 0 or args.batch_size <= 0 or args.eval_batch_size <= 0:
        raise ValueError("target epoch and batch sizes must be positive")
    if args.seed <= 0:
        raise ValueError("seed must be positive")
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    caches = base.load_terminal_cache(args.cache_dir)
    payload = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
    resume_epoch = int(payload["epoch"])
    if resume_epoch >= args.target_epoch:
        raise ValueError(
            f"resume checkpoint is already at epoch {resume_epoch}, "
            f"target={args.target_epoch}"
        )
    if int(payload.get("seed", args.seed)) != args.seed:
        raise ValueError("resume checkpoint seed does not match requested seed")
    config = base.GeometricCriticConfig(**payload["critic_config"])
    pocket = base.load_pocket(args, device)
    critic = base.TwinQCritic(config).to(device)
    critic.load_state_dict(payload["critic_state_dict"])
    optimizer = torch.optim.Adam(critic.parameters(), lr=args.learning_rate)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    move_optimizer_state_to_device(optimizer, device)
    for group in optimizer.param_groups:
        group["lr"] = args.learning_rate

    train_cache = caches["train"]
    train_count = len(train_cache["sample_ids"])
    steps_per_epoch = int(np.ceil(train_count / args.batch_size))
    global_step = resume_epoch * steps_per_epoch
    rows = load_previous_metrics(args.old_metrics)
    best_pairwise = float(payload["validation"]["pairwise_preferred_gt_non_preferred"])
    best_loss = float(payload["validation"]["q_loss"])
    best_epoch = resume_epoch
    started = time.perf_counter()
    LOGGER.info(
        "starting seed=%d continuation resume_epoch=%d target_epoch=%d "
        "lr=%.6g device=%s gpu=%s train_rows=%d",
        args.seed,
        resume_epoch,
        args.target_epoch,
        args.learning_rate,
        device,
        torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        train_count,
    )

    for epoch in range(resume_epoch + 1, args.target_epoch + 1):
        critic.train()
        generator = torch.Generator(device="cpu").manual_seed(args.seed + epoch)
        permutation = torch.randperm(train_count, generator=generator)
        epoch_loss = 0.0
        epoch_started = time.perf_counter()
        for start in range(0, train_count, args.batch_size):
            stop = min(start + args.batch_size, train_count)
            indices = permutation[start:stop]
            transitions, targets = base.make_terminal_batch(train_cache, indices, device)
            inputs = base.build_geometric_critic_input(
                transitions,
                pocket,
                num_diffusion_steps=base.TERMINAL_DECISION_INDEX,
            )
            q1, q2 = critic(inputs)
            loss = base.compute_q_loss(q1, q2, targets, "mse")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            global_step += 1
            batch_count = stop - start
            epoch_loss += float(loss.detach().item()) * batch_count
            if global_step % 100 == 0:
                LOGGER.info(
                    "seed=%d epoch=%d/%d step=%d train_loss=%.6f elapsed=%.1fs",
                    args.seed,
                    epoch,
                    args.target_epoch,
                    global_step,
                    float(loss.detach().item()),
                    time.perf_counter() - started,
                )
        epoch_loss /= train_count
        train_metrics = base.evaluate_split(
            critic,
            train_cache,
            pocket,
            batch_size=args.eval_batch_size,
            device=device,
            num_diffusion_steps=base.TERMINAL_DECISION_INDEX,
        )
        validation_metrics = base.evaluate_split(
            critic,
            caches["validation"],
            pocket,
            batch_size=args.eval_batch_size,
            device=device,
            num_diffusion_steps=base.TERMINAL_DECISION_INDEX,
        )
        for split, split_metrics in (
            ("train", train_metrics),
            ("validation", validation_metrics),
        ):
            rows.append(
                {
                    "seed": args.seed,
                    "epoch": epoch,
                    "global_step": global_step,
                    "split": split,
                    "epoch_train_loss": epoch_loss,
                    **split_metrics,
                }
            )
        checkpoint_path = args.output_dir / "checkpoints" / f"epoch_{epoch:03d}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        base.save_checkpoint(
            checkpoint_path,
            critic,
            optimizer,
            seed=args.seed,
            epoch=epoch,
            critic_config=config,
            validation=validation_metrics,
        )
        val_pairwise = validation_metrics["pairwise_preferred_gt_non_preferred"]
        val_loss = validation_metrics["q_loss"]
        if val_pairwise > best_pairwise or (
            val_pairwise == best_pairwise and val_loss < best_loss
        ):
            best_pairwise = val_pairwise
            best_loss = val_loss
            best_epoch = epoch
            base.save_checkpoint(
                args.output_dir / "best_geometric_critic.pt",
                critic,
                optimizer,
                seed=args.seed,
                epoch=epoch,
                critic_config=config,
                validation=validation_metrics,
            )
        LOGGER.info(
            "seed=%d epoch=%d train_loss=%.6f train_pairwise=%.4f "
            "val_loss=%.6f val_pairwise=%.4f val_margin=%.4f "
            "epoch_time=%.1fs best_epoch=%d",
            args.seed,
            epoch,
            epoch_loss,
            train_metrics["pairwise_preferred_gt_non_preferred"],
            val_loss,
            val_pairwise,
            validation_metrics["preferred_minus_non_preferred"],
            time.perf_counter() - epoch_started,
            best_epoch,
        )

    if not (args.output_dir / "best_geometric_critic.pt").is_file():
        base.save_checkpoint(
            args.output_dir / "best_geometric_critic.pt",
            critic,
            optimizer,
            seed=args.seed,
            epoch=args.target_epoch,
            critic_config=config,
            validation=validation_metrics,
        )
    best_payload = torch.load(
        args.output_dir / "best_geometric_critic.pt",
        map_location=device,
        weights_only=False,
    )
    critic.load_state_dict(best_payload["critic_state_dict"])
    final_rows: list[dict[str, Any]] = []
    for split in base.SPLITS:
        split_metrics = base.evaluate_split(
            critic,
            caches[split],
            pocket,
            batch_size=args.eval_batch_size,
            device=device,
            num_diffusion_steps=base.TERMINAL_DECISION_INDEX,
        )
        final_row = {
            "seed": args.seed,
            "epoch": best_epoch,
            "global_step": global_step,
            "split": split,
            "epoch_train_loss": float("nan"),
            **split_metrics,
        }
        final_rows.append(final_row)
        rows.append(final_row)
    write_metrics(args.output_dir / "metrics.csv", rows)
    plot_history(args.output_dir, rows)
    summary = {
        "seed": args.seed,
        "resume_epoch": resume_epoch,
        "target_epoch": args.target_epoch,
        "best_epoch": best_epoch,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "best_validation_pairwise": best_pairwise,
        "best_validation_q_loss": best_loss,
        "final_rows": final_rows,
        "global_steps": global_step,
        "elapsed_seconds": time.perf_counter() - started,
        "actor_started": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    LOGGER.info(
        "completed seed=%d target_epoch=%d best_epoch=%d "
        "best_val_pairwise=%.4f test_pairwise=%.4f actor_started=False",
        args.seed,
        args.target_epoch,
        best_epoch,
        best_pairwise,
        next(row["pairwise_preferred_gt_non_preferred"] for row in final_rows if row["split"] == "test"),
    )


if __name__ == "__main__":
    main()
