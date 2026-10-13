"""Train the existing geometric terminal critic on every binary-labelled row.

This is an isolated diagnosis of whether the current DiffSBDD critic input can
learn the frozen rank-40% valid-only label.  It is not an IQL actor run.  The
trajectory shards are read on CPU once into a compact terminal-transition
cache; geometric graph construction and critic optimization run in GPU
mini-batches.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import random
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


WORKTREE_ROOT = Path("collected_iql_benchmark_100k_260806/iql_worktree").resolve()
sys.path.insert(0, str(WORKTREE_ROOT))

from ddpo.integration.diffsbdd_model import DiffSBDDModel  # noqa: E402
from ddpo.offline_rl.critic_adapter import (  # noqa: E402
    PocketContext,
    build_geometric_critic_input,
)
from ddpo.offline_rl.dataset import load_trajectory_shard  # noqa: E402
from ddpo.offline_rl.losses import compute_q_loss  # noqa: E402
from ddpo.offline_rl.models import GeometricCriticConfig, TwinQCritic  # noqa: E402
from ddpo.offline_rl.protocol import DenoisingTransitionBatch  # noqa: E402


LOGGER = logging.getLogger(__name__)
SPLITS = ("train", "validation", "test")
LABELS = ("preferred", "non_preferred")
TARGETS = {"preferred": 1.0, "non_preferred": -1.0}
DEFAULT_MAX_NODES = 30
TERMINAL_DECISION_INDEX = 1000


def parse_args() -> argparse.Namespace:
    """Parse full-data binary terminal-critic configuration."""
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
        "--output-dir",
        type=Path,
        default=Path(
            "analysis/offline_iql_binary_rank40_diagnosis_260821/"
            "full_binary_geometric_critic"
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Compact terminal cache directory; defaults to output-dir/terminal_cache.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt"),
    )
    parser.add_argument("--pocket-pdb", type=Path, default=Path("data/8GJL.pdb"))
    parser.add_argument("--ref-ligand", default="C:1101")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--eval-every-epochs", type=int, default=1)
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()


def configure_logging(output_dir: Path) -> None:
    """Configure a durable log and rank-zero console progress."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "full_binary_geometric_critic.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a local evidence file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    """Set deterministic Python, NumPy, and Torch seeds."""
    if seed <= 0:
        raise ValueError("seeds must be positive")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_manifest(path: Path) -> pd.DataFrame:
    """Load and validate the frozen valid-only binary manifest."""
    frame = pd.read_csv(path, compression="gzip")
    required = {
        "trajectory_id",
        "trajectory_shard",
        "mol_index",
        "split",
        "binary_label",
        "binary_target",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"manifest is missing required columns: {missing}")
    if frame["trajectory_id"].duplicated().any():
        raise ValueError("manifest contains duplicate trajectory_id values")
    if not set(frame["split"].unique()).issubset(SPLITS):
        raise ValueError("manifest contains an unknown split")
    if not set(frame["binary_label"].unique()).issubset(LABELS):
        raise ValueError("manifest contains an unknown binary label")
    frame = frame.reset_index(drop=True)
    frame["binary_target"] = pd.to_numeric(frame["binary_target"], errors="raise")
    frame["_cache_index"] = -1
    for split in SPLITS:
        indices = frame.index[frame["split"] == split].to_numpy()
        frame.loc[indices, "_cache_index"] = np.arange(len(indices), dtype=np.int64)
    if (frame["_cache_index"] < 0).any():
        raise ValueError("failed to assign compact-cache indices")
    return frame


def _empty_cache(
    frame: pd.DataFrame,
    split: str,
    *,
    max_n_nodes: int,
    latent_dim: int,
) -> dict[str, Any]:
    """Allocate one split's compact CPU terminal cache."""
    subset = frame.loc[frame["split"] == split]
    count = len(subset)
    return {
        "state_latent": torch.empty((count, max_n_nodes, latent_dim), dtype=torch.float32),
        "action_latent": torch.empty((count, max_n_nodes, latent_dim), dtype=torch.float32),
        "nodesxsample": torch.empty((count,), dtype=torch.int64),
        "recorded_behavior_score": torch.empty((count,), dtype=torch.float32),
        "binary_target": torch.tensor(subset["binary_target"].to_numpy(), dtype=torch.float32),
        "mol_index": torch.empty((count,), dtype=torch.int64),
        "sample_ids": [""] * count,
    }


def build_terminal_cache(manifest_path: Path, cache_dir: Path) -> dict[str, dict[str, Any]]:
    """Read every manifest row and cache only its final denoising transition."""
    frame = load_manifest(manifest_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    first_path = Path(str(frame.iloc[0]["trajectory_shard"]))
    first_shard = load_trajectory_shard(first_path)
    max_n_nodes = int(first_shard.max_n_nodes)
    latent_dim = int(first_shard.latents.shape[-1])
    num_diffusion_steps = int(first_shard.num_diffusion_steps)
    if num_diffusion_steps != TERMINAL_DECISION_INDEX:
        raise ValueError(
            "this diagnostic expects the current 1000-step collector; "
            f"found num_diffusion_steps={num_diffusion_steps}"
        )
    caches = {
        split: _empty_cache(
            frame,
            split,
            max_n_nodes=max_n_nodes,
            latent_dim=latent_dim,
        )
        for split in SPLITS
    }
    loaded_rows = 0
    started = time.perf_counter()
    grouped = frame.groupby("trajectory_shard", sort=True)
    for shard_number, (shard_path_value, group) in enumerate(grouped, start=1):
        ordered = group.sort_values("mol_index").reset_index(drop=True)
        shard_path = Path(str(shard_path_value))
        shard = load_trajectory_shard(shard_path)
        if int(shard.max_n_nodes) != max_n_nodes or int(shard.latents.shape[-1]) != latent_dim:
            raise ValueError(f"trajectory shape changed in {shard_path}")
        episode_indices = ordered["mol_index"].astype(int).tolist()
        selected = shard.select_transitions(
            episode_indices,
            [TERMINAL_DECISION_INDEX] * len(episode_indices),
        )
        expected_ids = tuple(str(value) for value in ordered["trajectory_id"])
        if selected.sample_ids != expected_ids:
            raise ValueError(f"sample ordering mismatch in {shard_path}")
        for split in SPLITS:
            split_mask = ordered["split"].eq(split).to_numpy()
            if not bool(split_mask.any()):
                continue
            source_positions = np.flatnonzero(split_mask)
            destination = ordered.loc[split_mask, "_cache_index"].to_numpy(dtype=np.int64)
            target_cache = caches[split]
            target_cache["state_latent"][destination] = selected.state_latent[source_positions]
            target_cache["action_latent"][destination] = selected.action_latent[source_positions]
            target_cache["nodesxsample"][destination] = selected.nodesxsample[source_positions]
            target_cache["recorded_behavior_score"][destination] = selected.recorded_behavior_score[source_positions]
            target_cache["mol_index"][destination] = torch.tensor(
                ordered.loc[split_mask, "mol_index"].to_numpy(dtype=np.int64),
                dtype=torch.int64,
            )
            target_cache["sample_ids"] = [
                str(value) if index not in ("", None) else ""
                for index, value in enumerate(target_cache["sample_ids"])
            ]
            for local_position, cache_position in zip(source_positions, destination):
                target_cache["sample_ids"][int(cache_position)] = expected_ids[int(local_position)]
            loaded_rows += int(source_positions.size)
        if shard_number == 1 or shard_number % 250 == 0 or shard_number == len(grouped):
            elapsed = time.perf_counter() - started
            LOGGER.info(
                "cache progress shards=%d/%d rows=%d/%d elapsed=%.1fs",
                shard_number,
                len(grouped),
                loaded_rows,
                len(frame),
                elapsed,
            )
    if loaded_rows != len(frame):
        raise RuntimeError(f"cache loaded {loaded_rows} rows, expected {len(frame)}")
    for split, cache in caches.items():
        if any(not sample_id for sample_id in cache["sample_ids"]):
            raise RuntimeError(f"cache has missing sample IDs in split={split}")
        torch.save(cache, cache_dir / f"{split}.pt")
    metadata = {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "cache_dir": str(cache_dir),
        "rows": {split: len(caches[split]["sample_ids"]) for split in SPLITS},
        "num_shards": int(frame["trajectory_shard"].nunique()),
        "max_n_nodes": max_n_nodes,
        "latent_dim": latent_dim,
        "num_diffusion_steps": num_diffusion_steps,
        "terminal_decision_index": TERMINAL_DECISION_INDEX,
        "label_definition": (
            "preferred = train-frozen nearest-ChEMBL UMAP distance <= train q40 "
            "and structural_hard_bad == false; non_preferred = every remaining "
            "valid row; invalid rows excluded"
        ),
    }
    (cache_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    LOGGER.info(
        "terminal cache completed rows=%d shards=%d elapsed=%.1fs path=%s",
        len(frame),
        int(frame["trajectory_shard"].nunique()),
        time.perf_counter() - started,
        cache_dir,
    )
    return caches


def load_terminal_cache(cache_dir: Path) -> dict[str, dict[str, Any]]:
    """Load all compact split caches onto CPU."""
    caches: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        path = cache_dir / f"{split}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"missing terminal cache: {path}")
        cache = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(cache, dict):
            raise TypeError(f"invalid cache payload: {path}")
        caches[split] = cache
    return caches


def make_terminal_batch(
    cache: dict[str, Any],
    indices: torch.Tensor,
    device: torch.device,
) -> tuple[DenoisingTransitionBatch, torch.Tensor]:
    """Construct a terminal transition batch and binary targets on one device."""
    if indices.device.type != "cpu":
        indices = indices.cpu()
    state = cache["state_latent"].index_select(0, indices).to(device=device, non_blocking=True)
    action = cache["action_latent"].index_select(0, indices).to(device=device, non_blocking=True)
    nodes = cache["nodesxsample"].index_select(0, indices).to(device=device, non_blocking=True)
    behavior = cache["recorded_behavior_score"].index_select(0, indices).to(device=device, non_blocking=True)
    targets = cache["binary_target"].index_select(0, indices).to(device=device, non_blocking=True)
    batch_size, max_n_nodes, latent_dim = state.shape
    node_ids = torch.arange(max_n_nodes, device=device).unsqueeze(0)
    node_mask = (node_ids < nodes.unsqueeze(1)).to(dtype=state.dtype).unsqueeze(-1)
    action_mask = node_mask.expand_as(action).clone()
    action_mask[:, :, 3:] = 0
    zeros_long = torch.zeros(batch_size, dtype=torch.int64, device=device)
    transition = DenoisingTransitionBatch(
        state_latent=state,
        action_latent=action,
        next_state_latent=action.clone(),
        action_mask=action_mask,
        recorded_behavior_score=behavior,
        recorded_timestep=zeros_long,
        source_diffusion_timestep=zeros_long,
        target_diffusion_timestep=torch.full_like(zeros_long, -1),
        decision_index=torch.full_like(zeros_long, TERMINAL_DECISION_INDEX),
        episode_index=cache["mol_index"].index_select(0, indices).to(device=device, non_blocking=True),
        nodesxsample=nodes,
        node_mask=node_mask,
        done=torch.ones(batch_size, dtype=torch.bool, device=device),
        is_final_decode=torch.ones(batch_size, dtype=torch.bool, device=device),
        sample_ids=tuple(cache["sample_ids"][int(index)] for index in indices.tolist()),
    )
    if int(state.shape[-1]) != latent_dim:
        raise RuntimeError("terminal cache latent dimension changed unexpectedly")
    return transition, targets


def pairwise_preferred_gt_non_preferred(scores: torch.Tensor, targets: torch.Tensor) -> float:
    """Calculate strict preferred-vs-non-preferred pairwise ordering."""
    scores = scores.detach().cpu()
    targets = targets.detach().cpu()
    preferred = scores[targets > 0]
    non_preferred = scores[targets < 0]
    if preferred.numel() == 0 or non_preferred.numel() == 0:
        return float("nan")
    sorted_negative = torch.sort(non_preferred).values
    wins = torch.searchsorted(sorted_negative, preferred, right=False).sum()
    return float(wins.item() / (preferred.numel() * non_preferred.numel()))


def evaluate_split(
    critic: TwinQCritic,
    cache: dict[str, Any],
    pocket: PocketContext,
    *,
    batch_size: int,
    device: torch.device,
    num_diffusion_steps: int,
) -> dict[str, float]:
    """Evaluate one full split in GPU mini-batches."""
    critic.eval()
    count = len(cache["sample_ids"])
    scores: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    loss_sum = 0.0
    q1_sum = 0.0
    q2_sum = 0.0
    disagreement_sum = 0.0
    with torch.no_grad():
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            indices = torch.arange(start, stop, dtype=torch.int64)
            transitions, batch_targets = make_terminal_batch(cache, indices, device)
            inputs = build_geometric_critic_input(
                transitions,
                pocket,
                num_diffusion_steps=num_diffusion_steps,
            )
            q1, q2 = critic(inputs)
            loss = compute_q_loss(q1, q2, batch_targets, "mse")
            score = torch.minimum(q1, q2)
            scores.append(score.cpu())
            targets.append(batch_targets.cpu())
            batch_count = stop - start
            loss_sum += float(loss.item()) * batch_count
            q1_sum += float(q1.mean().item()) * batch_count
            q2_sum += float(q2.mean().item()) * batch_count
            disagreement_sum += float(torch.abs(q1 - q2).mean().item()) * batch_count
    all_scores = torch.cat(scores)
    all_targets = torch.cat(targets)
    preferred_scores = all_scores[all_targets > 0]
    non_preferred_scores = all_scores[all_targets < 0]
    return {
        "rows": float(count),
        "q_loss": loss_sum / count,
        "q1_mean": q1_sum / count,
        "q2_mean": q2_sum / count,
        "q_disagreement_mean": disagreement_sum / count,
        "preferred_q_mean": float(preferred_scores.mean().item()),
        "non_preferred_q_mean": float(non_preferred_scores.mean().item()),
        "preferred_minus_non_preferred": float(
            preferred_scores.mean().item() - non_preferred_scores.mean().item()
        ),
        "pairwise_preferred_gt_non_preferred": pairwise_preferred_gt_non_preferred(
            all_scores,
            all_targets,
        ),
    }


def build_critic_config(cache: dict[str, Any], pocket: PocketContext) -> GeometricCriticConfig:
    """Reuse the existing small geometric critic architecture."""
    latent_dim = int(cache["state_latent"].shape[-1])
    return GeometricCriticConfig(
        ligand_feature_dim=latent_dim - 3,
        pocket_feature_dim=int(pocket.features.shape[-1]),
        hidden_dim=32,
        num_message_layers=1,
        num_rbf=8,
        time_embedding_dim=16,
        max_neighbors=16,
    )


def load_pocket(args: argparse.Namespace, device: torch.device) -> PocketContext:
    """Load the project DiffSBDD pocket representation on the target device."""
    policy = DiffSBDDModel(
        checkpoint_path=str(args.checkpoint),
        pocket_pdb_path=str(args.pocket_pdb),
        ref_ligand_path=args.ref_ligand,
        device=str(device),
        dataset_name="crossdock",
    )
    pocket = PocketContext.from_diffsbdd_mapping(
        policy.export_single_pocket_context()
    ).to(device)
    del policy
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return pocket


def save_checkpoint(
    path: Path,
    critic: TwinQCritic,
    optimizer: torch.optim.Optimizer,
    *,
    seed: int,
    epoch: int,
    critic_config: GeometricCriticConfig,
    validation: dict[str, float],
) -> None:
    """Save a resumable per-epoch diagnostic checkpoint."""
    torch.save(
        {
            "critic_state_dict": critic.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "seed": seed,
            "epoch": epoch,
            "critic_config": critic_config.__dict__,
            "validation": validation,
        },
        path,
    )


def train_seed(
    args: argparse.Namespace,
    caches: dict[str, dict[str, Any]],
    pocket: PocketContext,
    device: torch.device,
    seed: int,
    output_dir: Path,
) -> dict[str, float]:
    """Train one seed over every training row and evaluate held-out splits."""
    set_seed(seed)
    seed_dir = output_dir / f"seed_{seed}"
    (seed_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    config = build_critic_config(caches["train"], pocket)
    critic = TwinQCritic(config).to(device)
    optimizer = torch.optim.Adam(critic.parameters(), lr=args.learning_rate)
    train_cache = caches["train"]
    train_count = len(train_cache["sample_ids"])
    metrics: list[dict[str, Any]] = []
    best_pairwise = -float("inf")
    best_loss = float("inf")
    best_epoch = -1
    best_checkpoint: Path | None = None
    global_step = 0
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        critic.train()
        permutation = torch.randperm(train_count, generator=torch.Generator().manual_seed(seed + epoch))
        epoch_loss = 0.0
        epoch_started = time.perf_counter()
        for start in range(0, train_count, args.batch_size):
            stop = min(start + args.batch_size, train_count)
            indices = permutation[start:stop]
            transitions, targets = make_terminal_batch(train_cache, indices, device)
            inputs = build_geometric_critic_input(
                transitions,
                pocket,
                num_diffusion_steps=TERMINAL_DECISION_INDEX,
            )
            q1, q2 = critic(inputs)
            loss = compute_q_loss(q1, q2, targets, "mse")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            global_step += 1
            batch_count = stop - start
            epoch_loss += float(loss.detach().item()) * batch_count
            if global_step % 100 == 0:
                LOGGER.info(
                    "seed=%d epoch=%d/%d step=%d/%d train_loss=%.6f elapsed=%.1fs",
                    seed,
                    epoch,
                    args.epochs,
                    global_step,
                    int(np.ceil(train_count / args.batch_size)),
                    float(loss.detach().item()),
                    time.perf_counter() - started,
                )
        epoch_loss /= train_count
        if epoch % args.eval_every_epochs == 0 or epoch == args.epochs:
            train_metrics = evaluate_split(
                critic,
                train_cache,
                pocket,
                batch_size=args.eval_batch_size,
                device=device,
                num_diffusion_steps=TERMINAL_DECISION_INDEX,
            )
            validation_metrics = evaluate_split(
                critic,
                caches["validation"],
                pocket,
                batch_size=args.eval_batch_size,
                device=device,
                num_diffusion_steps=TERMINAL_DECISION_INDEX,
            )
            for split, split_metrics in (
                ("train", train_metrics),
                ("validation", validation_metrics),
            ):
                metrics.append(
                    {
                        "seed": seed,
                        "epoch": epoch,
                        "global_step": global_step,
                        "split": split,
                        "epoch_train_loss": epoch_loss,
                        **split_metrics,
                    }
                )
            epoch_checkpoint = seed_dir / "checkpoints" / f"epoch_{epoch:03d}.pt"
            save_checkpoint(
                epoch_checkpoint,
                critic,
                optimizer,
                seed=seed,
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
                best_checkpoint = seed_dir / "best_geometric_critic.pt"
                save_checkpoint(
                    best_checkpoint,
                    critic,
                    optimizer,
                    seed=seed,
                    epoch=epoch,
                    critic_config=config,
                    validation=validation_metrics,
                )
            LOGGER.info(
                "seed=%d epoch=%d train_loss=%.6f train_pairwise=%.4f "
                "val_loss=%.6f val_pairwise=%.4f val_margin=%.4f epoch_time=%.1fs",
                seed,
                epoch,
                epoch_loss,
                train_metrics["pairwise_preferred_gt_non_preferred"],
                val_loss,
                val_pairwise,
                validation_metrics["preferred_minus_non_preferred"],
                time.perf_counter() - epoch_started,
            )

    if best_checkpoint is None or best_epoch < 0:
        raise RuntimeError(f"seed={seed} did not produce a validation checkpoint")
    best_payload = torch.load(best_checkpoint, map_location=device, weights_only=False)
    critic.load_state_dict(best_payload["critic_state_dict"])
    final_rows: list[dict[str, Any]] = []
    for split in SPLITS:
        split_metrics = evaluate_split(
            critic,
            caches[split],
            pocket,
            batch_size=args.eval_batch_size,
            device=device,
            num_diffusion_steps=TERMINAL_DECISION_INDEX,
        )
        final_row = {
            "seed": seed,
            "epoch": best_epoch,
            "global_step": global_step,
            "split": split,
            "epoch_train_loss": float("nan"),
            **split_metrics,
        }
        final_rows.append(final_row)
        metrics.append(final_row)
    metrics_path = seed_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(metrics[0])
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)
    summary = {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_pairwise": best_pairwise,
        "best_validation_q_loss": best_loss,
        "train_final_pairwise": next(row["pairwise_preferred_gt_non_preferred"] for row in final_rows if row["split"] == "train"),
        "validation_final_pairwise": next(row["pairwise_preferred_gt_non_preferred"] for row in final_rows if row["split"] == "validation"),
        "test_final_pairwise": next(row["pairwise_preferred_gt_non_preferred"] for row in final_rows if row["split"] == "test"),
        "test_final_margin": next(row["preferred_minus_non_preferred"] for row in final_rows if row["split"] == "test"),
        "global_steps": global_step,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (seed_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def plot_results(output_dir: Path, summaries: list[dict[str, float]]) -> None:
    """Plot validation curves and final held-out pairwise results."""
    rows: list[dict[str, Any]] = []
    for seed_dir in sorted(output_dir.glob("seed_*/metrics.csv")):
        rows.extend(pd.read_csv(seed_dir).to_dict(orient="records"))
    if not rows:
        return
    frame = pd.DataFrame(rows)
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.8), constrained_layout=True)
    validation = frame[frame["split"] == "validation"]
    for seed, group in validation.groupby("seed"):
        history = group[group["epoch_train_loss"].notna()]
        axes[0].plot(
            history["epoch"],
            history["pairwise_preferred_gt_non_preferred"],
            marker="o",
            linewidth=2.0,
            label=f"seed {int(seed)}",
        )
    axes[0].axhline(0.5, color="#555555", linestyle="--", linewidth=1.5)
    axes[0].set_title("Full-data validation pairwise ordering", fontsize=16)
    axes[0].set_xlabel("Epoch", fontsize=13)
    axes[0].set_ylabel("preferred > non-preferred", fontsize=13)
    axes[0].set_ylim(0, 1.02)
    axes[0].legend(fontsize=11)
    final_frame = frame[frame["epoch_train_loss"].isna()].copy()
    final_frame["seed_label"] = final_frame["seed"].map(lambda value: f"seed {int(value)}")
    test = final_frame[final_frame["split"] == "test"]
    positions = np.arange(len(test))
    axes[1].bar(
        positions,
        test["pairwise_preferred_gt_non_preferred"],
        color="#2F6B9A",
        alpha=0.86,
    )
    axes[1].axhline(0.5, color="#555555", linestyle="--", linewidth=1.5)
    axes[1].set_xticks(positions, test["seed_label"])
    axes[1].set_title("Held-out test pairwise ordering", fontsize=16)
    axes[1].set_ylabel("preferred > non-preferred", fontsize=13)
    axes[1].set_ylim(0, 1.02)
    for position, value in zip(positions, test["pairwise_preferred_gt_non_preferred"]):
        axes[1].text(position, float(value) + 0.025, f"{value:.3f}", ha="center", fontsize=12)
    for axis in axes:
        axis.grid(axis="y", alpha=0.22)
        axis.tick_params(labelsize=11)
    figure.suptitle(
        "Rank-40% binary label: full valid-data geometric terminal critic",
        fontsize=19,
    )
    figure.savefig(output_dir / "full_binary_geometric_critic_results.png", dpi=190)
    figure.savefig(output_dir / "full_binary_geometric_critic_results.svg")
    plt.close(figure)
    pd.DataFrame(summaries).to_csv(output_dir / "seed_summary.csv", index=False)


def main() -> None:
    """Build the all-row cache and run the GPU mini-batch diagnosis."""
    args = parse_args()
    configure_logging(args.output_dir)
    if args.epochs <= 0 or args.batch_size <= 0 or args.eval_batch_size <= 0:
        raise ValueError("epochs and batch sizes must be positive")
    if args.eval_every_epochs <= 0:
        raise ValueError("eval-every-epochs must be positive")
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA requested but unavailable; do not silently run CPU")
    device = torch.device(args.device)
    cache_dir = args.cache_dir or (args.output_dir / "terminal_cache")
    seeds = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
    if not seeds or any(seed <= 0 for seed in seeds):
        raise ValueError("--seeds must contain positive integers")
    LOGGER.info(
        "starting full-data binary terminal critic device=%s gpu=%s seeds=%s",
        device,
        torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        seeds,
    )
    if args.rebuild_cache or not (cache_dir / "metadata.json").is_file():
        caches = build_terminal_cache(args.manifest, cache_dir)
    else:
        caches = load_terminal_cache(cache_dir)
        LOGGER.info("reusing terminal cache path=%s", cache_dir)
    LOGGER.info(
        "all-row cache counts=%s",
        {split: len(caches[split]["sample_ids"]) for split in SPLITS},
    )
    pocket = load_pocket(args, device)
    summaries = [
        train_seed(args, caches, pocket, device, seed, args.output_dir)
        for seed in seeds
    ]
    plot_results(args.output_dir, summaries)
    run_metadata = {
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "pocket_pdb": str(args.pocket_pdb),
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "seeds": seeds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.learning_rate,
        "cache_dir": str(cache_dir),
        "counts": {split: len(caches[split]["sample_ids"]) for split in SPLITS},
        "valid_only": True,
        "actor_started": False,
        "summaries": summaries,
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("completed full-data binary terminal critic; actor_started=False")


if __name__ == "__main__":
    main()
