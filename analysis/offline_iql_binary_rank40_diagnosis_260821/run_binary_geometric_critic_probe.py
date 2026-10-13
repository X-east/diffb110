"""Run a selected-time outcome probe with the existing DiffSBDD twin-Q critic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


WORKTREE_ROOT = Path(
    "collected_iql_benchmark_100k_260806/iql_worktree"
).resolve()
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
LABELS = ("preferred", "non_preferred")
TARGETS = {"preferred": 1.0, "non_preferred": -1.0}
DECISION_INDICES = (0, 250, 500, 750, 900, 990, 1000)


def parse_args() -> argparse.Namespace:
    """Parse probe configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("analysis/offline_iql_scaffold_split_260821/scaffold_split_manifest.csv.gz"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/offline_iql_geometric_critic_state_probe_260821"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt"),
    )
    parser.add_argument("--pocket-pdb", type=Path, default=Path("data/8GJL.pdb"))
    parser.add_argument("--ref-ligand", default="C:1101")
    parser.add_argument("--per-label", type=int, default=64)
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eval-every", type=int, default=25)
    return parser.parse_args()


def configure_logging(output_dir: Path) -> None:
    """Configure durable rank-agnostic probe logging."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "probe.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def sha256(path: Path) -> str:
    """Hash an input file for the local manifest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_rows(
    frame: pd.DataFrame,
    split: str,
    per_label: int,
    seed: int,
) -> pd.DataFrame:
    """Select a deterministic balanced trajectory sample for one split."""
    rng = np.random.default_rng(seed)
    selected: list[pd.DataFrame] = []
    for label in LABELS:
        candidates = frame[(frame["split"] == split) & (frame["binary_label"] == label)]
        if len(candidates) < per_label:
            raise ValueError(
                f"split={split} label={label} has {len(candidates)} rows, "
                f"less than requested {per_label}"
            )
        indices = rng.choice(len(candidates), size=per_label, replace=False)
        selected.append(candidates.iloc[np.sort(indices)])
    return pd.concat(selected, ignore_index=True)


def concat_batches(
    batches: list[DenoisingTransitionBatch],
) -> DenoisingTransitionBatch:
    """Concatenate structural transition batches while keeping them reward-free."""
    if not batches:
        raise ValueError("cannot concatenate an empty batch list")
    fields = (
        "state_latent",
        "action_latent",
        "next_state_latent",
        "action_mask",
        "recorded_behavior_score",
        "recorded_timestep",
        "source_diffusion_timestep",
        "target_diffusion_timestep",
        "decision_index",
        "episode_index",
        "nodesxsample",
        "node_mask",
        "done",
        "is_final_decode",
    )
    values = {
        name: torch.cat([getattr(batch, name) for batch in batches], dim=0)
        for name in fields
    }
    return DenoisingTransitionBatch(
        **values,
        sample_ids=tuple(
            sample_id for batch in batches for sample_id in batch.sample_ids
        ),
    )


def load_probe_batch(
    rows: pd.DataFrame,
    *,
    decision_indices: tuple[int, ...],
) -> tuple[DenoisingTransitionBatch, torch.Tensor, list[str], list[int]]:
    """Load selected states/actions and replicate each label across time points."""
    grouped: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for row in rows.itertuples(index=False):
        for decision_index in decision_indices:
            grouped[str(row.trajectory_shard)].append(
                (int(row.mol_index), str(row.binary_label), decision_index)
            )

    batches: list[DenoisingTransitionBatch] = []
    labels: list[str] = []
    times: list[int] = []
    for shard_path, selections in grouped.items():
        shard = load_trajectory_shard(Path(shard_path))
        episode_indices = [selection[0] for selection in selections]
        decisions = [selection[2] for selection in selections]
        batch = shard.select_transitions(episode_indices, decisions)
        expected_ids = [shard.sample_ids[index] for index in episode_indices]
        if tuple(expected_ids) != batch.sample_ids:
            raise RuntimeError(f"sample ordering mismatch in {shard_path}")
        batches.append(batch)
        labels.extend(selection[1] for selection in selections)
        times.extend(selection[2] for selection in selections)

    batch = concat_batches(batches)
    targets = torch.tensor([TARGETS[label] for label in labels], dtype=torch.float32)
    if len(labels) != batch.batch_size:
        raise RuntimeError("label count does not match transition count")
    return batch, targets, labels, times


def load_pocket(args: argparse.Namespace, device: torch.device) -> PocketContext:
    """Load the same DiffSBDD pocket context used by the IQL route."""
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


def evaluate(
    critic: TwinQCritic,
    inputs: object,
    targets: torch.Tensor,
    labels: list[str],
    times: list[int],
    *,
    split: str,
    update: int,
) -> list[dict[str, object]]:
    """Evaluate binary Q ordering by time and preferred label."""
    with torch.no_grad():
        q1, q2 = critic(inputs)
        prediction = torch.minimum(q1, q2)
        loss = compute_q_loss(q1, q2, targets, "mse")
    rows: list[dict[str, object]] = []
    for decision_index in DECISION_INDICES:
        mask = torch.tensor(
            [time == decision_index for time in times],
            dtype=torch.bool,
            device=prediction.device,
        )
        values = {
            label: prediction[mask & torch.tensor(
                [item == label for item in labels],
                dtype=torch.bool,
                device=prediction.device,
            )]
            for label in LABELS
        }
        if any(value.numel() == 0 for value in values.values()):
            continue
        preferred = values["preferred"]
        non_preferred = values["non_preferred"]
        rows.append(
            {
                "update": update,
                "split": split,
                "decision_index": decision_index,
                "q_loss": float(loss.detach().cpu()),
                "preferred_q_mean": float(preferred.mean().cpu()),
                "non_preferred_q_mean": float(non_preferred.mean().cpu()),
                "pairwise_preferred_gt_non_preferred": float(
                    (preferred[:, None] > non_preferred[None, :]).float().mean().cpu()
                ),
            }
        )
    return rows


def main() -> None:
    """Train and evaluate the existing geometric twin-Q critic."""
    args = parse_args()
    configure_logging(args.output_dir)
    if args.per_label <= 0 or args.updates <= 0:
        raise ValueError("per-label and updates must be positive")
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA requested but unavailable; record an explicit CPU run instead")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    frame = pd.read_csv(args.manifest, compression="gzip")
    for required in ("trajectory_shard", "mol_index", "split", "binary_label"):
        if required not in frame.columns:
            raise ValueError(f"manifest missing required column: {required}")

    sampled = {
        split: select_rows(frame, split, args.per_label, args.seed + index)
        for index, split in enumerate(("train", "validation", "test"))
    }
    LOGGER.info(
        "selected rows per split: %s",
        {split: len(rows) for split, rows in sampled.items()},
    )
    batches: dict[str, DenoisingTransitionBatch] = {}
    targets: dict[str, torch.Tensor] = {}
    labels: dict[str, list[str]] = {}
    times: dict[str, list[int]] = {}
    for split, rows in sampled.items():
        batch, target, split_labels, split_times = load_probe_batch(
            rows,
            decision_indices=DECISION_INDICES,
        )
        batches[split] = batch.to(device)
        targets[split] = target.to(device)
        labels[split] = split_labels
        times[split] = split_times
        LOGGER.info(
            "%s transitions=%d unique_samples=%d",
            split,
            batch.batch_size,
            len(set(batch.sample_ids)),
        )

    pocket = load_pocket(args, device)
    critic_config = GeometricCriticConfig(
        ligand_feature_dim=int(batches["train"].state_latent.shape[-1]) - 3,
        pocket_feature_dim=int(pocket.features.shape[-1]),
        hidden_dim=32,
        num_message_layers=1,
        num_rbf=8,
        time_embedding_dim=16,
        max_neighbors=16,
    )
    critic = TwinQCritic(critic_config).to(device).train()
    inputs = {
        split: build_geometric_critic_input(
            batch,
            pocket,
            num_diffusion_steps=1000,
        )
        for split, batch in batches.items()
    }
    optimizer = torch.optim.Adam(critic.parameters(), lr=args.learning_rate)
    metrics: list[dict[str, object]] = []
    best_val = (-float("inf"), float("inf"), -1)
    best_state: dict[str, torch.Tensor] | None = None

    for update in range(args.updates + 1):
        should_evaluate = (
            update == 0
            or update % args.eval_every == 0
            or update == args.updates
        )
        if should_evaluate:
            critic.eval()
            for split in ("train", "validation", "test"):
                metrics.extend(
                    evaluate(
                        critic,
                        inputs[split],
                        targets[split],
                        labels[split],
                        times[split],
                        split=split,
                        update=update,
                    )
                )
            current = [
                row for row in metrics
                if row["update"] == update and row["split"] == "validation"
            ]
            if current:
                val_pairwise = float(
                    np.mean(
                        [row["pairwise_preferred_gt_non_preferred"] for row in current]
                    )
                )
                val_loss = float(np.mean([row["q_loss"] for row in current]))
                if (val_pairwise, -val_loss) > (best_val[0], -best_val[1]):
                    best_val = (val_pairwise, val_loss, update)
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in critic.state_dict().items()
                    }
                LOGGER.info(
                    "update=%d val_pairwise=%.4f val_q_loss=%.6f",
                    update,
                    val_pairwise,
                    val_loss,
                )
        if update == args.updates:
            break
        critic.train()
        q1, q2 = critic(inputs["train"])
        loss = compute_q_loss(q1, q2, targets["train"], "mse")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if update % args.eval_every == 0:
            LOGGER.info("update=%d train_q_loss=%.6f", update, float(loss.detach().cpu()))

    if best_state is None:
        raise RuntimeError("no validation checkpoint was selected")
    critic.load_state_dict(best_state)
    final_metrics: list[dict[str, object]] = []
    critic.eval()
    for split in ("train", "validation", "test"):
        final_metrics.extend(
            evaluate(
                critic,
                inputs[split],
                targets[split],
                labels[split],
                times[split],
                split=split,
                update=best_val[2],
            )
        )
    metrics_path = args.output_dir / "probe_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(final_metrics[0])
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_metrics)
    torch.save(
        {
            "critic_state_dict": critic.state_dict(),
            "critic_config": critic_config.__dict__,
            "best_validation_pairwise": best_val[0],
            "best_validation_q_loss": best_val[1],
            "best_update": best_val[2],
            "decision_indices": DECISION_INDICES,
        },
        args.output_dir / "best_geometric_critic.pt",
    )
    manifest = {
        "purpose": "selected_time_outcome_probe_existing_diffsbdd_geometric_twin_q",
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "seed": args.seed,
        "decision_indices": DECISION_INDICES,
        "per_label_per_split": args.per_label,
        "target_map": TARGETS,
        "label_definition": (
            "binary_rank40_manifest; train-frozen distance q40 plus structural-hard-bad exclusion; "
            "valid rows only; invalid branch not included"
        ),
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "pocket_pdb": str(args.pocket_pdb),
        "best_validation_pairwise_mean": best_val[0],
        "best_validation_q_loss": best_val[1],
        "best_update": best_val[2],
        "train_rows": len(sampled["train"]),
        "validation_rows": len(sampled["validation"]),
        "test_rows": len(sampled["test"]),
    }
    (args.output_dir / "probe_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )

    metric_frame = pd.DataFrame(final_metrics)
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
    test = metric_frame[metric_frame["split"] == "test"]
    for label, color in zip(LABELS, ("#2F6B9A", "#C96B3C")):
        axes[0].plot(
            test["decision_index"],
            test[f"{label}_q_mean"],
            marker="o",
            linewidth=2.2,
            label=label,
            color=color,
        )
    axes[0].set_title("Held-out Q means by decision index", fontsize=15)
    axes[0].set_xlabel("Decision index", fontsize=13)
    axes[0].set_ylabel("min(Q1, Q2)", fontsize=13)
    axes[0].legend(fontsize=11)
    axes[1].plot(
        test["decision_index"],
        test["pairwise_preferred_gt_non_preferred"],
        marker="o",
        linewidth=2.2,
        label="preferred > non-preferred",
        color="#2F6B9A",
    )
    axes[1].axhline(0.5, color="#555555", linestyle="--", linewidth=1.5)
    axes[1].set_title("Held-out pairwise ordering", fontsize=15)
    axes[1].set_xlabel("Decision index", fontsize=13)
    axes[1].set_ylabel("Pairwise probability", fontsize=13)
    axes[1].set_ylim(0, 1.02)
    axes[1].legend(fontsize=10)
    for axis in axes:
        axis.grid(axis="y", alpha=0.22)
        axis.tick_params(labelsize=11)
    figure.suptitle("Rank-40% binary geometric critic: selected-time outcome probe", fontsize=18)
    figure.savefig(args.output_dir / "state_quality_probe.png", dpi=180)
    plt.close(figure)
    LOGGER.info("completed best_update=%d val_pairwise=%.4f test_rows=%d", best_val[2], best_val[0], len(final_metrics))


if __name__ == "__main__":
    main()
