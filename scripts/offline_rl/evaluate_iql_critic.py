#!/usr/bin/env python3
"""Evaluate a frozen offline-IQL critic checkpoint on a held-out split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import tempfile
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch


_BOOTSTRAP_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "diffb110_iql_matplotlib"
_BOOTSTRAP_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_BOOTSTRAP_MPLCONFIGDIR))

from ddpo.offline_rl.build import (
    build_offline_iql_runtime,
    load_offline_iql_config,
    resolve_project_path,
    sha256_file,
)
from ddpo.offline_rl.critic_adapter import (
    build_geometric_critic_input,
    build_next_geometric_critic_input,
)
from ddpo.offline_rl.critic_evaluation import (
    CriticPredictionFacts,
    build_calibration_rows,
    build_confounding_metrics,
    build_critic_metric_rows,
    build_quality_order_rows,
    build_stratified_metric_rows,
    fit_critic_baselines,
    regression_metrics,
)
from ddpo.offline_rl.dataset import load_trajectory_shard
from ddpo.offline_rl.index import (
    EpisodeIndexRow,
    load_episode_index,
    load_split_manifest,
)
from ddpo.offline_rl.reward_sidecar import load_reward_sidecar
from ddpo.offline_rl.losses import compute_td_target
from ddpo.runtime import DistributedContext


LOGGER = logging.getLogger("OfflineCriticEvaluation")


def _parse_args() -> argparse.Namespace:
    """Parse immutable inputs and bounded held-out sampling controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--decisions-per-time-bin", type=int, default=1)
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def _configure_logging(output_dir: Path) -> None:
    """Create a durable timestamped evaluation log."""
    output_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    LOGGER.handlers.clear()
    LOGGER.setLevel(logging.INFO)
    for handler in (
        logging.FileHandler(output_dir / "critic_evaluation.log", encoding="utf-8"),
        logging.StreamHandler(),
    ):
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)


def _load_checkpoint(path: Path) -> dict[str, Any]:
    """Load one exact rank checkpoint without assuming a torch version."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("trainer"), dict):
        raise ValueError("offline IQL checkpoint must contain a trainer mapping")
    return payload


def _write_csv(
    path: Path,
    rows: Iterable[Mapping[str, object]],
) -> None:
    """Write homogeneous non-empty row mappings to CSV."""
    materialized = [dict(row) for row in rows]
    if not materialized:
        raise ValueError(f"cannot write empty CSV {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    """Atomically write one JSON result with an fsync boundary."""
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _plot_evaluation(
    rows: tuple[CriticPredictionFacts, ...],
    time_metrics: tuple[dict[str, float | int | str], ...],
    output_path: Path,
) -> None:
    """Plot held-out target agreement and timestep-stratified ranking."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    targets = [row.reward_target for row in rows]
    q_values = [row.q_min for row in rows]
    lower = min((*targets, *q_values))
    upper = max((*targets, *q_values))
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    axes[0].scatter(targets, q_values, alpha=0.45, s=30, color="#287271")
    axes[0].plot([lower, upper], [lower, upper], linestyle="--", color="#d1495b")
    axes[0].set_title("Held-out min-Q vs terminal reward", fontsize=17)
    axes[0].set_xlabel("Terminal training reward", fontsize=14)
    axes[0].set_ylabel("min(Q1, Q2)", fontsize=14)
    axes[0].tick_params(labelsize=12)
    axes[0].grid(alpha=0.2)

    labels = [str(row["time_bin"]) for row in time_metrics]
    spearman = [float(row["spearman"]) for row in time_metrics]
    axes[1].bar(labels, spearman, color="#e09f3e")
    axes[1].axhline(0.0, color="#333333", linewidth=1)
    axes[1].set_ylim(-1.0, 1.0)
    axes[1].set_title("min-Q rank correlation by timestep", fontsize=17)
    axes[1].set_xlabel("Decision-index bin", fontsize=14)
    axes[1].set_ylabel("Spearman", fontsize=14)
    axes[1].tick_params(axis="x", labelrotation=35, labelsize=10)
    axes[1].tick_params(axis="y", labelsize=12)
    axes[1].grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _decision_index(
    *,
    sample_id: str,
    start: int,
    end: int,
    draw_index: int,
    seed: int,
) -> int:
    """Choose one stable per-episode decision without mutable RNG state."""
    digest = hashlib.sha256(
        f"{seed}|{sample_id}|{start}|{end}|{draw_index}".encode("utf-8")
    ).digest()
    return start + int.from_bytes(digest[:8], byteorder="big") % (end - start + 1)


def main() -> None:
    """Load a frozen checkpoint and write denominator-explicit held-out evidence."""
    args = _parse_args()
    if min(
        args.batch_size,
        args.decisions_per_time_bin,
        args.calibration_bins,
        args.seed,
    ) <= 0:
        raise ValueError("sampling, calibration, and seed controls must be positive")
    if args.max_shards is not None and args.max_shards <= 0:
        raise ValueError("max_shards must be positive or omitted")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"evaluation output is not empty: {args.output_dir}")
    matplotlib_config_dir = args.output_dir / "runtime_cache" / "matplotlib"
    matplotlib_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(matplotlib_config_dir)
    _configure_logging(args.output_dir)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but CUDA is unavailable")
    torch.manual_seed(args.seed)

    config_path = args.config.resolve()
    checkpoint_path = args.checkpoint.resolve()
    data_root = args.data_root.resolve()
    config = load_offline_iql_config(config_path)
    dist_ctx = DistributedContext(
        enabled=False,
        rank=0,
        local_rank=0,
        world_size=1,
        device=device,
    )
    LOGGER.info(
        "start config=%s checkpoint=%s split=%s device=%s seed=%d",
        config_path,
        checkpoint_path,
        args.split,
        device,
        args.seed,
    )
    bundle = build_offline_iql_runtime(
        config,
        project_root=data_root,
        dist_ctx=dist_ctx,
    )
    checkpoint = _load_checkpoint(checkpoint_path)
    bundle.trainer.load_state_dict(checkpoint["trainer"])
    bundle.trainer.online_q.eval()
    bundle.trainer.value.eval()

    offline_config = dict(config["offline_iql"])
    data_config = dict(offline_config["data"])
    reward_config = dict(offline_config["reward"])
    index_path = resolve_project_path(data_config["episode_index"], data_root)
    split_path = resolve_project_path(data_config["split_manifest"], data_root)
    sidecar_path = resolve_project_path(reward_config["sidecar"], data_root)
    episode_index = load_episode_index(index_path)
    split_manifest = load_split_manifest(split_path)
    split_manifest.validate_against(episode_index)
    reward_sidecar = load_reward_sidecar(
        sidecar_path,
        expected_reward_version=reward_config.get("expected_version"),
    )
    baseline_facts = fit_critic_baselines(
        episode_index,
        split_manifest,
        reward_sidecar,
        split="train",
    )
    source_run_dir = resolve_project_path(data_config["run_dir"], data_root)
    rows_by_shard: dict[tuple[str, str], list[EpisodeIndexRow]] = defaultdict(list)
    for index_row in episode_index.rows:
        if split_manifest.split_for(index_row.sample_id) == args.split:
            rows_by_shard[
                (index_row.shard_path, index_row.source_shard_sha256.lower())
            ].append(index_row)
    if not rows_by_shard:
        raise ValueError(f"held-out split {args.split!r} contains no episodes")
    selected_shards = sorted(rows_by_shard.items())
    if args.max_shards is not None:
        selected_shards = selected_shards[: args.max_shards]
    selected_episode_count = sum(len(rows) for _, rows in selected_shards)
    predictions: list[CriticPredictionFacts] = []
    started = time.perf_counter()
    for shard_number, ((shard_path, shard_sha256), index_rows) in enumerate(
        selected_shards,
        start=1,
    ):
        shard = load_trajectory_shard(
            source_run_dir / shard_path,
            expected_sha256=shard_sha256,
        )
        joined = reward_sidecar.join_shard(shard)
        transition_requests: list[tuple[EpisodeIndexRow, int, str]] = []
        for index_row in sorted(index_rows, key=lambda row: row.episode_index):
            for time_bin in bundle.sampler.time_bins:
                time_bin_name = f"{time_bin.start}-{time_bin.end}"
                for draw_index in range(args.decisions_per_time_bin):
                    transition_requests.append(
                        (
                            index_row,
                            _decision_index(
                                sample_id=index_row.sample_id,
                                start=time_bin.start,
                                end=time_bin.end,
                                draw_index=draw_index,
                                seed=args.seed,
                            ),
                            time_bin_name,
                        )
                    )
        for request_start in range(0, len(transition_requests), args.batch_size):
            request_batch = transition_requests[
                request_start : request_start + args.batch_size
            ]
            transitions = joined.select_transitions(
                [row.episode_index for row, _, _ in request_batch],
                [decision_index for _, decision_index, _ in request_batch],
            )
            transitions = transitions.to(
                device,
                non_blocking=device.type == "cuda",
            )
            if transitions.episode_return is None:
                raise ValueError("held-out evaluation requires episode_return labels")
            critic_input = build_geometric_critic_input(
                transitions,
                bundle.trainer.pocket,
                num_diffusion_steps=bundle.trainer.config.num_diffusion_steps,
            )
            next_critic_input = build_next_geometric_critic_input(
                transitions,
                bundle.trainer.pocket,
                num_diffusion_steps=bundle.trainer.config.num_diffusion_steps,
            )
            with torch.no_grad():
                q1, q2 = bundle.trainer.online_q(critic_input)
                value = bundle.trainer.value(critic_input)
                next_value = bundle.trainer.value(next_critic_input)
                if transitions.reward is None:
                    raise ValueError("held-out evaluation requires sparse rewards")
                td_target = compute_td_target(
                    transitions.reward,
                    transitions.done,
                    next_value,
                    bundle.trainer.config.discount,
                )
            for row_index, (index_row, decision_index, time_bin_name) in enumerate(
                request_batch
            ):
                sample_id = transitions.sample_ids[row_index]
                if sample_id != index_row.sample_id:
                    raise ValueError("held-out transition/sample order mismatch")
                label = reward_sidecar.get(sample_id)
                predictions.append(
                    CriticPredictionFacts(
                        sample_id=sample_id,
                        time_bin=time_bin_name,
                        decision_index=decision_index,
                        num_nodes=index_row.num_nodes,
                        rdkit_valid=label.rdkit_valid,
                        pb_evaluated=label.pb_evaluated,
                        pb_valid=label.pb_valid,
                        docking_success=label.docking_success,
                        actual_good=label.actual_good,
                        logp=label.logp,
                        reward_target=float(transitions.episode_return[row_index]),
                        td_target=float(td_target[row_index]),
                        q1=float(q1[row_index]),
                        q2=float(q2[row_index]),
                        value=float(value[row_index]),
                    )
                )
        if (
            shard_number == 1
            or shard_number % 100 == 0
            or shard_number == len(selected_shards)
        ):
            elapsed = time.perf_counter() - started
            shards_per_second = shard_number / max(elapsed, 1e-12)
            eta_seconds = (len(selected_shards) - shard_number) / shards_per_second
            LOGGER.info(
                "progress shards=%d/%d episodes=%d/%d predictions=%d "
                "elapsed_seconds=%.1f shards_per_second=%.3f eta_seconds=%.1f",
                shard_number,
                len(selected_shards),
                len({row.sample_id for row in predictions}),
                selected_episode_count,
                len(predictions),
                elapsed,
                shards_per_second,
                eta_seconds,
            )

    prediction_rows = tuple(predictions)
    model_metrics = build_critic_metric_rows(prediction_rows, baseline_facts)
    time_metrics = build_stratified_metric_rows(prediction_rows)
    quality_rows = build_quality_order_rows(prediction_rows)
    calibration_rows = build_calibration_rows(
        prediction_rows,
        bins=args.calibration_bins,
    )
    confounding_metrics = build_confounding_metrics(prediction_rows)
    _write_csv(
        args.output_dir / "predictions.csv",
        (
            {**asdict(row), "q_min": row.q_min, "quality_stratum": row.quality_stratum}
            for row in prediction_rows
        ),
    )
    _write_csv(args.output_dir / "model_metrics.csv", model_metrics)
    _write_csv(args.output_dir / "time_bin_metrics.csv", time_metrics)
    _write_csv(args.output_dir / "quality_order.csv", quality_rows)
    _write_csv(args.output_dir / "calibration.csv", calibration_rows)
    _plot_evaluation(
        prediction_rows,
        time_metrics,
        args.output_dir / "heldout_critic_evaluation.png",
    )
    full_split_episode_count = sum(
        split_manifest.split_for(row.sample_id) == args.split
        for row in episode_index.rows
    )
    q_min_metrics = next(row for row in model_metrics if row["model"] == "q_min")
    bellman_metrics = regression_metrics(
        [row.td_target for row in prediction_rows],
        [row.q_min for row in prediction_rows],
    )
    summary = {
        "status": "completed",
        "scope": "heldout_critic_evaluation_not_actor_quality_claim",
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_global_step": int(checkpoint["trainer"]["global_step"]),
        "episode_index_sha256": episode_index.sha256,
        "split_manifest_sha256": sha256_file(split_path),
        "reward_sidecar_sha256": sha256_file(sidecar_path),
        "reward_version": reward_sidecar.reward_version,
        "split": args.split,
        "split_episode_count": full_split_episode_count,
        "evaluated_episode_count": selected_episode_count,
        "episode_coverage_rate": selected_episode_count / full_split_episode_count,
        "eligible_shards": len(rows_by_shard),
        "evaluated_shards": len(selected_shards),
        "sampled_transitions": len(prediction_rows),
        "sampled_unique_episodes": len({row.sample_id for row in prediction_rows}),
        "sampling_with_replacement": False,
        "batch_size": args.batch_size,
        "decisions_per_time_bin": args.decisions_per_time_bin,
        "time_bins": [asdict(time_bin) for time_bin in bundle.sampler.time_bins],
        "seed": args.seed,
        "device": str(device),
        "baseline_facts": asdict(baseline_facts),
        "q_min_metrics": q_min_metrics,
        "bellman_metrics": bellman_metrics,
        "confounding_metrics": confounding_metrics,
        "q_disagreement_mean": float(
            sum(abs(row.q1 - row.q2) for row in prediction_rows)
            / len(prediction_rows)
        ),
        "scientific_gate_decision": "not_assigned_by_evaluator",
    }
    if not all(
        math.isfinite(float(value))
        for value in q_min_metrics.values()
        if isinstance(value, float)
    ):
        raise FloatingPointError("held-out q_min metrics contain NaN or Inf")
    _atomic_write_json(args.output_dir / "summary.json", summary)
    LOGGER.info("completed summary=%s", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
