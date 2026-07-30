#!/usr/bin/env python3
"""Run a real-index CPU smoke test for the offline transition sampler."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from ddpo.offline_rl.index import load_episode_index, load_split_manifest
from ddpo.offline_rl.sampler import (
    OfflineTransitionSampler,
    default_diffsbdd_time_bins,
)


LOGGER = logging.getLogger("OfflineSamplerSmoke")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--episode-index", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cache-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def _configure_logging(output_dir: Path) -> None:
    """Configure durable file and terminal logging."""
    output_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    for handler in (
        logging.FileHandler(output_dir / "sampler_smoke.log", encoding="utf-8"),
        logging.StreamHandler(),
    ):
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)


def main() -> None:
    """Sample real transitions and write bounded diagnostics."""
    args = _parse_args()
    if args.batches <= 0 or args.batch_size <= 0:
        raise ValueError("batches and batch-size must be positive")
    _configure_logging(args.output_dir)
    episode_index = load_episode_index(args.episode_index)
    split_manifest = load_split_manifest(args.split_manifest)
    time_bins = default_diffsbdd_time_bins()
    sampler = OfflineTransitionSampler(
        run_dir=args.run_dir,
        episode_index=episode_index,
        split_manifest=split_manifest,
        split=args.split,
        time_bins=time_bins,
        cache_size=args.cache_size,
        seed=args.seed,
    )
    LOGGER.info(
        "start split=%s eligible_episodes=%d eligible_shards=%d batches=%d "
        "batch_size=%d",
        args.split,
        sampler.num_eligible_episodes,
        sampler.num_eligible_shards,
        args.batches,
        args.batch_size,
    )
    decision_counts: Counter[int] = Counter()
    time_bin_counts: Counter[str] = Counter()
    final_count = 0
    for batch_index in range(args.batches):
        batch = sampler.sample_batch(args.batch_size)
        if batch.reward is not None:
            raise RuntimeError("unscored sampler smoke unexpectedly returned rewards")
        if not bool(torch.isfinite(batch.state_latent).all()) or not bool(
            torch.isfinite(batch.action_latent).all()
        ):
            raise RuntimeError("sampled transition contains NaN or Inf")
        sampled_decisions = [int(value) for value in batch.decision_index]
        decision_counts.update(sampled_decisions)
        for decision in sampled_decisions:
            matching_bins = [
                time_bin
                for time_bin in time_bins
                if time_bin.start <= decision <= time_bin.end
            ]
            if len(matching_bins) != 1:
                raise RuntimeError(
                    f"decision index {decision} matches {len(matching_bins)} bins"
                )
            time_bin = matching_bins[0]
            time_bin_counts[f"{time_bin.start}-{time_bin.end}"] += 1
        final_count += int(batch.is_final_decode.sum().item())
        if (batch_index + 1) % max(args.batches // 5, 1) == 0:
            LOGGER.info(
                "progress batch=%d/%d cache_hit_rate=%.3f",
                batch_index + 1,
                args.batches,
                sampler.stats().cache_hit_rate,
            )

    stats = sampler.stats()
    summary = {
        "status": "passed",
        "scope": "real_partial_index_unscored_sampler_smoke",
        "formal_training": False,
        "episode_index_sha256": episode_index.sha256,
        "split": args.split,
        "eligible_episodes": sampler.num_eligible_episodes,
        "eligible_shards": sampler.num_eligible_shards,
        "batches": args.batches,
        "batch_size": args.batch_size,
        "transitions_sampled": stats.transitions_sampled,
        "final_transition_count": final_count,
        "cache_hits": stats.cache_hits,
        "cache_misses": stats.cache_misses,
        "cache_hit_rate": stats.cache_hit_rate,
        "shard_load_seconds": stats.shard_load_seconds,
        "sample_seconds": stats.sample_seconds,
        "seed": args.seed,
        "warning": (
            "The source index/split is a partial unscored smoke artifact, not "
            "the frozen formal training split."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "decision_counts.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("decision_index", "count"),
        )
        writer.writeheader()
        writer.writerows(
            {
                "decision_index": decision,
                "count": count,
            }
            for decision, count in sorted(decision_counts.items())
        )
    with (args.output_dir / "time_bin_counts.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("time_bin", "count"),
        )
        writer.writeheader()
        writer.writerows(
            {
                "time_bin": f"{time_bin.start}-{time_bin.end}",
                "count": time_bin_counts[
                    f"{time_bin.start}-{time_bin.end}"
                ],
            }
            for time_bin in time_bins
        )

    figure, axis = plt.subplots(figsize=(11, 5.5))
    labels = [f"{time_bin.start}-{time_bin.end}" for time_bin in time_bins]
    axis.bar(
        labels,
        [time_bin_counts[label] for label in labels],
        color="#457b9d",
    )
    axis.set_title("Sampled DiffSBDD decision-time bins", fontsize=18)
    axis.set_xlabel("Inclusive decision-index bin", fontsize=15)
    axis.set_ylabel("Transitions", fontsize=15)
    axis.tick_params(axis="x", labelrotation=25, labelsize=12)
    axis.tick_params(axis="y", labelsize=13)
    figure.tight_layout()
    figure.savefig(args.output_dir / "decision_sampling_counts.png", dpi=180)
    plt.close(figure)
    LOGGER.info(
        "passed transitions=%d cache_hits=%d cache_misses=%d "
        "cache_hit_rate=%.3f",
        stats.transitions_sampled,
        stats.cache_hits,
        stats.cache_misses,
        stats.cache_hit_rate,
    )


if __name__ == "__main__":
    main()
