#!/usr/bin/env python3
"""Build an audited episode index and optional explicitly unscored split smoke."""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

from ddpo.offline_rl.index import (
    EpisodeIndex,
    SplitManifest,
    build_episode_index,
    build_split_manifest,
    write_episode_index,
    write_split_manifest,
)


LOGGER = logging.getLogger("OfflineEpisodeIndexBuilder")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-shards-per-rank", type=int, default=None)
    parser.add_argument(
        "--write-unscored-split",
        action="store_true",
        help=(
            "Write a diagnostic split without reward strata. This must not be "
            "used as the frozen formal split."
        ),
    )
    parser.add_argument("--split-version", default="split_smoke_unscored_v1")
    parser.add_argument("--split-salt", default="diffsbdd-offline-iql-v1")
    return parser.parse_args()


def _configure_logging(output_dir: Path) -> None:
    """Write timestamped progress to terminal and a durable local log."""
    output_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    file_handler = logging.FileHandler(
        output_dir / "build_episode_index.log",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(stream_handler)


def _write_summary(
    output_dir: Path,
    episode_index: EpisodeIndex,
    split_manifest: SplitManifest | None,
    *,
    run_dir: Path,
    max_shards_per_rank: int | None,
) -> None:
    """Write summary JSON and presentation-readable diagnostic plots."""
    validity_counts = Counter(
        "valid" if row.rdkit_valid else "invalid"
        for row in episode_index.rows
    )
    node_counts = Counter(row.num_nodes for row in episode_index.rows)
    split_counts: Counter[str] = Counter()
    if split_manifest is not None:
        split_counts.update(row.split for row in split_manifest.rows)
    summary = {
        "status": "completed",
        "scope": (
            "partial_unscored_index_split_smoke"
            if max_shards_per_rank is not None
            else "full_manifest_snapshot_unscored_split_not_formal"
        ),
        "formal_split": False,
        "source_run_dir": str(run_dir.resolve()),
        "source_run_id": episode_index.source_run_id,
        "episodes": len(episode_index.rows),
        "shards": len(
            {
                (row.source_shard_sha256, row.shard_path)
                for row in episode_index.rows
            }
        ),
        "episode_index_sha256": episode_index.sha256,
        "max_shards_per_rank": max_shards_per_rank,
        "validity_counts": dict(sorted(validity_counts.items())),
        "node_counts": {
            str(key): value for key, value in sorted(node_counts.items())
        },
        "split_counts": dict(sorted(split_counts.items())),
        "warning": (
            "No reward strata were used. Do not freeze this diagnostic split "
            "for formal model selection."
        ),
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    validity_labels = ["valid", "invalid"]
    axes[0].bar(
        validity_labels,
        [validity_counts[label] for label in validity_labels],
        color=["#2a9d8f", "#e76f51"],
    )
    axes[0].set_title("RDKit validity in indexed episodes", fontsize=17)
    axes[0].set_ylabel("Episodes", fontsize=15)
    axes[0].tick_params(labelsize=13)
    axes[1].bar(
        [str(node) for node in sorted(node_counts)],
        [node_counts[node] for node in sorted(node_counts)],
        color="#457b9d",
    )
    axes[1].set_title("Ligand node-count distribution", fontsize=17)
    axes[1].set_xlabel("Ligand nodes", fontsize=15)
    axes[1].set_ylabel("Episodes", fontsize=15)
    axes[1].tick_params(axis="x", labelrotation=45, labelsize=11)
    axes[1].tick_params(axis="y", labelsize=13)
    figure.tight_layout()
    figure.savefig(output_dir / "index_smoke_summary.png", dpi=180)
    plt.close(figure)


def main() -> None:
    """Build index artifacts from one collector manifest snapshot."""
    args = _parse_args()
    _configure_logging(args.output_dir)
    LOGGER.info(
        "building index run_dir=%s max_shards_per_rank=%s",
        args.run_dir,
        args.max_shards_per_rank,
    )
    episode_index = build_episode_index(
        args.run_dir,
        max_shards_per_rank=args.max_shards_per_rank,
    )
    index_path = write_episode_index(
        args.output_dir / "episode_index.jsonl",
        episode_index,
    )
    split_manifest = None
    if args.write_unscored_split:
        split_manifest = build_split_manifest(
            episode_index,
            split_version=args.split_version,
            salt=args.split_salt,
        )
        write_split_manifest(
            args.output_dir / "split_manifest.jsonl",
            split_manifest,
        )
    _write_summary(
        args.output_dir,
        episode_index,
        split_manifest,
        run_dir=args.run_dir,
        max_shards_per_rank=args.max_shards_per_rank,
    )
    LOGGER.info(
        "completed episodes=%d index_sha256=%s output=%s",
        len(episode_index.rows),
        episode_index.sha256,
        index_path,
    )


if __name__ == "__main__":
    main()
