"""Build readable rank-40 filtered-BC inputs from the frozen trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IQL_WORKTREE = (
    PROJECT_ROOT / "collected_iql_benchmark_100k_260806/iql_worktree"
)
if str(IQL_WORKTREE) not in sys.path:
    sys.path.insert(0, str(IQL_WORKTREE))

from ddpo.offline_rl.dataset import TRAJECTORY_SHARD_FORMAT  # noqa: E402
from ddpo.offline_rl.index import (  # noqa: E402
    EpisodeIndexProgressFacts,
    build_episode_index,
    build_split_manifest,
    write_episode_index,
    write_split_manifest,
)
from ddpo.offline_rl.reward_sidecar import (  # noqa: E402
    REWARD_SIDECAR_SCHEMA_VERSION,
    EpisodeRewardFacts,
    write_reward_sidecar,
)


LOGGER = logging.getLogger("rank40_filtered_bc_inputs")
REWARD_VERSION = "rank40_binary_allpreferred_one_nonpreferred_zero_v1"


def _parse_args() -> argparse.Namespace:
    """Parse immutable source and output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=PROJECT_ROOT
        / "analysis/offline_iql_frozen_original_100k_260819/run_view",
    )
    parser.add_argument(
        "--binary-manifest",
        type=Path,
        default=PROJECT_ROOT
        / "analysis/offline_iql_binary_rank40_diagnosis_260821"
        / "binary_rank40_manifest.csv.gz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "analysis/offline_iql_filtered_bc_rank40_260822/inputs",
    )
    parser.add_argument("--max-shards-per-rank", type=int, default=1250)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    """Hash one generated or source artifact in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _configure_logging(output_dir: Path) -> None:
    """Write durable progress while retaining rank-zero terminal updates."""
    output_dir.mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.FileHandler(output_dir / "prepare.log", encoding="utf-8"),
        logging.StreamHandler(),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def _progress(facts: EpisodeIndexProgressFacts) -> None:
    """Log bounded full-index reconstruction progress."""
    LOGGER.info(
        "index shards=%d/%d episodes=%d elapsed=%.1fs eta=%.1fs",
        facts.completed_shards,
        facts.total_shards,
        facts.indexed_episodes,
        facts.elapsed_seconds,
        facts.eta_seconds,
    )


def _write_plot(frame: pd.DataFrame, output_dir: Path) -> None:
    """Plot the exact preferred/non-preferred and split counts."""
    counts = pd.crosstab(frame["split"], frame["binary_label"]).reindex(
        index=["train", "validation", "test"], fill_value=0
    )
    axis = counts.plot.bar(figsize=(11, 7), color=["#D99A2B", "#3977A8"])
    axis.set_title("Rank-40 binary trajectory labels", fontsize=20)
    axis.set_xlabel("Frozen rank-40 split", fontsize=16)
    axis.set_ylabel("Terminal molecules / trajectories", fontsize=16)
    axis.tick_params(axis="both", labelsize=13)
    axis.legend(title="Label", fontsize=13, title_fontsize=13)
    axis.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / "rank40_filtered_bc_label_counts.png", dpi=180)
    plt.savefig(output_dir / "rank40_filtered_bc_label_counts.svg")
    plt.close()


def main() -> None:
    """Rebuild a readable index and binary sidecar without changing sources."""
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    _configure_logging(output_dir)
    started = time.time()
    LOGGER.info(
        "start run_dir=%s binary_manifest=%s output=%s",
        args.run_dir,
        args.binary_manifest,
        output_dir,
    )
    frame = pd.read_csv(args.binary_manifest)
    required = {"trajectory_id", "binary_label", "split"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"binary manifest missing columns: {sorted(missing)}")
    if frame["trajectory_id"].duplicated().any():
        raise ValueError("binary manifest contains duplicate trajectory_id")
    labels = set(frame["binary_label"].astype(str))
    if labels != {"preferred", "non_preferred"}:
        raise ValueError(f"unexpected binary labels: {sorted(labels)}")

    episode_index = build_episode_index(
        args.run_dir,
        max_shards_per_rank=args.max_shards_per_rank,
        progress_every_shards=250,
        progress_callback=_progress,
    )
    index_path = output_dir / "episode_index.jsonl"
    write_episode_index(index_path, episode_index)

    known_ids = {row.sample_id for row in episode_index.rows}
    manifest_ids = set(frame["trajectory_id"].astype(str))
    unknown = sorted(manifest_ids - known_ids)
    if unknown:
        raise ValueError(f"rank40 IDs absent from episode index: {unknown[:5]}")
    preferred_ids = sorted(
        frame.loc[
            frame["binary_label"].eq("preferred"), "trajectory_id"
        ].astype(str)
    )
    preferred_set = set(preferred_ids)
    eligible_path = output_dir / "preferred_all_sample_ids.json"
    eligible_path.write_text(
        json.dumps(preferred_ids, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    split_manifest = build_split_manifest(
        episode_index,
        split_version="rank40_all_episodes_train_for_leaky_filtered_bc_v1",
        salt="rank40-all-preferred-filtered-bc-260822",
        fractions=(1.0, 0.0, 0.0),
    )
    split_path = output_dir / "all_train_split_manifest.jsonl"
    write_split_manifest(split_path, split_manifest)

    reward_rows = []
    for row in episode_index.rows:
        preferred = row.sample_id in preferred_set
        reward_rows.append(
            EpisodeRewardFacts(
                sample_id=row.sample_id,
                source_trajectory_format=TRAJECTORY_SHARD_FORMAT,
                source_checkpoint_sha256=row.source_checkpoint_sha256,
                source_shard_sha256=row.source_shard_sha256,
                reward_schema_version=REWARD_SIDECAR_SCHEMA_VERSION,
                reward_version=REWARD_VERSION,
                reconstruction_success=row.rdkit_valid,
                sanitize_success=row.rdkit_valid,
                rdkit_valid=row.rdkit_valid,
                pb_evaluated=False,
                pb_valid=False,
                docking_input_success=False,
                docking_success=False,
                actual_good=False,
                reward_total_train=1.0 if preferred else 0.0,
                reward_total_raw=1.0 if preferred else 0.0,
                failure_reason=(
                    "rank40_preferred"
                    if preferred
                    else f"rank40_nonpreferred_or_unlabeled:{row.failure_type}"
                ),
            )
        )
    sidecar_path = output_dir / "binary_reward_sidecar.csv"
    write_reward_sidecar(sidecar_path, reward_rows)
    _write_plot(frame, output_dir)

    raw_counts = pd.crosstab(frame["split"], frame["binary_label"]).to_dict()
    counts = {
        str(label): {str(split): int(count) for split, count in values.items()}
        for label, values in raw_counts.items()
    }
    summary = {
        "status": "complete",
        "purpose": "all-preferred trajectory filtered BC inputs",
        "data_leakage_allowed_by_user": True,
        "episode_index_rows": len(episode_index.rows),
        "rank40_labeled_rows": len(frame),
        "preferred_eligible_rows": len(preferred_ids),
        "non_preferred_rows": int(frame["binary_label"].eq("non_preferred").sum()),
        "binary_split_counts": counts,
        "all_training_split": True,
        "reward_version": REWARD_VERSION,
        "source_binary_manifest": str(args.binary_manifest.resolve()),
        "source_binary_manifest_sha256": _sha256(args.binary_manifest),
        "episode_index_sha256": episode_index.sha256,
        "outputs": {
            "episode_index": str(index_path),
            "episode_index_file_sha256": _sha256(index_path),
            "split_manifest": str(split_path),
            "split_manifest_sha256": _sha256(split_path),
            "eligible_ids": str(eligible_path),
            "eligible_ids_sha256": _sha256(eligible_path),
            "reward_sidecar": str(sidecar_path),
            "reward_sidecar_sha256": _sha256(sidecar_path),
        },
        "elapsed_seconds": time.time() - started,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    LOGGER.info(
        "complete episodes=%d preferred=%d elapsed=%.1fs",
        len(episode_index.rows),
        len(preferred_ids),
        summary["elapsed_seconds"],
    )


if __name__ == "__main__":
    main()
