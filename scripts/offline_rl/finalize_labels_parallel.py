#!/usr/bin/env python3
"""Finalize completed offline label shards with parallel validation.

This is an aggregation-only accelerator for the established offline-label
format.  It does not run reconstruction, docking, or PoseBusters.  Per-shard
status, input fingerprints, sidecar hashes, sample coverage, and provenance
are validated in parallel; the final sidecar and split manifest are written
in deterministic episode-index order by the parent process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from ddpo.offline_rl.index import build_split_manifest, load_episode_index, write_split_manifest
from ddpo.offline_rl.labeling import merge_reward_sidecar_shards
from ddpo.offline_rl.labeling_run import (
    LABEL_SHARD_STATUS_FORMAT,
    LABEL_RUN_SUMMARY_FORMAT,
    _atomic_write_json,
    _label_counts,
    _reward_strata,
    _write_summary_csv,
    _write_summary_plot,
    build_shard_label_tasks,
    load_labeling_run_config,
    sha256_file,
    _task_dir,
)
from ddpo.offline_rl.reward_sidecar import (
    EpisodeRewardFacts,
    RewardSidecar,
    load_reward_sidecar,
    write_reward_sidecar,
)


LOGGER = logging.getLogger("OfflineLabelingParallelFinalize")


def _sample_ids_sha256(sample_ids: list[str]) -> str:
    """Hash one shard's ordered sample IDs using the worker schema."""
    return hashlib.sha256(("\n".join(sample_ids) + "\n").encode("utf-8")).hexdigest()


def _validate_one_task(payload: dict[str, Any]) -> tuple[int, list[dict[str, Any]]]:
    """Validate and parse one shard in a worker process."""
    status_path = Path(payload["status_path"])
    sidecar_path = Path(payload["sidecar_path"])
    reconstruction_path = Path(payload["reconstruction_path"])
    molecules_path = Path(payload["molecules_path"])
    status = json.loads(status_path.read_text(encoding="utf-8"))
    expected = dict(payload["expected"])
    mismatches = [name for name, value in expected.items() if status.get(name) != value]
    if mismatches:
        raise ValueError(f"{status_path}: status input mismatch: {mismatches}")
    if status.get("status") != "completed":
        raise ValueError(f"{status_path}: status is not completed")
    if not reconstruction_path.is_file() or not molecules_path.is_file():
        raise FileNotFoundError(
            f"missing immutable input for rank={payload['rank']} shard={payload['shard_index']}"
        )
    if not sidecar_path.is_file():
        raise FileNotFoundError(f"missing completed sidecar: {sidecar_path}")
    if status.get("reconstruction_csv_sha256") != sha256_file(reconstruction_path):
        raise ValueError(f"{status_path}: reconstruction CSV SHA mismatch")
    if status.get("molecules_sdf_sha256") != sha256_file(molecules_path):
        raise ValueError(f"{status_path}: molecule SDF SHA mismatch")
    if status.get("sidecar_sha256") != sha256_file(sidecar_path):
        raise ValueError(f"{status_path}: sidecar SHA mismatch")
    sidecar = load_reward_sidecar(
        sidecar_path,
        expected_reward_version=str(expected["reward_version"]),
    )
    indexed = {
        row["sample_id"]: row
        for row in payload["rows"]
    }
    labels = {row.sample_id: row for row in sidecar.rows}
    if set(labels) != set(indexed):
        raise ValueError(
            f"{sidecar_path}: sample coverage mismatch; "
            f"expected={len(indexed)} observed={len(labels)}"
        )
    for sample_id, index_row in indexed.items():
        label = labels[sample_id]
        if (
            label.source_shard_sha256.lower() != str(index_row["source_shard_sha256"]).lower()
            or label.source_checkpoint_sha256.lower()
            != str(index_row["source_checkpoint_sha256"]).lower()
        ):
            raise ValueError(f"{sidecar_path}: provenance mismatch for {sample_id}")
    return int(payload["ordinal"]), [asdict(row) for row in sidecar.rows]


def _build_payloads(episode_index: Any, tasks: tuple[Any, ...], run_config: Any) -> list[dict[str, Any]]:
    """Build small serializable worker payloads without copying the full index."""
    backend_sha256: dict[str, str] = {}
    if run_config.posebusters is not None:
        backend_sha256["posebusters_receptor_pdb"] = sha256_file(
            run_config.posebusters.receptor_pdb
        )
    if run_config.docking is not None:
        backend_sha256["docking_receptor_pdbqt"] = sha256_file(
            run_config.docking.receptor_pdbqt
        )
        backend_sha256["unidock_binary"] = sha256_file(run_config.docking.unidock_bin)

    payloads: list[dict[str, Any]] = []
    for ordinal, task in enumerate(tasks):
        task_dir = _task_dir(run_config.output_dir, task)
        rows = [
            {
                "sample_id": row.sample_id,
                "source_shard_sha256": row.source_shard_sha256,
                "source_checkpoint_sha256": row.source_checkpoint_sha256,
            }
            for row in task.rows
        ]
        expected = {
            "format": LABEL_SHARD_STATUS_FORMAT,
            "config_sha256": run_config.config_sha256,
            "episode_index_sha256": episode_index.sha256,
            "source_run_id": episode_index.source_run_id,
            "rank": task.rank,
            "shard_index": task.shard_index,
            "sample_count": len(task.rows),
            "sample_ids_sha256": _sample_ids_sha256([row["sample_id"] for row in rows]),
            "source_shard_sha256": rows[0]["source_shard_sha256"].lower(),
            "source_checkpoint_sha256": rows[0]["source_checkpoint_sha256"].lower(),
            "backend_sha256": backend_sha256,
            "reward_version": run_config.policy.reward_version,
        }
        payloads.append(
            {
                "ordinal": ordinal,
                "rank": task.rank,
                "shard_index": task.shard_index,
                "rows": rows,
                "expected": expected,
                "status_path": str(task_dir / "status.json"),
                "sidecar_path": str(task_dir / "reward_sidecar.csv"),
                "reconstruction_path": str(task.reconstruction_csv),
                "molecules_path": str(task.molecules_sdf),
            }
        )
    return payloads


def _write_progress(output_dir: Path, *, completed: int, total: int, workers: int) -> None:
    """Persist a durable progress record for external monitoring."""
    _atomic_write_json(
        output_dir / "parallel_finalize_status.json",
        {
            "status": "running" if completed < total else "validated",
            "completed_shards": completed,
            "total_shards": total,
            "workers": workers,
        },
    )


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    """Run parallel validation and deterministic final aggregation."""
    run_config = load_labeling_run_config(
        args.config,
        data_root=args.data_root,
        worker_id=0,
        gpu_id_override=0,
    )
    run_config = replace(run_config, output_dir=args.output_dir.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_snapshot = args.output_dir / "config" / "resolved_config.yaml"
    config_snapshot.parent.mkdir(parents=True, exist_ok=True)
    if not config_snapshot.exists():
        shutil.copyfile(args.config, config_snapshot)

    episode_index = load_episode_index(run_config.episode_index_path)
    tasks = build_shard_label_tasks(
        episode_index,
        source_run_dir=run_config.source_run_dir,
    )
    payloads = _build_payloads(episode_index, tasks, run_config)
    total = len(payloads)
    rows_by_ordinal: dict[int, list[dict[str, Any]]] = {}
    _write_progress(args.output_dir, completed=0, total=total, workers=args.workers)
    LOGGER.info("parallel validation start shards=%d workers=%d", total, args.workers)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_validate_one_task, payload) for payload in payloads]
        for completed, future in enumerate(as_completed(futures), start=1):
            ordinal, rows = future.result()
            rows_by_ordinal[ordinal] = rows
            if completed == 1 or completed % 250 == 0 or completed == total:
                _write_progress(
                    args.output_dir,
                    completed=completed,
                    total=total,
                    workers=args.workers,
                )
                LOGGER.info("validated shards=%d/%d", completed, total)

    ordered_rows = [
        EpisodeRewardFacts(**row)
        for ordinal in range(total)
        for row in rows_by_ordinal[ordinal]
    ]
    sidecar = RewardSidecar(ordered_rows)
    episode_rows_by_id = {row.sample_id: row for row in episode_index.rows}
    expected_ids = set(episode_rows_by_id)
    observed_ids = {row.sample_id for row in sidecar.rows}
    if expected_ids != observed_ids:
        raise ValueError(
            f"final sidecar coverage mismatch: missing={len(expected_ids - observed_ids)} "
            f"extra={len(observed_ids - expected_ids)}"
        )
    for row in sidecar.rows:
        index_row = episode_rows_by_id[row.sample_id]
        if (
            row.source_shard_sha256.lower() != index_row.source_shard_sha256.lower()
            or row.source_checkpoint_sha256.lower() != index_row.source_checkpoint_sha256.lower()
        ):
            raise ValueError(f"final sidecar provenance mismatch for {row.sample_id}")

    merged_path = args.output_dir / "reward_sidecar.csv"
    write_reward_sidecar(merged_path, sidecar.rows)
    split_counts: Counter[str] = Counter()
    split_path: Path | None = None
    if run_config.split_enabled:
        split = build_split_manifest(
            episode_index,
            split_version=run_config.split_version,
            salt=run_config.split_salt,
            fractions=run_config.split_fractions,
            reward_strata=_reward_strata(sidecar),
        )
        split_path = write_split_manifest(args.output_dir / "split_manifest.jsonl", split)
        split_counts.update(row.split for row in split.rows)

    counts = _label_counts(sidecar)
    attempts = counts["attempts"]
    scores = [
        float(row.docking_raw)
        for row in sidecar.rows
        if row.docking_success and row.docking_raw is not None
    ]
    rewards = [float(row.reward_total_train) for row in sidecar.rows]
    summary: dict[str, Any] = {
        "format": LABEL_RUN_SUMMARY_FORMAT,
        "status": "completed",
        "scope": "full_episode_index",
        "formal_labels": run_config.policy.formal_labels,
        "formal_dataset_complete": bool(run_config.policy.formal_labels),
        "config_path": str(run_config.config_path),
        "config_sha256": run_config.config_sha256,
        "episode_index_path": str(run_config.episode_index_path),
        "episode_index_sha256": episode_index.sha256,
        "source_episode_index_sha256": episode_index.sha256,
        "source_run_id": episode_index.source_run_id,
        "shards_completed": total,
        "shards_expected": total,
        "reward_sidecar": str(merged_path),
        "reward_sidecar_sha256": sha256_file(merged_path),
        "split_manifest": str(split_path) if split_path is not None else None,
        "counts": counts,
        "rates": {
            "rdkit_valid_rate_attempts": counts["rdkit_valid"] / attempts,
            "pb_valid_rate_attempts": counts["pb_valid"] / attempts,
            "pb_valid_rate_rdkit_valid": counts["pb_valid"] / counts["rdkit_valid"] if counts["rdkit_valid"] else 0.0,
            "docking_success_rate_attempts": counts["docking_success"] / attempts,
            "docking_success_rate_inputs": counts["docking_success"] / counts["docking_input_success"] if counts["docking_input_success"] else 0.0,
            "actual_good_rate_attempts": counts["actual_good"] / attempts,
        },
        "reward": {
            "mean": sum(rewards) / len(rewards),
            "median": sorted(rewards)[len(rewards) // 2],
            "min": min(rewards),
            "max": max(rewards),
        },
        "docking": {
            "score_count": len(scores),
            "score_mean": sum(scores) / len(scores) if scores else None,
            "score_median": sorted(scores)[len(scores) // 2] if scores else None,
            "score_best_min": min(scores) if scores else None,
            "score_worst_max": max(scores) if scores else None,
        },
        "split_counts": dict(sorted(split_counts.items())),
        "failure_counts": dict(sorted(Counter(row.failure_reason for row in sidecar.rows).items())),
        "parallel_workers": args.workers,
    }
    _atomic_write_json(args.output_dir / "summary.json", summary)
    _write_summary_csv(args.output_dir / "summary.csv", summary)
    _write_summary_plot(args.output_dir / "label_summary.png", sidecar)
    _write_progress(args.output_dir, completed=total, total=total, workers=args.workers)
    return summary


def main() -> None:
    """Parse arguments, configure logging, and finalize labels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=28)
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.output_dir / "logs" / "parallel_finalize_28.log", encoding="utf-8"),
        ],
        force=True,
    )
    summary = finalize(args)
    LOGGER.info(
        "completed shards=%d attempts=%d sidecar=%s split=%s",
        summary["shards_completed"],
        summary["counts"]["attempts"],
        summary["reward_sidecar"],
        summary["split_manifest"],
    )


if __name__ == "__main__":
    main()
