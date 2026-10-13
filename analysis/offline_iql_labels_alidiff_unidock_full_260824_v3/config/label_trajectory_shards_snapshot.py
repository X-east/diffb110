#!/usr/bin/env python3
"""Label DiffSBDD trajectory shards with resumable PB and UniDock facts."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

from ddpo.offline_rl.index import load_episode_index
from ddpo.offline_rl.labeling_run import (
    build_shard_label_tasks,
    finalize_labeling_run,
    label_one_task,
    load_labeling_run_config,
    select_worker_tasks,
)
from ddpo.offline_rl.labeling_backends import PoseBustersLabeler, UniDockLabeler


LOGGER = logging.getLogger("OfflineLabeling")


def _parse_args() -> argparse.Namespace:
    """Parse shard-worker and finalization controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def _configure_logging(output_dir: Path, worker_id: int) -> None:
    """Write timestamped progress to terminal and a durable worker log."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    chemistry_log_dir = log_dir / f"worker_{worker_id:02d}_chemistry"
    chemistry_log_dir.mkdir(parents=True, exist_ok=True)
    os.environ["DIFFSBDD_LOG_DIR"] = str(chemistry_log_dir)
    matplotlib_config_dir = output_dir / "runtime_cache" / "matplotlib"
    matplotlib_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config_dir))
    formatter = logging.Formatter(
        f"%(asctime)s %(levelname)s worker={worker_id} %(name)s: %(message)s"
    )
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    file_handler = logging.FileHandler(
        log_dir / f"worker_{worker_id:02d}.log",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(stream_handler)


def _git_value(project_root: Path, *args: str) -> str:
    """Return one non-mutating git value for run provenance."""
    result = subprocess.run(
        ("git", "-C", str(project_root), *args),
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    return (result.stdout or result.stderr).strip()


def _snapshot_config(config_path: Path, output_dir: Path) -> None:
    """Create or validate the immutable run-local YAML snapshot."""
    snapshot_path = output_dir / "config" / "resolved_config.yaml"
    source_bytes = config_path.read_bytes()
    if snapshot_path.exists():
        if snapshot_path.read_bytes() != source_bytes:
            raise ValueError(
                f"run config snapshot differs from requested config: {snapshot_path}"
            )
        return
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=snapshot_path.parent,
            prefix=f".{snapshot_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(source_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        if snapshot_path.exists() and snapshot_path.read_bytes() != source_bytes:
            raise ValueError(
                f"run config snapshot differs from requested config: {snapshot_path}"
            )
        os.replace(temporary_path, snapshot_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> None:
    """Run one stable shard assignment and optionally finalize its sidecar."""
    args = _parse_args()
    if args.finalize_only and args.finalize:
        raise ValueError("--finalize and --finalize-only are mutually exclusive")
    if args.preflight_only and (args.finalize or args.finalize_only):
        raise ValueError("--preflight-only cannot be combined with finalization")
    run_config = load_labeling_run_config(
        args.config,
        data_root=args.data_root,
        worker_id=args.worker_id,
        gpu_id_override=args.gpu_id,
    )
    if args.output_dir is not None:
        output_dir = (
            args.output_dir.resolve()
            if args.output_dir.is_absolute()
            else (args.data_root / args.output_dir).resolve()
        )
        run_config = replace(run_config, output_dir=output_dir)
    _configure_logging(run_config.output_dir, args.worker_id)
    _snapshot_config(run_config.config_path, run_config.output_dir)

    project_root = Path(__file__).resolve().parents[2]
    LOGGER.info(
        "start config=%s config_sha256=%s output=%s conda_env=%s "
        "pythonpath=%s cuda_visible_devices=%s git_commit=%s git_status=%s",
        run_config.config_path,
        run_config.config_sha256,
        run_config.output_dir,
        os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
        os.environ.get("PYTHONPATH", ""),
        os.environ.get("CUDA_VISIBLE_DEVICES", "all-visible"),
        _git_value(project_root, "rev-parse", "HEAD"),
        _git_value(project_root, "status", "--short").replace("\n", "; "),
    )
    episode_index = load_episode_index(run_config.episode_index_path)
    all_tasks = build_shard_label_tasks(
        episode_index,
        source_run_dir=run_config.source_run_dir,
    )
    tasks = select_worker_tasks(
        all_tasks,
        worker_id=args.worker_id,
        num_workers=args.num_workers,
        max_shards=args.max_shards,
    )
    LOGGER.info(
        "assignment worker=%d/%d gpu_id=%s selected_shards=%d "
        "all_shards=%d max_shards=%s",
        args.worker_id,
        args.num_workers,
        run_config.docking.gpu_id if run_config.docking is not None else "cpu",
        len(tasks),
        len(all_tasks),
        args.max_shards,
    )
    if args.preflight_only:
        LOGGER.info("preflight completed successfully")
        return
    if not args.finalize_only:
        pb_labeler = (
            PoseBustersLabeler(run_config.posebusters)
            if run_config.posebusters is not None
            else None
        )
        docking_labeler = (
            UniDockLabeler(run_config.docking)
            if run_config.docking is not None
            else None
        )
        try:
            for task_number, task in enumerate(tasks, start=1):
                state, sidecar_path, counts = label_one_task(
                    task,
                    run_config=run_config,
                    episode_index=episode_index,
                    resume=args.resume,
                    posebusters_labeler=pb_labeler,
                    docking_labeler=docking_labeler,
                )
                LOGGER.info(
                    "shard %d/%d rank=%d shard_index=%d state=%s "
                    "counts=%s sidecar=%s",
                    task_number,
                    len(tasks),
                    task.rank,
                    task.shard_index,
                    state,
                    counts,
                    sidecar_path,
                )
        finally:
            if pb_labeler is not None:
                pb_labeler.close()

    if args.finalize or args.finalize_only:
        if args.num_workers != 1 and args.max_shards is not None:
            raise ValueError(
                "partial finalization requires num_workers=1 for unambiguous coverage"
            )
        final_tasks = tasks if args.max_shards is not None else all_tasks
        summary = finalize_labeling_run(
            final_tasks,
            all_task_count=len(all_tasks),
            episode_index=episode_index,
            run_config=run_config,
        )
        LOGGER.info(
            "finalize completed scope=%s shards=%d/%d attempts=%d actual_good=%d",
            summary["scope"],
            summary["shards_completed"],
            summary["shards_expected"],
            summary["counts"]["attempts"],
            summary["counts"]["actual_good"],
        )
    LOGGER.info("completed successfully")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        LOGGER.exception("labeling failed")
        sys.exit(1)
