#!/usr/bin/env python3
"""Generate 1,000 fixed-attempt molecules from every tenth Rank40 BC checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger("rank40_bc_checkpoint_generation")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONDA = Path("/home/md666/miniforge3/bin/conda")
CHECKPOINT_NUMBERS = tuple(range(10, 101, 10))
GPU_IDS = tuple(range(8))


def _parse_args() -> argparse.Namespace:
    """Parse training and generation output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    return parser.parse_args()


def _utc_now() -> str:
    """Return the current UTC time in ISO-8601 form."""
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one JSON status file atomically."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    """Return the SHA256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sdf_records(path: Path) -> int:
    """Count explicit SDF record delimiters without RDKit filtering."""
    count = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip() == "$$$$":
                count += 1
    return count


def _wait_for_training(training_dir: Path, poll_seconds: int) -> None:
    """Wait until epoch 99 and its complete global manifest are durable."""
    status_path = training_dir / "status.json"
    manifest_path = training_dir / "checkpoints" / "epoch_0099" / "manifest.json"
    last_epoch: int | None = None
    while True:
        status: dict[str, Any] = {}
        if status_path.is_file():
            status = json.loads(status_path.read_text(encoding="utf-8"))
        epoch = status.get("epoch")
        if epoch != last_epoch:
            LOGGER.info(
                "training wait status=%s epoch=%s global_step=%s",
                status.get("status"),
                epoch,
                status.get("global_step"),
            )
            last_epoch = epoch
        if status.get("status") == "completed" and int(epoch or -1) >= 99 and manifest_path.is_file():
            LOGGER.info("training complete manifest=%s", manifest_path)
            return
        if status.get("status") == "failed":
            raise RuntimeError(f"training failed before epoch 99: {status}")
        time.sleep(max(5, poll_seconds))


def _run_export(offline_checkpoint: Path, run_dir: Path) -> Path:
    """Export one rank-zero actor to a native DiffSBDD checkpoint."""
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    exported = checkpoint_dir / "diffsbdd_actor.ckpt"
    metadata = checkpoint_dir / "export_metadata.json"
    if exported.is_file() and metadata.is_file():
        LOGGER.info("reuse exported checkpoint=%s", exported)
        return exported
    if exported.exists() or metadata.exists():
        raise FileExistsError(f"partial export exists in {checkpoint_dir}")
    command = [
        str(CONDA), "run", "-n", "dif", "python",
        str(PROJECT_ROOT / "scripts" / "export_offline_actor_to_diffsbdd_checkpoint.py"),
        "--base-checkpoint", str(PROJECT_ROOT / "DiffSBDD" / "checkpoints" / "crossdocked_fullatom_cond.ckpt"),
        "--offline-checkpoint", str(offline_checkpoint),
        "--output-checkpoint", str(exported),
        "--metadata-json", str(metadata),
    ]
    with (run_dir / "logs" / "export.log").open("a", encoding="utf-8") as log_handle:
        subprocess.run(command, cwd=PROJECT_ROOT, stdout=log_handle, stderr=subprocess.STDOUT, check=True)
    return exported


def _generate_shards(exported: Path, run_dir: Path) -> list[Path]:
    """Generate eight fixed-seed shards concurrently on eight GPUs."""
    shard_dir = run_dir / "shards"
    log_dir = run_dir / "logs"
    shard_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    processes: list[tuple[int, subprocess.Popen[bytes], Any]] = []
    shard_paths: list[Path] = []
    for gpu_id in GPU_IDS:
        shard_path = shard_dir / f"gpu_{gpu_id:02d}.sdf"
        shard_paths.append(shard_path)
        if shard_path.is_file() and _sdf_records(shard_path) == 125:
            LOGGER.info("reuse complete shard gpu=%d path=%s", gpu_id, shard_path)
            continue
        if shard_path.exists():
            shard_path.unlink()
        command = [
            str(CONDA), "run", "-n", "dif", "python",
            str(PROJECT_ROOT / "DiffSBDD" / "generate_ligands.py"),
            str(exported),
            "--pdbfile", str(PROJECT_ROOT / "data" / "8GJL.pdb"),
            "--ref_ligand", "C:1101",
            "--outfile", str(shard_path),
            "--n_samples", "125",
            "--batch_size", "25",
            "--all_frags",
            "--timesteps", "1000",
            "--resamplings", "10",
            "--jump_length", "1",
            "--reconstruction_method", "openbabel",
            "--seed", str(gpu_id + 1),
        ]
        environment = os.environ.copy()
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": str(gpu_id),
                "BABEL_LIBDIR": "/home/md666/miniforge3/envs/dif/lib/openbabel/3.1.0",
                "PYTHONUNBUFFERED": "1",
            }
        )
        log_handle = (log_dir / f"gpu_{gpu_id:02d}.log").open("ab")
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((gpu_id, process, log_handle))
    failures: list[tuple[int, int]] = []
    for gpu_id, process, log_handle in processes:
        return_code = process.wait()
        log_handle.close()
        if return_code != 0:
            failures.append((gpu_id, return_code))
    if failures:
        raise RuntimeError(f"generation shard failures: {failures}")
    observed = {path.name: _sdf_records(path) for path in shard_paths}
    if any(count != 125 for count in observed.values()):
        raise RuntimeError(f"unexpected shard SDF counts: {observed}")
    return shard_paths


def _combine_shards(shard_paths: list[Path], output_path: Path) -> None:
    """Concatenate validated shards into one 1,000-attempt SDF."""
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("wb") as output_handle:
        for path in shard_paths:
            with path.open("rb") as input_handle:
                shutil.copyfileobj(input_handle, output_handle)
    if _sdf_records(temporary) != 1000:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("combined SDF does not contain exactly 1,000 records")
    temporary.replace(output_path)


def _run_checkpoint(training_dir: Path, output_dir: Path, completed_epoch: int) -> dict[str, Any]:
    """Export and generate one completed-epoch checkpoint panel."""
    epoch_index = completed_epoch - 1
    label = f"epoch_{completed_epoch:03d}"
    run_dir = output_dir / label
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    completion_path = run_dir / "completion.json"
    combined = run_dir / "generated_1000_attempts.sdf"
    if completion_path.is_file() and combined.is_file():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("status") == "completed" and _sdf_records(combined) == 1000:
            LOGGER.info("skip completed checkpoint=%d", completed_epoch)
            return completion
    offline_checkpoint = (
        training_dir / "ranks" / "rank_0000" / "checkpoints"
        / f"epoch_{epoch_index:04d}_rank_0000.pt"
    )
    global_manifest = training_dir / "checkpoints" / f"epoch_{epoch_index:04d}" / "manifest.json"
    if not offline_checkpoint.is_file() or not global_manifest.is_file():
        raise FileNotFoundError(f"missing epoch {epoch_index} checkpoint evidence")
    started = time.time()
    LOGGER.info("checkpoint generation start completed_epoch=%d", completed_epoch)
    exported = _run_export(offline_checkpoint, run_dir)
    shards = _generate_shards(exported, run_dir)
    _combine_shards(shards, combined)
    completion = {
        "status": "completed",
        "completed_epoch": completed_epoch,
        "checkpoint_epoch_index": epoch_index,
        "fixed_attempts": 1000,
        "gpus": 8,
        "attempts_per_gpu": 125,
        "timesteps": 1000,
        "resamplings": 10,
        "seeds": [gpu_id + 1 for gpu_id in GPU_IDS],
        "written_sdf_records": _sdf_records(combined),
        "combined_sdf": str(combined.resolve()),
        "combined_sdf_sha256": _sha256(combined),
        "global_manifest": str(global_manifest.resolve()),
        "offline_checkpoint": str(offline_checkpoint.resolve()),
        "elapsed_seconds": time.time() - started,
        "completed_at": _utc_now(),
    }
    _atomic_json(completion_path, completion)
    LOGGER.info(
        "checkpoint generation completed completed_epoch=%d elapsed=%.1fs",
        completed_epoch,
        completion["elapsed_seconds"],
    )
    return completion


def main() -> int:
    """Wait for training, then run the resumable ten-checkpoint generation panel."""
    args = _parse_args()
    training_dir = args.training_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "generation_monitor.log", mode="a", encoding="utf-8"),
        ],
    )
    status_path = output_dir / "pipeline_status.json"
    _atomic_json(
        status_path,
        {
            "status": "waiting_for_training",
            "training_dir": str(training_dir),
            "output_dir": str(output_dir),
            "checkpoint_completed_epochs": list(CHECKPOINT_NUMBERS),
            "started_at": _utc_now(),
        },
    )
    try:
        _wait_for_training(training_dir, args.poll_seconds)
        results: list[dict[str, Any]] = []
        for completed_epoch in CHECKPOINT_NUMBERS:
            _atomic_json(
                status_path,
                {
                    "status": "generating",
                    "current_completed_epoch": completed_epoch,
                    "completed_checkpoints": [item["completed_epoch"] for item in results],
                    "updated_at": _utc_now(),
                },
            )
            results.append(_run_checkpoint(training_dir, output_dir, completed_epoch))
        summary = {
            "status": "completed",
            "training_dir": str(training_dir),
            "output_dir": str(output_dir),
            "fixed_attempts_per_checkpoint": 1000,
            "checkpoint_count": len(results),
            "total_fixed_attempts": 1000 * len(results),
            "checkpoints": results,
            "completed_at": _utc_now(),
        }
        _atomic_json(output_dir / "generation_summary.json", summary)
        _atomic_json(status_path, summary)
        return 0
    except Exception as exc:
        LOGGER.exception("generation monitor failed")
        _atomic_json(
            status_path,
            {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "updated_at": _utc_now()},
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
