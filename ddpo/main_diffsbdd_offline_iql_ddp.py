"""Train DiffSBDD offline IQL from immutable denoising trajectories with DDP."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ddpo.offline_rl.build import (  # noqa: E402
    build_offline_iql_runtime,
    load_offline_iql_config,
    module_sha256,
    resolve_project_path,
)
from ddpo.offline_rl.ddp import OfflineIQLDDPCoordinator  # noqa: E402
from ddpo.runtime import (  # noqa: E402
    all_gather_objects,
    barrier_if_needed,
    destroy_distributed,
    init_distributed_from_env,
    is_main_process,
)


LOGGER = logging.getLogger("DiffSBDDOfflineIQL")


def _parse_args() -> argparse.Namespace:
    """Parse configuration, output, resume, and bounded validation controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resume-manifest", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser.parse_args()


def _utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically persist one small JSON evidence object."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _setup_logging(output_dir: Path, *, rank: int, level: str) -> None:
    """Create rank-local durable logging and rank-zero terminal progress."""
    LOGGER.handlers.clear()
    LOGGER.setLevel(getattr(logging, level))
    LOGGER.propagate = False
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | rank=%(rank)s | %(name)s | %(message)s"
    )
    log_path = output_dir / "logs" / f"entry_rank_{rank:04d}.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.addFilter(_RankFilter(rank))
    LOGGER.addHandler(file_handler)
    if rank == 0:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.addFilter(_RankFilter(rank))
        LOGGER.addHandler(stream_handler)


class _RankFilter(logging.Filter):
    """Inject a fixed distributed rank into entrypoint log records."""

    def __init__(self, rank: int) -> None:
        """Store the rank value added to every record."""
        super().__init__()
        self.rank = rank

    def filter(self, record: logging.LogRecord) -> bool:
        """Attach rank and retain the record."""
        record.rank = self.rank
        return True


def _prepare_output(
    output_dir: Path,
    *,
    resume_manifest: Path | None,
    rank: int,
) -> None:
    """Create bootstrap directories without overwriting training evidence."""
    if rank != 0:
        return
    if resume_manifest is None and output_dir.exists():
        allowed = {"config", "logs", "launch.log", "run_manifest.json"}
        unexpected = sorted(
            path.name for path in output_dir.iterdir() if path.name not in allowed
        )
        if unexpected:
            raise FileExistsError(
                f"fresh output contains prior evidence: {unexpected}"
            )
    if resume_manifest is not None and not output_dir.exists():
        raise FileNotFoundError(f"resume output does not exist: {output_dir}")
    (output_dir / "config").mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)


def _git_facts() -> dict[str, Any]:
    """Capture the current commit and concise worktree status."""
    def run(*args: str) -> str:
        return subprocess.run(
            args,
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    try:
        return {
            "commit": run("git", "rev-parse", "HEAD"),
            "branch": run("git", "branch", "--show-current"),
            "status_short": run("git", "status", "--short").splitlines(),
        }
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _write_config_snapshot(
    output_dir: Path,
    config: dict[str, Any],
    *,
    epochs: int,
    resume: bool,
) -> Path:
    """Persist the exact resolved invocation configuration."""
    name = (
        f"resolved_config_resume_to_epoch_{epochs - 1:04d}.yaml"
        if resume
        else "resolved_config.yaml"
    )
    path = output_dir / "config" / name
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def _update_run_manifest(
    output_dir: Path,
    *,
    config_path: Path,
    snapshot_path: Path,
    epochs: int,
    resume_manifest: Path | None,
    runtime_identity: dict[str, Any],
    formal_training: bool,
) -> None:
    """Append one launch invocation to durable local provenance."""
    path = output_dir / "run_manifest.json"
    existing: dict[str, Any] = {}
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            existing = payload
    invocations = list(existing.get("invocations", []))
    invocations.append(
        {
            "started_at": _utc_now(),
            "config_path": str(config_path),
            "config_snapshot_path": str(snapshot_path),
            "target_epochs": epochs,
            "resume_manifest": (
                None if resume_manifest is None else str(resume_manifest)
            ),
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_cuda_version": torch.version.cuda,
            "gpu_count_visible": torch.cuda.device_count(),
        }
    )
    _atomic_write_json(
        path,
        {
            "format": "diffsbdd_offline_iql_run_manifest_v1",
            "formal_training": formal_training,
            "git": _git_facts(),
            "run_identity": runtime_identity,
            "invocations": invocations,
        },
    )


def _parameter_sync_facts(bundle: Any, rank: int) -> dict[str, Any]:
    """Return hashes for the three synchronized trainable modules."""
    return {
        "rank": rank,
        "global_step": bundle.trainer.global_step,
        "policy_sha256": module_sha256(bundle.trainer.actor_updater.policy),
        "online_q_sha256": module_sha256(bundle.trainer.online_q),
        "value_sha256": module_sha256(bundle.trainer.value),
    }


def main() -> None:
    """Build one torchrun runtime, train or resume, and verify rank agreement."""
    args = _parse_args()
    config_path = args.config.resolve()
    config = load_offline_iql_config(config_path)
    dist_ctx = init_distributed_from_env(config)
    try:
        offline_config = config.get("offline_iql", {})
        run_config = offline_config.get("run", {})
        configured_output = offline_config.get("output_dir")
        if args.output_dir is None and not configured_output:
            raise ValueError("output directory is required by CLI or YAML")
        output_dir = resolve_project_path(
            args.output_dir if args.output_dir is not None else configured_output,
            PROJECT_ROOT,
        )
        data_root = (
            PROJECT_ROOT
            if args.data_root is None
            else args.data_root.resolve()
        )
        epochs = int(
            run_config.get("epochs") if args.epochs is None else args.epochs
        )
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        resume_manifest = (
            None
            if args.resume_manifest is None
            else args.resume_manifest.resolve()
        )
        _prepare_output(
            output_dir,
            resume_manifest=resume_manifest,
            rank=dist_ctx.rank,
        )
        barrier_if_needed(dist_ctx)
        _setup_logging(output_dir, rank=dist_ctx.rank, level=args.log_level)
        LOGGER.info(
            "preflight config=%s output=%s env=%s world_size=%d device=%s resume=%s",
            config_path,
            output_dir,
            os.environ.get("CONDA_DEFAULT_ENV"),
            dist_ctx.world_size,
            dist_ctx.device,
            resume_manifest,
        )
        bundle = build_offline_iql_runtime(
            config,
            project_root=data_root,
            dist_ctx=dist_ctx,
            epochs_override=epochs,
        )
        if is_main_process(dist_ctx):
            resolved_config = json.loads(json.dumps(config))
            resolved_config["offline_iql"]["output_dir"] = str(output_dir)
            resolved_config["offline_iql"]["resolved_data_root"] = str(data_root)
            resolved_config["offline_iql"]["run"]["epochs"] = epochs
            snapshot = _write_config_snapshot(
                output_dir,
                resolved_config,
                epochs=epochs,
                resume=resume_manifest is not None,
            )
            _update_run_manifest(
                output_dir,
                config_path=config_path,
                snapshot_path=snapshot,
                epochs=epochs,
                resume_manifest=resume_manifest,
                runtime_identity=bundle.run_identity,
                formal_training=bundle.formal_training,
            )
        barrier_if_needed(dist_ctx)
        LOGGER.info(
            "runtime ready formal_training=%s reward_version=%s "
            "eligible_episodes=%d eligible_shards=%d rank_seed=%d",
            bundle.formal_training,
            bundle.reward_version,
            bundle.sampler.num_eligible_episodes,
            bundle.sampler.num_eligible_shards,
            dist_ctx.rank + 1,
        )
        if args.preflight_only:
            LOGGER.info("preflight-only completed without optimizer steps")
            return

        coordinator = OfflineIQLDDPCoordinator(
            trainer=bundle.trainer,
            sampler=bundle.sampler,
            runner_config=bundle.runner_config,
            dist_ctx=dist_ctx,
            output_dir=output_dir,
            run_identity=bundle.run_identity,
            shared_decision_scheduler=bundle.scheduler,
        )
        coordinator.run(resume_manifest=resume_manifest)
        sync_facts = all_gather_objects(
            _parameter_sync_facts(bundle, dist_ctx.rank),
            dist_ctx,
        )
        reference = {
            key: sync_facts[0][key]
            for key in ("global_step", "policy_sha256", "online_q_sha256", "value_sha256")
        }
        synchronized = all(
            all(item[key] == expected for key, expected in reference.items())
            for item in sync_facts
        )
        if is_main_process(dist_ctx):
            _atomic_write_json(
                output_dir / "parameter_sync.json",
                {
                    "format": "diffsbdd_offline_iql_parameter_sync_v1",
                    "synchronized": synchronized,
                    "ranks": sync_facts,
                    "checked_at": _utc_now(),
                },
            )
        if not synchronized:
            raise RuntimeError(f"DDP trainable module hashes diverged: {sync_facts}")
        LOGGER.info(
            "training completed global_step=%d parameter_sync=%s",
            bundle.trainer.global_step,
            synchronized,
        )
    except Exception:
        LOGGER.exception("offline IQL entrypoint failed")
        raise
    finally:
        destroy_distributed()


if __name__ == "__main__":
    main()
