"""Run the configuration-driven DiffSBDD T=1000 Gaussian-DPO trainer."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

# Make the repository root explicit when this file is called by absolute path.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import yaml

from ddpo.diffusion_dpo.pairs import load_pair_manifest, validate_pair_manifests
from ddpo.diffusion_dpo.scorer import DiffSBDDGaussianTransitionScorer
from ddpo.diffusion_dpo.trainer import GaussianDPOTrainer, TrainerConfig, sha256_file


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse the YAML path and explicit smoke/resume overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pair-count", type=int, default=None, help="Use a small prefix for a controlled smoke run.")
    parser.add_argument("--allow-small-pairs", action="store_true", help="Allow pair-count below the formal validation minimum.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-pairs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--mode", choices=["single_transition", "sampled_multistep", "full_valid_trajectory"], default=None)
    parser.add_argument("--timesteps-per-batch", type=int, default=None)
    parser.add_argument("--evaluation-mode", choices=["single_transition", "sampled_multistep", "full_valid_trajectory"], default=None)
    parser.add_argument("--evaluation-timesteps-per-batch", type=int, default=None)
    parser.add_argument("--aggregation", choices=["mean", "sum"], default=None)
    parser.add_argument("--resume", action="store_true", help="Resume from output-dir/checkpoint_latest.json.")
    return parser.parse_args()


def resolve(root: Path, value: str | Path) -> Path:
    """Resolve a project-relative path from YAML or a CLI argument."""
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def _git_facts(root: Path) -> dict[str, Any]:
    """Collect compact git provenance without dumping the full worktree."""
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=False, capture_output=True, text=True)
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=root, check=False, capture_output=True, text=True)
    status = subprocess.run(["git", "status", "--short", "--untracked-files=no"], cwd=root, check=False, capture_output=True, text=True)
    return {
        "commit": commit.stdout.strip(),
        "branch": branch.stdout.strip(),
        "dirty": bool(status.stdout.strip()),
        "tracked_change_count": len(status.stdout.splitlines()),
    }


def _cuda_facts(device: str) -> dict[str, Any]:
    """Record the PyTorch CUDA facts used by this process."""
    facts: dict[str, Any] = {
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "requested_device": device,
    }
    if device.startswith("cuda") and torch.cuda.is_available():
        index = torch.device(device).index
        if index is None:
            index = torch.cuda.current_device()
        if index < torch.cuda.device_count():
            facts["device_name"] = torch.cuda.get_device_name(index)
    return facts


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a stable indented JSON artifact."""
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def main() -> None:
    """Build current/reference models, preflight the run, and train."""
    args = parse_args()
    config_path = resolve(ROOT, args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_cfg = config["run"]
    model_cfg = config["model"]
    data_cfg = config["data"]
    dpo_cfg = config["dpo"]
    training_cfg = config["training"]
    if int(model_cfg["num_timesteps"]) != 1000:
        raise ValueError("this entrypoint requires model.num_timesteps=1000")
    device = str(args.device or model_cfg["device"])
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the formal T=1000 run")
    output_dir = resolve(ROOT, args.output_dir or run_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    completion = output_dir / "completion.json"
    if completion.exists() and not args.resume:
        raise FileExistsError(f"run is already completed; use --resume only for an incomplete run: {output_dir}")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(output_dir / "train.log", encoding="utf-8")],
        force=True,
    )

    pair_dir = resolve(ROOT, data_cfg["pair_dir"])
    train_manifest = pair_dir / "train_pairs.csv"
    validation_manifest = pair_dir / "validation_pairs.csv"
    train_pairs = load_pair_manifest(train_manifest)
    validation_pairs = load_pair_manifest(validation_manifest)
    if args.pair_count is not None:
        if args.pair_count <= 0:
            raise ValueError("--pair-count must be positive")
        train_pairs = train_pairs[: args.pair_count]
        validation_pairs = validation_pairs[: args.pair_count]
    minimum_validation_pairs = int(data_cfg.get("minimum_validation_pairs", 2000))
    if len(validation_pairs) < minimum_validation_pairs and not args.allow_small_pairs:
        raise ValueError(
            f"validation pairs={len(validation_pairs)} below formal minimum={minimum_validation_pairs}; "
            "use --allow-small-pairs only for a smoke run"
        )
    validate_pair_manifests(train_pairs, validation_pairs, source=pair_dir)

    checkpoint_path = resolve(ROOT, model_cfg["checkpoint"])
    pocket_path = resolve(ROOT, model_cfg["pocket_pdb"])
    if not checkpoint_path.is_file() or not pocket_path.is_file():
        raise FileNotFoundError(f"model inputs missing: checkpoint={checkpoint_path} pocket={pocket_path}")
    resume_path: Path | None = None
    if args.resume:
        latest_path = output_dir / "checkpoint_latest.json"
        if not latest_path.is_file():
            raise FileNotFoundError(f"--resume requested but missing {latest_path}")
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
        resume_path = resolve(ROOT, latest["path"])

    resolved_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    resolved_config["run"]["output_dir"] = str(output_dir)
    resolved_config["data"]["train_pair_manifest"] = str(train_manifest)
    resolved_config["data"]["validation_pair_manifest"] = str(validation_manifest)
    resolved_config["training"]["mode"] = args.mode or training_cfg.get("mode", dpo_cfg.get("mode", "sampled_multistep"))
    resolved_config["training"]["batch_pairs"] = args.batch_pairs or training_cfg["batch_pairs"]
    resolved_config["training"]["timesteps_per_batch"] = args.timesteps_per_batch or training_cfg.get("timesteps_per_batch", 8)
    resolved_config["training"]["evaluation_mode"] = args.evaluation_mode or training_cfg.get("evaluation_mode", "sampled_multistep")
    resolved_config["training"]["evaluation_timesteps_per_batch"] = args.evaluation_timesteps_per_batch or training_cfg.get("evaluation_timesteps_per_batch", 16)
    resolved_config["training"]["aggregation"] = args.aggregation or training_cfg.get("aggregation", "mean")
    (output_dir / "config.snapshot.yaml").write_text(yaml.safe_dump(resolved_config, sort_keys=False, allow_unicode=True), encoding="utf-8")

    manifest: dict[str, Any] = {
        "status": "running",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config_path": str(config_path),
        "config_snapshot": str(output_dir / "config.snapshot.yaml"),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "pocket_pdb": str(pocket_path),
        "train_pair_manifest": str(train_manifest),
        "validation_pair_manifest": str(validation_manifest),
        "train_pair_manifest_sha256": sha256_file(train_manifest),
        "validation_pair_manifest_sha256": sha256_file(validation_manifest),
        "train_pairs": len(train_pairs),
        "validation_pairs": len(validation_pairs),
        "num_timesteps": 1000,
        "device": device,
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
        "git": _git_facts(ROOT),
        "cuda": _cuda_facts(device),
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    _write_json(output_dir / "run_status.json", {"status": "initializing", "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})

    LOGGER.info("config=%s output_dir=%s", config_path, output_dir)
    LOGGER.info("pairs train=%d validation=%d minimum_validation=%d", len(train_pairs), len(validation_pairs), minimum_validation_pairs)
    LOGGER.info("cuda=%s", manifest["cuda"])
    current = DiffSBDDGaussianTransitionScorer(
        str(checkpoint_path), str(pocket_path), str(model_cfg["ref_ligand"]),
        device=device, dataset_name=str(model_cfg["dataset_name"]), num_timesteps=1000,
    )
    reference = DiffSBDDGaussianTransitionScorer(
        str(checkpoint_path), str(pocket_path), str(model_cfg["ref_ligand"]),
        device=device, dataset_name=str(model_cfg["dataset_name"]), num_timesteps=1000,
    )
    valid_timesteps = current.valid_transition_timesteps()
    manifest.update({"effective_valid_steps": len(valid_timesteps), "valid_timesteps": valid_timesteps})
    _write_json(output_dir / "run_manifest.json", manifest)

    trainer = GaussianDPOTrainer(
        current=current,
        reference=reference,
        train_pairs=train_pairs,
        validation_pairs=validation_pairs,
        config=TrainerConfig(
            output_dir=output_dir,
            epochs=int(args.epochs or training_cfg["epochs"]),
            batch_pairs=int(args.batch_pairs or training_cfg["batch_pairs"]),
            learning_rate=float(args.learning_rate or training_cfg["learning_rate"]),
            beta=float(args.beta or dpo_cfg["beta"]),
            max_grad_norm=float(training_cfg["max_grad_norm"]),
            num_timesteps=1000,
            reduction=str(dpo_cfg["reduction"]),
            include_normal_constant=bool(dpo_cfg["include_normal_constant"]),
            include_sigma_term=bool(dpo_cfg["include_sigma_term"]),
            x_weight=float(dpo_cfg.get("x_weight", 1.0)),
            v_weight=float(dpo_cfg.get("v_weight", 1.0)),
            mode=str(resolved_config["training"]["mode"]),  # type: ignore[arg-type]
            train_timesteps_per_batch=int(resolved_config["training"]["timesteps_per_batch"]),
            evaluation_mode=str(resolved_config["training"]["evaluation_mode"]),  # type: ignore[arg-type]
            evaluation_timesteps_per_batch=int(resolved_config["training"]["evaluation_timesteps_per_batch"]),
            aggregation=str(resolved_config["training"]["aggregation"]),  # type: ignore[arg-type]
            seed=int(run_cfg["seed"]),
            resume_checkpoint=resume_path,
        ),
        checkpoint_source=str(checkpoint_path),
    )
    trainer.train()
    manifest.update({"status": "completed", "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    _write_json(output_dir / "run_manifest.json", manifest)


if __name__ == "__main__":
    main()
