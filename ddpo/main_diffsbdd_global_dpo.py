"""Run endpoint-based DiffSBDD Global DPO."""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml

from ddpo.global_dpo.endpoint_data import load_pair_manifest, validate_split_pairs
from ddpo.global_dpo.generation import generate_epoch
from ddpo.global_dpo.trainer import GlobalDPOTrainer, GlobalDPOTrainerConfig, sha256_file
from ddpo.integration.diffsbdd_model import DiffSBDDModel


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse configuration and controlled run overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--pair-count", type=int, default=None)
    parser.add_argument("--allow-small-pairs", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-pairs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--objective", choices=("binary_dpo", "e2po"), default=None)
    parser.add_argument("--reward-temperature", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--train-timesteps-per-batch", type=int, default=None)
    parser.add_argument("--validation-timesteps", type=int, nargs="+", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--disable-generation", action="store_true")
    parser.add_argument("--save-every-updates", type=int, default=None)
    return parser.parse_args()


def resolve(root: Path, value: str | Path) -> Path:
    """Resolve project-relative paths."""
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def _git_facts(root: Path) -> dict[str, Any]:
    """Collect compact git provenance."""
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
    """Collect PyTorch CUDA facts."""
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
    """Write JSON with stable readable formatting."""
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _environment_name() -> str:
    """Report the environment from the executable when direct conda Python is used."""
    configured = os.environ.get("CONDA_DEFAULT_ENV", "")
    executable = str(Path(sys.executable).resolve())
    if "/envs/dif/" in executable:
        return "dif"
    return configured or "unknown"


def _seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible pilot scheduling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    """Build current/reference models and run endpoint Global DPO."""
    args = parse_args()
    config_path = resolve(ROOT, args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_cfg = config["run"]
    model_cfg = config["model"]
    data_cfg = config["data"]
    diffusion_cfg = config["diffusion"]
    dpo_cfg = config["dpo"]
    training_cfg = config["training"]
    variants = config["variants"]
    variant_name = str(args.variant or run_cfg.get("variant", "native_joint"))
    if variant_name not in variants:
        raise ValueError(f"unknown variant={variant_name}; expected one of {sorted(variants)}")
    device = str(args.device or model_cfg["device"])
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the formal Global DPO run")
    seed = int(run_cfg["seed"])
    _seed_everything(seed)
    output_dir = resolve(ROOT, args.output_dir or run_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "completion.json").exists() and not args.resume:
        raise FileExistsError(f"completed run exists; use --resume or a new output directory: {output_dir}")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(output_dir / "train.log", encoding="utf-8")],
        force=True,
    )

    pair_dir = resolve(ROOT, data_cfg["pair_dir"])
    train_manifest = pair_dir / "train_pairs.csv"
    validation_manifest = pair_dir / "validation_pairs.csv"
    test_manifest = pair_dir / "test_pairs.csv"
    reward_sidecar_value = data_cfg.get("reward_sidecar")
    reward_sidecar = resolve(ROOT, reward_sidecar_value) if reward_sidecar_value else None
    reward_key = str(data_cfg.get("reward_key", "reward_total_raw"))
    reward_sign = float(dpo_cfg.get("reward_sign", 1.0))
    allow_endpoint_reuse = bool(data_cfg.get("allow_endpoint_reuse", False))
    if reward_sidecar is not None and not reward_sidecar.is_file():
        raise FileNotFoundError(f"reward sidecar not found: {reward_sidecar}")
    train_pairs = load_pair_manifest(
        train_manifest,
        reward_sidecar=reward_sidecar,
        reward_key=reward_key,
        reward_sign=reward_sign,
        allow_endpoint_reuse=allow_endpoint_reuse,
    )
    validation_pairs = load_pair_manifest(
        validation_manifest,
        reward_sidecar=reward_sidecar,
        reward_key=reward_key,
        reward_sign=reward_sign,
        allow_endpoint_reuse=allow_endpoint_reuse,
    )
    test_pairs = (
        load_pair_manifest(
            test_manifest,
            reward_sidecar=reward_sidecar,
            reward_key=reward_key,
            reward_sign=reward_sign,
            allow_endpoint_reuse=allow_endpoint_reuse,
        )
        if test_manifest.is_file()
        else None
    )
    if args.pair_count is not None:
        if args.pair_count <= 0:
            raise ValueError("--pair-count must be positive")
        train_pairs = train_pairs[: args.pair_count]
        validation_pairs = validation_pairs[: args.pair_count]
        if test_pairs is not None:
            test_pairs = test_pairs[: args.pair_count]
    minimum_validation_pairs = int(data_cfg.get("minimum_validation_pairs", 2000))
    if len(validation_pairs) < minimum_validation_pairs and not args.allow_small_pairs:
        raise ValueError(
            f"validation pairs={len(validation_pairs)} below formal minimum={minimum_validation_pairs}; "
            "use --allow-small-pairs only for a controlled smoke"
        )
    validate_split_pairs(train_pairs, validation_pairs, test_pairs)

    checkpoint_path = resolve(ROOT, model_cfg["checkpoint"])
    pocket_path = resolve(ROOT, model_cfg["pocket_pdb"])
    if not checkpoint_path.is_file() or not pocket_path.is_file():
        raise FileNotFoundError(f"missing model input: checkpoint={checkpoint_path}, pocket={pocket_path}")

    timestep_grid = int(diffusion_cfg.get("timestep_grid", 500))
    validation_timesteps = tuple(
        int(value)
        for value in (
            args.validation_timesteps
            or training_cfg.get("validation_timesteps", [50, 125, 250, 375, 500])
        )
    )
    objective = str(args.objective or dpo_cfg.get("objective", "binary_dpo"))
    reward_temperature = float(
        args.reward_temperature
        if args.reward_temperature is not None
        else dpo_cfg.get("reward_temperature", 1.0)
    )
    if objective == "e2po":
        all_pairs = train_pairs + validation_pairs + (test_pairs or [])
        missing_reward_ids = [
            pair.winner_id
            for pair in all_pairs
            if pair.reward_winner is None or pair.reward_loser is None
        ]
        if missing_reward_ids:
            raise ValueError(
                "E2PO requires rewards for every endpoint; missing rewards for "
                f"{missing_reward_ids[:3]}"
            )
        reward_gaps = np.asarray(
            [float(pair.reward_winner) - float(pair.reward_loser) for pair in all_pairs],
            dtype=np.float64,
        )
        if not np.isfinite(reward_gaps).all():
            raise ValueError("E2PO reward gaps must be finite")
        if np.allclose(reward_gaps, reward_gaps[0]):
            LOGGER.warning(
                "E2PO reward gaps are constant at %.6f; this run behaves like "
                "a fixed-soft-label DPO and is not a continuous-energy test",
                float(reward_gaps[0]),
            )
        reward_gap_facts = {
            "min": float(reward_gaps.min()),
            "max": float(reward_gaps.max()),
            "mean": float(reward_gaps.mean()),
            "unique_count": int(np.unique(reward_gaps).size),
        }
    else:
        reward_gap_facts = None
    variant = variants[variant_name]
    x_weight = float(variant["x_weight"])
    v_weight = float(variant["v_weight"])
    freeze_cfg = training_cfg.get("freeze", {})
    freeze_encoder_decoder = bool(freeze_cfg.get("encoder_decoder", False))
    freeze_egnn_first_n_blocks = int(freeze_cfg.get("egnn_first_n_blocks", 0))
    noise_coupling = str(training_cfg.get("noise_coupling", "common_random"))
    if freeze_egnn_first_n_blocks < 0:
        raise ValueError("training.freeze.egnn_first_n_blocks must be non-negative")
    resolved_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    resolved_config["run"]["output_dir"] = str(output_dir)
    resolved_config["run"]["variant"] = variant_name
    resolved_config["data"]["train_pair_manifest"] = str(train_manifest)
    resolved_config["data"]["validation_pair_manifest"] = str(validation_manifest)
    resolved_config["data"]["test_pair_manifest"] = str(test_manifest) if test_manifest.is_file() else None
    resolved_config["data"]["reward_sidecar"] = str(reward_sidecar) if reward_sidecar is not None else None
    resolved_config["data"]["reward_key"] = reward_key
    resolved_config["training"]["validation_timesteps"] = list(validation_timesteps)
    resolved_config["training"]["x_weight"] = x_weight
    resolved_config["training"]["v_weight"] = v_weight
    resolved_config["training"].setdefault("freeze", {})["encoder_decoder"] = freeze_encoder_decoder
    resolved_config["training"]["freeze"]["egnn_first_n_blocks"] = freeze_egnn_first_n_blocks
    resolved_config["training"]["batch_pairs"] = int(args.batch_pairs or training_cfg["batch_pairs"])
    resolved_config["training"]["save_every_updates"] = int(args.save_every_updates or training_cfg.get("save_every_updates", 0))
    resolved_config["dpo"]["objective"] = objective
    resolved_config["dpo"]["reward_temperature"] = reward_temperature
    resolved_config["dpo"]["reward_sign"] = reward_sign
    resolved_config["diffusion"]["timestep_grid"] = timestep_grid
    (output_dir / "config.snapshot.yaml").write_text(
        yaml.safe_dump(resolved_config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    resume_path: Path | None = None
    if args.resume:
        latest_path = output_dir / "checkpoint_latest.json"
        if not latest_path.is_file():
            raise FileNotFoundError(f"--resume requested but missing {latest_path}")
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
        resume_path = resolve(ROOT, latest["path"])

    manifest: dict[str, Any] = {
        "status": "running",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config_path": str(config_path),
        "config_snapshot": str(output_dir / "config.snapshot.yaml"),
        "variant": variant_name,
        "objective": objective,
        "reward_temperature": reward_temperature,
        "reward_key": reward_key,
        "reward_sign": reward_sign,
        "x_weight": x_weight,
        "v_weight": v_weight,
        "freeze": {
            "encoder_decoder": freeze_encoder_decoder,
            "egnn_first_n_blocks": freeze_egnn_first_n_blocks,
        },
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "pocket_pdb": str(pocket_path),
        "train_pair_manifest": str(train_manifest),
        "validation_pair_manifest": str(validation_manifest),
        "test_pair_manifest": str(test_manifest) if test_manifest.is_file() else None,
        "train_pair_manifest_sha256": sha256_file(train_manifest),
        "validation_pair_manifest_sha256": sha256_file(validation_manifest),
        "test_pair_manifest_sha256": sha256_file(test_manifest) if test_manifest.is_file() else None,
        "train_pairs": len(train_pairs),
        "validation_pairs": len(validation_pairs),
        "test_pairs": None if test_pairs is None else len(test_pairs),
        "timestep_grid": timestep_grid,
        "reward_sidecar": str(reward_sidecar) if reward_sidecar is not None else None,
        "reward_sidecar_sha256": sha256_file(reward_sidecar) if reward_sidecar is not None else None,
        "reward_gap_facts": reward_gap_facts,
        "validation_timesteps": list(validation_timesteps),
        "device": device,
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "conda_environment": _environment_name(),
        "git": _git_facts(ROOT),
        "cuda": _cuda_facts(device),
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    _write_json(output_dir / "run_status.json", {"status": "initializing", "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    LOGGER.info(
        "variant=%s objective=%s train_pairs=%d validation_pairs=%d",
        variant_name,
        objective,
        len(train_pairs),
        len(validation_pairs),
    )
    LOGGER.info(
        "reward_key=%s reward_sign=%.4f reward_temperature=%.6f timestep_grid=%d",
        reward_key,
        reward_sign,
        reward_temperature,
        timestep_grid,
    )
    LOGGER.info("x_weight=%.8f v_weight=%.8f validation_timesteps=%s", x_weight, v_weight, validation_timesteps)

    generation_cfg = config.get("generation", {})
    generation_enabled = bool(generation_cfg.get("enabled", False)) and not args.disable_generation
    if generation_enabled:
        LOGGER.info(
            "per_epoch_generation=enabled attempts=%d device=%s",
            int(generation_cfg.get("attempts", 1000)),
            str(generation_cfg.get("device", "cuda:0")),
        )

    current = DiffSBDDModel(
        checkpoint_path=str(checkpoint_path),
        pocket_pdb_path=str(pocket_path),
        ref_ligand_path=str(model_cfg["ref_ligand"]),
        device=device,
        dataset_name=str(model_cfg["dataset_name"]),
    )
    reference = DiffSBDDModel(
        checkpoint_path=str(checkpoint_path),
        pocket_pdb_path=str(pocket_path),
        ref_ligand_path=str(model_cfg["ref_ligand"]),
        device=device,
        dataset_name=str(model_cfg["dataset_name"]),
    )
    trainer = GlobalDPOTrainer(
        current=current,
        reference=reference,
        train_pairs=train_pairs,
        validation_pairs=validation_pairs,
        test_pairs=test_pairs,
        config=GlobalDPOTrainerConfig(
            output_dir=output_dir,
            epochs=int(args.epochs or training_cfg["epochs"]),
            batch_pairs=int(args.batch_pairs or training_cfg["batch_pairs"]),
            learning_rate=float(args.learning_rate or training_cfg["learning_rate"]),
            beta=float(args.beta or dpo_cfg["beta"]),
            reward_temperature=reward_temperature,
            max_grad_norm=float(training_cfg["max_grad_norm"]),
            timestep_grid=timestep_grid,
            train_timesteps_per_batch=int(args.train_timesteps_per_batch or training_cfg.get("train_timesteps_per_batch", 1)),
            validation_timesteps=validation_timesteps,
            train_eval_pair_count=int(training_cfg.get("train_eval_pair_count", 512)),
            x_weight=x_weight,
        v_weight=v_weight,
        noise_coupling=noise_coupling,
            freeze_encoder_decoder=freeze_encoder_decoder,
            freeze_egnn_first_n_blocks=freeze_egnn_first_n_blocks,
            objective=objective,
            variant=variant_name,
            seed=seed,
            resume_checkpoint=resume_path,
            save_every_updates=int(args.save_every_updates or training_cfg.get("save_every_updates", 0)),
        ),
        checkpoint_source=str(checkpoint_path),
        epoch_callback=(
            lambda _epoch, checkpoint: generate_epoch(
                dpo_checkpoint=checkpoint,
                base_checkpoint=checkpoint_path,
                output_root=output_dir / "generation",
                pocket_pdb=pocket_path,
                ref_ligand=str(model_cfg["ref_ligand"]),
                generation_device=str(generation_cfg.get("device", "cuda:0")),
                attempts=int(generation_cfg.get("attempts", 1000)),
                batch_size=int(generation_cfg.get("batch_size", 25)),
                timesteps=int(generation_cfg.get("timesteps", timestep_grid)),
                seed=int(generation_cfg.get("seed", 1)),
                reconstruction_method=str(generation_cfg.get("reconstruction_method", "openbabel")),
                resamplings=int(generation_cfg.get("resamplings", 10)),
                jump_length=int(generation_cfg.get("jump_length", 1)),
                all_frags=bool(generation_cfg.get("all_frags", True)),
            )
        ) if generation_enabled else None,
    )
    try:
        trainer.train()
    except BaseException as exc:
        manifest.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        _write_json(output_dir / "run_manifest.json", manifest)
        raise
    manifest.update({"status": "completed", "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    _write_json(output_dir / "run_manifest.json", manifest)


if __name__ == "__main__":
    main()
