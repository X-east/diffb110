#!/usr/bin/env python3
"""Launch 6-GPU DDPO training with all common knobs at the top.

Edit the constants in the USER SETTINGS section, then run this script from a
normal user terminal or tmux session. The script writes a generated YAML config
under launch/generated/python_launch/ and launches torchrun with that config.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

# =============================================================================
# USER SETTINGS — edit these values for experiments
# =============================================================================

# Paths and runtime
PROJECT_ROOT = Path("/home/md666/data/disk6/DiffB110")
BASE_CONFIG_PATH = PROJECT_ROOT / "launch/6gpu_dock.yaml"
GENERATED_CONFIG_DIR = PROJECT_ROOT / "launch/generated/python_launch"
PYTHON_BIN = Path("/home/md666/miniforge3/envs/dif/bin/python")
TORCHRUN_BIN = Path("/home/md666/miniforge3/envs/dif/bin/torchrun")
BABEL_LIBDIR = "/home/md666/miniforge3/envs/dif/lib/openbabel/3.1.0"
TRAIN_ENTRYPOINT = "ddpo/main_diffsbdd_ddp.py"
LOG_LEVEL = "INFO"
MASTER_ADDR = "127.0.0.1"
MASTER_PORT = "29500"

# Run identity
RUN_NAME = "6gpu-dock-chembl-50-50"
SEED = 1
EPOCHS = 25

# GPU / DDP
NUM_GPUS = 6
GPU_IDS = "0,1,2,3,4,5"
FIND_UNUSED_PARAMETERS = True
PARAM_SYNC_CHECK = False
GRAD_SYNC_PROBE = True

# Reward weights. Components with weight <= 0 are disabled by reward factory.
# Do not set all weights to 0, because factory falls back to qed+sa.
QED_WEIGHT = 0.0
SA_WEIGHT = 0.0
DOCKING_WEIGHT = 0.5
CHEMBL_PRIOR_WEIGHT = 0.5

# ChEMBL prior / similarity penalty
CHEMBL_PRIOR_ENABLE = True
CHEMBL_PRIOR_CACHE = "./data/chembl/chembl_36_morgan_fingerprints.pkl.gz"
CHEMBL_PRIOR_TOP_K = 5
CHEMBL_PRIOR_CHUNK_SIZE = 100000
CHEMBL_PRIOR_LOW_THRESHOLD = 0.20
CHEMBL_PRIOR_LOW_WEIGHT = 0.5
CHEMBL_PRIOR_HIGH_THRESHOLD = 0.85
CHEMBL_PRIOR_HIGH_WEIGHT = 0.1
CHEMBL_PRIOR_FAILURE_PENALTY = -1.0

# Docking reward settings. Keep these explicit and config-backed.
DOCKING_BACKEND = "unidock"
DOCKING_RECEPTOR_FILE = "./data/8gjl.pdbqt"
DOCKING_NORMALIZATION_FACTOR = -10.0
DOCKING_CENTER_X = 207.293
DOCKING_CENTER_Y = 212.430
DOCKING_CENTER_Z = 172.771
DOCKING_SIZE_X = 20.0
DOCKING_SIZE_Y = 20.0
DOCKING_SIZE_Z = 20.0
DOCKING_EXHAUSTIVENESS = 8
DOCKING_SEARCH_MODE = "fast"

# Pocket metadata
POCKET_PDB_PATH = "./data/8GJL.pdb"
POCKET_REF_LIGAND = "C:1101"
POCKET_PDBQT_PATH = "./data/8gjl.pdbqt"

# Dataloader and molecule size sampling
BATCH_SIZE = 24
TRAIN_MICRO_BATCH_SIZE = 12
MIN_NODES = 10
MAX_NODES = 30
MAX_N_NODES = 106
NODE_SAMPLING = "size_distribution"

# PPO / DDPO optimization
LEARNING_RATE = 1.0e-5
KL_COEF = 0.0
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_WEIGHT_DECAY = 1.0e-4
ADAM_EPSILON = 1.0e-8
CLIP_RANGE = 0.1
MAX_GRAD_NORM = 1.0
PPO_EPOCHS = 4
PPO_SHUFFLE_MICROBATCH = False
CHECKPOINT_INTERVAL = 5

# Reward normalization
REWARD_NORMALIZATION_ENABLED = True
REWARD_NORMALIZATION_METHOD = "zscore_total"
REWARD_NORMALIZATION_EPSILON = 1.0e-6
REWARD_FAILURE_RAW_PENALTY = -5.0
REWARD_FAILURE_INCLUDE_IN_STATS = False
REWARD_FAILURE_ADVANTAGE_OVERRIDE = -1.5
REWARD_CLAMP_ENABLED = True
REWARD_CLAMP_MIN = -3.0
REWARD_CLAMP_MAX = 3.0

# Safety. Keep DRY_RUN=True to only write and print the generated config.
DRY_RUN = False

# =============================================================================
# Implementation below. Usually no need to edit.
# =============================================================================


def _require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def _positive_reward_weights(weights: dict[str, float]) -> dict[str, float]:
    return {name: float(value) for name, value in weights.items() if float(value) > 0.0}


def _validate_reward_weights(weights: dict[str, float]) -> None:
    enabled = _positive_reward_weights(weights)
    if not enabled:
        raise ValueError(
            "At least one reward weight must be > 0. "
            "Do not set all reward weights to 0 because reward factory falls back to qed+sa."
        )
    if "docking" in enabled and DOCKING_BACKEND == "autodock_gpu" and "CUDA_VISIBLE_DEVICES" in os.environ:
        raise RuntimeError(
            "This launcher defaults to UniDock. If you intentionally switch DOCKING_BACKEND to "
            "autodock_gpu, review AutoDock-GPU's 1-indexed device mapping before using "
            "CUDA_VISIBLE_DEVICES."
        )


def _load_base_config() -> dict[str, Any]:
    _require_file(BASE_CONFIG_PATH, "Base config")
    with BASE_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _build_config() -> dict[str, Any]:
    data = _load_base_config()

    data["name"] = RUN_NAME
    data["dataset"] = data.get("dataset", "crossdock")

    model = data.setdefault("model", {})
    model.setdefault("checkpoint", "./DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt")
    model["time_step"] = int(model.get("time_step", 500))
    model["device"] = "cuda"

    pocket = data.setdefault("pocket", {})
    pocket["pdb_path"] = POCKET_PDB_PATH
    pocket["ref_ligand"] = POCKET_REF_LIGAND
    pocket["pdbqt_path"] = POCKET_PDBQT_PATH

    reward = data.setdefault("reward", {})
    weights = {
        "qed": float(QED_WEIGHT),
        "sa": float(SA_WEIGHT),
        "docking": float(DOCKING_WEIGHT),
        "chembl_prior": float(CHEMBL_PRIOR_WEIGHT) if CHEMBL_PRIOR_ENABLE else 0.0,
    }
    _validate_reward_weights(weights)
    reward["weights"] = weights

    reward["normalization"] = {
        "enabled": bool(REWARD_NORMALIZATION_ENABLED),
        "method": REWARD_NORMALIZATION_METHOD,
        "epsilon": float(REWARD_NORMALIZATION_EPSILON),
        "use_rewards_key": "reward_total_raw",
        "raw_output_key": "reward_total_raw",
        "norm_output_key": "reward_total_norm",
        "clamped_output_key": "reward_total_clamped",
        "map_rewards_field": "reward_total",
        "advantages_key": "advantages",
        "override_rewards_field": True,
        "stats_prefix": "reward",
        "failure": {
            "raw_penalty": float(REWARD_FAILURE_RAW_PENALTY),
            "include_in_stats": bool(REWARD_FAILURE_INCLUDE_IN_STATS),
            "advantage_override": float(REWARD_FAILURE_ADVANTAGE_OVERRIDE),
        },
        "clamp": {
            "enabled": bool(REWARD_CLAMP_ENABLED),
            "min": float(REWARD_CLAMP_MIN),
            "max": float(REWARD_CLAMP_MAX),
        },
    }

    reward["docking"] = {
        "backend": DOCKING_BACKEND,
        "receptor_file": DOCKING_RECEPTOR_FILE,
        "normalization_factor": float(DOCKING_NORMALIZATION_FACTOR),
        "center_x": float(DOCKING_CENTER_X),
        "center_y": float(DOCKING_CENTER_Y),
        "center_z": float(DOCKING_CENTER_Z),
        "size_x": float(DOCKING_SIZE_X),
        "size_y": float(DOCKING_SIZE_Y),
        "size_z": float(DOCKING_SIZE_Z),
        "exhaustiveness": int(DOCKING_EXHAUSTIVENESS),
        "search_mode": DOCKING_SEARCH_MODE,
    }

    reward["chembl_prior"] = {
        "fingerprint_cache": CHEMBL_PRIOR_CACHE,
        "top_k": int(CHEMBL_PRIOR_TOP_K),
        "chunk_size": int(CHEMBL_PRIOR_CHUNK_SIZE),
        "low_similarity_threshold": float(CHEMBL_PRIOR_LOW_THRESHOLD),
        "low_similarity_weight": float(CHEMBL_PRIOR_LOW_WEIGHT),
        "high_similarity_threshold": float(CHEMBL_PRIOR_HIGH_THRESHOLD),
        "high_similarity_weight": float(CHEMBL_PRIOR_HIGH_WEIGHT),
        "failure_penalty": float(CHEMBL_PRIOR_FAILURE_PENALTY),
    }

    data["dataloader"] = {
        "batch_size": int(BATCH_SIZE),
        "min_nodes": int(MIN_NODES),
        "max_nodes": int(MAX_NODES),
        "max_n_nodes": int(MAX_N_NODES),
        "node_sampling": NODE_SAMPLING,
    }

    data["train"] = {
        "train_micro_batch_size": int(TRAIN_MICRO_BATCH_SIZE),
        "learning_rate": float(LEARNING_RATE),
        "kl_coef": float(KL_COEF),
        "adam_beta1": float(ADAM_BETA1),
        "adam_beta2": float(ADAM_BETA2),
        "adam_weight_decay": float(ADAM_WEIGHT_DECAY),
        "adam_epsilon": float(ADAM_EPSILON),
        "clip_range": float(CLIP_RANGE),
        "max_grad_norm": float(MAX_GRAD_NORM),
        "ppo_epochs": int(PPO_EPOCHS),
        "ppo_shuffle_microbatch": bool(PPO_SHUFFLE_MICROBATCH),
        "checkpoint_interval": int(CHECKPOINT_INTERVAL),
        "seed": int(SEED),
        "epochs": int(EPOCHS),
    }

    data["ddp"] = {
        "enabled": True,
        "num_gpus": int(NUM_GPUS),
        "gpus": GPU_IDS,
        "find_unused_parameters": bool(FIND_UNUSED_PARAMETERS),
        "param_sync_check": bool(PARAM_SYNC_CHECK),
        "grad_sync_probe": bool(GRAD_SYNC_PROBE),
    }

    data["output"] = {"dir": "./OUT/<run_name>_<YYYYMMDD>_<seq>"}
    return data


def _write_config(data: dict[str, Any]) -> Path:
    GENERATED_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = GENERATED_CONFIG_DIR / f"{RUN_NAME}_{timestamp}.yaml"
    out_path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return out_path


def _print_launch_summary(config_path: Path, data: dict[str, Any]) -> None:
    weights = data["reward"]["weights"]
    enabled = _positive_reward_weights(weights)
    total = sum(enabled.values())
    normalized = {name: value / total for name, value in enabled.items()}

    print("=" * 60)
    print(f"Run name : {RUN_NAME}")
    print(f"Config   : {config_path}")
    print(f"GPUs     : {GPU_IDS} ({NUM_GPUS} ranks)")
    print(f"Rewards  : raw={weights}")
    print(f"Rewards  : normalized={normalized}")
    print(
        "ChEMBL  : "
        f"enable={CHEMBL_PRIOR_ENABLE} "
        f"weight={CHEMBL_PRIOR_WEIGHT} "
        f"low_thr={CHEMBL_PRIOR_LOW_THRESHOLD} "
        f"high_thr={CHEMBL_PRIOR_HIGH_THRESHOLD} "
        f"cache={CHEMBL_PRIOR_CACHE}"
    )
    print(
        "Docking : "
        f"backend={DOCKING_BACKEND} center=({DOCKING_CENTER_X}, {DOCKING_CENTER_Y}, {DOCKING_CENTER_Z}) "
        f"size=({DOCKING_SIZE_X}, {DOCKING_SIZE_Y}, {DOCKING_SIZE_Z})"
    )
    print(f"Batch    : per_rank={BATCH_SIZE} global={BATCH_SIZE * NUM_GPUS} micro={TRAIN_MICRO_BATCH_SIZE}")
    print(f"PPO      : epochs={PPO_EPOCHS} kl_coef={KL_COEF} lr={LEARNING_RATE} train_epochs={EPOCHS}")
    print(f"Start    : {time.strftime('%F %T')}")
    print("=" * 60)


def _build_command(config_path: Path) -> list[str]:
    torchrun = str(TORCHRUN_BIN if TORCHRUN_BIN.exists() else "torchrun")
    return [
        torchrun,
        f"--nproc_per_node={NUM_GPUS}",
        "--nnodes=1",
        "--node_rank=0",
        f"--master_addr={MASTER_ADDR}",
        f"--master_port={MASTER_PORT}",
        TRAIN_ENTRYPOINT,
        "--config_path",
        str(config_path),
        "--log_level",
        LOG_LEVEL,
    ]


def main() -> int:
    _require_file(PYTHON_BIN, "Python executable")
    _require_file(PROJECT_ROOT / TRAIN_ENTRYPOINT, "Training entrypoint")

    data = _build_config()
    config_path = _write_config(data)
    _print_launch_summary(config_path, data)

    command = _build_command(config_path)
    print("Command  :", " ".join(shlex.quote(part) for part in command))

    if DRY_RUN:
        print("DRY_RUN=True, not launching training.")
        return 0

    env = os.environ.copy()
    env["BABEL_LIBDIR"] = BABEL_LIBDIR
    env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = GPU_IDS

    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)
    print("=" * 60)
    print(f"Training completed. End: {time.strftime('%F %T')}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
