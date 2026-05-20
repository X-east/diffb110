#!/usr/bin/env python
"""Print shell assignments for launch settings derived from a YAML config."""

from __future__ import annotations

import argparse
import shlex

import yaml


def _gpu_list(num_gpus: int) -> str:
    return ",".join(str(i) for i in range(max(1, num_gpus)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    args = parser.parse_args()

    with open(args.config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    train = config.get("train", {}) or {}
    ddp = config.get("ddp", {}) or {}
    dataloader = config.get("dataloader", {}) or {}
    reward = config.get("reward", {}) or {}

    ddp_enabled = bool(ddp.get("enabled", False))
    num_gpus = int(ddp.get("num_gpus", 1) or 1)
    gpus = ddp.get("gpus")
    if isinstance(gpus, (list, tuple)):
        gpu_list = ",".join(str(item) for item in gpus)
    elif isinstance(gpus, str) and gpus.strip():
        gpu_list = gpus.strip()
    else:
        gpu_list = _gpu_list(num_gpus)

    per_rank_batch = int(dataloader.get("batch_size", train.get("train_micro_batch_size", 1)) or 1)
    global_batch = per_rank_batch * (num_gpus if ddp_enabled else 1)
    seed = int(train.get("seed", 42))
    run_name = str(config.get("name", "run"))
    rewards = reward.get("enabled", [])
    rewards_text = rewards if isinstance(rewards, str) else "+".join(str(item) for item in rewards)

    values = {
        "YAML_RUN_NAME": run_name,
        "YAML_REWARDS": rewards_text,
        "YAML_SEED": str(seed),
        "YAML_DDP_ENABLED": "1" if ddp_enabled else "0",
        "YAML_NUM_GPUS": str(num_gpus),
        "YAML_GPUS": gpu_list,
        "YAML_PER_RANK_BATCH_SIZE": str(per_rank_batch),
        "YAML_GLOBAL_BATCH_SIZE": str(global_batch),
    }
    for key, value in values.items():
        print(f"export {key}={shlex.quote(value)}")


if __name__ == "__main__":
    main()
