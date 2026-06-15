#!/usr/bin/env python
"""Generate DiffSBDD single-reward matrix configs for 2026-06-12 tests."""

from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "launch" / "diffsbdd_8gpu_anticarbon_guard_20260611.yaml"
OUT_DIR = ROOT / "launch" / "generated" / "diffsbdd_single_reward_matrix_20260612"
MATRIX_CSV = OUT_DIR / "matrix.csv"
MATRIX_JSON = OUT_DIR / "matrix.json"
README = OUT_DIR / "README.md"

COMMON_REWARD_WEIGHTS = {
    "docking": 0.0,
    "qed": 0.0,
    "sa": 0.0,
    "tpsa_distribution": 0.0,
    "chembl_prior_ecfp4": 0.0,
    "chembl_prior_fcfp4": 0.0,
    "logp_distribution": 0.0,
    "ligand_efficiency": 0.0,
    "chembl_classifier": 0.0,
    "diversity_ecfp4": 0.0,
    "diversity_fcfp4": 0.0,
    "chemistry_guard": 0.0,
}

MATRIX = [
    {
        "idx": 1,
        "slug": "docking",
        "reward": "docking",
        "purpose": "Test whether docking alone drives high-carbon / low-TPSA drift.",
    },
    {
        "idx": 2,
        "slug": "qed",
        "reward": "qed",
        "purpose": "Test whether QED alone restores generic drug-likeness.",
    },
    {
        "idx": 3,
        "slug": "sa",
        "reward": "sa",
        "purpose": "Test whether synthesis accessibility simplifies molecules too much.",
    },
    {
        "idx": 4,
        "slug": "tpsa_distribution",
        "reward": "tpsa_distribution",
        "purpose": "Test whether TPSA distribution matching prevents polarity collapse.",
    },
    {
        "idx": 5,
        "slug": "ecfp4_prior",
        "reward": "chembl_prior_ecfp4",
        "purpose": "Test whether structural ChEMBL prior improves validity or causes hydrophobic drift.",
    },
    {
        "idx": 6,
        "slug": "fcfp4_prior",
        "reward": "chembl_prior_fcfp4",
        "purpose": "Test whether feature ChEMBL prior preserves hetero/pharmacophore features.",
    },
    {
        "idx": 7,
        "slug": "logp_distribution",
        "reward": "logp_distribution",
        "purpose": "Test whether LogP distribution matching controls hydrophobic drift.",
    },
    {
        "idx": 8,
        "slug": "ligand_efficiency",
        "reward": "ligand_efficiency",
        "purpose": "Test whether ligand efficiency avoids the large-molecule docking shortcut.",
    },
    {
        "idx": 9,
        "slug": "chembl_classifier",
        "reward": "chembl_classifier",
        "purpose": "Test the latest classifier as a single reward and expose classifier blind spots.",
    },
]


def _load_base_config() -> dict[str, Any]:
    with BASE_CONFIG.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _common_config(base: dict[str, Any]) -> dict[str, Any]:
    cfg = deepcopy(base)
    cfg["model"]["time_step"] = 200

    dataloader = cfg.setdefault("dataloader", {})
    dataloader["batch_size"] = 18
    dataloader["rollout_micro_batch_size"] = 18
    dataloader["rollout_accum_steps"] = 1

    train = cfg.setdefault("train", {})
    train["train_micro_batch_size"] = 16
    train["gradient_reference_batch_size"] = 64
    train["learning_rate"] = 2.0e-5
    train["kl_coef"] = 0.05
    train["kl_mode"] = "analytic_current_ref"
    train["kl_k3_logratio_clip"] = 30.0
    train["ppo_epochs"] = 2
    train["epochs"] = 10
    train["checkpoint_interval"] = 5

    ddp = cfg.setdefault("ddp", {})
    ddp["enabled"] = True
    ddp["num_gpus"] = 8
    ddp["gpus"] = "0,1,2,3,4,5,6,7"

    pool = cfg.setdefault("good_molecule_pool", {})
    pool["enabled"] = False

    reward = cfg.setdefault("reward", {})
    reward["weights"] = deepcopy(COMMON_REWARD_WEIGHTS)
    reward["monitor_components"] = ["docking"]
    reward.setdefault("normalization", {}).setdefault("failure", {})["advantage_override"] = -2.0
    reward.setdefault("normalization", {}).setdefault("clamp", {})["enabled"] = True
    reward["normalization"]["clamp"]["min"] = -2.0
    reward["normalization"]["clamp"]["max"] = 2.0

    reward.setdefault("qed", {})["failure_penalty"] = -1.0
    reward.setdefault("sa", {})["failure_penalty"] = -1.0

    reward.setdefault("chembl_classifier", {})
    reward["chembl_classifier"].update(
        {
            "model_path": "./chembl_classifier/models/chembl_classifier_selected_v0_3_1_scaffold_strongpoly.pt",
            "failure_penalty": -1.0,
            "good_threshold": 0.5,
            "bad_threshold": 0.5,
            "cap_high": 0.8,
            "cap_low": 0.2,
            "max_reward": 1.0,
            "penalty": -1.0,
            "device": "cuda",
        }
    )

    reward.setdefault("tpsa_distribution", {})
    reward["tpsa_distribution"].update(
        {
            "chembl_csv": "./data/chembl/chembl_36_properties.csv.gz",
            "tpsa_bins": [
                -float("inf"),
                20.0,
                40.0,
                60.0,
                80.0,
                100.0,
                120.0,
                160.0,
                float("inf"),
            ],
            "failure_penalty": -1.0,
            "min_reward": -2.0,
            "max_reward": 2.0,
            "scale": 12.0,
            "ema_alpha": 0.0,
            "reward_prefix": "tpsa_distribution",
        }
    )

    if "chemistry_guard" in reward:
        reward["chemistry_guard"]["enabled"] = False

    return cfg


def _config_for_variant(base: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    cfg = _common_config(base)
    idx = int(variant["idx"])
    slug = str(variant["slug"])
    reward_name = str(variant["reward"])
    cfg["name"] = f"diffsbdd-single-reward-20260612-{idx:02d}-{slug}"
    cfg["train"]["seed"] = 26061200 + idx
    cfg["reward"]["weights"][reward_name] = 1.0
    return cfg


def _write_yaml(path: Path, cfg: dict[str, Any]) -> None:
    text = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False, width=1000)
    path.write_text(text, encoding="utf-8")


def _write_matrix_files(rows: list[dict[str, Any]]) -> None:
    fieldnames = ["idx", "slug", "reward", "config", "name", "purpose"]
    with MATRIX_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    MATRIX_JSON.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# DiffSBDD Single-Reward Matrix 20260612",
        "",
        "Common settings: 200 diffusion steps, 8 GPUs, per-rank batch 18, PPO epochs 2, 10 epochs, lr=2e-5, KL=0.05 analytic_current_ref.",
        "",
        "Only one reward weight is set to 1.0 in each config. Docking is still run as a monitor component in every batch, so docking and ligand-efficiency drift can be compared across all rewards without changing the optimized objective. Chemistry guard and GoodMoleculePool are disabled.",
        "",
        "| idx | slug | reward | config | purpose |",
        "|---:|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['idx']} | {row['slug']} | {row['reward']} | `{Path(row['config']).name}` | {row['purpose']} |"
        )
    README.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = _load_base_config()
    rows: list[dict[str, Any]] = []

    for variant in MATRIX:
        cfg = _config_for_variant(base, variant)
        idx = int(variant["idx"])
        slug = str(variant["slug"])
        config_path = OUT_DIR / f"{idx:02d}_{slug}.yaml"
        _write_yaml(config_path, cfg)
        rows.append(
            {
                "idx": idx,
                "slug": slug,
                "reward": variant["reward"],
                "config": str(config_path.relative_to(ROOT)),
                "name": cfg["name"],
                "purpose": variant["purpose"],
            }
        )

    _write_matrix_files(rows)
    print(OUT_DIR.relative_to(ROOT))


if __name__ == "__main__":
    main()
