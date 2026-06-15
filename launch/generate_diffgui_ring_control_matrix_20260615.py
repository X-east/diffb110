#!/usr/bin/env python
"""Generate 5 additional DiffGui ring-control matrix configs for 2026-06-15.

Groups:
  9  qed                                 (single)
  10 sa                                  (single)
  11 ring_complexity                     (single)
  12 le + ring_complexity + aromatic_scaffold  (composite)
  13 docking + ring_complexity + aromatic_scaffold  (composite)

The first 3 reuse existing single-reward templates with fresh seeds.
The last 2 are new composite-reward configs.
"""

from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "launch" / "diffgui_8gpu_b24_aromatic_last3_lr3e5.yaml"
OUT_DIR = ROOT / "launch" / "generated" / "diffgui_ring_control_matrix_20260615"

# 2026-06-15: 目录已从 analysis/8gjl_... 移动到 analysis/benchmark/8gjl_...
POCKET_PDB = "analysis/benchmark/8gjl_c1101_benchmark_20260603_500mol_500steps/inputs/8GJL_C1101_pocket8A.pdb"
MATRIX_CSV = OUT_DIR / "matrix.csv"
MATRIX_JSON = OUT_DIR / "matrix.json"

COMMON_ZERO_WEIGHTS = {
    "docking": 0.0,
    "ligand_efficiency": 0.0,
    "qed": 0.0,
    "sa": 0.0,
    "aromatic_scaffold": 0.0,
    "ring_complexity": 0.0,
    "ring_pathology": 0.0,
    "chembl_classifier": 0.0,
    "chemistry_guard": 0.0,
}

GROUPS = [
    {"idx": 9,  "slug": "qed",                "weights": {"qed": 1.0},  "purpose": "Test whether generic drug-likeness preserves useful ring composition."},
    {"idx": 10, "slug": "sa",                 "weights": {"sa": 1.0},   "purpose": "Test whether synthesis accessibility suppresses complex rings too strongly."},
    {"idx": 11, "slug": "ring_complexity",     "weights": {"ring_complexity": 1.0}, "purpose": "Softly penalize four-or-more rings and excessive fused/bridge/spiro topology."},
    {
        "idx": 12,
        "slug": "le_ring_aromatic",
        "weights": {"ligand_efficiency": 0.34, "ring_complexity": 0.33, "aromatic_scaffold": 0.33},
        "purpose": "Composite: LE avoids docking-size shortcut, ring_complexity penalizes excessive topology, aromatic_scaffold preserves benzene.",
    },
    {
        "idx": 13,
        "slug": "dock_ring_aromatic",
        "weights": {"docking": 0.34, "ring_complexity": 0.33, "aromatic_scaffold": 0.33},
        "purpose": "Composite: docking provides pocket pressure, ring_complexity controls topology, aromatic_scaffold preserves benzene.",
    },
]


def _load_base_config() -> dict[str, Any]:
    with BASE_CONFIG.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _common_config(base: dict[str, Any]) -> dict[str, Any]:
    """Return a deep-copied base with all shared settings populated."""
    cfg = deepcopy(base)
    cfg["name"] = "diffgui-ring-control-20260615-placeholder"
    cfg["diffgui_model"]["pocket_pdb"] = POCKET_PDB

    training = cfg.setdefault("diffgui_training", {})
    training["epochs"] = 10
    training["batch_size"] = 18
    training["checkpoint_every_epochs"] = 5

    sweep = cfg.setdefault("diffgui_docking_batch_sweep", {})
    sweep["num_timesteps"] = 200
    sweep["auto_configure_timesteps"] = True
    sweep["ligand_size_mode"] = "priori"
    sweep["ligand_atoms"] = 24
    sweep["seed_base"] = 260615000
    sweep["reward_batch_size"] = None
    sweep["reward_timeout_seconds"] = 3600
    sweep["gui_strength"] = 3.0
    sweep["precision"] = "bf16"
    sweep["graph_micro_batch_size"] = 1
    sweep["normalize_advantages"] = True
    sweep["advantage_epsilon"] = 1.0e-6
    sweep["advantage_clip_min"] = -2.0
    sweep["advantage_clip_max"] = 2.0
    sweep["failure_advantage"] = -1.0

    train = cfg.setdefault("train", {})
    train["learning_rate"] = 3.0e-5
    train["ppo_epochs"] = 1
    train["precision"] = "bf16"
    train["graph_micro_batch_size"] = 1
    train["ddp_find_unused_parameters"] = False
    train.setdefault("freeze", {})
    train["freeze"]["mode"] = "last_n_blocks"
    train["freeze"]["train_last_n_blocks"] = 3

    reward = cfg.setdefault("reward", {})
    reward["weights"] = deepcopy(COMMON_ZERO_WEIGHTS)
    reward["monitor_components"] = ["docking"]
    reward.setdefault("per_component_normalization", {})["enabled"] = False
    reward.setdefault("normalization", {})["enabled"] = False

    reward.setdefault("qed", {})["failure_penalty"] = -1.0
    reward.setdefault("sa", {})["failure_penalty"] = -1.0

    reward.setdefault("ligand_efficiency", {}).update({
        "failure_penalty": -1.0, "min_le": 0.0, "max_le": 1.2, "scale": 1.0,
    })
    reward.setdefault("aromatic_scaffold", {}).update({
        "failure_penalty": -1.0,
        "base_reward": 0.0, "min_reward": -1.0, "max_reward": 1.0,
        "exact_single_benzene_reward": 1.0,
        "missing_benzene_penalty": 0.25,
        "max_benzene_ring_count": 2,
        "extra_benzene_penalty": 0.25,
        "max_heterocycle_ring_count": 999,
        "multi_heterocycle_penalty": 0.0,
        "max_fused_ring_system_size": 4, "fused_penalty": 0.15,
        "max_three_member_ring_count": 1, "three_member_penalty": 0.20,
        "max_total_halogen_count": 3, "total_halogen_penalty": 0.20,
        "max_aromatic_halogen_count": 2, "aromatic_halogen_penalty": 0.20,
    })
    reward.setdefault("ring_complexity", {}).update({
        "failure_penalty": -1.0,
        "base_reward": 1.0, "min_reward": -1.0, "max_reward": 1.0,
        "target_ring_count": 3,
        "target_aromatic_ring_count": 4,
        "target_fused_ring_system_size": 3,
        "max_three_member_ring_count": 1,
        "max_spiro_atom_count": 1,
        "max_bridgehead_atom_count": 2,
        "ring_penalty": 0.25,
        "aromatic_penalty": 0.0,
        "fused_penalty": 0.20,
        "three_member_penalty": 0.30,
        "spiro_penalty": 0.10,
        "bridgehead_penalty": 0.10,
    })
    reward.setdefault("ring_pathology", {}).update({
        "failure_penalty": -1.0,
        "base_reward": 1.0, "min_reward": -1.0, "max_reward": 1.0,
        "max_allowed_ring_count": 3, "min_large_ring_size": 7,
        "max_allowed_three_member_ring_count": 1,
        "ring_count_penalty": 0.25, "large_ring_penalty": 0.45,
        "large_ring_fused_six_penalty": 0.65,
        "extra_three_member_penalty": 0.45,
        "three_member_fused_or_spiro_penalty": 0.65,
    })
    reward.setdefault("chembl_classifier", {}).update({
        "model_path": "./chembl_classifier/models/chembl_classifier_selected_v0_3_1_scaffold_strongpoly.pt",
        "failure_penalty": -1.0,
        "good_threshold": 0.5, "bad_threshold": 0.5,
        "cap_high": 0.8, "cap_low": 0.2,
        "max_reward": 1.0, "penalty": -1.0,
        "device": "cuda",
    })

    if "chemistry_guard" in reward:
        reward["chemistry_guard"]["enabled"] = False

    return cfg


def _config_for_group(base: dict[str, Any], group: dict[str, Any]) -> dict[str, Any]:
    cfg = _common_config(base)
    idx = int(group["idx"])
    slug = str(group["slug"])
    cfg["name"] = f"diffgui-ring-control-20260615-{idx:02d}-{slug}"
    cfg["diffgui_docking_batch_sweep"]["seed_base"] = 260615000 + idx * 100
    cfg["reward"]["weights"] = deepcopy(COMMON_ZERO_WEIGHTS)
    for k, v in group["weights"].items():
        cfg["reward"]["weights"][k] = v
    return cfg


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = _load_base_config()
    rows: list[dict[str, Any]] = []
    for group in GROUPS:
        cfg = _config_for_group(base, group)
        idx = int(group["idx"])
        slug = str(group["slug"])
        config_path = OUT_DIR / f"{idx:02d}_{slug}.yaml"
        text = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False, width=1000)
        config_path.write_text(text, encoding="utf-8")
        rows.append({
            "idx": idx,
            "slug": slug,
            "weights": group["weights"],
            "config": str(config_path.relative_to(ROOT)),
            "name": cfg["name"],
            "purpose": group["purpose"],
        })
    with MATRIX_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["idx", "slug", "weights", "config", "name", "purpose"])
        writer.writeheader()
        writer.writerows(rows)
    MATRIX_JSON.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[generate] wrote {len(rows)} configs to {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
