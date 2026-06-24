#!/usr/bin/env python
"""Generate the 2026-06-18 six-run DiffGui next-test matrix."""

from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "launch" / "archive" / "diffgui_historical" / "diffgui_8gpu_b24_aromatic_last3_lr3e5.yaml"
OUT_DIR = ROOT / "launch" / "archive" / "generated" / "diffgui_next6_matrix_20260618"
MATRIX_CSV = OUT_DIR / "matrix.csv"
MATRIX_JSON = OUT_DIR / "matrix.json"
README = OUT_DIR / "README.md"

POCKET_PDB = "analysis/benchmark/8gjl_c1101_benchmark_20260603_500mol_500steps/inputs/8GJL_C1101_pocket8A.pdb"

ZERO_WEIGHTS = {
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

MATRIX = [
    {
        "idx": 1,
        "slug": "dock_aromatic_pathology",
        "weights": {"docking": 0.40, "aromatic_scaffold": 0.30, "ring_pathology": 0.30},
        "purpose": "Main P0: keep docking pressure while replacing ring_complexity with targeted pathology control.",
    },
    {
        "idx": 2,
        "slug": "le_aromatic_pathology",
        "weights": {"ligand_efficiency": 0.40, "aromatic_scaffold": 0.30, "ring_pathology": 0.30},
        "purpose": "Main P0: preserve LE distribution benefit and directly suppress pathological rings.",
    },
    {
        "idx": 3,
        "slug": "dock_le_aromatic_pathology",
        "weights": {"docking": 0.30, "ligand_efficiency": 0.25, "aromatic_scaffold": 0.25, "ring_pathology": 0.20},
        "purpose": "Main P0: balanced docking, LE, aromatic scaffold, and pathology control candidate.",
    },
    {
        "idx": 4,
        "slug": "ring_last3_pathology_guard",
        "weights": {
            "docking": 0.30,
            "ligand_efficiency": 0.20,
            "aromatic_scaffold": 0.25,
            "ring_pathology": 0.15,
            "chemistry_guard": 0.10,
        },
        "purpose": "Ring-last3 revision: keep the successful scaffold but replace ring_complexity and enable guard.",
    },
    {
        "idx": 5,
        "slug": "chembl_aromatic_pathology",
        "weights": {"chembl_classifier": 0.40, "aromatic_scaffold": 0.30, "ring_pathology": 0.30},
        "purpose": "Classifier route: test whether structural rewards can constrain the classifier positive signal.",
    },
    {
        "idx": 6,
        "slug": "dock_chembl_aromatic_pathology",
        "weights": {"docking": 0.30, "chembl_classifier": 0.25, "aromatic_scaffold": 0.25, "ring_pathology": 0.20},
        "purpose": "Classifier+docking route: combine pocket pressure, learned prior, and structural safeguards.",
    },
]


def _load_base_config() -> dict[str, Any]:
    """Load the previously validated DiffGui training template."""
    with BASE_CONFIG.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Base config is not a mapping: {BASE_CONFIG}")
    return data


def _common_config(base: dict[str, Any]) -> dict[str, Any]:
    """Return shared settings for all six formal next-test runs."""
    cfg = deepcopy(base)
    cfg["name"] = "diffgui-next6-20260618-placeholder"
    cfg.setdefault("diffgui_model", {})["pocket_pdb"] = POCKET_PDB

    training = cfg.setdefault("diffgui_training", {})
    training["output_root"] = "OUT"
    training["epochs"] = 20
    training["batch_size"] = 18
    training["checkpoint_every_epochs"] = 5

    sweep = cfg.setdefault("diffgui_docking_batch_sweep", {})
    sweep["num_timesteps"] = 200
    sweep["auto_configure_timesteps"] = True
    sweep["ligand_size_mode"] = "priori"
    sweep["ligand_atoms"] = 24
    sweep["seed_base"] = 260618000
    sweep["reward_env"] = "dif"
    sweep["reward_device"] = "cuda"
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
    sweep["memory_poll_interval"] = 5.0

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
    reward["weights"] = deepcopy(ZERO_WEIGHTS)
    reward["monitor_components"] = ["docking"]
    reward.setdefault("per_component_normalization", {})["enabled"] = False
    reward.setdefault("normalization", {})["enabled"] = False

    reward.setdefault("qed", {})["failure_penalty"] = -1.0
    reward.setdefault("sa", {})["failure_penalty"] = -1.0

    reward.setdefault("ligand_efficiency", {}).update(
        {
            "failure_penalty": -1.0,
            "min_le": 0.0,
            "max_le": 1.2,
            "scale": 1.0,
        }
    )
    reward.setdefault("aromatic_scaffold", {}).update(
        {
            "failure_penalty": -1.0,
            "base_reward": 0.0,
            "min_reward": -1.0,
            "max_reward": 1.0,
            "exact_single_benzene_reward": 1.0,
            "missing_benzene_penalty": 0.25,
            "max_benzene_ring_count": 2,
            "extra_benzene_penalty": 0.25,
            "max_heterocycle_ring_count": 999,
            "multi_heterocycle_penalty": 0.0,
            "max_fused_ring_system_size": 4,
            "fused_penalty": 0.15,
            "max_three_member_ring_count": 1,
            "three_member_penalty": 0.20,
            "max_total_halogen_count": 3,
            "total_halogen_penalty": 0.20,
            "max_aromatic_halogen_count": 2,
            "aromatic_halogen_penalty": 0.20,
        }
    )
    reward.setdefault("ring_complexity", {}).update(
        {
            "failure_penalty": -1.0,
            "base_reward": 1.0,
            "min_reward": -1.0,
            "max_reward": 1.0,
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
        }
    )
    reward.setdefault("ring_pathology", {}).update(
        {
            "failure_penalty": -1.0,
            "base_reward": 1.0,
            "min_reward": -1.0,
            "max_reward": 1.0,
            "max_allowed_ring_count": 3,
            "min_large_ring_size": 7,
            "max_allowed_three_member_ring_count": 1,
            "ring_count_penalty": 0.25,
            "large_ring_penalty": 0.45,
            "large_ring_fused_six_penalty": 0.65,
            "extra_three_member_penalty": 0.45,
            "three_member_fused_or_spiro_penalty": 0.65,
        }
    )
    reward.setdefault("chembl_classifier", {}).update(
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
    reward.setdefault("chemistry_guard", {}).update(
        {
            "enabled": False,
            "failure_penalty": -1.0,
            "base_reward": 1.0,
            "violation_penalty": 0.2,
            "min_reward": -1.0,
            "max_reward": 1.0,
            "max_carbon_fraction": 0.85,
            "min_hetero_atom_fraction": 0.05,
            "max_three_member_ring_count": 0,
            "max_fragment_count": 1,
            "max_bad_bond_length_fraction": 0.05,
            "max_halogen_bad_position_count": 0,
        }
    )
    return cfg


def _config_for_variant(base: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    """Build one concrete matrix config."""
    cfg = _common_config(base)
    idx = int(variant["idx"])
    slug = str(variant["slug"])
    cfg["name"] = f"diffgui-next6-20260618-{idx:02d}-{slug}"
    cfg["diffgui_docking_batch_sweep"]["seed_base"] = 260618000 + idx * 100
    cfg["reward"]["weights"] = deepcopy(ZERO_WEIGHTS)
    for name, weight in variant["weights"].items():
        cfg["reward"]["weights"][name] = float(weight)
    cfg["reward"]["chemistry_guard"]["enabled"] = float(cfg["reward"]["weights"].get("chemistry_guard", 0.0)) > 0.0
    return cfg


def _write_yaml(path: Path, cfg: dict[str, Any]) -> None:
    text = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False, width=1000)
    path.write_text(text, encoding="utf-8")


def _format_weights(weights: dict[str, float]) -> str:
    return " + ".join(f"{name}:{weight:g}" for name, weight in weights.items())


def _write_matrix_files(rows: list[dict[str, Any]]) -> None:
    fieldnames = ["idx", "slug", "weights", "config", "name", "purpose"]
    with MATRIX_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    MATRIX_JSON.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# DiffGui Next6 Matrix 20260618",
        "",
        "Common settings: 20 epochs, 8 GPUs, per-rank batch 18, 200 diffusion steps, lr=3e-5, PPO epochs 1, freeze last 3 blocks, bf16.",
        "",
        "| idx | slug | weights | config | purpose |",
        "|---:|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['idx']} | {row['slug']} | {row['weights']} | `{Path(row['config']).name}` | {row['purpose']} |"
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
        weights = {name: float(value) for name, value in variant["weights"].items()}
        rows.append(
            {
                "idx": idx,
                "slug": slug,
                "weights": _format_weights(weights),
                "config": str(config_path.relative_to(ROOT)),
                "name": cfg["name"],
                "purpose": variant["purpose"],
            }
        )
    _write_matrix_files(rows)
    print(f"[generate] wrote {len(rows)} configs to {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
