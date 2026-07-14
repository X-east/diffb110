"""PoseBusters-before-reward orchestration for DiffGui training."""

from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import yaml


PoseBustersRunner = Callable[..., Tuple[Dict[str, Any], List[Dict[str, Any]], Path]]
RewardRunner = Callable[..., List[Dict[str, Any]]]


@dataclass(frozen=True)
class PreRewardGateResult:
    """Outputs of the PoseBusters-before-reward stage."""

    posebusters_summary: dict[str, Any]
    posebusters_rows: list[dict[str, Any]]
    reward_rows: list[dict[str, Any]]
    audit: dict[str, int]
    timing: dict[str, float]


def run_posebusters_then_rewards(
    *,
    config: dict[str, Any],
    generated_sdf: Path,
    output_dir: Path,
    total_attempts: int,
    reward_batch_size: int,
    posebusters_runner: PoseBustersRunner,
    reward_runner: RewardRunner,
    logger_obj: logging.Logger | None = None,
) -> PreRewardGateResult:
    """Run the PB legality gate before any molecule reward calculation."""
    log = logger_obj or logging.getLogger(__name__)
    log.info("pre-reward gate phase=posebusters input=%s", generated_sdf)
    phase_start = time.perf_counter()
    posebusters_summary, posebusters_rows, pb_valid_sdf = posebusters_runner(
        config=config,
        input_sdf=generated_sdf,
        output_dir=output_dir,
        total_attempts=total_attempts,
    )
    posebusters_seconds = time.perf_counter() - phase_start
    log.info(
        "pre-reward gate phase=posebusters done pb_valid=%d seconds=%.3f",
        int(posebusters_summary.get("posebusters_pb_valid", 0)),
        posebusters_seconds,
    )

    phase_start = time.perf_counter()
    enabled = bool(posebusters_summary.get("posebusters_enabled", 0))
    log.info("pre-reward gate phase=reward input=%s", pb_valid_sdf)
    if enabled and int(posebusters_summary.get("posebusters_pb_valid", 0)) == 0:
        reward_rows = write_empty_reward_stage_outputs(
            config=config,
            input_sdf=pb_valid_sdf,
            output_dir=output_dir,
            logger_obj=logger_obj,
        )
    else:
        reward_rows = reward_runner(
            config=config,
            input_sdf=pb_valid_sdf,
            output_dir=output_dir,
            batch_size=reward_batch_size,
        )
    reward_seconds = time.perf_counter() - phase_start
    log.info(
        "pre-reward gate phase=reward done rows=%d seconds=%.3f",
        len(reward_rows),
        reward_seconds,
    )
    audit = validate_pre_reward_posebusters_gate(
        reward_rows=reward_rows,
        posebusters_rows=posebusters_rows,
        enabled=enabled,
    )
    return PreRewardGateResult(
        posebusters_summary=posebusters_summary,
        posebusters_rows=posebusters_rows,
        reward_rows=reward_rows,
        audit=audit,
        timing={
            "posebusters_seconds": float(posebusters_seconds),
            "reward_seconds": float(reward_seconds),
        },
    )


def write_empty_reward_stage_outputs(
    *,
    config: dict[str, Any],
    input_sdf: Path,
    output_dir: Path,
    logger_obj: logging.Logger | None = None,
) -> list[dict[str, Any]]:
    """Write durable empty reward artifacts when no molecule passes PB."""
    output_dir.mkdir(parents=True, exist_ok=True)
    reward_cfg = dict(config)
    reward_cfg["diffgui_reward_only"] = {
        "input_sdf": str(input_sdf),
        "output_dir": str(output_dir),
        "mol_id_prefix": "diffgui_dock",
        "input_records": 0,
        "skip_reason": "no_posebusters_valid_molecules",
        "scored_sdf": "docking_scored.sdf",
        "rewards_csv": "docking_rewards.csv",
        "summary_json": "docking_reward_summary.json",
        "manifest_json": "docking_reward_manifest.json",
    }
    (output_dir / "reward_only_config.yaml").write_text(
        yaml.safe_dump(reward_cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    _write_empty_reward_csv(output_dir / "docking_rewards.csv")
    summary = {
        "stage": "diffgui_reward_only_stage1",
        "status": "skipped",
        "skip_reason": "no_posebusters_valid_molecules",
        "input_sdf": str(input_sdf),
        "input_records": 0,
        "readable_records": 0,
        "rdkit_valid": 0,
        "rdkit_invalid": 0,
        "rdkit_valid_ratio": 0.0,
        "sdf_written": 0,
        "scored_sdf": str(output_dir / "docking_scored.sdf"),
        "rewards_csv": str(output_dir / "docking_rewards.csv"),
    }
    (output_dir / "docking_reward_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "docking_reward_manifest.json").write_text(
        json.dumps(
            {
                "stage": "diffgui_reward_only_stage1",
                "status": "skipped",
                "skip_reason": "no_posebusters_valid_molecules",
                "input_sdf": str(input_sdf),
                "outputs": {
                    "scored_sdf": "",
                    "rewards_csv": str(output_dir / "docking_rewards.csv"),
                    "summary_json": str(output_dir / "docking_reward_summary.json"),
                },
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    message = "reward-only skipped: no PoseBusters-valid molecules\n"
    (output_dir / "reward_only_subprocess.log").write_text(message, encoding="utf-8")
    (logger_obj or logging.getLogger(__name__)).info(
        "reward-only skipped input=%s reason=no_posebusters_valid_molecules",
        input_sdf,
    )
    return []


def validate_pre_reward_posebusters_gate(
    *,
    reward_rows: list[dict[str, Any]],
    posebusters_rows: list[dict[str, Any]],
    enabled: bool,
) -> dict[str, int]:
    """Verify that only PB-valid graph ids entered reward calculation."""
    if not enabled:
        return {
            "reward_input_records": int(len(reward_rows)),
            "reward_skipped_by_posebusters": 0,
            "reward_input_matches_pb_valid": 1,
            "pb_valid_without_reward_count": 0,
            "reward_without_pb_valid_count": 0,
        }

    pb_valid_graphs = {
        _row_int(row, "source_graph_index", default=-1)
        for row in posebusters_rows
        if _row_int(row, "pb_valid") == 1
    }
    pb_valid_graphs.discard(-1)
    reward_graph_list = [_row_int(row, "source_graph_index", default=-1) for row in reward_rows]
    if any(graph_index < 0 for graph_index in reward_graph_list):
        raise RuntimeError("Reward row missing source_graph_index after PoseBusters pre-filter")
    if len(set(reward_graph_list)) != len(reward_graph_list):
        raise RuntimeError("Duplicate source_graph_index values in PB-filtered reward rows")

    reward_graphs = set(reward_graph_list)
    missing = pb_valid_graphs - reward_graphs
    unexpected = reward_graphs - pb_valid_graphs
    if missing or unexpected:
        raise RuntimeError(
            "PoseBusters/reward graph mismatch "
            f"missing_rewards={sorted(missing)} unexpected_rewards={sorted(unexpected)}"
        )
    pb_evaluated = sum(_row_int(row, "pb_evaluated") for row in posebusters_rows)
    return {
        "reward_input_records": int(len(reward_rows)),
        "reward_skipped_by_posebusters": int(max(0, pb_evaluated - len(pb_valid_graphs))),
        "reward_input_matches_pb_valid": 1,
        "pb_valid_without_reward_count": 0,
        "reward_without_pb_valid_count": 0,
    }


def merge_molecule_posebusters_fields(
    molecule_rows: list[dict[str, Any]],
    posebusters_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach PB fields without conflating sanitize validity and PB validity."""
    fields_by_graph: dict[int, dict[str, Any]] = {}
    for pb_row in posebusters_rows:
        graph_index = _row_int(pb_row, "source_graph_index", default=-1)
        if graph_index < 0:
            continue
        fields: dict[str, Any] = {
            "posebusters_source_index": pb_row.get("source_index", ""),
            "posebusters_sanitize_status": pb_row.get("sanitize_status", ""),
            "sanitize_valid": _row_int(pb_row, "sanitize_valid"),
            "rdkit_valid": _row_int(pb_row, "sanitize_valid"),
            "pb_evaluated": _row_int(pb_row, "pb_evaluated"),
            "pb_valid": _row_int(pb_row, "pb_valid"),
            "reward_eligible": _row_int(pb_row, "pb_valid"),
            "final_legal": _row_int(pb_row, "pb_valid"),
            "posebusters_empty": _row_int(pb_row, "posebusters_empty"),
            "posebusters_error": pb_row.get("posebusters_error", ""),
        }
        fields.update({key: value for key, value in pb_row.items() if str(key).startswith("pb_check_")})
        fields_by_graph[graph_index] = fields

    merged: list[dict[str, Any]] = []
    for row in molecule_rows:
        out = dict(row)
        graph_index = _row_int(out, "local_mol_index", default=-1)
        fields = fields_by_graph.get(
            graph_index,
            {
                "posebusters_source_index": "",
                "posebusters_sanitize_status": "not_evaluated",
                "sanitize_valid": 0,
                "rdkit_valid": 0,
                "pb_evaluated": 0,
                "pb_valid": 0,
                "reward_eligible": 0,
                "final_legal": 0,
                "posebusters_empty": 0,
                "posebusters_error": "not_in_reconstructed_sdf",
            },
        )
        out.update(fields)
        merged.append(out)
    return merged


def _write_empty_reward_csv(path: Path) -> None:
    """Write an empty reward CSV with a stable header."""
    fieldnames = [
        "mol_id",
        "source_index",
        "source_graph_index",
        "original_name",
        "rdkit_valid",
        "sanitize_status",
        "smiles",
        "reward_total",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()


def _row_int(row: dict[str, Any], key: str, default: int = 0) -> int:
    """Read an int-like row value."""
    value = row.get(key, default)
    if value is None or value == "":
        value = default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)
