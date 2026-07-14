"""Tests for the DiffGui PoseBusters-before-reward contract."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from ddpo.diffgui.runtime.pre_reward_gate import (
    merge_molecule_posebusters_fields,
    run_posebusters_then_rewards,
    validate_pre_reward_posebusters_gate,
)


def test_posebusters_runs_before_reward_and_supplies_reward_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reward scoring must receive only the PB-valid SDF."""
    generated_sdf = tmp_path / "generated.sdf"
    generated_sdf.write_text("generated\n", encoding="utf-8")
    pb_valid_sdf = tmp_path / "pb_valid.sdf"
    calls: list[tuple[str, Path]] = []

    def fake_posebusters_gate(**kwargs: object) -> tuple[dict[str, int], list[dict[str, int]], Path]:
        input_sdf = Path(str(kwargs["input_sdf"]))
        calls.append(("posebusters", input_sdf))
        assert input_sdf == generated_sdf
        return (
            {"posebusters_enabled": 1, "posebusters_pb_valid": 1},
            [{"source_graph_index": 5, "pb_evaluated": 1, "pb_valid": 1}],
            pb_valid_sdf,
        )

    def fake_reward_score(**kwargs: object) -> list[dict[str, int]]:
        input_sdf = Path(str(kwargs["input_sdf"]))
        calls.append(("reward", input_sdf))
        assert calls[0] == ("posebusters", generated_sdf)
        assert input_sdf == pb_valid_sdf
        return [{"source_graph_index": 5, "reward_total": 1}]

    result = run_posebusters_then_rewards(
        config={},
        generated_sdf=generated_sdf,
        output_dir=tmp_path,
        total_attempts=8,
        reward_batch_size=8,
        posebusters_runner=fake_posebusters_gate,
        reward_runner=fake_reward_score,
    )

    assert calls == [("posebusters", generated_sdf), ("reward", pb_valid_sdf)]
    assert result.reward_rows == [{"source_graph_index": 5, "reward_total": 1}]
    assert result.audit["reward_input_records"] == 1
    assert result.audit["reward_input_matches_pb_valid"] == 1
    assert result.timing["posebusters_seconds"] >= 0.0
    assert result.timing["reward_seconds"] >= 0.0


def test_zero_pb_valid_skips_reward_subprocess_and_writes_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rank with zero PB-valid molecules should remain a valid epoch state."""
    generated_sdf = tmp_path / "generated.sdf"
    generated_sdf.write_text("generated\n", encoding="utf-8")
    pb_valid_sdf = tmp_path / "pb_valid.sdf"
    pb_valid_sdf.write_text("", encoding="utf-8")

    def fake_posebusters_gate(**_: object) -> tuple[dict[str, int], list[dict[str, int]], Path]:
        return (
            {"posebusters_enabled": 1, "posebusters_pb_valid": 0},
            [{"source_graph_index": 2, "pb_evaluated": 1, "pb_valid": 0}],
            pb_valid_sdf,
        )

    def fail_reward_score(**_: object) -> list[dict[str, int]]:
        raise AssertionError("reward subprocess must not run for an empty PB-valid set")

    result = run_posebusters_then_rewards(
        config={},
        generated_sdf=generated_sdf,
        output_dir=tmp_path,
        total_attempts=8,
        reward_batch_size=8,
        posebusters_runner=fake_posebusters_gate,
        reward_runner=fail_reward_score,
    )

    assert result.reward_rows == []
    assert result.audit["reward_input_records"] == 0
    assert result.audit["reward_skipped_by_posebusters"] == 1
    with (tmp_path / "docking_rewards.csv").open(newline="", encoding="utf-8") as handle:
        assert list(csv.DictReader(handle)) == []
    summary = json.loads((tmp_path / "docking_reward_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "skipped"
    assert summary["skip_reason"] == "no_posebusters_valid_molecules"


def test_reward_rows_outside_pb_valid_set_are_rejected() -> None:
    """A reward row for a PB-invalid graph must fail the invariant check."""
    with pytest.raises(RuntimeError, match="unexpected_rewards"):
        validate_pre_reward_posebusters_gate(
            reward_rows=[{"source_graph_index": 3}],
            posebusters_rows=[
                {"source_graph_index": 2, "pb_evaluated": 1, "pb_valid": 1},
                {"source_graph_index": 3, "pb_evaluated": 1, "pb_valid": 0},
            ],
            enabled=True,
        )


def test_pb_sanitize_valid_is_not_conflated_with_pb_valid() -> None:
    """A PB-invalid molecule can remain RDKit-valid while skipping reward."""
    merged = merge_molecule_posebusters_fields(
        [
            {
                "local_mol_index": 4,
                "rdkit_valid": 0,
                "reward_found": 0,
                "docking_success": 0,
            }
        ],
        [
            {
                "source_index": 0,
                "source_graph_index": 4,
                "sanitize_status": "ok",
                "sanitize_valid": 1,
                "pb_evaluated": 1,
                "pb_valid": 0,
            }
        ],
    )

    assert merged[0]["rdkit_valid"] == 1
    assert merged[0]["sanitize_valid"] == 1
    assert merged[0]["pb_valid"] == 0
    assert merged[0]["reward_eligible"] == 0
    assert merged[0]["reward_found"] == 0
