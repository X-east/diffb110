"""Tests for versioned reward sidecars and immutable shard joins."""

from __future__ import annotations

import csv
import hashlib
import os
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from ddpo.offline_rl.dataset import load_trajectory_shard
from ddpo.offline_rl.reward_sidecar import (
    REWARD_SIDECAR_SCHEMA_VERSION,
    EpisodeRewardFacts,
    RewardSidecar,
    RewardSidecarError,
    load_reward_sidecar,
    write_reward_sidecar,
)


CHECKPOINT_SHA256 = "c" * 64
REAL_SHARD_ENV = os.environ.get("DIFFSBDD_TEST_TRAJECTORY_SHARD")
REAL_SHARD_PATH = Path(REAL_SHARD_ENV) if REAL_SHARD_ENV else None


def _make_payload() -> dict[str, object]:
    """Build a two-episode trajectory fixture."""
    batch_size = 2
    diffusion_steps = 3
    num_actions = diffusion_steps + 1
    max_nodes = 4
    feature_dim = 5
    latents = torch.randn(
        batch_size,
        num_actions + 1,
        max_nodes,
        feature_dim,
    )
    nodesxsample = torch.tensor([2, 4], dtype=torch.long)
    latents[0, :, 2:] = 0
    x = torch.randn(batch_size, max_nodes, 3)
    categorical = torch.ones(batch_size, max_nodes, 2)
    x[0, 2:] = 0
    categorical[0, 2:] = 0
    return {
        "format": "diffsbdd_frozen_policy_trajectory_v1",
        "latents": latents,
        "logps": torch.zeros(batch_size, num_actions),
        "timesteps": torch.tensor([2, 1, 0, 0]).repeat(batch_size, 1),
        "nodesxsample": nodesxsample,
        "x": x,
        "categorical": categorical,
        "sample_ids": ["sample-0", "sample-1"],
        "metadata": {
            "rank": 0,
            "shard_index": 0,
            "seed": 1,
            "num_timesteps": diffusion_steps,
            "max_n_nodes": max_nodes,
            "checkpoint_sha256": CHECKPOINT_SHA256,
        },
        "reconstruction_rows": [
            {
                "sample_id": "sample-0",
                "rdkit_valid": 1,
                "failure_type": "success",
                "smiles": "CC",
            },
            {
                "sample_id": "sample-1",
                "rdkit_valid": 0,
                "failure_type": "sanitize_failed",
                "smiles": "",
            },
        ],
    }


def _save_shard(tmp_path: Path) -> tuple[Path, str]:
    """Save a trajectory fixture and return its digest."""
    path = tmp_path / "shard_0000.pt"
    torch.save(_make_payload(), path)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _reward_row(
    sample_id: str,
    shard_sha256: str,
    *,
    reward: float,
    valid: bool,
) -> EpisodeRewardFacts:
    """Build one valid reward row."""
    return EpisodeRewardFacts(
        sample_id=sample_id,
        source_trajectory_format="diffsbdd_frozen_policy_trajectory_v1",
        source_checkpoint_sha256=CHECKPOINT_SHA256,
        source_shard_sha256=shard_sha256,
        reward_schema_version=REWARD_SIDECAR_SCHEMA_VERSION,
        reward_version="reward_v1_test",
        reconstruction_success=valid,
        sanitize_success=valid,
        rdkit_valid=valid,
        pb_evaluated=valid,
        pb_valid=valid,
        docking_input_success=valid,
        docking_success=valid,
        actual_good=valid,
        reward_total_train=reward,
        failure_reason="" if valid else "sanitize_failed",
        docking_raw=-8.0 if valid else None,
        ligand_efficiency=0.5 if valid else None,
        qed=0.7 if valid else None,
        sa_raw=2.5 if valid else None,
        logp=2.0 if valid else None,
        tpsa=50.0 if valid else None,
        hba=2 if valid else None,
        hbd=1 if valid else None,
        fragment_count=1 if valid else None,
        carbon_fraction=0.75 if valid else None,
        hetero_atom_count=2 if valid else None,
        ring_count=1 if valid else None,
        aromatic_ring_count=1 if valid else None,
        three_member_ring_count=0 if valid else None,
        fused_ring_count=0 if valid else None,
        complex_ring_count=0 if valid else None,
        brenk_count=0 if valid else None,
        chembl_prior=0.6 if valid else None,
        reward_total_raw=1.2 if valid else None,
    )


def test_reward_sidecar_csv_round_trip_preserves_nullable_fields(
    tmp_path: Path,
) -> None:
    """Canonical CSV I/O should preserve types and missing raw measurements."""
    _, digest = _save_shard(tmp_path)
    rows = (
        _reward_row("sample-0", digest, reward=1.25, valid=True),
        _reward_row("sample-1", digest, reward=-1.0, valid=False),
    )
    path = write_reward_sidecar(tmp_path / "reward_sidecar.csv", rows)
    loaded = load_reward_sidecar(
        path,
        expected_reward_version="reward_v1_test",
    )

    assert loaded.rows == rows
    assert loaded.rows[1].docking_raw is None
    assert loaded.reward_version == "reward_v1_test"


def test_join_reorders_by_sample_id_and_keeps_source_bytes_unchanged(
    tmp_path: Path,
) -> None:
    """Lazy join should use sample IDs, verify provenance, and never rewrite PT."""
    shard_path, digest = _save_shard(tmp_path)
    bytes_before = shard_path.read_bytes()
    mtime_before = shard_path.stat().st_mtime_ns
    shard = load_trajectory_shard(shard_path)
    sidecar = RewardSidecar(
        (
            _reward_row("sample-1", digest, reward=-1.0, valid=False),
            _reward_row("sample-0", digest, reward=2.0, valid=True),
        )
    )

    joined = sidecar.join_shard(shard)
    ordinary = joined.select_transition(0, 0)
    terminal = joined.select_transition(0, shard.final_decision_index)

    assert tuple(row.sample_id for row in joined.reward_rows) == shard.sample_ids
    assert joined.episode_rewards.tolist() == pytest.approx([2.0, -1.0])
    assert ordinary.reward is not None and ordinary.reward.item() == 0.0
    assert ordinary.episode_return is not None
    assert ordinary.episode_return.item() == 2.0
    assert terminal.reward is not None and terminal.reward.item() == 2.0
    assert terminal.episode_return is not None
    assert terminal.episode_return.item() == 2.0
    assert shard_path.read_bytes() == bytes_before
    assert shard_path.stat().st_mtime_ns == mtime_before


def test_join_rejects_missing_reward_and_provenance_mismatch(
    tmp_path: Path,
) -> None:
    """Missing rows and wrong shard/checkpoint digests must fail closed."""
    shard_path, digest = _save_shard(tmp_path)
    shard = load_trajectory_shard(shard_path)

    missing = RewardSidecar(
        (_reward_row("sample-0", digest, reward=1.0, valid=True),)
    )
    with pytest.raises(RewardSidecarError, match="missing sample_id"):
        missing.join_shard(shard)

    wrong_shard_rows = (
        replace(
            _reward_row("sample-0", digest, reward=1.0, valid=True),
            source_shard_sha256="a" * 64,
        ),
        replace(
            _reward_row("sample-1", digest, reward=-1.0, valid=False),
            source_shard_sha256="a" * 64,
        ),
    )
    with pytest.raises(RewardSidecarError, match="source shard SHA"):
        RewardSidecar(wrong_shard_rows).join_shard(shard)

    wrong_checkpoint_rows = tuple(
        replace(row, source_checkpoint_sha256="b" * 64)
        for row in (
            _reward_row("sample-0", digest, reward=1.0, valid=True),
            _reward_row("sample-1", digest, reward=-1.0, valid=False),
        )
    )
    with pytest.raises(RewardSidecarError, match="checkpoint SHA"):
        RewardSidecar(wrong_checkpoint_rows).join_shard(shard)


def test_reward_facts_reject_nonfinite_and_inconsistent_success_flags(
    tmp_path: Path,
) -> None:
    """Training rewards and success implications must be explicit and valid."""
    _, digest = _save_shard(tmp_path)
    valid = _reward_row("sample-0", digest, reward=1.0, valid=True)

    with pytest.raises(RewardSidecarError, match="reward_total_train"):
        replace(valid, reward_total_train=float("nan"))
    with pytest.raises(RewardSidecarError, match="reward_total_train"):
        replace(valid, reward_total_train="1.0")  # type: ignore[arg-type]
    with pytest.raises(RewardSidecarError, match="rdkit_valid must be bool"):
        replace(valid, rdkit_valid=1)  # type: ignore[arg-type]
    with pytest.raises(RewardSidecarError, match="qed"):
        replace(valid, qed=float("inf"))
    with pytest.raises(RewardSidecarError, match="hba"):
        replace(valid, hba=1.5)  # type: ignore[arg-type]
    with pytest.raises(RewardSidecarError, match="carbon_fraction"):
        replace(valid, carbon_fraction=1.1)
    with pytest.raises(RewardSidecarError, match="rdkit_valid requires"):
        replace(valid, sanitize_success=False)
    with pytest.raises(RewardSidecarError, match="docking_success requires"):
        replace(valid, actual_good=False, docking_input_success=False)
    with pytest.raises(RewardSidecarError, match="actual_good requires"):
        replace(valid, pb_valid=False)


def test_sidecar_rejects_duplicates_mixed_versions_and_empty() -> None:
    """A sidecar must be non-empty, unique, and single-version."""
    digest = "d" * 64
    first = _reward_row("sample-0", digest, reward=1.0, valid=True)

    with pytest.raises(RewardSidecarError, match="cannot be empty"):
        RewardSidecar(())
    with pytest.raises(RewardSidecarError, match="duplicate"):
        RewardSidecar((first, first))
    with pytest.raises(RewardSidecarError, match="one reward_version"):
        RewardSidecar(
            (
                first,
                replace(
                    _reward_row("sample-1", digest, reward=-1.0, valid=False),
                    reward_version="reward_v2_test",
                ),
            )
        )


def test_csv_loader_rejects_missing_column_invalid_bool_and_version(
    tmp_path: Path,
) -> None:
    """Malformed CSV schema and values should fail before any trajectory join."""
    _, digest = _save_shard(tmp_path)
    canonical_path = write_reward_sidecar(
        tmp_path / "canonical.csv",
        (_reward_row("sample-0", digest, reward=1.0, valid=True),),
    )
    with canonical_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        field_names = list(reader.fieldnames or ())
        raw_rows = list(reader)

    missing_path = tmp_path / "missing.csv"
    missing_fields = [
        name for name in field_names if name != "reward_total_train"
    ]
    with missing_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=missing_fields)
        writer.writeheader()
        writer.writerow(
            {name: raw_rows[0][name] for name in missing_fields}
        )
    with pytest.raises(RewardSidecarError, match="missing required columns"):
        load_reward_sidecar(missing_path)

    raw_rows[0]["rdkit_valid"] = "maybe"
    invalid_bool_path = tmp_path / "invalid_bool.csv"
    with invalid_bool_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(raw_rows)
    with pytest.raises(RewardSidecarError, match="0/1/true/false"):
        load_reward_sidecar(invalid_bool_path)

    with pytest.raises(RewardSidecarError, match="reward_version mismatch"):
        load_reward_sidecar(
            canonical_path,
            expected_reward_version="different_reward",
        )


@pytest.mark.skipif(
    REAL_SHARD_PATH is None or not REAL_SHARD_PATH.is_file(),
    reason="set DIFFSBDD_TEST_TRAJECTORY_SHARD for the real-shard join test",
)
def test_real_shard_accepts_complete_provenance_bound_sidecar() -> None:
    """A formal shard should join all rows without changing trajectory bytes."""
    assert REAL_SHARD_PATH is not None
    shard = load_trajectory_shard(REAL_SHARD_PATH)
    digest = shard.source_sha256()
    rows = tuple(
        EpisodeRewardFacts(
            sample_id=facts.sample_id,
            source_trajectory_format="diffsbdd_frozen_policy_trajectory_v1",
            source_checkpoint_sha256=facts.checkpoint_sha256,
            source_shard_sha256=digest,
            reward_schema_version=REWARD_SIDECAR_SCHEMA_VERSION,
            reward_version="reward_contract_probe_v1",
            reconstruction_success=facts.rdkit_valid,
            sanitize_success=facts.rdkit_valid,
            rdkit_valid=facts.rdkit_valid,
            pb_evaluated=False,
            pb_valid=False,
            docking_input_success=False,
            docking_success=False,
            actual_good=False,
            reward_total_train=1.0 if facts.rdkit_valid else -1.0,
            failure_reason=facts.failure_type,
        )
        for facts in shard.all_episode_facts()
    )

    joined = RewardSidecar(rows).join_shard(shard)

    assert len(joined.reward_rows) == shard.batch_size
    assert joined.episode_rewards.shape == (shard.batch_size,)
    assert bool(torch.isfinite(joined.episode_rewards).all())
