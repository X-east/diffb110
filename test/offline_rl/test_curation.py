"""Tests for drift-focused positive and negative episode curation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from ddpo.offline_rl.curation import (
    CURATION_MANIFEST_SCHEMA_VERSION,
    CurationConfig,
    CurationError,
    build_curation_manifest,
    load_curation_manifest,
    write_curation_manifest,
)
from ddpo.offline_rl.index import (
    EPISODE_INDEX_SCHEMA_VERSION,
    EpisodeIndex,
    EpisodeIndexRow,
)
from ddpo.offline_rl.reward_sidecar import (
    REWARD_SIDECAR_SCHEMA_VERSION,
    EpisodeRewardFacts,
    RewardSidecar,
)


CHECKPOINT_SHA256 = "c" * 64
SIDECAR_SHA256 = "d" * 64


def _index(sample_ids: tuple[str, ...]) -> EpisodeIndex:
    """Build one exact fixture index with parseable valid SMILES."""
    rows = []
    for position, sample_id in enumerate(sample_ids):
        valid = sample_id != "invalid"
        rows.append(
            EpisodeIndexRow(
                index_schema_version=EPISODE_INDEX_SCHEMA_VERSION,
                source_run_id="fixture-run",
                manifest_snapshot_sha256="a" * 64,
                sample_id=sample_id,
                shard_path="shards/rank_00/shard_0000.pt",
                source_shard_sha256="b" * 64,
                episode_index=position,
                rank=0,
                shard_index=0,
                seed=1,
                num_nodes=6,
                num_diffusion_steps=3,
                max_n_nodes=10,
                rdkit_valid=valid,
                failure_type="success" if valid else "sanitize_failed",
                smiles="c1ccccc1N" if valid else None,
                source_checkpoint_sha256=CHECKPOINT_SHA256,
            )
        )
    return EpisodeIndex(rows)


def _row(sample_id: str, *, actual_good: bool = False) -> EpisodeRewardFacts:
    """Build one PB-valid, target-good, property-good reward row."""
    return EpisodeRewardFacts(
        sample_id=sample_id,
        source_trajectory_format="diffsbdd_frozen_policy_trajectory_v1",
        source_checkpoint_sha256=CHECKPOINT_SHA256,
        source_shard_sha256="b" * 64,
        reward_schema_version=REWARD_SIDECAR_SCHEMA_VERSION,
        reward_version="fixture_reward_v1",
        reconstruction_success=True,
        sanitize_success=True,
        rdkit_valid=True,
        pb_evaluated=True,
        pb_valid=True,
        docking_input_success=True,
        docking_success=True,
        actual_good=actual_good,
        reward_total_train=0.5,
        failure_reason="success",
        docking_raw=-9.0,
        ligand_efficiency=0.3,
        qed=0.6,
        sa_raw=3.0,
        logp=2.0,
        tpsa=50.0,
        hba=1,
        hbd=1,
        fragment_count=1,
        carbon_fraction=0.8,
        hetero_atom_count=1,
        ring_count=1,
        aromatic_ring_count=1,
        three_member_ring_count=0,
        fused_ring_count=0,
        complex_ring_count=0,
        brenk_count=0,
    )


def _fixtures() -> tuple[EpisodeIndex, RewardSidecar]:
    """Build one episode for every registered disjoint quality class."""
    sample_ids = (
        "positive",
        "invalid",
        "target-negative",
        "property-negative",
        "technical-failure",
    )
    index = _index(sample_ids)
    invalid = replace(
        _row("invalid"),
        reconstruction_success=True,
        sanitize_success=False,
        rdkit_valid=False,
        pb_evaluated=False,
        pb_valid=False,
        docking_input_success=False,
        docking_success=False,
        reward_total_train=-1.0,
        failure_reason="sanitize_failed",
        docking_raw=None,
        ligand_efficiency=None,
        qed=None,
        sa_raw=None,
        logp=None,
        tpsa=None,
        hba=None,
        hbd=None,
        fragment_count=None,
        carbon_fraction=None,
        hetero_atom_count=None,
        ring_count=None,
        aromatic_ring_count=None,
        three_member_ring_count=None,
        fused_ring_count=None,
        complex_ring_count=None,
        brenk_count=None,
    )
    technical = replace(
        _row("technical-failure"),
        pb_evaluated=False,
        pb_valid=False,
        docking_input_success=False,
        docking_success=False,
        reward_total_train=-1.0,
        failure_reason="docking_backend_failed",
        docking_raw=None,
        ligand_efficiency=None,
    )
    sidecar = RewardSidecar(
        (
            _row("positive", actual_good=True),
            invalid,
            replace(_row("target-negative"), docking_raw=-6.0),
            replace(_row("property-negative"), qed=0.2),
            technical,
        )
    )
    return index, sidecar


def test_curation_assigns_disjoint_training_roles_and_round_trips(
    tmp_path: Path,
) -> None:
    """Positives, scientific negatives, and technical failures must not mix."""
    index, sidecar = _fixtures()
    manifest = build_curation_manifest(
        index,
        sidecar,
        reward_sidecar_sha256=SIDECAR_SHA256,
        config=CurationConfig(curation_version="fixture_curation_v1"),
    )

    assert manifest.counts() == {
        "positive": 1,
        "structural_negative": 1,
        "target_negative": 1,
        "property_negative": 1,
        "technical_failure": 1,
    }
    assert manifest.get("positive").actor_bc_keep
    assert manifest.get("invalid").structure_label == 0
    assert manifest.get("target-negative").target_label == 0
    assert manifest.get("property-negative").target_label == 1
    assert not manifest.get("technical-failure").critic_keep
    assert (
        manifest.get("positive").curation_schema_version
        == CURATION_MANIFEST_SCHEMA_VERSION
    )
    assert manifest.get("positive").scaffold_key == "c1ccccc1"
    assert manifest.sample_ids_for_classes(("positive",)) == frozenset({"positive"})

    path = write_curation_manifest(tmp_path / "curation.jsonl", manifest)
    loaded = load_curation_manifest(path)
    assert loaded.rows == manifest.rows


def test_curation_rejects_actual_good_policy_drift() -> None:
    """A label passing the same thresholds cannot silently remain non-positive."""
    index = _index(("mismatch",))
    sidecar = RewardSidecar((_row("mismatch", actual_good=False),))
    with pytest.raises(CurationError, match="policies do not match"):
        build_curation_manifest(
            index,
            sidecar,
            reward_sidecar_sha256=SIDECAR_SHA256,
            config=CurationConfig(curation_version="fixture_curation_v1"),
        )


def test_curation_rejects_unknown_or_empty_class_selection() -> None:
    """Training selection must name at least one registered class."""
    index, sidecar = _fixtures()
    manifest = build_curation_manifest(
        index,
        sidecar,
        reward_sidecar_sha256=SIDECAR_SHA256,
        config=CurationConfig(curation_version="fixture_curation_v1"),
    )
    with pytest.raises(CurationError, match="unsupported requested"):
        manifest.sample_ids_for_classes(("bad",))
    with pytest.raises(CurationError, match="at least one"):
        manifest.sample_ids_for_classes(())
