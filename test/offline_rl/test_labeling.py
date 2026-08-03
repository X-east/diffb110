"""Tests for provenance-bound offline trajectory molecule labeling."""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Geometry import Point3D

from ddpo.offline_rl.index import (
    EPISODE_INDEX_SCHEMA_VERSION,
    EpisodeIndex,
    EpisodeIndexRow,
)
from ddpo.offline_rl.labeling import (
    DockingLabelFacts,
    LabelPolicyConfig,
    PoseBustersLabelFacts,
    label_trajectory_shard,
    merge_reward_sidecar_shards,
    reconstruction_facts_from_row,
)
from ddpo.offline_rl.labeling_backends import _load_unidock_pose
from ddpo.offline_rl.reward_sidecar import write_reward_sidecar
from ddpo.rewards.interfaces import DockingResult


CHECKPOINT_SHA256 = "c" * 64
SHARD_SHA256 = "d" * 64


def _index_row(
    sample_id: str,
    episode_index: int,
    *,
    rdkit_valid: bool,
    failure_type: str,
    shard_sha256: str = SHARD_SHA256,
) -> EpisodeIndexRow:
    """Build one minimal valid episode-index row."""
    return EpisodeIndexRow(
        index_schema_version=EPISODE_INDEX_SCHEMA_VERSION,
        source_run_id="collector-test",
        manifest_snapshot_sha256="a" * 64,
        sample_id=sample_id,
        shard_path="trajectories/rank_00/shard_0000.pt",
        source_shard_sha256=shard_sha256,
        episode_index=episode_index,
        rank=0,
        shard_index=0,
        seed=1,
        num_nodes=3,
        num_diffusion_steps=4,
        max_n_nodes=4,
        rdkit_valid=rdkit_valid,
        failure_type=failure_type,
        smiles="CCO" if rdkit_valid else None,
        source_checkpoint_sha256=CHECKPOINT_SHA256,
    )


def _molecule(sample_id: str, smiles: str = "CCO") -> Chem.Mol:
    """Build a sanitized molecule with deterministic 3D coordinates."""
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    conformer = Chem.Conformer(molecule.GetNumAtoms())
    for atom_index in range(molecule.GetNumAtoms()):
        conformer.SetAtomPosition(
            atom_index,
            Point3D(float(atom_index) * 1.4, 0.0, 0.0),
        )
    molecule.AddConformer(conformer)
    molecule.SetProp("sample_id", sample_id)
    return molecule


def _write_inputs(
    tmp_path: Path,
    reconstruction_rows: list[dict[str, object]],
    molecules: list[Chem.Mol],
) -> tuple[Path, Path]:
    """Write collector-shaped reconstruction CSV and SDF fixtures."""
    csv_path = tmp_path / "reconstruction.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("sample_id", "rdkit_valid", "failure_type"),
        )
        writer.writeheader()
        writer.writerows(reconstruction_rows)
    sdf_path = tmp_path / "molecules.sdf"
    writer_sdf = Chem.SDWriter(str(sdf_path))
    for molecule in molecules:
        writer_sdf.write(molecule)
    writer_sdf.close()
    return csv_path, sdf_path


def test_load_unidock_pose_reads_retained_sdf_coordinates(tmp_path: Path) -> None:
    """The labeling adapter must recover UniDock's canonical ``_out.sdf`` pose."""
    ligand_path = tmp_path / "ligand_0000.sdf"
    pose_dir = tmp_path / "poses"
    pose_dir.mkdir()
    pose = _molecule("pose")
    conformer = pose.GetConformer()
    conformer.SetAtomPosition(0, Point3D(207.0, 212.0, 173.0))
    writer = Chem.SDWriter(str(pose_dir / "ligand_0000_out.sdf"))
    writer.write(pose)
    writer.close()

    loaded = _load_unidock_pose(pose_dir, ligand_path)

    assert loaded is not None
    position = loaded.GetConformer().GetAtomPosition(0)
    assert (position.x, position.y, position.z) == pytest.approx((207.0, 212.0, 173.0))


def _formal_config() -> LabelPolicyConfig:
    """Return permissive thresholds that isolate conjunction mechanics."""
    return LabelPolicyConfig(
        reward_version="test_formal_reward_v1",
        formal_labels=True,
        posebusters_enabled=True,
        docking_enabled=True,
        failure_penalty=-3.0,
        valid_base_reward=0.2,
        qed_weight=0.3,
        sa_weight=0.4,
        docking_weight=0.5,
        pb_valid_weight=0.6,
        actual_good_docking_max=-1.0,
        actual_good_ligand_efficiency_min=0.01,
        actual_good_qed_min=0.0,
        actual_good_sa_max=10.0,
        actual_good_logp_min=-10.0,
        actual_good_logp_max=10.0,
        actual_good_tpsa_min=0.0,
        actual_good_tpsa_max=200.0,
        actual_good_max_fused_rings=20,
        actual_good_max_complex_rings=20,
        actual_good_max_brenk=20,
    )


@pytest.mark.parametrize(
    ("rdkit_valid", "failure_type", "expected"),
    [
        ("1", "success", (True, True, True)),
        ("0", "fragmented_molecule", (True, True, False)),
        ("0", "sanitize_failed", (True, False, False)),
        ("0", "openbabel_reconstruct_failed", (False, False, False)),
    ],
)
def test_reconstruction_facts_follow_canonical_collector_semantics(
    rdkit_valid: str,
    failure_type: str,
    expected: tuple[bool, bool, bool],
) -> None:
    """Collector failure types must recover distinct reconstruction stages."""
    facts = reconstruction_facts_from_row(
        {"rdkit_valid": rdkit_valid, "failure_type": failure_type}
    )

    assert (
        facts.reconstruction_success,
        facts.sanitize_success,
        facts.rdkit_valid,
    ) == expected


def test_reconstruction_facts_reject_unknown_or_inconsistent_rows() -> None:
    """Ambiguous collector rows must not be converted into guessed labels."""
    with pytest.raises(ValueError, match="unsupported collector failure_type"):
        reconstruction_facts_from_row(
            {"rdkit_valid": "0", "failure_type": "unknown_failure"}
        )
    with pytest.raises(ValueError, match="success row"):
        reconstruction_facts_from_row(
            {"rdkit_valid": "0", "failure_type": "success"}
        )
    with pytest.raises(ValueError, match="cannot have rdkit_valid=1"):
        reconstruction_facts_from_row(
            {"rdkit_valid": "1", "failure_type": "sanitize_failed"}
        )


def test_label_shard_preserves_invalid_stage_facts(tmp_path: Path) -> None:
    """Fragmented and sanitize-failed samples must receive finite penalties."""
    rows = (
        _index_row("valid", 0, rdkit_valid=True, failure_type="success"),
        _index_row(
            "fragmented", 1, rdkit_valid=False, failure_type="fragmented_molecule"
        ),
        _index_row(
            "sanitize", 2, rdkit_valid=False, failure_type="sanitize_failed"
        ),
    )
    csv_path, sdf_path = _write_inputs(
        tmp_path,
        [
            {"sample_id": "valid", "rdkit_valid": 1, "failure_type": "success"},
            {
                "sample_id": "fragmented",
                "rdkit_valid": 0,
                "failure_type": "fragmented_molecule",
            },
            {
                "sample_id": "sanitize",
                "rdkit_valid": 0,
                "failure_type": "sanitize_failed",
            },
        ],
        [_molecule("valid")],
    )

    labels = label_trajectory_shard(
        rows,
        reconstruction_csv=csv_path,
        molecules_sdf=sdf_path,
        config=LabelPolicyConfig(reward_version="descriptor_test_v1"),
    )
    by_id = {label.sample_id: label for label in labels}

    assert by_id["fragmented"].reconstruction_success
    assert by_id["fragmented"].sanitize_success
    assert not by_id["fragmented"].rdkit_valid
    assert by_id["fragmented"].reward_total_train == -2.0
    assert by_id["sanitize"].reconstruction_success
    assert not by_id["sanitize"].sanitize_success
    assert by_id["valid"].qed is not None
    assert by_id["valid"].reward_total_train == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("pb_facts", "docking_facts", "expected"),
    [
        (
            PoseBustersLabelFacts(evaluated=True, valid=True),
            DockingLabelFacts(
                input_success=True,
                result=DockingResult.succeeded("unidock", -8.0),
                pose_molecule=_molecule("pose-success"),
            ),
            (True, True, True, True, "success"),
        ),
        (
            PoseBustersLabelFacts(
                evaluated=False,
                valid=False,
                failure_reason="posebusters_error:timeout",
            ),
            DockingLabelFacts(
                input_success=True,
                result=DockingResult.succeeded("unidock", -8.0),
                pose_molecule=_molecule("pose-error"),
            ),
            (False, False, True, False, "posebusters_error:timeout"),
        ),
        (
            PoseBustersLabelFacts(
                evaluated=True,
                valid=False,
                failure_reason="posebusters_invalid",
            ),
            DockingLabelFacts(
                input_success=True,
                result=DockingResult.succeeded("unidock", -8.0),
                pose_molecule=_molecule("pose-invalid"),
            ),
            (True, False, True, False, "posebusters_invalid"),
        ),
        (
            PoseBustersLabelFacts(evaluated=True, valid=True),
            DockingLabelFacts(
                input_success=True,
                result=DockingResult.failed("unidock", "docking_backend_failed"),
            ),
            (False, False, True, False, "docking_backend_failed"),
        ),
        (
            PoseBustersLabelFacts(evaluated=True, valid=True),
            DockingLabelFacts(
                input_success=False,
                result=DockingResult.failed("unidock", "docking_input_failed"),
            ),
            (False, False, False, False, "docking_input_failed"),
        ),
    ],
)
def test_label_shard_distinguishes_pb_and_docking_boundaries(
    tmp_path: Path,
    pb_facts: PoseBustersLabelFacts,
    docking_facts: DockingLabelFacts,
    expected: tuple[bool, bool, bool, bool, str],
) -> None:
    """PB errors and docking input/backend failures must remain distinguishable."""
    row = _index_row("valid", 0, rdkit_valid=True, failure_type="success")
    csv_path, sdf_path = _write_inputs(
        tmp_path,
        [{"sample_id": "valid", "rdkit_valid": 1, "failure_type": "success"}],
        [_molecule("valid")],
    )

    label = label_trajectory_shard(
        (row,),
        reconstruction_csv=csv_path,
        molecules_sdf=sdf_path,
        config=_formal_config(),
        posebusters=lambda _: pb_facts,
        docking=lambda _: {"valid": docking_facts},
    )[0]

    assert (
        label.pb_evaluated,
        label.pb_valid,
        label.docking_input_success,
        label.actual_good,
        label.failure_reason,
    ) == expected
    if docking_facts.result.success:
        assert label.reward_total_train > -3.0
    else:
        assert label.reward_total_train == -3.0


def test_formal_reward_can_gate_pb_invalid_molecules(tmp_path: Path) -> None:
    """PB-gated formal rewards must keep invalid poses as finite hard negatives."""
    row = _index_row("valid", 0, rdkit_valid=True, failure_type="success")
    csv_path, sdf_path = _write_inputs(
        tmp_path,
        [{"sample_id": "valid", "rdkit_valid": 1, "failure_type": "success"}],
        [_molecule("valid")],
    )
    config = replace(_formal_config(), require_pb_valid_for_reward=True)

    label = label_trajectory_shard(
        (row,),
        reconstruction_csv=csv_path,
        molecules_sdf=sdf_path,
        config=config,
        posebusters=lambda _: PoseBustersLabelFacts(
            evaluated=True,
            valid=False,
            failure_reason="posebusters_invalid:bond_lengths",
        ),
        docking=lambda _: {
            "valid": DockingLabelFacts(
                input_success=True,
                result=DockingResult.succeeded("unidock", -9.0),
                pose_molecule=_molecule("pose"),
            )
        },
    )[0]

    assert label.reward_total_train == config.failure_penalty
    assert label.reward_total_raw == config.failure_penalty
    assert label.docking_raw == -9.0
    assert label.failure_reason == "posebusters_invalid:bond_lengths"


def test_label_shard_rejects_missing_inputs_and_backend_coverage(
    tmp_path: Path,
) -> None:
    """Every indexed sample and submitted molecule must have exact coverage."""
    row = _index_row("valid", 0, rdkit_valid=True, failure_type="success")
    missing_csv, sdf_path = _write_inputs(tmp_path, [], [_molecule("valid")])
    with pytest.raises(ValueError, match="reconstruction CSV coverage mismatch"):
        label_trajectory_shard(
            (row,),
            reconstruction_csv=missing_csv,
            molecules_sdf=sdf_path,
            config=LabelPolicyConfig(reward_version="coverage_test_v1"),
        )

    csv_path, sdf_path = _write_inputs(
        tmp_path,
        [{"sample_id": "valid", "rdkit_valid": 1, "failure_type": "success"}],
        [_molecule("valid")],
    )
    with pytest.raises(ValueError, match="do not exactly cover"):
        label_trajectory_shard(
            (row,),
            reconstruction_csv=csv_path,
            molecules_sdf=sdf_path,
            config=_formal_config(),
            posebusters=lambda _: PoseBustersLabelFacts(True, True),
            docking=lambda _: {},
        )


def test_merge_sidecars_requires_exact_coverage_and_provenance(
    tmp_path: Path,
) -> None:
    """Final merge must fail on missing samples or changed source hashes."""
    rows = (
        _index_row("sample-0", 0, rdkit_valid=True, failure_type="success"),
        _index_row("sample-1", 1, rdkit_valid=True, failure_type="success"),
    )
    episode_index = EpisodeIndex(rows)
    csv_path, sdf_path = _write_inputs(
        tmp_path,
        [
            {"sample_id": row.sample_id, "rdkit_valid": 1, "failure_type": "success"}
            for row in rows
        ],
        [_molecule(row.sample_id) for row in rows],
    )
    labels = label_trajectory_shard(
        rows,
        reconstruction_csv=csv_path,
        molecules_sdf=sdf_path,
        config=LabelPolicyConfig(reward_version="merge_test_v1"),
    )
    first_path = write_reward_sidecar(tmp_path / "first.csv", labels[:1])
    second_path = write_reward_sidecar(tmp_path / "second.csv", labels[1:])

    merged = merge_reward_sidecar_shards(
        (first_path, second_path),
        episode_index=episode_index,
        output_path=tmp_path / "merged.csv",
    )
    assert {row.sample_id for row in merged.rows} == {"sample-0", "sample-1"}

    with pytest.raises(ValueError, match="coverage does not match"):
        merge_reward_sidecar_shards(
            (first_path,),
            episode_index=episode_index,
            output_path=tmp_path / "missing.csv",
        )

    wrong = replace(labels[1], source_shard_sha256="e" * 64)
    wrong_path = write_reward_sidecar(tmp_path / "wrong.csv", (wrong,))
    with pytest.raises(ValueError, match="provenance mismatch"):
        merge_reward_sidecar_shards(
            (first_path, wrong_path),
            episode_index=episode_index,
            output_path=tmp_path / "wrong-merged.csv",
        )
