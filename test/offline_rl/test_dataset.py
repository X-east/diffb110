"""Tests for the read-only DiffSBDD offline trajectory dataset boundary."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest
import torch

from ddpo.offline_rl.dataset import (
    OfflineTrajectoryDataError,
    load_trajectory_shard,
)


REAL_SHARD_ENV = os.environ.get("DIFFSBDD_TEST_TRAJECTORY_SHARD")
REAL_SHARD_PATH = Path(REAL_SHARD_ENV) if REAL_SHARD_ENV else None


def _make_payload() -> dict[str, object]:
    """Build a small valid v1 trajectory payload."""
    batch_size = 2
    diffusion_steps = 3
    num_actions = diffusion_steps + 1
    max_nodes = 4
    atom_features = 2
    latent_features = 3 + atom_features

    latents = torch.arange(
        batch_size
        * (num_actions + 1)
        * max_nodes
        * latent_features,
        dtype=torch.float32,
    ).reshape(batch_size, num_actions + 1, max_nodes, latent_features)
    x = torch.arange(
        batch_size * max_nodes * 3,
        dtype=torch.float32,
    ).reshape(batch_size, max_nodes, 3)
    categorical = torch.ones(
        batch_size,
        max_nodes,
        atom_features,
        dtype=torch.float32,
    )
    nodesxsample = torch.tensor([2, 4], dtype=torch.long)
    latents[0, :, 2:] = 0
    x[0, 2:] = 0
    categorical[0, 2:] = 0

    return {
        "format": "diffsbdd_frozen_policy_trajectory_v1",
        "latents": latents,
        "logps": torch.arange(
            batch_size * num_actions,
            dtype=torch.float32,
        ).reshape(batch_size, num_actions),
        "timesteps": torch.tensor([2, 1, 0, 0], dtype=torch.long).repeat(
            batch_size,
            1,
        ),
        "nodesxsample": nodesxsample,
        "x": x,
        "categorical": categorical,
        "sample_ids": ["sample-0", "sample-1"],
        "metadata": {
            "rank": 2,
            "local_rank": 0,
            "shard_index": 4,
            "seed": 3,
            "num_timesteps": diffusion_steps,
            "max_n_nodes": max_nodes,
            "condition": False,
            "checkpoint_sha256": "checkpoint-digest",
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


def _save_payload(tmp_path: Path, payload: dict[str, object]) -> Path:
    """Persist one test payload and return its path."""
    path = tmp_path / "shard_0000.pt"
    torch.save(payload, path)
    return path


def _sha256(path: Path) -> str:
    """Hash one test file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_load_shard_validates_structure_without_modifying_source(
    tmp_path: Path,
) -> None:
    """Loading should expose v1 facts while leaving source bytes unchanged."""
    path = _save_payload(tmp_path, _make_payload())
    digest_before = _sha256(path)
    mtime_before = path.stat().st_mtime_ns

    shard = load_trajectory_shard(
        path,
        expected_sha256=digest_before,
        validate_full_tensors=True,
    )

    assert shard.batch_size == 2
    assert shard.num_diffusion_steps == 3
    assert shard.num_actions == 4
    assert shard.final_decision_index == 3
    assert shard.max_n_nodes == 4
    assert _sha256(path) == digest_before
    assert path.stat().st_mtime_ns == mtime_before


def test_episode_facts_preserve_reconstruction_semantics(tmp_path: Path) -> None:
    """Episode facts should join metadata and reconstruction rows by index."""
    shard = load_trajectory_shard(_save_payload(tmp_path, _make_payload()))

    valid = shard.episode_facts(0)
    invalid = shard.episode_facts(1)

    assert valid.sample_id == "sample-0"
    assert valid.rank == 2
    assert valid.shard_index == 4
    assert valid.seed == 3
    assert valid.num_nodes == 2
    assert valid.rdkit_valid is True
    assert valid.failure_type == "success"
    assert valid.smiles == "CC"
    assert invalid.rdkit_valid is False
    assert invalid.failure_type == "sanitize_failed"
    assert invalid.smiles is None
    assert len(shard.all_episode_facts()) == 2


def test_select_transitions_distinguishes_ordinary_and_final_actions(
    tmp_path: Path,
) -> None:
    """Only final decode transitions should be terminal and receive reward."""
    shard = load_trajectory_shard(_save_payload(tmp_path, _make_payload()))

    batch = shard.select_transitions(
        episode_indices=[0, 1],
        decision_indices=[1, 3],
        episode_rewards=[10.0, 20.0],
    )

    torch.testing.assert_close(
        batch.state_latent,
        shard.latents[torch.tensor([0, 1]), torch.tensor([1, 3])],
    )
    torch.testing.assert_close(
        batch.action_latent,
        shard.latents[torch.tensor([0, 1]), torch.tensor([2, 4])],
    )
    torch.testing.assert_close(batch.next_state_latent, batch.action_latent)
    assert batch.recorded_behavior_score.tolist() == pytest.approx([1.0, 7.0])
    assert batch.recorded_timestep.tolist() == [1, 0]
    assert batch.source_diffusion_timestep.tolist() == [2, 0]
    assert batch.target_diffusion_timestep.tolist() == [1, -1]
    assert batch.decision_index.tolist() == [1, 3]
    assert batch.done.tolist() == [False, True]
    assert batch.is_final_decode.tolist() == [False, True]
    assert batch.reward is not None
    assert batch.reward.tolist() == pytest.approx([0.0, 20.0])
    assert batch.sample_ids == ("sample-0", "sample-1")
    assert batch.node_mask[:, :, 0].tolist() == [
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
    ]
    assert batch.action_mask[0].tolist() == [
        [1.0] * 5,
        [1.0] * 5,
        [0.0] * 5,
        [0.0] * 5,
    ]
    assert batch.action_mask[1].tolist() == [
        [1.0, 1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 0.0, 0.0],
    ]


def test_unscored_transition_keeps_reward_missing(tmp_path: Path) -> None:
    """Absence of a reward sidecar must not silently become zero reward."""
    shard = load_trajectory_shard(_save_payload(tmp_path, _make_payload()))

    batch = shard.select_transition(0, shard.final_decision_index)

    assert batch.reward is None
    assert batch.done.item() is True


def test_single_transition_applies_episode_reward_only_at_terminal(
    tmp_path: Path,
) -> None:
    """A supplied episode reward should remain terminal-only."""
    shard = load_trajectory_shard(_save_payload(tmp_path, _make_payload()))

    ordinary = shard.select_transition(0, 0, episode_reward=7.0)
    terminal = shard.select_transition(0, 3, episode_reward=7.0)

    assert ordinary.reward is not None
    assert ordinary.reward.item() == 0.0
    assert terminal.reward is not None
    assert terminal.reward.item() == 7.0


def test_transition_selection_rejects_invalid_indices_and_rewards(
    tmp_path: Path,
) -> None:
    """Out-of-range pairs and malformed reward sidecars should fail early."""
    shard = load_trajectory_shard(_save_payload(tmp_path, _make_payload()))

    with pytest.raises(IndexError, match="episode_index"):
        shard.select_transition(2, 0)
    with pytest.raises(OfflineTrajectoryDataError, match="decision index"):
        shard.select_transition(0, 4)
    with pytest.raises(OfflineTrajectoryDataError, match="integer indices"):
        shard.select_transitions([0.5], [0])
    with pytest.raises(OfflineTrajectoryDataError, match="integer indices"):
        shard.select_transition(0.5, 0)
    with pytest.raises(OfflineTrajectoryDataError, match="length"):
        shard.select_transitions([0], [0], episode_rewards=[1.0])
    with pytest.raises(OfflineTrajectoryDataError, match="finite"):
        shard.select_transitions(
            [0],
            [0],
            episode_rewards=[float("nan"), 1.0],
        )


def test_load_rejects_wrong_format_and_shape(tmp_path: Path) -> None:
    """Format and trajectory-length drift must not enter the offline dataset."""
    wrong_format = _make_payload()
    wrong_format["format"] = "unknown"
    with pytest.raises(OfflineTrajectoryDataError, match="unsupported"):
        load_trajectory_shard(_save_payload(tmp_path, wrong_format))

    wrong_length = _make_payload()
    wrong_length["latents"] = wrong_length["latents"][:, :-1]  # type: ignore[index]
    with pytest.raises(OfflineTrajectoryDataError, match="latents length"):
        load_trajectory_shard(_save_payload(tmp_path, wrong_length))


def test_load_rejects_timestep_and_sample_id_drift(tmp_path: Path) -> None:
    """Decision order and reconstruction joins are part of the v1 contract."""
    wrong_time = _make_payload()
    wrong_time["timesteps"] = torch.tensor([2, 0, 1, 0]).repeat(2, 1)
    with pytest.raises(OfflineTrajectoryDataError, match="timesteps"):
        load_trajectory_shard(_save_payload(tmp_path, wrong_time))

    wrong_ids = _make_payload()
    rows = wrong_ids["reconstruction_rows"]
    assert isinstance(rows, list)
    rows[1]["sample_id"] = "different"  # type: ignore[index]
    with pytest.raises(OfflineTrajectoryDataError, match="sample_ids"):
        load_trajectory_shard(_save_payload(tmp_path, wrong_ids))


def test_full_validation_detects_nonfinite_and_nonzero_padding(
    tmp_path: Path,
) -> None:
    """Expensive audit checks should identify corrupted numerical tensors."""
    nonfinite = _make_payload()
    nonfinite_latents = nonfinite["latents"]
    assert isinstance(nonfinite_latents, torch.Tensor)
    nonfinite_latents[0, 0, 0, 0] = float("nan")
    shard = load_trajectory_shard(_save_payload(tmp_path, nonfinite))
    with pytest.raises(OfflineTrajectoryDataError, match="NaN or Inf"):
        shard.validate_full_tensors()

    nonzero_padding = _make_payload()
    padded_latents = nonzero_padding["latents"]
    assert isinstance(padded_latents, torch.Tensor)
    padded_latents[0, 0, 3, 0] = 1.0
    shard = load_trajectory_shard(_save_payload(tmp_path, nonzero_padding))
    with pytest.raises(OfflineTrajectoryDataError, match="nonzero padding"):
        shard.validate_full_tensors()


def test_sha256_mismatch_fails_before_payload_use(tmp_path: Path) -> None:
    """Manifest digest mismatches should reject a shard."""
    path = _save_payload(tmp_path, _make_payload())

    with pytest.raises(OfflineTrajectoryDataError, match="SHA-256"):
        load_trajectory_shard(path, expected_sha256="0" * 64)


@pytest.mark.skipif(
    REAL_SHARD_PATH is None or not REAL_SHARD_PATH.is_file(),
    reason="set DIFFSBDD_TEST_TRAJECTORY_SHARD for the real-shard contract test",
)
def test_real_formal_shard_matches_offline_transition_contract() -> None:
    """A formal 1000-step shard should expose 1000 ordinary plus one final action."""
    assert REAL_SHARD_PATH is not None
    shard = load_trajectory_shard(
        REAL_SHARD_PATH,
        validate_full_tensors=True,
    )

    assert shard.batch_size == 10
    assert shard.num_diffusion_steps == 1000
    assert shard.num_actions == 1001
    assert shard.final_decision_index == 1000

    final = shard.select_transition(0, shard.final_decision_index)
    torch.testing.assert_close(final.state_latent[0], shard.latents[0, -2])
    torch.testing.assert_close(final.action_latent[0], shard.latents[0, -1])
    assert final.recorded_timestep.item() == 0
    assert final.source_diffusion_timestep.item() == 0
    assert final.target_diffusion_timestep.item() == -1
    assert final.done.item() is True
    assert final.is_final_decode.item() is True
    assert final.reward is None
    num_nodes = int(shard.nodesxsample[0].item())
    assert torch.equal(
        shard.categorical[0, :num_nodes].argmax(dim=-1),
        shard.latents[0, -2, :num_nodes, 3:].argmax(dim=-1),
    )
    assert bool(torch.all(final.action_mask[0, :num_nodes, :3] == 1))
    assert bool(torch.all(final.action_mask[0, :num_nodes, 3:] == 0))
