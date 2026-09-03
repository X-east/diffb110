"""Pure tests for endpoint-based DiffSBDD Global DPO primitives."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from ddpo.global_dpo.endpoint_data import (
    EndpointStore,
    GlobalPreferencePair,
    load_pair_manifest,
    validate_pairs,
    validate_split_pairs,
)
from ddpo.global_dpo.global_score import component_epsilon_losses, score_from_component_losses
from ddpo.global_dpo.loss import compute_global_dpo_loss, compute_global_e2po_loss
from ddpo.global_dpo.paired_noise import build_paired_noisy_batch
from ddpo.global_dpo.trainer import GlobalDPOTrainer, GlobalDPOTrainerConfig


def _pair(index: int, prefix: str = "") -> GlobalPreferencePair:
    """Build one minimal valid pair for validation tests."""
    return GlobalPreferencePair(
        winner_id=f"{prefix}w{index}",
        loser_id=f"{prefix}l{index}",
        winner_shard="winner.pt",
        loser_shard="loser.pt",
        winner_index=index,
        loser_index=index,
        n_nodes=2,
    )


def test_component_losses_are_mean_per_atom_and_component() -> None:
    """Coordinate and atom-feature losses use their own feature dimensions."""
    prediction = torch.zeros(3, 5)
    epsilon = torch.ones_like(prediction)
    mask = torch.tensor([0, 0, 1])
    sizes = torch.tensor([2, 1])
    x_loss, v_loss = component_epsilon_losses(prediction, epsilon, mask, sizes, coordinate_dim=3)
    assert torch.allclose(x_loss, torch.ones(2))
    assert torch.allclose(v_loss, torch.ones(2))


def test_global_score_is_larger_for_lower_denoising_error() -> None:
    """The preference score is the negative weighted denoising error."""
    score = score_from_component_losses(
        torch.tensor([1.0, 2.0]),
        torch.tensor([3.0, 4.0]),
        x_weight=0.25,
        v_weight=0.75,
    )
    assert torch.allclose(score, torch.tensor([-2.5, -3.5]))


def test_global_dpo_identity_has_log_two_loss() -> None:
    """Current/reference identity is neutral DPO."""
    scores = torch.tensor([[1.0, -1.0], [3.0, 2.0]])
    facts = compute_global_dpo_loss(scores, scores, beta=1.0)
    assert torch.allclose(facts.relative_margin, torch.zeros(2))
    assert facts.loss.item() == pytest.approx(math.log(2.0))


def test_global_e2po_equal_rewards_matches_neutral_binary_pair() -> None:
    """Equal endpoint rewards produce equal soft preference weights."""
    current = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    reference = torch.zeros_like(current)
    rewards = torch.tensor([1.0, 1.0])
    facts = compute_global_e2po_loss(
        current,
        reference,
        rewards,
        rewards,
        beta=1.0,
        reward_temperature=1.0,
    )
    assert torch.allclose(facts.winner_weight, torch.full((2,), 0.5))
    assert torch.allclose(facts.loser_weight, torch.full((2,), 0.5))
    assert torch.isfinite(facts.loss)


def test_global_e2po_reward_gap_increases_winner_weight() -> None:
    """A larger winner reward must produce a larger winner soft label."""
    current = torch.zeros(1, 2)
    reference = torch.zeros_like(current)
    facts = compute_global_e2po_loss(
        current,
        reference,
        torch.tensor([2.0]),
        torch.tensor([0.0]),
        beta=1.0,
        reward_temperature=1.0,
    )
    assert facts.winner_weight.item() > 0.5
    assert facts.loser_weight.item() < 0.5


def test_load_pair_manifest_can_join_reward_sidecar(tmp_path) -> None:
    """Legacy pair manifests can receive rewards from sample-id sidecars."""
    pair_path = tmp_path / "pairs.csv"
    pair_path.write_text(
        "winner_id,loser_id,winner_shard,loser_shard,winner_index,loser_index,n_nodes\n"
        "w0,l0,w.pt,l.pt,0,0,2\n",
        encoding="utf-8",
    )
    sidecar_path = tmp_path / "reward_sidecar.csv"
    sidecar_path.write_text(
        "sample_id,reward_total_raw\n"
        "w0,2.0\n"
        "l0,-1.0\n",
        encoding="utf-8",
    )
    pairs = load_pair_manifest(pair_path, reward_sidecar=sidecar_path)
    assert pairs[0].reward_winner == pytest.approx(2.0)
    assert pairs[0].reward_loser == pytest.approx(-1.0)


def test_load_pair_manifest_applies_reward_sign(tmp_path) -> None:
    """Lower-is-better scores can be converted to larger-is-better rewards."""
    pair_path = tmp_path / "pairs.csv"
    pair_path.write_text(
        "winner_id,loser_id,winner_shard,loser_shard,winner_index,loser_index,n_nodes\n"
        "w0,l0,w.pt,l.pt,0,0,2\n",
        encoding="utf-8",
    )
    sidecar_path = tmp_path / "reward_sidecar.csv"
    sidecar_path.write_text(
        "sample_id,docking_raw\n"
        "w0,-8.0\n"
        "l0,-6.0\n",
        encoding="utf-8",
    )
    pairs = load_pair_manifest(
        pair_path,
        reward_sidecar=sidecar_path,
        reward_key="docking_raw",
        reward_sign=-1.0,
    )
    assert pairs[0].reward_winner == pytest.approx(8.0)
    assert pairs[0].reward_loser == pytest.approx(6.0)


def test_endpoint_store_rejects_manifest_id_mismatch(tmp_path) -> None:
    """A stale pair index must not silently substitute another endpoint."""
    shard = tmp_path / "shard.pt"
    torch.save(
        {
            "sample_ids": ["actual_id"],
            "x": torch.zeros(1, 2, 3),
            "categorical": torch.zeros(1, 2, 10),
            "nodesxsample": torch.tensor([2]),
        },
        shard,
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        EndpointStore().sample(str(shard), 0, expected_sample_id="stale_id")


def test_endpoint_store_accepts_matching_manifest_id(tmp_path) -> None:
    """A matching pair ID is accepted after endpoint validation."""
    shard = tmp_path / "shard.pt"
    torch.save(
        {
            "sample_ids": ["actual_id"],
            "x": torch.zeros(1, 2, 3),
            "categorical": torch.zeros(1, 2, 10),
            "nodesxsample": torch.tensor([2]),
        },
        shard,
    )
    sample = EndpointStore().sample(str(shard), 0, expected_sample_id="actual_id")
    assert sample.sample_id == "actual_id"


class _ParityDDPM:
    """Minimal ConditionalDDPM-shaped object for native-noise parity tests."""

    n_dims = 3

    @staticmethod
    def normalize(ligand, pocket):
        """Mirror the no-op normalization used by this synthetic fixture."""
        return ligand, pocket

    @staticmethod
    def remove_mean_batch(x_lig, x_pocket, lig_indices, pocket_indices):
        """Mirror ConditionalDDPM's ligand-center projection."""
        means = []
        for batch_index in range(int(lig_indices.max().item()) + 1):
            means.append(x_lig[lig_indices == batch_index].mean(dim=0))
        mean = torch.stack(means)
        return x_lig - mean[lig_indices], x_pocket - mean[pocket_indices]

    @staticmethod
    def gamma(t):
        """Return a zero schedule so alpha/sigma are deterministic here."""
        return torch.zeros_like(t)

    @staticmethod
    def inflate_batch_array(value, target):
        """Expand one timestep value to the flat ligand atom layout."""
        return value.repeat_interleave(target.shape[0] // value.shape[0], dim=0)

    @staticmethod
    def alpha(gamma, target):
        """Use unit signal coefficient in the synthetic reference."""
        return torch.ones_like(target)

    @staticmethod
    def sigma(gamma, target):
        """Use a fixed noise coefficient in the synthetic reference."""
        return torch.full_like(target, 0.5)


class _ParityModel:
    """Minimal model wrapper exposing ConditionalDDPM-native semantics."""

    def __init__(self) -> None:
        self.device = "cpu"
        self.ddpm = _ParityDDPM()

    @staticmethod
    def _replicate_pocket(batch_size: int):
        """Return two pocket atoms per endpoint with four features."""
        pocket_x = torch.tensor([[10.0, 0.0, 0.0], [12.0, 0.0, 0.0]])
        pocket_v = torch.zeros(2, 4)
        return {
            "x": pocket_x.repeat(batch_size, 1),
            "one_hot": pocket_v.repeat(batch_size, 1),
            "size": torch.full((batch_size,), 2, dtype=torch.long),
            "mask": torch.repeat_interleave(torch.arange(batch_size), 2),
        }


def test_paired_noise_matches_condition_model_semantics() -> None:
    """Paired endpoint noise keeps clean pocket conditioning and projects COM."""
    model = _ParityModel()
    items = [
        SimpleNamespace(
            sample_id="w",
            n_nodes=2,
            x=torch.tensor([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
            v=torch.zeros(2, 4),
        ),
        SimpleNamespace(
            sample_id="l",
            n_nodes=2,
            x=torch.tensor([[5.0, 0.0, 0.0], [7.0, 0.0, 0.0]]),
            v=torch.zeros(2, 4),
        ),
    ]
    batch = build_paired_noisy_batch(model, items, 250, timestep_grid=500, noise_seed=17)

    # The pair uses common random numbers, but each endpoint still has the
    # native ligand epsilon marginal and the pocket remains a clean condition.
    assert torch.allclose(batch.epsilon[:2], batch.epsilon[2:])
    assert not torch.allclose(batch.xh0_pocket[:2], batch.xh0_pocket[2:])
    assert torch.allclose(batch.xh0_pocket[:, 3:], torch.zeros_like(batch.xh0_pocket[:, 3:]))
    for batch_index in range(2):
        assert torch.allclose(
            batch.z_t[batch.ligand_mask == batch_index, :3].mean(dim=0),
            torch.zeros(3),
            atol=1e-6,
        )
    assert torch.allclose(batch.timestep, torch.full((2, 1), 0.5))
    assert torch.isfinite(batch.z_t).all()


def test_pair_validation_rejects_endpoint_reuse() -> None:
    """An endpoint cannot appear in two preference pairs."""
    first = _pair(0)
    second = GlobalPreferencePair(
        winner_id=first.winner_id,
        loser_id="other",
        winner_shard="winner.pt",
        loser_shard="loser.pt",
        winner_index=0,
        loser_index=1,
        n_nodes=2,
    )
    with pytest.raises(ValueError, match="reused"):
        validate_pairs([first, second])


def test_split_validation_rejects_endpoint_leakage() -> None:
    """Train and validation endpoint IDs must be disjoint."""
    train = [_pair(0)]
    validation = [_pair(0)]
    with pytest.raises(ValueError, match="leakage"):
        validate_split_pairs(train, validation)


class _DummyEgnn(torch.nn.Module):
    """Minimal EGNN-shaped module for freeze-policy testing."""

    def __init__(self) -> None:
        super().__init__()
        self.e_block_0 = torch.nn.Linear(2, 2)
        self.e_block_1 = torch.nn.Linear(2, 2)
        self.e_block_2 = torch.nn.Linear(2, 2)
        self.e_block_3 = torch.nn.Linear(2, 2)
        self.e_block_4 = torch.nn.Linear(2, 2)


class _DummyDynamics(torch.nn.Module):
    """Minimal DiffSBDD dynamics-shaped module for freeze-policy testing."""

    def __init__(self) -> None:
        super().__init__()
        self.atom_encoder = torch.nn.Linear(2, 2)
        self.residue_encoder = torch.nn.Linear(2, 2)
        self.egnn = _DummyEgnn()
        self.atom_decoder = torch.nn.Linear(2, 2)
        self.residue_decoder = torch.nn.Linear(2, 2)


class _DummyCore(torch.nn.Module):
    """Minimal DiffSBDD-model-shaped container for freeze-policy testing."""

    def __init__(self) -> None:
        super().__init__()
        self.ddpm = torch.nn.Module()
        self.ddpm.dynamics = _DummyDynamics()


def test_freeze_policy_keeps_only_last_two_egnn_blocks_trainable(tmp_path) -> None:
    """Freeze the requested front and output modules without touching last blocks."""
    core = _DummyCore()
    trainer = object.__new__(GlobalDPOTrainer)
    trainer.current = SimpleNamespace(model=core)
    trainer.config = GlobalDPOTrainerConfig(
        output_dir=tmp_path,
        freeze_encoder_decoder=True,
        freeze_egnn_first_n_blocks=3,
    )

    report = trainer._apply_freeze_policy()

    assert report["trainable_parameters"] > 0
    assert not core.ddpm.dynamics.atom_encoder.weight.requires_grad
    assert not core.ddpm.dynamics.residue_encoder.weight.requires_grad
    assert not core.ddpm.dynamics.atom_decoder.weight.requires_grad
    assert not core.ddpm.dynamics.residue_decoder.weight.requires_grad
    assert not core.ddpm.dynamics.egnn.e_block_0.weight.requires_grad
    assert not core.ddpm.dynamics.egnn.e_block_1.weight.requires_grad
    assert not core.ddpm.dynamics.egnn.e_block_2.weight.requires_grad
    assert core.ddpm.dynamics.egnn.e_block_3.weight.requires_grad
    assert core.ddpm.dynamics.egnn.e_block_4.weight.requires_grad
