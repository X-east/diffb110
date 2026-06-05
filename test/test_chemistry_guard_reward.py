"""Tests for ChemistryGuardReward facts and factory integration."""

from __future__ import annotations

import logging

import pytest
import torch
from rdkit import Chem
from rdkit.Geometry import Point3D

from ddpo.rewards.composite import CompositeReward
from ddpo.rewards.factory import build_rewarder, get_enabled_rewards
from ddpo.rewards.chemistry_guard_reward import ChemistryGuardReward
from ddpo.runtime.protocol import DataProto, TensorDict


def _mol_with_linear_conformer(smiles: str, spacing: float = 1.45) -> Chem.Mol:
    """Create a small RDKit molecule with deterministic heavy-atom coordinates."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    conf = Chem.Conformer(mol.GetNumAtoms())
    for idx in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(idx, Point3D(float(idx) * spacing, 0.0, 0.0))
    mol.RemoveAllConformers()
    mol.AddConformer(conf)
    return mol


def _cyclopropane_with_conformer() -> Chem.Mol:
    """Create cyclopropane with a compact triangular conformer."""
    mol = Chem.MolFromSmiles("C1CC1")
    assert mol is not None
    conf = Chem.Conformer(mol.GetNumAtoms())
    positions = [Point3D(0.0, 0.0, 0.0), Point3D(1.5, 0.0, 0.0), Point3D(0.75, 1.3, 0.0)]
    for idx, position in enumerate(positions):
        conf.SetAtomPosition(idx, position)
    mol.RemoveAllConformers()
    mol.AddConformer(conf)
    return mol


def _batch_from_mols(mols: list[Chem.Mol | None], valid: list[int] | None = None) -> DataProto:
    """Create minimal DataProto with precomputed RDKit semantics."""
    valid = valid or [1 if mol is not None else 0 for mol in mols]
    batch_size = len(mols)
    batch = TensorDict(
        {
            "x": torch.zeros(batch_size, 4, 3),
            "categorical": torch.zeros(batch_size, 4, 5),
            "nodesxsample": torch.full((batch_size,), 4, dtype=torch.long),
        },
        batch_size=batch_size,
    )
    return DataProto(
        batch=batch,
        non_tensor_batch={
            "rdmols": mols,
            "rdkit_valid": valid,
            "smiles": [Chem.MolToSmiles(mol) if mol is not None else "" for mol in mols],
            "failure_type": ["success" if flag else "sanitize_failed" for flag in valid],
        },
        meta_info={"max_n_nodes": 4, "condition": False},
    )


def test_chemistry_guard_reward_writes_core_fact_fields() -> None:
    """Valid molecules should receive finite guard facts; invalid ones get penalty."""
    reward = ChemistryGuardReward(max_carbon_fraction=0.90)
    data = _batch_from_mols([_mol_with_linear_conformer("CCO"), None])

    result = reward.calculate_rewards(data)

    assert result.batch["chemistry_guard_reward"].shape == (2,)
    assert result.batch["chemistry_guard_success"].tolist() == [1, 0]
    assert result.batch["heavy_atom_count"].tolist() == pytest.approx([3.0, 0.0])
    assert result.batch["carbon_fraction"][0].item() == pytest.approx(2.0 / 3.0)
    assert result.batch["hetero_atom_count"].tolist() == pytest.approx([1.0, 0.0])
    assert result.batch["chemistry_guard_reward"][1].item() == pytest.approx(-1.0)


def test_chemistry_guard_penalizes_three_member_ring() -> None:
    """Three-member rings should be counted and lower the guard reward."""
    reward = ChemistryGuardReward(max_three_member_ring_count=0, violation_penalty=0.4)
    data = _batch_from_mols([
        _mol_with_linear_conformer("CCO"),
        _cyclopropane_with_conformer(),
    ])

    result = reward.calculate_rewards(data)

    assert result.batch["three_member_ring_count"].tolist() == pytest.approx([0.0, 1.0])
    assert result.batch["chemistry_guard_violation_count"].tolist() == pytest.approx([0.0, 2.0])
    assert result.batch["chemistry_guard_reward"][1].item() < result.batch["chemistry_guard_reward"][0].item()


def test_chemistry_guard_factory_can_run_as_zero_weight_provider(tmp_path) -> None:
    """enabled=true should instantiate chemistry_guard even when its weight is zero."""
    config = {
        "reward": {
            "weights": {"qed": 1.0, "chemistry_guard": 0.0},
            "chemistry_guard": {"enabled": True},
        }
    }

    enabled = get_enabled_rewards(config)
    rewarder = build_rewarder(config=config, dataset_info={}, logger=logging.getLogger(__name__))
    composite = rewarder.base_reward

    assert "chemistry_guard" in enabled
    assert isinstance(composite, CompositeReward)
    assert "chemistry_guard" in composite.rewards

    result = rewarder.calculate_rewards(_batch_from_mols([_mol_with_linear_conformer("CCO")]))

    assert "qed_reward" in result.batch
    assert "chemistry_guard_reward" in result.batch
    assert result.batch["reward_total_raw"].item() == pytest.approx(result.batch["qed_reward"].item())
