"""Tests for decoupled DiffGui rollout-to-actor plumbing."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIFFGUI_ROOT = PROJECT_ROOT / "reference project" / "DiffGui"
if str(DIFFGUI_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFGUI_ROOT))

from models.transition import ContigousTransition, GeneralCategoricalTransition  # noqa: E402

from ddpo.diffgui.runtime.actor import DiffGuiActor, DiffGuiPolicyBatch  # noqa: E402
from ddpo.diffgui.runtime.rollout import (  # noqa: E402
    DiffGuiPromptBatch,
    DiffGuiRollout,
    make_diffgui_ligand_placeholder,
)


class _FakeConfig:
    """Minimal config object for a DiffGui-like model."""

    train_mode = "no_lab"


class _TrainableFakeDiffGui(torch.nn.Module):
    """Small trainable DiffGui-like model for rollout smoke tests."""

    def __init__(self) -> None:
        super().__init__()
        betas = np.linspace(0.02, 0.12, 6, dtype=np.float64)
        self.pos_transition = ContigousTransition(betas)
        self.node_transition = GeneralCategoricalTransition(betas, num_classes=5, init_prob="tomask")
        self.edge_transition = GeneralCategoricalTransition(betas, num_classes=6, init_prob="absorb")
        self.config = _FakeConfig()
        self.categorical_space = "discrete"
        self.center_pos_mode = "protein"
        self.num_timesteps = 3
        self.scale = torch.nn.Parameter(torch.tensor(0.25))
        self.node_bias = torch.nn.Parameter(torch.zeros(5))
        self.edge_bias = torch.nn.Parameter(torch.zeros(6))

    def forward(
        self,
        protein_node: torch.Tensor,
        protein_pos: torch.Tensor,
        protein_batch: torch.Tensor,
        ligand_node_pert: torch.Tensor,
        ligand_pos_pert: torch.Tensor,
        ligand_batch: torch.Tensor,
        ligand_edge_pert: torch.Tensor,
        ligand_edge_index: torch.Tensor,
        ligand_edge_batch: torch.Tensor,
        time_step: torch.Tensor,
        batch_lab: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """Return trainable predictions with DiffGui-compatible keys."""
        del protein_node, protein_pos, protein_batch, ligand_batch
        del ligand_edge_index, ligand_edge_batch, time_step, batch_lab
        n_halfedges = ligand_edge_pert.shape[0] // 2
        return {
            "pred_ligand_pos": ligand_pos_pert * self.scale,
            "pred_ligand_node": ligand_node_pert + self.node_bias,
            "pred_ligand_halfedge": ligand_edge_pert[:n_halfedges] + self.edge_bias,
        }


def _make_prompt() -> DiffGuiPromptBatch:
    """Build a valid two-graph DiffGui prompt with all ligand halfedges."""
    ligand_batch, halfedge_index, halfedge_batch = make_diffgui_ligand_placeholder([3, 3])
    return DiffGuiPromptBatch(
        protein_node=torch.zeros(2, 4),
        protein_pos=torch.tensor([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
        protein_batch=torch.tensor([0, 1], dtype=torch.long),
        ligand_batch=ligand_batch,
        halfedge_index=halfedge_index,
        halfedge_batch=halfedge_batch,
    )


def test_make_diffgui_ligand_placeholder() -> None:
    """The ligand placeholder should build per-graph complete halfedges."""
    ligand_batch, halfedge_index, halfedge_batch = make_diffgui_ligand_placeholder([2, 3])

    assert ligand_batch.tolist() == [0, 0, 1, 1, 1]
    assert halfedge_index.shape == (2, 4)
    assert halfedge_batch.tolist() == [0, 1, 1, 1]
    assert halfedge_index[:, 0].tolist() == [0, 1]
    assert halfedge_index[:, -1].tolist() == [3, 4]


def test_diffgui_rollout_builds_policy_batch() -> None:
    """Rollout should produce a valid DiffGuiPolicyBatch on CPU by default."""
    torch.manual_seed(31)
    model = _TrainableFakeDiffGui()
    rollout = DiffGuiRollout(
        model,
        {
            "diffgui_rollout": {
                "synthetic_advantage_mode": "linear",
                "synthetic_advantage_scale": 2.0,
            }
        },
    )

    policy_batch = rollout.generate_policy_batch(_make_prompt())

    assert isinstance(policy_batch, DiffGuiPolicyBatch)
    assert policy_batch.advantages.device.type == "cpu"
    assert policy_batch.advantages.tolist() == [1.0, 2.0]
    assert policy_batch.old_logprob.shape == (model.num_timesteps, 2)
    policy_batch.validate()


def test_diffgui_rollout_dataproto_bridge_updates_actor() -> None:
    """Rollout DataProto bridge should feed directly into DiffGuiActor."""
    torch.manual_seed(37)
    model = _TrainableFakeDiffGui()
    rollout = DiffGuiRollout(model, {"diffgui_rollout": {"synthetic_advantage_mode": "ones"}})
    data = rollout.generate_samples(_make_prompt())
    actor = DiffGuiActor(
        model,
        {
            "train": {
                "learning_rate": 1e-3,
                "clip_range": 0.2,
                "max_grad_norm": 1.0,
                "ppo_epochs": 1,
            }
        },
    )

    metrics = actor.update_policy(data)

    assert data.batch["nodesxsample"].tolist() == [3, 3]
    assert data.meta_info["diffgui_policy_batch_kind"] == "xhe_joint"
    assert metrics["PPO_Epochs"] == 1.0
    assert metrics["TrainBatchSize"] == 2.0
    assert np.isfinite(metrics["Loss"])
