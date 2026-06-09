"""Tests for the decoupled DiffGui PPO actor."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIFFGUI_ROOT = PROJECT_ROOT / "reference project" / "DiffGui"
if str(DIFFGUI_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFGUI_ROOT))

from models.transition import ContigousTransition, GeneralCategoricalTransition  # noqa: E402

from ddpo.diffgui.runtime.actor import DiffGuiActor, DiffGuiPolicyBatch  # noqa: E402
from ddpo.diffgui.runtime.logprob_adapter import diffgui_sample_with_logprob  # noqa: E402


class _FakeConfig:
    """Minimal config object for the fake DiffGui model."""

    train_mode = "no_lab"


class _TrainableFakeDiffGui(torch.nn.Module):
    """Small trainable DiffGui-like model for actor smoke tests."""

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
        """Return predictions that depend on trainable parameters."""
        del protein_node, protein_pos, protein_batch, ligand_batch, ligand_edge_index, ligand_edge_batch, time_step, batch_lab
        n_halfedges = ligand_edge_pert.shape[0] // 2
        return {
            "pred_ligand_pos": ligand_pos_pert * self.scale,
            "pred_ligand_node": ligand_node_pert + self.node_bias,
            "pred_ligand_halfedge": ligand_edge_pert[:n_halfedges] + self.edge_bias,
        }


class _FreezeFakeDenoiser(torch.nn.Module):
    """Minimal DiffGui-like denoiser with block ModuleLists."""

    def __init__(self) -> None:
        super().__init__()
        self.node_blocks_with_edge = torch.nn.ModuleList([torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)])
        self.edge_embs = torch.nn.ModuleList([torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)])
        self.edge_blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)])
        self.pos_blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)])


class _FreezeFakeDiffGui(torch.nn.Module):
    """Minimal DiffGui-like module for freeze policy tests."""

    def __init__(self) -> None:
        super().__init__()
        self.input_embedder = torch.nn.Linear(2, 2)
        self.denoiser = _FreezeFakeDenoiser()
        self.ligand_node_decoder = torch.nn.Linear(2, 2)
        self.ligand_edge_decoder = torch.nn.Linear(2, 2)


def _make_policy_batch(model: _TrainableFakeDiffGui) -> DiffGuiPolicyBatch:
    """Generate a tiny old-policy rollout batch."""
    torch.manual_seed(23)
    ligand_batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    halfedge_index = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)
    halfedge_batch = torch.tensor([0, 1], dtype=torch.long)
    protein_node = torch.zeros(2, 4)
    protein_pos = torch.tensor([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    protein_batch = torch.tensor([0, 1], dtype=torch.long)
    outputs = diffgui_sample_with_logprob(
        model,
        n_graphs=2,
        protein_node=protein_node,
        protein_pos=protein_pos,
        protein_batch=protein_batch,
        ligand_batch=ligand_batch,
        halfedge_index=halfedge_index,
        halfedge_batch=halfedge_batch,
    )
    return DiffGuiPolicyBatch.from_sample_outputs(
        outputs=outputs,
        advantages=torch.tensor([1.0, 0.5]),
        protein_node=protein_node,
        protein_pos=protein_pos,
        protein_batch=protein_batch,
        ligand_batch=ligand_batch,
        halfedge_index=halfedge_index,
        halfedge_batch=halfedge_batch,
    )


def test_diffgui_actor_update_policy_smoke() -> None:
    """DiffGuiActor should run one PPO update and return finite metrics."""
    model = _TrainableFakeDiffGui()
    batch = _make_policy_batch(model)
    actor = DiffGuiActor(
        model,
        {
            "train": {
                "learning_rate": 1e-3,
                "adam_beta1": 0.9,
                "adam_beta2": 0.999,
                "adam_weight_decay": 0.0,
                "adam_epsilon": 1e-8,
                "clip_range": 0.2,
                "max_grad_norm": 1.0,
                "ppo_epochs": 1,
            }
        },
    )

    before = model.scale.detach().clone()
    metrics = actor.update_policy(batch)

    assert metrics["PPO_Epochs"] == 1.0
    assert metrics["TrainBatchSize"] == 2.0
    assert metrics["NumSteps"] == float(model.num_timesteps)
    assert np.isfinite(metrics["Loss"])
    assert np.isfinite(metrics["PPO_ApproxKL"])
    assert not torch.allclose(before, model.scale.detach())


def test_diffgui_policy_batch_select_graphs_rebases_halfedges() -> None:
    """Graph slicing should compact graph ids and ligand node indices."""
    model = _TrainableFakeDiffGui()
    batch = _make_policy_batch(model)

    selected = batch.select_graphs([1])

    assert selected.advantages.tolist() == [0.5]
    assert selected.protein_batch.tolist() == [0]
    assert selected.ligand_batch.tolist() == [0, 0]
    assert selected.halfedge_batch.tolist() == [0]
    assert selected.halfedge_index.tolist() == [[0], [1]]
    assert selected.old_logprob.shape == (model.num_timesteps, 1)
    selected.validate()


def test_diffgui_actor_update_policy_graph_microbatch_smoke() -> None:
    """DiffGuiActor should stream graph microbatches without changing batch semantics."""
    model = _TrainableFakeDiffGui()
    batch = _make_policy_batch(model)
    actor = DiffGuiActor(
        model,
        {
            "train": {
                "learning_rate": 1e-3,
                "clip_range": 0.2,
                "max_grad_norm": 1.0,
                "ppo_epochs": 1,
                "graph_micro_batch_size": 1,
            }
        },
    )

    metrics = actor.update_policy(batch)

    assert metrics["PPO_Epochs"] == 1.0
    assert metrics["TrainBatchSize"] == 2.0
    assert metrics["GraphMicroBatchSize"] == 1.0
    assert metrics["NumSteps"] == float(model.num_timesteps)
    assert np.isfinite(metrics["Loss"])


def test_diffgui_freeze_policy_heads_only() -> None:
    """heads_only should freeze backbone parameters and train decoders."""
    model = _FreezeFakeDiffGui()

    report = DiffGuiActor._apply_freeze_policy(model, {"mode": "heads_only"})

    assert report["trainable_params"] > 0
    assert report["trainable_params"] < report["total_params"]
    assert not any(parameter.requires_grad for parameter in model.input_embedder.parameters())
    assert not any(parameter.requires_grad for parameter in model.denoiser.parameters())
    assert all(parameter.requires_grad for parameter in model.ligand_node_decoder.parameters())
    assert all(parameter.requires_grad for parameter in model.ligand_edge_decoder.parameters())


def test_diffgui_freeze_policy_last_n_blocks() -> None:
    """last_n_blocks should train decoders and only the final denoiser block group."""
    model = _FreezeFakeDiffGui()

    report = DiffGuiActor._apply_freeze_policy(
        model,
        {"mode": "last_n_blocks", "train_last_n_blocks": 1},
    )

    assert report["train_last_n_blocks"] == 1
    assert not any(parameter.requires_grad for parameter in model.input_embedder.parameters())
    assert not any(parameter.requires_grad for parameter in model.denoiser.node_blocks_with_edge[0].parameters())
    assert all(parameter.requires_grad for parameter in model.denoiser.node_blocks_with_edge[-1].parameters())
    assert all(parameter.requires_grad for parameter in model.denoiser.edge_embs[-1].parameters())
    assert all(parameter.requires_grad for parameter in model.denoiser.edge_blocks[-1].parameters())
    assert all(parameter.requires_grad for parameter in model.denoiser.pos_blocks[-1].parameters())
    assert all(parameter.requires_grad for parameter in model.ligand_node_decoder.parameters())
    assert all(parameter.requires_grad for parameter in model.ligand_edge_decoder.parameters())


def test_diffgui_actor_accepts_meta_info_bridge() -> None:
    """The actor should accept a metadata bridge carrying DiffGuiPolicyBatch."""
    model = _TrainableFakeDiffGui()
    batch = _make_policy_batch(model)
    data = SimpleNamespace(meta_info={"diffgui_policy_batch": batch})
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

    assert metrics["PPO_Epochs"] == 1.0
    assert np.isfinite(metrics["Loss"])
