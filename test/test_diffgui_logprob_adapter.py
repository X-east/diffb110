"""Tests for the decoupled DiffGui log-probability adapter."""

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

from ddpo.diffgui.runtime.logprob_adapter import (  # noqa: E402
    diffgui_forward_step_logprob,
    diffgui_sample_with_logprob,
    recompute_discrete_reverse_logprob,
    sample_discrete_reverse_step,
)


def _make_transitions() -> tuple[ContigousTransition, GeneralCategoricalTransition, GeneralCategoricalTransition]:
    """Create small DiffGui reference transitions for tests."""
    betas = np.linspace(0.02, 0.12, 6, dtype=np.float64)
    pos_transition = ContigousTransition(betas)
    node_transition = GeneralCategoricalTransition(betas, num_classes=5, init_prob="tomask")
    edge_transition = GeneralCategoricalTransition(betas, num_classes=6, init_prob="absorb")
    return pos_transition, node_transition, edge_transition


class _FakeConfig:
    """Minimal config object for fake DiffGui sampling tests."""

    train_mode = "no_lab"


class _FakeDiffGui:
    """Small callable object with the DiffGui attributes used by the adapter."""

    def __init__(self) -> None:
        self.pos_transition, self.node_transition, self.edge_transition = _make_transitions()
        self.config = _FakeConfig()
        self.categorical_space = "discrete"
        self.center_pos_mode = "protein"
        self.num_timesteps = 3

    def __call__(
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
        """Return deterministic predictions with reference-compatible shapes."""
        del protein_node, protein_pos, protein_batch, ligand_edge_index, ligand_edge_batch, time_step, batch_lab
        n_halfedges = ligand_edge_pert.shape[0] // 2
        return {
            "pred_ligand_pos": ligand_pos_pert * 0.25,
            "pred_ligand_node": ligand_node_pert + 0.1,
            "pred_ligand_halfedge": ligand_edge_pert[:n_halfedges] + 0.2,
        }


def test_sampled_step_logprob_can_be_recomputed_exactly() -> None:
    """A sampled transition's old logprob must match same-policy recomputation."""
    torch.manual_seed(7)
    pos_transition, node_transition, edge_transition = _make_transitions()
    ligand_batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    halfedge_batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    time_step = torch.tensor([5, 3], dtype=torch.long)
    num_graphs = 2

    pos_t = torch.randn(len(ligand_batch), 3)
    pred_pos = torch.randn_like(pos_t)
    pred_node_logits = torch.randn(len(ligand_batch), 5)
    pred_halfedge_logits = torch.randn(len(halfedge_batch), 6)
    _, _node_init, node_log_state = node_transition.sample_init(len(ligand_batch))
    _, _edge_init, edge_log_state = edge_transition.sample_init(len(halfedge_batch))

    step = sample_discrete_reverse_step(
        pos_transition=pos_transition,
        node_transition=node_transition,
        edge_transition=edge_transition,
        pos_t=pos_t,
        node_log_state_t=node_log_state,
        halfedge_log_state_t=edge_log_state,
        pred_pos=pred_pos,
        pred_node_logits=pred_node_logits,
        pred_halfedge_logits=pred_halfedge_logits,
        time_step=time_step,
        ligand_batch=ligand_batch,
        halfedge_batch=halfedge_batch,
        num_graphs=num_graphs,
    )
    recomputed = recompute_discrete_reverse_logprob(
        pos_transition=pos_transition,
        node_transition=node_transition,
        edge_transition=edge_transition,
        pos_t=pos_t,
        node_log_state_t=node_log_state,
        halfedge_log_state_t=edge_log_state,
        pred_pos=pred_pos,
        pred_node_logits=pred_node_logits,
        pred_halfedge_logits=pred_halfedge_logits,
        time_step=time_step,
        ligand_batch=ligand_batch,
        halfedge_batch=halfedge_batch,
        pos_prev=step.pos_prev,
        node_prev=step.node_prev,
        halfedge_prev=step.halfedge_prev,
        num_graphs=num_graphs,
    )

    assert torch.allclose(step.logprob_total, recomputed.logprob_total, atol=1e-6)
    assert torch.allclose(step.logprob_pos, recomputed.logprob_pos, atol=1e-6)
    assert torch.allclose(step.logprob_node, recomputed.logprob_node, atol=1e-6)
    assert torch.allclose(step.logprob_halfedge, recomputed.logprob_halfedge, atol=1e-6)
    assert torch.isfinite(step.logprob_total).all()


def test_t0_position_logprob_is_finite_and_zero() -> None:
    """DiffGui's deterministic t=0 position transition should not create inf logprob."""
    torch.manual_seed(11)
    pos_transition, node_transition, edge_transition = _make_transitions()
    ligand_batch = torch.tensor([0, 0, 1], dtype=torch.long)
    halfedge_batch = torch.tensor([0, 1], dtype=torch.long)
    time_step = torch.tensor([0, 0], dtype=torch.long)

    _, _node_init, node_log_state = node_transition.sample_init(len(ligand_batch))
    _, _edge_init, edge_log_state = edge_transition.sample_init(len(halfedge_batch))
    step = sample_discrete_reverse_step(
        pos_transition=pos_transition,
        node_transition=node_transition,
        edge_transition=edge_transition,
        pos_t=torch.randn(len(ligand_batch), 3),
        node_log_state_t=node_log_state,
        halfedge_log_state_t=edge_log_state,
        pred_pos=torch.randn(len(ligand_batch), 3),
        pred_node_logits=torch.randn(len(ligand_batch), 5),
        pred_halfedge_logits=torch.randn(len(halfedge_batch), 6),
        time_step=time_step,
        ligand_batch=ligand_batch,
        halfedge_batch=halfedge_batch,
        num_graphs=2,
    )

    assert torch.allclose(step.logprob_pos, torch.zeros_like(step.logprob_pos), atol=1e-7)
    assert torch.isfinite(step.logprob_node).all()
    assert torch.isfinite(step.logprob_halfedge).all()
    assert torch.isfinite(step.logprob_total).all()


def test_sample_with_logprob_records_recomputable_transition() -> None:
    """The functional DiffGui sampler should save enough state to recompute a step."""
    torch.manual_seed(17)
    model = _FakeDiffGui()
    ligand_batch = torch.tensor([0, 0, 1], dtype=torch.long)
    halfedge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    halfedge_batch = torch.tensor([0, 1], dtype=torch.long)
    outputs = diffgui_sample_with_logprob(
        model,
        n_graphs=2,
        protein_node=torch.zeros(2, 4),
        protein_pos=torch.tensor([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
        protein_batch=torch.tensor([0, 1], dtype=torch.long),
        ligand_batch=ligand_batch,
        halfedge_index=halfedge_index,
        halfedge_batch=halfedge_batch,
    )

    assert outputs["logprob"]["total"].shape == (model.num_timesteps, 2)
    assert outputs["traj"][0].shape[0] == model.num_timesteps + 1
    assert outputs["policy_traj"]["node_log_state"].shape == outputs["traj"][0].shape
    assert torch.isfinite(outputs["logprob"]["total"]).all()

    transition_idx = 1
    recomputed = diffgui_forward_step_logprob(
        model,
        protein_node=torch.zeros(2, 4),
        protein_pos=torch.tensor([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
        protein_batch=torch.tensor([0, 1], dtype=torch.long),
        ligand_node_t=outputs["policy_traj"]["node"][transition_idx],
        ligand_pos_t=outputs["policy_traj"]["pos_centered"][transition_idx],
        ligand_batch=ligand_batch,
        ligand_halfedge_t=outputs["policy_traj"]["halfedge"][transition_idx],
        halfedge_index=halfedge_index,
        halfedge_batch=halfedge_batch,
        node_log_state_t=outputs["policy_traj"]["node_log_state"][transition_idx],
        halfedge_log_state_t=outputs["policy_traj"]["halfedge_log_state"][transition_idx],
        ligand_pos_prev=outputs["policy_traj"]["pos_centered"][transition_idx + 1],
        ligand_node_prev=outputs["policy_traj"]["node"][transition_idx + 1],
        ligand_halfedge_prev=outputs["policy_traj"]["halfedge"][transition_idx + 1],
        time_step=outputs["logprob"]["timesteps"][transition_idx],
    )
    assert torch.allclose(
        recomputed.logprob_total,
        outputs["logprob"]["total"][transition_idx],
        atol=1e-6,
    )
