"""DiffGui reverse-transition sampling and log-probability helpers.

The functions here mirror the local reference implementation in
``reference project/DiffGui/models/model.py`` and
``reference project/DiffGui/models/transition.py`` while keeping the code under
the decoupled ``ddpo.diffgui`` namespace.  They provide the missing bridge
between DiffGui's denovo sampler and DDPO/PPO, namely a per-step log-probability
for the sampled ``pos/node/halfedge`` reverse transition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

try:
    from torch_scatter import scatter_mean
except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
    raise ModuleNotFoundError(
        "DiffGui logprob adapter requires torch_scatter, matching the DiffGui reference project."
    ) from exc


LOG_TWO_PI = math.log(2.0 * math.pi)


@dataclass
class DiffGuiReverseStep:
    """Outputs from one DiffGui reverse transition."""

    pos_prev: torch.Tensor
    node_prev: torch.Tensor
    halfedge_prev: torch.Tensor
    node_log_state_prev: torch.Tensor
    halfedge_log_state_prev: torch.Tensor
    logprob_total: torch.Tensor
    logprob_pos: torch.Tensor
    logprob_node: torch.Tensor
    logprob_halfedge: torch.Tensor
    pos_mu: torch.Tensor
    pos_sigma: torch.Tensor
    node_log_probs: torch.Tensor
    halfedge_log_probs: torch.Tensor
    node_prev_type: torch.Tensor
    halfedge_prev_type: torch.Tensor


@dataclass
class DiffGuiLogProbResult:
    """Log-probability result for an already sampled transition."""

    logprob_total: torch.Tensor
    logprob_pos: torch.Tensor
    logprob_node: torch.Tensor
    logprob_halfedge: torch.Tensor
    pos_mu: torch.Tensor
    pos_sigma: torch.Tensor
    node_log_probs: torch.Tensor
    halfedge_log_probs: torch.Tensor


def continuous_reverse_stats(
    transition: Any,
    *,
    x_t: torch.Tensor,
    x_recon: torch.Tensor,
    t: torch.Tensor,
    batch: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``mu`` and ``sigma`` used by DiffGui's continuous reverse step."""
    coef_x0 = _extract_transition_value(transition.coef_x0, t, batch)
    coef_xt = _extract_transition_value(transition.coef_xt, t, batch)
    mu = coef_x0 * x_recon + coef_xt * x_t
    sigma = _extract_transition_value(transition.std, t, batch)
    return mu, sigma


def gaussian_logprob_by_graph(
    sample: torch.Tensor,
    *,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    t: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
    reduction: str = "mean",
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute per-graph Gaussian transition log-probability.

    DiffGui's position transition is deterministic when ``t == 0``.  That step
    is assigned zero position log-probability rather than an improper infinite
    density; node and halfedge categorical terms remain stochastic at ``t == 0``.
    """
    sigma_safe = torch.clamp(sigma, min=eps)
    sample_detached = sample.detach()
    elem_logp = -0.5 * (((sample_detached - mu) / sigma_safe) ** 2 + 2.0 * torch.log(sigma_safe) + LOG_TWO_PI)
    token_logp = elem_logp.sum(dim=-1)
    time_zero = t[batch] == 0
    token_logp = torch.where(time_zero, torch.zeros_like(token_logp), token_logp)
    return _reduce_by_graph(token_logp, batch=batch, num_graphs=num_graphs, reduction=reduction)


def categorical_logprob_by_graph(
    log_probs: torch.Tensor,
    sample: torch.Tensor,
    *,
    batch: torch.Tensor,
    num_graphs: int,
    reduction: str = "mean",
) -> torch.Tensor:
    """Gather categorical log-probability for one-hot or index samples."""
    sample_type = _as_type_index(sample)
    token_logp = log_probs.gather(dim=-1, index=sample_type.unsqueeze(-1)).squeeze(-1)
    return _reduce_by_graph(token_logp, batch=batch, num_graphs=num_graphs, reduction=reduction)


def sample_discrete_reverse_step(
    *,
    pos_transition: Any,
    node_transition: Any,
    edge_transition: Any,
    pos_t: torch.Tensor,
    node_log_state_t: torch.Tensor,
    halfedge_log_state_t: torch.Tensor,
    pred_pos: torch.Tensor,
    pred_node_logits: torch.Tensor,
    pred_halfedge_logits: torch.Tensor,
    time_step: torch.Tensor,
    ligand_batch: torch.Tensor,
    halfedge_batch: torch.Tensor,
    num_graphs: int,
    reduction: str = "mean",
    prev_pos: torch.Tensor | None = None,
    prev_node_type: torch.Tensor | None = None,
    prev_halfedge_type: torch.Tensor | None = None,
) -> DiffGuiReverseStep:
    """Sample or score one discrete DiffGui reverse transition.

    Args:
        pos_transition: DiffGui ``ContigousTransition`` for coordinates.
        node_transition: DiffGui ``GeneralCategoricalTransition`` for atoms.
        edge_transition: DiffGui ``GeneralCategoricalTransition`` for halfedges.
        pos_t: Current centered ligand positions.
        node_log_state_t: The rolling log posterior state used by DiffGui.
        halfedge_log_state_t: The rolling halfedge log posterior state.
        pred_pos: Model-predicted denoised positions.
        pred_node_logits: Model-predicted node logits.
        pred_halfedge_logits: Model-predicted halfedge logits.
        time_step: Per-graph integer timestep tensor.
        ligand_batch: Node-to-graph indices.
        halfedge_batch: Halfedge-to-graph indices.
        num_graphs: Number of graphs in the batch.
        reduction: ``"mean"`` or ``"sum"`` graph aggregation.
        prev_pos: Optional already sampled previous positions for recomputing.
        prev_node_type: Optional already sampled previous node type indices.
        prev_halfedge_type: Optional already sampled previous halfedge indices.

    Returns:
        DiffGuiReverseStep with sampled previous state and old-policy logprob.
    """
    pos_mu, pos_sigma = continuous_reverse_stats(
        pos_transition,
        x_t=pos_t,
        x_recon=pred_pos,
        t=time_step,
        batch=ligand_batch,
    )
    if prev_pos is None:
        sampled_pos = pos_mu + pos_sigma * torch.randn_like(pos_mu)
        time_zero = (time_step[ligand_batch] == 0).unsqueeze(-1)
        pos_prev = torch.where(time_zero, pos_mu, sampled_pos)
    else:
        pos_prev = prev_pos

    node_log_recon = F.log_softmax(pred_node_logits, dim=-1)
    node_log_probs = node_transition.q_v_posterior(
        node_log_recon,
        node_log_state_t,
        time_step,
        ligand_batch,
        v0_prob=True,
    )
    if prev_node_type is None:
        node_prev_type = _sample_categorical_from_log_probs(node_log_probs)
    else:
        node_prev_type = prev_node_type.to(device=node_log_probs.device, dtype=torch.long)
    node_prev = node_transition.onehot_encode(node_prev_type)

    halfedge_log_recon = F.log_softmax(pred_halfedge_logits, dim=-1)
    halfedge_log_probs = edge_transition.q_v_posterior(
        halfedge_log_recon,
        halfedge_log_state_t,
        time_step,
        halfedge_batch,
        v0_prob=True,
    )
    if prev_halfedge_type is None:
        halfedge_prev_type = _sample_categorical_from_log_probs(halfedge_log_probs)
    else:
        halfedge_prev_type = prev_halfedge_type.to(device=halfedge_log_probs.device, dtype=torch.long)
    halfedge_prev = edge_transition.onehot_encode(halfedge_prev_type)

    logprob_pos = gaussian_logprob_by_graph(
        pos_prev,
        mu=pos_mu,
        sigma=pos_sigma,
        t=time_step,
        batch=ligand_batch,
        num_graphs=num_graphs,
        reduction=reduction,
    )
    logprob_node = categorical_logprob_by_graph(
        node_log_probs,
        node_prev_type,
        batch=ligand_batch,
        num_graphs=num_graphs,
        reduction=reduction,
    )
    logprob_halfedge = categorical_logprob_by_graph(
        halfedge_log_probs,
        halfedge_prev_type,
        batch=halfedge_batch,
        num_graphs=num_graphs,
        reduction=reduction,
    )
    return DiffGuiReverseStep(
        pos_prev=pos_prev,
        node_prev=node_prev,
        halfedge_prev=halfedge_prev,
        node_log_state_prev=node_log_probs,
        halfedge_log_state_prev=halfedge_log_probs,
        logprob_total=logprob_pos + logprob_node + logprob_halfedge,
        logprob_pos=logprob_pos,
        logprob_node=logprob_node,
        logprob_halfedge=logprob_halfedge,
        pos_mu=pos_mu,
        pos_sigma=pos_sigma,
        node_log_probs=node_log_probs,
        halfedge_log_probs=halfedge_log_probs,
        node_prev_type=node_prev_type,
        halfedge_prev_type=halfedge_prev_type,
    )


def recompute_discrete_reverse_logprob(
    *,
    pos_transition: Any,
    node_transition: Any,
    edge_transition: Any,
    pos_t: torch.Tensor,
    node_log_state_t: torch.Tensor,
    halfedge_log_state_t: torch.Tensor,
    pred_pos: torch.Tensor,
    pred_node_logits: torch.Tensor,
    pred_halfedge_logits: torch.Tensor,
    time_step: torch.Tensor,
    ligand_batch: torch.Tensor,
    halfedge_batch: torch.Tensor,
    pos_prev: torch.Tensor,
    node_prev: torch.Tensor,
    halfedge_prev: torch.Tensor,
    num_graphs: int,
    reduction: str = "mean",
) -> DiffGuiLogProbResult:
    """Recompute log-probability for an already sampled DiffGui transition."""
    step = sample_discrete_reverse_step(
        pos_transition=pos_transition,
        node_transition=node_transition,
        edge_transition=edge_transition,
        pos_t=pos_t,
        node_log_state_t=node_log_state_t,
        halfedge_log_state_t=halfedge_log_state_t,
        pred_pos=pred_pos,
        pred_node_logits=pred_node_logits,
        pred_halfedge_logits=pred_halfedge_logits,
        time_step=time_step,
        ligand_batch=ligand_batch,
        halfedge_batch=halfedge_batch,
        num_graphs=num_graphs,
        reduction=reduction,
        prev_pos=pos_prev,
        prev_node_type=_as_type_index(node_prev),
        prev_halfedge_type=_as_type_index(halfedge_prev),
    )
    return DiffGuiLogProbResult(
        logprob_total=step.logprob_total,
        logprob_pos=step.logprob_pos,
        logprob_node=step.logprob_node,
        logprob_halfedge=step.logprob_halfedge,
        pos_mu=step.pos_mu,
        pos_sigma=step.pos_sigma,
        node_log_probs=step.node_log_probs,
        halfedge_log_probs=step.halfedge_log_probs,
    )


@torch.no_grad()
def diffgui_sample_with_logprob(
    model: Any,
    *,
    n_graphs: int,
    protein_node: torch.Tensor,
    protein_pos: torch.Tensor,
    protein_batch: torch.Tensor,
    ligand_batch: torch.Tensor,
    halfedge_index: torch.Tensor,
    halfedge_batch: torch.Tensor,
    batch_lab: torch.Tensor | None = None,
    gui_strength: float | None = None,
    reduction: str = "mean",
    guidance: Any = None,
) -> dict[str, Any]:
    """Run DiffGui denovo sampling while recording per-step transition logprob.

    This mirrors ``DiffGui.sample`` for the common denovo discrete mode.  Bond
    guidance is intentionally rejected because it mutates ``pos_prev`` after the
    stochastic transition, making the recorded transition log-probability no
    longer match the actual stored action.
    """
    core_model = _core_model(model)
    _require_discrete_diffgui(core_model)
    if guidance is not None:
        raise ValueError("DiffGui PPO rollout must disable bond guidance for logprob consistency.")

    device = ligand_batch.device
    n_nodes_all = len(ligand_batch)
    n_halfedges_all = len(halfedge_batch)
    _, ligand_node_h_pert, node_log_state = core_model.node_transition.sample_init(n_nodes_all)
    _, ligand_halfedge_h_pert, halfedge_log_state = core_model.edge_transition.sample_init(n_halfedges_all)

    pocket_center_pos = scatter_mean(protein_pos, protein_batch, dim=0)
    ligand_center_pos = pocket_center_pos[ligand_batch]
    ligand_pos_init = ligand_center_pos + torch.randn_like(ligand_center_pos)
    protein_pos_centered, ligand_pos_pert, offset = _center_pos(
        protein_pos,
        ligand_pos_init,
        protein_batch,
        ligand_batch,
        mode=core_model.center_pos_mode,
    )

    num_timesteps = int(core_model.num_timesteps)
    ligand_node_traj = torch.zeros(
        [num_timesteps + 1, n_nodes_all, ligand_node_h_pert.shape[-1]],
        dtype=ligand_node_h_pert.dtype,
        device=device,
    )
    ligand_pos_traj = torch.zeros(
        [num_timesteps + 1, n_nodes_all, 3],
        dtype=ligand_pos_pert.dtype,
        device=device,
    )
    ligand_pos_centered_traj = torch.zeros_like(ligand_pos_traj)
    ligand_halfedge_traj = torch.zeros(
        [num_timesteps + 1, n_halfedges_all, ligand_halfedge_h_pert.shape[-1]],
        dtype=ligand_halfedge_h_pert.dtype,
        device=device,
    )
    node_log_state_traj = torch.zeros_like(ligand_node_traj)
    halfedge_log_state_traj = torch.zeros_like(ligand_halfedge_traj)

    logprob = torch.zeros([num_timesteps, n_graphs], dtype=ligand_pos_pert.dtype, device=device)
    logprob_pos = torch.zeros_like(logprob)
    logprob_node = torch.zeros_like(logprob)
    logprob_halfedge = torch.zeros_like(logprob)
    timesteps = torch.zeros([num_timesteps, n_graphs], dtype=torch.long, device=device)

    ligand_node_traj[0] = ligand_node_h_pert
    ligand_pos_traj[0] = ligand_pos_pert + offset[ligand_batch]
    ligand_pos_centered_traj[0] = ligand_pos_pert
    ligand_halfedge_traj[0] = ligand_halfedge_h_pert
    node_log_state_traj[0] = node_log_state
    halfedge_log_state_traj[0] = halfedge_log_state

    ligand_edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=1)
    ligand_edge_batch = torch.cat([halfedge_batch, halfedge_batch], dim=0)

    pred_ligand_pos = pred_ligand_node = pred_ligand_halfedge = None
    for traj_idx, step in enumerate(range(num_timesteps)[::-1]):
        time_step = torch.full(size=(n_graphs,), fill_value=step, dtype=torch.long, device=device)
        ligand_edge_h_pert = torch.cat([ligand_halfedge_h_pert, ligand_halfedge_h_pert], dim=0)
        pred_ligand_pos, pred_ligand_node, pred_ligand_halfedge = _predict_diffgui_step(
            model,
            protein_node=protein_node,
            protein_pos=protein_pos_centered,
            protein_batch=protein_batch,
            ligand_node_pert=ligand_node_h_pert,
            ligand_pos_pert=ligand_pos_pert,
            ligand_batch=ligand_batch,
            ligand_edge_pert=ligand_edge_h_pert,
            ligand_edge_index=ligand_edge_index,
            ligand_edge_batch=ligand_edge_batch,
            time_step=time_step,
            batch_lab=batch_lab,
            gui_strength=gui_strength,
        )
        reverse_step = sample_discrete_reverse_step(
            pos_transition=core_model.pos_transition,
            node_transition=core_model.node_transition,
            edge_transition=core_model.edge_transition,
            pos_t=ligand_pos_pert,
            node_log_state_t=node_log_state,
            halfedge_log_state_t=halfedge_log_state,
            pred_pos=pred_ligand_pos,
            pred_node_logits=pred_ligand_node,
            pred_halfedge_logits=pred_ligand_halfedge,
            time_step=time_step,
            ligand_batch=ligand_batch,
            halfedge_batch=halfedge_batch,
            num_graphs=n_graphs,
            reduction=reduction,
        )

        ligand_pos_pert = reverse_step.pos_prev
        ligand_node_h_pert = reverse_step.node_prev
        ligand_halfedge_h_pert = reverse_step.halfedge_prev
        node_log_state = reverse_step.node_log_state_prev
        halfedge_log_state = reverse_step.halfedge_log_state_prev

        write_idx = traj_idx + 1
        ligand_node_traj[write_idx] = ligand_node_h_pert
        ligand_pos_traj[write_idx] = ligand_pos_pert + offset[ligand_batch]
        ligand_pos_centered_traj[write_idx] = ligand_pos_pert
        ligand_halfedge_traj[write_idx] = ligand_halfedge_h_pert
        node_log_state_traj[write_idx] = node_log_state
        halfedge_log_state_traj[write_idx] = halfedge_log_state
        logprob[traj_idx] = reverse_step.logprob_total
        logprob_pos[traj_idx] = reverse_step.logprob_pos
        logprob_node[traj_idx] = reverse_step.logprob_node
        logprob_halfedge[traj_idx] = reverse_step.logprob_halfedge
        timesteps[traj_idx] = time_step

    assert pred_ligand_pos is not None
    return {
        "pred": [
            pred_ligand_node,
            pred_ligand_pos + offset[ligand_batch],
            pred_ligand_halfedge,
        ],
        "traj": [ligand_node_traj, ligand_pos_traj, ligand_halfedge_traj],
        "policy_traj": {
            "node": ligand_node_traj,
            "pos_centered": ligand_pos_centered_traj,
            "halfedge": ligand_halfedge_traj,
            "node_log_state": node_log_state_traj,
            "halfedge_log_state": halfedge_log_state_traj,
        },
        "logprob": {
            "total": logprob,
            "pos": logprob_pos,
            "node": logprob_node,
            "halfedge": logprob_halfedge,
            "timesteps": timesteps,
        },
    }


def diffgui_forward_step_logprob(
    model: Any,
    *,
    protein_node: torch.Tensor,
    protein_pos: torch.Tensor,
    protein_batch: torch.Tensor,
    ligand_node_t: torch.Tensor,
    ligand_pos_t: torch.Tensor,
    ligand_batch: torch.Tensor,
    ligand_halfedge_t: torch.Tensor,
    halfedge_index: torch.Tensor,
    halfedge_batch: torch.Tensor,
    node_log_state_t: torch.Tensor,
    halfedge_log_state_t: torch.Tensor,
    ligand_pos_prev: torch.Tensor,
    ligand_node_prev: torch.Tensor,
    ligand_halfedge_prev: torch.Tensor,
    time_step: torch.Tensor,
    batch_lab: torch.Tensor | None = None,
    gui_strength: float | None = None,
    reduction: str = "mean",
) -> DiffGuiLogProbResult:
    """Recompute current-policy logprob for one stored DiffGui transition."""
    core_model = _core_model(model)
    _require_discrete_diffgui(core_model)
    ligand_edge_h_t = torch.cat([ligand_halfedge_t, ligand_halfedge_t], dim=0)
    ligand_edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=1)
    ligand_edge_batch = torch.cat([halfedge_batch, halfedge_batch], dim=0)
    pred_pos, pred_node, pred_halfedge = _predict_diffgui_step(
        model,
        protein_node=protein_node,
        protein_pos=protein_pos,
        protein_batch=protein_batch,
        ligand_node_pert=ligand_node_t,
        ligand_pos_pert=ligand_pos_t,
        ligand_batch=ligand_batch,
        ligand_edge_pert=ligand_edge_h_t,
        ligand_edge_index=ligand_edge_index,
        ligand_edge_batch=ligand_edge_batch,
        time_step=time_step,
        batch_lab=batch_lab,
        gui_strength=gui_strength,
    )
    return recompute_discrete_reverse_logprob(
        pos_transition=core_model.pos_transition,
        node_transition=core_model.node_transition,
        edge_transition=core_model.edge_transition,
        pos_t=ligand_pos_t,
        node_log_state_t=node_log_state_t,
        halfedge_log_state_t=halfedge_log_state_t,
        pred_pos=pred_pos,
        pred_node_logits=pred_node,
        pred_halfedge_logits=pred_halfedge,
        time_step=time_step,
        ligand_batch=ligand_batch,
        halfedge_batch=halfedge_batch,
        pos_prev=ligand_pos_prev,
        node_prev=ligand_node_prev,
        halfedge_prev=ligand_halfedge_prev,
        num_graphs=int(time_step.shape[0]),
        reduction=reduction,
    )


def _predict_diffgui_step(
    model: Any,
    *,
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
    gui_strength: float | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the same model prediction branch as DiffGui.sample."""
    core_model = _core_model(model)
    train_mode = str(core_model.config.train_mode)
    if train_mode in ("ori", "no_bond"):
        preds_cond = model(
            protein_node,
            protein_pos,
            protein_batch,
            ligand_node_pert,
            ligand_pos_pert,
            ligand_batch,
            ligand_edge_pert,
            ligand_edge_index,
            ligand_edge_batch,
            time_step,
            batch_lab,
        )
        # The reference DiffGui classifier_free branch creates a zero label but
        # calls the unconditional forward with batch_lab again.  Reusing the same
        # tensor is numerically equivalent for this checkpoint and avoids a
        # second DDP forward for every denoising step.
        preds_uncond = preds_cond
        pred_eps_cond = core_model._predict_eps_from_x0(
            xt=ligand_pos_pert,
            t=time_step,
            pred_x0=preds_cond["pred_ligand_pos"],
            batch=ligand_batch,
        )
        pred_eps_uncond = core_model._predict_eps_from_x0(
            xt=ligand_pos_pert,
            t=time_step,
            pred_x0=preds_uncond["pred_ligand_pos"],
            batch=ligand_batch,
        )
        pred_eps = (1 + float(gui_strength)) * pred_eps_cond - float(gui_strength) * pred_eps_uncond
        pred_pos = core_model._predict_x0_from_eps(
            xt=ligand_pos_pert,
            t=time_step,
            eps=pred_eps,
            batch=ligand_batch,
        )
        pred_node = preds_cond["pred_ligand_node"] + preds_uncond["pred_ligand_node"]
        pred_halfedge = preds_cond["pred_ligand_halfedge"] + preds_uncond["pred_ligand_halfedge"]
        return pred_pos, pred_node, pred_halfedge
    if train_mode in ("no_lab", "no_both"):
        preds = model(
            protein_node,
            protein_pos,
            protein_batch,
            ligand_node_pert,
            ligand_pos_pert,
            ligand_batch,
            ligand_edge_pert,
            ligand_edge_index,
            ligand_edge_batch,
            time_step,
            batch_lab,
        )
        return preds["pred_ligand_pos"], preds["pred_ligand_node"], preds["pred_ligand_halfedge"]
    raise ValueError(f"Unsupported DiffGui train_mode={train_mode!r}")


def _core_model(model: Any) -> Any:
    """Return the bare DiffGui module from optional DDP/DataParallel wrappers."""
    return getattr(model, "module", model)


def _extract_transition_value(coef: torch.Tensor, t: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Match DiffGui ``extract(..., ndim=2)`` for [items, features] tensors."""
    return coef[t][batch].unsqueeze(-1)


def _sample_categorical_from_log_probs(log_probs: torch.Tensor) -> torch.Tensor:
    """Sample categorical indices using DiffGui's Gumbel argmax trick."""
    uniform = torch.rand_like(log_probs)
    gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
    return (gumbel_noise + log_probs).argmax(dim=-1)


def _as_type_index(sample: torch.Tensor) -> torch.Tensor:
    """Convert one-hot or index samples to class indices."""
    if sample.ndim >= 2:
        return sample.argmax(dim=-1).long()
    return sample.long()


def _reduce_by_graph(
    values: torch.Tensor,
    *,
    batch: torch.Tensor,
    num_graphs: int,
    reduction: str,
) -> torch.Tensor:
    """Reduce token-level values into graph-level log-probabilities."""
    if reduction not in {"mean", "sum"}:
        raise ValueError(f"Unsupported reduction={reduction!r}; expected 'mean' or 'sum'.")
    out = torch.zeros(num_graphs, dtype=values.dtype, device=values.device)
    out.index_add_(0, batch.long(), values)
    if reduction == "sum":
        return out
    counts = torch.zeros(num_graphs, dtype=values.dtype, device=values.device)
    counts.index_add_(0, batch.long(), torch.ones_like(values, dtype=values.dtype))
    return out / torch.clamp(counts, min=1.0)


def _center_pos(
    protein_pos: torch.Tensor,
    ligand_pos: torch.Tensor,
    protein_batch: torch.Tensor,
    ligand_batch: torch.Tensor,
    *,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Local copy of DiffGui center_pos for denovo sampling."""
    if mode == "none":
        max_graph = int(torch.max(torch.cat([protein_batch, ligand_batch])).item()) + 1
        offset = torch.zeros(max_graph, 3, dtype=protein_pos.dtype, device=protein_pos.device)
        return protein_pos, ligand_pos, offset
    if mode == "protein":
        offset = scatter_mean(protein_pos, protein_batch, dim=0)
        return protein_pos - offset[protein_batch], ligand_pos - offset[ligand_batch], offset
    raise ValueError(f"Unsupported DiffGui center_pos_mode={mode!r}")


def _require_discrete_diffgui(model: Any) -> None:
    """Reject unsupported DiffGui categorical modes early."""
    categorical_space = getattr(model, "categorical_space", None)
    if categorical_space != "discrete":
        raise ValueError(
            "DiffGui stage-2 adapter currently supports categorical_space='discrete' only, "
            f"got {categorical_space!r}."
        )
