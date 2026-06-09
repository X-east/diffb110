"""PPO actor for the decoupled DiffGui RL path.

This actor intentionally does not reuse ``ddpo.runtime.actor.EDMActor`` because
DiffGui's policy transition has three coupled state families:
coordinates, node types, and halfedge types.  The actor consumes a
``DiffGuiPolicyBatch`` produced by the DiffGui rollout path and computes the
standard PPO clipped objective from per-step transition log-probabilities.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from ddpo.diffgui.runtime.logprob_adapter import diffgui_forward_step_logprob


logger = logging.getLogger(__name__)


@dataclass
class DiffGuiPolicyBatch:
    """DiffGui rollout tensors needed by PPO.

    All trajectory tensors use the same step order as
    ``diffgui_sample_with_logprob``: transition ``j`` maps
    ``traj[j] -> traj[j + 1]`` with integer timestep ``timesteps[j]``.
    """

    advantages: torch.Tensor
    protein_node: torch.Tensor
    protein_pos: torch.Tensor
    protein_batch: torch.Tensor
    ligand_batch: torch.Tensor
    halfedge_index: torch.Tensor
    halfedge_batch: torch.Tensor
    node_traj: torch.Tensor
    pos_centered_traj: torch.Tensor
    halfedge_traj: torch.Tensor
    node_log_state_traj: torch.Tensor
    halfedge_log_state_traj: torch.Tensor
    old_logprob: torch.Tensor
    timesteps: torch.Tensor
    batch_lab: torch.Tensor | None = None

    @classmethod
    def from_sample_outputs(
        cls,
        *,
        outputs: dict[str, Any],
        advantages: torch.Tensor,
        protein_node: torch.Tensor,
        protein_pos: torch.Tensor,
        protein_batch: torch.Tensor,
        ligand_batch: torch.Tensor,
        halfedge_index: torch.Tensor,
        halfedge_batch: torch.Tensor,
        batch_lab: torch.Tensor | None = None,
    ) -> "DiffGuiPolicyBatch":
        """Build a policy batch from ``diffgui_sample_with_logprob`` outputs."""
        policy_traj = outputs["policy_traj"]
        logprob = outputs["logprob"]
        return cls(
            advantages=advantages,
            protein_node=protein_node,
            protein_pos=protein_pos,
            protein_batch=protein_batch,
            ligand_batch=ligand_batch,
            halfedge_index=halfedge_index,
            halfedge_batch=halfedge_batch,
            node_traj=policy_traj["node"],
            pos_centered_traj=policy_traj["pos_centered"],
            halfedge_traj=policy_traj["halfedge"],
            node_log_state_traj=policy_traj["node_log_state"],
            halfedge_log_state_traj=policy_traj["halfedge_log_state"],
            old_logprob=logprob["total"],
            timesteps=logprob["timesteps"],
            batch_lab=batch_lab,
        )

    def validate(self) -> None:
        """Validate tensor ranks and leading dimensions before PPO update."""
        num_graphs = int(self.advantages.shape[0])
        num_steps = int(self.old_logprob.shape[0])
        if self.old_logprob.shape != self.timesteps.shape:
            raise ValueError("old_logprob and timesteps must have the same shape [S, B].")
        if int(self.old_logprob.shape[1]) != num_graphs:
            raise ValueError("old_logprob second dimension must match advantages length.")
        for name, tensor in {
            "node_traj": self.node_traj,
            "pos_centered_traj": self.pos_centered_traj,
            "halfedge_traj": self.halfedge_traj,
            "node_log_state_traj": self.node_log_state_traj,
            "halfedge_log_state_traj": self.halfedge_log_state_traj,
        }.items():
            if int(tensor.shape[0]) != num_steps + 1:
                raise ValueError(f"{name} first dimension must be num_steps + 1.")
        if int(self.timesteps.shape[1]) != num_graphs:
            raise ValueError("timesteps second dimension must match advantages length.")

    def to(self, device: torch.device | str) -> "DiffGuiPolicyBatch":
        """Move all tensor fields to a device and return a new batch."""
        target = torch.device(device)
        return DiffGuiPolicyBatch(
            advantages=self.advantages.to(target),
            protein_node=self.protein_node.to(target),
            protein_pos=self.protein_pos.to(target),
            protein_batch=self.protein_batch.to(target),
            ligand_batch=self.ligand_batch.to(target),
            halfedge_index=self.halfedge_index.to(target),
            halfedge_batch=self.halfedge_batch.to(target),
            node_traj=self.node_traj.to(target),
            pos_centered_traj=self.pos_centered_traj.to(target),
            halfedge_traj=self.halfedge_traj.to(target),
            node_log_state_traj=self.node_log_state_traj.to(target),
            halfedge_log_state_traj=self.halfedge_log_state_traj.to(target),
            old_logprob=self.old_logprob.to(target),
            timesteps=self.timesteps.to(target),
            batch_lab=None if self.batch_lab is None else self.batch_lab.to(target),
        )


class DiffGuiActor:
    """PPO actor for DiffGui transition batches."""

    def __init__(self, model: torch.nn.Module, config: dict[str, Any]) -> None:
        """Initialize the actor with an optimizer and PPO hyperparameters."""
        super().__init__()
        self.model = model
        self.config = config
        train_cfg = config.get("train", {})
        self.clip_range = float(train_cfg.get("clip_range", 0.1))
        self.max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
        self.num_ppo_epochs = max(1, int(train_cfg.get("ppo_epochs", 1)))
        self.reduction = str(train_cfg.get("diffgui_logprob_reduction", "mean"))
        self._optimizer_params = [param for param in self.model.parameters() if param.requires_grad]
        if not self._optimizer_params:
            raise RuntimeError("DiffGuiActor model has no trainable parameters.")
        self.optimizer = torch.optim.AdamW(
            self._optimizer_params,
            lr=float(train_cfg.get("learning_rate", 1e-5)),
            betas=(
                float(train_cfg.get("adam_beta1", 0.9)),
                float(train_cfg.get("adam_beta2", 0.999)),
            ),
            weight_decay=float(train_cfg.get("adam_weight_decay", 0.0)),
            eps=float(train_cfg.get("adam_epsilon", 1e-8)),
        )

    def update_policy(self, data: Any | DiffGuiPolicyBatch) -> dict[str, float]:
        """Update DiffGui policy from a rollout batch using PPO clipping."""
        policy_batch = self._policy_batch_from_input(data)
        policy_batch.validate()
        device = self._training_device()
        policy_batch = policy_batch.to(device)
        self.model.train()

        epoch_metrics: list[dict[str, float]] = []
        for ppo_epoch in range(self.num_ppo_epochs):
            self.optimizer.zero_grad(set_to_none=True)
            loss, metrics = self._loss_for_batch(policy_batch)
            loss.backward()
            grad_norm = clip_grad_norm_(self._optimizer_params, max_norm=self.max_grad_norm)
            self.optimizer.step()
            metrics["GradNorm"] = float(_to_scalar(grad_norm))
            metrics["PPO_Epoch"] = float(ppo_epoch)
            epoch_metrics.append(metrics)
        return self._merge_metrics(epoch_metrics)

    def _loss_for_batch(self, batch: DiffGuiPolicyBatch) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute accumulated PPO loss across all DiffGui reverse steps."""
        losses: list[torch.Tensor] = []
        clipfracs: list[torch.Tensor] = []
        approx_kls: list[torch.Tensor] = []
        ratio_means: list[torch.Tensor] = []
        logprob_current_means: list[torch.Tensor] = []
        logprob_old_means: list[torch.Tensor] = []
        component_means: dict[str, list[torch.Tensor]] = {
            "LogProbPos": [],
            "LogProbNode": [],
            "LogProbHalfedge": [],
        }

        num_steps = int(batch.old_logprob.shape[0])
        for step_idx in range(num_steps):
            result = diffgui_forward_step_logprob(
                self.model,
                protein_node=batch.protein_node,
                protein_pos=batch.protein_pos,
                protein_batch=batch.protein_batch,
                ligand_node_t=batch.node_traj[step_idx],
                ligand_pos_t=batch.pos_centered_traj[step_idx],
                ligand_batch=batch.ligand_batch,
                ligand_halfedge_t=batch.halfedge_traj[step_idx],
                halfedge_index=batch.halfedge_index,
                halfedge_batch=batch.halfedge_batch,
                node_log_state_t=batch.node_log_state_traj[step_idx],
                halfedge_log_state_t=batch.halfedge_log_state_traj[step_idx],
                ligand_pos_prev=batch.pos_centered_traj[step_idx + 1],
                ligand_node_prev=batch.node_traj[step_idx + 1],
                ligand_halfedge_prev=batch.halfedge_traj[step_idx + 1],
                time_step=batch.timesteps[step_idx],
                batch_lab=batch.batch_lab,
                reduction=self.reduction,
            )
            logprob_old = batch.old_logprob[step_idx].detach()
            logratio = result.logprob_total - logprob_old
            ratio = torch.exp(logratio)
            unclipped = batch.advantages * ratio
            clipped = batch.advantages * torch.clamp(
                ratio,
                min=1.0 - self.clip_range,
                max=1.0 + self.clip_range,
            )
            losses.append(-torch.min(unclipped, clipped).mean())
            clipfracs.append((torch.abs(ratio - 1.0) > self.clip_range).float().mean().detach())
            approx_kls.append((0.5 * logratio.pow(2)).mean().detach())
            ratio_means.append(ratio.detach().mean())
            logprob_current_means.append(result.logprob_total.detach().mean())
            logprob_old_means.append(logprob_old.mean())
            component_means["LogProbPos"].append(result.logprob_pos.detach().mean())
            component_means["LogProbNode"].append(result.logprob_node.detach().mean())
            component_means["LogProbHalfedge"].append(result.logprob_halfedge.detach().mean())

        loss = torch.stack(losses).mean()
        metrics = {
            "Loss": float(loss.detach().cpu().item()),
            "PPO_Loss": float(loss.detach().cpu().item()),
            "ClipFrac": _mean_metric(clipfracs),
            "PPO_ApproxKL": _mean_metric(approx_kls),
            "RatioMean": _mean_metric(ratio_means),
            "LogProbCurrentMean": _mean_metric(logprob_current_means),
            "LogProbOldMean": _mean_metric(logprob_old_means),
            "NumSteps": float(num_steps),
            "TrainBatchSize": float(batch.advantages.shape[0]),
        }
        for key, values in component_means.items():
            metrics[key] = _mean_metric(values)
        return loss, metrics

    def _training_device(self) -> torch.device:
        """Return the device of trainable model parameters."""
        return self._optimizer_params[0].device

    @staticmethod
    def _policy_batch_from_input(data: Any | DiffGuiPolicyBatch) -> DiffGuiPolicyBatch:
        """Extract a DiffGuiPolicyBatch from supported actor input types."""
        if isinstance(data, DiffGuiPolicyBatch):
            return data
        meta_info = getattr(data, "meta_info", None)
        if isinstance(meta_info, dict):
            candidate = meta_info.get("diffgui_policy_batch")
            if isinstance(candidate, DiffGuiPolicyBatch):
                return candidate
            raise ValueError("data.meta_info['diffgui_policy_batch'] must contain a DiffGuiPolicyBatch.")
        raise TypeError(f"Unsupported DiffGuiActor input type: {type(data)!r}")

    @staticmethod
    def _merge_metrics(epoch_metrics: list[dict[str, float]]) -> dict[str, float]:
        """Average scalar metrics across PPO epochs."""
        if not epoch_metrics:
            return {}
        keys = sorted({key for row in epoch_metrics for key in row})
        merged: dict[str, float] = {}
        for key in keys:
            if key == "PPO_Epoch":
                continue
            values = [row[key] for row in epoch_metrics if key in row]
            merged[key] = float(np.mean(np.asarray(values, dtype=np.float64)))
        merged["PPO_Epochs"] = float(len(epoch_metrics))
        return merged


def _mean_metric(values: list[torch.Tensor]) -> float:
    """Return a Python float mean for detached scalar tensors."""
    if not values:
        return 0.0
    stacked = torch.stack([value.detach().float().reshape(()) for value in values])
    return float(stacked.mean().cpu().item())


def _to_scalar(value: torch.Tensor | float) -> float:
    """Convert a scalar tensor or float to a Python float."""
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().cpu().item())
    return float(value)
