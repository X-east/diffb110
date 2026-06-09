"""Rollout helpers for the decoupled DiffGui PPO path.

This module stays model-agnostic: it expects a DiffGui-like model exposing the
same ``pos/node/edge`` transition attributes as the reference implementation,
but it does not import the reference project or DiffSBDD runtime modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ddpo.diffgui.runtime.actor import DiffGuiPolicyBatch
from ddpo.diffgui.runtime.logprob_adapter import diffgui_sample_with_logprob
from ddpo.diffgui.runtime.timestep_config import configure_diffgui_model_timesteps


@dataclass
class DiffGuiDataBridge:
    """Minimal DataProto-compatible bridge for DiffGui actor updates."""

    batch: dict[str, torch.Tensor]
    meta_info: dict[str, Any]


@dataclass(frozen=True)
class DiffGuiPromptBatch:
    """Protein condition and ligand placeholder tensors for DiffGui sampling."""

    protein_node: torch.Tensor
    protein_pos: torch.Tensor
    protein_batch: torch.Tensor
    ligand_batch: torch.Tensor
    halfedge_index: torch.Tensor
    halfedge_batch: torch.Tensor
    batch_lab: torch.Tensor | None = None

    @property
    def n_graphs(self) -> int:
        """Return the number of ligand graphs represented by this prompt."""
        if self.ligand_batch.numel() == 0:
            raise ValueError("DiffGuiPromptBatch.ligand_batch must not be empty.")
        return int(self.ligand_batch.max().detach().cpu().item()) + 1

    def node_counts(self) -> torch.Tensor:
        """Return per-graph ligand node counts."""
        return torch.bincount(self.ligand_batch.long(), minlength=self.n_graphs)

    def validate(self) -> None:
        """Validate shape consistency before rollout."""
        if self.protein_node.ndim != 2:
            raise ValueError("protein_node must have shape [N_protein, F].")
        if self.protein_pos.shape != (self.protein_node.shape[0], 3):
            raise ValueError("protein_pos must have shape [N_protein, 3].")
        if self.protein_batch.shape[0] != self.protein_node.shape[0]:
            raise ValueError("protein_batch length must match protein_node.")
        if self.halfedge_index.ndim != 2 or self.halfedge_index.shape[0] != 2:
            raise ValueError("halfedge_index must have shape [2, E].")
        if self.halfedge_batch.shape[0] != self.halfedge_index.shape[1]:
            raise ValueError("halfedge_batch length must match halfedge_index columns.")
        if self.batch_lab is not None and self.batch_lab.shape[0] != self.n_graphs:
            raise ValueError("batch_lab first dimension must match n_graphs.")

    def to(self, device: torch.device | str) -> "DiffGuiPromptBatch":
        """Move prompt tensors to a device and return a new prompt."""
        target = torch.device(device)
        return DiffGuiPromptBatch(
            protein_node=self.protein_node.to(target),
            protein_pos=self.protein_pos.to(target),
            protein_batch=self.protein_batch.to(target),
            ligand_batch=self.ligand_batch.to(target),
            halfedge_index=self.halfedge_index.to(target),
            halfedge_batch=self.halfedge_batch.to(target),
            batch_lab=None if self.batch_lab is None else self.batch_lab.to(target),
        )


class DiffGuiRollout:
    """Generate DiffGui policy batches with old-policy log-probabilities."""

    def __init__(self, model: torch.nn.Module, config: dict[str, Any]) -> None:
        """Initialize rollout settings from the DiffGui-specific config."""
        self.model = model
        rollout_cfg = config.get("diffgui_rollout", {})
        self.num_timesteps = int(rollout_cfg.get("num_timesteps", getattr(model, "num_timesteps", 0)))
        model_timesteps = int(getattr(model, "num_timesteps", 0))
        if self.num_timesteps <= 0:
            raise ValueError("diffgui_rollout.num_timesteps must be positive.")
        if model_timesteps != self.num_timesteps:
            if bool(rollout_cfg.get("auto_configure_timesteps", True)):
                self.timestep_config_report = configure_diffgui_model_timesteps(model, self.num_timesteps)
                model_timesteps = int(getattr(model, "num_timesteps", 0))
            else:
                raise ValueError(
                    "diffgui_rollout.num_timesteps must match model.num_timesteps. "
                    "Set diffgui_rollout.auto_configure_timesteps=true to regenerate "
                    "DiffGui timestep-dependent buffers after checkpoint loading."
                )
        else:
            self.timestep_config_report = {
                "changed": False,
                "num_timesteps": self.num_timesteps,
                "reason": "already_configured",
            }
        if model_timesteps != self.num_timesteps:
            raise RuntimeError(
                "DiffGui timestep auto-configuration did not update model.num_timesteps "
                f"to {self.num_timesteps}; got {model_timesteps}."
            )
        self.reduction = str(rollout_cfg.get("logprob_reduction", "mean"))
        self.gui_strength = rollout_cfg.get("gui_strength")
        train_mode = str(getattr(getattr(model, "config", None), "train_mode", ""))
        if train_mode in ("ori", "no_bond") and self.gui_strength is None:
            raise ValueError(
                "DiffGui train_mode requires diffgui_rollout.gui_strength for classifier-free prediction; "
                "set it to 0.0 to disable guidance while keeping rollout and PPO logprob aligned."
            )
        self.cpu_offload = bool(rollout_cfg.get("cpu_offload", True))
        self.advantage_mode = str(rollout_cfg.get("synthetic_advantage_mode", "ones"))
        self.advantage_scale = float(rollout_cfg.get("synthetic_advantage_scale", 1.0))
        if rollout_cfg.get("guidance") is not None:
            raise ValueError("DiffGui PPO rollout must keep guidance disabled for logprob consistency.")

    def generate_policy_batch(
        self,
        prompt: DiffGuiPromptBatch,
        *,
        advantages: torch.Tensor | None = None,
    ) -> DiffGuiPolicyBatch:
        """Sample a DiffGui trajectory and return the actor-ready policy batch."""
        prompt.validate()
        device = self._model_device()
        prompt_on_device = prompt.to(device)
        outputs = diffgui_sample_with_logprob(
            self.model,
            n_graphs=prompt_on_device.n_graphs,
            protein_node=prompt_on_device.protein_node,
            protein_pos=prompt_on_device.protein_pos,
            protein_batch=prompt_on_device.protein_batch,
            ligand_batch=prompt_on_device.ligand_batch,
            halfedge_index=prompt_on_device.halfedge_index,
            halfedge_batch=prompt_on_device.halfedge_batch,
            batch_lab=prompt_on_device.batch_lab,
            gui_strength=self.gui_strength,
            reduction=self.reduction,
        )
        if advantages is None:
            advantages = self._synthetic_advantages(prompt_on_device.n_graphs, device=device)
        else:
            advantages = advantages.to(device=device, dtype=prompt_on_device.protein_pos.dtype)
        batch = DiffGuiPolicyBatch.from_sample_outputs(
            outputs=outputs,
            advantages=advantages,
            protein_node=prompt_on_device.protein_node,
            protein_pos=prompt_on_device.protein_pos,
            protein_batch=prompt_on_device.protein_batch,
            ligand_batch=prompt_on_device.ligand_batch,
            halfedge_index=prompt_on_device.halfedge_index,
            halfedge_batch=prompt_on_device.halfedge_batch,
            batch_lab=prompt_on_device.batch_lab,
            gui_strength=self.gui_strength,
        )
        batch.validate()
        return batch.to("cpu") if self.cpu_offload else batch

    def generate_samples(
        self,
        prompt: DiffGuiPromptBatch,
        *,
        advantages: torch.Tensor | None = None,
    ) -> DiffGuiDataBridge:
        """Return a lightweight bridge carrying a DiffGuiPolicyBatch in metadata."""
        policy_batch = self.generate_policy_batch(prompt, advantages=advantages)
        node_counts = prompt.node_counts().detach().cpu()
        bridge_batch = {
            "advantages": policy_batch.advantages.detach().cpu(),
            "reward_total": policy_batch.advantages.detach().cpu().clone(),
            "nodesxsample": node_counts,
        }
        return DiffGuiDataBridge(
            batch=bridge_batch,
            meta_info={
                "diffgui_policy_batch": policy_batch,
                "diffgui_policy_batch_kind": "xhe_joint",
                "diffgui_num_timesteps": self.num_timesteps,
                "diffgui_timestep_config_report": self.timestep_config_report,
            },
        )

    def _model_device(self) -> torch.device:
        """Infer the model device from its parameters."""
        try:
            return next(self.model.parameters()).device
        except StopIteration as exc:
            raise RuntimeError("DiffGui rollout model has no parameters.") from exc

    def _synthetic_advantages(self, n_graphs: int, *, device: torch.device) -> torch.Tensor:
        """Build deterministic synthetic advantages for rollout/actor smoke tests."""
        if self.advantage_mode == "ones":
            values = torch.ones(n_graphs, dtype=torch.float32, device=device)
        elif self.advantage_mode == "linear":
            values = torch.linspace(0.5, 1.0, steps=n_graphs, dtype=torch.float32, device=device)
        elif self.advantage_mode == "zeros":
            values = torch.zeros(n_graphs, dtype=torch.float32, device=device)
        else:
            raise ValueError(
                f"Unsupported synthetic_advantage_mode={self.advantage_mode!r}; "
                "expected 'ones', 'linear', or 'zeros'."
            )
        return values * self.advantage_scale


def make_diffgui_ligand_placeholder(
    nodes_per_graph: list[int] | torch.Tensor,
    *,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build DiffGui ligand batch, halfedge index, and halfedge batch tensors."""
    if isinstance(nodes_per_graph, torch.Tensor):
        counts = [int(value) for value in nodes_per_graph.detach().cpu().tolist()]
    else:
        counts = [int(value) for value in nodes_per_graph]
    if not counts:
        raise ValueError("nodes_per_graph must not be empty.")
    if any(count <= 0 for count in counts):
        raise ValueError("All nodes_per_graph values must be positive.")

    target = None if device is None else torch.device(device)
    ligand_batch_parts: list[torch.Tensor] = []
    halfedge_parts: list[torch.Tensor] = []
    halfedge_batch_parts: list[torch.Tensor] = []
    node_offset = 0
    for graph_idx, count in enumerate(counts):
        ligand_batch_parts.append(torch.full((count,), graph_idx, dtype=torch.long))
        halfedges = torch.triu_indices(count, count, offset=1, dtype=torch.long)
        if halfedges.numel() > 0:
            halfedge_parts.append(halfedges + node_offset)
            halfedge_batch_parts.append(torch.full((halfedges.shape[1],), graph_idx, dtype=torch.long))
        node_offset += count

    ligand_batch = torch.cat(ligand_batch_parts, dim=0)
    if halfedge_parts:
        halfedge_index = torch.cat(halfedge_parts, dim=1)
        halfedge_batch = torch.cat(halfedge_batch_parts, dim=0)
    else:
        halfedge_index = torch.empty((2, 0), dtype=torch.long)
        halfedge_batch = torch.empty((0,), dtype=torch.long)

    if target is not None:
        ligand_batch = ligand_batch.to(target)
        halfedge_index = halfedge_index.to(target)
        halfedge_batch = halfedge_batch.to(target)
    return ligand_batch, halfedge_index, halfedge_batch
