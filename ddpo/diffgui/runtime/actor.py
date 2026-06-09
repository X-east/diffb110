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
    pred_node: torch.Tensor | None = None
    pred_pos: torch.Tensor | None = None
    pred_halfedge: torch.Tensor | None = None
    batch_lab: torch.Tensor | None = None
    gui_strength: float | None = None

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
        gui_strength: float | None = None,
    ) -> "DiffGuiPolicyBatch":
        """Build a policy batch from ``diffgui_sample_with_logprob`` outputs."""
        policy_traj = outputs["policy_traj"]
        logprob = outputs["logprob"]
        pred = outputs.get("pred")
        pred_node = pred[0] if pred is not None else None
        pred_pos = pred[1] if pred is not None else None
        pred_halfedge = pred[2] if pred is not None else None
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
            pred_node=pred_node,
            pred_pos=pred_pos,
            pred_halfedge=pred_halfedge,
            batch_lab=batch_lab,
            gui_strength=gui_strength,
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
        if self.pred_node is not None and int(self.pred_node.shape[0]) != int(self.ligand_batch.shape[0]):
            raise ValueError("pred_node first dimension must match ligand_batch length.")
        if self.pred_pos is not None and int(self.pred_pos.shape[0]) != int(self.ligand_batch.shape[0]):
            raise ValueError("pred_pos first dimension must match ligand_batch length.")
        if self.pred_halfedge is not None and int(self.pred_halfedge.shape[0]) != int(self.halfedge_batch.shape[0]):
            raise ValueError("pred_halfedge first dimension must match halfedge_batch length.")

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
            pred_node=None if self.pred_node is None else self.pred_node.to(target),
            pred_pos=None if self.pred_pos is None else self.pred_pos.to(target),
            pred_halfedge=None if self.pred_halfedge is None else self.pred_halfedge.to(target),
            batch_lab=None if self.batch_lab is None else self.batch_lab.to(target),
            gui_strength=self.gui_strength,
        )

    def select_graphs(self, graph_indices: list[int] | torch.Tensor) -> "DiffGuiPolicyBatch":
        """Return a graph-sliced policy batch with node/edge indices re-based."""
        device = self.advantages.device
        if isinstance(graph_indices, torch.Tensor):
            graph_ids = graph_indices.to(device=device, dtype=torch.long).flatten()
        else:
            graph_ids = torch.tensor([int(index) for index in graph_indices], dtype=torch.long, device=device)
        if graph_ids.numel() == 0:
            raise ValueError("graph_indices must not be empty.")

        node_mask = _mask_for_graph_ids(self.ligand_batch, graph_ids)
        halfedge_mask = _mask_for_graph_ids(self.halfedge_batch, graph_ids)
        protein_mask = _mask_for_graph_ids(self.protein_batch, graph_ids)
        node_indices = torch.nonzero(node_mask, as_tuple=False).flatten()
        halfedge_indices = torch.nonzero(halfedge_mask, as_tuple=False).flatten()
        protein_indices = torch.nonzero(protein_mask, as_tuple=False).flatten()

        node_map = torch.full((self.ligand_batch.shape[0],), -1, dtype=torch.long, device=device)
        node_map[node_indices] = torch.arange(node_indices.shape[0], dtype=torch.long, device=device)
        rebased_halfedge_index = node_map[self.halfedge_index[:, halfedge_indices]]
        if (rebased_halfedge_index < 0).any():
            raise ValueError("Selected halfedges reference nodes outside graph_indices.")

        return DiffGuiPolicyBatch(
            advantages=self.advantages[graph_ids],
            protein_node=self.protein_node[protein_indices],
            protein_pos=self.protein_pos[protein_indices],
            protein_batch=_remap_graph_ids(self.protein_batch[protein_indices], graph_ids),
            ligand_batch=_remap_graph_ids(self.ligand_batch[node_indices], graph_ids),
            halfedge_index=rebased_halfedge_index,
            halfedge_batch=_remap_graph_ids(self.halfedge_batch[halfedge_indices], graph_ids),
            node_traj=self.node_traj[:, node_indices],
            pos_centered_traj=self.pos_centered_traj[:, node_indices],
            halfedge_traj=self.halfedge_traj[:, halfedge_indices],
            node_log_state_traj=self.node_log_state_traj[:, node_indices],
            halfedge_log_state_traj=self.halfedge_log_state_traj[:, halfedge_indices],
            old_logprob=self.old_logprob[:, graph_ids],
            timesteps=self.timesteps[:, graph_ids],
            pred_node=None if self.pred_node is None else self.pred_node[node_indices],
            pred_pos=None if self.pred_pos is None else self.pred_pos[node_indices],
            pred_halfedge=None if self.pred_halfedge is None else self.pred_halfedge[halfedge_indices],
            batch_lab=None if self.batch_lab is None else self.batch_lab[graph_ids],
            gui_strength=self.gui_strength,
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
        self.precision = str(train_cfg.get("precision", "fp32")).lower()
        self.amp_dtype = self._resolve_amp_dtype(self.precision)
        self.graph_micro_batch_size = int(train_cfg.get("graph_micro_batch_size", 0))
        if self.graph_micro_batch_size < 0:
            raise ValueError("train.graph_micro_batch_size must be non-negative.")
        self.freeze_report = self._apply_freeze_policy(self.model, train_cfg.get("freeze", {}))
        self._optimizer_params = [param for param in self.model.parameters() if param.requires_grad]
        if not self._optimizer_params:
            raise RuntimeError("DiffGuiActor model has no trainable parameters.")
        logger.info("DiffGuiActor freeze_report=%s", self.freeze_report)
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
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=self.amp_dtype is torch.float16)

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
            metrics = self._backward_for_batch(policy_batch)
            if self.grad_scaler.is_enabled():
                self.grad_scaler.unscale_(self.optimizer)
            grad_norm = clip_grad_norm_(self._optimizer_params, max_norm=self.max_grad_norm)
            if self.grad_scaler.is_enabled():
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                self.optimizer.step()
            metrics["GradNorm"] = float(_to_scalar(grad_norm))
            metrics["PPO_Epoch"] = float(ppo_epoch)
            metrics["TrainableParams"] = float(self.freeze_report["trainable_params"])
            metrics["FrozenParams"] = float(self.freeze_report["frozen_params"])
            epoch_metrics.append(metrics)
        return self._merge_metrics(epoch_metrics)

    def _backward_for_batch(self, batch: DiffGuiPolicyBatch) -> dict[str, float]:
        """Backpropagate PPO loss, optionally streaming graph microbatches."""
        total_graphs = int(batch.advantages.shape[0])
        if 0 < self.graph_micro_batch_size < total_graphs:
            rows: list[dict[str, float]] = []
            for start in range(0, total_graphs, self.graph_micro_batch_size):
                end = min(start + self.graph_micro_batch_size, total_graphs)
                sub_batch = batch.select_graphs(list(range(start, end)))
                rows.append(
                    self._backward_for_batch_steps(
                        sub_batch,
                        graph_loss_weight=float(end - start) / float(total_graphs),
                    )
                )
            merged = self._merge_microbatch_metrics(rows)
            merged["GraphMicroBatchSize"] = float(self.graph_micro_batch_size)
            return merged
        metrics = self._backward_for_batch_steps(batch, graph_loss_weight=1.0)
        metrics["GraphMicroBatchSize"] = float(total_graphs)
        return metrics

    def _backward_for_batch_steps(self, batch: DiffGuiPolicyBatch, *, graph_loss_weight: float) -> dict[str, float]:
        """Backpropagate PPO loss step-by-step for one graph microbatch."""
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
        amp_enabled = self._amp_enabled()
        for step_idx in range(num_steps):
            with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=self.amp_dtype or torch.float16):
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
                    gui_strength=batch.gui_strength,
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
                step_loss = -torch.min(unclipped, clipped).mean()
            scaled_step_loss = step_loss * float(graph_loss_weight) / float(num_steps)
            if self.grad_scaler.is_enabled():
                self.grad_scaler.scale(scaled_step_loss).backward()
            else:
                scaled_step_loss.backward()
            losses.append(step_loss.detach())
            clipfracs.append((torch.abs(ratio - 1.0) > self.clip_range).float().mean().detach())
            approx_kls.append((0.5 * logratio.pow(2)).mean().detach())
            ratio_means.append(ratio.detach().mean())
            logprob_current_means.append(result.logprob_total.detach().mean())
            logprob_old_means.append(logprob_old.mean())
            component_means["LogProbPos"].append(result.logprob_pos.detach().mean())
            component_means["LogProbNode"].append(result.logprob_node.detach().mean())
            component_means["LogProbHalfedge"].append(result.logprob_halfedge.detach().mean())
            del result, step_loss, logratio, ratio, unclipped, clipped

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
            "AMPEnabled": float(amp_enabled),
        }
        for key, values in component_means.items():
            metrics[key] = _mean_metric(values)
        return metrics

    def _training_device(self) -> torch.device:
        """Return the device of trainable model parameters."""
        return self._optimizer_params[0].device

    @staticmethod
    def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
        """Return the bare DiffGui module from optional DDP/DataParallel wrappers."""
        return getattr(model, "module", model)

    @staticmethod
    def _merge_microbatch_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
        """Merge graph microbatch metrics weighted by graph count."""
        if not rows:
            return {}
        total = float(sum(row.get("TrainBatchSize", 0.0) for row in rows))
        if total <= 0.0:
            total = float(len(rows))
        merged: dict[str, float] = {}
        keys = sorted({key for row in rows for key in row})
        for key in keys:
            if key == "TrainBatchSize":
                merged[key] = total
                continue
            values = [row[key] * row.get("TrainBatchSize", 1.0) for row in rows if key in row]
            merged[key] = float(sum(values) / total) if values else 0.0
        return merged

    def _amp_enabled(self) -> bool:
        """Return whether CUDA autocast should be used for PPO replay."""
        return self.amp_dtype is not None and self._training_device().type == "cuda"

    @staticmethod
    def _resolve_amp_dtype(precision: str) -> torch.dtype | None:
        """Resolve a precision string to an AMP dtype."""
        normalized = precision.lower()
        if normalized in ("fp32", "float32", "32", "none", "off"):
            return None
        if normalized in ("fp16", "float16", "16"):
            return torch.float16
        if normalized in ("bf16", "bfloat16"):
            return torch.bfloat16
        raise ValueError("train.precision must be one of fp32, fp16, or bf16.")

    @classmethod
    def _apply_freeze_policy(cls, model: torch.nn.Module, freeze_cfg: Any) -> dict[str, Any]:
        """Apply a DiffGui-specific parameter freeze policy before optimizer setup."""
        core_model = cls._unwrap_model(model)
        cfg = freeze_cfg if isinstance(freeze_cfg, dict) else {}
        mode = str(cfg.get("mode", "none")).lower()
        if mode in ("none", "off", "false"):
            return cls._freeze_report(core_model, mode="none", train_last_n_blocks=0)

        for parameter in core_model.parameters():
            parameter.requires_grad_(False)

        train_last_n_blocks = int(cfg.get("train_last_n_blocks", cfg.get("last_n_blocks", 1)))
        if mode in ("heads_only", "head", "heads"):
            cls._enable_named_module(core_model, "ligand_node_decoder")
            cls._enable_named_module(core_model, "ligand_edge_decoder")
            train_last_n_blocks = 0
        elif mode in ("last_n_blocks", "last_blocks"):
            if train_last_n_blocks <= 0:
                raise ValueError("freeze.train_last_n_blocks must be positive for last_n_blocks mode.")
            cls._enable_named_module(core_model, "ligand_node_decoder")
            cls._enable_named_module(core_model, "ligand_edge_decoder")
            cls._enable_last_denoiser_blocks(core_model, train_last_n_blocks)
        else:
            raise ValueError("train.freeze.mode must be one of none, heads_only, or last_n_blocks.")
        return cls._freeze_report(core_model, mode=mode, train_last_n_blocks=train_last_n_blocks)

    @staticmethod
    def _enable_named_module(model: torch.nn.Module, module_name: str) -> None:
        """Enable gradients for a direct child module if it exists."""
        module = getattr(model, module_name, None)
        if module is None:
            return
        for parameter in module.parameters():
            parameter.requires_grad_(True)

    @classmethod
    def _enable_last_denoiser_blocks(cls, model: torch.nn.Module, train_last_n_blocks: int) -> None:
        """Enable gradients for the last N DiffGui EGNN block groups."""
        denoiser = getattr(model, "denoiser", None)
        if denoiser is None:
            return
        for list_name in ("node_blocks_with_edge", "edge_embs", "edge_blocks", "pos_blocks"):
            modules = getattr(denoiser, list_name, None)
            if modules is None:
                continue
            start = max(0, len(modules) - int(train_last_n_blocks))
            for module in list(modules)[start:]:
                cls._enable_module(module)

    @staticmethod
    def _enable_module(module: torch.nn.Module) -> None:
        """Enable gradients for all parameters in a module."""
        for parameter in module.parameters():
            parameter.requires_grad_(True)

    @staticmethod
    def _freeze_report(model: torch.nn.Module, *, mode: str, train_last_n_blocks: int) -> dict[str, Any]:
        """Summarize trainable and frozen parameter counts."""
        total_params = sum(parameter.numel() for parameter in model.parameters())
        trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        return {
            "mode": mode,
            "train_last_n_blocks": int(train_last_n_blocks),
            "total_params": int(total_params),
            "trainable_params": int(trainable_params),
            "frozen_params": int(total_params - trainable_params),
            "trainable_ratio": float(trainable_params / total_params) if total_params else 0.0,
        }

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


def _mask_for_graph_ids(values: torch.Tensor, graph_ids: torch.Tensor) -> torch.Tensor:
    """Return a boolean mask selecting entries whose graph id is listed."""
    mask = torch.zeros(values.shape, dtype=torch.bool, device=values.device)
    for graph_id in graph_ids:
        mask = mask | (values == graph_id)
    return mask


def _remap_graph_ids(values: torch.Tensor, graph_ids: torch.Tensor) -> torch.Tensor:
    """Map original graph ids to a compact 0..B-1 range."""
    remapped = torch.empty_like(values)
    for new_id, old_id in enumerate(graph_ids):
        remapped[values == old_id] = int(new_id)
    return remapped
