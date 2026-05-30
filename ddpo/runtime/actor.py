from __future__ import annotations

"""PPO actor used by the native DiffSBDD + DDPO runtime."""

import os
import random
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm as tq

from ddpo.runtime.logging_utils import log_cuda_memory
from ddpo.runtime.protocol import DataProto


class BaseActor(ABC):
    """Minimal local actor interface for DDPO policy updates."""

    @abstractmethod
    def update_policy(self, data: DataProto) -> Dict:
        """Update the policy from a rollout batch and return scalar metrics."""
        raise NotImplementedError


class EDMActor(BaseActor):
    def __init__(self, model, config, model_impl=None, ref_model=None):
        """
        Initialize the EDM rollout worker.

        Args:
            model: The training model reference used for optimizer parameters.
            config: Configuration parameters for the rollout
            model_impl: Optional bare model implementation used for custom
                diffusion helper methods such as get_mask() and
                sample_p_zs_given_zt(). If omitted, defaults to model.
            ref_model: Optional frozen reference model for KL penalty.
                When None or kl_coef <= 0, KL computation is fully disabled.
        """
        super().__init__()
        self._rank = int(os.environ.get("RANK", "0"))
        self.model = model
        self.model_impl = model_impl if model_impl is not None else self._unwrap_model(model)
        self.ref_model = ref_model
        self.kl_coef = float(config.get("train", {}).get("kl_coef", 0.0))
        self.config = config
        self.last_grad_probe = []
        optimizer_cls = torch.optim.AdamW
        self.optimizer = optimizer_cls(
            self.model.parameters(),
            lr=self.config["train"]["learning_rate"],
            betas=(self.config["train"]["adam_beta1"], self.config["train"]["adam_beta2"]),
            weight_decay=self.config["train"]["adam_weight_decay"],
            eps=self.config["train"]["adam_epsilon"],
        )
        self.train_micro_batch_size = self.config["train"]["train_micro_batch_size"]
        self.max_grad_norm = self.config["train"]["max_grad_norm"]
        self.clip_range = self.config["train"]["clip_range"]
        self.num_timesteps = self.config["model"]["time_step"]
        self.num_ppo_epochs = max(1, int(self.config.get("train", {}).get("ppo_epochs", 1)))
        self.ppo_shuffle_microbatch = bool(self.config.get("train", {}).get("ppo_shuffle_microbatch", False))

    @staticmethod
    def _unwrap_model(model):
        return getattr(model, "module", model)

    def set_model_impl(self, model_impl) -> None:
        self.model_impl = model_impl

    def _training_device(self) -> torch.device:
        """Return the device that owns trainable model parameters."""
        try:
            return next(self.model.parameters()).device
        except StopIteration as exc:
            raise RuntimeError("EDMActor model has no parameters; cannot infer training device") from exc

    def _loss_scale(self, batch_size: int) -> float:
        """Scale accumulated micro-batch gradients to a reference local batch."""
        train_cfg = self.config.get("train", {})
        explicit = train_cfg.get("loss_scale")
        if explicit is not None:
            return float(explicit)

        reference = train_cfg.get("gradient_reference_batch_size")
        if reference is None:
            return 1.0

        reference_value = float(reference)
        if reference_value <= 0:
            raise ValueError(f"train.gradient_reference_batch_size must be > 0, got {reference!r}")
        return min(1.0, reference_value / float(max(batch_size, 1)))

    def _microbatch_indices(self, batch_size: int) -> list[torch.Tensor]:
        """Build CPU index tensors for streaming training micro-batches."""
        micro_batch = max(1, int(self.train_micro_batch_size))
        return [
            torch.arange(start, min(start + micro_batch, batch_size), dtype=torch.long)
            for start in range(0, batch_size, micro_batch)
        ]

    @staticmethod
    def _take_to_device(tensor: torch.Tensor, indices: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Index a batch tensor and move only that slice to the training device."""
        index = indices.to(tensor.device) if tensor.device.type != "cpu" else indices
        return tensor[index].to(device, non_blocking=(tensor.device.type == "cpu" and device.type == "cuda"))

    def _training_microbatch(
        self,
        data: DataProto,
        indices: torch.Tensor,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Materialize one PPO training micro-batch on the target device."""
        batch = data.batch
        latents = self._take_to_device(batch["latents"], indices, device)
        sample = {
            "advantages": self._take_to_device(batch["advantages"], indices, device),
            "latents": latents[:, : self.num_timesteps + 1],
            "next_latents": latents[:, 1:],
            "timesteps": self._take_to_device(batch["timesteps"], indices, device),
            "logps": self._take_to_device(batch["logps"], indices, device),
            "nodesxsample": self._take_to_device(batch["nodesxsample"], indices, device),
        }
        if self.condition:
            sample["context"] = self._take_to_device(batch["context"], indices, device)
        return sample

    def capture_grad_probe(self):
        """Capture a small local gradient summary before optimizer.step()."""
        probe_cfg = self.config.get("ddp", {})
        if not bool(probe_cfg.get("grad_sync_probe", False)):
            self.last_grad_probe = []
            return []

        max_params = int(probe_cfg.get("grad_sync_probe_max_params", 8))
        grad_probe = []
        for idx, (name, param) in enumerate(self.model.named_parameters()):
            if idx >= max_params:
                break
            grad = param.grad
            grad_probe.append(
                {
                    "name": name,
                    "param_norm": float(param.detach().float().norm().item()),
                    "grad_is_none": grad is None,
                    "grad_norm": float(grad.detach().float().norm().item()) if grad is not None else None,
                    "grad_mean": float(grad.detach().float().mean().item()) if grad is not None else None,
                }
            )
        self.last_grad_probe = grad_probe
        return grad_probe

    def train_batched_samples(
        self,
        data: DataProto,
        microbatch_indices: list[torch.Tensor],
        device: torch.device,
        loss_scale: float,
        ppo_epoch_idx: int = 0,
    ):
        """
        Train on rollout samples by streaming one micro-batch to GPU at a time.

        Args:
            data: Full rollout batch, typically stored on CPU.
            microbatch_indices: CPU index tensors defining the micro-batch slices.
            device: Device that owns the training model.
            loss_scale: Scalar multiplier applied before backward to preserve
                the configured update scale when effective rollout batch grows.
            ppo_epoch_idx (int): Zero-based index of the current PPO inner epoch (for logging).

        Side Effects:
            - Model weights are updated via self.optimizer.step().
            - Logs the statistics to the accelerator trackers.

        Returns:
            dict: Metric dict containing ClipFrac, Loss, PPO_Loss, KL_Penalty, PPO_Epoch.
        """
        info = defaultdict(list)
        num_steps = self.num_timesteps + 1
        desc_suffix = f" [PPO {ppo_epoch_idx}]" if ppo_epoch_idx > 0 else ""
        for _i, indices in tq(
            enumerate(microbatch_indices),
            desc=f"Training{desc_suffix}",
            unit="Batch",
            leave=False,
            file=sys.stdout,
            disable=(self._rank != 0),
        ):
            if _i == 0:
                log_cuda_memory(
                    f"train_batched_samples:ppo{ppo_epoch_idx}:before_first_microbatch",
                    component="actor",
                    device=device,
                )
            sample = self._training_microbatch(data, indices, device)
            if _i == 0:
                log_cuda_memory(
                    f"train_batched_samples:ppo{ppo_epoch_idx}:after_first_microbatch_materialized",
                    component="actor",
                    device=device,
                )
            for j in tq(
                range(num_steps),
                desc=f"Training Batch{desc_suffix}",
                unit="timesteps",
                leave=False,
                file=sys.stdout,
                disable=(self._rank != 0),
            ):
                if self.condition:
                    context = sample["context"]
                else:
                    context = None
                loss, clipfrac, ppo_loss_val, kl_val = self.calculate_loss(
                    sample["latents"][:, j],
                    sample["timesteps"][:, j],
                    sample["next_latents"][:, j],
                    sample["logps"][:, j],
                    sample["advantages"],
                    sample["nodesxsample"],
                    context=context,
                )
                info["clipfrac"].append(clipfrac.item())
                info["loss"].append(loss.item())
                info["ppo_loss"].append(ppo_loss_val.item())
                if self.kl_coef > 0.0:
                    info["kl_penalty"].append(kl_val.item())
                (loss * loss_scale).backward()
                clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
            del sample
            if _i == 0:
                log_cuda_memory(
                    f"train_batched_samples:ppo{ppo_epoch_idx}:after_first_microbatch_released",
                    component="actor",
                    device=device,
                )
        self.capture_grad_probe()
        self.optimizer.step()
        self.optimizer.zero_grad()

        metric = {}
        metric["ClipFrac"] = np.mean(np.array(info["clipfrac"]))
        metric["Loss"] = np.mean(np.array(info["loss"]))
        metric["PPO_Loss"] = np.mean(np.array(info["ppo_loss"]))
        if info["kl_penalty"]:
            metric["KL_Penalty"] = np.mean(np.array(info["kl_penalty"]))
        metric["PPO_Epoch"] = ppo_epoch_idx
        metric["LossScale"] = float(loss_scale)
        metric["TrainMicroBatches"] = float(len(microbatch_indices))

        return metric

    def calculate_loss(self, latents, timesteps, next_latents, log_prob_old, advantages, nodesxsample, context=None):
        """
        Calculate the loss for a batch of an unpacked sample.

        When ``self.kl_coef > 0`` and ``self.ref_model`` is not None,
        a Gaussian KL penalty term is added to prevent policy drift away
        from the frozen pretrained reference model:

            total_loss = ppo_loss + kl_coef * KL(ref || current)

        Args:
            latents:     [B, N, D]  latent at time t.
            timesteps:   [B]        normalized timestep.
            next_latents:[B, N, D]  latent at time s (s < t).
            log_prob_old:[B]        log p(z_s | z_t) under the *old* (rollout) policy.
            advantages:  [B]        PPO advantage estimates.
            nodesxsample: [B]       number of valid nodes per sample.
            context:     optional   condition embedding.

        Returns:
            (loss, clipfrac, ppo_loss, kl_penalty) — all scalar tensors.
        """
        s_array = timesteps
        t_array = s_array + 1
        s_array = s_array / self.num_timesteps
        t_array = t_array / self.num_timesteps
        current_model = self.model
        helper_model = self.model_impl
        node_mask, edge_mask = helper_model.get_mask(nodesxsample, latents.shape[0], self.max_n_nodes)
        node_mask = node_mask.to(latents.device)
        edge_mask = edge_mask.to(latents.device)

        kwargs = dict(
            mode="ddpo_log_prob",
            s=s_array,
            t=t_array,
            zt=latents,
            node_mask=node_mask,
            edge_mask=edge_mask,
            prev_sample=next_latents,
            context=context,
        )

        # --- Current model forward: returns (log_prob, mu, sigma) ---
        log_prob_current, mu_new, sigma_new = current_model(**kwargs)

        # --- Standard PPO loss (unchanged) ---
        dif_logp = log_prob_current - log_prob_old
        ratio = torch.exp(dif_logp)
        ppo_loss = self.loss(advantages, self.clip_range, ratio)
        clipfrac = torch.mean((torch.abs(ratio - 1.0) > self.clip_range).float())

        # --- Optional KL penalty against frozen reference model ---
        kl_penalty = torch.tensor(0.0, device=latents.device)
        if self.kl_coef > 0.0 and self.ref_model is not None:
            with torch.no_grad():
                _, mu_ref, sigma_ref = self.ref_model(**kwargs)

            from ddpo.kl import compute_gaussian_kl

            kl_penalty = compute_gaussian_kl(
                mu_ref=mu_ref,
                sigma_ref=sigma_ref,
                mu_new=mu_new,
                sigma_new=sigma_new,
                mask=node_mask,
            )

        loss = ppo_loss + self.kl_coef * kl_penalty
        return loss, clipfrac, ppo_loss, kl_penalty

    def loss(
        self,
        advantages: torch.Tensor,
        clip_range: float,
        ratio: torch.Tensor,
    ):
        unclipped_loss = -1.0 * advantages * ratio
        clipped_loss = -1.0 * advantages * torch.clamp(
            ratio,
            1.0 - clip_range,
            1.0 + clip_range,
        )

        return torch.mean(torch.maximum(unclipped_loss, clipped_loss))

    def update_policy(self, data: DataProto) -> Dict:
        """
        Update the policy with PPO, optionally reusing the same rollout data
        for multiple inner epochs.

        Log-probabilities from the rollout policy (``log_prob_old``) are kept
        **frozen** throughout all PPO inner epochs, following the standard
        importance-sampling formulation (Schulman et al. 2017).  The PPO clip
        mechanism naturally prevents ratio explosion when the policy drifts.

        When ``num_ppo_epochs > 1`` and ``ppo_shuffle_microbatch`` is enabled,
        the order of micro-batches is shuffled before each inner epoch to
        improve gradient diversity.

        Args:
            data (DataProto): Rollout data with ``latents``, ``timesteps``,
                ``logps``, ``advantages``, and meta info.

        Returns:
            Dict: Averaged metrics across all PPO epochs
                (``Loss``, ``ClipFrac``, ``PPO_Loss``, ``KL_Penalty``,
                 ``PPO_Epochs``).
        """
        # ── Step 1: inspect rollout batch and build streaming plan ─────
        self.max_n_nodes = data.meta_info["max_n_nodes"]
        self.batch_size = data.batch.batch_size[0]
        self.condition = data.meta_info["condition"]
        log_cuda_memory("update_policy:start", component="actor")
        device = self._training_device()
        loss_scale = self._loss_scale(int(self.batch_size))
        microbatch_indices = self._microbatch_indices(int(self.batch_size))

        log_cuda_memory("update_policy:after_streaming_plan", component="actor", device=device)

        # ── Step 3: multi-epoch PPO training ──────────────────────────
        # Each inner epoch performs a full gradient accumulation + optimizer step
        # over the same rollout data.  ``log_prob_old`` is fixed throughout.
        all_metrics: dict[str, list[float]] = defaultdict(list)

        for ppo_epoch in range(self.num_ppo_epochs):
            # Optional micro-batch shuffle for gradient diversity
            epoch_indices = list(microbatch_indices)
            if self.ppo_shuffle_microbatch and len(epoch_indices) > 1:
                random.shuffle(epoch_indices)

            metric = self.train_batched_samples(
                data=data,
                microbatch_indices=epoch_indices,
                device=device,
                loss_scale=loss_scale,
                ppo_epoch_idx=ppo_epoch,
            )

            for key, value in metric.items():
                all_metrics[key].append(value)

        # ── Step 4: aggregate metrics across PPO epochs ───────────────
        final_metric: dict[str, float] = {}
        for key, values in all_metrics.items():
            if key == "PPO_Epoch":
                continue  # metadata, not an averaged scalar
            final_metric[key] = float(np.mean(values))
        final_metric["PPO_Epochs"] = self.num_ppo_epochs
        final_metric["TrainBatchSize"] = float(self.batch_size)

        log_cuda_memory("update_policy:after_train", component="actor")
        return final_metric
