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
    """Minimal local actor interface retained from the RLPF integration."""

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

    def train_batched_samples(self, batched_samples, ppo_epoch_idx: int = 0):
        """
        Train on a batch of samples. Main training segment.

        Args:
            batched_samples (list[dict[str, torch.Tensor]]): The batched samples to train on.
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
        for _i, sample in tq(
            enumerate(batched_samples),
            desc=f"Training{desc_suffix}",
            unit="Batch",
            leave=False,
            file=sys.stdout,
            disable=(self._rank != 0),
        ):
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
                loss.backward()
                clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
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
        # ── Step 1: extract and prepare raw data ──────────────────────
        samples: dict[str, torch.Tensor] = {}
        self.max_n_nodes = data.meta_info["max_n_nodes"]
        self.batch_size = data.batch.batch_size[0]
        self.condition = data.meta_info["condition"]
        log_cuda_memory("update_policy:start", component="actor")
        samples["advantages"] = data.batch["advantages"]
        device = samples["advantages"].device

        latents = data.batch["latents"].to(device)
        samples["latents"] = latents[:, : self.num_timesteps + 1]
        samples["next_latents"] = latents[:, 1:]
        samples["timesteps"] = data.batch["timesteps"].to(device)
        samples["logps"] = data.batch["logps"].to(device)
        samples["nodesxsample"] = data.batch["nodesxsample"].to(device)
        if self.condition:
            samples["context"] = data.batch["context"].to(device)

        log_cuda_memory("update_policy:after_view_setup", component="actor")

        # ── Step 2: split into micro-batches ─────────────────────────
        original_keys = list(samples.keys())
        original_values = list(samples.values())
        samples_batched: list[dict[str, torch.Tensor]] = []

        if self.train_micro_batch_size >= self.batch_size:
            samples_batched.append(samples)
        else:
            n_complete_batches = self.batch_size // self.train_micro_batch_size
            remaining_samples = self.batch_size % self.train_micro_batch_size

            if n_complete_batches > 0:
                n_complete = n_complete_batches * self.train_micro_batch_size
                complete_values = [
                    v[:n_complete].reshape(-1, self.train_micro_batch_size, *v.shape[1:])
                    for v in original_values
                ]
                transposed_values = zip(*complete_values)
                samples_batched.extend(
                    dict(zip(original_keys, row_values))
                    for row_values in transposed_values
                )

            if remaining_samples > 0:
                remaining_values = [v[n_complete:].unsqueeze(0) for v in original_values]
                samples_batched.append(dict(zip(original_keys, remaining_values)))

        # ── Step 3: multi-epoch PPO training ──────────────────────────
        # Each inner epoch performs a full gradient accumulation + optimizer step
        # over the same rollout data.  ``log_prob_old`` is fixed throughout.
        all_metrics: dict[str, list[float]] = defaultdict(list)

        for ppo_epoch in range(self.num_ppo_epochs):
            # Optional micro-batch shuffle for gradient diversity
            if self.ppo_shuffle_microbatch and len(samples_batched) > 1:
                random.shuffle(samples_batched)

            metric = self.train_batched_samples(samples_batched, ppo_epoch_idx=ppo_epoch)

            for key, value in metric.items():
                all_metrics[key].append(value)

        # ── Step 4: aggregate metrics across PPO epochs ───────────────
        final_metric: dict[str, float] = {}
        for key, values in all_metrics.items():
            if key == "PPO_Epoch":
                continue  # metadata, not an averaged scalar
            final_metric[key] = float(np.mean(values))
        final_metric["PPO_Epochs"] = self.num_ppo_epochs

        log_cuda_memory("update_policy:after_train", component="actor")
        return final_metric
