import os
import sys
from typing import Dict, Optional
from tqdm import tqdm as tq
from collections import defaultdict
from verl_diffusion.protocol import DataProto
from .base import BaseActor
import torch
from torch.nn.utils import clip_grad_norm_
import numpy as np

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

    def train_batched_samples(self, batched_samples):
        """
        Train on a batch of samples. Main training segment

        Args:
            global_step (int): The current global step
            batched_samples (list[dict[str, torch.Tensor]]): The batched samples to train on

        Side Effects:
            - Model weights are updated
            - Logs the statistics to the accelerator trackers.

        Returns:
            global_step (int): The updated global step
        """
        info = defaultdict(list)
        self.T = self.num_timesteps + 1
        for _i, sample in tq(enumerate(batched_samples), desc="Training", unit="Batch", leave=False, file=sys.stdout, disable=(self._rank != 0)):

            for j in tq(range(self.T), desc="Training Batch", unit="timesteps", leave=False, file=sys.stdout, disable=(self._rank != 0)):
                if self.condition :
                    context = sample["context"]
                else:
                    context = None
                loss, clipfrac = self.calculate_loss(
                        sample["latents"][:, j],
                        sample["timesteps"][:, j],
                        sample["next_latents"][:, j],
                        sample["logps"][:, j],
                        sample["advantages"],
                        sample["nodesxsample"],
                        context = context
                    )
                info["clipfrac"].append(clipfrac.item())
                info["loss"].append(loss.item())
                loss.backward()
                clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
        self.capture_grad_probe()
        self.optimizer.step()
        self.optimizer.zero_grad()
            
        metric = {}
        metric["ClipFrac"] = np.mean(np.array(info["clipfrac"]))
        metric["Loss"] = np.mean(np.array(info["loss"]))

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
            (loss, clipfrac) — both scalar tensors.
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
        return loss, clipfrac
    
    def loss(
        self,
        advantages: torch.Tensor,
        clip_range: float,
        ratio: torch.Tensor,
    ):
        unclipped_loss =    -1.0 * advantages * ratio
        clipped_loss =   -1.0  * advantages * torch.clamp(
            ratio,
            1.0 - clip_range,
            1.0 + clip_range,
        )
        
        return torch.mean(torch.maximum(unclipped_loss, clipped_loss))
    
    def update_policy(self, data: DataProto) -> Dict:
        samples = {}
        self.max_n_nodes = data.meta_info["max_n_nodes"]
        self.batch_size = data.batch.batch_size[0]
        self.condition = data.meta_info["condition"]
        samples["advantages"] = data.batch["advantages"]
        
        samples["latents"] = data.batch["latents"][:,:self.num_timesteps+1].clone()
        samples["next_latents"] = data.batch["latents"][:,1:].clone()
        samples["timesteps"] = data.batch["timesteps"]
        samples["logps"] = data.batch["logps"]
        samples["nodesxsample"] = data.batch["nodesxsample"]
        if self.condition:
           samples["context"] = data.batch["context"]

        original_keys = samples.keys()
        original_values = samples.values()
        # rebatch them as user defined train_batch_size is different from sample_batch_size
        samples_batched = []
        
        if self.train_micro_batch_size >= self.batch_size:
            # If train_micro_batch_size is larger than or equal to batch_size,
            # just use the entire batch as one micro batch
            samples_batched.append(samples)
        else:
            # Calculate number of complete batches and remaining samples
            n_complete_batches = self.batch_size // self.train_micro_batch_size
            remaining_samples = self.batch_size % self.train_micro_batch_size
            
            # Process complete batches
            if n_complete_batches > 0:
                reshaped_values = [v.reshape(-1, self.train_micro_batch_size, *v.shape[1:]) for v in original_values]
                transposed_values = zip(*reshaped_values)
                samples_batched.extend([dict(zip(original_keys, row_values)) for row_values in transposed_values])
            
            # Process remaining samples if any
            if remaining_samples > 0:
                remaining_values = [v[-remaining_samples:].unsqueeze(0) for v in original_values]
                samples_batched.append(dict(zip(original_keys, remaining_values)))
        
        # Train each batch
        metric = self.train_batched_samples(samples_batched)
        
        return metric