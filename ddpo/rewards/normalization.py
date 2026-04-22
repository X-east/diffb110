from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ddpo.runtime.reward_stats import RewardStats, compute_global_reward_stats
from ddpo.utils.molecule_semantics import FAILURE_SUCCESS


@dataclass(frozen=True)
class RewardNormalizationConfig:
    enabled: bool
    method: str
    epsilon: float
    use_rewards_key: str
    raw_output_key: str
    norm_output_key: str
    clamped_output_key: str
    stats_prefix: str
    failure_penalty: float
    include_failures_in_stats: bool
    clamp_enabled: bool
    clamp_min: float
    clamp_max: float
    map_rewards_field: str
    advantages_key: str
    override_rewards_field: bool

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "RewardNormalizationConfig":
        reward_cfg = config.get("reward", {})
        norm_cfg = reward_cfg.get("normalization", {})
        clamp_cfg = norm_cfg.get("clamp", {})
        failure_cfg = norm_cfg.get("failure", {})
        return cls(
            enabled=bool(norm_cfg.get("enabled", False)),
            method=str(norm_cfg.get("method", "zscore_total")),
            epsilon=float(norm_cfg.get("epsilon", 1.0e-6)),
            use_rewards_key=str(norm_cfg.get("use_rewards_key", "rewards_raw")),
            raw_output_key=str(norm_cfg.get("raw_output_key", "rewards_raw")),
            norm_output_key=str(norm_cfg.get("norm_output_key", "rewards_norm")),
            clamped_output_key=str(norm_cfg.get("clamped_output_key", "rewards_clamped")),
            stats_prefix=str(norm_cfg.get("stats_prefix", "reward")),
            failure_penalty=float(failure_cfg.get("raw_penalty", -5.0)),
            include_failures_in_stats=bool(failure_cfg.get("include_in_stats", True)),
            clamp_enabled=bool(clamp_cfg.get("enabled", True)),
            clamp_min=float(clamp_cfg.get("min", -5.0)),
            clamp_max=float(clamp_cfg.get("max", 5.0)),
            map_rewards_field=str(norm_cfg.get("map_rewards_field", "rewards")),
            advantages_key=str(norm_cfg.get("advantages_key", "advantages")),
            override_rewards_field=bool(norm_cfg.get("override_rewards_field", True)),
        )


class RewardNormalizer:
    """Apply YAML-driven total-reward normalization after reward composition.

    Policy:
    - component rewards remain untouched and inspectable
    - failure samples receive one explicit raw penalty
    - z-score is applied on total reward only
    - clamped normalized reward becomes training advantage by default
    """

    def __init__(self, config: RewardNormalizationConfig, logger: Any | None = None) -> None:
        self.config = config
        self.logger = logger

    def _log(self, message: str, *args: Any) -> None:
        if self.logger is not None:
            self.logger.info(message, *args)

    def _build_failure_mask(self, batch: dict[str, Any], batch_size: int, device: torch.device) -> torch.Tensor:
        failure_type = batch.get("failure_type")
        if failure_type is not None:
            mask = torch.tensor(
                [str(item) != FAILURE_SUCCESS for item in list(failure_type)],
                device=device,
                dtype=torch.bool,
            )
            if mask.numel() == batch_size:
                return mask

        rdkit_valid = batch.get("rdkit_valid")
        if rdkit_valid is not None:
            if isinstance(rdkit_valid, torch.Tensor):
                valid = rdkit_valid.detach().to(device=device).reshape(-1).to(dtype=torch.int64)
                if valid.numel() == batch_size:
                    return valid == 0
            valid = torch.tensor(list(rdkit_valid), device=device, dtype=torch.int64).reshape(-1)
            if valid.numel() == batch_size:
                return valid == 0

        return torch.zeros(batch_size, device=device, dtype=torch.bool)

    def _resolve_raw_rewards(self, batch: dict[str, Any], device: torch.device) -> torch.Tensor:
        rewards = batch.get(self.config.use_rewards_key)
        if rewards is None and self.config.use_rewards_key != "rewards":
            rewards = batch.get("rewards")
        if rewards is None:
            raise ValueError(
                f"[RewardNormalizer] Missing total reward field '{self.config.use_rewards_key}' in batch."
            )
        if isinstance(rewards, torch.Tensor):
            return rewards.detach().to(device=device, dtype=torch.float32).reshape(-1)
        return torch.tensor(list(rewards), device=device, dtype=torch.float32).reshape(-1)

    def _stats_input(self, rewards_raw: torch.Tensor, failure_mask: torch.Tensor) -> torch.Tensor:
        if self.config.include_failures_in_stats:
            return rewards_raw
        return rewards_raw[~failure_mask]

    def _write_stats(self, batch: dict[str, Any], raw_stats: RewardStats, norm_tensor: torch.Tensor, raw_tensor: torch.Tensor) -> None:
        prefix = self.config.stats_prefix
        batch[f"{prefix}_mean_raw"] = raw_stats.mean
        batch[f"{prefix}_std_raw"] = raw_stats.std
        batch[f"{prefix}_min_raw"] = raw_stats.minimum
        batch[f"{prefix}_max_raw"] = raw_stats.maximum
        batch[f"{prefix}_count_raw"] = raw_stats.count

        if norm_tensor.numel() > 0:
            batch[f"{prefix}_mean_norm"] = float(norm_tensor.mean().item())
            batch[f"{prefix}_std_norm"] = float(norm_tensor.std(unbiased=False).item()) if norm_tensor.numel() > 1 else 0.0
            batch[f"{prefix}_min_norm"] = float(norm_tensor.min().item())
            batch[f"{prefix}_max_norm"] = float(norm_tensor.max().item())
        else:
            batch[f"{prefix}_mean_norm"] = 0.0
            batch[f"{prefix}_std_norm"] = 0.0
            batch[f"{prefix}_min_norm"] = 0.0
            batch[f"{prefix}_max_norm"] = 0.0

        batch[f"{prefix}_failure_count"] = int((raw_tensor == self.config.failure_penalty).sum().item())

    def apply(self, data: Any, dist_ctx: Any) -> Any:
        if not self.config.enabled:
            return data
        if self.config.method != "zscore_total":
            raise ValueError(
                f"Unsupported reward.normalization.method='{self.config.method}'. "
                "Currently only 'zscore_total' is implemented."
            )

        batch = data.batch
        if "x" in batch and isinstance(batch["x"], torch.Tensor):
            device = batch["x"].device
            batch_size = int(batch["x"].shape[0])
        else:
            raw_probe = self._resolve_raw_rewards(batch, device=getattr(dist_ctx, "device", torch.device("cpu")))
            device = raw_probe.device
            batch_size = int(raw_probe.numel())

        rewards_raw = self._resolve_raw_rewards(batch, device=device)
        if rewards_raw.numel() != batch_size:
            raise ValueError(
                f"[RewardNormalizer] Reward vector size mismatch: expected {batch_size}, got {rewards_raw.numel()}"
            )

        failure_mask = self._build_failure_mask(batch, batch_size=batch_size, device=device)
        rewards_raw = rewards_raw.clone()
        rewards_raw[failure_mask] = float(self.config.failure_penalty)

        stats_values = self._stats_input(rewards_raw, failure_mask)
        raw_stats = compute_global_reward_stats(stats_values, dist_ctx=dist_ctx, epsilon=self.config.epsilon)

        rewards_norm = (rewards_raw - float(raw_stats.mean)) / float(raw_stats.std)
        if self.config.clamp_enabled:
            rewards_clamped = rewards_norm.clamp(self.config.clamp_min, self.config.clamp_max)
        else:
            rewards_clamped = rewards_norm

        batch[self.config.raw_output_key] = rewards_raw
        batch[self.config.norm_output_key] = rewards_norm
        batch[self.config.clamped_output_key] = rewards_clamped
        batch[self.config.advantages_key] = rewards_clamped
        batch["reward_failure_mask"] = failure_mask

        if self.config.override_rewards_field:
            batch[self.config.map_rewards_field] = rewards_clamped

        self._write_stats(batch, raw_stats=raw_stats, norm_tensor=rewards_norm, raw_tensor=rewards_raw)
        self._log(
            "[RewardNormalizer] raw_mean=%.4f raw_std=%.4f raw_min=%.4f raw_max=%.4f failures=%d clamp=%s [%.2f, %.2f]",
            raw_stats.mean,
            raw_stats.std,
            raw_stats.minimum,
            raw_stats.maximum,
            int(failure_mask.sum().item()),
            self.config.clamp_enabled,
            self.config.clamp_min,
            self.config.clamp_max,
        )
        return data
