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
    failure_advantage_override: float | None  # None = use z-score result; float = assign fixed advantage to failures

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
            use_rewards_key=str(norm_cfg.get("use_rewards_key", "reward_total_raw")),
            raw_output_key=str(norm_cfg.get("raw_output_key", "reward_total_raw")),
            norm_output_key=str(norm_cfg.get("norm_output_key", "reward_total_norm")),
            clamped_output_key=str(norm_cfg.get("clamped_output_key", "reward_total_clamped")),
            stats_prefix=str(norm_cfg.get("stats_prefix", "reward")),
            failure_penalty=float(failure_cfg.get("raw_penalty", -5.0)),
            include_failures_in_stats=bool(failure_cfg.get("include_in_stats", True)),
            clamp_enabled=bool(clamp_cfg.get("enabled", True)),
            clamp_min=float(clamp_cfg.get("min", -5.0)),
            clamp_max=float(clamp_cfg.get("max", 5.0)),
            map_rewards_field=str(norm_cfg.get("map_rewards_field", "reward_total")),
            advantages_key=str(norm_cfg.get("advantages_key", "advantages")),
            override_rewards_field=bool(norm_cfg.get("override_rewards_field", True)),
            failure_advantage_override=float(failure_cfg["advantage_override"])
                if "advantage_override" in failure_cfg else None,
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

    def _build_failure_mask(self, data: Any, batch: dict[str, Any], batch_size: int, device: torch.device) -> torch.Tensor:
        # Read failure_type from non_tensor_batch first (Python-native str list),
        # fall back to batch dict for backward compatibility.
        ntb = getattr(data, "non_tensor_batch", None)
        source = ntb if isinstance(ntb, dict) else batch
        failure_type = source.get("failure_type")
        if failure_type is not None:
            mask = torch.tensor(
                [str(item) != FAILURE_SUCCESS for item in list(failure_type)],
                device=device,
                dtype=torch.bool,
            )
            if mask.numel() == batch_size:
                return mask

        rdkit_valid = source.get("rdkit_valid")
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
        if rewards is None and self.config.use_rewards_key != "reward_total":
            rewards = batch.get("reward_total")
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
        # TensorDict with non-empty batch_size rejects Python scalars (shape=[]).
        # Broadcast all global-stat scalars into [batch_size] tensors so they
        # are compatible with both plain dicts and TensorDict containers.
        bs = int(batch.get("x", raw_tensor).shape[0]) if "x" in batch and isinstance(batch.get("x"), torch.Tensor) else int(raw_tensor.numel())
        device = raw_tensor.device

        def _broadcast(v: Any) -> torch.Tensor:
            t = v if isinstance(v, torch.Tensor) else torch.tensor(float(v), dtype=torch.float32)
            return t.expand(bs).clone().to(device)

        def _broadcast_int(v: Any) -> torch.Tensor:
            t = v if isinstance(v, torch.Tensor) else torch.tensor(int(v), dtype=torch.int64)
            return t.expand(bs).clone().to(device)

        batch[f"{prefix}_mean_raw"] = _broadcast(raw_stats.mean)
        batch[f"{prefix}_std_raw"] = _broadcast(raw_stats.std)
        batch[f"{prefix}_min_raw"] = _broadcast(raw_stats.minimum)
        batch[f"{prefix}_max_raw"] = _broadcast(raw_stats.maximum)
        batch[f"{prefix}_count_raw"] = _broadcast_int(raw_stats.count)

        if norm_tensor.numel() > 0:
            batch[f"{prefix}_mean_norm"] = _broadcast(float(norm_tensor.mean().item()))
            batch[f"{prefix}_std_norm"] = _broadcast(float(norm_tensor.std(unbiased=False).item()) if norm_tensor.numel() > 1 else 0.0)
            batch[f"{prefix}_min_norm"] = _broadcast(float(norm_tensor.min().item()))
            batch[f"{prefix}_max_norm"] = _broadcast(float(norm_tensor.max().item()))
        else:
            batch[f"{prefix}_mean_norm"] = _broadcast(0.0)
            batch[f"{prefix}_std_norm"] = _broadcast(0.0)
            batch[f"{prefix}_min_norm"] = _broadcast(0.0)
            batch[f"{prefix}_max_norm"] = _broadcast(0.0)

        batch[f"{prefix}_failure_count"] = _broadcast_int(int((raw_tensor == self.config.failure_penalty).sum().item()))

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

        failure_mask = self._build_failure_mask(data, batch, batch_size=batch_size, device=device)
        rewards_raw = rewards_raw.clone()
        rewards_raw[failure_mask] = float(self.config.failure_penalty)

        stats_values = self._stats_input(rewards_raw, failure_mask)
        raw_stats = compute_global_reward_stats(stats_values, dist_ctx=dist_ctx, epsilon=self.config.epsilon)

        rewards_norm = (rewards_raw - float(raw_stats.mean)) / float(raw_stats.std)
        if self.config.clamp_enabled:
            rewards_clamped = rewards_norm.clamp(self.config.clamp_min, self.config.clamp_max)
        else:
            rewards_clamped = rewards_norm

        # Override failure advantage with a fixed value if configured.
        # This allows excluding failures from z-score stats (preventing outlier distortion)
        # while still assigning a reasonable negative signal so the model learns to avoid them.
        if self.config.failure_advantage_override is not None and failure_mask.any():
            rewards_clamped[failure_mask] = float(self.config.failure_advantage_override)
            rewards_norm[failure_mask] = float(self.config.failure_advantage_override)

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
