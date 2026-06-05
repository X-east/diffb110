"""Per-component running z-score normalizer for CompositeReward.

Normalises each reward component to ~N(0,1) before weighted aggregation
so that weight == exact gradient influence, immune to per-component scale
differences (e.g. chembl_prior spanning [-2, +2] vs diversity [-1, 0]).

Design:
- Each component tracks its own EMA running mean/variance across batches.
- Cold-start: first valid batch directly initialises stats (no EMA blend).
- NaN/Inf-safe: skipped in update; normalised to zero if uninitialised.
- DDP-safe: each rank tracks independently; identical data distributions
  make this equivalent to a synchronised normalizer.
"""

from __future__ import annotations

import torch


class RunningNormalizer:
    """EMA-based running z-score normalizer for a single reward component.

    Args:
        momentum: EMA update rate (0.01 ≈ 100-batch effective window). Must
            be in (0, 1].
        epsilon: Division safety floor to prevent /0 on zero-variance batches.
    """

    def __init__(self, momentum: float = 0.01, epsilon: float = 1e-6) -> None:
        if not 0.0 < momentum <= 1.0:
            raise ValueError(f"momentum must be in (0, 1], got {momentum}")
        self.momentum: float = momentum
        self.epsilon: float = epsilon
        self.mean: float | None = None
        self.var: float | None = None
        self._initialized: bool = False

    def update(self, values: torch.Tensor) -> None:
        """Update running statistics with a batch of per-molecule values.

        Skips NaN/Inf entries. First invocation directly sets mean/var
        (cold-start initialisation).
        """
        valid = values[torch.isfinite(values)]
        if valid.numel() == 0:
            return

        batch_mean = float(valid.mean().item())
        batch_var = float(valid.var(unbiased=False).item()) if valid.numel() > 1 else 0.0

        if not self._initialized:
            self.mean = batch_mean
            self.var = batch_var
            self._initialized = True
        else:
            a = self.momentum
            self.mean = (1.0 - a) * (self.mean or 0.0) + a * batch_mean
            self.var = (1.0 - a) * (self.var or 1.0) + a * batch_var

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        """Apply running z-score: (x - mean) / sqrt(var + epsilon).

        Returns zeros if the normalizer is uninitialised (no valid molecules seen yet).
        """
        if not self._initialized or self.mean is None or self.var is None:
            return torch.zeros_like(values)
        std = (self.var + self.epsilon) ** 0.5
        return (values - self.mean) / std

    @property
    def initialized(self) -> bool:
        return self._initialized


class ComponentNormalizer:
    """Per-component running normalizer collection.

    Maps component names → RunningNormalizer instances and provides a
    convenience ``update_and_normalize(name, values)`` method that chains
    the stat update and normalisation in one call.
    """

    def __init__(
        self,
        component_names: list[str],
        momentum: float = 0.01,
        epsilon: float = 1e-6,
    ) -> None:
        self._normalizers: dict[str, RunningNormalizer] = {
            name: RunningNormalizer(momentum=momentum, epsilon=epsilon)
            for name in component_names
        }

    def update_and_normalize(self, name: str, values: torch.Tensor) -> torch.Tensor:
        """Update running stats for *name* and return normalised values."""
        normalizer = self._normalizers.get(name)
        if normalizer is None:
            raise KeyError(
                f"ComponentNormalizer has no normalizer for '{name}'. "
                f"Registered: {list(self._normalizers.keys())}"
            )
        normalizer.update(values)
        return normalizer.normalize(values)

    def get_stats(self, name: str) -> tuple[float | None, float | None]:
        """Return (mean, var) for a component, or (None, None) if uninitialised."""
        n = self._normalizers.get(name)
        if n is None:
            return None, None
        return n.mean, n.var

    @property
    def initialized(self) -> bool:
        return all(n.initialized for n in self._normalizers.values())
