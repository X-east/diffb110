from __future__ import annotations

"""Minimal model interface used by the native DiffSBDD + DDPO adapter."""

from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def sample(self, *args, **kwargs):
        """Sample trajectories from the model."""
        raise NotImplementedError

    @abstractmethod
    def compute_log_p_zs_given_zt(self, *args, **kwargs):
        """Compute log probability of z_s given z_t."""
        raise NotImplementedError

    @abstractmethod
    def sample_p_zs_given_zt(self, *args, **kwargs):
        """Sample z_s given z_t."""
        raise NotImplementedError
