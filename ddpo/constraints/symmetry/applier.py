"""Tensor-level application of paired-sidechain copy constraints."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ddpo.constraints.symmetry.spec import SymmetrySpec


@dataclass
class SymmetryConstraintApplier:
    """Apply R1-to-R2 h-channel copying and build matching action masks."""

    spec: SymmetrySpec

    @property
    def enabled(self) -> bool:
        """Return whether this applier should modify tensors."""
        return bool(self.spec.enabled)

    def _index_tensor(self, indices: tuple[int, ...], device: torch.device) -> torch.Tensor:
        """Return a long index tensor on the target device."""
        return torch.tensor(indices, dtype=torch.long, device=device)

    def validate_tensor(self, tensor: torch.Tensor, node_mask: torch.Tensor | None = None) -> None:
        """Validate that a padded latent tensor can carry the configured slots."""
        if tensor.ndim != 3:
            raise ValueError(f"Expected [B, N, D] tensor, got shape {tuple(tensor.shape)}")
        max_index = max((*self.spec.r1_indices, *self.spec.r2_indices), default=-1)
        if max_index >= int(tensor.shape[1]):
            raise ValueError(
                f"Symmetry slot index {max_index} is outside padded node dimension {int(tensor.shape[1])}"
            )
        if self.spec.h_start >= int(tensor.shape[-1]):
            raise ValueError(
                f"Symmetry h_start={self.spec.h_start} is outside feature dimension {int(tensor.shape[-1])}"
            )
        if node_mask is not None:
            if node_mask.shape[:2] != tensor.shape[:2]:
                raise ValueError("node_mask and tensor must share [B, N] dimensions")
            required = torch.cat(
                [
                    self._index_tensor(self.spec.r1_indices, tensor.device),
                    self._index_tensor(self.spec.r2_indices, tensor.device),
                ],
                dim=0,
            )
            valid_values = node_mask.to(device=tensor.device)[..., 0].index_select(1, required)
            if bool((valid_values <= 0).any().item()):
                raise ValueError("Configured symmetry slots include padded/invalid ligand nodes")

    def apply(self, tensor: torch.Tensor, node_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Return a tensor where R2 h-channels are copied from the paired R1 slots."""
        if not self.enabled:
            return tensor
        self.validate_tensor(tensor, node_mask=node_mask)
        result = tensor.clone()
        r1 = self._index_tensor(self.spec.r1_indices, tensor.device)
        r2 = self._index_tensor(self.spec.r2_indices, tensor.device)
        h_start = self.spec.h_start
        h_stop = self.spec.h_stop_for(int(tensor.shape[-1]))
        result[:, r2, h_start:h_stop] = result[:, r1, h_start:h_stop]
        if node_mask is not None:
            result = result * node_mask.to(device=result.device, dtype=result.dtype)
        return result

    def build_action_mask(self, node_mask: torch.Tensor, feature_dim: int) -> torch.Tensor:
        """Build a `[B, N, D]` mask with copied R2 h-channels excluded from PPO actions."""
        if not self.enabled:
            return node_mask.expand(*node_mask.shape[:2], int(feature_dim)).clone()
        if node_mask.ndim != 3 or node_mask.shape[-1] != 1:
            raise ValueError(f"Expected node_mask [B, N, 1], got shape {tuple(node_mask.shape)}")
        max_index = max((*self.spec.r1_indices, *self.spec.r2_indices), default=-1)
        if max_index >= int(node_mask.shape[1]):
            raise ValueError(
                f"Symmetry slot index {max_index} is outside padded node dimension {int(node_mask.shape[1])}"
            )
        action_mask = node_mask.expand(*node_mask.shape[:2], int(feature_dim)).clone()
        r2 = self._index_tensor(self.spec.r2_indices, node_mask.device)
        h_start = self.spec.h_start
        h_stop = self.spec.h_stop_for(int(feature_dim))
        action_mask[:, r2, h_start:h_stop] = 0
        return action_mask

