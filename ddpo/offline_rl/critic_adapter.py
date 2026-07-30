"""Typed geometric-critic inputs for DiffSBDD offline IQL.

The adapter is the only layer that knows how a
:class:`DenoisingTransitionBatch` is split into coordinates, scalar ligand
features, action features, and normalized diffusion time. Critic networks do
not depend on DiffSBDD policy internals or trajectory-shard storage.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

import torch

from .protocol import DenoisingTransitionBatch


@dataclass(frozen=True)
class PocketContext:
    """One padded or unpadded pocket represented by coordinates and features."""

    coordinates: torch.Tensor
    features: torch.Tensor
    node_mask: torch.Tensor

    def __post_init__(self) -> None:
        """Validate the static pocket representation."""
        if self.coordinates.ndim != 2 or int(self.coordinates.shape[-1]) != 3:
            raise ValueError("pocket coordinates must have shape [P,3]")
        if self.features.ndim != 2:
            raise ValueError("pocket features must have shape [P,F]")
        if int(self.features.shape[0]) != int(self.coordinates.shape[0]):
            raise ValueError("pocket coordinates and features must share P")
        expected_mask_shape = (int(self.coordinates.shape[0]), 1)
        if tuple(self.node_mask.shape) != expected_mask_shape:
            raise ValueError(
                f"pocket node_mask must have shape {expected_mask_shape}"
            )
        if int(self.coordinates.shape[0]) == 0:
            raise ValueError("pocket context cannot be empty")
        if self.coordinates.device != self.features.device:
            raise ValueError("pocket coordinates and features must share a device")
        if self.coordinates.device != self.node_mask.device:
            raise ValueError("pocket tensors must share a device")
        if self.coordinates.dtype != self.features.dtype:
            raise TypeError("pocket coordinates and features must share a dtype")
        if not bool(torch.isfinite(self.coordinates).all()):
            raise ValueError("pocket coordinates must be finite")
        if not bool(torch.isfinite(self.features).all()):
            raise ValueError("pocket features must be finite")
        if not bool(
            torch.logical_or(self.node_mask == 0, self.node_mask == 1).all()
        ):
            raise ValueError("pocket node_mask must contain only zero or one")
        if not bool(torch.any(self.node_mask == 1)):
            raise ValueError("pocket context must contain at least one valid node")

    @classmethod
    def from_diffsbdd_mapping(
        cls,
        pocket: Mapping[str, torch.Tensor],
    ) -> "PocketContext":
        """Build a context from DiffSBDD ``prepare_pocket`` output.

        DiffSBDD's ``mask`` field is a graph-assignment index, not a validity
        mask. A single prepared pocket therefore uses an explicit all-one
        validity mask here.
        """
        coordinates = pocket.get("x")
        features = pocket.get("one_hot")
        if not isinstance(coordinates, torch.Tensor):
            raise TypeError("DiffSBDD pocket field 'x' must be a tensor")
        if not isinstance(features, torch.Tensor):
            raise TypeError("DiffSBDD pocket field 'one_hot' must be a tensor")
        features = features.to(
            device=coordinates.device,
            dtype=coordinates.dtype,
        )
        node_mask = coordinates.new_ones((int(coordinates.shape[0]), 1))
        return cls(
            coordinates=coordinates,
            features=features,
            node_mask=node_mask,
        )

    def to(
        self,
        device: str | torch.device,
        *,
        dtype: torch.dtype | None = None,
    ) -> "PocketContext":
        """Return a pocket copy on the requested device and floating dtype."""
        target_dtype = self.coordinates.dtype if dtype is None else dtype
        return PocketContext(
            coordinates=self.coordinates.to(device=device, dtype=target_dtype),
            features=self.features.to(device=device, dtype=target_dtype),
            node_mask=self.node_mask.to(device=device, dtype=target_dtype),
        )


@dataclass(frozen=True)
class GeometricCriticInput:
    """Batched state/action/pocket tensors consumed by geometric critics."""

    state_coordinates: torch.Tensor
    state_features: torch.Tensor
    action_coordinates: torch.Tensor
    action_features: torch.Tensor
    pocket_coordinates: torch.Tensor
    pocket_features: torch.Tensor
    ligand_mask: torch.Tensor
    pocket_mask: torch.Tensor
    normalized_timestep: torch.Tensor
    is_final_decode: torch.Tensor

    def __post_init__(self) -> None:
        """Validate critic input shapes and finite-value invariants."""
        if self.state_coordinates.ndim != 3:
            raise ValueError("state_coordinates must have shape [B,N,3]")
        batch_size, ligand_nodes, coordinate_dim = self.state_coordinates.shape
        if coordinate_dim != 3:
            raise ValueError("state coordinate dimension must equal three")
        if tuple(self.action_coordinates.shape) != tuple(
            self.state_coordinates.shape
        ):
            raise ValueError("action_coordinates must match state coordinates")
        if self.state_features.ndim != 3:
            raise ValueError("state_features must have shape [B,N,F]")
        if tuple(self.state_features.shape[:2]) != (batch_size, ligand_nodes):
            raise ValueError("state_features must match state B,N dimensions")
        if tuple(self.action_features.shape) != tuple(self.state_features.shape):
            raise ValueError("action_features must match state_features")
        if self.pocket_coordinates.ndim != 3:
            raise ValueError("pocket_coordinates must have shape [B,P,3]")
        if int(self.pocket_coordinates.shape[0]) != batch_size:
            raise ValueError("pocket coordinates must share critic batch size")
        if int(self.pocket_coordinates.shape[-1]) != 3:
            raise ValueError("pocket coordinate dimension must equal three")
        if self.pocket_features.ndim != 3:
            raise ValueError("pocket_features must have shape [B,P,Fp]")
        if tuple(self.pocket_features.shape[:2]) != tuple(
            self.pocket_coordinates.shape[:2]
        ):
            raise ValueError("pocket features must match pocket B,P dimensions")
        if tuple(self.ligand_mask.shape) != (batch_size, ligand_nodes, 1):
            raise ValueError("ligand_mask must have shape [B,N,1]")
        expected_pocket_mask = (
            batch_size,
            int(self.pocket_coordinates.shape[1]),
            1,
        )
        if tuple(self.pocket_mask.shape) != expected_pocket_mask:
            raise ValueError(
                f"pocket_mask must have shape {expected_pocket_mask}"
            )
        if tuple(self.normalized_timestep.shape) != (batch_size,):
            raise ValueError("normalized_timestep must have shape [B]")
        if tuple(self.is_final_decode.shape) != (batch_size,):
            raise ValueError("is_final_decode must have shape [B]")
        if self.is_final_decode.dtype != torch.bool:
            raise TypeError("is_final_decode must be torch.bool")
        if bool(
            torch.any(self.normalized_timestep < 0)
            or torch.any(self.normalized_timestep > 1)
        ):
            raise ValueError("normalized_timestep must lie in [0,1]")
        for name, mask in (
            ("ligand_mask", self.ligand_mask),
            ("pocket_mask", self.pocket_mask),
        ):
            if not bool(torch.logical_or(mask == 0, mask == 1).all()):
                raise ValueError(f"{name} must contain only zero or one")
        if bool(torch.any(self.ligand_mask.sum(dim=1) == 0)):
            raise ValueError("every critic row must contain a ligand node")
        if bool(torch.any(self.pocket_mask.sum(dim=1) == 0)):
            raise ValueError("every critic row must contain a pocket node")

        tensors = (
            self.state_coordinates,
            self.state_features,
            self.action_coordinates,
            self.action_features,
            self.pocket_coordinates,
            self.pocket_features,
            self.ligand_mask,
            self.pocket_mask,
            self.normalized_timestep,
        )
        devices = {tensor.device for tensor in tensors}
        if len(devices) != 1:
            raise ValueError("all geometric critic tensors must share a device")
        floating_dtypes = {
            tensor.dtype for tensor in tensors if torch.is_floating_point(tensor)
        }
        if len(floating_dtypes) != 1:
            raise TypeError("all geometric critic floating tensors must share a dtype")
        for tensor in tensors:
            if torch.is_floating_point(tensor) and not bool(
                torch.isfinite(tensor).all()
            ):
                raise ValueError("geometric critic inputs must all be finite")

    @property
    def batch_size(self) -> int:
        """Return the transition batch size."""
        return int(self.state_coordinates.shape[0])


def build_geometric_critic_input(
    transitions: DenoisingTransitionBatch,
    pocket: PocketContext,
    *,
    num_diffusion_steps: int,
) -> GeometricCriticInput:
    """Convert structural offline transitions to policy-independent critic input.

    Final DiffSBDD actions contain stochastic coordinates but atom identities
    are decoded from ``z0``. Consequently final action features are removed
    using the transition's action mask before they reach Q.

    Recorded ligand latents use a center-of-mass frame while
    ``prepare_pocket`` returns crystallographic coordinates. The fixed pocket
    is therefore centered by its valid-node centroid here so radius-based
    pocket-ligand edges are defined in the same canonical local frame.
    """
    if num_diffusion_steps <= 0:
        raise ValueError("num_diffusion_steps must be positive")
    if int(transitions.state_latent.shape[-1]) <= 3:
        raise ValueError("DiffSBDD latent must contain xyz and atom features")

    state_dtype = transitions.state_latent.dtype
    aligned_pocket = pocket.to(transitions.device, dtype=state_dtype)
    batch_size = transitions.batch_size
    ligand_mask = transitions.node_mask.to(dtype=state_dtype)
    pocket_mask = aligned_pocket.node_mask.unsqueeze(0).expand(
        batch_size,
        -1,
        -1,
    )
    pocket_centroid = torch.sum(
        aligned_pocket.coordinates * aligned_pocket.node_mask,
        dim=0,
        keepdim=True,
    ) / torch.sum(aligned_pocket.node_mask).clamp_min(1.0)
    centered_pocket_coordinates = (
        aligned_pocket.coordinates - pocket_centroid
    ) * aligned_pocket.node_mask
    pocket_coordinates = centered_pocket_coordinates.unsqueeze(0).expand(
        batch_size,
        -1,
        -1,
    )
    pocket_features = aligned_pocket.features.unsqueeze(0).expand(
        batch_size,
        -1,
        -1,
    )

    state_coordinates = transitions.state_latent[..., :3] * ligand_mask
    state_features = transitions.state_latent[..., 3:] * ligand_mask
    action_coordinates = (
        transitions.action_latent[..., :3]
        * transitions.action_mask[..., :3]
    )
    action_features = (
        transitions.action_latent[..., 3:]
        * transitions.action_mask[..., 3:]
    )
    normalized_timestep = (
        transitions.source_diffusion_timestep.to(dtype=state_dtype)
        / float(num_diffusion_steps)
    )
    return GeometricCriticInput(
        state_coordinates=state_coordinates,
        state_features=state_features,
        action_coordinates=action_coordinates,
        action_features=action_features,
        pocket_coordinates=pocket_coordinates,
        pocket_features=pocket_features,
        ligand_mask=ligand_mask,
        pocket_mask=pocket_mask,
        normalized_timestep=normalized_timestep,
        is_final_decode=transitions.is_final_decode,
    )


def build_next_geometric_critic_input(
    transitions: DenoisingTransitionBatch,
    pocket: PocketContext,
    *,
    num_diffusion_steps: int,
) -> GeometricCriticInput:
    """Build the successor-state input used by the IQL TD target.

    The successor of an ordinary reverse step is its sampled action latent.
    If that successor is ``z0`` then its next decision is final decode. Terminal
    successors are still represented by finite tensors, but ``done`` removes
    their value from the TD target.
    """
    current = build_geometric_critic_input(
        transitions,
        pocket,
        num_diffusion_steps=num_diffusion_steps,
    )
    ligand_mask = current.ligand_mask
    next_state_coordinates = (
        transitions.next_state_latent[..., :3] * ligand_mask
    )
    next_state_features = (
        transitions.next_state_latent[..., 3:] * ligand_mask
    )
    next_timestep = transitions.target_diffusion_timestep.clamp_min(0).to(
        dtype=transitions.state_latent.dtype
    ) / float(num_diffusion_steps)
    next_is_final_decode = torch.logical_and(
        ~transitions.done,
        transitions.target_diffusion_timestep == 0,
    )
    return replace(
        current,
        state_coordinates=next_state_coordinates,
        state_features=next_state_features,
        action_coordinates=torch.zeros_like(current.action_coordinates),
        action_features=torch.zeros_like(current.action_features),
        normalized_timestep=next_timestep,
        is_final_decode=next_is_final_decode,
    )
