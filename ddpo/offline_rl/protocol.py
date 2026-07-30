"""Typed data boundaries for DiffSBDD denoising offline reinforcement learning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class DenoisingEpisodeFacts:
    """Immutable metadata for one trajectory episode.

    This object intentionally contains no reward or mutable molecule object.
    Reward sidecars will be joined in a later layer using ``sample_id``.
    """

    sample_id: str
    shard_path: Path
    episode_index: int
    rank: int
    shard_index: int
    seed: int
    num_nodes: int
    num_diffusion_steps: int
    max_n_nodes: int
    condition: bool
    rdkit_valid: bool
    failure_type: str
    smiles: str | None
    checkpoint_sha256: str


@dataclass(frozen=True)
class DenoisingTransitionBatch:
    """A batch of structural DiffSBDD denoising MDP transitions.

    ``next_state_latent`` equals the sampled action for DiffSBDD's deterministic
    state transition. For final decode actions it represents the terminal
    ``xh_final`` observation; ``done=True`` prevents value bootstrapping.
    ``action_mask`` marks action dimensions that affect the MDP: all latent
    dimensions for ordinary reverse steps and xyz only for final decode, because
    the frozen DiffSBDD decoder derives atom types from ``z0`` rather than the
    sampled final h channels.
    ``reward`` remains ``None`` until an explicit episode-reward sidecar is
    supplied by the caller.
    """

    state_latent: torch.Tensor
    action_latent: torch.Tensor
    next_state_latent: torch.Tensor
    action_mask: torch.Tensor
    recorded_behavior_score: torch.Tensor
    recorded_timestep: torch.Tensor
    source_diffusion_timestep: torch.Tensor
    target_diffusion_timestep: torch.Tensor
    decision_index: torch.Tensor
    episode_index: torch.Tensor
    nodesxsample: torch.Tensor
    node_mask: torch.Tensor
    done: torch.Tensor
    is_final_decode: torch.Tensor
    sample_ids: tuple[str, ...]
    reward: torch.Tensor | None = None

    def __post_init__(self) -> None:
        """Validate shapes, devices, and terminal semantics."""
        batch_size = int(self.state_latent.shape[0])
        if self.state_latent.ndim != 3:
            raise ValueError(
                "state_latent must have shape [B, N, D], "
                f"got {tuple(self.state_latent.shape)}"
            )
        expected_latent_shape = tuple(self.state_latent.shape)
        for name, tensor in (
            ("action_latent", self.action_latent),
            ("next_state_latent", self.next_state_latent),
            ("action_mask", self.action_mask),
        ):
            if tuple(tensor.shape) != expected_latent_shape:
                raise ValueError(
                    f"{name} must match state_latent shape "
                    f"{expected_latent_shape}, got {tuple(tensor.shape)}"
                )

        expected_vector_shape = (batch_size,)
        for name, tensor in (
            ("recorded_behavior_score", self.recorded_behavior_score),
            ("recorded_timestep", self.recorded_timestep),
            ("source_diffusion_timestep", self.source_diffusion_timestep),
            ("target_diffusion_timestep", self.target_diffusion_timestep),
            ("decision_index", self.decision_index),
            ("episode_index", self.episode_index),
            ("nodesxsample", self.nodesxsample),
            ("done", self.done),
            ("is_final_decode", self.is_final_decode),
        ):
            if tuple(tensor.shape) != expected_vector_shape:
                raise ValueError(
                    f"{name} must have shape {expected_vector_shape}, "
                    f"got {tuple(tensor.shape)}"
                )

        expected_mask_shape = (
            batch_size,
            int(self.state_latent.shape[1]),
            1,
        )
        if tuple(self.node_mask.shape) != expected_mask_shape:
            raise ValueError(
                f"node_mask must have shape {expected_mask_shape}, "
                f"got {tuple(self.node_mask.shape)}"
            )
        if self.done.dtype != torch.bool:
            raise TypeError(f"done must be torch.bool, got {self.done.dtype}")
        if self.is_final_decode.dtype != torch.bool:
            raise TypeError(
                "is_final_decode must be torch.bool, "
                f"got {self.is_final_decode.dtype}"
            )
        if not torch.equal(self.done, self.is_final_decode):
            raise ValueError(
                "DiffSBDD v1 transitions require done == is_final_decode"
            )
        for name, tensor in (
            ("recorded_timestep", self.recorded_timestep),
            ("source_diffusion_timestep", self.source_diffusion_timestep),
            ("target_diffusion_timestep", self.target_diffusion_timestep),
            ("decision_index", self.decision_index),
            ("episode_index", self.episode_index),
            ("nodesxsample", self.nodesxsample),
        ):
            if tensor.dtype not in (torch.int32, torch.int64):
                raise TypeError(f"{name} must be an integer tensor, got {tensor.dtype}")
        if not bool(torch.isfinite(self.recorded_behavior_score).all()):
            raise ValueError("recorded_behavior_score must contain only finite values")
        if not bool(
            torch.logical_or(self.node_mask == 0, self.node_mask == 1).all()
        ):
            raise ValueError("node_mask must contain only zero or one")
        if not bool(
            torch.logical_or(self.action_mask == 0, self.action_mask == 1).all()
        ):
            raise ValueError("action_mask must contain only zero or one")
        expected_node_mask = (
            torch.arange(
                int(self.state_latent.shape[1]),
                device=self.device,
            ).unsqueeze(0)
            < self.nodesxsample.unsqueeze(1)
        ).to(dtype=self.node_mask.dtype).unsqueeze(-1)
        if not torch.equal(self.node_mask, expected_node_mask):
            raise ValueError("node_mask must match nodesxsample")

        expected_action_mask = self.node_mask.expand_as(self.action_mask).clone()
        final_rows = self.is_final_decode
        if bool(final_rows.any()):
            expected_action_mask[final_rows, :, 3:] = 0
        if not torch.equal(self.action_mask, expected_action_mask):
            raise ValueError(
                "action_mask must cover all ordinary latent dimensions and "
                "only xyz dimensions for final decode"
            )
        ordinary_rows = ~self.is_final_decode
        if bool(
            torch.any(
                self.source_diffusion_timestep[ordinary_rows]
                != self.target_diffusion_timestep[ordinary_rows] + 1
            )
        ):
            raise ValueError(
                "ordinary transitions require source timestep = target timestep + 1"
            )
        if bool(
            torch.any(
                self.recorded_timestep[ordinary_rows]
                != self.target_diffusion_timestep[ordinary_rows]
            )
        ):
            raise ValueError(
                "ordinary recorded timestep must equal target diffusion timestep"
            )
        if bool(final_rows.any()) and (
            bool(torch.any(self.recorded_timestep[final_rows] != 0))
            or bool(torch.any(self.source_diffusion_timestep[final_rows] != 0))
            or bool(torch.any(self.target_diffusion_timestep[final_rows] != -1))
        ):
            raise ValueError(
                "final decode requires recorded/source timestep 0 and target timestep -1"
            )
        if len(self.sample_ids) != batch_size:
            raise ValueError(
                f"sample_ids length {len(self.sample_ids)} "
                f"does not match batch size {batch_size}"
            )
        if self.reward is not None and tuple(self.reward.shape) != expected_vector_shape:
            raise ValueError(
                f"reward must have shape {expected_vector_shape}, "
                f"got {tuple(self.reward.shape)}"
            )
        if self.reward is not None and not bool(torch.isfinite(self.reward).all()):
            raise ValueError("reward must contain only finite values")

        tensor_devices = {
            tensor.device
            for tensor in self._tensor_values()
            if tensor is not None
        }
        if len(tensor_devices) != 1:
            raise ValueError(
                "all DenoisingTransitionBatch tensors must share one device, "
                f"got {sorted(str(device) for device in tensor_devices)}"
            )

    @property
    def batch_size(self) -> int:
        """Return the number of transitions."""
        return int(self.state_latent.shape[0])

    @property
    def device(self) -> torch.device:
        """Return the common tensor device."""
        return self.state_latent.device

    def to(
        self,
        device: str | torch.device,
        *,
        non_blocking: bool = False,
    ) -> "DenoisingTransitionBatch":
        """Return a copy with all tensors moved to ``device``."""
        target = torch.device(device)
        return DenoisingTransitionBatch(
            state_latent=self.state_latent.to(target, non_blocking=non_blocking),
            action_latent=self.action_latent.to(target, non_blocking=non_blocking),
            next_state_latent=self.next_state_latent.to(
                target,
                non_blocking=non_blocking,
            ),
            action_mask=self.action_mask.to(target, non_blocking=non_blocking),
            recorded_behavior_score=self.recorded_behavior_score.to(
                target,
                non_blocking=non_blocking,
            ),
            recorded_timestep=self.recorded_timestep.to(
                target,
                non_blocking=non_blocking,
            ),
            source_diffusion_timestep=self.source_diffusion_timestep.to(
                target,
                non_blocking=non_blocking,
            ),
            target_diffusion_timestep=self.target_diffusion_timestep.to(
                target,
                non_blocking=non_blocking,
            ),
            decision_index=self.decision_index.to(
                target,
                non_blocking=non_blocking,
            ),
            episode_index=self.episode_index.to(
                target,
                non_blocking=non_blocking,
            ),
            nodesxsample=self.nodesxsample.to(
                target,
                non_blocking=non_blocking,
            ),
            node_mask=self.node_mask.to(target, non_blocking=non_blocking),
            done=self.done.to(target, non_blocking=non_blocking),
            is_final_decode=self.is_final_decode.to(
                target,
                non_blocking=non_blocking,
            ),
            sample_ids=self.sample_ids,
            reward=(
                None
                if self.reward is None
                else self.reward.to(target, non_blocking=non_blocking)
            ),
        )

    def _tensor_values(self) -> tuple[torch.Tensor | None, ...]:
        """Return all tensor fields for invariant checks."""
        return (
            self.state_latent,
            self.action_latent,
            self.next_state_latent,
            self.action_mask,
            self.recorded_behavior_score,
            self.recorded_timestep,
            self.source_diffusion_timestep,
            self.target_diffusion_timestep,
            self.decision_index,
            self.episode_index,
            self.nodesxsample,
            self.node_mask,
            self.done,
            self.is_final_decode,
            self.reward,
        )
