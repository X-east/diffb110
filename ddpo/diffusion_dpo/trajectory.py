"""Typed access to saved DiffSBDD reverse-diffusion trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch


LATENT_DIM = 13
COORDINATE_DIM = 3


class TrajectoryProtocolError(ValueError):
    """Raised when a saved trajectory violates the DiffSBDD protocol."""


@dataclass(frozen=True)
class ReverseTransition:
    """One saved reverse transition from ``z_t`` to ``z_s``."""

    z_t: torch.Tensor
    z_s: torch.Tensor
    t: int
    s: int
    node_mask: torch.Tensor


@dataclass(frozen=True)
class TrajectorySample:
    """One unpadded DiffSBDD trajectory, excluding the final decoder state."""

    sample_id: str
    n_nodes: int
    states: torch.Tensor
    step_s: torch.Tensor
    terminal_decode: torch.Tensor

    @property
    def num_timesteps(self) -> int:
        """Return the number of reverse Gaussian transitions."""
        return int(self.states.shape[0] - 1)

    def transition(self, step_index: int) -> ReverseTransition:
        """Return transition ``z_t -> z_s`` at a reverse-step index."""
        if not 0 <= step_index < self.num_timesteps:
            raise IndexError(
                f"step_index={step_index} outside [0, {self.num_timesteps})"
            )
        s = int(self.step_s[step_index].item())
        t = s + 1
        if t <= s or t > self.num_timesteps:
            raise TrajectoryProtocolError(
                f"invalid reverse timestep pair t={t}, s={s} at index {step_index}"
            )
        node_mask = torch.ones(self.n_nodes, 1, dtype=self.states.dtype)
        return ReverseTransition(
            z_t=self.states[step_index],
            z_s=self.states[step_index + 1],
            t=t,
            s=s,
            node_mask=node_mask,
        )

    def transition_for_timestep(self, timestep_t: int) -> ReverseTransition:
        """Return the saved transition whose upper endpoint is ``timestep_t``."""
        if not 1 <= timestep_t <= self.num_timesteps:
            raise IndexError(f"timestep_t={timestep_t} outside [1, {self.num_timesteps}]")
        return self.transition(self.num_timesteps - int(timestep_t))


def collate_transitions(
    transitions: list[ReverseTransition],
    *,
    device: torch.device | str | None = None,
) -> dict[str, torch.Tensor]:
    """Pad equal-timestep transitions into the adapter's batch format."""
    if not transitions:
        raise ValueError("cannot collate an empty transition list")
    timesteps = {(item.t, item.s) for item in transitions}
    if len(timesteps) != 1:
        raise ValueError(f"a batch must use one timestep, got {sorted(timesteps)}")
    max_nodes = max(item.z_t.shape[0] for item in transitions)
    dtype = transitions[0].z_t.dtype
    target_device = torch.device(device) if device is not None else transitions[0].z_t.device
    z_t = torch.zeros(len(transitions), max_nodes, transitions[0].z_t.shape[-1], dtype=dtype, device=target_device)
    z_s = torch.zeros_like(z_t)
    node_mask = torch.zeros(len(transitions), max_nodes, 1, dtype=dtype, device=target_device)
    for index, item in enumerate(transitions):
        n_nodes = item.z_t.shape[0]
        z_t[index, :n_nodes] = item.z_t.to(target_device)
        z_s[index, :n_nodes] = item.z_s.to(target_device)
        node_mask[index, :n_nodes] = 1.0
    t, s = next(iter(timesteps))
    return {
        "z_t": z_t,
        "z_s": z_s,
        "node_mask": node_mask,
        "t": torch.full((len(transitions), 1), t, dtype=torch.long, device=target_device),
        "s": torch.full((len(transitions), 1), s, dtype=torch.long, device=target_device),
    }


def _require_tensor(payload: dict[str, object], key: str) -> torch.Tensor:
    """Return a required tensor from one trajectory shard."""
    value = payload.get(key)
    if not isinstance(value, torch.Tensor):
        raise TrajectoryProtocolError(f"payload[{key!r}] must be a torch.Tensor")
    return value


def load_trajectory_sample(
    shard_path: str | Path,
    sample_index: int,
    *,
    expected_timesteps: int = 1000,
    expected_latent_dim: int = LATENT_DIM,
) -> TrajectorySample:
    """Load and validate one saved DiffSBDD trajectory.

    The collector stores ``[z_T, z_{T-1}, ..., z_0, xh_final]`` and records
    the reverse-step ``s`` values as ``[T-1, ..., 0, 0]``.  The formal DPO
    protocol uses only the first ``T+1`` states, leaving ``xh_final`` out of
    the Gaussian reverse-transition score.
    """
    path = Path(shard_path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    latents = _require_tensor(payload, "latents")
    timesteps = _require_tensor(payload, "timesteps")
    nodes = _require_tensor(payload, "nodesxsample")
    sample_ids = payload.get("sample_ids")
    if not isinstance(sample_ids, list):
        raise TrajectoryProtocolError("payload['sample_ids'] must be a list")
    if not 0 <= sample_index < len(sample_ids):
        raise IndexError(f"sample_index={sample_index} outside shard")
    if latents.ndim != 4 or timesteps.ndim != 2 or nodes.ndim != 1:
        raise TrajectoryProtocolError(
            f"unexpected ranks latents={latents.ndim}, timesteps={timesteps.ndim}, nodes={nodes.ndim}"
        )
    n_nodes = int(nodes[sample_index].item())
    if not 1 <= n_nodes <= latents.shape[2]:
        raise TrajectoryProtocolError(f"invalid node count {n_nodes}")
    expected_latents = expected_timesteps + 2
    expected_steps = expected_timesteps + 1
    if tuple(latents.shape[1:]) != (expected_latents, latents.shape[2], expected_latent_dim):
        raise TrajectoryProtocolError(
            f"latent shape {tuple(latents.shape)} does not match T={expected_timesteps}, D={expected_latent_dim}"
        )
    if tuple(timesteps.shape[1:]) != (expected_steps,):
        raise TrajectoryProtocolError(
            f"timestep shape {tuple(timesteps.shape)} does not match T={expected_timesteps}"
        )
    states = latents[sample_index, : expected_timesteps + 1, :n_nodes].float().contiguous()
    step_s = timesteps[sample_index, :expected_timesteps].long().contiguous()
    terminal_decode = latents[sample_index, expected_timesteps + 1, :n_nodes].float().contiguous()
    if not torch.isfinite(states).all() or not torch.isfinite(terminal_decode).all():
        raise TrajectoryProtocolError(f"non-finite latent values in {path} index {sample_index}")
    expected_s = torch.arange(expected_timesteps - 1, -1, -1, dtype=torch.long)
    if not torch.equal(step_s.cpu(), expected_s):
        raise TrajectoryProtocolError(
            f"unexpected reverse timestep sequence at {path} index {sample_index}"
        )
    if states[:, :, :COORDINATE_DIM].shape[1] != n_nodes:
        raise TrajectoryProtocolError("unpadded state has an inconsistent node count")
    return TrajectorySample(
        sample_id=str(sample_ids[sample_index]),
        n_nodes=n_nodes,
        states=states,
        step_s=step_s,
        terminal_decode=terminal_decode,
    )


@lru_cache(maxsize=16)
def _cached_trajectory_sample(
    shard_path: str,
    sample_index: int,
    expected_timesteps: int,
) -> TrajectorySample:
    """Cache a bounded number of decoded samples for pair training."""
    return load_trajectory_sample(
        shard_path,
        sample_index,
        expected_timesteps=expected_timesteps,
    )


class TrajectoryStore:
    """Bounded-cache access to endpoint trajectory samples."""

    def __init__(self, *, expected_timesteps: int = 1000) -> None:
        """Configure the protocol version used for every loaded sample."""
        self.expected_timesteps = expected_timesteps

    def sample(self, shard_path: str, sample_index: int) -> TrajectorySample:
        """Load one validated trajectory sample from a shard."""
        return _cached_trajectory_sample(shard_path, int(sample_index), self.expected_timesteps)
