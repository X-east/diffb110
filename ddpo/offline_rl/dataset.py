"""Read-only DiffSBDD trajectory shard access for offline RL.

The reader consumes the exact
``diffsbdd_frozen_policy_trajectory_v1`` payload emitted by
``ddpo/main_diffsbdd_collect_ddp.py``. It never writes to, augments, or
rewrites the source shard and does not depend on PPO replay utilities.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence

import torch

from .protocol import DenoisingEpisodeFacts, DenoisingTransitionBatch


TRAJECTORY_SHARD_FORMAT = "diffsbdd_frozen_policy_trajectory_v1"
REQUIRED_TENSOR_KEYS = (
    "latents",
    "logps",
    "timesteps",
    "nodesxsample",
    "x",
    "categorical",
)
REQUIRED_PAYLOAD_KEYS = (
    "format",
    *REQUIRED_TENSOR_KEYS,
    "sample_ids",
    "metadata",
    "reconstruction_rows",
)


class OfflineTrajectoryDataError(RuntimeError):
    """Raised when a trajectory shard violates the offline MDP contract."""


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_tensor(payload: dict[str, Any], key: str) -> torch.Tensor:
    """Return one required CPU tensor or raise a typed data error."""
    value = payload.get(key)
    if not isinstance(value, torch.Tensor):
        raise OfflineTrajectoryDataError(
            f"payload field {key!r} must be a torch.Tensor, got {type(value)!r}"
        )
    if value.device.type != "cpu":
        raise OfflineTrajectoryDataError(
            f"payload field {key!r} must load on CPU, got {value.device}"
        )
    return value


def _as_cpu_indices(
    values: Sequence[int] | torch.Tensor,
    *,
    name: str,
) -> torch.Tensor:
    """Normalize exact integer index inputs to a flat CPU long tensor."""
    raw = torch.as_tensor(values).detach().to(device="cpu").reshape(-1)
    if raw.dtype == torch.bool:
        raise OfflineTrajectoryDataError(f"{name} must contain integer indices")
    if torch.is_floating_point(raw):
        if not bool(torch.isfinite(raw).all()) or not torch.equal(raw, raw.round()):
            raise OfflineTrajectoryDataError(
                f"{name} must contain finite integer indices"
            )
    return raw.to(dtype=torch.long)


class DenoisingTrajectoryShard:
    """Validated in-memory view of one immutable trajectory shard."""

    def __init__(self, path: Path, payload: dict[str, Any]) -> None:
        """Initialize from a payload already loaded on CPU."""
        self.path = path
        self._source_sha256: str | None = None
        self.latents = _require_tensor(payload, "latents")
        self.logps = _require_tensor(payload, "logps")
        self.timesteps = _require_tensor(payload, "timesteps")
        self.nodesxsample = _require_tensor(payload, "nodesxsample")
        self.x = _require_tensor(payload, "x")
        self.categorical = _require_tensor(payload, "categorical")

        sample_ids = payload["sample_ids"]
        metadata = payload["metadata"]
        reconstruction_rows = payload["reconstruction_rows"]
        if not isinstance(sample_ids, (list, tuple)):
            raise OfflineTrajectoryDataError("sample_ids must be a list or tuple")
        if not isinstance(metadata, dict):
            raise OfflineTrajectoryDataError("metadata must be a dict")
        if not isinstance(reconstruction_rows, (list, tuple)):
            raise OfflineTrajectoryDataError(
                "reconstruction_rows must be a list or tuple"
            )

        self.sample_ids = tuple(str(value) for value in sample_ids)
        self.metadata = dict(metadata)
        self.reconstruction_rows = tuple(
            dict(row) if isinstance(row, dict) else row
            for row in reconstruction_rows
        )
        self._validate_structure()

    @property
    def batch_size(self) -> int:
        """Return the number of episodes in the shard."""
        return int(self.latents.shape[0])

    @property
    def num_actions(self) -> int:
        """Return the number of MDP actions per episode."""
        return int(self.logps.shape[1])

    @property
    def num_diffusion_steps(self) -> int:
        """Return the number of ordinary reverse-diffusion actions."""
        return int(self.metadata["num_timesteps"])

    @property
    def final_decision_index(self) -> int:
        """Return the decision index of ``z0 -> xh_final``."""
        return self.num_actions - 1

    @property
    def max_n_nodes(self) -> int:
        """Return the padded ligand-node dimension."""
        return int(self.metadata["max_n_nodes"])

    def source_sha256(self) -> str:
        """Return and cache the digest of the immutable source shard."""
        if self._source_sha256 is None:
            self._source_sha256 = _sha256_file(self.path)
        return self._source_sha256

    def episode_facts(self, episode_index: int) -> DenoisingEpisodeFacts:
        """Return immutable metadata for one episode."""
        index = self._validate_episode_index(episode_index)
        row = self.reconstruction_rows[index]
        if not isinstance(row, dict):
            raise OfflineTrajectoryDataError(
                f"reconstruction row {index} is not a dict"
            )
        smiles_value = row.get("smiles")
        smiles = None if smiles_value in (None, "") else str(smiles_value)
        return DenoisingEpisodeFacts(
            sample_id=self.sample_ids[index],
            shard_path=self.path,
            episode_index=index,
            rank=int(self.metadata["rank"]),
            shard_index=int(self.metadata["shard_index"]),
            seed=int(self.metadata["seed"]),
            num_nodes=int(self.nodesxsample[index].item()),
            num_diffusion_steps=self.num_diffusion_steps,
            max_n_nodes=self.max_n_nodes,
            condition=bool(self.metadata.get("condition", False)),
            rdkit_valid=bool(int(row.get("rdkit_valid", 0))),
            failure_type=str(row.get("failure_type", "unknown")),
            smiles=smiles,
            checkpoint_sha256=str(self.metadata["checkpoint_sha256"]),
        )

    def all_episode_facts(self) -> tuple[DenoisingEpisodeFacts, ...]:
        """Return metadata for every episode without expanding transitions."""
        return tuple(self.episode_facts(index) for index in range(self.batch_size))

    def select_transitions(
        self,
        episode_indices: Sequence[int] | torch.Tensor,
        decision_indices: Sequence[int] | torch.Tensor,
        *,
        episode_rewards: Sequence[float] | torch.Tensor | None = None,
    ) -> DenoisingTransitionBatch:
        """Select arbitrary episode/decision pairs as one structural batch.

        Args:
            episode_indices: Episode index for each requested transition.
            decision_indices: Decision index in ``[0, num_actions)``. The final
                index selects ``z0 -> xh_final``.
            episode_rewards: Optional terminal reward for every episode in this
                shard. Ordinary transitions receive zero; final transitions
                receive their corresponding episode reward.

        Returns:
            A CPU :class:`DenoisingTransitionBatch`.
        """
        episodes = _as_cpu_indices(episode_indices, name="episode_indices")
        decisions = _as_cpu_indices(decision_indices, name="decision_indices")
        if int(episodes.numel()) == 0:
            raise OfflineTrajectoryDataError("transition selection cannot be empty")
        if tuple(episodes.shape) != tuple(decisions.shape):
            raise OfflineTrajectoryDataError(
                "episode_indices and decision_indices must have identical lengths"
            )
        if bool(torch.any(episodes < 0)) or bool(torch.any(episodes >= self.batch_size)):
            raise OfflineTrajectoryDataError(
                f"episode index outside [0, {self.batch_size})"
            )
        if bool(torch.any(decisions < 0)) or bool(
            torch.any(decisions >= self.num_actions)
        ):
            raise OfflineTrajectoryDataError(
                f"decision index outside [0, {self.num_actions})"
            )

        state = self.latents[episodes, decisions]
        action = self.latents[episodes, decisions + 1]
        recorded_behavior_score = self.logps[episodes, decisions]
        recorded_timestep = self.timesteps[episodes, decisions]
        nodes = self.nodesxsample[episodes]
        node_ids = torch.arange(self.max_n_nodes).unsqueeze(0)
        node_mask = (
            (node_ids < nodes.unsqueeze(1))
            .to(dtype=state.dtype)
            .unsqueeze(-1)
        )
        is_final = decisions == self.final_decision_index
        source_diffusion_timestep = recorded_timestep + 1
        source_diffusion_timestep = torch.where(
            is_final,
            torch.zeros_like(source_diffusion_timestep),
            source_diffusion_timestep,
        )
        target_diffusion_timestep = torch.where(
            is_final,
            torch.full_like(recorded_timestep, -1),
            recorded_timestep,
        )
        action_mask = node_mask.expand_as(action).clone()
        action_mask[is_final, :, 3:] = 0

        reward: torch.Tensor | None = None
        episode_return: torch.Tensor | None = None
        if episode_rewards is not None:
            rewards = (
                torch.as_tensor(episode_rewards)
                .detach()
                .to(device="cpu", dtype=torch.float32)
                .reshape(-1)
            )
            if int(rewards.numel()) != self.batch_size:
                raise OfflineTrajectoryDataError(
                    f"episode_rewards length {int(rewards.numel())} "
                    f"does not match shard batch size {self.batch_size}"
                )
            if not bool(torch.isfinite(rewards).all()):
                raise OfflineTrajectoryDataError("episode_rewards must all be finite")
            reward = torch.where(
                is_final,
                rewards[episodes],
                rewards.new_zeros(is_final.shape),
            )
            episode_return = rewards[episodes]

        return DenoisingTransitionBatch(
            state_latent=state,
            action_latent=action,
            next_state_latent=action,
            action_mask=action_mask,
            recorded_behavior_score=recorded_behavior_score,
            recorded_timestep=recorded_timestep,
            source_diffusion_timestep=source_diffusion_timestep,
            target_diffusion_timestep=target_diffusion_timestep,
            decision_index=decisions,
            episode_index=episodes,
            nodesxsample=nodes,
            node_mask=node_mask,
            done=is_final,
            is_final_decode=is_final,
            sample_ids=tuple(self.sample_ids[int(index)] for index in episodes),
            reward=reward,
            episode_return=episode_return,
        )

    def select_transition(
        self,
        episode_index: int,
        decision_index: int,
        *,
        episode_reward: float | None = None,
    ) -> DenoisingTransitionBatch:
        """Select one transition while preserving a batch dimension."""
        index = self._validate_episode_index(episode_index)
        rewards: torch.Tensor | None = None
        if episode_reward is not None:
            rewards = torch.zeros(self.batch_size, dtype=torch.float32)
            rewards[index] = float(episode_reward)
        return self.select_transitions(
            [index],
            [decision_index],
            episode_rewards=rewards,
        )

    def validate_full_tensors(self, *, check_padding: bool = True) -> None:
        """Run finite-value and optional zero-padding validation.

        This scan is intended for dataset auditing and fixtures, not for every
        shard load in a high-throughput training loop.
        """
        for name, tensor in (
            ("latents", self.latents),
            ("logps", self.logps),
            ("x", self.x),
            ("categorical", self.categorical),
        ):
            if not bool(torch.isfinite(tensor).all()):
                raise OfflineTrajectoryDataError(f"{name} contains NaN or Inf")

        if not check_padding:
            return
        node_ids = torch.arange(self.max_n_nodes).unsqueeze(0)
        padding = node_ids >= self.nodesxsample.unsqueeze(1)
        latent_padding = padding[:, None, :, None].expand_as(self.latents)
        x_padding = padding[:, :, None].expand_as(self.x)
        categorical_padding = padding[:, :, None].expand_as(self.categorical)
        if bool(torch.any(self.latents[latent_padding] != 0)):
            raise OfflineTrajectoryDataError("latents contain nonzero padding")
        if bool(torch.any(self.x[x_padding] != 0)):
            raise OfflineTrajectoryDataError("x contains nonzero padding")
        if bool(torch.any(self.categorical[categorical_padding] != 0)):
            raise OfflineTrajectoryDataError("categorical contains nonzero padding")

    def _validate_episode_index(self, episode_index: int) -> int:
        """Return one in-range episode index."""
        normalized = _as_cpu_indices([episode_index], name="episode_index")
        index = int(normalized.item())
        if index < 0 or index >= self.batch_size:
            raise IndexError(
                f"episode_index {index} outside [0, {self.batch_size})"
            )
        return index

    def _validate_structure(self) -> None:
        """Validate the v1 shard shape and timestep contract."""
        if self.latents.ndim != 4:
            raise OfflineTrajectoryDataError(
                f"latents must have shape [B,L,N,D], got {tuple(self.latents.shape)}"
            )
        if self.logps.ndim != 2 or self.timesteps.ndim != 2:
            raise OfflineTrajectoryDataError(
                "logps and timesteps must have shape [B,A]"
            )
        if self.nodesxsample.ndim != 1:
            raise OfflineTrajectoryDataError("nodesxsample must have shape [B]")
        if self.x.ndim != 3 or int(self.x.shape[-1]) != 3:
            raise OfflineTrajectoryDataError("x must have shape [B,N,3]")
        if self.categorical.ndim != 3:
            raise OfflineTrajectoryDataError(
                "categorical must have shape [B,N,H]"
            )

        batch_size = int(self.latents.shape[0])
        action_count = int(self.logps.shape[1])
        max_nodes = int(self.latents.shape[2])
        feature_dim = int(self.latents.shape[3])
        expected_shapes = {
            "logps": (batch_size, action_count),
            "timesteps": (batch_size, action_count),
            "nodesxsample": (batch_size,),
            "x": (batch_size, max_nodes, 3),
            "categorical": (
                batch_size,
                max_nodes,
                int(self.categorical.shape[-1]),
            ),
        }
        actual_shapes = {
            "logps": tuple(self.logps.shape),
            "timesteps": tuple(self.timesteps.shape),
            "nodesxsample": tuple(self.nodesxsample.shape),
            "x": tuple(self.x.shape),
            "categorical": tuple(self.categorical.shape),
        }
        for name, expected in expected_shapes.items():
            if actual_shapes[name] != expected:
                raise OfflineTrajectoryDataError(
                    f"{name} shape mismatch: expected {expected}, "
                    f"got {actual_shapes[name]}"
                )
        if int(self.latents.shape[1]) != action_count + 1:
            raise OfflineTrajectoryDataError(
                "latents length must equal action count + 1"
            )
        if feature_dim != 3 + int(self.categorical.shape[-1]):
            raise OfflineTrajectoryDataError(
                "latent feature dimension must equal 3 + categorical dimension"
            )

        required_metadata = (
            "rank",
            "shard_index",
            "seed",
            "num_timesteps",
            "max_n_nodes",
            "checkpoint_sha256",
        )
        missing_metadata = [
            key for key in required_metadata if key not in self.metadata
        ]
        if missing_metadata:
            raise OfflineTrajectoryDataError(
                f"metadata missing required fields: {missing_metadata}"
            )
        if self.num_diffusion_steps + 1 != action_count:
            raise OfflineTrajectoryDataError(
                "action count must equal num_timesteps + final decode action"
            )
        if self.num_diffusion_steps <= 0:
            raise OfflineTrajectoryDataError("num_timesteps must be positive")
        if self.max_n_nodes != max_nodes:
            raise OfflineTrajectoryDataError(
                f"metadata max_n_nodes={self.max_n_nodes} "
                f"does not match tensor dimension {max_nodes}"
            )
        if bool(torch.any(self.nodesxsample <= 0)) or bool(
            torch.any(self.nodesxsample > max_nodes)
        ):
            raise OfflineTrajectoryDataError(
                f"nodesxsample must lie in [1, {max_nodes}]"
            )
        if self.timesteps.dtype not in (torch.int32, torch.int64):
            raise OfflineTrajectoryDataError(
                f"timesteps must be an integer tensor, got {self.timesteps.dtype}"
            )
        if self.nodesxsample.dtype not in (torch.int32, torch.int64):
            raise OfflineTrajectoryDataError(
                "nodesxsample must be an integer tensor, "
                f"got {self.nodesxsample.dtype}"
            )

        expected_timesteps = torch.cat(
            (
                torch.arange(
                    self.num_diffusion_steps - 1,
                    -1,
                    -1,
                    dtype=self.timesteps.dtype,
                ),
                torch.zeros(1, dtype=self.timesteps.dtype),
            )
        )
        if not torch.equal(
            self.timesteps,
            expected_timesteps.unsqueeze(0).expand(batch_size, -1),
        ):
            raise OfflineTrajectoryDataError(
                "timesteps must equal [T-1, ..., 0, 0] for every episode"
            )

        if len(self.sample_ids) != batch_size:
            raise OfflineTrajectoryDataError(
                f"sample_ids length {len(self.sample_ids)} "
                f"does not match batch size {batch_size}"
            )
        if len(set(self.sample_ids)) != batch_size:
            raise OfflineTrajectoryDataError(
                "sample_ids must be unique within a shard"
            )
        if len(self.reconstruction_rows) != batch_size:
            raise OfflineTrajectoryDataError(
                f"reconstruction_rows length {len(self.reconstruction_rows)} "
                f"does not match batch size {batch_size}"
            )
        row_sample_ids = []
        for index, row in enumerate(self.reconstruction_rows):
            if not isinstance(row, dict):
                raise OfflineTrajectoryDataError(
                    f"reconstruction row {index} is not a dict"
                )
            row_sample_ids.append(str(row.get("sample_id", "")))
        if tuple(row_sample_ids) != self.sample_ids:
            raise OfflineTrajectoryDataError(
                "reconstruction row sample_ids must match sample_ids in order"
            )


def load_trajectory_shard(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    validate_full_tensors: bool = False,
    check_padding: bool = True,
) -> DenoisingTrajectoryShard:
    """Load one trusted project shard on CPU without modifying it.

    Args:
        path: Path to a collector-produced ``.pt`` shard.
        expected_sha256: Optional manifest digest to verify before loading.
        validate_full_tensors: Whether to scan all tensor values for finiteness.
        check_padding: Whether a full validation also requires exact zero
            padding.

    Returns:
        A validated read-only shard view.

    Raises:
        OfflineTrajectoryDataError: If the file, digest, payload, or structural
            contract is invalid.
    """
    shard_path = Path(path)
    if not shard_path.is_file():
        raise OfflineTrajectoryDataError(f"trajectory shard not found: {shard_path}")
    if expected_sha256 is not None:
        actual_sha256 = _sha256_file(shard_path)
        if actual_sha256.lower() != str(expected_sha256).lower():
            raise OfflineTrajectoryDataError(
                f"SHA-256 mismatch for {shard_path}: "
                f"expected={expected_sha256} actual={actual_sha256}"
            )

    try:
        payload = torch.load(
            shard_path,
            map_location="cpu",
            weights_only=False,
        )
    except Exception as exc:
        raise OfflineTrajectoryDataError(
            f"failed to load trajectory shard {shard_path}"
        ) from exc
    if not isinstance(payload, dict):
        raise OfflineTrajectoryDataError(
            f"trajectory payload must be a dict, got {type(payload)!r}"
        )
    missing = [key for key in REQUIRED_PAYLOAD_KEYS if key not in payload]
    if missing:
        raise OfflineTrajectoryDataError(
            f"trajectory payload missing required fields: {missing}"
        )
    if payload["format"] != TRAJECTORY_SHARD_FORMAT:
        raise OfflineTrajectoryDataError(
            f"unsupported shard format {payload['format']!r}; "
            f"expected {TRAJECTORY_SHARD_FORMAT!r}"
        )

    shard = DenoisingTrajectoryShard(shard_path, payload)
    if expected_sha256 is not None:
        shard._source_sha256 = actual_sha256
    if validate_full_tensors:
        shard.validate_full_tensors(check_padding=check_padding)
    return shard
