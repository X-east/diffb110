"""Versioned reward sidecars and immutable trajectory joins for offline RL."""

from __future__ import annotations

import csv
import math
import os
import tempfile
from dataclasses import dataclass, fields
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Iterable, Mapping, Sequence

import torch

from .dataset import (
    TRAJECTORY_SHARD_FORMAT,
    DenoisingTrajectoryShard,
)
from .protocol import DenoisingTransitionBatch

if TYPE_CHECKING:
    from os import PathLike


REWARD_SIDECAR_SCHEMA_VERSION = "diffsbdd_episode_reward_sidecar_v2"
OPTIONAL_FLOAT_FIELDS = (
    "docking_raw",
    "ligand_efficiency",
    "qed",
    "sa_raw",
    "logp",
    "tpsa",
    "chembl_prior",
    "reward_total_raw",
)
OPTIONAL_INTEGER_FIELDS = (
    "hba",
    "hbd",
    "fragment_count",
    "hetero_atom_count",
    "ring_count",
    "aromatic_ring_count",
    "three_member_ring_count",
    "fused_ring_count",
    "complex_ring_count",
    "brenk_count",
)
OPTIONAL_FRACTION_FIELDS = ("carbon_fraction",)
BOOLEAN_FIELDS = (
    "reconstruction_success",
    "sanitize_success",
    "rdkit_valid",
    "pb_evaluated",
    "pb_valid",
    "docking_input_success",
    "docking_success",
    "actual_good",
)


class RewardSidecarError(RuntimeError):
    """Raised when reward facts or their trajectory join are invalid."""


@dataclass(frozen=True)
class EpisodeRewardFacts:
    """Versioned finite training reward and audit facts for one episode."""

    sample_id: str
    source_trajectory_format: str
    source_checkpoint_sha256: str
    source_shard_sha256: str
    reward_schema_version: str
    reward_version: str
    reconstruction_success: bool
    sanitize_success: bool
    rdkit_valid: bool
    pb_evaluated: bool
    pb_valid: bool
    docking_input_success: bool
    docking_success: bool
    actual_good: bool
    reward_total_train: float
    failure_reason: str
    docking_raw: float | None = None
    ligand_efficiency: float | None = None
    qed: float | None = None
    sa_raw: float | None = None
    logp: float | None = None
    tpsa: float | None = None
    hba: int | None = None
    hbd: int | None = None
    fragment_count: int | None = None
    carbon_fraction: float | None = None
    hetero_atom_count: int | None = None
    ring_count: int | None = None
    aromatic_ring_count: int | None = None
    three_member_ring_count: int | None = None
    fused_ring_count: int | None = None
    complex_ring_count: int | None = None
    brenk_count: int | None = None
    chembl_prior: float | None = None
    reward_total_raw: float | None = None

    def __post_init__(self) -> None:
        """Validate identifiers, numeric values, and success implications."""
        for name in (
            "sample_id",
            "source_trajectory_format",
            "source_checkpoint_sha256",
            "source_shard_sha256",
            "reward_schema_version",
            "reward_version",
        ):
            if not str(getattr(self, name)).strip():
                raise RewardSidecarError(f"{name} must be non-empty")
        if self.source_trajectory_format != TRAJECTORY_SHARD_FORMAT:
            raise RewardSidecarError(
                "unsupported source_trajectory_format "
                f"{self.source_trajectory_format!r}"
            )
        if self.reward_schema_version != REWARD_SIDECAR_SCHEMA_VERSION:
            raise RewardSidecarError(
                "unsupported reward_schema_version "
                f"{self.reward_schema_version!r}"
            )
        for name in ("source_checkpoint_sha256", "source_shard_sha256"):
            value = str(getattr(self, name))
            if len(value) != 64 or any(
                character not in "0123456789abcdefABCDEF" for character in value
            ):
                raise RewardSidecarError(f"{name} must be a 64-character hex digest")
        for name in BOOLEAN_FIELDS:
            if not isinstance(getattr(self, name), bool):
                raise RewardSidecarError(f"{name} must be bool")
        if (
            isinstance(self.reward_total_train, bool)
            or not isinstance(self.reward_total_train, (int, float))
            or not math.isfinite(float(self.reward_total_train))
        ):
            raise RewardSidecarError("reward_total_train must be finite")
        for name in (*OPTIONAL_FLOAT_FIELDS, *OPTIONAL_FRACTION_FIELDS):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise RewardSidecarError(
                    f"{name} must be a finite number or missing"
                )
        for name in OPTIONAL_INTEGER_FIELDS:
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise RewardSidecarError(
                    f"{name} must be a non-negative integer or missing"
                )
        if self.carbon_fraction is not None and not 0.0 <= self.carbon_fraction <= 1.0:
            raise RewardSidecarError("carbon_fraction must lie in [0,1]")
        if self.rdkit_valid and not (
            self.reconstruction_success and self.sanitize_success
        ):
            raise RewardSidecarError(
                "rdkit_valid requires reconstruction_success and sanitize_success"
            )
        if self.docking_success and not self.docking_input_success:
            raise RewardSidecarError(
                "docking_success requires docking_input_success"
            )
        if self.pb_valid and not self.pb_evaluated:
            raise RewardSidecarError("pb_valid requires pb_evaluated")
        if self.actual_good and not (
            self.rdkit_valid
            and self.pb_evaluated
            and self.pb_valid
            and self.docking_success
        ):
            raise RewardSidecarError(
                "actual_good requires rdkit_valid, pb_valid, and docking_success"
            )


@dataclass(frozen=True)
class RewardJoinedTrajectoryShard:
    """Read-only shard view with episode rewards ordered by shard sample IDs."""

    shard: DenoisingTrajectoryShard
    reward_rows: tuple[EpisodeRewardFacts, ...]
    episode_rewards: torch.Tensor

    def __post_init__(self) -> None:
        """Validate the ordered immutable join."""
        expected_size = self.shard.batch_size
        if len(self.reward_rows) != expected_size:
            raise RewardSidecarError(
                f"joined reward row count {len(self.reward_rows)} "
                f"does not match shard batch size {expected_size}"
            )
        if tuple(self.episode_rewards.shape) != (expected_size,):
            raise RewardSidecarError(
                "episode_rewards must have shape "
                f"({expected_size},), got {tuple(self.episode_rewards.shape)}"
            )
        if self.episode_rewards.device.type != "cpu":
            raise RewardSidecarError("joined episode_rewards must remain on CPU")
        if not bool(torch.isfinite(self.episode_rewards).all()):
            raise RewardSidecarError("joined episode_rewards must all be finite")
        if tuple(row.sample_id for row in self.reward_rows) != self.shard.sample_ids:
            raise RewardSidecarError(
                "joined reward rows must follow shard sample_id order"
            )

    def select_transitions(
        self,
        episode_indices: Sequence[int] | torch.Tensor,
        decision_indices: Sequence[int] | torch.Tensor,
    ) -> DenoisingTransitionBatch:
        """Select transitions with terminal-only rewards from the sidecar."""
        return self.shard.select_transitions(
            episode_indices,
            decision_indices,
            episode_rewards=self.episode_rewards,
        )

    def select_transition(
        self,
        episode_index: int,
        decision_index: int,
    ) -> DenoisingTransitionBatch:
        """Select one transition with its joined terminal reward."""
        return self.select_transitions([episode_index], [decision_index])


class RewardSidecar:
    """Validated in-memory episode reward index keyed by stable sample ID."""

    def __init__(self, rows: Iterable[EpisodeRewardFacts]) -> None:
        """Build a unique, single-version reward index."""
        ordered_rows = tuple(rows)
        if not ordered_rows:
            raise RewardSidecarError("reward sidecar cannot be empty")
        by_sample_id: dict[str, EpisodeRewardFacts] = {}
        for row in ordered_rows:
            if row.sample_id in by_sample_id:
                raise RewardSidecarError(
                    f"duplicate reward row for sample_id {row.sample_id!r}"
                )
            by_sample_id[row.sample_id] = row
        reward_versions = {row.reward_version for row in ordered_rows}
        if len(reward_versions) != 1:
            raise RewardSidecarError(
                "one reward sidecar must contain exactly one reward_version"
            )
        checkpoint_digests = {
            row.source_checkpoint_sha256 for row in ordered_rows
        }
        if len(checkpoint_digests) != 1:
            raise RewardSidecarError(
                "one reward sidecar must reference exactly one source checkpoint"
            )
        self._rows = ordered_rows
        self._by_sample_id: Mapping[str, EpisodeRewardFacts] = MappingProxyType(
            by_sample_id
        )

    @property
    def rows(self) -> tuple[EpisodeRewardFacts, ...]:
        """Return rows in sidecar file order."""
        return self._rows

    @property
    def reward_version(self) -> str:
        """Return the single reward formula version."""
        return self._rows[0].reward_version

    def get(self, sample_id: str) -> EpisodeRewardFacts:
        """Return one row or fail rather than silently supplying zero reward."""
        try:
            return self._by_sample_id[sample_id]
        except KeyError as exc:
            raise RewardSidecarError(
                f"reward sidecar missing sample_id {sample_id!r}"
            ) from exc

    def join_shard(
        self,
        shard: DenoisingTrajectoryShard,
    ) -> RewardJoinedTrajectoryShard:
        """Join one immutable shard by sample ID and verify exact provenance."""
        source_sha256 = shard.source_sha256()
        rows = tuple(self.get(sample_id) for sample_id in shard.sample_ids)
        for row in rows:
            if row.source_shard_sha256.lower() != source_sha256.lower():
                raise RewardSidecarError(
                    f"source shard SHA mismatch for sample_id {row.sample_id!r}"
                )
            if (
                row.source_checkpoint_sha256.lower()
                != str(shard.metadata["checkpoint_sha256"]).lower()
            ):
                raise RewardSidecarError(
                    f"checkpoint SHA mismatch for sample_id {row.sample_id!r}"
                )
        episode_rewards = torch.tensor(
            [row.reward_total_train for row in rows],
            dtype=torch.float32,
            device="cpu",
        )
        return RewardJoinedTrajectoryShard(
            shard=shard,
            reward_rows=rows,
            episode_rewards=episode_rewards,
        )


def _parse_bool(value: str, *, field_name: str, row_number: int) -> bool:
    """Parse one strict CSV boolean."""
    normalized = value.strip().lower()
    if normalized in {"1", "true"}:
        return True
    if normalized in {"0", "false"}:
        return False
    raise RewardSidecarError(
        f"row {row_number} field {field_name} must be 0/1/true/false"
    )


def _parse_optional_float(
    value: str,
    *,
    field_name: str,
    row_number: int,
) -> float | None:
    """Parse one nullable finite float."""
    if value.strip() == "":
        return None
    try:
        parsed = float(value)
    except ValueError as exc:
        raise RewardSidecarError(
            f"row {row_number} field {field_name} must be numeric or empty"
        ) from exc
    if not math.isfinite(parsed):
        raise RewardSidecarError(
            f"row {row_number} field {field_name} must be finite or empty"
        )
    return parsed


def _parse_optional_integer(
    value: str,
    *,
    field_name: str,
    row_number: int,
) -> int | None:
    """Parse one nullable non-negative integer."""
    if value.strip() == "":
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise RewardSidecarError(
            f"row {row_number} field {field_name} must be an integer or empty"
        ) from exc
    if parsed < 0:
        raise RewardSidecarError(
            f"row {row_number} field {field_name} must be non-negative or empty"
        )
    return parsed


def _row_from_csv(raw: Mapping[str, str], row_number: int) -> EpisodeRewardFacts:
    """Convert one DictReader row into validated reward facts."""
    values: dict[str, object] = {
        name: str(raw.get(name, "")).strip()
        for name in (
            "sample_id",
            "source_trajectory_format",
            "source_checkpoint_sha256",
            "source_shard_sha256",
            "reward_schema_version",
            "reward_version",
            "failure_reason",
        )
    }
    for name in BOOLEAN_FIELDS:
        values[name] = _parse_bool(
            str(raw.get(name, "")),
            field_name=name,
            row_number=row_number,
        )
    raw_train_reward = str(raw.get("reward_total_train", ""))
    parsed_train_reward = _parse_optional_float(
        raw_train_reward,
        field_name="reward_total_train",
        row_number=row_number,
    )
    if parsed_train_reward is None:
        raise RewardSidecarError(
            f"row {row_number} reward_total_train cannot be empty"
        )
    values["reward_total_train"] = parsed_train_reward
    for name in (*OPTIONAL_FLOAT_FIELDS, *OPTIONAL_FRACTION_FIELDS):
        values[name] = _parse_optional_float(
            str(raw.get(name, "")),
            field_name=name,
            row_number=row_number,
        )
    for name in OPTIONAL_INTEGER_FIELDS:
        values[name] = _parse_optional_integer(
            str(raw.get(name, "")),
            field_name=name,
            row_number=row_number,
        )
    return EpisodeRewardFacts(**values)  # type: ignore[arg-type]


def load_reward_sidecar(
    path: str | PathLike[str],
    *,
    expected_reward_version: str | None = None,
) -> RewardSidecar:
    """Load and validate a versioned episode reward CSV."""
    sidecar_path = Path(path)
    if not sidecar_path.is_file():
        raise RewardSidecarError(f"reward sidecar not found: {sidecar_path}")
    required_columns = {field.name for field in fields(EpisodeRewardFacts)}
    try:
        with sidecar_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            present_columns = set(reader.fieldnames or ())
            missing_columns = sorted(required_columns - present_columns)
            if missing_columns:
                raise RewardSidecarError(
                    f"reward sidecar missing required columns: {missing_columns}"
                )
            rows = tuple(
                _row_from_csv(raw, row_number)
                for row_number, raw in enumerate(reader, start=2)
            )
    except OSError as exc:
        raise RewardSidecarError(
            f"failed to read reward sidecar {sidecar_path}"
        ) from exc
    sidecar = RewardSidecar(rows)
    if (
        expected_reward_version is not None
        and sidecar.reward_version != expected_reward_version
    ):
        raise RewardSidecarError(
            "reward_version mismatch: "
            f"expected={expected_reward_version!r} "
            f"actual={sidecar.reward_version!r}"
        )
    return sidecar


def write_reward_sidecar(
    path: str | PathLike[str],
    rows: Iterable[EpisodeRewardFacts],
) -> Path:
    """Atomically write validated reward rows using the canonical CSV schema."""
    sidecar = RewardSidecar(rows)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    field_names = [field.name for field in fields(EpisodeRewardFacts)]
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=field_names)
            writer.writeheader()
            for row in sidecar.rows:
                values = {
                    field_name: getattr(row, field_name)
                    for field_name in field_names
                }
                for field_name in BOOLEAN_FIELDS:
                    values[field_name] = int(bool(values[field_name]))
                for field_name, value in tuple(values.items()):
                    if value is None:
                        values[field_name] = ""
                writer.writerow(values)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
    except OSError as exc:
        raise RewardSidecarError(
            f"failed to write reward sidecar {output_path}"
        ) from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return output_path
