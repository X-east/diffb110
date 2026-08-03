"""Global episode indexes and deterministic leakage-free split manifests."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Iterable, Mapping, Sequence

from .dataset import OfflineTrajectoryDataError, load_trajectory_shard


EPISODE_INDEX_SCHEMA_VERSION = "diffsbdd_episode_index_v1"
SPLIT_MANIFEST_SCHEMA_VERSION = "diffsbdd_episode_split_v1"
VALID_SPLITS = ("train", "validation", "test")


class EpisodeIndexError(RuntimeError):
    """Raised when source manifests, episode indexes, or splits are invalid."""


@dataclass(frozen=True)
class EpisodeIndexRow:
    """One immutable trajectory episode locator and reconstruction summary."""

    index_schema_version: str
    source_run_id: str
    manifest_snapshot_sha256: str
    sample_id: str
    shard_path: str
    source_shard_sha256: str
    episode_index: int
    rank: int
    shard_index: int
    seed: int
    num_nodes: int
    num_diffusion_steps: int
    max_n_nodes: int
    rdkit_valid: bool
    failure_type: str
    smiles: str | None
    source_checkpoint_sha256: str

    def __post_init__(self) -> None:
        """Validate one index row without loading its shard."""
        if self.index_schema_version != EPISODE_INDEX_SCHEMA_VERSION:
            raise EpisodeIndexError(
                f"unsupported index schema {self.index_schema_version!r}"
            )
        for name in (
            "source_run_id",
            "manifest_snapshot_sha256",
            "sample_id",
            "shard_path",
            "source_shard_sha256",
            "source_checkpoint_sha256",
        ):
            if not str(getattr(self, name)).strip():
                raise EpisodeIndexError(f"{name} must be non-empty")
        for name in (
            "manifest_snapshot_sha256",
            "source_shard_sha256",
            "source_checkpoint_sha256",
        ):
            value = str(getattr(self, name))
            if len(value) != 64 or any(
                character not in "0123456789abcdefABCDEF" for character in value
            ):
                raise EpisodeIndexError(f"{name} must be a 64-character hex digest")
        relative_shard_path = Path(self.shard_path)
        if relative_shard_path.is_absolute() or ".." in relative_shard_path.parts:
            raise EpisodeIndexError(
                "shard_path must be a run-relative path without parent traversal"
            )
        for name in (
            "episode_index",
            "rank",
            "shard_index",
            "seed",
            "num_nodes",
            "num_diffusion_steps",
            "max_n_nodes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise EpisodeIndexError(f"{name} must be an integer")
        if min(self.episode_index, self.rank, self.shard_index) < 0:
            raise EpisodeIndexError(
                "episode_index, rank, and shard_index must be non-negative"
            )
        if self.seed <= 0:
            raise EpisodeIndexError("seed must be positive")
        if not 0 < self.num_nodes <= self.max_n_nodes:
            raise EpisodeIndexError("num_nodes must lie in [1,max_n_nodes]")
        if self.num_diffusion_steps <= 0:
            raise EpisodeIndexError("num_diffusion_steps must be positive")
        if not isinstance(self.rdkit_valid, bool):
            raise EpisodeIndexError("rdkit_valid must be bool")
        if not self.failure_type.strip():
            raise EpisodeIndexError("failure_type must be non-empty")
        if self.smiles is not None and not isinstance(self.smiles, str):
            raise EpisodeIndexError("smiles must be str or None")


@dataclass(frozen=True)
class EpisodeIndexProgressFacts:
    """One immutable index-build progress snapshot for local logging."""

    completed_shards: int
    total_shards: int
    indexed_episodes: int
    elapsed_seconds: float
    shards_per_second: float
    eta_seconds: float


@dataclass(frozen=True)
class SplitAssignment:
    """One episode-level deterministic split assignment."""

    split_schema_version: str
    split_version: str
    episode_index_sha256: str
    sample_id: str
    split: str
    stratum: str

    def __post_init__(self) -> None:
        """Validate one split row."""
        if self.split_schema_version != SPLIT_MANIFEST_SCHEMA_VERSION:
            raise EpisodeIndexError(
                f"unsupported split schema {self.split_schema_version!r}"
            )
        if not self.split_version.strip():
            raise EpisodeIndexError("split_version must be non-empty")
        if self.split not in VALID_SPLITS:
            raise EpisodeIndexError(f"unsupported split {self.split!r}")
        if not self.sample_id.strip() or not self.stratum.strip():
            raise EpisodeIndexError("sample_id and stratum must be non-empty")
        if len(self.episode_index_sha256) != 64 or any(
            character not in "0123456789abcdefABCDEF"
            for character in self.episode_index_sha256
        ):
            raise EpisodeIndexError(
                "episode_index_sha256 must be a 64-character hex digest"
            )


def _canonical_digest(rows: Iterable[object]) -> str:
    """Hash dataclass rows using stable compact JSON lines."""
    digest = hashlib.sha256()
    for row in rows:
        payload = asdict(row)  # type: ignore[arg-type]
        digest.update(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


class EpisodeIndex:
    """Validated global index with unique sample and shard episode locators."""

    def __init__(self, rows: Iterable[EpisodeIndexRow]) -> None:
        """Validate and store index rows in deterministic sample order."""
        ordered = tuple(sorted(rows, key=lambda row: row.sample_id))
        if not ordered:
            raise EpisodeIndexError("episode index cannot be empty")
        sample_ids = [row.sample_id for row in ordered]
        if len(sample_ids) != len(set(sample_ids)):
            raise EpisodeIndexError("episode index contains duplicate sample_id")
        locators = [
            (row.source_shard_sha256.lower(), row.episode_index)
            for row in ordered
        ]
        if len(locators) != len(set(locators)):
            raise EpisodeIndexError("episode index contains duplicate shard locator")
        for name, values in (
            ("source_run_id", {row.source_run_id for row in ordered}),
            (
                "manifest_snapshot_sha256",
                {row.manifest_snapshot_sha256 for row in ordered},
            ),
            (
                "source_checkpoint_sha256",
                {row.source_checkpoint_sha256 for row in ordered},
            ),
        ):
            if len(values) != 1:
                raise EpisodeIndexError(
                    f"episode index must contain exactly one {name}"
                )
        self._rows = ordered
        self._by_sample_id: Mapping[str, EpisodeIndexRow] = MappingProxyType(
            {row.sample_id: row for row in ordered}
        )
        self._sha256 = _canonical_digest(ordered)

    @property
    def rows(self) -> tuple[EpisodeIndexRow, ...]:
        """Return rows sorted by stable sample ID."""
        return self._rows

    @property
    def sha256(self) -> str:
        """Return the canonical content digest."""
        return self._sha256

    @property
    def source_run_id(self) -> str:
        """Return the indexed collector run identifier."""
        return self._rows[0].source_run_id

    def get(self, sample_id: str) -> EpisodeIndexRow:
        """Return one indexed episode or fail closed."""
        try:
            return self._by_sample_id[sample_id]
        except KeyError as exc:
            raise EpisodeIndexError(
                f"episode index missing sample_id {sample_id!r}"
            ) from exc


class SplitManifest:
    """Validated split assignments bound to one exact episode index."""

    def __init__(self, rows: Iterable[SplitAssignment]) -> None:
        """Validate uniqueness and consistent split provenance."""
        ordered = tuple(sorted(rows, key=lambda row: row.sample_id))
        if not ordered:
            raise EpisodeIndexError("split manifest cannot be empty")
        sample_ids = [row.sample_id for row in ordered]
        if len(sample_ids) != len(set(sample_ids)):
            raise EpisodeIndexError("split manifest contains duplicate sample_id")
        for name, values in (
            ("split_version", {row.split_version for row in ordered}),
            (
                "episode_index_sha256",
                {row.episode_index_sha256 for row in ordered},
            ),
        ):
            if len(values) != 1:
                raise EpisodeIndexError(
                    f"split manifest must contain exactly one {name}"
                )
        self._rows = ordered
        self._by_sample_id: Mapping[str, SplitAssignment] = MappingProxyType(
            {row.sample_id: row for row in ordered}
        )

    @property
    def rows(self) -> tuple[SplitAssignment, ...]:
        """Return assignments sorted by sample ID."""
        return self._rows

    def validate_against(self, episode_index: EpisodeIndex) -> None:
        """Require exact one-to-one coverage of one immutable episode index."""
        expected_ids = {row.sample_id for row in episode_index.rows}
        actual_ids = set(self._by_sample_id)
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        if missing or extra:
            raise EpisodeIndexError(
                "split/index sample coverage mismatch: "
                f"missing={missing[:5]} extra={extra[:5]}"
            )
        if self._rows[0].episode_index_sha256.lower() != episode_index.sha256:
            raise EpisodeIndexError("split manifest episode index SHA mismatch")

    def split_for(self, sample_id: str) -> str:
        """Return the assigned split for one episode."""
        try:
            return self._by_sample_id[sample_id].split
        except KeyError as exc:
            raise EpisodeIndexError(
                f"split manifest missing sample_id {sample_id!r}"
            ) from exc


def _read_manifest_snapshot(
    run_dir: Path,
) -> tuple[tuple[dict[str, object], ...], str]:
    """Read all current rank manifests once and hash the completed snapshot."""
    manifest_paths = sorted((run_dir / "manifests").glob("rank_*.jsonl"))
    if not manifest_paths:
        raise EpisodeIndexError(f"no rank manifests found under {run_dir}")
    rows: list[dict[str, object]] = []
    for path in manifest_paths:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
            parsed = [
                json.loads(line)
                for line in lines
                if line.strip()
            ]
        except (OSError, json.JSONDecodeError) as exc:
            raise EpisodeIndexError(f"failed to read manifest {path}") from exc
        for row in parsed:
            if not isinstance(row, dict):
                raise EpisodeIndexError(f"manifest {path} contains a non-object row")
            if row.get("status") == "completed":
                missing = {
                    "rank",
                    "shard_index",
                    "shard_path",
                    "shard_sha256",
                } - set(row)
                if missing:
                    raise EpisodeIndexError(
                        f"completed manifest row missing fields {sorted(missing)}"
                    )
                rows.append(row)
    if not rows:
        raise EpisodeIndexError("manifest snapshot contains no completed shards")
    rows.sort(key=lambda row: (int(row["rank"]), int(row["shard_index"])))
    locators = [(int(row["rank"]), int(row["shard_index"])) for row in rows]
    if len(locators) != len(set(locators)):
        raise EpisodeIndexError("manifest snapshot contains duplicate rank/shard")
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return tuple(rows), digest.hexdigest()


def build_episode_index(
    run_dir: str | Path,
    *,
    max_shards_per_rank: int | None = None,
    progress_every_shards: int = 100,
    progress_callback: Callable[[EpisodeIndexProgressFacts], None] | None = None,
) -> EpisodeIndex:
    """Build an exact index from a stable snapshot of completed manifests."""
    if progress_every_shards <= 0:
        raise EpisodeIndexError("progress_every_shards must be positive")
    source_dir = Path(run_dir).resolve()
    manifest_rows, snapshot_sha256 = _read_manifest_snapshot(source_dir)
    if max_shards_per_rank is not None:
        if max_shards_per_rank <= 0:
            raise EpisodeIndexError("max_shards_per_rank must be positive or None")
        per_rank_count: dict[int, int] = defaultdict(int)
        selected: list[dict[str, object]] = []
        for row in manifest_rows:
            rank = int(row["rank"])
            if per_rank_count[rank] < max_shards_per_rank:
                selected.append(row)
                per_rank_count[rank] += 1
        manifest_rows = tuple(selected)

    rows: list[EpisodeIndexRow] = []
    total_shards = len(manifest_rows)
    started = time.perf_counter()
    for completed_shards, manifest_row in enumerate(manifest_rows, start=1):
        try:
            relative_path = str(manifest_row["shard_path"])
            shard_sha256 = str(manifest_row["shard_sha256"])
            expected_rank = int(manifest_row["rank"])
            expected_shard_index = int(manifest_row["shard_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise EpisodeIndexError(
                f"malformed completed manifest row: {manifest_row}"
            ) from exc
        try:
            shard = load_trajectory_shard(
                source_dir / relative_path,
                expected_sha256=shard_sha256,
            )
        except OfflineTrajectoryDataError as exc:
            raise EpisodeIndexError(
                f"failed to validate indexed shard {relative_path}"
            ) from exc
        if int(shard.metadata["rank"]) != expected_rank or int(
            shard.metadata["shard_index"]
        ) != expected_shard_index:
            raise EpisodeIndexError(
                f"shard metadata locator mismatch for {relative_path}"
            )
        for facts in shard.all_episode_facts():
            rows.append(
                EpisodeIndexRow(
                    index_schema_version=EPISODE_INDEX_SCHEMA_VERSION,
                    source_run_id=source_dir.name,
                    manifest_snapshot_sha256=snapshot_sha256,
                    sample_id=facts.sample_id,
                    shard_path=relative_path,
                    source_shard_sha256=shard_sha256,
                    episode_index=facts.episode_index,
                    rank=facts.rank,
                    shard_index=facts.shard_index,
                    seed=facts.seed,
                    num_nodes=facts.num_nodes,
                    num_diffusion_steps=facts.num_diffusion_steps,
                    max_n_nodes=facts.max_n_nodes,
                    rdkit_valid=facts.rdkit_valid,
                    failure_type=facts.failure_type,
                    smiles=facts.smiles,
                    source_checkpoint_sha256=facts.checkpoint_sha256,
                )
            )
        if progress_callback is not None and (
            completed_shards % progress_every_shards == 0
            or completed_shards == total_shards
        ):
            elapsed_seconds = time.perf_counter() - started
            shards_per_second = completed_shards / max(elapsed_seconds, 1e-12)
            progress_callback(
                EpisodeIndexProgressFacts(
                    completed_shards=completed_shards,
                    total_shards=total_shards,
                    indexed_episodes=len(rows),
                    elapsed_seconds=elapsed_seconds,
                    shards_per_second=shards_per_second,
                    eta_seconds=(
                        (total_shards - completed_shards) / shards_per_second
                    ),
                )
            )
    return EpisodeIndex(rows)


def _allocate_split_counts(
    size: int,
    fractions: tuple[float, float, float],
) -> tuple[int, int, int]:
    """Allocate exact per-stratum counts with largest remainders."""
    raw = [size * fraction for fraction in fractions]
    counts = [math.floor(value) for value in raw]
    remaining = size - sum(counts)
    order = sorted(
        range(3),
        key=lambda index: (-(raw[index] - counts[index]), index),
    )
    for index in order[:remaining]:
        counts[index] += 1
    return counts[0], counts[1], counts[2]


def _allocate_stratified_counts(
    stratum_sizes: Mapping[str, int],
    fractions: tuple[float, float, float],
) -> dict[str, tuple[int, int, int]]:
    """Approximate each stratum while enforcing exact global split totals."""
    total_size = sum(stratum_sizes.values())
    global_targets = _allocate_split_counts(total_size, fractions)
    counts: dict[str, list[int]] = {}
    remaining_by_stratum: dict[str, int] = {}
    fractional_remainders: dict[tuple[str, int], float] = {}
    allocated_extra: set[tuple[str, int]] = set()
    for stratum in sorted(stratum_sizes):
        size = stratum_sizes[stratum]
        raw = [size * fraction for fraction in fractions]
        base = [math.floor(value) for value in raw]
        counts[stratum] = base
        remaining_by_stratum[stratum] = size - sum(base)
        for split_index in range(3):
            fractional_remainders[(stratum, split_index)] = (
                raw[split_index] - base[split_index]
            )

    column_counts = [
        sum(counts[stratum][split_index] for stratum in counts)
        for split_index in range(3)
    ]
    deficits = [
        global_targets[index] - column_counts[index]
        for index in range(3)
    ]
    while sum(remaining_by_stratum.values()) > 0:
        candidates = [
            (
                fractional_remainders[(stratum, split_index)],
                -split_index,
                stratum,
                split_index,
            )
            for stratum in sorted(stratum_sizes)
            for split_index in range(3)
            if remaining_by_stratum[stratum] > 0
            and deficits[split_index] > 0
            and (stratum, split_index) not in allocated_extra
        ]
        if not candidates:
            raise EpisodeIndexError(
                "failed to satisfy exact global split counts during stratification"
            )
        _, _, stratum, split_index = max(candidates)
        counts[stratum][split_index] += 1
        remaining_by_stratum[stratum] -= 1
        deficits[split_index] -= 1
        allocated_extra.add((stratum, split_index))
    if deficits != [0, 0, 0]:
        raise EpisodeIndexError(
            f"stratified split allocation left global deficits {deficits}"
        )
    return {
        stratum: (values[0], values[1], values[2])
        for stratum, values in counts.items()
    }


def build_split_manifest(
    episode_index: EpisodeIndex,
    *,
    split_version: str,
    salt: str,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    reward_strata: Mapping[str, str] | None = None,
) -> SplitManifest:
    """Assign whole episodes deterministically within validity/node/reward strata."""
    if not split_version.strip() or not salt:
        raise EpisodeIndexError("split_version and salt must be non-empty")
    if len(fractions) != 3 or any(value < 0 for value in fractions):
        raise EpisodeIndexError("split fractions must be three non-negative values")
    if not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise EpisodeIndexError("split fractions must sum to one")
    if reward_strata is not None:
        expected_ids = {row.sample_id for row in episode_index.rows}
        actual_ids = set(reward_strata)
        if expected_ids != actual_ids:
            raise EpisodeIndexError(
                "reward strata must cover the episode index exactly"
            )

    grouped: dict[str, list[EpisodeIndexRow]] = defaultdict(list)
    for row in episode_index.rows:
        reward_stratum = (
            "unscored"
            if reward_strata is None
            else str(reward_strata[row.sample_id]).strip()
        )
        if not reward_stratum:
            raise EpisodeIndexError("reward stratum must be non-empty")
        stratum = (
            f"reward={reward_stratum}|valid={int(row.rdkit_valid)}"
            f"|nodes={row.num_nodes}"
        )
        grouped[stratum].append(row)

    split_counts_by_stratum = _allocate_stratified_counts(
        {stratum: len(rows) for stratum, rows in grouped.items()},
        fractions,
    )
    assignments: list[SplitAssignment] = []
    for stratum in sorted(grouped):
        stratum_rows = sorted(
            grouped[stratum],
            key=lambda row: hashlib.sha256(
                f"{salt}|{row.sample_id}".encode("utf-8")
            ).hexdigest(),
        )
        train_count, validation_count, _ = split_counts_by_stratum[stratum]
        for position, row in enumerate(stratum_rows):
            if position < train_count:
                split = "train"
            elif position < train_count + validation_count:
                split = "validation"
            else:
                split = "test"
            assignments.append(
                SplitAssignment(
                    split_schema_version=SPLIT_MANIFEST_SCHEMA_VERSION,
                    split_version=split_version,
                    episode_index_sha256=episode_index.sha256,
                    sample_id=row.sample_id,
                    split=split,
                    stratum=stratum,
                )
            )
    manifest = SplitManifest(assignments)
    manifest.validate_against(episode_index)
    return manifest


def _write_jsonl_atomic(path: Path, rows: Iterable[object]) -> Path:
    """Atomically write dataclass rows as canonical JSON lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            for row in rows:
                handle.write(
                    json.dumps(
                        asdict(row),  # type: ignore[arg-type]
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except OSError as exc:
        raise EpisodeIndexError(f"failed to write JSONL {path}") from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return path


def write_episode_index(path: str | Path, episode_index: EpisodeIndex) -> Path:
    """Write a canonical episode index JSONL."""
    return _write_jsonl_atomic(Path(path), episode_index.rows)


def write_split_manifest(path: str | Path, manifest: SplitManifest) -> Path:
    """Write a canonical split manifest JSONL."""
    return _write_jsonl_atomic(Path(path), manifest.rows)


def _load_jsonl_dataclass(path: Path, row_type: type[object]) -> tuple[object, ...]:
    """Load exact dataclass fields from one JSONL artifact."""
    expected_fields = {field.name for field in fields(row_type)}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise EpisodeIndexError(f"failed to read JSONL {path}") from exc
    rows: list[object] = []
    for row_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EpisodeIndexError(
                f"invalid JSON at {path}:{row_number}"
            ) from exc
        if not isinstance(raw, dict) or set(raw) != expected_fields:
            raise EpisodeIndexError(
                f"schema mismatch at {path}:{row_number}"
            )
        try:
            rows.append(row_type(**raw))
        except (TypeError, ValueError) as exc:
            raise EpisodeIndexError(
                f"invalid row at {path}:{row_number}"
            ) from exc
    return tuple(rows)


def load_episode_index(path: str | Path) -> EpisodeIndex:
    """Load a validated episode index JSONL."""
    rows = _load_jsonl_dataclass(Path(path), EpisodeIndexRow)
    return EpisodeIndex(rows)  # type: ignore[arg-type]


def load_split_manifest(path: str | Path) -> SplitManifest:
    """Load a validated split manifest JSONL."""
    rows = _load_jsonl_dataclass(Path(path), SplitAssignment)
    return SplitManifest(rows)  # type: ignore[arg-type]
