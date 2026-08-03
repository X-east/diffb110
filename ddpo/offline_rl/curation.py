"""Auditable positive and negative episode curation for offline training."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Mapping

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from .index import (
    SPLIT_MANIFEST_SCHEMA_VERSION,
    EpisodeIndex,
    SplitAssignment,
    SplitManifest,
)
from .reward_sidecar import EpisodeRewardFacts, RewardSidecar


CURATION_MANIFEST_SCHEMA_VERSION = "diffsbdd_offline_curation_manifest_v1"
QUALITY_CLASSES = (
    "positive",
    "structural_negative",
    "target_negative",
    "property_negative",
    "technical_failure",
)
SPLIT_NAMES = ("train", "validation", "test")


class CurationError(RuntimeError):
    """Raised when curated labels are incomplete or internally inconsistent."""


@dataclass(frozen=True)
class CurationConfig:
    """Frozen thresholds used to derive disjoint training-quality classes."""

    curation_version: str
    docking_max: float = -8.0
    ligand_efficiency_min: float = 0.25
    qed_min: float = 0.4
    sa_max: float = 6.0
    logp_min: float = -1.0
    logp_max: float = 5.0
    tpsa_min: float = 20.0
    tpsa_max: float = 140.0
    max_three_member_rings: int = 0
    max_fused_rings: int = 4
    max_complex_rings: int = 1
    max_brenk: int = 0

    def __post_init__(self) -> None:
        """Validate class thresholds before any episode is assigned."""
        if not self.curation_version.strip():
            raise CurationError("curation_version cannot be empty")
        numeric = (
            self.docking_max,
            self.ligand_efficiency_min,
            self.qed_min,
            self.sa_max,
            self.logp_min,
            self.logp_max,
            self.tpsa_min,
            self.tpsa_max,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise CurationError("curation thresholds must be finite")
        if self.logp_min > self.logp_max or self.tpsa_min > self.tpsa_max:
            raise CurationError("curation descriptor ranges are inverted")
        integer_limits = (
            self.max_three_member_rings,
            self.max_fused_rings,
            self.max_complex_rings,
            self.max_brenk,
        )
        if any(isinstance(value, bool) or value < 0 for value in integer_limits):
            raise CurationError("curation structural limits must be non-negative")


@dataclass(frozen=True)
class EpisodeCurationFacts:
    """One immutable quality class and its intended offline-training roles."""

    curation_schema_version: str
    curation_version: str
    episode_index_sha256: str
    reward_sidecar_sha256: str
    sample_id: str
    quality_class: str
    actor_bc_keep: bool
    critic_keep: bool
    structure_label: int | None
    target_label: int | None
    primary_reason: str
    failed_checks: str
    canonical_smiles: str | None
    scaffold_key: str | None

    def __post_init__(self) -> None:
        """Validate one class row and fail closed on role-label conflicts."""
        if self.curation_schema_version != CURATION_MANIFEST_SCHEMA_VERSION:
            raise CurationError(
                f"unsupported curation schema {self.curation_schema_version!r}"
            )
        for name in (
            "curation_version",
            "episode_index_sha256",
            "reward_sidecar_sha256",
            "sample_id",
            "quality_class",
            "primary_reason",
        ):
            if not str(getattr(self, name)).strip():
                raise CurationError(f"{name} must be non-empty")
        for name in ("episode_index_sha256", "reward_sidecar_sha256"):
            value = str(getattr(self, name))
            if len(value) != 64 or any(
                character not in "0123456789abcdefABCDEF" for character in value
            ):
                raise CurationError(f"{name} must be a 64-character hex digest")
        if self.quality_class not in QUALITY_CLASSES:
            raise CurationError(f"unsupported quality_class {self.quality_class!r}")
        if not isinstance(self.actor_bc_keep, bool) or not isinstance(
            self.critic_keep, bool
        ):
            raise CurationError("training role flags must be bool")
        if self.structure_label not in (None, 0, 1):
            raise CurationError("structure_label must be 0, 1, or missing")
        if self.target_label not in (None, 0, 1):
            raise CurationError("target_label must be 0, 1, or missing")
        if self.quality_class == "positive" and not (
            self.actor_bc_keep
            and self.critic_keep
            and self.structure_label == 1
            and self.target_label == 1
        ):
            raise CurationError("positive rows require both training roles and labels")
        if self.quality_class == "structural_negative" and not (
            not self.actor_bc_keep
            and self.critic_keep
            and self.structure_label == 0
            and self.target_label is None
        ):
            raise CurationError("structural negatives have inconsistent roles")
        if self.quality_class == "target_negative" and not (
            not self.actor_bc_keep
            and self.critic_keep
            and self.structure_label == 1
            and self.target_label == 0
        ):
            raise CurationError("target negatives have inconsistent roles")
        if self.quality_class == "property_negative" and not (
            not self.actor_bc_keep
            and self.critic_keep
            and self.structure_label == 1
            and self.target_label == 1
        ):
            raise CurationError("property negatives have inconsistent roles")
        if self.quality_class == "technical_failure" and (
            self.actor_bc_keep
            or self.critic_keep
            or self.structure_label is not None
            or self.target_label is not None
        ):
            raise CurationError("technical failures must be excluded from training")


class CurationManifest:
    """Validated curation rows keyed by immutable episode sample ID."""

    def __init__(self, rows: Iterable[EpisodeCurationFacts]) -> None:
        """Require unique rows bound to one index, sidecar, and policy version."""
        ordered = tuple(sorted(rows, key=lambda row: row.sample_id))
        if not ordered:
            raise CurationError("curation manifest cannot be empty")
        sample_ids = [row.sample_id for row in ordered]
        if len(sample_ids) != len(set(sample_ids)):
            raise CurationError("curation manifest contains duplicate sample_id")
        for name in (
            "curation_version",
            "episode_index_sha256",
            "reward_sidecar_sha256",
        ):
            if len({getattr(row, name) for row in ordered}) != 1:
                raise CurationError(f"curation manifest mixes {name}")
        self._rows = ordered
        self._by_sample_id: Mapping[str, EpisodeCurationFacts] = MappingProxyType(
            {row.sample_id: row for row in ordered}
        )

    @property
    def rows(self) -> tuple[EpisodeCurationFacts, ...]:
        """Return rows in stable sample-ID order."""
        return self._rows

    @property
    def curation_version(self) -> str:
        """Return the single curation policy version."""
        return self._rows[0].curation_version

    @property
    def episode_index_sha256(self) -> str:
        """Return the canonical episode-index digest."""
        return self._rows[0].episode_index_sha256

    @property
    def reward_sidecar_sha256(self) -> str:
        """Return the exact sidecar file digest."""
        return self._rows[0].reward_sidecar_sha256

    def get(self, sample_id: str) -> EpisodeCurationFacts:
        """Return one curation row or fail closed."""
        try:
            return self._by_sample_id[sample_id]
        except KeyError as exc:
            raise CurationError(
                f"curation manifest missing sample_id {sample_id!r}"
            ) from exc

    def sample_ids_for_classes(self, classes: Iterable[str]) -> frozenset[str]:
        """Return IDs assigned to the requested explicit quality classes."""
        requested = frozenset(str(value) for value in classes)
        unknown = sorted(requested - set(QUALITY_CLASSES))
        if unknown:
            raise CurationError(f"unsupported requested quality classes: {unknown}")
        if not requested:
            raise CurationError("at least one quality class must be requested")
        selected = frozenset(
            row.sample_id for row in self._rows if row.quality_class in requested
        )
        if not selected:
            raise CurationError(
                f"requested quality classes contain no episodes: {sorted(requested)}"
            )
        return selected

    def validate_against(
        self,
        episode_index: EpisodeIndex,
        reward_sidecar: RewardSidecar,
        *,
        reward_sidecar_sha256: str,
    ) -> None:
        """Require exact coverage and provenance of the index and reward sidecar."""
        expected_ids = {row.sample_id for row in episode_index.rows}
        actual_ids = set(self._by_sample_id)
        if expected_ids != actual_ids:
            raise CurationError(
                "curation/index coverage mismatch: "
                f"missing={sorted(expected_ids - actual_ids)[:5]} "
                f"extra={sorted(actual_ids - expected_ids)[:5]}"
            )
        reward_ids = {row.sample_id for row in reward_sidecar.rows}
        if reward_ids != expected_ids:
            raise CurationError("reward sidecar does not cover the episode index")
        if self.episode_index_sha256.lower() != episode_index.sha256.lower():
            raise CurationError("curation episode-index SHA mismatch")
        if self.reward_sidecar_sha256.lower() != reward_sidecar_sha256.lower():
            raise CurationError("curation reward-sidecar SHA mismatch")

    def counts(self) -> dict[str, int]:
        """Return denominator-explicit counts for all registered classes."""
        observed = Counter(row.quality_class for row in self._rows)
        return {name: observed.get(name, 0) for name in QUALITY_CLASSES}


def _required_value(row: EpisodeRewardFacts, name: str) -> float | int:
    """Return one descriptor required after successful PB and docking."""
    value = getattr(row, name)
    if value is None:
        raise CurationError(
            f"sample {row.sample_id!r} is missing required fact {name!r}"
        )
    return value


def _scaffold(smiles: str | None, *, sample_id: str) -> tuple[str | None, str | None]:
    """Return canonical SMILES and a stable scaffold group for valid molecules."""
    if not smiles:
        return None, None
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise CurationError(f"indexed SMILES is unreadable for {sample_id!r}")
    canonical = Chem.MolToSmiles(molecule, canonical=True)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule)
    return canonical, scaffold if scaffold else f"acyclic:{canonical}"


def _failed_checks(
    row: EpisodeRewardFacts,
    config: CurationConfig,
) -> tuple[list[str], list[str], list[str]]:
    """Return structural, target, and molecular-property failures separately."""
    structural: list[str] = []
    target: list[str] = []
    properties: list[str] = []
    if int(_required_value(row, "fragment_count")) != 1:
        structural.append("fragment_count")
    if int(_required_value(row, "three_member_ring_count")) > config.max_three_member_rings:
        structural.append("three_member_ring_count")
    if int(_required_value(row, "fused_ring_count")) > config.max_fused_rings:
        structural.append("fused_ring_count")
    if int(_required_value(row, "complex_ring_count")) > config.max_complex_rings:
        structural.append("complex_ring_count")
    if int(_required_value(row, "brenk_count")) > config.max_brenk:
        structural.append("brenk_count")
    if float(_required_value(row, "docking_raw")) > config.docking_max:
        target.append("docking_raw")
    if float(_required_value(row, "ligand_efficiency")) < config.ligand_efficiency_min:
        target.append("ligand_efficiency")
    if float(_required_value(row, "qed")) < config.qed_min:
        properties.append("qed")
    if float(_required_value(row, "sa_raw")) > config.sa_max:
        properties.append("sa_raw")
    logp = float(_required_value(row, "logp"))
    if not config.logp_min <= logp <= config.logp_max:
        properties.append("logp")
    tpsa = float(_required_value(row, "tpsa"))
    if not config.tpsa_min <= tpsa <= config.tpsa_max:
        properties.append("tpsa")
    return structural, target, properties


def _classify(
    row: EpisodeRewardFacts,
    config: CurationConfig,
) -> tuple[str, bool, bool, int | None, int | None, str, str]:
    """Assign one disjoint class with conservative technical-failure handling."""
    if not row.rdkit_valid:
        reason = row.failure_reason or "rdkit_invalid"
        return "structural_negative", False, True, 0, None, reason, reason
    if not row.docking_input_success or not row.docking_success:
        reason = row.failure_reason or "docking_not_successful"
        return "technical_failure", False, False, None, None, reason, reason
    if not row.pb_evaluated:
        reason = row.failure_reason or "posebusters_not_evaluated"
        return "technical_failure", False, False, None, None, reason, reason
    if not row.pb_valid:
        reason = row.failure_reason or "posebusters_invalid"
        return "structural_negative", False, True, 0, None, reason, reason

    structural, target, properties = _failed_checks(row, config)
    all_failures = structural + target + properties
    if structural:
        return (
            "structural_negative",
            False,
            True,
            0,
            None,
            structural[0],
            ";".join(all_failures),
        )
    if target:
        return (
            "target_negative",
            False,
            True,
            1,
            0,
            target[0],
            ";".join(all_failures),
        )
    if properties:
        return (
            "property_negative",
            False,
            True,
            1,
            1,
            properties[0],
            ";".join(all_failures),
        )
    if not row.actual_good:
        raise CurationError(
            f"sample {row.sample_id!r} passes curation thresholds but actual_good=0; "
            "the label and curation policies do not match"
        )
    return "positive", True, True, 1, 1, "actual_good", ""


def build_curation_manifest(
    episode_index: EpisodeIndex,
    reward_sidecar: RewardSidecar,
    *,
    reward_sidecar_sha256: str,
    config: CurationConfig,
) -> CurationManifest:
    """Derive disjoint quality classes without modifying trajectories or rewards."""
    reward_ids = {row.sample_id for row in reward_sidecar.rows}
    index_ids = {row.sample_id for row in episode_index.rows}
    if reward_ids != index_ids:
        raise CurationError("reward sidecar must exactly cover the episode index")
    rows: list[EpisodeCurationFacts] = []
    for index_row in episode_index.rows:
        reward_row = reward_sidecar.get(index_row.sample_id)
        if reward_row.rdkit_valid != index_row.rdkit_valid:
            raise CurationError(
                f"RDKit-valid mismatch for sample {index_row.sample_id!r}"
            )
        classification = _classify(reward_row, config)
        canonical_smiles, scaffold_key = _scaffold(
            index_row.smiles if index_row.rdkit_valid else None,
            sample_id=index_row.sample_id,
        )
        rows.append(
            EpisodeCurationFacts(
                curation_schema_version=CURATION_MANIFEST_SCHEMA_VERSION,
                curation_version=config.curation_version,
                episode_index_sha256=episode_index.sha256,
                reward_sidecar_sha256=reward_sidecar_sha256,
                sample_id=index_row.sample_id,
                quality_class=classification[0],
                actor_bc_keep=classification[1],
                critic_keep=classification[2],
                structure_label=classification[3],
                target_label=classification[4],
                primary_reason=classification[5],
                failed_checks=classification[6],
                canonical_smiles=canonical_smiles,
                scaffold_key=scaffold_key,
            )
        )
    manifest = CurationManifest(rows)
    manifest.validate_against(
        episode_index,
        reward_sidecar,
        reward_sidecar_sha256=reward_sidecar_sha256,
    )
    return manifest


def _validate_split_fractions(
    fractions: tuple[float, float, float],
) -> None:
    """Validate train/validation/test fractions for grouped curation splits."""
    if len(fractions) != 3 or any(
        not math.isfinite(value) or value < 0.0 for value in fractions
    ):
        raise CurationError("split fractions must be three finite non-negative values")
    if not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise CurationError("split fractions must sum to one")


def build_curation_split_manifest(
    episode_index: EpisodeIndex,
    curation_manifest: CurationManifest,
    *,
    split_version: str,
    salt: str,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> SplitManifest:
    """Assign whole molecular scaffolds with deterministic class balancing.

    Valid molecules sharing one Murcko scaffold can never cross splits. Episodes
    without a valid molecular representation remain individually assignable so
    reconstruction failures still retain their fixed-attempt denominator.
    """
    if not split_version.strip() or not salt:
        raise CurationError("split_version and salt must be non-empty")
    _validate_split_fractions(fractions)
    index_ids = {row.sample_id for row in episode_index.rows}
    curation_ids = {row.sample_id for row in curation_manifest.rows}
    if index_ids != curation_ids:
        raise CurationError("curation split requires exact index coverage")
    if curation_manifest.episode_index_sha256.lower() != episode_index.sha256.lower():
        raise CurationError("curation split episode-index SHA mismatch")

    index_by_id = {row.sample_id: row for row in episode_index.rows}
    groups: dict[str, list[EpisodeCurationFacts]] = defaultdict(list)
    for row in curation_manifest.rows:
        group_key = (
            f"scaffold:{row.scaffold_key}"
            if row.scaffold_key is not None
            else f"episode:{row.sample_id}"
        )
        groups[group_key].append(row)

    total_class_counts = Counter(
        row.quality_class for row in curation_manifest.rows
    )
    total_episodes = len(curation_manifest.rows)
    target_totals = {
        split: total_episodes * fraction
        for split, fraction in zip(SPLIT_NAMES, fractions)
    }
    target_classes = {
        split: {
            quality_class: total_class_counts[quality_class] * fraction
            for quality_class in QUALITY_CLASSES
        }
        for split, fraction in zip(SPLIT_NAMES, fractions)
    }
    assigned_totals = Counter({split: 0 for split in SPLIT_NAMES})
    assigned_classes = {
        split: Counter({quality_class: 0 for quality_class in QUALITY_CLASSES})
        for split in SPLIT_NAMES
    }
    group_to_split: dict[str, str] = {}

    def objective() -> float:
        """Return normalized total and per-class deviation from split targets."""
        score = 0.0
        for split in SPLIT_NAMES:
            total_target = target_totals[split]
            score += (
                (assigned_totals[split] - total_target) ** 2
                / max(total_target, 1.0)
            )
            for quality_class in QUALITY_CLASSES:
                class_target = target_classes[split][quality_class]
                score += (
                    (
                        assigned_classes[split][quality_class]
                        - class_target
                    )
                    ** 2
                    / max(class_target, 1.0)
                )
        return score

    ordered_groups = sorted(
        groups,
        key=lambda group_key: (
            -len(groups[group_key]),
            hashlib.sha256(f"{salt}|{group_key}".encode("utf-8")).hexdigest(),
            group_key,
        ),
    )
    for group_key in ordered_groups:
        class_counts = Counter(row.quality_class for row in groups[group_key])
        candidates: list[tuple[float, int, str]] = []
        for split_index, split in enumerate(SPLIT_NAMES):
            assigned_totals[split] += len(groups[group_key])
            assigned_classes[split].update(class_counts)
            candidates.append((objective(), split_index, split))
            assigned_totals[split] -= len(groups[group_key])
            assigned_classes[split].subtract(class_counts)
        _, _, selected_split = min(candidates)
        group_to_split[group_key] = selected_split
        assigned_totals[selected_split] += len(groups[group_key])
        assigned_classes[selected_split].update(class_counts)

    assignments: list[SplitAssignment] = []
    for group_key in sorted(groups):
        split = group_to_split[group_key]
        group_digest = hashlib.sha256(group_key.encode("utf-8")).hexdigest()[:16]
        for curation_row in groups[group_key]:
            index_row = index_by_id[curation_row.sample_id]
            assignments.append(
                SplitAssignment(
                    split_schema_version=SPLIT_MANIFEST_SCHEMA_VERSION,
                    split_version=split_version,
                    episode_index_sha256=episode_index.sha256,
                    sample_id=curation_row.sample_id,
                    split=split,
                    stratum=(
                        f"quality={curation_row.quality_class}"
                        f"|valid={int(index_row.rdkit_valid)}"
                        f"|nodes={index_row.num_nodes}"
                        f"|group={group_digest}"
                    ),
                )
            )
    manifest = SplitManifest(assignments)
    manifest.validate_against(episode_index)
    for group_key, group_rows in groups.items():
        observed = {
            manifest.split_for(row.sample_id) for row in group_rows
        }
        if len(observed) != 1:
            raise CurationError(
                f"curation scaffold group crosses splits: {group_key!r}"
            )
    return manifest


def write_curation_manifest(path: str | Path, manifest: CurationManifest) -> Path:
    """Atomically write a canonical curation JSONL artifact."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            for row in manifest.rows:
                handle.write(
                    json.dumps(asdict(row), ensure_ascii=False, sort_keys=True) + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
    except OSError as exc:
        raise CurationError(f"failed to write curation manifest {output_path}") from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return output_path


def load_curation_manifest(path: str | Path) -> CurationManifest:
    """Load an exact-schema curation JSONL artifact."""
    source = Path(path)
    expected_fields = {field.name for field in fields(EpisodeCurationFacts)}
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise CurationError(f"failed to read curation manifest {source}") from exc
    rows: list[EpisodeCurationFacts] = []
    for row_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CurationError(f"invalid JSON at {source}:{row_number}") from exc
        if not isinstance(raw, dict) or set(raw) != expected_fields:
            raise CurationError(f"schema mismatch at {source}:{row_number}")
        try:
            rows.append(EpisodeCurationFacts(**raw))
        except (TypeError, ValueError) as exc:
            raise CurationError(f"invalid row at {source}:{row_number}") from exc
    return CurationManifest(rows)


def sha256_file(path: str | Path) -> str:
    """Return the SHA256 digest of one curation input or output file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
