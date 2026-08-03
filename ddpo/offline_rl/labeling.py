"""Provenance-bound molecule labeling for offline DiffSBDD trajectories."""

from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from rdkit import Chem
from rdkit.Chem import Descriptors, FilterCatalog, QED, rdMolDescriptors

from ddpo.rewards.chemistry_guard_reward import ChemistryGuardReward
from ddpo.rewards.ring_complexity_reward import RingComplexityReward
from ddpo.rewards.interfaces import DockingResult
from ddpo.rewards.sa_reward import calculate_sa_score
from ddpo.utils.molecule_semantics import (
    FAILURE_FRAGMENTED,
    FAILURE_OPENBABEL_RECONSTRUCT,
    FAILURE_SANITIZE,
    FAILURE_SUCCESS,
)

from .dataset import TRAJECTORY_SHARD_FORMAT
from .index import EpisodeIndex, EpisodeIndexRow
from .reward_sidecar import (
    REWARD_SIDECAR_SCHEMA_VERSION,
    EpisodeRewardFacts,
    RewardSidecar,
    load_reward_sidecar,
    write_reward_sidecar,
)


LOGGER = logging.getLogger("OfflineLabeling")


@dataclass(frozen=True)
class ReconstructionFacts:
    """Collector reconstruction state recovered from its canonical CSV fields."""

    reconstruction_success: bool
    sanitize_success: bool
    rdkit_valid: bool
    failure_reason: str


@dataclass(frozen=True)
class PoseBustersLabelFacts:
    """One auditable PoseBusters outcome, including backend errors."""

    evaluated: bool
    valid: bool
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        """Validate PoseBusters success implications."""
        if self.valid and not self.evaluated:
            raise ValueError("PoseBusters valid requires evaluated=True")
        if self.valid and self.failure_reason:
            raise ValueError("PoseBusters valid result cannot have a failure reason")


@dataclass(frozen=True)
class DockingLabelFacts:
    """One docking result plus whether a ligand input reached the backend."""

    input_success: bool
    result: DockingResult
    pose_molecule: Chem.Mol | None = None

    def __post_init__(self) -> None:
        """Validate docking input and result implications."""
        if self.result.success and not self.input_success:
            raise ValueError("docking success requires input_success=True")
        if self.pose_molecule is not None and not self.result.success:
            raise ValueError("docking pose requires a successful docking result")


PoseBustersFunction = Callable[[Chem.Mol], PoseBustersLabelFacts]
DockingFunction = Callable[
    [Sequence[tuple[str, Chem.Mol]]],
    Mapping[str, DockingLabelFacts],
]


@dataclass(frozen=True)
class LabelPolicyConfig:
    """Finite reward formula and strict actual-good thresholds."""

    reward_version: str
    formal_labels: bool = False
    posebusters_enabled: bool = False
    docking_enabled: bool = False
    posebusters_input: str = "docked_pose"
    require_pb_valid_for_reward: bool = False
    failure_penalty: float = -2.0
    valid_base_reward: float = 0.0
    qed_weight: float = 0.0
    sa_weight: float = 0.0
    docking_weight: float = 0.0
    pb_valid_weight: float = 0.0
    docking_normalization_factor: float = -10.0
    actual_good_docking_max: float = -8.0
    actual_good_ligand_efficiency_min: float = 0.25
    actual_good_qed_min: float = 0.4
    actual_good_sa_max: float = 6.0
    actual_good_logp_min: float = -1.0
    actual_good_logp_max: float = 5.0
    actual_good_tpsa_min: float = 20.0
    actual_good_tpsa_max: float = 140.0
    actual_good_max_three_member_rings: int = 0
    actual_good_max_fused_rings: int = 4
    actual_good_max_complex_rings: int = 1
    actual_good_max_brenk: int = 0

    def __post_init__(self) -> None:
        """Validate finite controls and formal backend completeness."""
        if not self.reward_version.strip():
            raise ValueError("reward_version cannot be empty")
        numeric = (
            self.failure_penalty,
            self.valid_base_reward,
            self.qed_weight,
            self.sa_weight,
            self.docking_weight,
            self.pb_valid_weight,
            self.docking_normalization_factor,
            self.actual_good_docking_max,
            self.actual_good_ligand_efficiency_min,
            self.actual_good_qed_min,
            self.actual_good_sa_max,
            self.actual_good_logp_min,
            self.actual_good_logp_max,
            self.actual_good_tpsa_min,
            self.actual_good_tpsa_max,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("label policy numeric values must be finite")
        if self.docking_normalization_factor == 0:
            raise ValueError("docking_normalization_factor cannot be zero")
        if self.actual_good_logp_min > self.actual_good_logp_max:
            raise ValueError("actual-good LogP range is inverted")
        if self.actual_good_tpsa_min > self.actual_good_tpsa_max:
            raise ValueError("actual-good TPSA range is inverted")
        if self.formal_labels and not (
            self.posebusters_enabled and self.docking_enabled
        ):
            raise ValueError(
                "formal labels require both PoseBusters and docking"
            )
        if self.posebusters_input not in {"docked_pose", "reconstructed"}:
            raise ValueError(
                "posebusters_input must be 'docked_pose' or 'reconstructed'"
            )
        if self.formal_labels and self.posebusters_input != "docked_pose":
            raise ValueError("formal labels require PoseBusters on docked poses")


@dataclass(frozen=True)
class MoleculeDescriptorFacts:
    """Per-molecule descriptors reused by reward and actual-good labeling."""

    qed: float
    sa_raw: float
    logp: float
    tpsa: float
    hba: int
    hbd: int
    heavy_atom_count: int
    fragment_count: int
    carbon_fraction: float
    hetero_atom_count: int
    ring_count: int
    aromatic_ring_count: int
    three_member_ring_count: int
    fused_ring_count: int
    complex_ring_count: int
    brenk_count: int


def compute_molecule_descriptor_facts(mol: Chem.Mol) -> MoleculeDescriptorFacts:
    """Compute deterministic RDKit facts from one sanitized molecule."""
    guard = ChemistryGuardReward().compute_facts(mol)
    complexity = RingComplexityReward().compute_facts(mol)
    ring_sets = [set(ring) for ring in mol.GetRingInfo().AtomRings()]
    fused_members: set[int] = set()
    for left_index, left in enumerate(ring_sets):
        for right_index in range(left_index + 1, len(ring_sets)):
            if len(left & ring_sets[right_index]) >= 2:
                fused_members.update((left_index, right_index))
    complex_ring_count = int(complexity.max_fused_ring_system_size >= 3)
    complex_ring_count += int(complexity.spiro_atom_count > 0)
    complex_ring_count += int(complexity.bridgehead_atom_count > 2)
    catalog_params = FilterCatalog.FilterCatalogParams()
    catalog_params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
    brenk_count = len(FilterCatalog.FilterCatalog(catalog_params).GetMatches(mol))
    values = MoleculeDescriptorFacts(
        qed=float(QED.qed(mol)),
        sa_raw=float(calculate_sa_score(mol)),
        logp=float(Descriptors.MolLogP(mol)),
        tpsa=float(rdMolDescriptors.CalcTPSA(mol)),
        hba=int(rdMolDescriptors.CalcNumHBA(mol)),
        hbd=int(rdMolDescriptors.CalcNumHBD(mol)),
        heavy_atom_count=int(guard.heavy_atom_count),
        fragment_count=int(guard.fragment_count),
        carbon_fraction=float(guard.carbon_fraction),
        hetero_atom_count=int(guard.hetero_atom_count),
        ring_count=int(complexity.ring_count),
        aromatic_ring_count=int(complexity.aromatic_ring_count),
        three_member_ring_count=int(complexity.three_member_ring_count),
        fused_ring_count=len(fused_members),
        complex_ring_count=complex_ring_count,
        brenk_count=brenk_count,
    )
    if not all(
        math.isfinite(value)
        for value in (
            values.qed,
            values.sa_raw,
            values.logp,
            values.tpsa,
            values.carbon_fraction,
        )
    ):
        raise FloatingPointError("molecule descriptors contain NaN or Inf")
    return values


def _read_reconstruction_rows(path: Path) -> dict[str, dict[str, str]]:
    """Read one collector reconstruction CSV keyed by sample ID."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_sample = {row["sample_id"]: row for row in rows}
    if len(by_sample) != len(rows):
        raise ValueError(f"duplicate sample_id in {path}")
    return by_sample


def reconstruction_facts_from_row(row: Mapping[str, str]) -> ReconstructionFacts:
    """Recover canonical collector states without guessing from validity alone."""
    failure_reason = str(row.get("failure_type", "")).strip()
    rdkit_valid_text = str(row.get("rdkit_valid", "")).strip()
    if rdkit_valid_text not in {"0", "1"}:
        raise ValueError("collector rdkit_valid must be exactly 0 or 1")
    rdkit_valid = rdkit_valid_text == "1"
    if failure_reason == FAILURE_SUCCESS:
        if not rdkit_valid:
            raise ValueError("collector success row must have rdkit_valid=1")
        return ReconstructionFacts(True, True, True, FAILURE_SUCCESS)
    if rdkit_valid:
        raise ValueError(
            f"collector failure {failure_reason!r} cannot have rdkit_valid=1"
        )
    if failure_reason == FAILURE_FRAGMENTED:
        return ReconstructionFacts(True, True, False, failure_reason)
    if failure_reason == FAILURE_SANITIZE:
        return ReconstructionFacts(True, False, False, failure_reason)
    if failure_reason == FAILURE_OPENBABEL_RECONSTRUCT:
        return ReconstructionFacts(False, False, False, failure_reason)
    raise ValueError(
        f"unsupported collector failure_type {failure_reason!r}; refusing to infer state"
    )


def _read_sdf_molecules(path: Path) -> dict[str, Chem.Mol]:
    """Read collector molecules by their durable sample_id property."""
    by_sample: dict[str, Chem.Mol] = {}
    for molecule in Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False):
        if molecule is None or not molecule.HasProp("sample_id"):
            raise ValueError(f"SDF contains an unreadable or unnamed molecule: {path}")
        sample_id = molecule.GetProp("sample_id")
        if sample_id in by_sample:
            raise ValueError(f"duplicate sample_id {sample_id!r} in {path}")
        by_sample[sample_id] = molecule
    return by_sample


def _finite_reward(
    facts: MoleculeDescriptorFacts,
    *,
    pb_valid: bool,
    docking_result: DockingResult,
    config: LabelPolicyConfig,
) -> float:
    """Compute the configured continuous finite training reward."""
    if config.docking_enabled and not docking_result.success:
        return config.failure_penalty
    if (
        config.posebusters_enabled
        and config.require_pb_valid_for_reward
        and not pb_valid
    ):
        return config.failure_penalty
    sa_reward = (10.0 - facts.sa_raw) / 9.0
    docking_reward = (
        docking_result.score_raw / config.docking_normalization_factor
        if docking_result.success
        else 0.0
    )
    reward = (
        config.valid_base_reward
        + config.qed_weight * facts.qed
        + config.sa_weight * sa_reward
        + config.docking_weight * docking_reward
        + config.pb_valid_weight * float(pb_valid)
    )
    if not math.isfinite(reward):
        raise FloatingPointError("computed training reward is not finite")
    return float(reward)


def _actual_good(
    facts: MoleculeDescriptorFacts,
    *,
    pb_valid: bool,
    docking_result: DockingResult,
    config: LabelPolicyConfig,
) -> bool:
    """Apply the explicit strict conjunction used only as final label."""
    ligand_efficiency = (
        -docking_result.score_raw / facts.heavy_atom_count
        if docking_result.success and facts.heavy_atom_count > 0
        else float("-inf")
    )
    return bool(
        pb_valid
        and docking_result.success
        and docking_result.score_raw <= config.actual_good_docking_max
        and ligand_efficiency >= config.actual_good_ligand_efficiency_min
        and facts.qed >= config.actual_good_qed_min
        and facts.sa_raw <= config.actual_good_sa_max
        and config.actual_good_logp_min <= facts.logp <= config.actual_good_logp_max
        and config.actual_good_tpsa_min <= facts.tpsa <= config.actual_good_tpsa_max
        and facts.fragment_count == 1
        and facts.three_member_ring_count
        <= config.actual_good_max_three_member_rings
        and facts.fused_ring_count <= config.actual_good_max_fused_rings
        and facts.complex_ring_count <= config.actual_good_max_complex_rings
        and facts.brenk_count <= config.actual_good_max_brenk
    )


def label_trajectory_shard(
    rows: Sequence[EpisodeIndexRow],
    *,
    reconstruction_csv: str | Path,
    molecules_sdf: str | Path,
    config: LabelPolicyConfig,
    posebusters: PoseBustersFunction | None = None,
    docking: DockingFunction | None = None,
) -> tuple[EpisodeRewardFacts, ...]:
    """Label indexed episodes from one exact collector molecule shard."""
    if not rows:
        raise ValueError("cannot label an empty shard row set")
    if config.posebusters_enabled != (posebusters is not None):
        raise ValueError("PoseBusters enablement and callback disagree")
    if config.docking_enabled != (docking is not None):
        raise ValueError("docking enablement and callback disagree")
    reconstruction = _read_reconstruction_rows(Path(reconstruction_csv))
    molecules = _read_sdf_molecules(Path(molecules_sdf))
    requested_ids = {row.sample_id for row in rows}
    missing_reconstruction = requested_ids - set(reconstruction)
    extra_reconstruction = set(reconstruction) - requested_ids
    if missing_reconstruction or extra_reconstruction:
        raise ValueError(
            "reconstruction CSV coverage mismatch: "
            f"missing={sorted(missing_reconstruction)[:5]} "
            f"extra={sorted(extra_reconstruction)[:5]}"
        )
    expected_sdf_ids = {
        sample_id
        for sample_id, reconstruction_row in reconstruction.items()
        if reconstruction_facts_from_row(reconstruction_row).rdkit_valid
    }
    if set(molecules) != expected_sdf_ids:
        raise ValueError(
            "molecule SDF coverage mismatch: "
            f"missing={sorted(expected_sdf_ids - set(molecules))[:5]} "
            f"extra={sorted(set(molecules) - expected_sdf_ids)[:5]}"
        )

    descriptors: dict[str, MoleculeDescriptorFacts] = {}
    sanitized: list[tuple[str, Chem.Mol]] = []
    for row in rows:
        reconstruction_row = reconstruction[row.sample_id]
        reconstruction_facts = reconstruction_facts_from_row(reconstruction_row)
        if not reconstruction_facts.rdkit_valid:
            continue
        molecule = molecules.get(row.sample_id)
        if molecule is None:
            raise ValueError(
                f"valid reconstruction {row.sample_id!r} is missing from SDF"
            )
        molecule = Chem.Mol(molecule)
        Chem.SanitizeMol(molecule)
        descriptors[row.sample_id] = compute_molecule_descriptor_facts(molecule)
        sanitized.append((row.sample_id, molecule))

    LOGGER.info(
        "label stages prepared attempts=%d rdkit_valid=%d",
        len(rows),
        len(sanitized),
    )

    if docking is not None and sanitized:
        LOGGER.info("docking start molecules=%d", len(sanitized))
        docking_results: Mapping[str, DockingLabelFacts] = docking(sanitized)
        LOGGER.info(
            "docking completed inputs=%d success=%d poses=%d",
            sum(facts.input_success for facts in docking_results.values()),
            sum(facts.result.success for facts in docking_results.values()),
            sum(facts.pose_molecule is not None for facts in docking_results.values()),
        )
    else:
        docking_results = {}
    if docking is not None and set(docking_results) != {
        sample_id for sample_id, _ in sanitized
    }:
        raise ValueError("docking results do not exactly cover valid molecules")

    pb_labels: dict[str, PoseBustersLabelFacts] = {}
    pb_total = len(sanitized)
    for pb_index, (sample_id, molecule) in enumerate(sanitized, start=1):
        if posebusters is None:
            pb_labels[sample_id] = PoseBustersLabelFacts(False, False)
            continue
        pb_molecule = molecule
        if config.posebusters_input == "docked_pose":
            docking_facts = docking_results.get(sample_id)
            if docking_facts is None or not docking_facts.result.success:
                pb_labels[sample_id] = PoseBustersLabelFacts(
                    evaluated=False,
                    valid=False,
                    failure_reason="posebusters_not_run:docking_failed",
                )
                continue
            if docking_facts.pose_molecule is None:
                pb_labels[sample_id] = PoseBustersLabelFacts(
                    evaluated=False,
                    valid=False,
                    failure_reason="posebusters_not_run:docked_pose_missing",
                )
                continue
            pb_molecule = docking_facts.pose_molecule
        pb_labels[sample_id] = posebusters(pb_molecule)
        if pb_index == 1 or pb_index % 5 == 0 or pb_index == pb_total:
            LOGGER.info(
                "posebusters progress=%d/%d evaluated=%d valid=%d",
                pb_index,
                pb_total,
                sum(facts.evaluated for facts in pb_labels.values()),
                sum(facts.valid for facts in pb_labels.values()),
            )

    output: list[EpisodeRewardFacts] = []
    for row in rows:
        reconstruction_row = reconstruction[row.sample_id]
        reconstruction_facts = reconstruction_facts_from_row(reconstruction_row)
        facts = descriptors.get(row.sample_id)
        if facts is None:
            output.append(
                EpisodeRewardFacts(
                    sample_id=row.sample_id,
                    source_trajectory_format=TRAJECTORY_SHARD_FORMAT,
                    source_checkpoint_sha256=row.source_checkpoint_sha256,
                    source_shard_sha256=row.source_shard_sha256,
                    reward_schema_version=REWARD_SIDECAR_SCHEMA_VERSION,
                    reward_version=config.reward_version,
                    reconstruction_success=reconstruction_facts.reconstruction_success,
                    sanitize_success=reconstruction_facts.sanitize_success,
                    rdkit_valid=reconstruction_facts.rdkit_valid,
                    pb_evaluated=False,
                    pb_valid=False,
                    docking_input_success=False,
                    docking_success=False,
                    actual_good=False,
                    reward_total_train=config.failure_penalty,
                    failure_reason=reconstruction_facts.failure_reason,
                )
            )
            continue
        docking_facts = docking_results.get(
            row.sample_id,
            DockingLabelFacts(
                input_success=False,
                result=DockingResult.failed("disabled", "docking_not_run"),
            ),
        )
        docking_result = docking_facts.result
        pb_facts = pb_labels[row.sample_id]
        pb_valid = pb_facts.valid
        ligand_efficiency = (
            -docking_result.score_raw / facts.heavy_atom_count
            if docking_result.success and facts.heavy_atom_count > 0
            else None
        )
        actual_good = _actual_good(
            facts,
            pb_valid=pb_valid,
            docking_result=docking_result,
            config=config,
        )
        failure_reason = "success"
        if config.docking_enabled and not docking_facts.input_success:
            failure_reason = docking_result.failure_reason or "docking_input_failed"
        elif config.docking_enabled and not docking_result.success:
            failure_reason = docking_result.failure_reason or "docking_failed"
        elif config.posebusters_enabled and not pb_facts.evaluated:
            failure_reason = pb_facts.failure_reason or "posebusters_error"
        elif config.posebusters_enabled and not pb_facts.valid:
            failure_reason = pb_facts.failure_reason or "posebusters_invalid"
        output.append(
            EpisodeRewardFacts(
                sample_id=row.sample_id,
                source_trajectory_format=TRAJECTORY_SHARD_FORMAT,
                source_checkpoint_sha256=row.source_checkpoint_sha256,
                source_shard_sha256=row.source_shard_sha256,
                reward_schema_version=REWARD_SIDECAR_SCHEMA_VERSION,
                reward_version=config.reward_version,
                reconstruction_success=True,
                sanitize_success=True,
                rdkit_valid=True,
                pb_evaluated=pb_facts.evaluated,
                pb_valid=pb_valid,
                docking_input_success=docking_facts.input_success,
                docking_success=docking_result.success,
                actual_good=actual_good,
                reward_total_train=_finite_reward(
                    facts,
                    pb_valid=pb_valid,
                    docking_result=docking_result,
                    config=config,
                ),
                failure_reason=failure_reason,
                docking_raw=(
                    docking_result.score_raw if docking_result.success else None
                ),
                ligand_efficiency=ligand_efficiency,
                qed=facts.qed,
                sa_raw=facts.sa_raw,
                logp=facts.logp,
                tpsa=facts.tpsa,
                hba=facts.hba,
                hbd=facts.hbd,
                fragment_count=facts.fragment_count,
                carbon_fraction=facts.carbon_fraction,
                hetero_atom_count=facts.hetero_atom_count,
                ring_count=facts.ring_count,
                aromatic_ring_count=facts.aromatic_ring_count,
                three_member_ring_count=facts.three_member_ring_count,
                fused_ring_count=facts.fused_ring_count,
                complex_ring_count=facts.complex_ring_count,
                brenk_count=facts.brenk_count,
                reward_total_raw=_finite_reward(
                    facts,
                    pb_valid=pb_valid,
                    docking_result=docking_result,
                    config=config,
                ),
            )
        )
    return tuple(output)


def merge_reward_sidecar_shards(
    shard_paths: Iterable[str | Path],
    *,
    episode_index: EpisodeIndex,
    output_path: str | Path,
) -> RewardSidecar:
    """Merge complete shard labels only after exact index/provenance validation."""
    rows: list[EpisodeRewardFacts] = []
    for path in sorted(Path(value) for value in shard_paths):
        rows.extend(load_reward_sidecar(path).rows)
    sidecar = RewardSidecar(rows)
    expected = {row.sample_id: row for row in episode_index.rows}
    observed = {row.sample_id: row for row in sidecar.rows}
    if set(expected) != set(observed):
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        raise ValueError(
            "label coverage does not match episode index: "
            f"missing={missing[:5]} extra={extra[:5]}"
        )
    for sample_id, index_row in expected.items():
        label = observed[sample_id]
        if (
            label.source_checkpoint_sha256.lower()
            != index_row.source_checkpoint_sha256.lower()
            or label.source_shard_sha256.lower()
            != index_row.source_shard_sha256.lower()
        ):
            raise ValueError(f"label provenance mismatch for {sample_id!r}")
    write_reward_sidecar(output_path, sidecar.rows)
    return sidecar
