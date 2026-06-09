"""Reward-only evaluation pipeline for DiffGui-generated SDF files.

This module is intentionally decoupled from the existing DiffSBDD rollout and
actor code.  It adapts RDKit molecules from an SDF into the minimal DataProto
shape expected by the shared reward stack, then writes reward-annotated CSV,
SDF, and JSON summary artifacts.
"""

from __future__ import annotations

import csv
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rdkit import Chem

from ddpo.rewards.factory import build_rewarder
from ddpo.runtime.protocol import DataProto


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SdfMoleculeRecord:
    """One SDF record plus its sanitized reward-ready molecule."""

    source_index: int
    original_mol: Chem.Mol | None
    reward_mol: Chem.Mol | None
    mol_id: str
    original_name: str
    rdkit_valid: int
    sanitize_status: str
    smiles: str


@dataclass(frozen=True)
class RewardOnlyOutputs:
    """Paths produced by the reward-only evaluation pipeline."""

    scored_sdf: Path
    rewards_csv: Path
    summary_json: Path
    manifest_json: Path


def run_reward_only_evaluation(
    config: dict[str, Any],
    *,
    project_root: Path,
    dataset_info: dict[str, Any],
    logger_obj: logging.Logger | None = None,
) -> dict[str, Any]:
    """Evaluate DiffGui SDF molecules with the shared reward factory.

    Args:
        config: Full YAML config dictionary.
        project_root: Repository root used for resolving relative paths.
        dataset_info: DiffSBDD dataset constants passed to reward factory.
        logger_obj: Optional logger used by reward construction and progress.

    Returns:
        JSON-serializable summary dictionary.
    """
    log = logger_obj or logger
    eval_cfg = config.get("diffgui_reward_only", {})
    input_sdf = _resolve_path(project_root, eval_cfg.get("input_sdf"))
    output_dir = _resolve_path(project_root, eval_cfg.get("output_dir", "analysis/diffgui_reward_only_stage1"))
    batch_size = int(eval_cfg.get("batch_size", 128))
    mol_id_prefix = str(eval_cfg.get("mol_id_prefix", "diffgui"))
    remove_hs = bool(eval_cfg.get("remove_hs", False))
    strict_connectivity = bool(eval_cfg.get("strict_connectivity", True))
    device = _resolve_torch_device(str(eval_cfg.get("device", "cpu")))

    if batch_size <= 0:
        raise ValueError("diffgui_reward_only.batch_size must be positive")
    if input_sdf is None:
        raise ValueError("diffgui_reward_only.input_sdf is required")
    if not input_sdf.exists():
        raise FileNotFoundError(f"Input SDF does not exist: {input_sdf}")

    outputs = _resolve_outputs(output_dir, eval_cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_sdf_records(
        input_sdf,
        mol_id_prefix=mol_id_prefix,
        remove_hs=remove_hs,
        strict_connectivity=strict_connectivity,
    )
    log.info("Loaded %d SDF records from %s", len(records), input_sdf)

    rewarder = build_rewarder(config, dataset_info, log)
    rows: list[dict[str, Any]] = []
    atom_dim = len(dataset_info.get("atom_decoder", []))
    atom_encoder = dict(dataset_info.get("atom_encoder", {}))

    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        data = build_reward_dataproto(
            chunk,
            atom_dim=atom_dim,
            atom_encoder=atom_encoder,
            device=device,
        )
        scored = rewarder.calculate_rewards(data)
        rows.extend(extract_reward_rows(chunk, scored))
        log.info("Scored records %d-%d / %d", start, start + len(chunk), len(records))

    written_sdf = write_scored_sdf(rows, records, outputs.scored_sdf)
    write_reward_csv(rows, outputs.rewards_csv)
    summary = build_summary(
        rows,
        input_sdf=input_sdf,
        outputs=outputs,
        written_sdf=written_sdf,
        config=config,
    )
    outputs.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    outputs.manifest_json.write_text(
        json.dumps(
            {
                "stage": "diffgui_reward_only_stage1",
                "input_sdf": str(input_sdf),
                "outputs": {
                    "scored_sdf": str(outputs.scored_sdf),
                    "rewards_csv": str(outputs.rewards_csv),
                    "summary_json": str(outputs.summary_json),
                },
                "decoupled_from": [
                    "ddpo/main_diffsbdd_ddp.py",
                    "ddpo/runtime/rollout.py",
                    "ddpo/runtime/actor.py",
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return summary


def load_sdf_records(
    input_sdf: Path,
    *,
    mol_id_prefix: str,
    remove_hs: bool,
    strict_connectivity: bool,
) -> list[SdfMoleculeRecord]:
    """Load SDF records and make sanitized reward molecules when possible."""
    supplier = Chem.SDMolSupplier(str(input_sdf), sanitize=False, removeHs=remove_hs)
    records: list[SdfMoleculeRecord] = []
    for index, mol in enumerate(supplier):
        mol_id = f"{mol_id_prefix}_{index:05d}"
        if mol is None:
            records.append(
                SdfMoleculeRecord(
                    source_index=index,
                    original_mol=None,
                    reward_mol=None,
                    mol_id=mol_id,
                    original_name="",
                    rdkit_valid=0,
                    sanitize_status="supplier_none",
                    smiles="",
                )
            )
            continue

        original_name = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
        reward_mol, status = sanitize_sdf_mol(mol, strict_connectivity=strict_connectivity)
        rdkit_valid = int(reward_mol is not None)
        smiles = _safe_smiles(reward_mol) if reward_mol is not None else ""
        records.append(
            SdfMoleculeRecord(
                source_index=index,
                original_mol=mol,
                reward_mol=reward_mol,
                mol_id=mol_id,
                original_name=original_name,
                rdkit_valid=rdkit_valid,
                sanitize_status=status,
                smiles=smiles,
            )
        )
    return records


def sanitize_sdf_mol(mol: Chem.Mol, *, strict_connectivity: bool) -> tuple[Chem.Mol | None, str]:
    """Return a sanitized copy suitable for molecule-level rewards."""
    candidate = Chem.Mol(mol)
    try:
        candidate.UpdatePropertyCache(strict=False)
        status = Chem.SanitizeMol(candidate, catchErrors=True)
    except Exception as exc:
        return None, f"sanitize_exception:{type(exc).__name__}"
    if status != Chem.SanitizeFlags.SANITIZE_NONE:
        return None, f"sanitize_failed:{int(status)}"
    if strict_connectivity:
        fragments = Chem.GetMolFrags(candidate)
        if len(fragments) > 1:
            return None, f"fragmented:{len(fragments)}"
    return candidate, "ok"


def build_reward_dataproto(
    records: list[SdfMoleculeRecord],
    *,
    atom_dim: int,
    atom_encoder: dict[str, int],
    device: torch.device,
) -> DataProto:
    """Build the minimal DataProto expected by shared molecule rewards."""
    batch_size = len(records)
    max_nodes = max([_record_num_atoms(record) for record in records] + [1])
    x = torch.zeros((batch_size, max_nodes, 3), dtype=torch.float32, device=device)
    categorical = torch.zeros((batch_size, max_nodes, atom_dim), dtype=torch.float32, device=device)
    nodesxsample = torch.zeros((batch_size,), dtype=torch.long, device=device)

    for row_idx, record in enumerate(records):
        mol = record.reward_mol or record.original_mol
        if mol is None:
            continue
        node_count = mol.GetNumAtoms()
        nodesxsample[row_idx] = int(node_count)
        _copy_conformer_positions(mol, x[row_idx])
        _copy_atom_features(mol, categorical[row_idx], atom_encoder)

    rdkit_valid = np.asarray([record.rdkit_valid for record in records], dtype=np.int64)
    source_index = np.asarray([record.source_index for record in records], dtype=np.int64)
    non_tensors = {
        "rdmols": [record.reward_mol for record in records],
        "rdkit_valid": rdkit_valid,
        "smiles": [record.smiles for record in records],
        "mol_id": [record.mol_id for record in records],
        "source_index": source_index,
        "sanitize_status": [record.sanitize_status for record in records],
        "original_name": [record.original_name for record in records],
    }
    data = DataProto.from_dict(
        tensors={
            "x": x,
            "categorical": categorical,
            "nodesxsample": nodesxsample,
        },
        non_tensors=non_tensors,
    )
    # DataProto.from_dict intentionally stores non-tensors as object arrays.
    # Shared StabilityAwareReward expects rdkit_valid to be numeric-convertible.
    data.non_tensor_batch["rdkit_valid"] = rdkit_valid
    data.non_tensor_batch["source_index"] = source_index
    return data


def extract_reward_rows(records: list[SdfMoleculeRecord], scored: DataProto) -> list[dict[str, Any]]:
    """Convert a scored DataProto chunk into CSV/SDF rows."""
    scalar_fields = _extract_scalar_batch_fields(scored)
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        row: dict[str, Any] = {
            "mol_id": record.mol_id,
            "source_index": record.source_index,
            "original_name": record.original_name,
            "rdkit_valid": record.rdkit_valid,
            "sanitize_status": record.sanitize_status,
            "smiles": record.smiles,
        }
        for key, values in scalar_fields.items():
            if index < len(values):
                row[key] = values[index]
        rows.append(row)
    return rows


def write_scored_sdf(
    rows: list[dict[str, Any]],
    records: list[SdfMoleculeRecord],
    output_sdf: Path,
) -> int:
    """Write readable input molecules with reward fields attached."""
    output_sdf.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(output_sdf))
    writer.SetKekulize(False)
    written = 0
    try:
        for row, record in zip(rows, records):
            if record.original_mol is None:
                continue
            mol = Chem.Mol(record.original_mol)
            mol.SetProp("_Name", _build_sdf_name(row))
            for key, value in row.items():
                _safe_set_prop(mol, key, value)
            writer.write(mol)
            written += 1
    finally:
        writer.close()
    return written


def write_reward_csv(rows: list[dict[str, Any]], output_csv: Path) -> None:
    """Write reward rows as a stable union-schema CSV."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _ordered_fieldnames(rows)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fieldnames})


def build_summary(
    rows: list[dict[str, Any]],
    *,
    input_sdf: Path,
    outputs: RewardOnlyOutputs,
    written_sdf: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Build a compact JSON summary for the reward-only run."""
    readable_records = sum(1 for row in rows if row["sanitize_status"] != "supplier_none")
    valid_rows = [row for row in rows if int(row.get("rdkit_valid", 0)) == 1]
    summary: dict[str, Any] = {
        "stage": "diffgui_reward_only_stage1",
        "input_sdf": str(input_sdf),
        "input_records": len(rows),
        "readable_records": readable_records,
        "rdkit_valid": len(valid_rows),
        "rdkit_invalid": len(rows) - len(valid_rows),
        "rdkit_valid_ratio": float(len(valid_rows) / len(rows)) if rows else 0.0,
        "sdf_written": written_sdf,
        "scored_sdf": str(outputs.scored_sdf),
        "rewards_csv": str(outputs.rewards_csv),
        "summary_json": str(outputs.summary_json),
        "reward_weights": config.get("reward", {}).get("weights", {}),
    }
    for key in _numeric_summary_keys(rows):
        values = _finite_values(rows, key)
        if values.size == 0:
            continue
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_median"] = float(np.median(values))
        summary[f"{key}_min"] = float(np.min(values))
        summary[f"{key}_max"] = float(np.max(values))
        summary[f"{key}_q10"] = float(np.quantile(values, 0.10))
        summary[f"{key}_q90"] = float(np.quantile(values, 0.90))
    return summary


def _resolve_outputs(output_dir: Path, eval_cfg: dict[str, Any]) -> RewardOnlyOutputs:
    """Resolve output file names from config."""
    return RewardOnlyOutputs(
        scored_sdf=output_dir / str(eval_cfg.get("scored_sdf", "diffgui_reward_only_scored.sdf")),
        rewards_csv=output_dir / str(eval_cfg.get("rewards_csv", "diffgui_reward_only_rewards.csv")),
        summary_json=output_dir / str(eval_cfg.get("summary_json", "diffgui_reward_only_summary.json")),
        manifest_json=output_dir / str(eval_cfg.get("manifest_json", "manifest.json")),
    )


def _resolve_path(project_root: Path, value: Any) -> Path | None:
    """Resolve a config path relative to the project root."""
    if value is None:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else project_root / path


def _resolve_torch_device(value: str) -> torch.device:
    """Resolve torch device with a CUDA availability guard."""
    if value.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested for reward-only DataProto but unavailable; using CPU.")
        return torch.device("cpu")
    return torch.device(value)


def _record_num_atoms(record: SdfMoleculeRecord) -> int:
    mol = record.reward_mol or record.original_mol
    return int(mol.GetNumAtoms()) if mol is not None else 0


def _copy_conformer_positions(mol: Chem.Mol, target: torch.Tensor) -> None:
    """Copy conformer coordinates into the padded coordinate tensor."""
    if mol.GetNumConformers() == 0:
        return
    conformer = mol.GetConformer()
    limit = min(mol.GetNumAtoms(), target.shape[0])
    for atom_idx in range(limit):
        pos = conformer.GetAtomPosition(atom_idx)
        target[atom_idx, 0] = float(pos.x)
        target[atom_idx, 1] = float(pos.y)
        target[atom_idx, 2] = float(pos.z)


def _copy_atom_features(mol: Chem.Mol, target: torch.Tensor, atom_encoder: dict[str, int]) -> None:
    """Copy one-hot atom features when symbols exist in the dataset encoder."""
    if target.shape[-1] == 0:
        return
    limit = min(mol.GetNumAtoms(), target.shape[0])
    for atom_idx in range(limit):
        symbol = mol.GetAtomWithIdx(atom_idx).GetSymbol()
        encoded = atom_encoder.get(symbol)
        if encoded is not None and 0 <= encoded < target.shape[-1]:
            target[atom_idx, encoded] = 1.0


def _safe_smiles(mol: Chem.Mol | None) -> str:
    """Return canonical SMILES or an empty marker on failure."""
    if mol is None:
        return ""
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return ""


def _extract_scalar_batch_fields(data: DataProto) -> dict[str, list[Any]]:
    """Extract batch fields with one scalar value per molecule."""
    output: dict[str, list[Any]] = {}
    batch_size = len(data)
    skip = {"x", "categorical"}
    for key, value in data.batch.items():
        if key in skip or not isinstance(value, torch.Tensor):
            continue
        tensor = value.detach().cpu()
        if tensor.shape[:1] != (batch_size,):
            continue
        if tensor.ndim == 1:
            output[key] = [_python_scalar(v) for v in tensor]
        elif tensor.ndim == 2 and tensor.shape[1] == 1:
            output[key] = [_python_scalar(v) for v in tensor[:, 0]]
    return output


def _python_scalar(value: torch.Tensor) -> int | float:
    """Convert a scalar tensor to a Python scalar."""
    item = value.item()
    if isinstance(item, int):
        return int(item)
    if isinstance(item, float):
        return float(item)
    return float(item)


def _ordered_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    """Return stable CSV field order with metadata first."""
    base = [
        "mol_id",
        "source_index",
        "original_name",
        "rdkit_valid",
        "sanitize_status",
        "smiles",
        "nodesxsample",
        "reward_total",
        "reward_total_raw",
        "stability",
    ]
    keys: list[str] = []
    for key in base:
        if any(key in row for row in rows):
            keys.append(key)
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    return keys


def _build_sdf_name(row: dict[str, Any]) -> str:
    """Build compact SDF molecule name carrying reward and validity."""
    reward = _short_float(row.get("reward_total"))
    valid = row.get("rdkit_valid", 0)
    status = row.get("sanitize_status", "")
    return f"{row.get('mol_id', 'diffgui')}|reward={reward}|valid={valid}|{status}"


def _safe_set_prop(mol: Chem.Mol, key: str, value: Any) -> None:
    """Set an SDF property using an RDKit-safe string."""
    mol.SetProp(str(key), _sdf_value(value))


def _sdf_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    return str(value)


def _csv_value(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    return value


def _short_float(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return "NA"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(numeric) or math.isinf(numeric):
        return "NA"
    return f"{numeric:.{digits}f}"


def _numeric_summary_keys(rows: list[dict[str, Any]]) -> list[str]:
    """Pick scalar numeric fields worth summarizing."""
    preferred = [
        "reward_total",
        "reward_total_raw",
        "qed_raw",
        "qed_reward",
        "sa_raw",
        "sa_reward",
        "chembl_classifier_reward",
        "chemistry_guard_reward",
        "heavy_atom_count",
        "carbon_fraction",
        "hetero_atom_count",
        "hetero_atom_fraction",
        "aromatic_ring_count",
        "three_member_ring_count",
        "fragment_count",
        "bad_bond_length_fraction",
        "halogen_bad_position_count",
        "chemistry_guard_violation_count",
        "stability",
    ]
    present = [key for key in preferred if any(key in row for row in rows)]
    extras = []
    for row in rows:
        for key, value in row.items():
            if key in present or key in extras:
                continue
            if _is_number_like(value):
                extras.append(key)
    return present + extras


def _finite_values(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    values: list[float] = []
    for row in rows:
        if key not in row:
            continue
        try:
            value = float(row[key])
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=np.float64)


def _is_number_like(value: Any) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric)
