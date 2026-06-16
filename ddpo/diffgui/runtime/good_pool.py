"""Good-molecule metadata pool for DiffGui training runs."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rdkit import Chem

from ddpo.diffgui.runtime.training_filter import TrainingFilterSelection


DEFAULT_SCORE_KEY = "reward_total"
DEFAULT_RECORD_FIELDS = (
    "reward_total",
    "reward_total_raw",
    "reward_for_advantage",
    "advantage",
    "docking_raw",
    "docking_reward",
    "ligand_efficiency_raw",
    "rdkit_valid",
    "reconstruction_success",
    "fragment_count",
    "ring_count",
    "aromatic_ring_count",
    "three_member_ring_count",
    "max_fused_ring_system_size",
    "spiro_atom_count",
    "bridgehead_atom_count",
)


@dataclass
class GoodPoolSelection:
    """Selected good-molecule graph indices and records."""

    indices: list[int] = field(default_factory=list)
    records: list[dict[str, Any]] = field(default_factory=list)
    retained_records: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    def __len__(self) -> int:
        """Return selected count."""
        return len(self.indices)


class DiffGuiGoodMoleculePool:
    """Append-only rank-local good-molecule metadata pool."""

    def __init__(
        self,
        root_dir: str | Path,
        config: dict[str, Any] | None = None,
        *,
        rank: int = 0,
    ) -> None:
        """Initialize from ``diffgui_good_molecule_pool`` YAML config."""
        cfg = config or {}
        self.root_dir = Path(root_dir)
        self.rank = int(rank)
        self.enabled = bool(cfg.get("enabled", False))
        self.top_k_per_rank_per_epoch = max(
            0,
            int(cfg.get("top_k_per_rank_per_epoch", cfg.get("top_k_per_epoch", cfg.get("store_top_k", 4)))),
        )
        self.retained_top_k = max(0, int(cfg.get("retained_top_k", 128)))
        self.retention_window_epochs = max(1, int(cfg.get("retention_window_epochs", 10)))
        self.score_key = str(cfg.get("score_key", DEFAULT_SCORE_KEY))
        self.min_score = cfg.get("min_score")
        self.require_training_keep = bool(cfg.get("require_training_keep", True))
        self.require_rdkit_valid = bool(cfg.get("require_rdkit_valid", True))
        self.require_reconstruction_success = bool(cfg.get("require_reconstruction_success", True))
        self.deduplicate_smiles = bool(cfg.get("deduplicate_smiles", True))
        self.deduplicate_retained_smiles = bool(cfg.get("deduplicate_retained_smiles", True))
        self.quality_filters = dict(cfg.get("quality_filters", {}) or {})
        self.missing_quality_filter_policy = str(cfg.get("missing_quality_filter_policy", "ignore"))
        self.record_fields = tuple(
            dict.fromkeys(str(field) for field in (*cfg.get("record_fields", DEFAULT_RECORD_FIELDS), *self.quality_filters))
        )
        self.manifest_path = self.root_dir / f"good_molecules_rank{self.rank:02d}.jsonl"
        self.retained_jsonl_path = self.root_dir / f"good_molecules_retained_rank{self.rank:02d}.jsonl"
        self.retained_csv_path = self.root_dir / f"good_molecules_retained_rank{self.rank:02d}.csv"
        if self.enabled:
            self.root_dir.mkdir(parents=True, exist_ok=True)

    def add_epoch(
        self,
        *,
        molecule_rows: list[dict[str, Any]],
        training_selection: TrainingFilterSelection,
        epoch: int,
        generated_sdf: Path | None = None,
        top_sdf_path: Path | None = None,
    ) -> GoodPoolSelection:
        """Select good molecules from one epoch and persist pool artifacts."""
        selection = self.select(molecule_rows=molecule_rows, training_selection=training_selection, epoch=epoch)
        if not self.enabled:
            return selection
        if selection.records:
            with self.manifest_path.open("a", encoding="utf-8") as handle:
                for record in selection.records:
                    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        retained = self.retained_records(epoch)
        selection.retained_records = retained
        self._write_jsonl(self.retained_jsonl_path, retained)
        self._write_csv(self.retained_csv_path, retained)
        if generated_sdf is not None and top_sdf_path is not None:
            self.write_top_sdf(generated_sdf=generated_sdf, output_path=top_sdf_path, records=selection.records)
        selection.metrics.update(self._retained_metrics(retained))
        return selection

    def select(
        self,
        *,
        molecule_rows: list[dict[str, Any]],
        training_selection: TrainingFilterSelection,
        epoch: int,
    ) -> GoodPoolSelection:
        """Return top good molecules and compact records."""
        if self.top_k_per_rank_per_epoch <= 0:
            return GoodPoolSelection(metrics={"good_pool_selected_count": 0.0})
        training_kept = set(training_selection.train_indices)
        candidates: list[tuple[int, float, dict[str, Any]]] = []
        for fallback_index, row in enumerate(molecule_rows):
            graph_index = _row_index(row, fallback=fallback_index)
            score = _score_value(row, self.score_key)
            if score is None and self.score_key != "reward_for_advantage":
                score = _score_value(row, "reward_for_advantage")
            if score is None:
                continue
            if self.min_score is not None and score < float(self.min_score):
                continue
            if self.require_training_keep and graph_index not in training_kept:
                continue
            if self.require_reconstruction_success and _row_int(row, "reconstruction_success") != 1:
                continue
            if self.require_rdkit_valid and _row_int(row, "rdkit_valid") != 1:
                continue
            if not self._passes_quality_filters(row):
                continue
            record = self._build_record(row=row, graph_index=graph_index, epoch=epoch, score=score)
            candidates.append((graph_index, score, record))

        candidates.sort(key=lambda item: item[1], reverse=True)
        selected_indices: list[int] = []
        selected_records: list[dict[str, Any]] = []
        seen_smiles: set[str] = set()
        for graph_index, _score, record in candidates:
            smiles_key = _smiles_key(record)
            if self.deduplicate_smiles and smiles_key and smiles_key in seen_smiles:
                continue
            if smiles_key:
                seen_smiles.add(smiles_key)
            selected_indices.append(graph_index)
            selected_records.append(record)
            if len(selected_indices) >= self.top_k_per_rank_per_epoch:
                break
        scores = [float(record["score"]) for record in selected_records]
        return GoodPoolSelection(
            indices=selected_indices,
            records=selected_records,
            metrics={
                "good_pool_enabled": float(self.enabled),
                "good_pool_selected_count": float(len(selected_indices)),
                "good_pool_selected_score_mean": float(sum(scores) / len(scores)) if scores else 0.0,
                "good_pool_selected_score_max": float(max(scores)) if scores else 0.0,
            },
        )

    def retained_records(self, epoch: int) -> list[dict[str, Any]]:
        """Return rolling retained good-molecule records."""
        min_epoch = int(epoch) - self.retention_window_epochs + 1
        window_records = [
            record
            for record in self._load_manifest_records()
            if min_epoch <= int(record.get("epoch", -1)) <= int(epoch)
        ]
        window_records.sort(key=lambda record: float(record.get("score", 0.0)), reverse=True)
        if self.deduplicate_retained_smiles:
            window_records = _deduplicate_records(window_records)
        if self.retained_top_k > 0:
            window_records = window_records[: self.retained_top_k]
        return window_records

    def write_top_sdf(self, *, generated_sdf: Path, output_path: Path, records: list[dict[str, Any]]) -> Path | None:
        """Write selected top molecules to SDF when the generated SDF is available."""
        if not records or not generated_sdf.exists():
            return None
        selected = {int(record["local_mol_index"]): record for record in records}
        supplier = Chem.SDMolSupplier(str(generated_sdf), sanitize=False, removeHs=False)
        mols: list[Chem.Mol] = []
        for mol in supplier:
            if mol is None:
                continue
            graph_index = _mol_index_from_mol(mol)
            if graph_index not in selected:
                continue
            for key, value in selected[graph_index].items():
                if isinstance(value, (str, int, float)) and value != "":
                    mol.SetProp(str(key), str(value))
            mols.append(mol)
        if not mols:
            return None
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = Chem.SDWriter(str(output_path))
        writer.SetKekulize(False)
        try:
            for mol in mols:
                writer.write(mol)
        finally:
            writer.close()
        return output_path

    def _build_record(self, *, row: dict[str, Any], graph_index: int, epoch: int, score: float) -> dict[str, Any]:
        """Build one manifest record."""
        record: dict[str, Any] = {
            "epoch": int(epoch),
            "rank": int(self.rank),
            "local_mol_index": int(graph_index),
            "global_mol_id": row.get("global_mol_id", ""),
            "score_key": self.score_key,
            "score": float(score),
            "smiles": row.get("smiles", ""),
            "canonical_smiles": row.get("canonical_smiles", ""),
            "inchikey": row.get("inchikey", ""),
            "murcko_scaffold": row.get("murcko_scaffold", ""),
        }
        for field_name in self.record_fields:
            if field_name in row:
                record[field_name] = _json_safe(row.get(field_name))
        return record

    def _passes_quality_filters(self, row: dict[str, Any]) -> bool:
        """Return whether a molecule row passes numeric quality filters."""
        for field_name, limits in self.quality_filters.items():
            value = _score_value(row, str(field_name))
            if value is None:
                if self.missing_quality_filter_policy == "reject":
                    return False
                continue
            if "min" in limits and value < float(limits["min"]):
                return False
            if "max" in limits and value > float(limits["max"]):
                return False
            if "equals" in limits and value != float(limits["equals"]):
                return False
        return True

    def _load_manifest_records(self) -> list[dict[str, Any]]:
        """Load append-only manifest records."""
        if not self.manifest_path.exists():
            return []
        records: list[dict[str, Any]] = []
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    @staticmethod
    def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
        """Rewrite JSONL records."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    @staticmethod
    def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
        """Write retained records as CSV."""
        fields: list[str] = []
        for record in records:
            for key in record:
                if key not in fields:
                    fields.append(str(key))
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for record in records:
                writer.writerow({key: _json_safe(record.get(key, "")) for key in fields})

    @staticmethod
    def _retained_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
        """Return retained-pool metrics."""
        scores = [float(record["score"]) for record in records if _score_value(record, "score") is not None]
        return {
            "good_pool_retained_count": float(len(records)),
            "good_pool_retained_score_mean": float(sum(scores) / len(scores)) if scores else 0.0,
            "good_pool_retained_score_max": float(max(scores)) if scores else 0.0,
        }


def _deduplicate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the highest-score record per canonical SMILES."""
    retained: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        key = _smiles_key(record)
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        retained.append(record)
    return retained


def _smiles_key(record: dict[str, Any]) -> str:
    """Return a deduplication key for a molecule record."""
    return str(record.get("canonical_smiles") or record.get("smiles") or "")


def _row_index(row: dict[str, Any], *, fallback: int) -> int:
    """Read local graph index from a molecule row."""
    for key in ("local_mol_index", "mol_index", "graph_index", "local_index"):
        try:
            return int(float(row.get(key, "")))
        except (TypeError, ValueError):
            continue
    return int(fallback)


def _row_int(row: dict[str, Any], key: str, default: int = 0) -> int:
    """Read an integer-like molecule field."""
    try:
        return int(float(row.get(key, default) or default))
    except (TypeError, ValueError):
        return int(default)


def _score_value(row: dict[str, Any], key: str) -> float | None:
    """Read a finite float-like field."""
    try:
        value = float(row.get(key, ""))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _json_safe(value: Any) -> Any:
    """Convert common values into JSON/CSV friendly scalars."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _mol_index_from_mol(mol: Chem.Mol) -> int:
    """Read local molecule index from SDF properties."""
    for prop_name in ("mol_index", "local_mol_index"):
        if mol.HasProp(prop_name):
            try:
                return int(float(mol.GetProp(prop_name)))
            except ValueError:
                continue
    name = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
    suffix = name.rsplit("_", 1)[-1]
    try:
        return int(suffix)
    except ValueError:
        return -1
