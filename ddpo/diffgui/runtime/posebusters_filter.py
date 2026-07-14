"""PoseBusters gate for DiffGui-generated SDF molecules."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem

import posebusters.posebusters as _pbmod
from posebusters import PoseBusters


logger = logging.getLogger(__name__)


def _get_name(self: object, mol_pred: Chem.Mol) -> str:
    """Return a robust molecule name across supported RDKit versions."""
    try:
        return str(mol_pred.GetProp("_Name"))
    except Exception:
        return "mol"


_pbmod.PoseBusters._get_name = _get_name


def run_posebusters_filter(
    *,
    input_sdf: Path,
    receptor_pdb: Path,
    output_sdf: Path,
    output_csv: Path,
    summary_json: Path,
    total_attempts: int,
    config_name: str = "dock",
    buster: Any | None = None,
) -> dict[str, Any]:
    """Write PB-valid molecules and return denominator-explicit metrics."""
    if not input_sdf.exists():
        raise FileNotFoundError(f"PoseBusters input SDF does not exist: {input_sdf}")
    if not receptor_pdb.exists():
        raise FileNotFoundError(f"PoseBusters receptor does not exist: {receptor_pdb}")
    if total_attempts < 0:
        raise ValueError("total_attempts must be non-negative")

    output_sdf.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    evaluator = buster or PoseBusters(config=config_name)
    rows: list[dict[str, Any]] = []
    writer = Chem.SDWriter(str(output_sdf))
    writer.SetKekulize(False)
    try:
        supplier = Chem.SDMolSupplier(str(input_sdf), sanitize=False, removeHs=False)
        for source_index, mol in enumerate(supplier):
            row: dict[str, Any] = {
                "source_index": int(source_index),
                "source_graph_index": int(source_index),
                "original_name": "",
                "sanitize_valid": 0,
                "pb_evaluated": 0,
                "pb_valid": 0,
                "posebusters_empty": 0,
                "posebusters_error": "",
            }
            if mol is None:
                row["posebusters_error"] = "supplier_none"
                rows.append(row)
                continue
            row["source_graph_index"] = _source_graph_index(mol, source_index)
            row["original_name"] = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
            sanitized, sanitize_status = _sanitize_molecule(mol)
            row["sanitize_status"] = sanitize_status
            row["sanitize_valid"] = int(sanitized is not None)
            if sanitized is None:
                rows.append(row)
                continue
            try:
                pb_fields = _bust_one_molecule(evaluator, receptor_pdb, sanitized)
                row.update(pb_fields)
            except Exception as exc:
                row["pb_evaluated"] = 1
                row["posebusters_error"] = f"{type(exc).__name__}:{exc}"
                logger.exception(
                    "PoseBusters failed source_index=%d graph_index=%d",
                    source_index,
                    row["source_graph_index"],
                )
            if int(row.get("pb_valid", 0)) == 1:
                valid_mol = Chem.Mol(sanitized)
                valid_mol.SetProp("source_graph_index", str(row["source_graph_index"]))
                valid_mol.SetProp("pb_valid", "1")
                writer.write(valid_mol)
            rows.append(row)
    finally:
        writer.close()

    _write_rows(output_csv, rows)
    sanitize_count = sum(int(row.get("sanitize_valid", 0)) for row in rows)
    pb_evaluated = sum(int(row.get("pb_evaluated", 0)) for row in rows)
    pb_valid = sum(int(row.get("pb_valid", 0)) for row in rows)
    attempts = int(total_attempts)
    summary = {
        "posebusters_enabled": 1,
        "posebusters_config": config_name,
        "posebusters_total_attempts": attempts,
        "posebusters_input_records": len(rows),
        "posebusters_sanitize_valid": sanitize_count,
        "posebusters_pb_evaluated": pb_evaluated,
        "posebusters_pb_valid": pb_valid,
        "posebusters_pb_invalid_sanitize_valid": max(0, sanitize_count - pb_valid),
        "posebusters_invalid_total": max(0, attempts - pb_valid),
        "posebusters_sanitize_rate_attempts": _rate(sanitize_count, attempts),
        "posebusters_pb_valid_rate_sanitize": _rate(pb_valid, sanitize_count),
        "posebusters_pb_valid_rate_attempts": _rate(pb_valid, attempts),
        "posebusters_input_sdf": str(input_sdf),
        "posebusters_receptor_pdb": str(receptor_pdb),
        "posebusters_valid_sdf": str(output_sdf),
        "posebusters_csv": str(output_csv),
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    logger.info(
        "PoseBusters gate attempts=%d sanitize=%d pb_valid=%d conditional=%.4f total=%.4f",
        attempts,
        sanitize_count,
        pb_valid,
        summary["posebusters_pb_valid_rate_sanitize"],
        summary["posebusters_pb_valid_rate_attempts"],
    )
    return summary


def _sanitize_molecule(mol: Chem.Mol) -> tuple[Chem.Mol | None, str]:
    """Return a sanitized molecule without imposing extra connectivity rules."""
    candidate = Chem.Mol(mol)
    try:
        candidate.UpdatePropertyCache(strict=False)
        status = Chem.SanitizeMol(candidate, catchErrors=True)
    except Exception as exc:
        return None, f"sanitize_exception:{type(exc).__name__}"
    if status != Chem.SanitizeFlags.SANITIZE_NONE:
        return None, f"sanitize_failed:{int(status)}"
    return candidate, "ok"


def _bust_one_molecule(buster: Any, receptor_pdb: Path, mol: Chem.Mol) -> dict[str, Any]:
    """Run the established dock checks for one molecule."""
    with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp:
        writer = Chem.SDWriter(tmp.name)
        writer.SetKekulize(False)
        writer.write(mol)
        writer.close()
        result_df = buster.bust(mol_pred=tmp.name, mol_cond=str(receptor_pdb))
    if result_df.empty:
        return {"pb_evaluated": 1, "pb_valid": 0, "posebusters_empty": 1}
    result_row = result_df.iloc[0]
    boolean_values = [value for value in result_row.values if isinstance(value, (bool, np.bool_))]
    fields: dict[str, Any] = {
        "pb_evaluated": 1,
        "pb_valid": int(bool(boolean_values) and all(bool(value) for value in boolean_values)),
        "posebusters_empty": 0,
    }
    for key, value in result_row.items():
        if isinstance(value, (bool, np.bool_)):
            fields[f"pb_check_{_metric_safe(key)}"] = int(bool(value))
    return fields


def _source_graph_index(mol: Chem.Mol, fallback: int) -> int:
    """Read the original policy graph index carried by the generated SDF."""
    for key in ("source_graph_index", "mol_index"):
        if not mol.HasProp(key):
            continue
        try:
            return int(float(mol.GetProp(key)))
        except (TypeError, ValueError):
            continue
    return int(fallback)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a union-schema per-molecule CSV."""
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _rate(numerator: int, denominator: int) -> float:
    """Return a finite rate with an explicit denominator."""
    return float(numerator / denominator) if denominator > 0 else 0.0


def _metric_safe(value: Any) -> str:
    """Convert PoseBusters column names into stable field suffixes."""
    text = str(value)
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-sdf", type=Path, required=True)
    parser.add_argument("--receptor-pdb", type=Path, required=True)
    parser.add_argument("--output-sdf", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--total-attempts", type=int, required=True)
    parser.add_argument("--config-name", default="dock")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return parser.parse_args()


def main() -> None:
    """Run the command-line PoseBusters gate."""
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger.info("cpu_affinity=%s", sorted(int(cpu) for cpu in os.sched_getaffinity(0)))
    run_posebusters_filter(
        input_sdf=args.input_sdf,
        receptor_pdb=args.receptor_pdb,
        output_sdf=args.output_sdf,
        output_csv=args.output_csv,
        summary_json=args.summary_json,
        total_attempts=args.total_attempts,
        config_name=args.config_name,
    )


if __name__ == "__main__":
    main()
