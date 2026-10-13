#!/usr/bin/env python3
"""Run one matched PoseBusters dock-protocol comparison."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

import posebusters.posebusters as pb_module
from posebusters import PoseBusters

RDLogger.DisableLog("rdApp.*")


def _get_name(self: object, mol: Chem.Mol) -> str:
    try:
        return str(mol.GetProp("_Name"))
    except Exception:
        return "mol"


pb_module.PoseBusters._get_name = _get_name


def _pb_valid(row: pd.Series) -> bool:
    flags = [bool(value) for value in row.values if isinstance(value, (bool, np.bool_))]
    return bool(flags) and all(flags)


def _read_docking_scores(path: Path) -> dict[int, float]:
    table = pd.read_csv(path)
    return {
        int(row["benchmark_index"]): float(row["score"])
        for _, row in table.iterrows()
        if str(row.get("success")) == "True" and math.isfinite(float(row["score"]))
    }


def evaluate(
    method: str,
    sdf_path: Path,
    docking_csv: Path,
    receptor: Path,
    checkpoint_path: Path,
    checkpoint_every: int,
) -> pd.DataFrame:
    buster = PoseBusters(config="dock")
    scores = _read_docking_scores(docking_csv)
    rows: list[dict[str, Any]] = []
    supplier = Chem.SDMolSupplier(str(sdf_path), sanitize=False, removeHs=False)
    for source_index, mol in enumerate(supplier):
        if mol is None:
            continue
        try:
            result = buster.bust(mol_pred=mol, mol_cond=str(receptor))
            if result.empty:
                row: dict[str, Any] = {"posebusters_empty": True, "pb_valid": False}
            else:
                pb_row = result.iloc[0]
                row = pb_row.to_dict()
                row["posebusters_empty"] = False
                row["pb_valid"] = _pb_valid(pb_row)
        except Exception as exc:
            row = {"posebusters_error": repr(exc), "pb_valid": False}
        row.update(
            {
                "method": method,
                "source_index": source_index,
                "benchmark_index": int(mol.GetProp("benchmark_index")) if mol.HasProp("benchmark_index") else source_index,
                "name": str(mol.GetProp("_Name")) if mol.HasProp("_Name") else f"mol_{source_index:06d}",
            }
        )
        row["docking_score"] = scores.get(int(row["benchmark_index"]), math.nan)
        rows.append(row)
        if len(rows) % checkpoint_every == 0:
            pd.DataFrame(rows).to_csv(checkpoint_path, index=False)
            print(f"{method}: processed={len(rows)} pb_valid={sum(bool(item['pb_valid']) for item in rows)}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(checkpoint_path, index=False)
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-sdf", type=Path, required=True)
    parser.add_argument("--original-docking-csv", type=Path, required=True)
    parser.add_argument("--model-sdf", type=Path, required=True)
    parser.add_argument("--model-docking-csv", type=Path, required=True)
    parser.add_argument("--receptor", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--attempts", type=int, default=1000)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    tables = []
    for method, sdf, docking in (
        ("Original", args.original_sdf, args.original_docking_csv),
        ("Matched structure x-only", args.model_sdf, args.model_docking_csv),
    ):
        output = args.output_dir / f"{method.lower().replace(' ', '_').replace('-', '_')}_posebusters.csv"
        tables.append(evaluate(method, sdf, docking, args.receptor, output, args.checkpoint_every))
    combined = pd.concat(tables, ignore_index=True, sort=False)
    combined.to_csv(args.output_dir / "posebusters_per_molecule.csv", index=False)
    boolean_columns = [
        column
        for column in combined.columns
        if column != "pb_valid" and combined[column].dropna().map(lambda value: isinstance(value, (bool, np.bool_))).all()
    ]
    summary_rows: list[dict[str, Any]] = []
    check_rows: list[dict[str, Any]] = []
    for method, table in combined.groupby("method", sort=False):
        pb_valid_count = int(table["pb_valid"].fillna(False).astype(bool).sum())
        summary_rows.append(
            {
                "method": method,
                "processed": len(table),
                "pb_valid_count": pb_valid_count,
                "pb_valid_rate_processed": pb_valid_count / len(table) if len(table) else math.nan,
                "pb_valid_rate_fixed_attempts": pb_valid_count / args.attempts,
                "posebusters_error_count": int(table.get("posebusters_error", pd.Series(index=table.index)).notna().sum()),
            }
        )
        for column in boolean_columns:
            values = table[column].dropna().astype(bool)
            check_rows.append(
                {
                    "method": method,
                    "check": column,
                    "evaluated": len(values),
                    "pass_count": int(values.sum()),
                    "pass_rate": float(values.mean()) if len(values) else math.nan,
                }
            )
    pd.DataFrame(summary_rows).to_csv(args.output_dir / "posebusters_summary.csv", index=False)
    pd.DataFrame(check_rows).to_csv(args.output_dir / "posebusters_check_summary.csv", index=False)
    manifest = {
        "status": "completed",
        "protocol": "PoseBusters(config='dock')",
        "receptor": str(args.receptor.resolve()),
        "fixed_attempts_per_method": args.attempts,
        "elapsed_seconds": time.time() - started,
        "summary": summary_rows,
    }
    (args.output_dir / "posebusters_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
