"""Compute PB-valid rates for DiffGui 200-step training endpoints."""

from __future__ import annotations

import argparse
import math
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem

import posebusters.posebusters as _pbmod
from posebusters import PoseBusters


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
RUN_DIR = ROOT / "runs/diffgui-deppa-full-lr1e5-wd1e4-b8x8-t200-100ep_20260626_165809"
RECEPTOR = ROOT / "analysis/benchmark/8gjl_c1101_benchmark_20260603_500mol_500steps/inputs/8GJL_C1101_pocket8A.pdb"
PER_EPOCH_DIR = OUT_DIR / "posebusters_per_epoch"
SUMMARY_CSV = OUT_DIR / "diffgui_t200_epoch_endpoint_pbvalid_summary.csv"
SUMMARY_MD = OUT_DIR / "diffgui_t200_epoch_endpoint_pbvalid_summary.md"


def _get_name(self: object, mol_pred: Chem.Mol) -> str:
    try:
        return str(mol_pred.GetProp("_Name"))
    except Exception:
        return "mol"


_pbmod.PoseBusters._get_name = _get_name


def _sdf_count(path: Path) -> int:
    count = 0
    with path.open("r", errors="replace") as handle:
        for line in handle:
            if line.strip() == "$$$$":
                count += 1
    return count


def _pb_valid_row(row: pd.Series) -> bool:
    bool_cols = [col for col, value in row.items() if isinstance(value, (bool, np.bool_))]
    return bool(bool_cols) and all(bool(row[col]) for col in bool_cols)


def _bust_one_molecule(buster: PoseBusters, receptor: Path, mol: Chem.Mol) -> dict[str, Any]:
    tmp = tempfile.NamedTemporaryFile(suffix=".sdf", delete=False)
    tmp_path = Path(tmp.name)
    try:
        writer = Chem.SDWriter(str(tmp_path))
        writer.write(mol)
        writer.close()
        df = buster.bust(mol_pred=str(tmp_path), mol_cond=str(receptor))
    finally:
        tmp.close()
        tmp_path.unlink(missing_ok=True)
    if df.empty:
        return {"pb_valid": False, "posebusters_empty": True}
    row = df.iloc[0]
    result = row.to_dict()
    result["pb_valid"] = _pb_valid_row(row)
    result["posebusters_empty"] = False
    return result


def _epoch_sdf(epoch: int) -> Path:
    path = RUN_DIR / "epochs" / f"epoch_{epoch:04d}" / "molecules.sdf"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _csv_path(epoch: int) -> Path:
    return PER_EPOCH_DIR / f"diffgui_t200_epoch_{epoch:04d}_posebusters.csv"


def _load_existing_summary(epoch: int, csv_path: Path) -> dict[str, Any]:
    df = pd.read_csv(csv_path)
    valid_count = int(df["pb_valid"].fillna(False).astype(bool).sum()) if not df.empty else 0
    return {
        "run_id": "diffgui_t200",
        "run_label": "DiffGui 200 steps",
        "timesteps_config": 200,
        "epoch": epoch,
        "endpoint": "epoch1" if epoch == 1 else "last_complete",
        "source_sdf": str(_epoch_sdf(epoch)),
        "sdf_records": _sdf_count(_epoch_sdf(epoch)),
        "processed": len(df),
        "pb_valid_count": valid_count,
        "pb_valid_rate": valid_count / len(df) if len(df) else math.nan,
        "elapsed_seconds": "",
        "per_molecule_csv": str(csv_path),
        "status": "existing",
    }


def _compute_epoch(epoch: int, *, force: bool) -> dict[str, Any]:
    PER_EPOCH_DIR.mkdir(parents=True, exist_ok=True)
    sdf_path = _epoch_sdf(epoch)
    csv_path = _csv_path(epoch)
    partial_path = csv_path.with_suffix(".partial.csv")
    if csv_path.exists() and not force:
        return _load_existing_summary(epoch, csv_path)

    buster = PoseBusters(config="dock")
    rows: list[dict[str, Any]] = []
    resume_after = -1
    if partial_path.exists() and not force:
        partial_df = pd.read_csv(partial_path)
        if not partial_df.empty and "source_index" in partial_df.columns:
            rows = partial_df.to_dict("records")
            resume_after = int(pd.to_numeric(partial_df["source_index"]).max())
            print(f"[resume] epoch={epoch} loaded {len(rows)} rows from {partial_path}; resume_after={resume_after}", flush=True)
    started = time.time()
    print(f"[posebusters] epoch={epoch} sdf={sdf_path} records={_sdf_count(sdf_path)}", flush=True)
    for source_index, mol in enumerate(Chem.SDMolSupplier(str(sdf_path), removeHs=False)):
        if source_index <= resume_after:
            continue
        if mol is None:
            continue
        row: dict[str, Any] = {
            "run_id": "diffgui_t200",
            "run_label": "DiffGui 200 steps",
            "epoch": epoch,
            "endpoint": "epoch1" if epoch == 1 else "last_complete",
            "source_index": source_index,
            "source_sdf": str(sdf_path),
            "name": mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{source_index:06d}",
        }
        try:
            row.update(_bust_one_molecule(buster, RECEPTOR, mol))
        except Exception as exc:
            row.update({"pb_valid": False, "posebusters_error": repr(exc)})
        rows.append(row)
        if len(rows) % 10 == 0:
            pd.DataFrame(rows).to_csv(partial_path, index=False)
            print(f"  processed {len(rows)}", flush=True)

    per_df = pd.DataFrame(rows)
    per_df.to_csv(csv_path, index=False)
    partial_path.unlink(missing_ok=True)
    valid_count = int(per_df["pb_valid"].fillna(False).astype(bool).sum()) if not per_df.empty else 0
    elapsed = time.time() - started
    return {
        "run_id": "diffgui_t200",
        "run_label": "DiffGui 200 steps",
        "timesteps_config": 200,
        "epoch": epoch,
        "endpoint": "epoch1" if epoch == 1 else "last_complete",
        "source_sdf": str(sdf_path),
        "sdf_records": _sdf_count(sdf_path),
        "processed": len(per_df),
        "pb_valid_count": valid_count,
        "pb_valid_rate": valid_count / len(per_df) if len(per_df) else math.nan,
        "elapsed_seconds": elapsed,
        "per_molecule_csv": str(csv_path),
        "status": "computed",
    }


def _write_markdown(summary: pd.DataFrame) -> None:
    table = summary.copy()
    table["PB-valid率（%）"] = table["pb_valid_rate"].map(lambda value: f"{float(value) * 100:.2f}")
    keep = ["epoch", "endpoint", "processed", "pb_valid_count", "PB-valid率（%）", "status"]
    lines = [
        "# DiffGui 200-step endpoint PB-valid",
        "",
        "| " + " | ".join(keep) + " |",
        "| " + " | ".join(["---"] * len(keep)) + " |",
    ]
    for _, row in table[keep].iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in keep) + " |")
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", nargs="*", type=int, default=[1, 89])
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not RECEPTOR.exists():
        raise FileNotFoundError(RECEPTOR)
    rows = [_compute_epoch(epoch, force=bool(args.force)) for epoch in args.epochs]
    summary = pd.DataFrame(rows).sort_values("epoch")
    summary.to_csv(SUMMARY_CSV, index=False)
    _write_markdown(summary)
    print(summary[["epoch", "processed", "pb_valid_count", "pb_valid_rate", "status"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
