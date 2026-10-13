"""Queued PoseBusters statistics for the formal five-model benchmark.

The queue processes one repeat/model SDF at a time and waits for system load to
drop before starting each task. This avoids the all-CPU contention caused by
large parallel PoseBusters runs.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem

import posebusters.posebusters as _pbmod
from posebusters import PoseBusters


BASE = Path(__file__).resolve().parent
RECEPTOR = Path(
    "/home/md666/data/disk6/DiffB110/analysis/benchmark/8gjl_c1101_benchmark_20260603_500mol_500steps/inputs/8GJL_C1101_pocket8A.pdb"
)
MODEL_ORDER = ["DiffGui", "DiffSBDD", "PMDM", "GCDM-SBDD", "TargetDiff"]
MODEL_SLUG = {
    "DiffGui": "diffgui",
    "DiffSBDD": "diffsbdd",
    "PMDM": "pmdm",
    "GCDM-SBDD": "gcdm_sbdd",
    "TargetDiff": "targetdiff",
}
REPEAT_ORDER = ["rep1", "rep2", "rep3"]
DOCKING_PROP = "DockingScore"


def _get_name(self: object, mol_pred: Chem.Mol) -> str:
    """Return a robust molecule name for current RDKit/PoseBusters versions."""
    try:
        return str(mol_pred.GetProp("_Name"))
    except Exception:
        return "mol"


_pbmod.PoseBusters._get_name = _get_name

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _sdf_count(sdf_path: Path) -> int:
    """Count SDF records using the standard delimiter."""
    count = 0
    with sdf_path.open("r", errors="replace") as handle:
        for line in handle:
            if line.strip() == "$$$$":
                count += 1
    return count


def _wait_for_load(max_load: float, check_interval: int) -> None:
    """Wait until 1-minute load average is below the configured threshold."""
    if max_load <= 0:
        return
    while True:
        load1, load5, load15 = os.getloadavg()
        if load1 <= max_load:
            logger.info("load ok: %.2f %.2f %.2f <= max_load %.2f", load1, load5, load15, max_load)
            return
        logger.info(
            "load high: %.2f %.2f %.2f > max_load %.2f; sleeping %ss",
            load1,
            load5,
            load15,
            max_load,
            check_interval,
        )
        time.sleep(check_interval)


def _pb_valid_row(df_row: pd.Series) -> bool:
    """Return True if all boolean PoseBusters checks pass for one molecule."""
    bool_cols = [col for col, value in df_row.items() if isinstance(value, (bool, np.bool_))]
    if not bool_cols:
        return False
    return all(bool(df_row[col]) for col in bool_cols)


def _mol_name(mol: Chem.Mol, index: int) -> str:
    """Return a stable molecule name."""
    if mol.HasProp("_Name"):
        return str(mol.GetProp("_Name"))
    return f"mol_{index:06d}"


def _docking_score(mol: Chem.Mol) -> float:
    """Read docking score from SDF properties when available."""
    try:
        return float(mol.GetProp(DOCKING_PROP))
    except Exception:
        return float("nan")


def _bust_one_molecule(buster: PoseBusters, receptor: Path, mol: Chem.Mol) -> dict[str, Any]:
    """Run PoseBusters for one molecule through a temporary SDF."""
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
    result: dict[str, Any] = row.to_dict()
    result["pb_valid"] = bool(_pb_valid_row(row))
    result["posebusters_empty"] = False
    return result


def _write_checkpoint(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a partial CSV checkpoint."""
    if not rows:
        return
    pd.DataFrame(rows).to_csv(path, index=False)


def process_task(
    *,
    rep: str,
    model: str,
    receptor: Path,
    out_dir: Path,
    limit: int | None,
    force: bool,
    checkpoint_every: int,
) -> Path:
    """Run one repeat/model task and return its per-molecule CSV path."""
    slug = MODEL_SLUG.get(model, model.lower())
    sdf_path = BASE / "docking" / rep / slug / "successful_docked_outputs.sdf"
    if not sdf_path.exists():
        raise FileNotFoundError(f"missing docked SDF: {sdf_path}")

    per_task = out_dir / "per_task"
    status_dir = out_dir / "status"
    per_task.mkdir(parents=True, exist_ok=True)
    status_dir.mkdir(parents=True, exist_ok=True)
    csv_path = per_task / f"{rep}_{slug}_posebusters.csv"
    partial_path = per_task / f"{rep}_{slug}_posebusters.partial.csv"
    done_path = status_dir / f"{rep}_{slug}.done.json"
    if done_path.exists() and csv_path.exists() and not force:
        logger.info("skip existing task %s/%s: %s", rep, model, csv_path)
        return csv_path

    total_sdf = _sdf_count(sdf_path)
    logger.info("processing %s/%s: sdf_records=%s limit=%s", rep, model, total_sdf, limit)
    buster = PoseBusters(config="dock")
    rows: list[dict[str, Any]] = []
    started = time.time()
    processed = 0
    for source_index, mol in enumerate(Chem.SDMolSupplier(str(sdf_path), removeHs=False)):
        if mol is None:
            continue
        if limit is not None and processed >= limit:
            break
        processed += 1
        row: dict[str, Any] = {
            "repeat": rep,
            "model": model,
            "model_slug": slug,
            "source_sdf": str(sdf_path),
            "source_index": source_index,
            "name": _mol_name(mol, source_index),
            "docking_score": _docking_score(mol),
        }
        try:
            row.update(_bust_one_molecule(buster, receptor, mol))
        except Exception as exc:  # keep the queue moving and record failures.
            row.update({"pb_valid": False, "posebusters_error": repr(exc)})
            logger.exception("PoseBusters failed for %s/%s source_index=%s", rep, model, source_index)
        rows.append(row)
        if processed % checkpoint_every == 0:
            _write_checkpoint(partial_path, rows)
            logger.info("%s/%s processed %s molecules", rep, model, processed)

    _write_checkpoint(csv_path, rows)
    partial_path.unlink(missing_ok=True)
    elapsed = time.time() - started
    valid_count = int(sum(bool(row.get("pb_valid", False)) for row in rows))
    done = {
        "repeat": rep,
        "model": model,
        "model_slug": slug,
        "source_sdf": str(sdf_path),
        "sdf_records": total_sdf,
        "processed": len(rows),
        "pb_valid_count": valid_count,
        "pb_valid_rate": valid_count / len(rows) if rows else math.nan,
        "elapsed_seconds": elapsed,
        "csv": str(csv_path),
    }
    done_path.write_text(json.dumps(done, ensure_ascii=True, indent=2) + "\n")
    logger.info("done %s/%s: processed=%s pb_valid=%s elapsed=%.1fs", rep, model, len(rows), valid_count, elapsed)
    return csv_path


def write_summaries(out_dir: Path) -> None:
    """Combine per-task CSV files into summary tables."""
    per_task = out_dir / "per_task"
    rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    for done_path in sorted((out_dir / "status").glob("*.done.json")):
        status_rows.append(json.loads(done_path.read_text()))
    for csv_path in sorted(per_task.glob("*_posebusters.csv")):
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        total = len(df)
        valid_count = int(df["pb_valid"].fillna(False).astype(bool).sum())
        valid_df = df[df["pb_valid"].fillna(False).astype(bool)]
        best_score = float("nan")
        best_name = ""
        if not valid_df.empty and "docking_score" in valid_df.columns:
            scores = pd.to_numeric(valid_df["docking_score"], errors="coerce")
            if scores.notna().any():
                best_idx = scores.idxmin()
                best_score = float(scores.loc[best_idx])
                best_name = str(valid_df.loc[best_idx, "name"])
        first = df.iloc[0]
        rows.append(
            {
                "repeat": first["repeat"],
                "model": first["model"],
                "model_slug": first["model_slug"],
                "source_sdf": first["source_sdf"],
                "total": total,
                "pb_valid_count": valid_count,
                "pb_valid_rate": valid_count / total if total else math.nan,
                "best_pb_valid_docking_score": best_score,
                "best_pb_valid_name": best_name,
                "per_molecule_csv": str(csv_path),
            }
        )
    if rows:
        summary = pd.DataFrame(rows)
        summary.to_csv(out_dir / "posebusters_summary.csv", index=False)
        model_summary = (
            summary.groupby("model", as_index=False)
            .agg(total=("total", "sum"), pb_valid_count=("pb_valid_count", "sum"))
        )
        model_summary["pb_valid_rate"] = model_summary["pb_valid_count"] / model_summary["total"]
        model_summary.to_csv(out_dir / "posebusters_model_summary.csv", index=False)
    if status_rows:
        pd.DataFrame(status_rows).to_csv(out_dir / "posebusters_task_status.csv", index=False)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receptor", default=str(RECEPTOR), help="Receptor PDB for PoseBusters dock config.")
    parser.add_argument("--out-dir", default=str(BASE / "posebusters_queue"), help="Output directory.")
    parser.add_argument("--models", nargs="*", default=MODEL_ORDER, help="Models to process.")
    parser.add_argument("--repeats", nargs="*", default=REPEAT_ORDER, help="Repeats to process.")
    parser.add_argument("--limit", type=int, default=None, help="Max molecules per task for smoke runs.")
    parser.add_argument("--max-load", type=float, default=20.0, help="Wait while 1-min load average exceeds this; <=0 disables waiting.")
    parser.add_argument("--check-interval", type=int, default=300, help="Seconds between load checks.")
    parser.add_argument("--sleep-between", type=int, default=60, help="Seconds to sleep between tasks.")
    parser.add_argument("--checkpoint-every", type=int, default=25, help="Partial CSV checkpoint interval.")
    parser.add_argument("--force", action="store_true", help="Recompute tasks with done markers.")
    parser.add_argument("--dry-run", action="store_true", help="List planned tasks and exit.")
    return parser.parse_args()


def main() -> None:
    """Run the queue."""
    args = parse_args()
    receptor = Path(args.receptor)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(rep, model) for rep in args.repeats for model in args.models]
    logger.info("planned tasks=%s out_dir=%s", len(tasks), out_dir)
    for rep, model in tasks:
        slug = MODEL_SLUG.get(model, model.lower())
        sdf_path = BASE / "docking" / rep / slug / "successful_docked_outputs.sdf"
        logger.info("task %s/%s sdf=%s records=%s", rep, model, sdf_path, _sdf_count(sdf_path) if sdf_path.exists() else "MISSING")
    if args.dry_run:
        return

    for task_index, (rep, model) in enumerate(tasks, start=1):
        logger.info("queue task %s/%s: %s/%s", task_index, len(tasks), rep, model)
        _wait_for_load(float(args.max_load), int(args.check_interval))
        process_task(
            rep=rep,
            model=model,
            receptor=receptor,
            out_dir=out_dir,
            limit=args.limit,
            force=bool(args.force),
            checkpoint_every=int(args.checkpoint_every),
        )
        write_summaries(out_dir)
        if task_index < len(tasks) and int(args.sleep_between) > 0:
            logger.info("sleeping %ss before next task", int(args.sleep_between))
            time.sleep(int(args.sleep_between))
    write_summaries(out_dir)
    logger.info("queue completed")


if __name__ == "__main__":
    main()
