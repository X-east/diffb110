"""PoseBusters PB-valid for DiffGui step-sweep 100/200/500 docked SDFs.

Mirrors the formal five-model benchmark PoseBusters queue
(analysis/benchmark/benchmark5/.../posebusters_queue.py) but targets the nine
successful_docked_outputs.sdf files produced by run_diffgui_step_docking_queue.py
under docking_100_200_500/.

The queue runs one task per subprocess worker and waits for system load to drop
before starting each task, avoiding the all-CPU contention that large parallel
PoseBusters runs cause.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem

import posebusters.posebusters as _pbmod
from posebusters import PoseBusters


SCRIPT_PATH = Path(__file__).resolve()
STATS_DIR = SCRIPT_PATH.parent
REPO_ROOT = SCRIPT_PATH.parents[3]
DOCKED_ROOT = STATS_DIR / "docking_100_200_500"
RECEPTOR = (
    REPO_ROOT
    / "analysis/benchmark/8gjl_c1101_benchmark_20260603_500mol_500steps/inputs/8GJL_C1101_pocket8A.pdb"
)
OUT_DIR = STATS_DIR / "posebusters_100_200_500"
DOCKING_PROP = "DockingScore"

STEPS = (100, 200, 500)


@dataclass(frozen=True)
class PBTask:
    """One step-sweep docked SDF to score with PoseBusters."""

    step: int
    repeat: int
    name: str

    @property
    def sdf_path(self) -> Path:
        return DOCKED_ROOT / self.name / "diffgui" / "successful_docked_outputs.sdf"


def _task_specs() -> list[PBTask]:
    specs: list[PBTask] = []
    for step in STEPS:
        for repeat in (1, 2, 3):
            seed_label = {1: "seed1_7", 2: "seed8_14", 3: "seed15_21"}[repeat]
            name = f"{step}steps_rep{repeat}_{seed_label}"
            specs.append(PBTask(step=step, repeat=repeat, name=name))
    return specs


def _get_name(self: object, mol_pred: Chem.Mol) -> str:
    """Return a robust molecule name for current RDKit/PoseBusters versions."""
    try:
        return str(mol_pred.GetProp("_Name"))
    except Exception:
        return "mol"


_pbmod.PoseBusters._get_name = _get_name

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("diffgui_step_posebusters")


def _sdf_count(sdf_path: Path) -> int:
    count = 0
    with sdf_path.open("r", errors="replace") as handle:
        for line in handle:
            if line.strip() == "$$$$":
                count += 1
    return count


def _wait_for_load(max_load: float, check_interval: int) -> None:
    if max_load <= 0:
        return
    while True:
        load1, _load5, _load15 = os.getloadavg()
        if load1 <= max_load:
            return
        logger.info("load high: %.2f > max_load %.2f; sleeping %ss", load1, max_load, check_interval)
        time.sleep(check_interval)


def _pb_valid_row(df_row: pd.Series) -> bool:
    bool_cols = [col for col, value in df_row.items() if isinstance(value, (bool, np.bool_))]
    if not bool_cols:
        return False
    return all(bool(df_row[col]) for col in bool_cols)


def _mol_name(mol: Chem.Mol, index: int) -> str:
    if mol.HasProp("_Name"):
        return str(mol.GetProp("_Name"))
    return f"mol_{index:06d}"


def _docking_score(mol: Chem.Mol) -> float:
    try:
        return float(mol.GetProp(DOCKING_PROP))
    except Exception:
        return float("nan")


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
    result: dict[str, Any] = row.to_dict()
    result["pb_valid"] = bool(_pb_valid_row(row))
    result["posebusters_empty"] = False
    return result


def process_task(task: PBTask, *, receptor: Path, force: bool, checkpoint_every: int) -> Path:
    """Run PoseBusters for one task and return its per-molecule CSV path."""
    per_task = OUT_DIR / "per_task"
    status_dir = OUT_DIR / "status"
    per_task.mkdir(parents=True, exist_ok=True)
    status_dir.mkdir(parents=True, exist_ok=True)
    csv_path = per_task / f"{task.name}_posebusters.csv"
    partial_path = per_task / f"{task.name}_posebusters.partial.csv"
    done_path = status_dir / f"{task.name}.done.json"

    if done_path.exists() and csv_path.exists() and not force:
        logger.info("skip existing task %s: %s", task.name, csv_path)
        return csv_path

    sdf_path = task.sdf_path
    if not sdf_path.exists():
        raise FileNotFoundError(f"missing docked SDF: {sdf_path}")

    total_sdf = _sdf_count(sdf_path)
    logger.info("processing %s: sdf_records=%s", task.name, total_sdf)
    buster = PoseBusters(config="dock")
    rows: list[dict[str, Any]] = []
    started = time.time()
    processed = 0
    for source_index, mol in enumerate(Chem.SDMolSupplier(str(sdf_path), removeHs=False)):
        if mol is None:
            continue
        processed += 1
        row: dict[str, Any] = {
            "step": task.step,
            "repeat": task.repeat,
            "name": task.name,
            "source_index": source_index,
            "mol_name": _mol_name(mol, source_index),
            "docking_score": _docking_score(mol),
        }
        try:
            row.update(_bust_one_molecule(buster, receptor, mol))
        except Exception as exc:  # keep the queue moving and record failures.
            row.update({"pb_valid": False, "posebusters_error": repr(exc)})
            logger.exception("PoseBusters failed for %s source_index=%s", task.name, source_index)
        rows.append(row)
        if processed % checkpoint_every == 0:
            pd.DataFrame(rows).to_csv(partial_path, index=False)
            logger.info("%s processed %s molecules", task.name, processed)

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    partial_path.unlink(missing_ok=True)
    elapsed = time.time() - started
    valid_count = int(sum(bool(row.get("pb_valid", False)) for row in rows))
    done = {
        "step": task.step,
        "repeat": task.repeat,
        "name": task.name,
        "source_sdf": str(sdf_path),
        "sdf_records": total_sdf,
        "processed": len(rows),
        "pb_valid_count": valid_count,
        "pb_valid_rate": valid_count / len(rows) if rows else math.nan,
        "elapsed_seconds": elapsed,
        "csv": str(csv_path),
    }
    done_path.write_text(json.dumps(done, ensure_ascii=True, indent=2) + "\n")
    logger.info("done %s: processed=%s pb_valid=%s rate=%.4f elapsed=%.1fs", task.name, len(rows), valid_count, done["pb_valid_rate"], elapsed)
    return csv_path


def write_summaries() -> None:
    per_task = OUT_DIR / "per_task"
    summary_rows: list[dict[str, Any]] = []
    for csv_path in sorted(per_task.glob("*_posebusters.csv")):
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        total = len(df)
        valid_count = int(df["pb_valid"].fillna(False).astype(bool).sum())
        first = df.iloc[0]
        summary_rows.append(
            {
                "step": int(first["step"]),
                "repeat": int(first["repeat"]),
                "name": first["name"],
                "total": total,
                "pb_valid_count": valid_count,
                "pb_valid_rate": valid_count / total if total else math.nan,
            }
        )
    if not summary_rows:
        return
    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "posebusters_per_task.csv", index=False)

    agg_rows: list[dict[str, Any]] = []
    for step in STEPS:
        step_rows = [row for row in summary_rows if row["step"] == step]
        rates = [row["pb_valid_rate"] for row in step_rows]
        if not rates:
            continue
        agg_rows.append(
            {
                "diffusion_steps": step,
                "repeat_count": len(rates),
                "pb_valid_rate_mean": statistics.fmean(rates),
                "pb_valid_rate_sd": statistics.stdev(rates) if len(rates) > 1 else 0.0,
                "pb_valid_rate_formatted": f"{statistics.fmean(rates) * 100:.2f} ± {statistics.stdev(rates) * 100:.2f}" if len(rates) > 1 else f"{rates[0] * 100:.2f} ± 0.00",
            }
        )
    pd.DataFrame(agg_rows).to_csv(OUT_DIR / "posebusters_step_summary.csv", index=False)
    logger.info("wrote summaries: %s", OUT_DIR / "posebusters_step_summary.csv")


def run_single_task(name: str, *, force: bool, checkpoint_every: int) -> int:
    task = next((t for t in _task_specs() if t.name == name), None)
    if task is None:
        raise KeyError(f"unknown task name: {name}")
    process_task(task, receptor=RECEPTOR, force=bool(force), checkpoint_every=int(checkpoint_every))
    return 0


def run_queue(*, max_parallel: int, force: bool, checkpoint_every: int, max_load: float, check_interval: int) -> int:
    if not RECEPTOR.exists():
        raise FileNotFoundError(f"receptor not found: {RECEPTOR}")
    tasks = _task_specs()
    missing = [str(t.sdf_path) for t in tasks if not t.sdf_path.exists()]
    if missing:
        raise FileNotFoundError("Missing docked SDFs:\n" + "\n".join(missing))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    running: list[dict[str, Any]] = []
    pending = list(tasks)
    started_at = time.time()

    while pending or running:
        while pending and len(running) < max_parallel:
            task = pending.pop(0)
            done_path = OUT_DIR / "status" / f"{task.name}.done.json"
            csv_path = OUT_DIR / "per_task" / f"{task.name}_posebusters.csv"
            if done_path.exists() and csv_path.exists() and not force:
                logger.info("skip %s (already done)", task.name)
                write_summaries()
                continue
            _wait_for_load(max_load, check_interval)
            cmd = [
                sys.executable,
                str(SCRIPT_PATH),
                "--single-task",
                task.name,
                "--checkpoint-every",
                str(checkpoint_every),
            ]
            if force:
                cmd.append("--force")
            log_path = OUT_DIR / "logs" / f"{task.name}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open("w", encoding="utf-8")
            log_handle.write("# command: " + " ".join(cmd) + "\n")
            log_handle.flush()
            env = os.environ.copy()
            env.setdefault("OMP_NUM_THREADS", "1")
            env.setdefault("MKL_NUM_THREADS", "1")
            env.setdefault("OPENBLAS_NUM_THREADS", "1")
            env.setdefault("NUMEXPR_NUM_THREADS", "1")
            proc = subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT, env=env)
            running.append({"task": task, "proc": proc, "log_handle": log_handle, "started": time.time()})
            logger.info("[run] %s", task.name)

        time.sleep(5)
        still_running: list[dict[str, Any]] = []
        for item in running:
            proc = item["proc"]
            if proc.poll() is None:
                still_running.append(item)
                continue
            item["log_handle"].close()
            task = item["task"]
            seconds = time.time() - float(item["started"])
            status = "ok" if proc.returncode == 0 else "failed"
            logger.info("[%s] %s returncode=%s seconds=%.1f", status, task.name, proc.returncode, seconds)
            if status != "ok":
                for other in still_running:
                    other["proc"].terminate()
                return 1
            write_summaries()
        running = still_running

    logger.info("queue completed in %.1fs", time.time() - started_at)
    write_summaries()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-task", default=None, help="Process a single task name (used by the queue worker).")
    parser.add_argument("--receptor", default=str(RECEPTOR), help="Receptor PDB for PoseBusters dock config.")
    parser.add_argument("--out-dir", default=str(OUT_DIR), help="Output directory.")
    parser.add_argument("--max-parallel", type=int, default=3, help="Max parallel task workers in queue mode.")
    parser.add_argument("--max-load", type=float, default=18.0, help="Wait while 1-min load average exceeds this.")
    parser.add_argument("--check-interval", type=int, default=120, help="Seconds between load checks.")
    parser.add_argument("--checkpoint-every", type=int, default=25, help="Partial CSV checkpoint interval.")
    parser.add_argument("--force", action="store_true", help="Recompute tasks with done markers.")
    return parser.parse_args()


def main() -> None:
    global RECEPTOR, OUT_DIR
    args = parse_args()
    RECEPTOR = Path(args.receptor)
    OUT_DIR = Path(args.out_dir)
    if args.single_task:
        raise SystemExit(run_single_task(args.single_task, force=args.force, checkpoint_every=args.checkpoint_every))
    raise SystemExit(
        run_queue(
            max_parallel=args.max_parallel,
            force=args.force,
            checkpoint_every=args.checkpoint_every,
            max_load=args.max_load,
            check_interval=args.check_interval,
        )
    )


if __name__ == "__main__":
    main()
