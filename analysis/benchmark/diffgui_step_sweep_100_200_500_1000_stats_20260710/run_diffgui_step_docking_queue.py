"""Run UniDock for DiffGui step-sweep sanitized molecules.

The queue intentionally uses Python subprocesses instead of a large shell
matrix so later tasks do not depend on fragile shell quoting.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DockingTask:
    """One DiffGui step-sweep repeat to dock."""

    step: int
    repeat: int
    seed_label: str
    input_sdf: Path

    @property
    def name(self) -> str:
        return f"{self.step}steps_rep{self.repeat}_{self.seed_label}"


SCRIPT_PATH = Path(__file__).resolve()
STATS_DIR = SCRIPT_PATH.parent
REPO_ROOT = SCRIPT_PATH.parents[3]
DOCK_SCRIPT = (
    REPO_ROOT
    / "analysis/benchmark/8gjl_c1101_benchmark_20260624_1000mol_500steps/scripts/dock_sdf_unidock.py"
)
RECEPTOR = REPO_ROOT / "data/8gjl.pdbqt"
UNIDOCK_BIN = Path("/opt/unidock-1.1.3/bin/unidock")
OUT_ROOT = STATS_DIR / "docking_100_200_500"
LOG_DIR = OUT_ROOT / "logs"
MANIFEST_CSV = OUT_ROOT / "docking_task_manifest.csv"
SUMMARY_JSON = OUT_ROOT / "docking_queue_summary.json"


def _task_specs() -> list[DockingTask]:
    root_100 = REPO_ROOT / "analysis/benchmark/diffgui_step_sweep_100_1000attempts_20260709"
    root_200_500 = REPO_ROOT / "analysis/benchmark/diffgui_step_sweep_500_200_1000attempts_20260707_run2"
    return [
        DockingTask(100, 1, "seed1_7", root_100 / "100steps_rep1_seed1_7/filtered_sanitize_valid/diffgui.sanitize_valid.sdf"),
        DockingTask(100, 2, "seed8_14", root_100 / "100steps_rep2_seed8_14/filtered_sanitize_valid/diffgui.sanitize_valid.sdf"),
        DockingTask(100, 3, "seed15_21", root_100 / "100steps_rep3_seed15_21/filtered_sanitize_valid/diffgui.sanitize_valid.sdf"),
        DockingTask(200, 1, "seed1_7", root_200_500 / "200steps_rep1_seed1_7/filtered_sanitize_valid/diffgui.sanitize_valid.sdf"),
        DockingTask(200, 2, "seed8_14", root_200_500 / "200steps_rep2_seed8_14/filtered_sanitize_valid/diffgui.sanitize_valid.sdf"),
        DockingTask(200, 3, "seed15_21", root_200_500 / "200steps_rep3_seed15_21/filtered_sanitize_valid/diffgui.sanitize_valid.sdf"),
        DockingTask(500, 1, "seed1_7", root_200_500 / "500steps_rep1_seed1_7/filtered_sanitize_valid/diffgui.sanitize_valid.sdf"),
        DockingTask(500, 2, "seed8_14", root_200_500 / "500steps_rep2_seed8_14/filtered_sanitize_valid/diffgui.sanitize_valid.sdf"),
        DockingTask(500, 3, "seed15_21", root_200_500 / "500steps_rep3_seed15_21/filtered_sanitize_valid/diffgui.sanitize_valid.sdf"),
    ]


def _read_env_list(name: str, default: str) -> list[str]:
    value = os.environ.get(name, default)
    items = [item.strip() for item in value.replace(",", " ").split() if item.strip()]
    if not items:
        raise ValueError(f"{name} must contain at least one GPU id")
    return items


def _run_summary_ok(out_dir: Path) -> bool:
    summary_path = out_dir / "run_summary.json"
    results_path = out_dir / "docking_results.csv"
    docked_path = out_dir / "successful_docked_outputs.sdf"
    if not (summary_path.is_file() and results_path.is_file() and docked_path.is_file()):
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return bool(summary.get("total_ligands")) and summary.get("success_count") == summary.get("merged_success_sdf_count")


def _write_manifest(rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "step",
        "repeat",
        "task",
        "gpu_id",
        "status",
        "input_sdf",
        "out_dir",
        "log_path",
        "seconds",
        "returncode",
    ]
    MANIFEST_CSV.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    max_parallel = int(os.environ.get("MAX_PARALLEL", "3"))
    batch_size = os.environ.get("BATCH_SIZE", "128")
    force = os.environ.get("FORCE", "0") == "1"
    gpu_ids = _read_env_list("GPU_IDS", "0 1 2")
    if max_parallel < 1:
        raise ValueError("MAX_PARALLEL must be >= 1")
    max_parallel = min(max_parallel, len(gpu_ids))

    for label, path in [("dock_script", DOCK_SCRIPT), ("receptor", RECEPTOR), ("unidock_bin", UNIDOCK_BIN)]:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    tasks = _task_specs()
    missing = [str(task.input_sdf) for task in tasks if not task.input_sdf.exists()]
    if missing:
        raise FileNotFoundError("Missing input SDFs:\n" + "\n".join(missing))

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    running: list[dict[str, object]] = []
    pending = list(tasks)
    started_at = time.time()

    print(
        json.dumps(
            {
                "event": "queue_start",
                "out_root": str(OUT_ROOT),
                "tasks": len(tasks),
                "gpu_ids": gpu_ids,
                "max_parallel": max_parallel,
                "batch_size": batch_size,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    while pending or running:
        while pending and len(running) < max_parallel:
            task = pending.pop(0)
            gpu_id = gpu_ids[len(rows + running) % len(gpu_ids)]
            out_dir = OUT_ROOT / task.name / "diffgui"
            log_path = LOG_DIR / f"{task.name}.log"
            if _run_summary_ok(out_dir) and not force:
                rows.append(
                    {
                        "step": task.step,
                        "repeat": task.repeat,
                        "task": task.name,
                        "gpu_id": gpu_id,
                        "status": "skipped_existing",
                        "input_sdf": str(task.input_sdf),
                        "out_dir": str(out_dir),
                        "log_path": str(log_path),
                        "seconds": 0.0,
                        "returncode": 0,
                    }
                )
                _write_manifest(rows)
                print(f"[skip] {task.name} existing {out_dir}", flush=True)
                continue

            out_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                str(DOCK_SCRIPT),
                "--model",
                "DiffGui",
                "--input-sdf",
                str(task.input_sdf),
                "--receptor",
                str(RECEPTOR),
                "--out-dir",
                str(out_dir),
                "--unidock-bin",
                str(UNIDOCK_BIN),
                "--gpu-id",
                str(gpu_id),
                "--batch-size",
                str(batch_size),
            ]
            log_handle = log_path.open("w", encoding="utf-8")
            log_handle.write("# command: " + " ".join(cmd) + "\n")
            log_handle.flush()
            env = os.environ.copy()
            env.setdefault("OMP_NUM_THREADS", "1")
            env.setdefault("MKL_NUM_THREADS", "1")
            env.setdefault("OPENBLAS_NUM_THREADS", "1")
            env.setdefault("NUMEXPR_NUM_THREADS", "1")
            proc = subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT, env=env)
            running.append(
                {
                    "task": task,
                    "gpu_id": gpu_id,
                    "out_dir": out_dir,
                    "log_path": log_path,
                    "log_handle": log_handle,
                    "proc": proc,
                    "started": time.time(),
                }
            )
            print(f"[run] {task.name} gpu={gpu_id} out={out_dir}", flush=True)

        time.sleep(5)
        still_running: list[dict[str, object]] = []
        for item in running:
            proc = item["proc"]
            returncode = proc.poll()
            if returncode is None:
                still_running.append(item)
                continue
            item["log_handle"].close()
            task = item["task"]
            seconds = time.time() - float(item["started"])
            status = "ok" if returncode == 0 and _run_summary_ok(item["out_dir"]) else "failed"
            row = {
                "step": task.step,
                "repeat": task.repeat,
                "task": task.name,
                "gpu_id": item["gpu_id"],
                "status": status,
                "input_sdf": str(task.input_sdf),
                "out_dir": str(item["out_dir"]),
                "log_path": str(item["log_path"]),
                "seconds": round(seconds, 3),
                "returncode": returncode,
            }
            rows.append(row)
            _write_manifest(rows)
            print(f"[{status}] {task.name} returncode={returncode} seconds={seconds:.1f}", flush=True)
            if status != "ok":
                for pending_item in still_running:
                    pending_item["proc"].terminate()
                SUMMARY_JSON.write_text(
                    json.dumps(
                        {
                            "status": "failed",
                            "failed_task": task.name,
                            "manifest": str(MANIFEST_CSV),
                            "seconds": time.time() - started_at,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                return 1
        running = still_running

    status = "ok" if all(row["status"] in {"ok", "skipped_existing"} for row in rows) else "failed"
    SUMMARY_JSON.write_text(
        json.dumps(
            {
                "status": status,
                "manifest": str(MANIFEST_CSV),
                "tasks": len(rows),
                "seconds": time.time() - started_at,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[queue] completed status={status} manifest={MANIFEST_CSV}", flush=True)
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
