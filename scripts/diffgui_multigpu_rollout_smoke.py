"""Launch multiple independent DiffGui 100-step smoke workers across GPUs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SMOKE_SCRIPT = PROJECT_ROOT / "scripts/diffgui_100step_rollout_smoke.py"


def parse_args() -> argparse.Namespace:
    """Parse multi-GPU smoke launcher arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "analysis/diffgui_100step_multigpu_smoke_20260609")
    parser.add_argument("--num-timesteps", type=int, default=100)
    parser.add_argument("--num-graphs-per-gpu", type=int, default=2)
    parser.add_argument("--ligand-atoms", type=int, default=24)
    parser.add_argument("--seed-base", type=int, default=260609)
    parser.add_argument("--gui-strength", type=float, default=0.0)
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--freeze-mode", type=str, default="last_n_blocks", choices=["none", "heads_only", "last_n_blocks"])
    parser.add_argument("--train-last-n-blocks", type=int, default=1)
    parser.add_argument("--graph-micro-batch-size", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    """Run one smoke process per requested GPU and aggregate summaries."""
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    gpu_ids = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpu_ids:
        raise ValueError("--gpus must contain at least one GPU id.")

    workers: list[dict[str, Any]] = []
    for rank, gpu_id in enumerate(gpu_ids):
        rank_dir = output_dir / f"rank{rank:02d}_gpu{gpu_id}"
        rank_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / f"rank{rank:02d}_gpu{gpu_id}.log"
        cmd = [
            sys.executable,
            str(SMOKE_SCRIPT),
            "--output-dir",
            str(rank_dir),
            "--device",
            "cuda:0",
            "--num-timesteps",
            str(args.num_timesteps),
            "--num-graphs",
            str(args.num_graphs_per_gpu),
            "--ligand-atoms",
            str(args.ligand_atoms),
            "--seed",
            str(args.seed_base + rank),
            "--gui-strength",
            str(args.gui_strength),
            "--precision",
            str(args.precision),
            "--graph-micro-batch-size",
            str(args.graph_micro_batch_size),
            "--freeze-mode",
            str(args.freeze_mode),
            "--train-last-n-blocks",
            str(args.train_last_n_blocks),
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env["PYTHONUNBUFFERED"] = "1"
        handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT)
        workers.append(
            {
                "rank": rank,
                "gpu": gpu_id,
                "rank_dir": rank_dir,
                "log_path": log_path,
                "process": process,
                "handle": handle,
            }
        )
        print(f"[launch] rank={rank} gpu={gpu_id} pid={process.pid} log={log_path}", flush=True)

    summaries: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for worker in workers:
        process = worker["process"]
        return_code = process.wait()
        worker["handle"].close()
        summary_path = worker["rank_dir"] / "summary.json"
        row = {
            "rank": worker["rank"],
            "gpu": worker["gpu"],
            "return_code": int(return_code),
            "rank_dir": str(worker["rank_dir"]),
            "log_path": str(worker["log_path"]),
        }
        if return_code == 0 and summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            row["summary"] = summary
            summaries.append(row)
            print(f"[done] rank={worker['rank']} gpu={worker['gpu']} ok", flush=True)
        else:
            failed.append(row)
            print(f"[fail] rank={worker['rank']} gpu={worker['gpu']} rc={return_code}", flush=True)

    aggregate = _aggregate(args, gpu_ids, summaries, failed)
    summary_output = output_dir / "multi_summary.json"
    summary_output.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(aggregate, indent=2, ensure_ascii=False), flush=True)
    if failed:
        raise SystemExit(1)


def _aggregate(
    args: argparse.Namespace,
    gpu_ids: list[str],
    summaries: list[dict[str, Any]],
    failed: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate per-GPU smoke summaries into one JSON object."""
    per_rank = [row["summary"] for row in summaries]
    total_graphs = sum(int(summary.get("num_graphs", 0)) for summary in per_rank)
    total_attempted = sum(int(summary.get("reconstruction", {}).get("attempted", 0)) for summary in per_rank)
    total_success = sum(int(summary.get("reconstruction", {}).get("success", 0)) for summary in per_rank)
    return {
        "stage": "diffgui_100step_multigpu_smoke",
        "requested_gpus": gpu_ids,
        "successful_workers": len(summaries),
        "failed_workers": len(failed),
        "total_graphs": total_graphs,
        "num_timesteps": int(args.num_timesteps),
        "num_graphs_per_gpu": int(args.num_graphs_per_gpu),
        "ligand_atoms": int(args.ligand_atoms),
        "seed_base": int(args.seed_base),
        "precision": str(args.precision),
        "freeze_mode": str(args.freeze_mode),
        "train_last_n_blocks": int(args.train_last_n_blocks),
        "graph_micro_batch_size": int(args.graph_micro_batch_size),
        "reconstruction_attempted": total_attempted,
        "reconstruction_success": total_success,
        "reconstruction_success_ratio": float(total_success / total_attempted) if total_attempted else 0.0,
        "numsteps_values": [summary.get("actor_metrics", {}).get("NumSteps") for summary in per_rank],
        "trainable_params_values": [summary.get("actor_metrics", {}).get("TrainableParams") for summary in per_rank],
        "failed": failed,
        "workers": [
            {
                "rank": row["rank"],
                "gpu": row["gpu"],
                "return_code": row["return_code"],
                "rank_dir": row["rank_dir"],
                "log_path": row["log_path"],
                "seed": int(args.seed_base + int(row["rank"])),
                "old_logprob_shape": row["summary"].get("old_logprob_shape"),
                "actor_metrics": row["summary"].get("actor_metrics", {}),
                "reconstruction": row["summary"].get("reconstruction", {}),
            }
            for row in summaries
        ],
    }


if __name__ == "__main__":
    main()
