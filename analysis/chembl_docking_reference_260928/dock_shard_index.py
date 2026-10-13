#!/usr/bin/env python
"""Dock one shard of ligands with a SINGLE UniDock ``--ligand_index`` call.

Replaces the old per-128 batch loop (``scripts/docking/dock_sdf_unidock.py``).
UniDock's own FAQ says the optimum is "one pocket x ~1000 ligands", that larger
batches are faster, and that ``--max_gpu_memory 0`` makes UniDock split the
ligand list into GPU batches by available memory. This script therefore submits
the whole shard once and lets UniDock batch internally, instead of launching a
new process every 64/128 ligands.

Failure handling
----------------
UniDock writes each ligand's result incrementally into ``--dir``. If a chunk
inside UniDock aborts, the already written results survive; this script parses
whatever exists and records the rest as ``missing_output`` so a re-run only
needs to resubmit the missing ligands (``--resume``).

Usage
-----
    CUDA_VISIBLE_DEVICES=0 conda run -n dif python dock_shard_index.py \\
        --ligand-dir <dir with one .sdf per ligand> \\
        --shard-id 0 --num-shards 8 \\
        --receptor data/8gjl.pdbqt \\
        --center-x 207.293 --center-y 212.430 --center-z 172.771 \\
        --size-x 20 --size-y 20 --size-z 20 \\
        --scoring vina --search-mode fast --num-modes 1 --seed 42 \\
        --max-gpu-memory 0 \\
        --out-dir <out dir for this shard>
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import subprocess
import time
from pathlib import Path

LOGGER = logging.getLogger("dock-shard")

DEFAULT_UNIDOCK = "/opt/unidock-1.1.3/bin/unidock"

SCORE_PATTERNS = [
    re.compile(r"REMARK\s+VINA RESULT:\s*([-+]?\d+(?:\.\d+)?)"),
    re.compile(r"ENERGY=\s*([-+]?\d+(?:\.\d+)?)"),
    re.compile(
        r">\s*<Uni-Dock RESULT>\s*\n(?:.*\n)*?ENERGY=\s*([-+]?\d+(?:\.\d+)?)",
        re.MULTILINE,
    ),
    re.compile(
        r">\s*<[Dd]ocking[_\s][Ss]core>\s*\n\s*([-+]?\d+(?:\.\d+)?)",
        re.MULTILINE,
    ),
    re.compile(r">\s*<[Ss]core>\s*\n\s*([-+]?\d+(?:\.\d+)?)", re.MULTILINE),
]


def _parse_score(path: Path) -> float | None:
    """Return the Vina/UniDock score from one output file, or ``None``."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    for pattern in SCORE_PATTERNS:
        match = pattern.search(text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
    return None


def _output_candidates(out_dir: Path, stem: str) -> list[Path]:
    """Candidate UniDock output paths for one ligand stem."""
    return [
        out_dir / f"{stem}_out.sdf",
        out_dir / f"{stem}.sdf",
        out_dir / f"{stem}_out.pdbqt",
        out_dir / f"{stem}.pdbqt",
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ligand-dir", type=Path, required=True,
                        help="Directory containing one single-molecule .sdf per ligand")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--receptor", type=Path, default=Path("data/8gjl.pdbqt"))
    parser.add_argument("--unidock-bin", type=Path, default=Path(DEFAULT_UNIDOCK))
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--center-x", type=float, default=207.293)
    parser.add_argument("--center-y", type=float, default=212.430)
    parser.add_argument("--center-z", type=float, default=172.771)
    parser.add_argument("--size-x", type=float, default=20.0)
    parser.add_argument("--size-y", type=float, default=20.0)
    parser.add_argument("--size-z", type=float, default=20.0)
    parser.add_argument("--scoring", default="vina")
    parser.add_argument("--search-mode", default="fast")
    parser.add_argument("--num-modes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-gpu-memory", type=int, default=0)
    parser.add_argument("--verbosity", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=86400)
    parser.add_argument("--resume", action="store_true",
                        help="Re-submit only ligands whose output is still missing")
    parser.add_argument("--retain-ligands", action="store_true",
                        help="Keep the per-ligand copied inputs (default: reference in place)")
    return parser.parse_args()


def _select_shard(ligands: list[Path], shard_id: int, num_shards: int) -> list[Path]:
    """Deterministic round-robin shard selection."""
    return [p for i, p in enumerate(ligands) if i % num_shards == shard_id]


def _build_command(args: argparse.Namespace, index_path: Path, out_dir: Path) -> list[str]:
    """Build one UniDock command using the ``--ligand_index`` form."""
    return [
        str(args.unidock_bin),
        "--receptor", str(args.receptor),
        "--ligand_index", str(index_path),
        "--center_x", str(args.center_x),
        "--center_y", str(args.center_y),
        "--center_z", str(args.center_z),
        "--size_x", str(args.size_x),
        "--size_y", str(args.size_y),
        "--size_z", str(args.size_z),
        "--scoring", args.scoring,
        "--search_mode", args.search_mode,
        "--num_modes", str(args.num_modes),
        "--seed", str(args.seed),
        "--max_gpu_memory", str(args.max_gpu_memory),
        "--verbosity", str(args.verbosity),
        "--dir", str(out_dir),
    ]


def main() -> None:
    """Run one shard through a single UniDock ``--ligand_index`` submission."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    dock_dir = out_dir / "docking_outputs"
    dock_dir.mkdir(parents=True, exist_ok=True)

    ligands = sorted(args.ligand_dir.glob("*.sdf"))
    if not ligands:
        raise SystemExit(f"no .sdf ligands found in {args.ligand_dir}")
    shard = _select_shard(ligands, args.shard_id, args.num_shards)
    LOGGER.info("shard %d/%d: %d of %d ligands", args.shard_id, args.num_shards,
                len(shard), len(ligands))

    if args.resume:
        pending = [p for p in shard
                   if not any(c.exists() for c in _output_candidates(dock_dir, p.stem))]
        LOGGER.info("resume: %d ligands still missing output", len(pending))
        shard = pending

    index_path = out_dir / "ligand_index.txt"
    index_path.write_text("\n".join(str(p.resolve()) for p in shard) + "\n",
                          encoding="utf-8")

    started = time.time()
    cmd = _build_command(args, index_path, dock_dir)
    (out_dir / "unidock_command.json").write_text(
        json.dumps({"cmd": cmd, "shard_id": args.shard_id,
                    "num_ligands": len(shard)}, indent=2) + "\n",
        encoding="utf-8")

    env_log = out_dir / "unidock.log"
    with env_log.open("w", encoding="utf-8") as log_handle:
        result = subprocess.run(cmd, stdout=log_handle, stderr=subprocess.STDOUT,
                                timeout=args.timeout, check=False)
    elapsed = time.time() - started

    rows: list[dict[str, object]] = []
    for ligand in shard:
        out_path = next((c for c in _output_candidates(dock_dir, ligand.stem)
                         if c.exists()), None)
        if out_path is None:
            rows.append({"ligand_stem": ligand.stem, "input_sdf": str(ligand),
                         "success": False, "score": "", "docking_output": "",
                         "failure_reason": "missing_output"})
            continue
        score = _parse_score(out_path)
        rows.append({
            "ligand_stem": ligand.stem,
            "input_sdf": str(ligand),
            "success": score is not None,
            "score": score if score is not None else "",
            "docking_output": str(out_path),
            "failure_reason": "" if score is not None else "score_not_found",
        })

    results_csv = out_dir / "docking_results.csv"
    with results_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "ligand_stem", "input_sdf", "success", "score",
            "docking_output", "failure_reason"])
        writer.writeheader()
        writer.writerows(rows)

    ok = [r for r in rows if r["success"]]
    summary = {
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "num_ligands": len(shard),
        "success_count": len(ok),
        "success_rate": len(ok) / len(shard) if shard else 0.0,
        "seconds": elapsed,
        "throughput_mol_per_s": len(shard) / elapsed if elapsed else 0.0,
        "unidock_returncode": result.returncode,
        "submit_mode": "ligand_index_single_call",
        "max_gpu_memory": args.max_gpu_memory,
        "search_mode": args.search_mode,
        "seed": args.seed,
        "box": {"center_x": args.center_x, "center_y": args.center_y,
                "center_z": args.center_z, "size_x": args.size_x,
                "size_y": args.size_y, "size_z": args.size_z},
        "results_csv": str(results_csv),
    }
    (out_dir / "shard_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    LOGGER.info("shard %d done: %d/%d ok, %.1fs, %.2f mol/s, returncode=%d",
                args.shard_id, len(ok), len(shard), elapsed,
                summary["throughput_mol_per_s"], result.returncode)


if __name__ == "__main__":
    main()
