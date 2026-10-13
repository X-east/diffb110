#!/usr/bin/env python
"""Stage 2: dock completed 3D shards, one UniDock ``--ligand_index`` call per shard.

Pipeline worker. It repeatedly claims the next un-docked shard produced by
``gen_3d_shards.py``, materialises that shard's molecules as per-molecule SDF
files (UniDock reads only the first record of a multi-record SDF, verified), runs
ONE UniDock call with ``--max_gpu_memory 0``, parses the results, deletes the
temporary per-molecule files, and marks the shard docked.

Concurrency
-----------
Shards are claimed with an ``O_EXCL`` claim file so two workers on different GPUs
never dock the same shard. A worker exits once the Stage-1 ``gen_done.json``
marker exists and no claimable shard remains.

Usage
-----
    CUDA_VISIBLE_DEVICES=0 conda run -n dif python dock_worker.py \\
        --out-root analysis/chembl_docking_reference_260928/full \\
        --gpu-id 0
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import subprocess
import time
from pathlib import Path

from rdkit import RDLogger

LOGGER = logging.getLogger("dock-worker")
DEFAULT_UNIDOCK = "/opt/unidock-1.1.3/bin/unidock"

SCORE_PATTERNS = [
    re.compile(r"REMARK\s+VINA RESULT:\s*([-+]?\d+(?:\.\d+)?)"),
    re.compile(r"ENERGY=\s*([-+]?\d+(?:\.\d+)?)"),
    re.compile(
        r">\s*<Uni-Dock RESULT>\s*\n(?:.*\n)*?ENERGY=\s*([-+]?\d+(?:\.\d+)?)",
        re.MULTILINE,
    ),
    re.compile(r">\s*<[Ss]core>\s*\n\s*([-+]?\d+(?:\.\d+)?)", re.MULTILINE),
]


def _parse_score(path: Path) -> float | None:
    """Return the docking score from one UniDock output file."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    for pattern in SCORE_PATTERNS:
        match = pattern.search(text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
    return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--receptor", type=Path, default=Path("data/8gjl.pdbqt"))
    parser.add_argument("--unidock-bin", type=Path, default=Path(DEFAULT_UNIDOCK))
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
    parser.add_argument("--timeout", type=int, default=14400)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-shards", type=int, default=0, help="0 = no limit")
    parser.add_argument("--redo-shards", default="",
                        help="Comma-separated shard indices to force-redo (ignores ok/failed state)")
    parser.add_argument("--redo-tag", default="",
                        help="Shared tag for the redo claim file (must match across workers)")
    return parser.parse_args()


def _claim(claim_dir: Path, shard_idx: int, suffix: str = "") -> bool:
    """Atomically claim a shard; False if another worker already claimed it."""
    path = claim_dir / f"shard_{shard_idx:05d}{suffix}.claim"
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        return True
    except FileExistsError:
        return False


def _shard_state(root: Path, idx: int) -> str:
    """Return ``'ok'``, ``'failed'`` or ``'none'`` for one shard's dock state."""
    summary = root / f"docking/shard_{idx:05d}/shard_summary.json"
    if not summary.exists():
        return "none"
    try:
        data = json.loads(summary.read_text(encoding="utf-8"))
    except Exception:
        return "failed"
    if data.get("returncode", 0) == 0 and data.get("success_count", 0) > 0:
        return "ok"
    return "failed"


def _needs_work(root: Path, claim_dir: Path, marker: Path) -> bool:
    """True while a shard has not reached a terminal state (ok / failed+retried)."""
    idx = int(marker.stem.split("_")[1])
    state = _shard_state(root, idx)
    if state == "ok":
        return False
    if state == "failed":
        return not (claim_dir / f"shard_{idx:05d}.retry.claim").exists()
    return True


SUB_BATCH_SIZE = 500


def _run_unidock(args: argparse.Namespace, ligand_paths: list[Path], index_path: Path,
                 out_dir: Path, log_path: Path) -> int:
    """Write the index, run one UniDock ``--ligand_index`` call, return rc."""
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text("\n".join(str(p.resolve()) for p in ligand_paths) + "\n",
                          encoding="utf-8")
    cmd = [
        str(args.unidock_bin), "--receptor", str(args.receptor),
        "--ligand_index", str(index_path),
        "--center_x", str(args.center_x), "--center_y", str(args.center_y),
        "--center_z", str(args.center_z), "--size_x", str(args.size_x),
        "--size_y", str(args.size_y), "--size_z", str(args.size_z),
        "--scoring", args.scoring, "--search_mode", args.search_mode,
        "--num_modes", str(args.num_modes), "--seed", str(args.seed),
        "--max_gpu_memory", str(args.max_gpu_memory),
        "--verbosity", str(args.verbosity), "--dir", str(out_dir),
    ]
    with log_path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT,
                                timeout=args.timeout, check=False)
    return result.returncode


def _collect(out_dir: Path, names: list[str]) -> dict[str, float | None]:
    """Parse docking scores for the given ligand stems from an output dir."""
    scores: dict[str, float | None] = {}
    for name in names:
        cand = [out_dir / f"{name}_out.sdf", out_dir / f"{name}.sdf",
                out_dir / f"{name}_out.pdbqt", out_dir / f"{name}.pdbqt"]
        out_path = next((c for c in cand if c.exists()), None)
        scores[name] = _parse_score(out_path) if out_path is not None else None
    return scores


def _dock_shard(args: argparse.Namespace, shard_idx: int) -> dict:
    """Dock one shard; on backend failure fall back to sub-batch isolation.

    A whole-shard UniDock segfault (return code -11 has been observed) would lose
    the entire 20k shard, so a failed shard is re-submitted in small sub-batches:
    a single bad ligand then only spoils its own sub-batch. No files are created
    or deleted by the split itself (deletion is blocked in this environment).
    """
    root = args.out_root
    ligand_dir = root / "ligands" / f"shard_{shard_idx:05d}"
    dock_dir = root / f"docking/shard_{shard_idx:05d}"
    out_dir = dock_dir / "outputs"
    dock_dir.mkdir(parents=True, exist_ok=True)

    ligand_paths = sorted(ligand_dir.glob("*.sdf"))
    names = [p.stem for p in ligand_paths]
    if not ligand_paths:
        return {"shard_idx": shard_idx, "num_ligands": 0, "success_count": 0,
                "seconds": 0.0, "throughput_mol_per_s": 0.0, "returncode": -1}

    started = time.time()
    rc = _run_unidock(args, ligand_paths, dock_dir / "ligand_index.txt",
                      out_dir, dock_dir / "unidock.log")
    scores = _collect(out_dir, names)
    found = sum(1 for v in scores.values() if v is not None)
    if rc != 0 or found == 0:
        LOGGER.warning("shard %d: whole-shard rc=%d found=%d/%d; "
                       "falling back to sub-batches of %d",
                       shard_idx, rc, found, len(names), SUB_BATCH_SIZE)
        scores = {}
        for bi, start in enumerate(range(0, len(ligand_paths), SUB_BATCH_SIZE)):
            sub_paths = ligand_paths[start:start + SUB_BATCH_SIZE]
            sub_names = [p.stem for p in sub_paths]
            sub_out = dock_dir / f"sub_{bi:05d}"
            sub_rc = _run_unidock(
                args, sub_paths, dock_dir / f"sub_{bi:05d}_index.txt",
                sub_out, dock_dir / f"sub_{bi:05d}.log")
            if sub_rc != 0:
                LOGGER.warning("shard %d sub %d rc=%d", shard_idx, bi, sub_rc)
            scores.update(_collect(sub_out, sub_names))
    elapsed = time.time() - started

    rows = []
    for name in names:
        score = scores.get(name)
        rows.append({"chembl_id": name, "success": score is not None,
                     "score": score if score is not None else "",
                     "failure_reason": "" if score is not None else "missing_output"})

    with (dock_dir / "docking_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "chembl_id", "success", "score", "failure_reason"])
        writer.writeheader()
        writer.writerows(rows)

    ok = sum(1 for r in rows if r["success"])
    effective_rc = 0 if ok > 0 else rc
    summary = {"shard_idx": shard_idx, "num_ligands": len(rows), "success_count": ok,
               "success_rate": ok / len(rows) if rows else 0.0, "seconds": elapsed,
               "throughput_mol_per_s": len(rows) / elapsed if elapsed else 0.0,
               "returncode": effective_rc, "whole_shard_rc": rc,
               "gpu_id": args.gpu_id}
    (dock_dir / "shard_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    """Claim and dock shards until the generator is done and none remain."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    RDLogger.DisableLog("rdApp.*")
    root = args.out_root
    marker_dir = root / "markers"
    claim_dir = root / "claims"
    claim_dir.mkdir(parents=True, exist_ok=True)
    gen_done = root / "gen_done.json"
    docked_this_worker = 0

    redo = {int(x) for x in str(args.redo_shards).split(",") if x.strip()}
    tag = args.redo_tag or ""
    idle = 0

    while True:
        markers = sorted(marker_dir.glob("shard_*.done"))
        target = None
        for marker in markers:
            idx = int(marker.stem.split("_")[1])
            state = _shard_state(root, idx)
            if state == "ok" and idx not in redo:
                continue
            if idx in redo:
                suffix = f".redo{tag}"
            elif state == "failed":
                suffix = f".retry{tag}"
            else:
                # 'none' (not yet docked). Use a per-run tag so a claim left over
                # from a killed previous run can never strand the shard.
                suffix = f".run{tag}"
            if _claim(claim_dir, idx, suffix):
                target = idx
                break
        if target is None:
            if gen_done.exists():
                idle += 1
                if idle >= 3:
                    LOGGER.info("generator finished and nothing claimable after %d "
                                "polls; exit (docked=%d)", idle, docked_this_worker)
                    return
            time.sleep(args.poll_seconds)
            continue
        idle = 0

        LOGGER.info("gpu=%d claiming shard %d", args.gpu_id, target)
        summary = _dock_shard(args, target)
        (root / "markers" / f"shard_{target:05d}.docked").write_text(
            json.dumps(summary) + "\n", encoding="utf-8")
        docked_this_worker += 1
        LOGGER.info("gpu=%d shard %d done: %d/%d ok, %.1fs, %.2f mol/s, rc=%d",
                    args.gpu_id, target, summary["success_count"],
                    summary["num_ligands"], summary["seconds"],
                    summary["throughput_mol_per_s"], summary["returncode"])
        if args.max_shards and docked_this_worker >= args.max_shards:
            LOGGER.info("reached --max-shards=%d; exit", args.max_shards)
            return


if __name__ == "__main__":
    main()
