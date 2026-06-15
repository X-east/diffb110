"""Dock a multi-record SDF with UniDock and write per-ligand scores.

This is an experiment-local wrapper. It keeps the DDPO training docking box
and backend settings explicit while avoiding any dependency on rollout objects.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import time
from pathlib import Path
from statistics import mean, median

from rdkit import Chem, RDLogger


RDLogger.DisableLog("rdApp.*")

SCORE_PATTERNS = [
    re.compile(r"REMARK\s+VINA RESULT:\s*([-+]?\d+(?:\.\d+)?)"),
    re.compile(r"ENERGY=\s*([-+]?\d+(?:\.\d+)?)"),
    re.compile(r">\s*<Uni-Dock RESULT>\s*\n(?:.*\n)*?ENERGY=\s*([-+]?\d+(?:\.\d+)?)", re.MULTILINE),
    re.compile(r">\s*<[Dd]ocking[_\s][Ss]core>\s*\n\s*([-+]?\d+(?:\.\d+)?)", re.MULTILINE),
    re.compile(r">\s*<[Ss]core>\s*\n\s*([-+]?\d+(?:\.\d+)?)", re.MULTILINE),
]


def _chunked(items: list[Path], size: int):
    if size <= 0:
        yield items
        return
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _safe_smiles(mol: Chem.Mol) -> str:
    try:
        copy = Chem.Mol(mol)
        Chem.SanitizeMol(copy)
        return Chem.MolToSmiles(Chem.RemoveHs(copy), canonical=True, isomericSmiles=True)
    except Exception:
        return ""


def split_sdf(input_sdf: Path, split_dir: Path, model: str) -> tuple[list[Path], dict[str, dict[str, str]]]:
    split_dir.mkdir(parents=True, exist_ok=True)
    supplier = Chem.SDMolSupplier(str(input_sdf), sanitize=False, removeHs=False)
    ligand_paths: list[Path] = []
    metadata: dict[str, dict[str, str]] = {}
    read_failed = 0
    write_failed = 0
    for idx, mol in enumerate(supplier):
        if mol is None:
            read_failed += 1
            continue
        stem = f"{model.lower()}_{idx:06d}"
        path = split_dir / f"{stem}.sdf"
        mol.SetProp("benchmark_model", model)
        mol.SetProp("benchmark_index", str(idx))
        mol.SetProp("_Name", stem)
        try:
            writer = Chem.SDWriter(str(path))
            writer.write(mol)
            writer.close()
        except Exception:
            write_failed += 1
            continue
        ligand_paths.append(path)
        metadata[stem] = {
            "model": model,
            "benchmark_index": str(idx),
            "source_sdf": str(input_sdf),
            "split_sdf": str(path),
            "input_smiles": _safe_smiles(mol),
        }
    metadata["__split_summary__"] = {
        "input_sdf_records_seen": str(idx + 1 if "idx" in locals() else 0),
        "split_ligands": str(len(ligand_paths)),
        "read_failed": str(read_failed),
        "write_failed": str(write_failed),
    }
    return ligand_paths, metadata


def build_cmd(args: argparse.Namespace, ligands: list[Path], out_dir: Path) -> list[str]:
    cmd = [
        str(args.unidock_bin),
        "--receptor", str(args.receptor),
        "--gpu_batch", *[str(path) for path in ligands],
        "--center_x", str(args.center_x),
        "--center_y", str(args.center_y),
        "--center_z", str(args.center_z),
        "--size_x", str(args.size_x),
        "--size_y", str(args.size_y),
        "--size_z", str(args.size_z),
        "--scoring", args.scoring,
        "--num_modes", str(args.num_modes),
        "--seed", str(args.seed),
        "--verbosity", str(args.verbosity),
        "--cpu", str(args.cpu),
        "--max_gpu_memory", str(args.max_gpu_memory),
        "--dir", str(out_dir),
    ]
    if args.search_mode:
        cmd += ["--search_mode", args.search_mode]
    else:
        cmd += ["--exhaustiveness", str(args.exhaustiveness)]
    return cmd


def parse_score(path: Path) -> float | None:
    text = path.read_text(encoding="utf-8", errors="ignore")
    for pattern in SCORE_PATTERNS:
        match = pattern.search(text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
    return None


def collect_batch_results(ligands: list[Path], out_dir: Path, metadata: dict[str, dict[str, str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for ligand in ligands:
        stem = ligand.stem
        meta = metadata.get(stem, {})
        candidates = [out_dir / f"{stem}_out.sdf", out_dir / f"{stem}.sdf", out_dir / f"{stem}_out.pdbqt"]
        output_file = next((path for path in candidates if path.exists()), None)
        if output_file is None:
            rows.append({**meta, "ligand_stem": stem, "success": False, "score": "", "output_file": "", "failure_reason": "missing_output_file"})
            continue
        score = parse_score(output_file)
        rows.append({
            **meta,
            "ligand_stem": stem,
            "success": score is not None,
            "score": score if score is not None else "",
            "output_file": str(output_file),
            "failure_reason": "" if score is not None else "score_not_found",
        })
    return rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    fields = [
        "model", "benchmark_index", "ligand_stem", "success", "score",
        "input_smiles", "source_sdf", "split_sdf", "output_file", "failure_reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_success_sdf(rows: list[dict[str, object]], path: Path) -> int:
    writer = Chem.SDWriter(str(path))
    count = 0
    for row in rows:
        if not row.get("success") or not row.get("output_file"):
            continue
        supplier = Chem.SDMolSupplier(str(row["output_file"]), sanitize=False, removeHs=False)
        wrote = False
        for mol in supplier:
            if mol is None:
                continue
            mol.SetProp("benchmark_model", str(row.get("model", "")))
            mol.SetProp("benchmark_index", str(row.get("benchmark_index", "")))
            mol.SetProp("DockingScore", str(row.get("score", "")))
            mol.SetProp("_Name", str(row.get("ligand_stem", "")))
            writer.write(mol)
            wrote = True
        if wrote:
            count += 1
    writer.close()
    return count


def summarize(rows: list[dict[str, object]], args: argparse.Namespace, started: float, merged_count: int, split_summary: dict[str, str]) -> dict[str, object]:
    scores = [float(row["score"]) for row in rows if row.get("success") and row.get("score") not in ("", None)]
    total = len(rows)
    return {
        "model": args.model,
        "input_sdf": str(args.input_sdf),
        "receptor": str(args.receptor),
        "output_dir": str(args.out_dir),
        "gpu_id": args.gpu_id,
        "batch_size": args.batch_size,
        "search_mode": args.search_mode,
        "scoring": args.scoring,
        "box": {
            "center_x": args.center_x,
            "center_y": args.center_y,
            "center_z": args.center_z,
            "size_x": args.size_x,
            "size_y": args.size_y,
            "size_z": args.size_z,
        },
        "total_ligands": total,
        "success_count": len(scores),
        "success_rate": len(scores) / total if total else 0.0,
        "merged_success_sdf_count": merged_count,
        "score_mean": mean(scores) if scores else None,
        "score_median": median(scores) if scores else None,
        "score_min_best": min(scores) if scores else None,
        "score_max_worst": max(scores) if scores else None,
        "score_top_1pct": sorted(scores)[: max(1, int(0.01 * len(scores)))] if scores else [],
        "seconds": time.time() - started,
        "split_summary": split_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-sdf", type=Path, required=True)
    parser.add_argument("--receptor", type=Path, default=Path("data/8gjl.pdbqt"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--unidock-bin", type=Path, default=Path("/opt/unidock-1.1.3/bin/unidock"))
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--center-x", type=float, default=207.293)
    parser.add_argument("--center-y", type=float, default=212.430)
    parser.add_argument("--center-z", type=float, default=172.771)
    parser.add_argument("--size-x", type=float, default=20.0)
    parser.add_argument("--size-y", type=float, default=20.0)
    parser.add_argument("--size-z", type=float, default=20.0)
    parser.add_argument("--scoring", default="vina")
    parser.add_argument("--search-mode", default="fast")
    parser.add_argument("--exhaustiveness", type=int, default=8)
    parser.add_argument("--num-modes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbosity", type=int, default=1)
    parser.add_argument("--cpu", type=int, default=0)
    parser.add_argument("--max-gpu-memory", type=int, default=0)
    parser.add_argument("--keep-split", action="store_true")
    args = parser.parse_args()

    for label, path in [("input_sdf", args.input_sdf), ("receptor", args.receptor), ("unidock_bin", args.unidock_bin)]:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    started = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    split_dir = args.out_dir / "split_ligands"
    batch_root = args.out_dir / "batch_outputs"
    log_dir = args.out_dir / "logs"
    batch_root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    ligand_paths, metadata = split_sdf(args.input_sdf, split_dir, args.model)
    split_summary = metadata.pop("__split_summary__")
    if not ligand_paths:
        raise RuntimeError(f"No ligands could be split from {args.input_sdf}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    all_rows: list[dict[str, object]] = []
    failed_batches: list[str] = []
    for batch_idx, batch_ligands in enumerate(_chunked(ligand_paths, args.batch_size), start=1):
        out_dir = batch_root / f"batch_{batch_idx:04d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = build_cmd(args, batch_ligands, out_dir)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout, env=env, check=False)
        (log_dir / f"batch_{batch_idx:04d}.stdout.log").write_text(result.stdout, encoding="utf-8")
        (log_dir / f"batch_{batch_idx:04d}.stderr.log").write_text(result.stderr, encoding="utf-8")
        if result.returncode != 0:
            failed_batches.append(f"batch_{batch_idx:04d}: returncode={result.returncode}")
            for ligand in batch_ligands:
                stem = ligand.stem
                all_rows.append({**metadata.get(stem, {}), "ligand_stem": stem, "success": False, "score": "", "output_file": "", "failure_reason": f"batch_returncode_{result.returncode}"})
            continue
        all_rows.extend(collect_batch_results(batch_ligands, out_dir, metadata))

    results_csv = args.out_dir / "docking_results.csv"
    docked_sdf = args.out_dir / "successful_docked_outputs.sdf"
    write_csv(all_rows, results_csv)
    merged_count = write_success_sdf(all_rows, docked_sdf)
    summary = summarize(all_rows, args, started, merged_count, split_summary)
    summary["failed_batches"] = failed_batches
    (args.out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
