#!/usr/bin/env python
"""Offline UniDock evaluation for every DiffSBDD single-reward epoch."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import pandas as pd
from rdkit import Chem, RDLogger


ROOT = Path(__file__).resolve().parents[1]
RUN_GLOB = "diffsbdd-single-reward-20260612-*"
EPOCH_PATTERN = re.compile(r"epoch_(\d+)")
SCORE_PATTERNS = [
    re.compile(r"REMARK\s+VINA RESULT:\s*([-+]?\d+(?:\.\d+)?)"),
    re.compile(r"ENERGY=\s*([-+]?\d+(?:\.\d+)?)"),
    re.compile(r">\s*<[Dd]ocking[_\s][Ss]core>\s*\n\s*([-+]?\d+(?:\.\d+)?)", re.MULTILINE),
    re.compile(r">\s*<[Ss]core>\s*\n\s*([-+]?\d+(?:\.\d+)?)", re.MULTILINE),
]

RDLogger.DisableLog("rdApp.*")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "analysis/diffsbdd_single_reward_matrix_20260613/offline_docking_all_epochs",
    )
    parser.add_argument("--receptor", type=Path, default=ROOT / "data/8gjl.pdbqt")
    parser.add_argument("--unidock-bin", type=Path, default=Path("/opt/unidock-1.1.3/bin/unidock"))
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--center-x", type=float, default=207.293)
    parser.add_argument("--center-y", type=float, default=212.430)
    parser.add_argument("--center-z", type=float, default=172.771)
    parser.add_argument("--size-x", type=float, default=20.0)
    parser.add_argument("--size-y", type=float, default=20.0)
    parser.add_argument("--size-z", type=float, default=20.0)
    parser.add_argument("--search-mode", default="fast")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-split", action="store_true")
    return parser.parse_args()


def _reward_label(run_dir: Path) -> str:
    name = run_dir.name
    match = re.search(r"diffsbdd-single-reward-20260612-\d+-(.+?)_\d{8}_\d{6}$", name)
    return match.group(1) if match else name


def _discover_epoch_sdfs() -> list[tuple[Path, str, int, Path]]:
    found: list[tuple[Path, str, int, Path]] = []
    for run_dir in sorted((ROOT / "OUT").glob(RUN_GLOB)):
        for sdf_path in sorted((run_dir / "epochs").glob("epoch_*/molecules.sdf")):
            match = EPOCH_PATTERN.fullmatch(sdf_path.parent.name)
            if match:
                found.append((run_dir, _reward_label(run_dir), int(match.group(1)), sdf_path))
    return found


def _sanitize_copy(mol: Chem.Mol) -> Chem.Mol | None:
    candidate = Chem.Mol(mol)
    try:
        candidate.UpdatePropertyCache(strict=False)
        if Chem.SanitizeMol(candidate, catchErrors=True) != Chem.SanitizeFlags.SANITIZE_NONE:
            return None
    except Exception:
        return None
    return candidate


def _split_all(
    epoch_sdfs: list[tuple[Path, str, int, Path]],
    split_dir: Path,
) -> tuple[list[Path], dict[str, dict[str, object]]]:
    split_dir.mkdir(parents=True, exist_ok=True)
    ligand_paths: list[Path] = []
    metadata: dict[str, dict[str, object]] = {}
    for run_dir, reward, epoch, sdf_path in epoch_sdfs:
        supplier = Chem.SDMolSupplier(str(sdf_path), sanitize=False, removeHs=False)
        for molecule_index, mol in enumerate(supplier):
            stem = f"{reward}_e{epoch:04d}_m{molecule_index:05d}"
            row: dict[str, object] = {
                "run": run_dir.name,
                "reward": reward,
                "epoch": epoch,
                "molecule_index": molecule_index,
                "source_sdf": str(sdf_path.relative_to(ROOT)),
                "ligand_stem": stem,
                "input_valid": False,
                "heavy_atom_count": 0,
            }
            if mol is None:
                metadata[stem] = row
                continue
            sanitized = _sanitize_copy(mol)
            if sanitized is None:
                metadata[stem] = row
                continue
            row["input_valid"] = True
            row["heavy_atom_count"] = int(sanitized.GetNumHeavyAtoms())
            row["smiles"] = Chem.MolToSmiles(sanitized, canonical=True, isomericSmiles=True)
            path = split_dir / f"{stem}.sdf"
            sanitized.SetProp("_Name", stem)
            writer = Chem.SDWriter(str(path))
            writer.write(sanitized)
            writer.close()
            ligand_paths.append(path)
            metadata[stem] = row
    return ligand_paths, metadata


def _chunks(items: list[Path], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _build_cmd(args: argparse.Namespace, ligands: list[Path], output_dir: Path) -> list[str]:
    return [
        str(args.unidock_bin),
        "--receptor",
        str(args.receptor),
        "--gpu_batch",
        *[str(path) for path in ligands],
        "--center_x",
        str(args.center_x),
        "--center_y",
        str(args.center_y),
        "--center_z",
        str(args.center_z),
        "--size_x",
        str(args.size_x),
        "--size_y",
        str(args.size_y),
        "--size_z",
        str(args.size_z),
        "--scoring",
        "vina",
        "--num_modes",
        "1",
        "--seed",
        str(args.seed),
        "--verbosity",
        "1",
        "--cpu",
        "0",
        "--max_gpu_memory",
        "0",
        "--dir",
        str(output_dir),
        "--search_mode",
        str(args.search_mode),
    ]


def _parse_score(path: Path) -> float | None:
    text = path.read_text(encoding="utf-8", errors="ignore")
    for pattern in SCORE_PATTERNS:
        match = pattern.search(text)
        if match:
            return float(match.group(1))
    return None


def _run_docking(args: argparse.Namespace, ligand_paths: list[Path]) -> dict[str, tuple[bool, float | None, str]]:
    batch_root = args.out_dir / "batch_outputs"
    log_dir = args.out_dir / "logs"
    batch_root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    results: dict[str, tuple[bool, float | None, str]] = {}
    total_batches = (len(ligand_paths) + args.batch_size - 1) // args.batch_size
    for batch_index, ligands in enumerate(_chunks(ligand_paths, args.batch_size), start=1):
        batch_dir = batch_root / f"batch_{batch_index:04d}"
        marker = batch_dir / "complete.json"
        if args.resume and marker.exists():
            existing = json.loads(marker.read_text(encoding="utf-8"))
            for stem, item in existing.items():
                results[stem] = (bool(item["success"]), item["score"], str(item["failure_reason"]))
            continue
        batch_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()
        completed = subprocess.run(
            _build_cmd(args, ligands, batch_dir),
            capture_output=True,
            text=True,
            timeout=args.timeout,
            env=env,
            check=False,
        )
        (log_dir / f"batch_{batch_index:04d}.stdout.log").write_text(completed.stdout, encoding="utf-8")
        (log_dir / f"batch_{batch_index:04d}.stderr.log").write_text(completed.stderr, encoding="utf-8")
        marker_rows: dict[str, dict[str, object]] = {}
        for ligand in ligands:
            candidates = [
                batch_dir / f"{ligand.stem}_out.sdf",
                batch_dir / f"{ligand.stem}.sdf",
                batch_dir / f"{ligand.stem}_out.pdbqt",
            ]
            output = next((path for path in candidates if path.exists()), None)
            score = _parse_score(output) if output else None
            success = completed.returncode == 0 and score is not None
            reason = "" if success else f"returncode_{completed.returncode}" if completed.returncode else "missing_score"
            results[ligand.stem] = (success, score, reason)
            marker_rows[ligand.stem] = {"success": success, "score": score, "failure_reason": reason}
        marker.write_text(json.dumps(marker_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(
            f"[{batch_index}/{total_batches}] ligands={len(ligands)} "
            f"success={sum(1 for item in marker_rows.values() if item['success'])} "
            f"seconds={time.time() - started:.1f}",
            flush=True,
        )
    return results


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(rows: list[dict[str, object]]) -> pd.DataFrame:
    grouped: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["run"]), str(row["reward"]), int(row["epoch"]))].append(row)
    summary: list[dict[str, object]] = []
    for (run, reward, epoch), items in sorted(grouped.items()):
        valid = [row for row in items if row.get("input_valid")]
        success = [row for row in items if row.get("docking_success")]
        scores = [float(row["docking_score"]) for row in success]
        le_values = [float(row["ligand_efficiency"]) for row in success]
        summary.append(
            {
                "run": run,
                "reward": reward,
                "epoch": epoch,
                "total_records": len(items),
                "input_valid_count": len(valid),
                "docking_success_count": len(success),
                "docking_success_rate_all": len(success) / len(items) if items else 0.0,
                "docking_success_rate_valid": len(success) / len(valid) if valid else 0.0,
                "avg_docking": mean(scores) if scores else None,
                "best_docking": min(scores) if scores else None,
                "avg_ligand_efficiency": mean(le_values) if le_values else None,
            }
        )
    return pd.DataFrame(summary)


def _plot(summary: pd.DataFrame, out_dir: Path) -> None:
    plt.rcParams.update({"figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.25})
    metrics = [
        ("avg_docking", "Mean docking score (lower is better)"),
        ("avg_ligand_efficiency", "Mean ligand efficiency (higher is better)"),
        ("docking_success_rate_all", "Docking success rate / all records"),
    ]
    figure, axes = plt.subplots(len(metrics), 1, figsize=(12, 12), sharex=True)
    for axis, (column, title) in zip(axes, metrics):
        for reward, frame in summary.groupby("reward", sort=False):
            axis.plot(frame["epoch"], frame[column], marker="o", linewidth=1.5, label=reward)
        axis.set_ylabel(column)
        axis.set_title(title)
    axes[-1].set_xlabel("Epoch")
    axes[0].legend(ncol=3, fontsize=8)
    figure.tight_layout()
    figure.savefig(out_dir / "offline_docking_trends.png")
    figure.savefig(out_dir / "offline_docking_trends.svg")
    plt.close(figure)

    final = summary.sort_values("epoch").groupby("reward", as_index=False).tail(1)
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, column, title in [
        (axes[0], "avg_docking", "Final mean docking"),
        (axes[1], "avg_ligand_efficiency", "Final mean LE"),
        (axes[2], "docking_success_rate_all", "Final docking success"),
    ]:
        axis.bar(final["reward"], pd.to_numeric(final[column], errors="coerce").fillna(0.0))
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=45)
    figure.tight_layout()
    figure.savefig(out_dir / "offline_docking_final_comparison.png")
    figure.savefig(out_dir / "offline_docking_final_comparison.svg")
    plt.close(figure)


def main() -> None:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.receptor.exists() or not args.unidock_bin.exists():
        raise FileNotFoundError(f"Missing receptor or UniDock binary: {args.receptor}, {args.unidock_bin}")
    epoch_sdfs = _discover_epoch_sdfs()
    if not epoch_sdfs:
        raise RuntimeError("No DiffSBDD single-reward epoch SDF files found")
    split_dir = args.out_dir / "split_ligands"
    ligand_paths, metadata = _split_all(epoch_sdfs, split_dir)
    results = _run_docking(args, ligand_paths)
    rows: list[dict[str, object]] = []
    for stem, row in metadata.items():
        success, score, reason = results.get(stem, (False, None, "not_submitted"))
        heavy = int(row.get("heavy_atom_count", 0))
        row["docking_success"] = success
        row["docking_score"] = score
        row["ligand_efficiency"] = (-float(score) / heavy) if success and score is not None and heavy > 0 else None
        row["failure_reason"] = reason
        rows.append(row)
    _write_csv(rows, args.out_dir / "offline_docking_per_molecule.csv")
    summary = _aggregate(rows)
    summary.to_csv(args.out_dir / "offline_docking_by_epoch.csv", index=False)
    final = summary.sort_values("epoch").groupby("reward", as_index=False).tail(1)
    final.to_csv(args.out_dir / "offline_docking_final_epoch.csv", index=False)
    _plot(summary, args.out_dir)
    run_summary = {
        "epoch_sdf_count": len(epoch_sdfs),
        "total_records": len(rows),
        "input_valid_count": sum(bool(row.get("input_valid")) for row in rows),
        "docking_success_count": sum(bool(row.get("docking_success")) for row in rows),
        "receptor": str(args.receptor),
        "box_center": [args.center_x, args.center_y, args.center_z],
        "box_size": [args.size_x, args.size_y, args.size_z],
        "search_mode": args.search_mode,
        "gpu_id": args.gpu_id,
    }
    (args.out_dir / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not args.keep_split:
        shutil.rmtree(split_dir, ignore_errors=True)
    print(json.dumps(run_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
