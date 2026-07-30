"""Score the formal five-model benchmark with DePPA-style QVina score-only."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

from openbabel import pybel
from rdkit import Chem, RDLogger


LOGGER = logging.getLogger("qvina_score_only")
AFFINITY_PATTERN = re.compile(r"Affinity:\s*([-+]?\d*\.?\d+)")
MODEL_ORDER = ["DiffGui", "DiffSBDD", "PMDM", "GCDM-SBDD", "TargetDiff"]
REPEAT_ORDER = ["rep1", "rep2", "rep3"]


@dataclass(frozen=True)
class SourceSpec:
    """One repeat/model SDF input and its attempt denominator."""

    repeat: str
    model: str
    sdf_path: Path
    generation_attempts: int
    is_reference: bool = False


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def build_source_specs(benchmark_root: Path, project_root: Path) -> list[SourceSpec]:
    """Return the exact SDFs used by the formal three-repeat benchmark."""
    rep1_diffgui = "8gjl_c1101_benchmark_20260701_diffgui_rep1_1000steps_seed1_7"
    rep1_other = "8gjl_c1101_benchmark_20260630_all_1000steps"
    rep2 = "8gjl_c1101_benchmark_20260701_all_rep2_1000steps_seed8_14"
    rep3 = "8gjl_c1101_benchmark_20260701_all_rep3_1000steps_seed15_21"
    filename = {
        "DiffGui": "diffgui.sanitize_valid.sdf",
        "DiffSBDD": "diffsbdd.sanitize_valid.sdf",
        "PMDM": "pmdm.sanitize_valid.sdf",
        "GCDM-SBDD": "gcdm.sanitize_valid.sdf",
        "TargetDiff": "targetdiff.sanitize_valid.sdf",
    }
    folders = {
        "rep1": {
            model: rep1_diffgui if model == "DiffGui" else rep1_other
            for model in MODEL_ORDER
        },
        "rep2": {model: rep2 for model in MODEL_ORDER},
        "rep3": {model: rep3 for model in MODEL_ORDER},
    }
    specs = []
    for repeat in REPEAT_ORDER:
        for model in MODEL_ORDER:
            specs.append(
                SourceSpec(
                    repeat=repeat,
                    model=model,
                    sdf_path=(
                        benchmark_root
                        / folders[repeat][model]
                        / "filtered_sanitize_valid"
                        / filename[model]
                    ),
                    generation_attempts=1000,
                )
            )
    reference_sdf = (
        project_root
        / "analysis"
        / "benchmark"
        / "8gjl_c1101_benchmark_20260624_1000mol_500steps"
        / "inputs"
        / "8GJL_C1101_ligand.sdf"
    )
    specs.append(
        SourceSpec(
            repeat="reference",
            model="C1101",
            sdf_path=reference_sdf,
            generation_attempts=1,
            is_reference=True,
        )
    )
    return specs


def _safe_smiles(mol: Chem.Mol) -> str:
    try:
        copy = Chem.Mol(mol)
        Chem.SanitizeMol(copy)
        return Chem.MolToSmiles(Chem.RemoveHs(copy), canonical=True, isomericSmiles=True)
    except Exception:
        return ""


def _extract_largest_fragment(mol: Chem.Mol) -> tuple[Chem.Mol, bool]:
    """Reduce a record to its largest connected fragment before docking scoring.

    DiffSBDD-family generators keep all fragments, so records may be
    disconnected. Docking a multi-fragment ligand as one entity inflates
    apparent affinity, so we keep only the largest connected component while
    preserving its 3D coordinates. Returns (molecule, fragment_reduced_flag).
    """
    try:
        fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    except Exception:
        return mol, False
    if len(fragments) <= 1:
        return mol, False
    largest = max(fragments, key=lambda m: m.GetNumAtoms())
    return largest, True


def prepare_ligands(
    specs: list[SourceSpec],
    prepared_root: Path,
    limit_per_task: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert every input conformer to a dedicated PDBQT file."""
    score_tasks: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    for spec in specs:
        if not spec.sdf_path.is_file():
            raise FileNotFoundError(f"Input SDF not found: {spec.sdf_path}")
        task_dir = prepared_root / spec.repeat / _slug(spec.model)
        task_dir.mkdir(parents=True, exist_ok=True)
        supplier = Chem.SDMolSupplier(str(spec.sdf_path), sanitize=True, removeHs=False)
        seen = 0
        prepared = 0
        fragment_reduced_count = 0
        for index, mol in enumerate(supplier):
            if limit_per_task is not None and seen >= limit_per_task:
                break
            seen += 1
            fragment_reduced = False
            if mol is not None and mol.GetNumConformers() > 0:
                mol, fragment_reduced = _extract_largest_fragment(mol)
            if fragment_reduced:
                fragment_reduced_count += 1
            task_id = f"{spec.repeat}_{_slug(spec.model)}_{index:06d}"
            base = {
                "task_id": task_id,
                "repeat": spec.repeat,
                "model": spec.model,
                "is_reference": spec.is_reference,
                "source_sdf": str(spec.sdf_path),
                "source_index": index,
                "generation_attempts": spec.generation_attempts,
                "fragment_reduced": fragment_reduced,
            }
            if mol is None or mol.GetNumConformers() == 0:
                failed_rows.append(
                    {
                        **base,
                        "smiles": "",
                        "pdbqt_path": "",
                        "success": False,
                        "affinity_kcal_mol": "",
                        "deppa_vina_reward": "",
                        "seconds": 0.0,
                        "failure_reason": "rdkit_read_or_conformer_failed",
                    }
                )
                continue
            pdbqt_path = task_dir / f"{task_id}.pdbqt"
            try:
                molblock = Chem.MolToMolBlock(mol, kekulize=False)
                obmol = pybel.readstring("sdf", molblock)
                obmol.write("pdbqt", str(pdbqt_path), overwrite=True)
                if not pdbqt_path.is_file() or pdbqt_path.stat().st_size == 0:
                    raise RuntimeError("OpenBabel produced an empty PDBQT")
            except Exception as exc:
                failed_rows.append(
                    {
                        **base,
                        "smiles": _safe_smiles(mol),
                        "pdbqt_path": str(pdbqt_path),
                        "success": False,
                        "affinity_kcal_mol": "",
                        "deppa_vina_reward": "",
                        "seconds": 0.0,
                        "failure_reason": f"pdbqt_conversion_failed:{type(exc).__name__}",
                    }
                )
                continue
            prepared += 1
            score_tasks.append(
                {
                    **base,
                    "smiles": _safe_smiles(mol),
                    "pdbqt_path": str(pdbqt_path),
                }
            )
        LOGGER.info(
            "prepared repeat=%s model=%s seen=%d prepared=%d failed=%d fragment_reduced=%d",
            spec.repeat,
            spec.model,
            seen,
            prepared,
            seen - prepared,
            fragment_reduced_count,
        )
    return score_tasks, failed_rows


def score_one(
    task: dict[str, Any],
    *,
    qvina_bin: str,
    receptor: str,
    center: tuple[float, float, float],
    size: float,
    exhaustiveness: int,
    timeout: int,
) -> dict[str, Any]:
    """Run one QuickVina2 score-only calculation and parse its affinity."""
    started = time.monotonic()
    pdbqt_text = Path(task["pdbqt_path"]).read_text(encoding="utf-8", errors="ignore")
    root_count = len(re.findall(r"(?m)^ROOT\s*$", pdbqt_text))
    if root_count != 1:
        return {
            **task,
            "success": False,
            "affinity_kcal_mol": "",
            "deppa_vina_reward": "",
            "seconds": time.monotonic() - started,
            "failure_reason": f"multiple_pdbqt_roots:{root_count}",
        }
    cmd = [
        qvina_bin,
        "--receptor",
        receptor,
        "--ligand",
        task["pdbqt_path"],
        "--center_x",
        f"{center[0]:.4f}",
        "--center_y",
        f"{center[1]:.4f}",
        "--center_z",
        f"{center[2]:.4f}",
        "--size_x",
        str(size),
        "--size_y",
        str(size),
        "--size_z",
        str(size),
        "--exhaustiveness",
        str(exhaustiveness),
        "--score_only",
    ]
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return {
            **task,
            "success": False,
            "affinity_kcal_mol": "",
            "deppa_vina_reward": "",
            "seconds": time.monotonic() - started,
            "failure_reason": "qvina_timeout",
        }
    combined_output = f"{result.stdout}\n{result.stderr}"
    match = AFFINITY_PATTERN.search(combined_output)
    if result.returncode != 0 or match is None:
        reason = (
            f"qvina_returncode_{result.returncode}"
            if result.returncode != 0
            else "affinity_not_found"
        )
        return {
            **task,
            "success": False,
            "affinity_kcal_mol": "",
            "deppa_vina_reward": "",
            "seconds": time.monotonic() - started,
            "failure_reason": reason,
        }
    affinity = float(match.group(1))
    return {
        **task,
        "success": True,
        "affinity_kcal_mol": affinity,
        "deppa_vina_reward": -affinity,
        "seconds": time.monotonic() - started,
        "failure_reason": "",
    }


RESULT_FIELDS = [
    "task_id",
    "repeat",
    "model",
    "is_reference",
    "source_sdf",
    "source_index",
    "generation_attempts",
    "fragment_reduced",
    "smiles",
    "pdbqt_path",
    "success",
    "affinity_kcal_mol",
    "deppa_vina_reward",
    "seconds",
    "failure_reason",
]


def write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    """Write deterministic per-ligand result rows."""
    repeat_rank = {name: idx for idx, name in enumerate(REPEAT_ORDER + ["reference"])}
    model_rank = {name: idx for idx, name in enumerate(MODEL_ORDER + ["C1101"])}
    ordered = sorted(
        rows,
        key=lambda row: (
            repeat_rank.get(str(row["repeat"]), 99),
            model_rank.get(str(row["model"]), 99),
            int(row["source_index"]),
        ),
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ordered)


def _best_fraction_mean(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    count = max(1, math.ceil(len(values) * fraction))
    return mean(sorted(values)[:count])


def summarize(
    rows: list[dict[str, Any]], specs: list[SourceSpec], output_dir: Path
) -> None:
    """Write per-repeat and pooled five-model score summaries."""
    per_repeat_rows = []
    for spec in specs:
        task_rows = [
            row
            for row in rows
            if row["repeat"] == spec.repeat and row["model"] == spec.model
        ]
        affinities = [
            float(row["affinity_kcal_mol"])
            for row in task_rows
            if row["success"] and row["affinity_kcal_mol"] != ""
        ]
        per_repeat_rows.append(
            {
                "repeat": spec.repeat,
                "model": spec.model,
                "is_reference": spec.is_reference,
                "generation_attempts": spec.generation_attempts,
                "sanitize_valid_inputs": len(task_rows),
                "score_success_count": len(affinities),
                "score_success_per_input": (
                    len(affinities) / len(task_rows) if task_rows else 0.0
                ),
                "score_success_per_generation_attempt": (
                    len(affinities) / spec.generation_attempts
                    if spec.generation_attempts
                    else 0.0
                ),
                "affinity_mean": mean(affinities) if affinities else None,
                "affinity_median": median(affinities) if affinities else None,
                "affinity_std_population": pstdev(affinities) if len(affinities) > 1 else 0.0,
                "affinity_best_min": min(affinities) if affinities else None,
                "affinity_best_5pct_mean": _best_fraction_mean(affinities, 0.05),
                "rate_affinity_le_minus10_per_scored": (
                    sum(value <= -10.0 for value in affinities) / len(affinities)
                    if affinities
                    else 0.0
                ),
                "deppa_reward_mean": mean([-value for value in affinities]) if affinities else None,
            }
        )
    summary_fields = list(per_repeat_rows[0])
    with (output_dir / "per_repeat_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(per_repeat_rows)

    model_rows = []
    for model in MODEL_ORDER:
        repeat_summaries = [
            row for row in per_repeat_rows if row["model"] == model
        ]
        model_scores = [
            float(row["affinity_kcal_mol"])
            for row in rows
            if row["model"] == model
            and row["success"]
            and row["affinity_kcal_mol"] != ""
        ]
        model_rows.append(
            {
                "model": model,
                "generation_attempts_total": sum(
                    int(row["generation_attempts"]) for row in repeat_summaries
                ),
                "sanitize_valid_inputs_total": sum(
                    int(row["sanitize_valid_inputs"]) for row in repeat_summaries
                ),
                "score_success_total": len(model_scores),
                "score_success_per_input": len(model_scores)
                / sum(int(row["sanitize_valid_inputs"]) for row in repeat_summaries),
                "score_success_per_generation_attempt": len(model_scores) / 3000,
                "affinity_pooled_mean": mean(model_scores),
                "affinity_pooled_median": median(model_scores),
                "affinity_pooled_std_population": pstdev(model_scores),
                "affinity_best_min": min(model_scores),
                "affinity_best_5pct_mean": _best_fraction_mean(model_scores, 0.05),
                "rate_affinity_le_minus10_per_scored": sum(
                    value <= -10.0 for value in model_scores
                )
                / len(model_scores),
                "repeat_affinity_mean_mean": mean(
                    float(row["affinity_mean"]) for row in repeat_summaries
                ),
                "repeat_affinity_mean_std": pstdev(
                    [float(row["affinity_mean"]) for row in repeat_summaries]
                ),
                "deppa_reward_pooled_mean": mean([-value for value in model_scores]),
            }
        )
    with (output_dir / "model_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(model_rows[0]))
        writer.writeheader()
        writer.writerows(model_rows)


def main() -> None:
    """Prepare, score, and summarize all formal benchmark molecules."""
    script_path = Path(__file__).resolve()
    benchmark_root = script_path.parents[2]
    project_root = script_path.parents[5]
    default_receptor = (
        project_root
        / "analysis"
        / "deppa_8gjl_full_replica_20260720"
        / "target"
        / "8GJL.pdbqt"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=script_path.parent / "results")
    parser.add_argument("--receptor", type=Path, default=default_receptor)
    parser.add_argument("--qvina-bin", type=Path, default=Path(shutil.which("qvina2") or "qvina2"))
    parser.add_argument("--workers", type=int, default=18)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--center-x", type=float, default=207.4519)
    parser.add_argument("--center-y", type=float, default=212.5211)
    parser.add_argument("--center-z", type=float, default=172.9314)
    parser.add_argument("--size", type=float, default=20.0)
    parser.add_argument("--exhaustiveness", type=int, default=16)
    parser.add_argument("--limit-per-task", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.output_dir = args.output_dir.resolve()
    args.receptor = args.receptor.resolve()
    args.qvina_bin = args.qvina_bin.resolve()
    if not args.receptor.is_file():
        raise FileNotFoundError(f"Receptor not found: {args.receptor}")
    if not args.qvina_bin.is_file():
        raise FileNotFoundError(f"QuickVina2 not found: {args.qvina_bin}")
    result_path = args.output_dir / "per_ligand_scores.csv"
    if result_path.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite existing results: {result_path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.output_dir / "score_qvina_only.log", encoding="utf-8"),
        ],
    )
    RDLogger.DisableLog("rdApp.*")
    specs = build_source_specs(benchmark_root, project_root)
    started = time.time()
    prepared_root = args.output_dir / "prepared_pdbqt"
    tasks, failed_rows = prepare_ligands(specs, prepared_root, args.limit_per_task)
    LOGGER.info(
        "scoring_start tasks=%d preparation_failures=%d workers=%d",
        len(tasks),
        len(failed_rows),
        args.workers,
    )
    rows = list(failed_rows)
    score_kwargs = {
        "qvina_bin": str(args.qvina_bin),
        "receptor": str(args.receptor),
        "center": (args.center_x, args.center_y, args.center_z),
        "size": args.size,
        "exhaustiveness": args.exhaustiveness,
        "timeout": args.timeout,
    }
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(score_one, task, **score_kwargs): task["task_id"]
            for task in tasks
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if completed % 250 == 0 or completed == len(futures):
                write_rows(rows, args.output_dir / "per_ligand_scores.partial.csv")
                LOGGER.info("progress completed=%d total=%d", completed, len(futures))
    write_rows(rows, result_path)
    summarize(rows, specs, args.output_dir)
    failure_counts: dict[str, int] = {}
    for row in rows:
        if row["success"]:
            continue
        reason = str(row["failure_reason"])
        failure_counts[reason] = failure_counts.get(reason, 0) + 1
    manifest = {
        "started_at_unix": started,
        "finished_at_unix": time.time(),
        "seconds": time.time() - started,
        "qvina_bin": str(args.qvina_bin),
        "qvina_version": subprocess.run(
            [str(args.qvina_bin), "--version"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip(),
        "receptor": str(args.receptor),
        "center": [args.center_x, args.center_y, args.center_z],
        "size": [args.size, args.size, args.size],
        "exhaustiveness": args.exhaustiveness,
        "score_mode": "score_only",
        "workers": args.workers,
        "limit_per_task": args.limit_per_task,
        "total_rows": len(rows),
        "success_count": sum(bool(row["success"]) for row in rows),
        "failure_counts": failure_counts,
        "sources": [{**asdict(spec), "sdf_path": str(spec.sdf_path)} for spec in specs],
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    LOGGER.info(
        "completed rows=%d success=%d failures=%d seconds=%.2f",
        len(rows),
        manifest["success_count"],
        len(rows) - manifest["success_count"],
        manifest["seconds"],
    )


if __name__ == "__main__":
    main()
