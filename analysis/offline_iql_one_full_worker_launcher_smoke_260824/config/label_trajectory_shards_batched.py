#!/usr/bin/env python3
"""Label DiffSBDD trajectory shards with one independent UniDock submission.

The existing offline-labeling contract writes one atomic sidecar per collector
shard.  This runner changes only the docking scheduling: all pending collector
shards assigned to one worker are prepared together, submitted to UniDock
through one ``--ligand_index`` file, and then committed back to their original
shard sidecars.  UniDock performs its own GPU-memory-aware internal splitting;
there is deliberately no application-level 4096-ligand batch limit here.
The UniDock runtime arguments intentionally follow the established five-model
8GJL/C1101 benchmark.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml
from rdkit import Chem, RDLogger

from ddpo.offline_rl.index import load_episode_index
from ddpo.offline_rl.labeling import (
    DockingLabelFacts,
    _read_reconstruction_rows,
    _read_sdf_molecules,
    reconstruction_facts_from_row,
)
from ddpo.offline_rl.labeling_run import (
    _task_dir,
    _task_inputs,
    _validate_completed_status,
    build_shard_label_tasks,
    label_one_task,
    load_labeling_run_config,
    select_worker_tasks,
)
from ddpo.rewards.interfaces import DockingResult
from ddpo.utils.molecule_semantics import (
    FAILURE_DOCKING_BACKEND,
    FAILURE_DOCKING_INPUT,
    prepare_mol_for_docking,
    write_rdmol_to_sdf,
)


LOGGER = logging.getLogger("OfflineLabelingBatched")
RDLogger.DisableLog("rdApp.*")

SCORE_PATTERNS = (
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
)


@dataclass(frozen=True)
class BenchmarkDockingConfig:
    """Fixed UniDock runtime controls inherited from the five-model benchmark."""

    submission_mode: str = "one_full_worker"
    scoring: str = "vina"
    search_mode: str = "fast"
    seed: int = 42
    verbosity: int = 1
    cpu: int = 0
    max_gpu_memory: int = 0
    timeout_seconds: int = 3600
    retain_docking_outputs: bool = True
    max_retries: int = 2

    @classmethod
    def from_yaml(cls, config_path: Path) -> "BenchmarkDockingConfig":
        """Load and validate the benchmark-compatible section of one YAML file."""
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        section = (
            raw.get("offline_labeling", {}).get("benchmark_docking", {})
            if isinstance(raw, Mapping)
            else {}
        )
        if not isinstance(section, Mapping):
            raise ValueError("offline_labeling.benchmark_docking must be a mapping")
        values = dict(section)
        legacy_batch_size = values.pop("batch_size", None)
        if legacy_batch_size not in (None, 0):
            raise ValueError(
                "application-level batch_size is disabled; submit one full worker "
                "chunk and let UniDock split internally"
            )
        values.pop("external_batch_size", None)
        result = cls(**values)
        if result.submission_mode != "one_full_worker":
            raise ValueError(
                "the full-label runner requires submission_mode=one_full_worker"
            )
        if result.scoring != "vina":
            raise ValueError("the locked 8GJL benchmark scoring function is vina")
        if result.search_mode != "fast":
            raise ValueError("the locked 8GJL benchmark search mode is fast")
        if result.timeout_seconds <= 0 or result.max_retries < 0:
            raise ValueError("invalid benchmark docking timeout/retry controls")
        return result


@dataclass(frozen=True)
class MoleculePreparationConfig:
    """Canonical molecule preparation controls used by the offline labeler."""

    add_hydrogens: bool = True
    assign_gasteiger: bool = True

    @classmethod
    def from_yaml(cls, config_path: Path) -> "MoleculePreparationConfig":
        """Load the explicit preparation semantics from the run YAML."""
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        section = (
            raw.get("offline_labeling", {}).get("molecule_preparation", {})
            if isinstance(raw, Mapping)
            else {}
        )
        if not isinstance(section, Mapping):
            raise ValueError(
                "offline_labeling.molecule_preparation must be a mapping"
            )
        return cls(**dict(section))


@dataclass(frozen=True)
class PreparedMolecule:
    """One sample and its prepared UniDock input, or an input-preparation failure."""

    sample_id: str
    prepared: Chem.Mol | None
    failure_reason: str | None = None


@dataclass(frozen=True)
class PreparedTask:
    """One complete collector shard plus all its docking candidates."""

    task: Any
    molecules: tuple[PreparedMolecule, ...]


class CachedDockingLabeler:
    """Return previously computed docking facts to the native shard labeler."""

    def __init__(self, facts: Mapping[str, DockingLabelFacts]) -> None:
        """Store the exact sample-to-docking mapping for one batch."""
        self._facts = dict(facts)

    def __call__(
        self,
        molecules: Sequence[tuple[str, Chem.Mol]],
    ) -> Mapping[str, DockingLabelFacts]:
        """Return facts with exact coverage of the native labeler's valid inputs."""
        requested = {sample_id for sample_id, _ in molecules}
        missing = requested - set(self._facts)
        if missing:
            raise ValueError(
                "batched docking cache is missing sample IDs: "
                f"{sorted(missing)[:5]}"
            )
        return {sample_id: self._facts[sample_id] for sample_id in requested}


def _parse_args() -> argparse.Namespace:
    """Parse one independent worker's controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument(
        "--max-shards",
        type=int,
        default=None,
        help="diagnostic limit; formal full runs must leave this unset",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def _configure_logging(output_dir: Path, worker_id: int) -> None:
    """Configure a durable worker log and rank-readable stream output."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    chemistry_dir = log_dir / f"worker_{worker_id:02d}_chemistry"
    chemistry_dir.mkdir(parents=True, exist_ok=True)
    os.environ["DIFFSBDD_LOG_DIR"] = str(chemistry_dir)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / "runtime_cache" / "matplotlib"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        f"%(asctime)s %(levelname)s worker={worker_id} %(name)s: %(message)s"
    )
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    file_handler = logging.FileHandler(
        log_dir / f"worker_{worker_id:02d}_batched.log",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(stream_handler)


def _sha256_lines(lines: Sequence[str]) -> str:
    """Hash an ordered sample manifest without loading ligand contents."""
    digest = hashlib.sha256()
    for line in lines:
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _parse_score(path: Path) -> float | None:
    """Extract the first UniDock/Vina score using the benchmark parser rules."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    for pattern in SCORE_PATTERNS:
        match = pattern.search(text)
        if match:
            return float(match.group(1))
    return None


def _output_path(output_dir: Path, stem: str) -> Path | None:
    """Find the standard UniDock output pose for one input stem."""
    candidates = (
        output_dir / f"{stem}_out.sdf",
        output_dir / f"{stem}.sdf",
        output_dir / f"{stem}_out.pdbqt",
    )
    return next((path for path in candidates if path.is_file()), None)


def _failed_docking(reason: str) -> DockingLabelFacts:
    """Build a finite failed docking fact."""
    return DockingLabelFacts(
        input_success=True,
        result=DockingResult.failed("unidock", reason),
    )


def _prepare_task(task: Any, *, add_hydrogens: bool, assign_gasteiger: bool) -> PreparedTask:
    """Prepare all collector-valid molecules in one shard without docking them."""
    reconstruction = _read_reconstruction_rows(Path(task.reconstruction_csv))
    molecules = _read_sdf_molecules(Path(task.molecules_sdf))
    requested_ids = {row.sample_id for row in task.rows}
    if requested_ids - set(reconstruction) or set(reconstruction) - requested_ids:
        raise ValueError(f"reconstruction coverage mismatch for {task.key}")
    expected_sdf_ids = {
        sample_id
        for sample_id, reconstruction_row in reconstruction.items()
        if reconstruction_facts_from_row(reconstruction_row).rdkit_valid
    }
    if set(molecules) != expected_sdf_ids:
        raise ValueError(f"molecule SDF coverage mismatch for {task.key}")

    prepared: list[PreparedMolecule] = []
    for row in task.rows:
        reconstruction_facts = reconstruction_facts_from_row(reconstruction[row.sample_id])
        if not reconstruction_facts.rdkit_valid:
            continue
        molecule = molecules[row.sample_id]
        candidate = prepare_mol_for_docking(
            molecule,
            add_hydrogens=add_hydrogens,
            assign_gasteiger=assign_gasteiger,
        )
        if candidate is None:
            prepared.append(
                PreparedMolecule(
                    sample_id=row.sample_id,
                    prepared=None,
                    failure_reason=FAILURE_DOCKING_INPUT,
                )
            )
        else:
            prepared.append(PreparedMolecule(row.sample_id, candidate))
    return PreparedTask(task=task, molecules=tuple(prepared))


def _write_batch_inputs(
    prepared: Sequence[PreparedMolecule],
    input_dir: Path,
) -> tuple[list[PreparedMolecule], dict[str, Path], Path]:
    """Write independent SDF inputs and one full-worker ligand index."""
    input_dir.mkdir(parents=True, exist_ok=True)
    ligand_index = input_dir / "ligands.txt"
    submitted: list[PreparedMolecule] = []
    paths: dict[str, Path] = {}
    seen_sample_ids: set[str] = set()
    with ligand_index.open("w", encoding="utf-8") as index_handle:
        for ordinal, item in enumerate(prepared):
            if item.prepared is None:
                continue
            if item.sample_id in seen_sample_ids:
                raise ValueError(f"duplicate sample_id in full-worker submission: {item.sample_id}")
            seen_sample_ids.add(item.sample_id)
            stem = f"ligand_{ordinal:08d}"
            path = input_dir / f"{stem}.sdf"
            molecule = Chem.Mol(item.prepared)
            molecule.SetProp("sample_id", item.sample_id)
            molecule.SetProp("_Name", stem)
            if not write_rdmol_to_sdf(molecule, path):
                raise RuntimeError(f"failed to write UniDock input: {path}")
            index_handle.write(f"{path}\n")
            submitted.append(item)
            paths[item.sample_id] = path
    return submitted, paths, ligand_index


def _run_unidock(
    *,
    ligand_index: Path,
    output_dir: Path,
    run_config: Any,
    benchmark: BenchmarkDockingConfig,
    gpu_id: int,
    batch_dir: Path,
) -> tuple[int, float, list[str]]:
    """Run one benchmark-compatible UniDock submission with bounded retries."""
    docking = run_config.docking
    if docking is None:
        raise ValueError("docking must be enabled for batched affinity labeling")
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(docking.unidock_bin),
        "--receptor",
        str(docking.receptor_pdbqt),
        "--ligand_index",
        str(ligand_index),
        "--center_x",
        str(docking.center_x),
        "--center_y",
        str(docking.center_y),
        "--center_z",
        str(docking.center_z),
        "--size_x",
        str(docking.size_x),
        "--size_y",
        str(docking.size_y),
        "--size_z",
        str(docking.size_z),
        "--scoring",
        benchmark.scoring,
        "--num_modes",
        str(docking.num_modes),
        "--seed",
        str(benchmark.seed),
        "--verbosity",
        str(benchmark.verbosity),
        "--cpu",
        str(benchmark.cpu),
        "--max_gpu_memory",
        str(benchmark.max_gpu_memory),
        "--dir",
        str(output_dir),
        "--search_mode",
        benchmark.search_mode,
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    command_json = batch_dir / "unidock_command.json"
    command_json.write_text(
        json.dumps(
            {
                "command": command,
                "cuda_visible_devices": str(gpu_id),
                "benchmark_parameters": benchmark.__dict__,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    started = time.perf_counter()
    return_code = -1
    for attempt in range(1, benchmark.max_retries + 2):
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = batch_dir / f"unidock_attempt_{attempt:02d}.stdout.log"
        stderr_path = batch_dir / f"unidock_attempt_{attempt:02d}.stderr.log"
        LOGGER.info(
            "unidock start attempt=%d/%d submitted_index=%s gpu=%d",
            attempt,
            benchmark.max_retries + 1,
            ligand_index,
            gpu_id,
        )
        with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr_handle:
            try:
                result = subprocess.run(
                    command,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    text=True,
                    timeout=benchmark.timeout_seconds,
                    env=environment,
                    check=False,
                )
                return_code = result.returncode
            except subprocess.TimeoutExpired:
                return_code = -9
        if return_code == 0:
            elapsed = time.perf_counter() - started
            LOGGER.info("unidock completed seconds=%.1f", elapsed)
            return return_code, elapsed, command
        LOGGER.warning(
            "unidock failed attempt=%d return_code=%d; retrying=%s",
            attempt,
            return_code,
            attempt <= benchmark.max_retries,
        )
    raise RuntimeError(
        f"UniDock failed after {benchmark.max_retries + 1} attempts "
        f"return_code={return_code}; see {batch_dir}"
    )


def _write_scores(
    submitted: Sequence[PreparedMolecule],
    input_paths: Mapping[str, Path],
    output_dir: Path,
    scores_path: Path,
) -> dict[str, DockingLabelFacts]:
    """Parse one completed UniDock output directory into sample-level facts."""
    scores: dict[str, DockingLabelFacts] = {}
    rows: list[dict[str, str]] = []
    successful = 0
    for item in submitted:
        input_path = input_paths[item.sample_id]
        output_path = _output_path(output_dir, input_path.stem)
        score = _parse_score(output_path) if output_path is not None else None
        # Match the existing UniDock backend: a zero score is treated as a
        # backend failure, while all nonzero finite scores are accepted.
        if score is not None and math.isfinite(score) and score != 0.0:
            facts = DockingLabelFacts(
                input_success=True,
                result=DockingResult.succeeded("unidock", score),
            )
            successful += 1
            failure_reason = ""
        else:
            facts = _failed_docking(FAILURE_DOCKING_BACKEND)
            failure_reason = FAILURE_DOCKING_BACKEND
        scores[item.sample_id] = facts
        rows.append(
            {
                "sample_id": item.sample_id,
                "success": str(int(facts.result.success)),
                "score_raw": "" if score is None else str(score),
                "failure_reason": failure_reason,
                "input_file": str(input_path),
                "output_file": "" if output_path is None else str(output_path),
            }
        )
    if submitted and successful == 0:
        raise RuntimeError("UniDock returned no parseable successful scores")
    with scores_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["sample_id"])
        writer.writeheader()
        writer.writerows(rows)
    return scores


def _load_scores(scores_path: Path) -> dict[str, DockingLabelFacts]:
    """Reload a previously completed batch score table for resume."""
    result: dict[str, DockingLabelFacts] = {}
    with scores_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            success = row.get("success") == "1"
            score_text = row.get("score_raw", "").strip()
            if success and score_text:
                result[row["sample_id"]] = DockingLabelFacts(
                    input_success=True,
                    result=DockingResult.succeeded("unidock", float(score_text)),
                )
            else:
                result[row["sample_id"]] = _failed_docking(
                    row.get("failure_reason") or FAILURE_DOCKING_BACKEND
                )
    return result


def _completed_task(
    task: Any,
    *,
    run_config: Any,
    episode_index: Any,
) -> bool:
    """Validate and identify an exact completed shard on resume."""
    status_path = _task_dir(run_config.output_dir, task) / "status.json"
    if not status_path.is_file():
        return False
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "completed":
        return False
    sidecar_path = status_path.parent / "reward_sidecar.csv"
    _validate_completed_status(
        status,
        expected=_task_inputs(task, run_config=run_config, episode_index=episode_index),
        sidecar_path=sidecar_path,
        task=task,
    )
    return True


def _submission_id_for_manifest(worker_dir: Path, manifest_sha256: str) -> int:
    """Reuse an interrupted submission with the same manifest when possible."""
    worker_dir.mkdir(parents=True, exist_ok=True)
    existing_ids: list[int] = []
    for path in worker_dir.glob("batch_*"):
        try:
            submission_id = int(path.name.split("_")[-1])
        except ValueError:
            continue
        existing_ids.append(submission_id)
        manifest_path = path / "manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("manifest_sha256") == manifest_sha256:
            return submission_id
    return max(existing_ids, default=-1) + 1


def _process_batch(
    prepared_tasks: Sequence[PreparedTask],
    *,
    run_config: Any,
    episode_index: Any,
    benchmark: BenchmarkDockingConfig,
    worker_id: int,
    gpu_id: int,
) -> None:
    """Submit one complete worker chunk and atomically commit its shards."""
    worker_dir = run_config.output_dir / "batches" / f"worker_{worker_id:02d}"
    all_prepared = [item for task in prepared_tasks for item in task.molecules]
    manifest_lines = [
        f"{task.task.rank}:{task.task.shard_index}:{item.sample_id}"
        for task in prepared_tasks
        for item in task.molecules
    ]
    manifest_hash = _sha256_lines(manifest_lines)
    batch_id = _submission_id_for_manifest(worker_dir, manifest_hash)
    batch_dir = worker_dir / f"batch_{batch_id:04d}"
    input_dir = batch_dir / "inputs"
    output_dir = batch_dir / "poses"
    batch_dir.mkdir(parents=True, exist_ok=True)
    (batch_dir / "manifest.json").write_text(
        json.dumps(
            {
                "worker_id": worker_id,
                "gpu_id": gpu_id,
                "batch_id": batch_id,
                "task_count": len(prepared_tasks),
                "attempt_count": sum(len(task.task.rows) for task in prepared_tasks),
                "candidate_count": len(all_prepared),
                "manifest_sha256": manifest_hash,
                "submission_mode": "one_full_worker",
                "benchmark_docking": benchmark.__dict__,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    cached_facts: dict[str, DockingLabelFacts] = {}
    for item in all_prepared:
        if item.prepared is None:
            cached_facts[item.sample_id] = DockingLabelFacts(
                input_success=False,
                result=DockingResult.failed(
                    "unidock",
                    item.failure_reason or FAILURE_DOCKING_INPUT,
                ),
            )

    submitted, input_paths, ligand_index = _write_batch_inputs(all_prepared, input_dir)
    scores_path = batch_dir / "scores.csv"
    complete_path = batch_dir / "docking_complete.json"
    if complete_path.is_file() and scores_path.is_file():
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if complete.get("manifest_sha256") != manifest_hash:
            raise ValueError(f"batch manifest mismatch on resume: {batch_dir}")
        cached_facts.update(_load_scores(scores_path))
        LOGGER.info(
            "reused completed full-worker submission=%d candidates=%d",
            batch_id,
            len(submitted),
        )
    else:
        if submitted:
            _, elapsed, _ = _run_unidock(
                ligand_index=ligand_index,
                output_dir=output_dir,
                run_config=run_config,
                benchmark=benchmark,
                gpu_id=gpu_id,
                batch_dir=batch_dir,
            )
            parsed = _write_scores(submitted, input_paths, output_dir, scores_path)
            cached_facts.update(parsed)
            success_count = sum(fact.result.success for fact in parsed.values())
        else:
            elapsed = 0.0
            success_count = 0
            scores_path.write_text("sample_id\n", encoding="utf-8")
        complete_path.write_text(
            json.dumps(
                {
                    "status": "completed",
                    "manifest_sha256": manifest_hash,
                    "candidate_count": len(submitted),
                    "success_count": success_count,
                    "seconds": elapsed,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    for prepared_task in prepared_tasks:
        labeler = CachedDockingLabeler(cached_facts)
        state, sidecar_path, counts = label_one_task(
            prepared_task.task,
            run_config=run_config,
            episode_index=episode_index,
            resume=True,
            docking_labeler=labeler,
        )
        LOGGER.info(
            "committed submission=%d shard=(%d,%d) state=%s attempts=%d rdkit_valid=%d "
            "docking_success=%d sidecar=%s",
            batch_id,
            prepared_task.task.rank,
            prepared_task.task.shard_index,
            state,
            counts["attempts"],
            counts["rdkit_valid"],
            counts["docking_success"],
            sidecar_path,
        )
    (batch_dir / "labels_complete.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "manifest_sha256": manifest_hash,
                "shards": [
                    {"rank": task.task.rank, "shard_index": task.task.shard_index}
                    for task in prepared_tasks
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Run one worker's stable shard assignment to completion."""
    args = _parse_args()
    if args.worker_id < 0 or args.worker_id >= args.num_workers:
        raise ValueError("worker-id must lie in [0,num-workers)")
    output_dir = args.output_dir.resolve()
    _configure_logging(output_dir, args.worker_id)
    benchmark = BenchmarkDockingConfig.from_yaml(args.config.resolve())
    preparation = MoleculePreparationConfig.from_yaml(args.config.resolve())
    run_config = load_labeling_run_config(
        args.config,
        data_root=args.data_root,
        worker_id=args.worker_id,
        gpu_id_override=args.gpu_id,
    )
    run_config = replace(run_config, output_dir=output_dir)
    LOGGER.info(
        "start config=%s output=%s worker=%d/%d gpu=%d submission_mode=%s "
        "cuda_visible_devices=%s benchmark=%s",
        args.config.resolve(),
        output_dir,
        args.worker_id,
        args.num_workers,
        args.gpu_id,
        benchmark.submission_mode,
        os.environ.get("CUDA_VISIBLE_DEVICES", "all-visible"),
        benchmark.__dict__,
    )
    episode_index = load_episode_index(run_config.episode_index_path)
    all_tasks = build_shard_label_tasks(
        episode_index,
        source_run_dir=run_config.source_run_dir,
    )
    tasks = select_worker_tasks(
        all_tasks,
        worker_id=args.worker_id,
        num_workers=args.num_workers,
        max_shards=args.max_shards,
    )
    LOGGER.info(
        "assignment selected_shards=%d all_shards=%d expected_attempts=%d "
        "max_shards=%s",
        len(tasks),
        len(all_tasks),
        sum(len(task.rows) for task in tasks),
        args.max_shards,
    )
    if args.preflight_only:
        if run_config.docking is None:
            raise ValueError("docking is disabled")
        LOGGER.info("preflight completed successfully")
        return

    pending = [
        task
        for task in tasks
        if not (
            args.resume
            and _completed_task(task, run_config=run_config, episode_index=episode_index)
        )
    ]
    LOGGER.info("resume completed=%d pending=%d", len(tasks) - len(pending), len(pending))
    prepared_tasks: list[PreparedTask] = []
    total_candidates = 0
    for task in pending:
        prepared_task = _prepare_task(
            task,
            add_hydrogens=preparation.add_hydrogens,
            assign_gasteiger=preparation.assign_gasteiger,
        )
        prepared_tasks.append(prepared_task)
        total_candidates += len(prepared_task.molecules)
    if prepared_tasks:
        _process_batch(
            prepared_tasks,
            run_config=run_config,
            episode_index=episode_index,
            benchmark=benchmark,
            worker_id=args.worker_id,
            gpu_id=args.gpu_id,
        )
    LOGGER.info(
        "completed worker pending_shards=%d full_worker_candidates=%d "
        "external_submissions=%d",
        len(pending),
        total_candidates,
        int(bool(prepared_tasks)),
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        LOGGER.exception("batched labeling failed")
        sys.exit(1)
