"""Recoverable shard orchestration for offline trajectory molecule labels."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence

import yaml

from .index import (
    EpisodeIndex,
    EpisodeIndexRow,
    build_split_manifest,
    load_episode_index,
    write_split_manifest,
)
from .labeling import (
    LabelPolicyConfig,
    label_trajectory_shard,
    merge_reward_sidecar_shards,
)
from .labeling_backends import (
    PoseBustersBackendConfig,
    PoseBustersLabeler,
    UniDockLabeler,
    UniDockLabelingConfig,
)
from .reward_sidecar import RewardSidecar, load_reward_sidecar


LABEL_SHARD_STATUS_FORMAT = "diffsbdd_offline_label_shard_status_v1"
LABEL_RUN_SUMMARY_FORMAT = "diffsbdd_offline_label_run_summary_v1"


@dataclass(frozen=True)
class ShardLabelTaskFacts:
    """One exact collector molecule shard and its indexed episodes."""

    rank: int
    shard_index: int
    rows: tuple[EpisodeIndexRow, ...]
    reconstruction_csv: Path
    molecules_sdf: Path

    @property
    def key(self) -> tuple[int, int]:
        """Return the stable rank/shard locator."""
        return self.rank, self.shard_index


@dataclass(frozen=True)
class LabelingRunConfig:
    """Resolved paths and scientific controls for one labeling run."""

    name: str
    config_path: Path
    config_sha256: str
    source_run_dir: Path
    episode_index_path: Path
    output_dir: Path
    policy: LabelPolicyConfig
    posebusters: PoseBustersBackendConfig | None
    docking: UniDockLabelingConfig | None
    split_enabled: bool
    split_version: str
    split_salt: str
    split_fractions: tuple[float, float, float]


def sha256_file(path: str | Path) -> str:
    """Hash one file without loading it fully into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_path(value: object, *, data_root: Path, name: str) -> Path:
    """Resolve a required configured path relative to the selected data root."""
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{name} cannot be empty")
    path = Path(text)
    return path.resolve() if path.is_absolute() else (data_root / path).resolve()


def _mapping(value: object, *, name: str) -> dict[str, Any]:
    """Return a plain mapping or reject malformed YAML sections."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a YAML mapping")
    return dict(value)


def load_labeling_run_config(
    config_path: str | Path,
    *,
    data_root: str | Path,
    worker_id: int,
    gpu_id_override: int | None = None,
) -> LabelingRunConfig:
    """Load one strict YAML config and resolve worker-specific GPU selection."""
    path = Path(config_path).resolve()
    root = Path(data_root).resolve()
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"failed to load labeling config {path}") from exc
    root_config = _mapping(raw, name="config")
    labeling = _mapping(root_config.get("offline_labeling"), name="offline_labeling")
    policy_raw = _mapping(labeling.get("policy"), name="offline_labeling.policy")
    pb_raw = _mapping(
        labeling.get("posebusters", {}),
        name="offline_labeling.posebusters",
    )
    docking_raw = _mapping(
        labeling.get("docking", {}),
        name="offline_labeling.docking",
    )
    split_raw = _mapping(
        labeling.get("split", {}),
        name="offline_labeling.split",
    )
    policy = LabelPolicyConfig(**policy_raw)
    pb_enabled = bool(pb_raw.pop("enabled", False))
    docking_enabled = bool(docking_raw.pop("enabled", False))
    if policy.posebusters_enabled != pb_enabled:
        raise ValueError("policy and PoseBusters enablement disagree")
    if policy.docking_enabled != docking_enabled:
        raise ValueError("policy and docking enablement disagree")

    posebusters = None
    if pb_enabled:
        posebusters = PoseBustersBackendConfig(
            receptor_pdb=_resolve_path(
                pb_raw.pop("receptor_pdb", None),
                data_root=root,
                name="posebusters.receptor_pdb",
            ),
            **pb_raw,
        )

    docking = None
    if docking_enabled:
        gpu_ids_raw = docking_raw.pop("gpu_ids", None)
        configured_gpu_ids = (
            tuple(int(value) for value in gpu_ids_raw)
            if isinstance(gpu_ids_raw, Sequence)
            and not isinstance(gpu_ids_raw, (str, bytes))
            else ()
        )
        if gpu_id_override is not None:
            gpu_id = gpu_id_override
        elif configured_gpu_ids:
            gpu_id = configured_gpu_ids[worker_id % len(configured_gpu_ids)]
        else:
            gpu_id = worker_id
        receptor_pdbqt = _resolve_path(
            docking_raw.pop("receptor_pdbqt", None),
            data_root=root,
            name="docking.receptor_pdbqt",
        )
        unidock_bin = Path(
            str(docking_raw.pop("unidock_bin", "/opt/unidock-1.1.3/bin/unidock"))
        ).resolve()
        docking = UniDockLabelingConfig(
            receptor_pdbqt=receptor_pdbqt,
            gpu_id=gpu_id,
            unidock_bin=unidock_bin,
            **docking_raw,
        )

    fractions_raw = split_raw.get("fractions", (0.8, 0.1, 0.1))
    if not isinstance(fractions_raw, Sequence) or isinstance(
        fractions_raw, (str, bytes)
    ):
        raise ValueError("split.fractions must contain three numbers")
    fractions = tuple(float(value) for value in fractions_raw)
    if len(fractions) != 3 or not math.isclose(sum(fractions), 1.0, abs_tol=1e-9):
        raise ValueError("split.fractions must contain three values summing to one")

    name = str(root_config.get("name", "")).strip()
    if not name:
        raise ValueError("config name cannot be empty")
    return LabelingRunConfig(
        name=name,
        config_path=path,
        config_sha256=sha256_file(path),
        source_run_dir=_resolve_path(
            labeling.get("source_run_dir"),
            data_root=root,
            name="offline_labeling.source_run_dir",
        ),
        episode_index_path=_resolve_path(
            labeling.get("episode_index"),
            data_root=root,
            name="offline_labeling.episode_index",
        ),
        output_dir=_resolve_path(
            labeling.get("output_dir"),
            data_root=root,
            name="offline_labeling.output_dir",
        ),
        policy=policy,
        posebusters=posebusters,
        docking=docking,
        split_enabled=bool(split_raw.get("enabled", True)),
        split_version=str(split_raw.get("version", "offline_label_split_v1")),
        split_salt=str(split_raw.get("salt", "diffsbdd-offline-labels-v1")),
        split_fractions=(fractions[0], fractions[1], fractions[2]),
    )


def build_shard_label_tasks(
    episode_index: EpisodeIndex,
    *,
    source_run_dir: str | Path,
) -> tuple[ShardLabelTaskFacts, ...]:
    """Bind indexed shard groups to the collector's exact molecule artifacts."""
    run_dir = Path(source_run_dir)
    grouped: dict[tuple[int, int], list[EpisodeIndexRow]] = defaultdict(list)
    for row in episode_index.rows:
        grouped[(row.rank, row.shard_index)].append(row)
    tasks: list[ShardLabelTaskFacts] = []
    for (rank, shard_index), rows in sorted(grouped.items()):
        source_shards = {row.source_shard_sha256.lower() for row in rows}
        shard_paths = {row.shard_path for row in rows}
        if len(source_shards) != 1 or len(shard_paths) != 1:
            raise ValueError(f"index shard group {(rank, shard_index)} is inconsistent")
        molecule_dir = run_dir / "molecules" / f"rank_{rank:02d}" / f"shard_{shard_index:04d}"
        reconstruction_csv = molecule_dir / "reconstruction.csv"
        molecules_sdf = molecule_dir / f"diffsbdd_collect_r{rank:02d}_s{shard_index:04d}.sdf"
        for name, path in (
            ("reconstruction CSV", reconstruction_csv),
            ("molecule SDF", molecules_sdf),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"{name} not found: {path}")
        tasks.append(
            ShardLabelTaskFacts(
                rank=rank,
                shard_index=shard_index,
                rows=tuple(sorted(rows, key=lambda row: row.episode_index)),
                reconstruction_csv=reconstruction_csv,
                molecules_sdf=molecules_sdf,
            )
        )
    return tuple(tasks)


def select_worker_tasks(
    tasks: Sequence[ShardLabelTaskFacts],
    *,
    worker_id: int,
    num_workers: int,
    max_shards: int | None = None,
) -> tuple[ShardLabelTaskFacts, ...]:
    """Assign globally ordered shards by modulo for stable multi-worker resume."""
    if num_workers <= 0 or not 0 <= worker_id < num_workers:
        raise ValueError("worker_id must lie in [0,num_workers)")
    if max_shards is not None and max_shards <= 0:
        raise ValueError("max_shards must be positive")
    selected = tuple(
        task for ordinal, task in enumerate(tasks) if ordinal % num_workers == worker_id
    )
    return selected if max_shards is None else selected[:max_shards]


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically persist one canonical JSON object."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _task_dir(output_dir: Path, task: ShardLabelTaskFacts) -> Path:
    """Return one stable rank/shard output directory."""
    return output_dir / "shards" / f"rank_{task.rank:02d}" / f"shard_{task.shard_index:04d}"


def _task_inputs(
    task: ShardLabelTaskFacts,
    *,
    run_config: LabelingRunConfig,
    episode_index: EpisodeIndex,
) -> dict[str, Any]:
    """Build immutable input fingerprints used to authorize resume skips."""
    backend_sha256: dict[str, str] = {}
    if run_config.posebusters is not None:
        backend_sha256["posebusters_receptor_pdb"] = sha256_file(
            run_config.posebusters.receptor_pdb
        )
    if run_config.docking is not None:
        backend_sha256["docking_receptor_pdbqt"] = sha256_file(
            run_config.docking.receptor_pdbqt
        )
        backend_sha256["unidock_binary"] = sha256_file(
            run_config.docking.unidock_bin
        )
    return {
        "format": LABEL_SHARD_STATUS_FORMAT,
        "config_sha256": run_config.config_sha256,
        "episode_index_sha256": episode_index.sha256,
        "source_run_id": episode_index.source_run_id,
        "rank": task.rank,
        "shard_index": task.shard_index,
        "sample_count": len(task.rows),
        "sample_ids_sha256": hashlib.sha256(
            ("\n".join(row.sample_id for row in task.rows) + "\n").encode("utf-8")
        ).hexdigest(),
        "source_shard_sha256": task.rows[0].source_shard_sha256.lower(),
        "source_checkpoint_sha256": task.rows[0].source_checkpoint_sha256.lower(),
        "reconstruction_csv_sha256": sha256_file(task.reconstruction_csv),
        "molecules_sdf_sha256": sha256_file(task.molecules_sdf),
        "backend_sha256": backend_sha256,
        "reward_version": run_config.policy.reward_version,
    }


def _validate_completed_status(
    status: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
    sidecar_path: Path,
    task: ShardLabelTaskFacts,
) -> None:
    """Require exact input, output, sample, and provenance agreement."""
    mismatches = [name for name, value in expected.items() if status.get(name) != value]
    if mismatches:
        raise ValueError(f"completed shard status input mismatch: {mismatches}")
    if status.get("status") != "completed":
        raise ValueError("shard status is not completed")
    if not sidecar_path.is_file():
        raise FileNotFoundError(f"completed shard sidecar not found: {sidecar_path}")
    if status.get("sidecar_sha256") != sha256_file(sidecar_path):
        raise ValueError("completed shard sidecar SHA mismatch")
    sidecar = load_reward_sidecar(
        sidecar_path,
        expected_reward_version=str(expected["reward_version"]),
    )
    labels = {row.sample_id: row for row in sidecar.rows}
    indexed = {row.sample_id: row for row in task.rows}
    if set(labels) != set(indexed):
        raise ValueError("completed shard sidecar sample coverage mismatch")
    for sample_id, index_row in indexed.items():
        label = labels[sample_id]
        if (
            label.source_shard_sha256.lower() != index_row.source_shard_sha256.lower()
            or label.source_checkpoint_sha256.lower()
            != index_row.source_checkpoint_sha256.lower()
        ):
            raise ValueError(f"completed shard provenance mismatch for {sample_id}")


def _label_counts(sidecar: RewardSidecar) -> dict[str, int]:
    """Count primary label-stage outcomes for status and summaries."""
    return {
        "attempts": len(sidecar.rows),
        "reconstruction_success": sum(row.reconstruction_success for row in sidecar.rows),
        "sanitize_success": sum(row.sanitize_success for row in sidecar.rows),
        "rdkit_valid": sum(row.rdkit_valid for row in sidecar.rows),
        "pb_evaluated": sum(row.pb_evaluated for row in sidecar.rows),
        "pb_valid": sum(row.pb_valid for row in sidecar.rows),
        "docking_input_success": sum(row.docking_input_success for row in sidecar.rows),
        "docking_success": sum(row.docking_success for row in sidecar.rows),
        "actual_good": sum(row.actual_good for row in sidecar.rows),
    }


def label_one_task(
    task: ShardLabelTaskFacts,
    *,
    run_config: LabelingRunConfig,
    episode_index: EpisodeIndex,
    resume: bool,
    posebusters_labeler: PoseBustersLabeler | None = None,
    docking_labeler: UniDockLabeler | None = None,
) -> tuple[str, Path, dict[str, int]]:
    """Label one shard atomically or validate an exact completed resume artifact."""
    task_dir = _task_dir(run_config.output_dir, task)
    sidecar_path = task_dir / "reward_sidecar.csv"
    status_path = task_dir / "status.json"
    expected = _task_inputs(task, run_config=run_config, episode_index=episode_index)
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") == "completed":
            if not resume:
                raise FileExistsError(
                    f"completed shard exists; pass --resume to validate and skip: {task_dir}"
                )
            _validate_completed_status(
                status,
                expected=expected,
                sidecar_path=sidecar_path,
                task=task,
            )
            return "skipped", sidecar_path, dict(status["counts"])
        if not resume:
            raise FileExistsError(
                f"non-completed shard status exists; pass --resume to recompute: {status_path}"
            )
    elif sidecar_path.exists() and not resume:
        raise FileExistsError(
            f"orphan shard sidecar exists; pass --resume to recompute: {sidecar_path}"
        )

    task_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        owns_posebusters_labeler = False
        pb_labeler = posebusters_labeler
        if pb_labeler is None and run_config.posebusters is not None:
            pb_labeler = PoseBustersLabeler(run_config.posebusters)
            owns_posebusters_labeler = True
        active_docking_labeler = docking_labeler
        if active_docking_labeler is None and run_config.docking is not None:
            active_docking_labeler = UniDockLabeler(run_config.docking)
        rows = label_trajectory_shard(
            task.rows,
            reconstruction_csv=task.reconstruction_csv,
            molecules_sdf=task.molecules_sdf,
            config=run_config.policy,
            posebusters=pb_labeler,
            docking=active_docking_labeler,
        )
        from .reward_sidecar import write_reward_sidecar

        write_reward_sidecar(sidecar_path, rows)
        sidecar = load_reward_sidecar(
            sidecar_path,
            expected_reward_version=run_config.policy.reward_version,
        )
        counts = _label_counts(sidecar)
        status = {
            **expected,
            "status": "completed",
            "seconds": time.perf_counter() - started,
            "counts": counts,
            "sidecar_file": sidecar_path.name,
            "sidecar_sha256": sha256_file(sidecar_path),
        }
        _atomic_write_json(status_path, status)
        return "completed", sidecar_path, counts
    except Exception as exc:
        _atomic_write_json(
            status_path,
            {
                **expected,
                "status": "failed",
                "seconds": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            },
        )
        raise
    finally:
        if "owns_posebusters_labeler" in locals() and owns_posebusters_labeler:
            pb_labeler.close()


def _reward_strata(sidecar: RewardSidecar) -> dict[str, str]:
    """Build explicit coarse strata from independent label stages."""
    strata: dict[str, str] = {}
    for row in sidecar.rows:
        if not row.rdkit_valid:
            stratum = f"invalid:{row.failure_reason}"
        elif not row.pb_evaluated:
            stratum = "pb_not_evaluated"
        elif not row.pb_valid:
            stratum = "pb_invalid"
        elif not row.docking_success:
            stratum = "docking_failed"
        elif row.actual_good:
            stratum = "actual_good"
        else:
            stratum = "scored_not_actual_good"
        strata[row.sample_id] = stratum
    return strata


def _write_summary_csv(path: Path, summary: Mapping[str, Any]) -> None:
    """Write one flat denominator-explicit summary row."""
    counts = dict(summary["counts"])
    row = {
        "status": summary["status"],
        "scope": summary["scope"],
        "formal_dataset_complete": int(bool(summary["formal_dataset_complete"])),
        **counts,
        "rdkit_valid_rate_attempts": summary["rates"]["rdkit_valid_rate_attempts"],
        "pb_valid_rate_attempts": summary["rates"]["pb_valid_rate_attempts"],
        "pb_valid_rate_rdkit_valid": summary["rates"]["pb_valid_rate_rdkit_valid"],
        "docking_success_rate_attempts": summary["rates"]["docking_success_rate_attempts"],
        "docking_success_rate_inputs": summary["rates"]["docking_success_rate_inputs"],
        "actual_good_rate_attempts": summary["rates"]["actual_good_rate_attempts"],
        "reward_mean": summary["reward"]["mean"],
        "reward_median": summary["reward"]["median"],
        "docking_score_mean": summary["docking"]["score_mean"],
        "docking_score_median": summary["docking"]["score_median"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def _write_summary_plot(path: Path, sidecar: RewardSidecar) -> None:
    """Plot stage counts and successful docking-score distribution."""
    import matplotlib.pyplot as plt

    counts = _label_counts(sidecar)
    stage_names = (
        "attempts",
        "rdkit_valid",
        "pb_valid",
        "docking_input_success",
        "docking_success",
        "actual_good",
    )
    scores = [
        float(row.docking_raw)
        for row in sidecar.rows
        if row.docking_success and row.docking_raw is not None
    ]
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    axes[0].bar(
        [name.replace("_", "\n") for name in stage_names],
        [counts[name] for name in stage_names],
        color=("#4c78a8", "#59a14f", "#f28e2b", "#76b7b2", "#e15759", "#b07aa1"),
    )
    axes[0].set_title("Offline label stage counts", fontsize=18)
    axes[0].set_ylabel("Molecules", fontsize=15)
    axes[0].tick_params(labelsize=12)
    if scores:
        axes[1].hist(scores, bins=min(30, max(5, int(len(scores) ** 0.5))), color="#4c78a8")
        axes[1].set_xlabel("UniDock score (kcal/mol)", fontsize=15)
        axes[1].set_ylabel("Molecules", fontsize=15)
    else:
        axes[1].text(0.5, 0.5, "No successful docking scores", ha="center", va="center", fontsize=15)
        axes[1].set_xticks([])
        axes[1].set_yticks([])
    axes[1].set_title("Successful docking score distribution", fontsize=18)
    axes[1].tick_params(labelsize=12)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def finalize_labeling_run(
    tasks: Sequence[ShardLabelTaskFacts],
    *,
    all_task_count: int,
    episode_index: EpisodeIndex,
    run_config: LabelingRunConfig,
) -> dict[str, Any]:
    """Validate selected shard statuses, merge labels, freeze split, and summarize."""
    if not tasks:
        raise ValueError("cannot finalize an empty task selection")
    selected_rows = tuple(row for task in tasks for row in task.rows)
    selected_index = EpisodeIndex(selected_rows)
    sidecar_paths: list[Path] = []
    for task in tasks:
        task_dir = _task_dir(run_config.output_dir, task)
        status_path = task_dir / "status.json"
        if not status_path.is_file():
            raise FileNotFoundError(f"missing shard status: {status_path}")
        status = json.loads(status_path.read_text(encoding="utf-8"))
        sidecar_path = task_dir / "reward_sidecar.csv"
        _validate_completed_status(
            status,
            expected=_task_inputs(
                task,
                run_config=run_config,
                episode_index=episode_index,
            ),
            sidecar_path=sidecar_path,
            task=task,
        )
        sidecar_paths.append(sidecar_path)

    merged_path = run_config.output_dir / "reward_sidecar.csv"
    sidecar = merge_reward_sidecar_shards(
        sidecar_paths,
        episode_index=selected_index,
        output_path=merged_path,
    )
    split_counts: Counter[str] = Counter()
    split_path: Path | None = None
    if run_config.split_enabled:
        split = build_split_manifest(
            selected_index,
            split_version=run_config.split_version,
            salt=run_config.split_salt,
            fractions=run_config.split_fractions,
            reward_strata=_reward_strata(sidecar),
        )
        split_path = write_split_manifest(
            run_config.output_dir / "split_manifest.jsonl",
            split,
        )
        split_counts.update(row.split for row in split.rows)

    counts = _label_counts(sidecar)
    attempts = counts["attempts"]
    scores = [
        float(row.docking_raw)
        for row in sidecar.rows
        if row.docking_success and row.docking_raw is not None
    ]
    rewards = [float(row.reward_total_train) for row in sidecar.rows]
    complete = len(tasks) == all_task_count and len(selected_index.rows) == len(
        episode_index.rows
    )
    summary: dict[str, Any] = {
        "format": LABEL_RUN_SUMMARY_FORMAT,
        "status": "completed",
        "scope": "full_episode_index" if complete else "partial_smoke",
        "formal_labels": run_config.policy.formal_labels,
        "formal_dataset_complete": bool(complete and run_config.policy.formal_labels),
        "config_path": str(run_config.config_path),
        "config_sha256": run_config.config_sha256,
        "episode_index_path": str(run_config.episode_index_path),
        "episode_index_sha256": selected_index.sha256,
        "source_episode_index_sha256": episode_index.sha256,
        "source_run_id": episode_index.source_run_id,
        "shards_completed": len(tasks),
        "shards_expected": all_task_count,
        "reward_sidecar": str(merged_path),
        "reward_sidecar_sha256": sha256_file(merged_path),
        "split_manifest": str(split_path) if split_path is not None else None,
        "counts": counts,
        "rates": {
            "rdkit_valid_rate_attempts": counts["rdkit_valid"] / attempts,
            "pb_valid_rate_attempts": counts["pb_valid"] / attempts,
            "pb_valid_rate_rdkit_valid": (
                counts["pb_valid"] / counts["rdkit_valid"]
                if counts["rdkit_valid"]
                else 0.0
            ),
            "docking_success_rate_attempts": counts["docking_success"] / attempts,
            "docking_success_rate_inputs": (
                counts["docking_success"] / counts["docking_input_success"]
                if counts["docking_input_success"]
                else 0.0
            ),
            "actual_good_rate_attempts": counts["actual_good"] / attempts,
        },
        "reward": {
            "mean": mean(rewards),
            "median": median(rewards),
            "min": min(rewards),
            "max": max(rewards),
        },
        "docking": {
            "score_count": len(scores),
            "score_mean": mean(scores) if scores else None,
            "score_median": median(scores) if scores else None,
            "score_best_min": min(scores) if scores else None,
            "score_worst_max": max(scores) if scores else None,
        },
        "split_counts": dict(sorted(split_counts.items())),
        "failure_counts": dict(
            sorted(Counter(row.failure_reason for row in sidecar.rows).items())
        ),
    }
    _atomic_write_json(run_config.output_dir / "summary.json", summary)
    _write_summary_csv(run_config.output_dir / "summary.csv", summary)
    _write_summary_plot(run_config.output_dir / "label_summary.png", sidecar)
    return summary
