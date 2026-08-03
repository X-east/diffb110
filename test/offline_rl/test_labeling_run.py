"""Tests for recoverable multi-worker offline label orchestration."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from rdkit import Chem

from ddpo.offline_rl.index import (
    EPISODE_INDEX_SCHEMA_VERSION,
    EpisodeIndex,
    EpisodeIndexRow,
    write_episode_index,
)
from ddpo.offline_rl.labeling_run import (
    build_shard_label_tasks,
    finalize_labeling_run,
    label_one_task,
    load_labeling_run_config,
    select_worker_tasks,
)


def _row(rank: int, sample_id: str, shard_sha256: str) -> EpisodeIndexRow:
    """Build a one-episode shard row."""
    return EpisodeIndexRow(
        index_schema_version=EPISODE_INDEX_SCHEMA_VERSION,
        source_run_id="collector-run",
        manifest_snapshot_sha256="a" * 64,
        sample_id=sample_id,
        shard_path=f"shards/rank_{rank:02d}/shard_0000.pt",
        source_shard_sha256=shard_sha256,
        episode_index=0,
        rank=rank,
        shard_index=0,
        seed=rank + 1,
        num_nodes=3,
        num_diffusion_steps=4,
        max_n_nodes=4,
        rdkit_valid=True,
        failure_type="success",
        smiles="CCO",
        source_checkpoint_sha256="c" * 64,
    )


def _write_collector_molecule_dir(
    run_dir: Path,
    *,
    rank: int,
    sample_id: str,
) -> None:
    """Write one canonical collector molecule directory."""
    shard_dir = run_dir / "molecules" / f"rank_{rank:02d}" / "shard_0000"
    shard_dir.mkdir(parents=True)
    with (shard_dir / "reconstruction.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("sample_id", "rdkit_valid", "failure_type"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": sample_id,
                "rdkit_valid": 1,
                "failure_type": "success",
            }
        )
    molecule = Chem.MolFromSmiles("CCO")
    assert molecule is not None
    molecule.SetProp("sample_id", sample_id)
    sdf_path = shard_dir / f"diffsbdd_collect_r{rank:02d}_s0000.sdf"
    writer_sdf = Chem.SDWriter(str(sdf_path))
    writer_sdf.write(molecule)
    writer_sdf.close()


def _write_config(
    path: Path,
    *,
    run_dir: Path,
    index_path: Path,
    output_dir: Path,
) -> None:
    """Write a descriptor-only orchestration fixture config."""
    path.write_text(
        f"""name: labeling_run_test
offline_labeling:
  source_run_dir: {run_dir}
  episode_index: {index_path}
  output_dir: {output_dir}
  policy:
    reward_version: descriptor_fixture_v1
    formal_labels: false
    posebusters_enabled: false
    docking_enabled: false
  posebusters:
    enabled: false
  docking:
    enabled: false
  split:
    enabled: false
""",
        encoding="utf-8",
    )


def test_worker_resume_hash_gate_and_partial_finalize(tmp_path: Path) -> None:
    """Resume skips exact outputs and rejects changed collector inputs."""
    run_dir = tmp_path / "collector"
    output_dir = tmp_path / "labels"
    rows = (
        _row(0, "sample-r0", "d" * 64),
        _row(1, "sample-r1", "e" * 64),
    )
    episode_index = EpisodeIndex(rows)
    index_path = write_episode_index(tmp_path / "episode_index.jsonl", episode_index)
    _write_collector_molecule_dir(run_dir, rank=0, sample_id="sample-r0")
    _write_collector_molecule_dir(run_dir, rank=1, sample_id="sample-r1")
    config_path = tmp_path / "labels.yaml"
    _write_config(
        config_path,
        run_dir=run_dir,
        index_path=index_path,
        output_dir=output_dir,
    )
    config = load_labeling_run_config(
        config_path,
        data_root=tmp_path,
        worker_id=0,
    )
    tasks = build_shard_label_tasks(episode_index, source_run_dir=run_dir)

    assert [task.rank for task in select_worker_tasks(
        tasks, worker_id=0, num_workers=2
    )] == [0]
    assert [task.rank for task in select_worker_tasks(
        tasks, worker_id=1, num_workers=2
    )] == [1]

    state, _, counts = label_one_task(
        tasks[0],
        run_config=config,
        episode_index=episode_index,
        resume=False,
    )
    assert state == "completed"
    assert counts["attempts"] == 1
    state, _, _ = label_one_task(
        tasks[0],
        run_config=config,
        episode_index=episode_index,
        resume=True,
    )
    assert state == "skipped"

    summary = finalize_labeling_run(
        (tasks[0],),
        all_task_count=len(tasks),
        episode_index=episode_index,
        run_config=config,
    )
    assert summary["scope"] == "partial_smoke"
    assert not summary["formal_dataset_complete"]
    assert (output_dir / "reward_sidecar.csv").is_file()
    assert (output_dir / "summary.csv").is_file()
    assert (output_dir / "label_summary.png").is_file()

    reconstruction_csv = tasks[0].reconstruction_csv
    reconstruction_csv.write_text(
        reconstruction_csv.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="input mismatch"):
        label_one_task(
            tasks[0],
            run_config=config,
            episode_index=episode_index,
            resume=True,
        )


def test_config_rejects_policy_backend_enablement_mismatch(tmp_path: Path) -> None:
    """A formal backend cannot be silently disabled outside the reward policy."""
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(
        """name: bad
offline_labeling:
  source_run_dir: run
  episode_index: index.jsonl
  output_dir: out
  policy:
    reward_version: bad_v1
    posebusters_enabled: true
    docking_enabled: false
  posebusters:
    enabled: false
  docking:
    enabled: false
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="PoseBusters enablement disagree"):
        load_labeling_run_config(config_path, data_root=tmp_path, worker_id=0)
