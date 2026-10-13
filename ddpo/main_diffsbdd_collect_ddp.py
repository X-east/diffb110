"""Collect frozen-policy DiffSBDD trajectories with PyTorch distributed workers.

Mirrors ``ddpo/main_diffgui_collect_ddp.py`` but for the DiffSBDD retained
baseline.  Each rank samples molecules with ``DiffSBDDRollout`` (which already
records the latent trajectory + per-step log-probabilities that ``EDMActor``
consumes during PPO), reconstructs the final molecules with the same
``annotate_batch_molecule_semantics`` path used by the RL pipeline, and persists
a CPU shard containing:

    latents      [B, T+2, N, D]   latent trajectory (z_T ... z_0, xh_final)
    logps        [B, T+1]         per-step old-policy log-probabilities
    timesteps    [B, T+1]         normalized timestep indices
    nodesxsample [B]              atom count per sample
    x            [B, N, 3]        final coordinates
    categorical  [B, N, atom_nf]  final atom-type one-hot
    advantages   None             filled by a later offline scoring pass

Each rank uses ``seed = local_rank + 1`` so every GPU draws a different RNG
stream (project rule: seed = gpu_id, 1-based).  The per-shard RNG state is
captured so the run can resume deterministically.

Reference basis (project "evidence-based change" rule):
  * ``ddpo/runtime/rollout.py::DiffSBDDRollout``   (trajectory producer)
  * ``ddpo/runtime/actor.py::EDMActor``            (trajectory consumer)
  * ``ddpo/runtime/replay.py``                      (format constraints)
  * ``ddpo/utils/molecule_semantics.py``            (reconstruction)
  * ``main_diffgui_collect_ddp.py``                 (collection framework)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import random
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import yaml


PROJECT_ROOT = Path(
    os.environ.get("DIFFB110_PROJECT_ROOT", Path(__file__).resolve().parent.parent)
).resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ddpo.integration.diffsbdd_model import DiffSBDDModel  # noqa: E402
from ddpo.runtime.protocol import DataProto, TensorDict  # noqa: E402
from ddpo.runtime.rollout import DiffSBDDRollout  # noqa: E402
from ddpo.runtime.trajectory_shard import TRAJECTORY_SHARD_FORMAT  # noqa: E402
from ddpo.utils.molecule_semantics import annotate_batch_molecule_semantics  # noqa: E402


LOGGER = logging.getLogger("DiffSBDDTrajectoryCollector")

SHARD_FORMAT = TRAJECTORY_SHARD_FORMAT


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Collection YAML config.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Exact shared run directory.")
    parser.add_argument("--resume", action="store_true", help="Resume from completed rank-local shards.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Distributed / filesystem helpers (pattern shared with main_diffgui_collect) #
# --------------------------------------------------------------------------- #
def _init_distributed() -> dict[str, int]:
    """Initialize torch.distributed from torchrun environment variables."""
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    return {"rank": rank, "local_rank": local_rank, "world_size": world_size}


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _broadcast_attempt_id(dist_info: dict[str, int]) -> str:
    value = uuid.uuid4().hex if int(dist_info["rank"]) == 0 else ""
    values = [value]
    if dist.is_available() and dist.is_initialized():
        dist.broadcast_object_list(values, src=0)
    return str(values[0])


def _resolve_run_dir(
    config: dict[str, Any], output_dir: Path | None, dist_info: dict[str, int]
) -> Path:
    """Resolve and broadcast one shared run directory."""
    if dist_info["rank"] == 0:
        if output_dir is not None:
            value = str(output_dir.resolve())
        else:
            cfg = config.get("diffsbdd_collection", {}) or {}
            root = Path(cfg.get("output_root", "runs/diffsbdd_collect"))
            name = str(config.get("name", "diffsbdd-trajectory-collection"))
            value = str(root / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    else:
        value = ""
    values = [value]
    if dist.is_available() and dist.is_initialized():
        dist.broadcast_object_list(values, src=0)
    return Path(values[0]).resolve()


def _prepare_dirs(run_dir: Path, dist_info: dict[str, int]) -> None:
    if dist_info["rank"] == 0:
        for subdir in ("config", "logs", "manifests", "molecules", "shards"):
            (run_dir / subdir).mkdir(parents=True, exist_ok=True)
    _barrier()
    (run_dir / "logs" / f"rank_{dist_info['rank']:02d}").mkdir(parents=True, exist_ok=True)


def _setup_logging(log_path: Path, level: str, dist_info: dict[str, int]) -> None:
    LOGGER.setLevel(getattr(logging, level))
    LOGGER.handlers.clear()
    formatter = logging.Formatter(
        f"%(asctime)s %(levelname)s %(name)s rank={dist_info['rank']}: %(message)s"
    )
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    if dist_info["rank"] == 0:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        LOGGER.addHandler(stream_handler)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary_path.replace(path)


def _model_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            digest.update(name.encode("utf-8"))
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# RNG determinism (per-card seed + per-shard continuation)                    #
# --------------------------------------------------------------------------- #
def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _capture_rng_state(device: torch.device) -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if device.type == "cuda":
        state["torch_cuda"] = torch.cuda.get_rng_state(device)
    return state


def _restore_rng_state(state: dict[str, Any], device: torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if device.type == "cuda" and "torch_cuda" in state:
        torch.cuda.set_rng_state(state["torch_cuda"], device)


def _restore_or_seed(
    run_dir: Path,
    *,
    rank: int,
    seed: int,
    resume: bool,
    device: torch.device,
    batch_size: int,
    num_timesteps: int,
    max_n_nodes: int,
) -> int:
    """Restore the last completed shard RNG state or initialize the rank seed once."""
    manifest_path = run_dir / "manifests" / f"rank_{rank:02d}.jsonl"
    records = _read_jsonl(manifest_path)
    if not resume or not records:
        if records and not resume:
            raise FileExistsError(f"Existing manifest requires --resume: {manifest_path}")
        _seed_all(seed)
        return 0
    for expected_index, row in enumerate(records):
        if int(row.get("shard_index", -1)) != expected_index:
            raise ValueError(
                f"Rank {rank} manifest is not contiguous at row {expected_index}: {row}"
            )
        if int(row.get("rank", -1)) != rank or int(row.get("seed", -1)) != seed:
            raise ValueError(f"Rank/seed mismatch in existing manifest row: {row}")
        if int(row.get("attempts", -1)) != batch_size:
            raise ValueError(f"Batch-size mismatch in existing manifest row: {row}")
        if int(row.get("num_timesteps", -1)) != num_timesteps:
            raise ValueError(f"Timestep mismatch in existing manifest row: {row}")
        shard_path = run_dir / str(row["shard_path"])
        if not shard_path.exists() or shard_path.stat().st_size != int(row["shard_bytes"]):
            raise FileNotFoundError(f"Missing or size-mismatched trajectory shard: {shard_path}")

    last = records[-1]
    shard_path = run_dir / str(last["shard_path"])
    payload = torch.load(shard_path, map_location="cpu", weights_only=False)
    expected_latent_shape = [batch_size, num_timesteps + 2, max_n_nodes, 13]
    expected_logps_shape = [batch_size, num_timesteps + 1]
    if list(payload["latents"].shape) != expected_latent_shape:
        raise ValueError(
            f"Resume latent shape mismatch: {list(payload['latents'].shape)} "
            f"!= {expected_latent_shape}"
        )
    if list(payload["logps"].shape) != expected_logps_shape:
        raise ValueError(
            f"Resume logps shape mismatch: {list(payload['logps'].shape)} "
            f"!= {expected_logps_shape}"
        )
    _restore_rng_state(payload["rng_state_after"], device)
    return int(last["shard_index"]) + 1


# --------------------------------------------------------------------------- #
# Model / prompts / reconstruction                                            #
# --------------------------------------------------------------------------- #
def _resolve_path(value: str) -> Path:
    p = Path(value)
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    return p


def _load_model(
    config: dict[str, Any], *, device: torch.device
) -> DiffSBDDModel:
    model_cfg = config.get("model", {}) or {}
    checkpoint = _resolve_path(model_cfg.get("checkpoint"))
    pocket_pdb = _resolve_path(model_cfg.get("pocket_pdb"))
    ref_ligand = str(model_cfg.get("ref_ligand", "C:1101"))
    dataset_name = str(model_cfg.get("dataset_name", "crossdock"))
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not pocket_pdb.exists():
        raise FileNotFoundError(f"Pocket PDB not found: {pocket_pdb}")
    model = DiffSBDDModel(
        checkpoint_path=str(checkpoint),
        pocket_pdb_path=str(pocket_pdb),
        ref_ligand_path=ref_ligand,
        device=str(device),
        dataset_name=dataset_name,
    )
    model.eval()
    return model


def _build_prompts(
    *,
    batch_size: int,
    min_nodes: int,
    max_nodes: int,
    max_n_nodes: int,
    device: torch.device,
) -> DataProto:
    """Build a prompts DataProto with per-sample node counts (seed-controlled)."""
    nodesxsample = torch.randint(
        int(min_nodes), int(max_nodes) + 1, (batch_size,), device=device
    )
    return DataProto(
        batch=TensorDict({"nodesxsample": nodesxsample}, batch_size=[batch_size]),
        meta_info={"max_n_nodes": int(max_n_nodes), "condition": False},
    )


def _write_reconstructed_outputs(
    *,
    output_dir: Path,
    rdmols: list[Any],
    sample_ids: list[str],
    rank: int,
    shard_index: int,
    seed: int,
    reconstruction_method: str,
    ntb: dict[str, Any],
) -> list[dict[str, Any]]:
    """Write final molecules to one SDF and a parallel CSV with stable ids."""
    from rdkit import Chem

    output_dir.mkdir(parents=True, exist_ok=True)
    sdf_path = output_dir / f"diffsbdd_collect_r{rank:02d}_s{shard_index:04d}.sdf"
    csv_path = output_dir / f"reconstruction.csv"

    smiles_list = ntb.get("smiles", [])
    valid_list = ntb.get("rdkit_valid", [])
    failure_list = ntb.get("failure_type", [])

    writer = Chem.SDWriter(str(sdf_path))
    writer.SetKekulize(False)
    rows: list[dict[str, Any]] = []
    try:
        for idx, mol in enumerate(rdmols):
            sample_id = sample_ids[idx]
            smi = smiles_list[idx] if idx < len(smiles_list) else None
            is_valid = bool(int(valid_list[idx])) if idx < len(valid_list) else False
            failure = failure_list[idx] if idx < len(failure_list) else "unknown"
            if mol is not None:
                mol.SetProp("sample_id", sample_id)
                mol.SetProp("generation_rank", str(rank))
                mol.SetProp("generation_seed", str(seed))
                mol.SetProp("generation_shard", str(shard_index))
                mol.SetProp("rdkit_valid", str(int(is_valid)))
                mol.SetProp("smiles", str(smi) if smi else "")
                writer.write(mol)
            rows.append(
                {
                    "sample_id": sample_id,
                    "rank": rank,
                    "shard_index": shard_index,
                    "rank_seed": seed,
                    "mol_index": idx,
                    "reconstruction_method": reconstruction_method,
                    "rdkit_valid": int(is_valid),
                    "failure_type": failure,
                    "smiles": str(smi) if smi else "",
                }
            )
    finally:
        writer.close()

    fieldnames = [
        "sample_id",
        "rank",
        "shard_index",
        "rank_seed",
        "mol_index",
        "reconstruction_method",
        "rdkit_valid",
        "failure_type",
        "smiles",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=fieldnames)
        writer_csv.writeheader()
        writer_csv.writerows(rows)
    return rows


# --------------------------------------------------------------------------- #
# Collection driver                                                          #
# --------------------------------------------------------------------------- #
def _collect(
    *,
    config: dict[str, Any],
    run_dir: Path,
    dist_info: dict[str, int],
    resume: bool,
    attempt_id: str,
) -> dict[str, Any]:
    """Run rank-local collection and aggregate a global summary."""
    collection_cfg = config.get("diffsbdd_collection", {}) or {}
    model_cfg = config.get("model", {}) or {}
    batch_size = int(collection_cfg.get("batch_size_per_rank", 1))
    num_shards = int(collection_cfg.get("num_shards_per_rank", 1))
    min_nodes = int(collection_cfg.get("min_nodes", 4))
    max_nodes = int(collection_cfg.get("max_nodes", 8))
    max_n_nodes = int(collection_cfg.get("max_n_nodes", 106))
    reconstruction_method = str(collection_cfg.get("reconstruction_method", "openbabel"))
    if batch_size <= 0 or num_shards <= 0:
        raise ValueError("Collection batch_size_per_rank and num_shards_per_rank must be positive.")
    if not (1 <= min_nodes <= max_nodes <= max_n_nodes):
        raise ValueError(
            "Expected 1 <= min_nodes <= max_nodes <= max_n_nodes, got "
            f"{min_nodes}, {max_nodes}, {max_n_nodes}"
        )

    rank = int(dist_info["rank"])
    local_rank = int(dist_info["local_rank"])
    world_size = int(dist_info["world_size"])
    expected_world_size = int(collection_cfg.get("expected_world_size", world_size))
    if world_size != expected_world_size:
        raise ValueError(
            f"Configured expected_world_size={expected_world_size}, launched world_size={world_size}"
        )
    expected_fixed_attempts = world_size * num_shards * batch_size
    configured_fixed_attempts = int(
        collection_cfg.get("fixed_attempts_total", expected_fixed_attempts)
    )
    if configured_fixed_attempts != expected_fixed_attempts:
        raise ValueError(
            f"Configured fixed_attempts_total={configured_fixed_attempts}, but "
            f"world_size*shards*batch={expected_fixed_attempts}"
        )
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("DiffSBDD trajectory collection requires CUDA.")

    model = _load_model(config, device=device)
    model_digest_before = _model_digest(model)
    checkpoint = _resolve_path(model_cfg.get("checkpoint"))
    checkpoint_sha256 = _sha256_file(checkpoint)
    num_timesteps = int(model_cfg.get("time_step", 500))

    rollout = DiffSBDDRollout(model, config)
    seed = rank + 1
    start_shard = _restore_or_seed(
        run_dir,
        rank=rank,
        seed=seed,
        resume=resume,
        device=device,
        batch_size=batch_size,
        num_timesteps=num_timesteps,
        max_n_nodes=max_n_nodes,
    )
    if start_shard > num_shards:
        raise ValueError(
            f"Rank {rank} already has {start_shard} shards, exceeding requested {num_shards}"
        )
    LOGGER.info(
        "collection start checkpoint=%s sha256=%s steps=%d batch=%d shards=%d seed=%d start_shard=%d",
        checkpoint,
        checkpoint_sha256,
        num_timesteps,
        batch_size,
        num_shards,
        seed,
        start_shard,
    )

    manifest_path = run_dir / "manifests" / f"rank_{rank:02d}.jsonl"
    rank_started = time.perf_counter()
    new_attempts = 0
    new_obabel_reconstructed = 0
    new_rdkit_valid = 0
    new_bytes = 0

    for shard_index in range(start_shard, num_shards):
        shard_started = time.perf_counter()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        prompts = _build_prompts(
            batch_size=batch_size,
            min_nodes=min_nodes,
            max_nodes=max_nodes,
            max_n_nodes=max_n_nodes,
            device=device,
        )
        data = rollout.generate_minibatch(prompts)
        data = annotate_batch_molecule_semantics(
            data, reconstruction_method=reconstruction_method
        )
        ntb = data.non_tensor_batch
        rdmols = ntb.get("rdmols", [None] * batch_size)
        obabel_flags = ntb.get("obabel_reconstruct_success", [0] * batch_size)
        sanitize_flags = ntb.get("sanitize_success", [0] * batch_size)
        valid_flags = ntb.get("rdkit_valid", [0] * batch_size)
        obabel_reconstructed = int(sum(int(v) for v in obabel_flags))
        sanitize_success = int(sum(int(v) for v in sanitize_flags))
        rdkit_valid = int(sum(int(v) for v in valid_flags))

        mol_dir = run_dir / "molecules" / f"rank_{rank:02d}" / f"shard_{shard_index:04d}"
        sample_ids = [
            f"r{rank:02d}_s{shard_index:04d}_m{idx:04d}" for idx in range(batch_size)
        ]
        rows = _write_reconstructed_outputs(
            output_dir=mol_dir,
            rdmols=rdmols,
            sample_ids=sample_ids,
            rank=rank,
            shard_index=shard_index,
            seed=seed,
            reconstruction_method=reconstruction_method,
            ntb=ntb,
        )

        rng_state_after = _capture_rng_state(device)
        shard_path = run_dir / "shards" / f"rank_{rank:02d}" / f"shard_{shard_index:04d}.pt"
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": SHARD_FORMAT,
            "latents": data.batch["latents"],
            "logps": data.batch["logps"],
            "timesteps": data.batch["timesteps"],
            "nodesxsample": data.batch["nodesxsample"],
            "x": data.batch["x"],
            "categorical": data.batch["categorical"],
            "advantages": None,
            "sample_ids": sample_ids,
            "metadata": {
                "rank": rank,
                "local_rank": local_rank,
                "shard_index": shard_index,
                "seed": seed,
                "num_timesteps": num_timesteps,
                "max_n_nodes": max_n_nodes,
                "condition": bool(prompts.meta_info.get("condition", False)),
                "reconstruction_method": reconstruction_method,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_sha256,
                "advantages_status": "missing_fill_after_offline_scoring",
            },
            "reconstruction_rows": rows,
            "rng_state_after": rng_state_after,
        }
        temporary_path = shard_path.with_suffix(".pt.tmp")
        torch.save(payload, temporary_path)
        temporary_path.replace(shard_path)

        shard_sha256 = _sha256_file(shard_path)
        shard_bytes = shard_path.stat().st_size
        trajectory_steps = int(data.batch["latents"].shape[1])
        manifest_row = {
            "rank": rank,
            "shard_index": shard_index,
            "seed": seed,
            "attempts": batch_size,
            "obabel_reconstructed": obabel_reconstructed,
            "sanitize_success": sanitize_success,
            "rdkit_valid": rdkit_valid,
            "sdf_written": rdkit_valid,
            "reconstructed": rdkit_valid,
            "reconstructed_field_semantics": "legacy alias of rdkit_valid",
            "num_timesteps": num_timesteps,
            "trajectory_steps": trajectory_steps,
            "latent_shape": list(data.batch["latents"].shape),
            "logps_shape": list(data.batch["logps"].shape),
            "shard_path": str(shard_path.relative_to(run_dir)),
            "shard_bytes": shard_bytes,
            "shard_sha256": shard_sha256,
            "elapsed_seconds": time.perf_counter() - shard_started,
            "torch_peak_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024.0**2),
            "status": "completed",
        }
        _append_jsonl(manifest_path, manifest_row)
        new_attempts += batch_size
        new_obabel_reconstructed += obabel_reconstructed
        new_rdkit_valid += rdkit_valid
        new_bytes += shard_bytes
        LOGGER.info("shard complete %s", json.dumps(manifest_row, ensure_ascii=False))

    model_digest_after = _model_digest(model)
    manifest_records = _read_jsonl(manifest_path)
    local_summary = {
        "attempt_id": attempt_id,
        "rank": rank,
        "seed": seed,
        "new_attempts": new_attempts,
        "new_obabel_reconstructed": new_obabel_reconstructed,
        "new_rdkit_valid": new_rdkit_valid,
        "new_shard_bytes": new_bytes,
        "attempts_total": sum(int(row.get("attempts", 0)) for row in manifest_records),
        "obabel_reconstructed_total": sum(
            int(row.get("obabel_reconstructed", row.get("reconstructed", 0)))
            for row in manifest_records
        ),
        "rdkit_valid_total": sum(
            int(row.get("rdkit_valid", row.get("reconstructed", 0)))
            for row in manifest_records
        ),
        "shard_bytes_total": sum(int(row.get("shard_bytes", 0)) for row in manifest_records),
        "completed_shards": sum(1 for row in manifest_records if row.get("status") == "completed"),
        "model_parameters_unchanged": model_digest_before == model_digest_after,
        "elapsed_seconds": time.perf_counter() - rank_started,
    }
    rank_summary_path = run_dir / "rank_summaries" / attempt_id / f"rank_{rank:02d}.json"
    _write_json_atomic(rank_summary_path, local_summary)
    LOGGER.info("rank collection complete summary=%s", json.dumps(local_summary, ensure_ascii=False))
    if rank != 0:
        return local_summary

    gathered = _wait_for_rank_summaries(
        run_dir,
        attempt_id,
        world_size,
        timeout_seconds=float(collection_cfg.get("finalize_timeout_seconds", 7200.0)),
    )
    attempts_total = sum(int(row["attempts_total"]) for row in gathered)
    completed_shards_total = sum(int(row["completed_shards"]) for row in gathered)
    expected_shards_total = world_size * num_shards
    complete = (
        len(gathered) == world_size
        and attempts_total == expected_fixed_attempts
        and completed_shards_total == expected_shards_total
        and all(bool(row["model_parameters_unchanged"]) for row in gathered)
    )
    summary = {
        "stage": "diffsbdd_frozen_policy_trajectory_collection",
        "status": "completed" if complete else "incomplete",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "world_size": world_size,
        "num_timesteps": num_timesteps,
        "batch_size_per_rank": batch_size,
        "num_shards_per_rank": num_shards,
        "expected_shards_total": expected_shards_total,
        "completed_shards_total": completed_shards_total,
        "expected_fixed_attempts": expected_fixed_attempts,
        "attempts_new": sum(int(row["new_attempts"]) for row in gathered),
        "attempts_total": attempts_total,
        "obabel_reconstructed_new": sum(
            int(row["new_obabel_reconstructed"]) for row in gathered
        ),
        "obabel_reconstructed_total": sum(
            int(row["obabel_reconstructed_total"]) for row in gathered
        ),
        "rdkit_valid_new": sum(int(row["new_rdkit_valid"]) for row in gathered),
        "rdkit_valid_total": sum(int(row["rdkit_valid_total"]) for row in gathered),
        "shard_bytes_new": sum(int(row["new_shard_bytes"]) for row in gathered),
        "shard_bytes_total": sum(int(row["shard_bytes_total"]) for row in gathered),
        "all_models_unchanged": all(bool(row["model_parameters_unchanged"]) for row in gathered),
        "rank_summaries": gathered,
    }
    if not complete:
        raise RuntimeError(f"Trajectory collection completion check failed: {summary}")
    return summary


def _wait_for_rank_summaries(run_dir: Path, attempt_id: str, world_size: int, timeout_seconds: float = 7200.0) -> list[dict[str, Any]]:
    summary_dir = run_dir / "rank_summaries" / attempt_id
    deadline = time.monotonic() + max(1.0, float(timeout_seconds))
    last_reported = -1
    while True:
        gathered: list[dict[str, Any]] = []
        for rank in range(int(world_size)):
            path = summary_dir / f"rank_{rank:02d}.json"
            if not path.exists():
                continue
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if row.get("attempt_id") == attempt_id and int(row.get("rank", -1)) == rank:
                gathered.append(row)
        if len(gathered) == int(world_size):
            return sorted(gathered, key=lambda row: int(row["rank"]))
        if len(gathered) != last_reported:
            LOGGER.info("waiting for rank summaries attempt_id=%s ready=%d/%d", attempt_id, len(gathered), world_size)
            last_reported = len(gathered)
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for rank summaries: attempt_id={attempt_id} ready={len(gathered)}/{world_size}")
        time.sleep(5.0)


def _write_run_manifest(
    args: argparse.Namespace,
    config: dict[str, Any],
    run_dir: Path,
    dist_info: dict[str, int],
    *,
    status: str,
    summary: dict[str, Any] | None = None,
) -> None:
    if dist_info["rank"] != 0:
        return
    resolved_config = run_dir / "config" / "resolved_config.yaml"
    resolved_config.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    payload = {
        "stage": "diffsbdd_frozen_policy_trajectory_collection",
        "status": status,
        "run_dir": str(run_dir),
        "config_path": str(args.config.resolve()),
        "world_size": int(dist_info["world_size"]),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "conda_prefix": os.environ.get("CONDA_PREFIX", ""),
        "python": sys.executable,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "summary": summary or {},
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    dist_info = _init_distributed()
    run_dir = _resolve_run_dir(config, args.output_dir, dist_info)
    _prepare_dirs(run_dir, dist_info)
    _setup_logging(run_dir / "logs" / f"rank_{dist_info['rank']:02d}" / "collect.log", args.log_level, dist_info)
    attempt_id = _broadcast_attempt_id(dist_info)

    completed = False
    try:
        summary = _collect(
            config=config,
            run_dir=run_dir,
            dist_info=dist_info,
            resume=args.resume,
            attempt_id=attempt_id,
        )
        if dist_info["rank"] == 0:
            (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
            _write_run_manifest(args, config, run_dir, dist_info, status="completed", summary=summary)
        completed = True
    finally:
        if dist_info["rank"] == 0 and not completed:
            _write_run_manifest(args, config, run_dir, dist_info, status="interrupted")
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
