"""Generate one resumable DiffSBDD rank with a fixed RNG stream."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rdkit import Chem


LOGGER = logging.getLogger("diffsbdd_generation")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pdbfile", type=Path, required=True)
    parser.add_argument("--ref-ligand", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--attempts-per-shard", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--timesteps", type=int, required=True)
    parser.add_argument("--reconstruction-method", default="openbabel")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(),
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state(state["torch_cuda"])


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def _append_manifest(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_sdf(path: Path, molecules: list[Chem.Mol], properties: dict[str, str]) -> None:
    temp = path.with_suffix(".sdf.tmp")
    writer = Chem.SDWriter(str(temp))
    writer.SetKekulize(False)
    try:
        for molecule_index, molecule in enumerate(molecules):
            for key, value in properties.items():
                molecule.SetProp(key, value)
            molecule.SetIntProp("reconstructed_index_in_shard", molecule_index)
            writer.write(molecule)
    finally:
        writer.close()
    temp.replace(path)


def _validate_existing_records(
    records: list[dict[str, Any]], output_dir: Path, expected_seed: int
) -> None:
    for expected_index, record in enumerate(records):
        if int(record["shard_index"]) != expected_index:
            raise ValueError("Manifest shard indexes are not contiguous")
        if int(record["seed"]) != expected_seed:
            raise ValueError("Manifest seed does not match requested seed")
        for key in ("sdf_path", "rng_state_path"):
            if not (output_dir / record[key]).exists():
                raise FileNotFoundError(output_dir / record[key])


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if not torch.cuda.is_available():
        raise RuntimeError("Formal DiffSBDD generation requires a visible CUDA GPU")
    if args.attempts_per_shard <= 0 or args.batch_size <= 0:
        raise ValueError("Shard and batch sizes must be positive")

    sys.path.insert(0, str((args.repo_root / "DiffSBDD").resolve()))
    from lightning_modules import LigandPocketDDPM

    output_dir = args.output_dir.resolve()
    shard_dir = output_dir / "shards"
    rng_dir = output_dir / "rng_states"
    shard_dir.mkdir(parents=True, exist_ok=True)
    rng_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    records = _read_manifest(manifest_path)
    _validate_existing_records(records, output_dir, args.seed)
    if records and not args.resume:
        raise FileExistsError(f"Existing manifest requires --resume: {manifest_path}")
    if len(records) > args.num_shards:
        raise ValueError("Existing manifest exceeds requested shard count")

    _seed_all(args.seed)
    device = torch.device("cuda")
    started = time.time()
    LOGGER.info("loading checkpoint=%s seed=%d", args.checkpoint, args.seed, extra={"rank": args.rank})
    model = LigandPocketDDPM.load_from_checkpoint(args.checkpoint, map_location=device)
    model = model.to(device)
    model.eval()

    start_shard = len(records)
    if records:
        state_path = output_dir / records[-1]["rng_state_path"]
        state_payload = torch.load(state_path, map_location="cpu")
        _restore_rng_state(state_payload["rng_state_after"])
        LOGGER.info("resumed after shard=%d", start_shard - 1, extra={"rank": args.rank})

    reconstructed_total = sum(int(record["reconstructed"]) for record in records)
    for shard_index in range(start_shard, args.num_shards):
        shard_started = time.time()
        molecules: list[Chem.Mol] = []
        remaining = args.attempts_per_shard
        with torch.no_grad():
            while remaining > 0:
                current_batch = min(args.batch_size, remaining)
                generated = model.generate_ligands(
                    str(args.pdbfile),
                    current_batch,
                    pocket_ids=None,
                    ref_ligand=args.ref_ligand,
                    num_nodes_lig=None,
                    sanitize=False,
                    largest_frag=False,
                    relax_iter=0,
                    timesteps=args.timesteps,
                    reconstruction_method=args.reconstruction_method,
                )
                molecules.extend(generated)
                remaining -= current_batch

        sdf_relative = Path("shards") / f"shard_{shard_index:05d}.sdf"
        state_relative = Path("rng_states") / f"shard_{shard_index:05d}.pt"
        attempt_start = shard_index * args.attempts_per_shard
        _write_sdf(
            output_dir / sdf_relative,
            molecules,
            {
                "generation_model": "DiffSBDD",
                "generation_rank": str(args.rank),
                "generation_seed": str(args.seed),
                "generation_shard": str(shard_index),
                "attempt_range_start": str(attempt_start),
                "attempt_range_end_exclusive": str(attempt_start + args.attempts_per_shard),
            },
        )
        _atomic_torch_save(
            output_dir / state_relative,
            {
                "rank": args.rank,
                "seed": args.seed,
                "completed_shard": shard_index,
                "rng_state_after": _capture_rng_state(),
            },
        )
        record = {
            "rank": args.rank,
            "seed": args.seed,
            "shard_index": shard_index,
            "attempts": args.attempts_per_shard,
            "reconstructed": len(molecules),
            "reconstruction_failures": args.attempts_per_shard - len(molecules),
            "timesteps": args.timesteps,
            "seconds": time.time() - shard_started,
            "sdf_path": str(sdf_relative),
            "rng_state_path": str(state_relative),
        }
        _append_manifest(manifest_path, record)
        records.append(record)
        reconstructed_total += len(molecules)
        LOGGER.info(
            "shard=%d/%d attempts=%d reconstructed=%d elapsed=%.1fs",
            shard_index + 1,
            args.num_shards,
            args.attempts_per_shard,
            len(molecules),
            time.time() - started,
            extra={"rank": args.rank},
        )

    summary = {
        "status": "complete",
        "rank": args.rank,
        "seed": args.seed,
        "completed_shards": len(records),
        "requested_shards": args.num_shards,
        "fixed_attempts": len(records) * args.attempts_per_shard,
        "reconstructed": reconstructed_total,
        "reconstruction_failures": len(records) * args.attempts_per_shard - reconstructed_total,
        "timesteps": args.timesteps,
        "determinism_scope": (
            "seed and per-shard RNG continuation are fixed; CUDA kernels may not be "
            "bitwise deterministic across processes"
        ),
        "seconds_this_invocation": time.time() - started,
    }
    _atomic_json(output_dir / "rank_summary.json", summary)
    LOGGER.info("rank complete summary=%s", json.dumps(summary, sort_keys=True), extra={"rank": args.rank})


if __name__ == "__main__":
    main()
