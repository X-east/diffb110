"""Per-epoch fixed-attempt generation for Global DPO checkpoints."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from scripts.export_diffsbdd_global_dpo_checkpoint import export_checkpoint


def _sdf_record_count(path: Path) -> int:
    """Count written SDF records without treating them as valid molecules."""
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return sum(line.strip() == "$$$$" for line in handle)


def generate_epoch(
    *,
    dpo_checkpoint: Path,
    base_checkpoint: Path,
    output_root: Path,
    pocket_pdb: Path,
    ref_ligand: str,
    generation_device: str,
    attempts: int,
    batch_size: int,
    timesteps: int,
    seed: int,
    reconstruction_method: str,
    resamplings: int,
    jump_length: int,
    all_frags: bool,
) -> dict[str, Any]:
    """Export one checkpoint and generate a fixed number of attempts."""
    if attempts <= 0 or batch_size <= 0 or attempts % batch_size != 0:
        raise ValueError("attempts must be positive and divisible by batch_size")
    epoch = int(torch_load_epoch(dpo_checkpoint))
    epoch_dir = output_root / f"epoch_{epoch:04d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    generation_checkpoint = epoch_dir / "global_dpo_generation.ckpt"
    molecule_sdf = epoch_dir / "molecules.sdf"
    generation_log = epoch_dir / "generation.log"
    if generation_checkpoint.exists() or molecule_sdf.exists():
        raise FileExistsError(f"generation output already exists: {epoch_dir}")
    export_result = export_checkpoint(dpo_checkpoint, base_checkpoint, generation_checkpoint)
    command = [
        "/home/md666/miniforge3/envs/dif/bin/python",
        str(Path(__file__).resolve().parents[2] / "DiffSBDD" / "generate_ligands.py"),
        str(generation_checkpoint),
        "--pdbfile", str(pocket_pdb),
        "--ref_ligand", ref_ligand,
        "--outfile", str(molecule_sdf),
        "--n_samples", str(attempts),
        "--batch_size", str(batch_size),
        "--timesteps", str(timesteps),
        "--seed", str(seed),
        "--resamplings", str(resamplings),
        "--jump_length", str(jump_length),
        "--reconstruction_method", reconstruction_method,
    ]
    if all_frags:
        command.append("--all_frags")
    environment = os.environ.copy()
    # Matrix runs mask each worker to one physical GPU. Preserve that mapping
    # for the generation subprocess instead of resetting every worker to GPU 0.
    environment.setdefault("CUDA_VISIBLE_DEVICES", generation_device.removeprefix("cuda:"))
    environment.update({
        "PYTHONPATH": f"{Path(__file__).resolve().parents[2]}:{Path(__file__).resolve().parents[2] / 'DiffSBDD'}",
        "PYTHONUNBUFFERED": "1",
        "BABEL_LIBDIR": "/home/md666/miniforge3/envs/dif/lib/openbabel/3.1.0",
    })
    with generation_log.open("w", encoding="utf-8") as handle:
        subprocess.run(command, cwd=Path(__file__).resolve().parents[2], env=environment, stdout=handle, stderr=subprocess.STDOUT, check=True)
    written_records = _sdf_record_count(molecule_sdf)
    manifest = {
        "status": "completed",
        "attempts": attempts,
        "written_sdf_records": written_records,
        "generation_device": generation_device,
        "cuda_visible_devices": environment.get("CUDA_VISIBLE_DEVICES", ""),
        "seed": seed,
        "timesteps": timesteps,
        "batch_size": batch_size,
        "checkpoint": str(generation_checkpoint),
        "source_dpo_checkpoint": str(dpo_checkpoint),
        "molecules_sdf": str(molecule_sdf),
        "command": command,
        "export": export_result,
    }
    (epoch_dir / "generation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def torch_load_epoch(path: Path) -> int:
    """Read only the epoch marker needed to name the generation directory."""
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    return int(payload["epoch"])
