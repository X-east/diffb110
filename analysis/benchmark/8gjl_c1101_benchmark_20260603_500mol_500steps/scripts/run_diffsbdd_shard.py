"""Run one DiffSBDD generation shard with flexible final batch size."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from rdkit import Chem


def _setup_import_path(repo_root: Path) -> None:
    sys.path.insert(0, str(repo_root / "DiffSBDD"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pdbfile", type=Path, required=True)
    parser.add_argument("--ref-ligand", default="C:1101")
    parser.add_argument("--outfile", type=Path, required=True)
    parser.add_argument("--n-samples", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--timesteps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--reconstruction-method", default="openbabel")
    args = parser.parse_args()

    _setup_import_path(args.repo_root.resolve())
    from lightning_modules import LigandPocketDDPM
    import utils

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    started = time.time()
    model = LigandPocketDDPM.load_from_checkpoint(args.checkpoint, map_location=device)
    model = model.to(device)
    model.eval()

    molecules: list[Chem.Mol] = []
    remaining = args.n_samples
    while remaining > 0:
        current = min(args.batch_size, remaining)
        batch = model.generate_ligands(
            str(args.pdbfile),
            current,
            pocket_ids=None,
            ref_ligand=args.ref_ligand,
            num_nodes_lig=None,
            sanitize=False,
            largest_frag=False,
            relax_iter=0,
            timesteps=args.timesteps,
            reconstruction_method=args.reconstruction_method,
        )
        molecules.extend(batch)
        remaining -= current

    args.outfile.parent.mkdir(parents=True, exist_ok=True)
    utils.write_sdf_file(args.outfile, molecules)
    summary = {
        "device": device,
        "requested": args.n_samples,
        "written_candidates": len(molecules),
        "batch_size": args.batch_size,
        "timesteps": args.timesteps,
        "seconds": time.time() - started,
        "outfile": str(args.outfile),
    }
    args.outfile.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
