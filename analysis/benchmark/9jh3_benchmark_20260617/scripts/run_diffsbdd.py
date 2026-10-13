"""DiffSBDD denovo generation shard for 9JH3 (200 steps, 500 mols)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from rdkit import Chem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pdbfile", type=Path, required=True)
    parser.add_argument("--ref-ligand", default="R:401")
    parser.add_argument("--outfile", type=Path, required=True)
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--timesteps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument("--reconstruction-method", default="openbabel")
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo_root.resolve() / "DiffSBDD"))
    from lightning_modules import LigandPocketDDPM
    import utils

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DiffSBDD] device={device}  samples={args.n_samples}  timesteps={args.timesteps}  seed={args.seed}")
    started = time.time()

    model = LigandPocketDDPM.load_from_checkpoint(str(args.checkpoint), map_location=device)
    model = model.to(device)
    model.eval()

    molecules: list[Chem.Mol] = []
    remaining = args.n_samples
    while remaining > 0:
        current = min(args.batch_size, remaining)
        batch = model.generate_ligands(
            str(args.pdbfile), current,
            pocket_ids=None, ref_ligand=args.ref_ligand,
            num_nodes_lig=None, sanitize=False, largest_frag=False,
            relax_iter=0, timesteps=args.timesteps,
            reconstruction_method=args.reconstruction_method,
        )
        molecules.extend(batch)
        remaining -= current
        elapsed = time.time() - started
        print(f"  [{len(molecules)}/{args.n_samples}]  {elapsed:.0f}s  ({len(molecules) / elapsed:.1f} mol/s)")

    args.outfile.parent.mkdir(parents=True, exist_ok=True)
    utils.write_sdf_file(str(args.outfile), molecules)
    elapsed = time.time() - started
    summary = {
        "device": device,
        "requested": args.n_samples,
        "written": len(molecules),
        "batch_size": args.batch_size,
        "timesteps": args.timesteps,
        "seed": args.seed,
        "seconds": elapsed,
        "mol_per_s": len(molecules) / elapsed if elapsed > 0 else 0,
        "outfile": str(args.outfile),
    }
    args.outfile.with_suffix(".json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
