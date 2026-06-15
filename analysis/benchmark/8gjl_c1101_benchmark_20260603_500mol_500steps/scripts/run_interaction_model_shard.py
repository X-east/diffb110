"""Run one DiffInt or DiffPharma shard with a runtime timestep override."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _load_runner(repo: Path):
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    import test_single as runner

    return runner


def _patch_timesteps(runner, timesteps: int) -> None:
    original = runner.ConditionalDDPM.sample_given_pocket

    def patched(self, *args, **kwargs):
        if kwargs.get("timesteps") is None:
            kwargs["timesteps"] = timesteps
        return original(self, *args, **kwargs)

    runner.ConditionalDDPM.sample_given_pocket = patched


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("diffint", "diffpharma"), required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pdb", type=Path, required=True)
    parser.add_argument("--sdf", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--n-samples", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--timesteps", type=int, default=500)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    args.workdir.mkdir(parents=True, exist_ok=True)
    runner = _load_runner(args.repo.resolve())
    _patch_timesteps(runner, args.timesteps)
    os.chdir(args.workdir.resolve())

    started = time.time()
    if args.kind == "diffint":
        tmp1 = runner.process_data(sdf_file=str(args.sdf), pdb_file=str(args.pdb))
        tmp2 = runner.process_data_h(sdf_file=str(args.sdf), pdb_file=str(args.pdb), npz_name=tmp1)
        runner.ligand_generation(
            test_file=tmp2,
            checkpoint=args.checkpoint,
            outdir=args.outdir,
            batch_size=args.batch_size,
            n_samples=args.n_samples,
            save=True,
        )
    else:
        runner.process_data(sdf_file=str(args.sdf), pdb_file=str(args.pdb))
        runner.process_data_h(sdf_file=str(args.sdf), pdb_file=str(args.pdb))
        runner.ligand_generation(
            test_file="tmp2.npz",
            checkpoint=args.checkpoint,
            outdir=args.outdir,
            batch_size=args.batch_size,
            n_samples=args.n_samples,
            save=True,
        )
    summary = {
        "kind": args.kind,
        "requested": args.n_samples,
        "batch_size": args.batch_size,
        "timesteps": args.timesteps,
        "seconds": time.time() - started,
        "outdir": str(args.outdir),
    }
    (args.outdir / "shard_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
