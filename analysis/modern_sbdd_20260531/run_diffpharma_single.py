"""Run DiffPharma single-pocket generation without editing upstream code."""

from __future__ import annotations

import argparse

import test_single as runner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", required=True)
    parser.add_argument("--sdf", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--n-samples", type=int, default=100)
    args = parser.parse_args()

    runner.process_data(sdf_file=args.sdf, pdb_file=args.pdb)
    runner.process_data_h(sdf_file=args.sdf, pdb_file=args.pdb)
    runner.ligand_generation(
        test_file="tmp2.npz",
        checkpoint=args.checkpoint,
        outdir=args.outdir,
        batch_size=args.batch_size,
        n_samples=args.n_samples,
        save=True,
    )


if __name__ == "__main__":
    main()
