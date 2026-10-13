"""Create per-GPU Pocket2Mol sampling configs for the 8GJL/C1101 benchmark."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml


def main() -> None:
    """Write one Pocket2Mol config per shard plus a manifest."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        default="/data/disk6/DiffB110/reference project/代码/Pocket2Mol/ckpt/pretrained_Pocket2Mol.pt",
    )
    parser.add_argument("--counts", default="143,143,143,143,143,143,142")
    parser.add_argument(
        "--base-seed",
        type=int,
        default=20260625,
        help="Deprecated: kept for CLI backward-compat. Seed is now gpu_id to "
        "align with DiffGui/DiffSBDD/PMDM/GCDM in launch_benchmark.sh.",
    )
    parser.add_argument("--beam-size", type=int, default=300)
    parser.add_argument("--max-steps", type=int, default=50)
    args = parser.parse_args()

    counts = [int(item) for item in args.counts.split(",") if item]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.result_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for gpu_id, count in enumerate(counts, start=1):
        result_path = args.result_root / f"gpu_{gpu_id}"
        config = {
            "model": {"checkpoint": args.checkpoint},
            "sample": {
                "seed": gpu_id,
                "num_samples": count,
                "beam_size": args.beam_size,
                "max_steps": args.max_steps,
                "threshold": {
                    "focal_threshold": 0.5,
                    "pos_threshold": 0.25,
                    "element_threshold": 0.3,
                    "hasatom_threshold": 0.6,
                    "bond_threshold": 0.4,
                },
            },
        }
        path = args.out_dir / f"pocket2mol_c1101_gpu{gpu_id}.yml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        rows.append(
            {
                "gpu_id": gpu_id,
                "count": count,
                "seed": config["sample"]["seed"],
                "config": str(path),
                "result_path": str(result_path),
            }
        )

    with (args.out_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["gpu_id", "count", "seed", "config", "result_path"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} Pocket2Mol configs to {args.out_dir}")


if __name__ == "__main__":
    main()
