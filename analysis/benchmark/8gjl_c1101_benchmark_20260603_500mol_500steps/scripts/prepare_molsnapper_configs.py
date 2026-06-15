"""Create MolSnapper per-shard sampling configs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--counts", default="72,72,72,71,71,71,71")
    parser.add_argument("--base-seed", type=int, default=20260603)
    parser.add_argument("--batch-size", type=int, default=48)
    args = parser.parse_args()

    counts = [int(item) for item in args.counts.split(",") if item]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for gpu_id, count in enumerate(counts):
        config = {
            "model": {"checkpoint": "ckpt/MolDiff.pt"},
            "bond_predictor": "ckpt/bondpred.pt",
            "sample": {
                "seed": args.base_seed + gpu_id * 101,
                "batch_size": args.batch_size,
                "num_mols": count,
                "save_traj_prob": 0.0,
                "guidance": ["uncertainty", 1e-4],
            },
        }
        path = args.out_dir / f"molsnapper_c1101_gpu{gpu_id}.yml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        rows.append({"gpu_id": gpu_id, "count": count, "seed": config["sample"]["seed"], "config": str(path)})

    with (args.out_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["gpu_id", "count", "seed", "config"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} MolSnapper configs to {args.out_dir}")


if __name__ == "__main__":
    main()
