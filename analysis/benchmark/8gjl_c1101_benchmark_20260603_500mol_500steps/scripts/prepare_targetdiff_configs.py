"""Create per-GPU TargetDiff configs for the 8GJL/C1101 benchmark."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--protein", type=Path, required=True)
    parser.add_argument("--ligand", type=Path, required=True)
    parser.add_argument("--checkpoint", default="./pretrained_models/pretrained_diffusion.pt")
    parser.add_argument("--counts", default="72,72,72,71,71,71,71")
    parser.add_argument("--base-seed", type=int, default=20260603)
    args = parser.parse_args()

    counts = [int(item) for item in args.counts.split(",") if item]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.result_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for gpu_id, count in enumerate(counts):
        result_path = args.result_root / f"gpu_{gpu_id}"
        config = {
            "model": {"checkpoint": args.checkpoint},
            "input": {
                "protein_path": str(args.protein.resolve()),
                "ligand_path": str(args.ligand.resolve()),
                "pocket_radius": 10.0,
            },
            "runtime": {
                "device": "cuda:0",
                "batch_size": 5,
                "result_path": str(result_path.resolve()),
            },
            "sample": {
                "seed": args.base_seed + gpu_id,
                "num_samples": count,
                "num_steps": 500,
                "pos_only": False,
                "center_pos_mode": "protein",
                "sample_num_atoms": "prior",
            },
        }
        path = args.out_dir / f"targetdiff_c1101_gpu{gpu_id}.yml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        rows.append({"gpu_id": gpu_id, "count": count, "seed": config["sample"]["seed"], "config": str(path), "result_path": str(result_path)})

    with (args.out_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["gpu_id", "count", "seed", "config", "result_path"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} TargetDiff configs to {args.out_dir}")


if __name__ == "__main__":
    main()
