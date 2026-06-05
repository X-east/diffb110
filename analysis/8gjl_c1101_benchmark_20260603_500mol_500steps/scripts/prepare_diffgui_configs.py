"""Create per-GPU DiffGui sampling configs for the 8GJL/C1101 benchmark."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--counts", default="72,72,72,71,71,71,71")
    parser.add_argument("--base-seed", type=int, default=20260603)
    args = parser.parse_args()

    counts = [int(item) for item in args.counts.split(",") if item]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for gpu_id, count in enumerate(counts):
        config = {
            "model": {
                "checkpoint": "ckpt/trained.pt",
                "target": str(args.target.resolve()),
                "ligand": "None",
                "frag": "None",
                "gen_mode": "denovo",
                "logp": 2.0,
                "tpsa": 100,
                "sa": 1.0,
                "qed": 0.8,
                "aff": 12.0,
            },
            "bond_predictor": "ckpt/bond_trained.pt",
            "sample": {
                "seed": args.base_seed + gpu_id * 101,
                "batch_size": 2,
                "num_mols": count,
                "save_traj_prob": 0.0,
                "sample": True,
                "sample_method": "priori",
                "mode": "pocket",
                "test_id": 0,
                "add_edge": "None",
                "gui_strength": 3.0,
                "guidance": ["uncertainty", 1e-4],
            },
            "data": {
                "name": "protein_ligand",
                "path": "data/PDBbind_v2020_pocket10",
                "split": "data/PDBbind_pocket10_split.pt",
                "protein_root": "data/PDBbind_v2020",
                "dataset": "pdbbind",
                "transform": {
                    "ligand_atom_mode": "aromatic",
                    "random_rot": False,
                    "sample": False,
                },
            },
        }
        path = args.out_dir / f"diffgui_c1101_gpu{gpu_id}.yml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        rows.append({"gpu_id": gpu_id, "count": count, "seed": config["sample"]["seed"], "config": str(path)})

    with (args.out_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["gpu_id", "count", "seed", "config"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} DiffGui configs to {args.out_dir}")


if __name__ == "__main__":
    main()
