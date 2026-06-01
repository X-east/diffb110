"""Create per-GPU DiffGui sampling configs for 500 molecules per step setting."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "configs"
COUNTS = [84, 84, 83, 83, 83, 83]
STEPS = [100, 200, 1000]
BASE_SEED = 20230531


BASE_CONFIG = {
    "model": {
        "checkpoint": "ckpt/trained.pt",
        "target": "sample/3ztx_pocket.pdb",
        "ligand": "None",
        "frag": "None",
        "gen_mode": "denovo",
        "logp": 2.00,
        "tpsa": 100,
        "sa": 1.00,
        "qed": 0.80,
        "aff": 12.00,
    },
    "bond_predictor": "ckpt/bond_trained.pt",
    "sample": {
        "seed": BASE_SEED,
        "batch_size": 5,
        "num_mols": 1,
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


def main() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for steps in STEPS:
        for gpu_id, count in enumerate(COUNTS):
            config = copy.deepcopy(BASE_CONFIG)
            config["sample"]["num_mols"] = int(count)
            config["sample"]["seed"] = BASE_SEED + steps * 100 + gpu_id * 17
            config["runtime"] = {
                "diffusion_steps": int(steps),
                "gpu_id": int(gpu_id),
                "target_count": int(count),
                "total_target_for_step": int(sum(COUNTS)),
            }
            path = CONFIG_DIR / f"diffgui_steps{steps}_gpu{gpu_id}.yml"
            with path.open("w") as handle:
                yaml.safe_dump(config, handle, sort_keys=False)
            manifest.append(f"{steps},{gpu_id},{count},{config['sample']['seed']},{path}")

    (CONFIG_DIR / "manifest.csv").write_text(
        "steps,gpu_id,target_count,seed,config_path\n" + "\n".join(manifest) + "\n"
    )
    print(f"Wrote {len(manifest)} configs to {CONFIG_DIR}")


if __name__ == "__main__":
    main()
