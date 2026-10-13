"""Test C: inference from a saved training checkpoint.

Loads epoch_0002.ckpt via Lightning's native load_from_checkpoint (exercises
the full custom checkpoint schema), then generates 8GJL molecules with
policy_update=False (no PPO update), and writes SDF + CSV.
"""

import sys
import copy
import csv
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import deppa_test_common as C  # noqa: E402
import utils  # noqa: E402


def main():
    here = Path(__file__).resolve().parent
    out_dir = here / "test_c_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Default to the checkpoint produced by test B; allow override.
    if len(sys.argv) > 1:
        ckpt_path = Path(sys.argv[1])
    else:
        ckpt_path = here / "test_b_out" / "checkpoints" / "epoch_0002.ckpt"
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path} (run test_b first)")

    print(f"[C] loading checkpoint via load_from_checkpoint: {ckpt_path}")
    model = C.load_model(str(ckpt_path))

    args = C.make_args(n_samples=4, timesteps=100, inference_interval=5)
    pdb_file, receptor_file, ref_ligand, pocket_ids = C.load_pocket()
    ppo_config = C.build_ppo_config(model, pdb_file, receptor_file, ref_ligand, args)
    ppo_config_eval = copy.deepcopy(ppo_config)
    ppo_config_eval.inference_interval = 1  # eval uses full-step observations

    with torch.no_grad():
        em, recs = C.run_one_rollout(
            model, args, pdb_file, pocket_ids, ppo_config_eval,
            optimizer=None, policy_update=False,
        )

    assert len(recs) > 0, "no molecules generated from checkpoint"
    sdf_path = out_dir / "epoch_0002_infer.sdf"
    utils.write_sdf_file(str(sdf_path), [r["mol"] for r in recs])
    csv_path = out_dir / "epoch_0002_infer.csv"
    fields = ["rank"] + list(recs[0]["metrics"].keys())
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for idx, rec in enumerate(recs):
            row = {"rank": idx + 1}
            row.update(rec["metrics"])
            writer.writerow(row)

    print("[C] PASSED", dict(
        n_molecules=len(recs),
        sdf=str(sdf_path),
        csv=str(csv_path),
    ))


if __name__ == "__main__":
    main()
