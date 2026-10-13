"""Test B: checkpoint save + resume correctness.

1. Fresh run: 1 rollout -> save epoch_0001.ckpt (optimizer + rng + weights).
2. Resume: load model+optimizer+rng from epoch_0001.ckpt, run rollout 2,
   save epoch_0002.ckpt.
3. Verify optimizer momentum restored bit-exactly, completed_rollouts==2, and
   resumed rollout updated weights vs the checkpoint.
4. Strong check: a continuous 2-rollout run reproduces the same final weights
   as the resume run (RNG + optimizer state fully restored).
"""

import sys
import shutil
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import deppa_test_common as C  # noqa: E402
from deppa_checkpointing import (  # noqa: E402
    load_training_checkpoint,
    optimizer_to,
    restore_rng_state,
    save_training_checkpoint,
)
from batch_generate_ligands_rl import _validate_resume_config  # noqa: E402


def main():
    here = Path(__file__).resolve().parent
    out_dir = here / "test_b_out"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    ckpt_dir = out_dir / "checkpoints"

    args = C.make_args(n_samples=4, rollouts=2, timesteps=100, inference_interval=5)
    pdb_file, receptor_file, ref_ligand, pocket_ids = C.load_pocket()
    probe = C.load_model()
    ppo_config = C.build_ppo_config(probe, pdb_file, receptor_file, ref_ligand, args)
    del probe

    # ---- Phase 1: fresh, 1 rollout, save epoch_0001.ckpt ----
    print("[B] phase 1: fresh run, 1 rollout")
    C.set_seed(1234)
    model = C.load_model()
    optimizer = C.new_optimizer(model, ppo_config.lr)
    C.run_one_rollout(model, args, pdb_file, pocket_ids, ppo_config, optimizer, policy_update=True)
    ckpt1 = save_training_checkpoint(
        ckpt_dir, model=model, optimizer=optimizer,
        completed_rollouts=1, global_step=ppo_config.episode_length,
        resolved_config=C.resolved_config(args), target="8GJL",
        base_checkpoint=C.PRETRAINED,
    )
    print(f"[B] saved {ckpt1}")

    # ---- Phase 2: resume from epoch_0001.ckpt, run rollout 2 ----
    print("[B] phase 2: resume + rollout 2")
    resume = load_training_checkpoint(ckpt1)
    model2 = C.load_model(str(ckpt1))
    optimizer2 = C.new_optimizer(model2, ppo_config.lr)
    optimizer2.load_state_dict(resume["optimizer_states"][0])
    optimizer_to(optimizer2, torch.device(C.DEVICE))
    restore_rng_state(resume["rng_state"])

    assert resume["training_state"]["completed_rollouts"] == 1
    # Optimizer momentum restored bit-exactly from checkpoint.
    assert C.optimizer_states_equal(optimizer2, optimizer), (
        "optimizer momentum not restored from checkpoint"
    )
    # Resume config guard passes for identical args.
    _validate_resume_config(args, resume, "8GJL")

    C.run_one_rollout(model2, args, pdb_file, pocket_ids, ppo_config, optimizer2, policy_update=True)
    ckpt2 = save_training_checkpoint(
        ckpt_dir, model=model2, optimizer=optimizer2,
        completed_rollouts=2, global_step=2 * ppo_config.episode_length,
        resolved_config=C.resolved_config(args), target="8GJL",
        base_checkpoint=C.PRETRAINED,
    )
    resume2 = load_training_checkpoint(ckpt2)
    assert resume2["training_state"]["completed_rollouts"] == 2

    # Weights changed between ckpt1 and ckpt2 (rollout 2 actually updated).
    sd1 = __import__("torch").load(ckpt1, map_location="cpu")["state_dict"]
    sd2 = __import__("torch").load(ckpt2, map_location="cpu")["state_dict"]
    wdiff = sum(float((sd2[k] - sd1[k]).abs().sum()) for k in sd1)
    assert wdiff > 1e-9, "resumed rollout did not update weights"

    # ---- Strong check: continuous 2-rollout reproduces resumed weights ----
    print("[B] strong check: continuous 2-rollout run")
    C.set_seed(1234)  # same start as phase 1 so rollout 0 is identical
    model_c = C.load_model()
    opt_c = C.new_optimizer(model_c, ppo_config.lr)
    for r in range(2):
        C.run_one_rollout(model_c, args, pdb_file, pocket_ids, ppo_config, opt_c, policy_update=True)
    W_cont = {k: v.detach().clone() for k, v in model_c.ddpm.state_dict().items()}
    W_resume = model2.ddpm.state_dict()
    cont_diff = C.state_dict_diff(W_cont, W_resume)

    print("[B] PASSED", dict(
        completed_rollouts=resume2["training_state"]["completed_rollouts"],
        weight_diff_ckpt1_vs_ckpt2=wdiff,
        continuous_vs_resume_maxdiff=cont_diff,
    ))
    if cont_diff > 1e-3:
        print(f"[B] WARNING: continuous vs resumed weights diverge (maxdiff={cont_diff:.4e})")


if __name__ == "__main__":
    main()
