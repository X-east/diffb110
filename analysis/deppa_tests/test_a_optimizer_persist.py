"""Test A: persistent optimizer across consecutive rollouts.

Verifies that a single optimizer object is reused for every rollout and that
AdamW step count keeps increasing (momentum is NOT reset between rollouts).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import deppa_test_common as C  # noqa: E402


def main():
    print("[A] loading model + building optimizer (single object, outside loop)")
    model = C.load_model()
    args = C.make_args(n_samples=4, rollouts=2, timesteps=100, inference_interval=5)
    pdb_file, receptor_file, ref_ligand, pocket_ids = C.load_pocket()
    ppo_config = C.build_ppo_config(model, pdb_file, receptor_file, ref_ligand, args)

    optimizer = C.new_optimizer(model, ppo_config.lr)
    opt_id = id(optimizer)
    before_sd = {k: v.detach().clone() for k, v in model.ddpm.state_dict().items()}

    step_counts = []
    for r in range(2):
        print(f"[A] rollout {r} (policy_update=True, optimizer reused)")
        em, recs = C.run_one_rollout(
            model, args, pdb_file, pocket_ids, ppo_config, optimizer,
            policy_update=True,
        )
        step_counts.append(C.optimizer_step_count(optimizer))

    after_sd = model.ddpm.state_dict()
    weight_diff = sum(
        float((after_sd[k] - before_sd[k]).abs().sum()) for k in before_sd
    )

    # Assertions
    assert id(optimizer) == opt_id, "optimizer object was replaced between rollouts"
    assert step_counts[0] == ppo_config.episode_length, (
        f"rollout 0 optimizer steps {step_counts[0]} != episode_length "
        f"{ppo_config.episode_length}"
    )
    assert step_counts[1] > step_counts[0], (
        f"optimizer step did not increase across rollouts: {step_counts}"
    )
    assert weight_diff > 1e-9, f"model weights did not change: diff={weight_diff}"

    print("[A] PASSED", dict(step_counts=step_counts, weight_diff=weight_diff))


if __name__ == "__main__":
    main()
