import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import deppa_test_common as C  # noqa: E402
import torch  # noqa: E402


def main():
    model = C.load_model()
    args = C.make_args(n_samples=4, timesteps=100, inference_interval=5)
    pdb_file, receptor_file, ref_ligand, pocket_ids = C.load_pocket()
    ppo_config = C.build_ppo_config(model, pdb_file, receptor_file, ref_ligand, args)
    optimizer = C.new_optimizer(model, ppo_config.lr)

    grads = {}
    params = list(model.ddpm.dynamics.parameters())
    for idx in (0, 5, -1):
        params[idx].register_hook(
            lambda g, i=idx: grads.__setitem__(i, g.detach().clone() if g is not None else None)
        )

    ref_before = {i: params[i].detach().clone() for i in (0, 5, -1)}

    em, recs = C.run_one_rollout(
        model, args, pdb_file, pocket_ids, ppo_config, optimizer, policy_update=True
    )

    print("loss:", em.get("loss"), "approx_kl:", em.get("approx_kl"))
    for i in (0, 5, -1):
        g = grads.get(i)
        print(
            f"param[{i}] grad is None={g is None}, grad_abs_sum="
            f"{None if g is None else float(g.abs().sum())}, "
            f"weight_changed={not torch.allclose(ref_before[i], params[i].detach())}"
        )


if __name__ == "__main__":
    main()
