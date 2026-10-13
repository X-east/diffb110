"""Shared helpers for DePPA PPO correctness tests (A/B/C).

Run with the ``deppa_ref`` conda environment from this directory.
All paths are absolute so the scripts can be launched from anywhere.
"""

import os

# Must be set before torch is imported anywhere in the process.
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "1")
os.environ["PATH"] = (
    "/home/md666/miniforge3/envs/diffgui/bin:" + os.environ.get("PATH", "")
)
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["NUMEXPR_THREADS"] = "4"

import sys
from pathlib import Path
from types import SimpleNamespace
from functools import partial

ROOT = Path("/data/disk6/DiffB110")
DEPPA_DIR = ROOT / "reference project/代码/DePPA"
sys.path.insert(0, str(DEPPA_DIR))

import torch  # noqa: E402
import numpy as np  # noqa: E402
import utils  # noqa: E402
from lightning_modules import LigandPocketDDPM  # noqa: E402

# Optional deterministic mode for verifying resume bit-exactness (set
# DETERMINISTIC=1). warn_only avoids hard errors on unsupported atomics.
if os.environ.get("DETERMINISTIC") == "1":
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

PRETRAINED = str(ROOT / "DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt")
TARGET_DIR = ROOT / "analysis/deppa_8gjl_single_gpu_smoke_20260720/target"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_pocket(target_dir=TARGET_DIR):
    """Return (pdb_file, receptor_file, ref_ligand, pocket_ids) for 8GJL."""
    target_dir = Path(target_dir)
    pdb_file = target_dir / "8GJL.pdb"
    receptor_file = target_dir / "8GJL.pdbqt"
    sdf_candidates = sorted(target_dir.glob("8GJL*.sdf"))
    if not sdf_candidates:
        raise FileNotFoundError(f"No 8GJL reference sdf in {target_dir}")
    ref_ligand = sdf_candidates[0]
    txt_file = ref_ligand.with_suffix(".txt")
    pocket_ids = txt_file.read_text().split()
    return pdb_file, receptor_file, ref_ligand, pocket_ids


def load_reference_ligand(ref_ligand, pdb_file):
    """Return (coords, mean_coord, n_atoms) for the reference ligand."""
    from rdkit import Chem

    rdmol = Chem.SDMolSupplier(str(ref_ligand))[0]
    coords = torch.from_numpy(rdmol.GetConformer().GetPositions()).float()
    return coords, coords.mean(axis=0), coords.shape[0]


def make_args(**overrides):
    defaults = dict(
        n_samples=4,
        rollouts=2,
        timesteps=100,
        inference_interval=5,
        ppo_batch_size=32,
        ppo_lr=1e-5,
        clip_range=0.2,
        max_grad_norm=0.5,
        kl_coeff_pretrain=0.0,
        reward_workers=4,
        sanitize=True,
        w_qed=0.2,
        w_sa=0.2,
        w_vina_score=0.5,
        w_distance=0.1,
        w_strain=0.0,
        mode_num_nodes_lig="sample",
        all_frags=False,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def build_ppo_config(model, pdb_file, receptor_file, ref_ligand, args):
    """Build the ppo_config namespace used by generate_ligands_rl."""
    _, mean_coord, _ = load_reference_ligand(ref_ligand, pdb_file)
    vina_center_xyz = mean_coord
    reward_fn_dict = {
        "qed": model.molecule_properties.calculate_qed,
        "sa": model.molecule_properties.calculate_sa,
        "vina_score": partial(
            model.molecule_properties.calculate_docking_score,
            center_xyz=vina_center_xyz,
            receptor_pdbqt_file=str(receptor_file),
            use_meeko=False,
            vina_mode="vina_score",
        ),
    }
    reward_weights = {
        "qed": args.w_qed,
        "sa": args.w_sa,
        "vina_score": args.w_vina_score,
        "distance": args.w_distance,
        "strain": args.w_strain,
    }
    actual_timesteps = model.T if args.timesteps is None else args.timesteps
    ppo_config = SimpleNamespace(
        clip_range=args.clip_range,
        max_grad_norm=args.max_grad_norm,
        max_time_steps=actual_timesteps,
        inference_interval=args.inference_interval,
        batch_size=args.ppo_batch_size,
        lr=args.ppo_lr,
        reward_fn_dict=reward_fn_dict,
        kl_coeff_pretrain=args.kl_coeff_pretrain,
        reward_num_workers=args.reward_workers,
        reward_weights=reward_weights,
    )
    actual_timesteps = model.T if args.timesteps is None else args.timesteps
    ppo_config.episode_length = actual_timesteps // args.inference_interval
    return ppo_config


def new_optimizer(model, lr):
    return torch.optim.AdamW(
        model.ddpm.parameters(),
        lr=lr,
        amsgrad=True,
        weight_decay=1e-4,
    )


def load_model(pretrained=PRETRAINED):
    model = LigandPocketDDPM.load_from_checkpoint(pretrained, map_location=DEVICE)
    model = model.to(DEVICE)
    model.eval()
    return model


def run_one_rollout(
    model, args, pdb_file, pocket_ids, ppo_config, optimizer, policy_update=True
):
    return model.generate_ligands_rl(
        str(pdb_file),
        args.n_samples,
        pocket_ids=pocket_ids,
        num_nodes_lig=None,
        sanitize=args.sanitize,
        largest_frag=not args.all_frags,
        timesteps=args.timesteps,
        ppo_config=ppo_config,
        optimizer=optimizer,
        policy_update=policy_update,
    )


def optimizer_step_count(optimizer):
    """Number of optimizer.step() calls (all params share the same step)."""
    steps = []
    for pg in optimizer.param_groups:
        for p in pg["params"]:
            st = optimizer.state.get(p)
            if st is not None and "step" in st:
                step = st["step"]
                steps.append(int(step) if not isinstance(step, torch.Tensor) else int(step.item()))
    # All params are updated every step, so the max equals the call count.
    return max(steps) if steps else 0


def optimizer_states_equal(opt_a, opt_b, tol=1e-6):
    """Compare AdamW per-parameter state (exp_avg/sq/step) for exact restore."""
    sa = opt_a.state_dict()["state"]
    sb = opt_b.state_dict()["state"]
    if set(sa.keys()) != set(sb.keys()):
        return False
    for k in sa:
        for field in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq", "step"):
            va, vb = sa[k].get(field), sb[k].get(field)
            if va is None and vb is None:
                continue
            if va is None or vb is None:
                return False
            if isinstance(va, torch.Tensor):
                if not torch.allclose(va.float(), vb.float(), atol=tol):
                    return False
            elif va != vb:
                return False
    return True


def resolved_config(args):
    return {
        k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
    }


def state_dict_diff(a_sd, b_sd):
    """Sum of absolute per-tensor max diffs between two state_dicts."""
    return sum(float((a_sd[k] - b_sd[k]).abs().max()) for k in a_sd)


def set_seed(seed):
    """Deterministic seeding across Python / NumPy / Torch (CPU + CUDA)."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
