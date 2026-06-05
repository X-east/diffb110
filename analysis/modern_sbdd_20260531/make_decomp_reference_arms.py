from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import torch


def _inject_local_packages(repo: Path) -> None:
    for name in ("utils", "datasets", "models"):
        module = types.ModuleType(name)
        module.__path__ = [str(repo / name)]
        sys.modules[name] = module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data-id", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))
    _inject_local_packages(repo)

    from datasets.pl_arm_sim_dataset import get_decomp_dataset
    from utils import misc

    config = misc.load_config(args.config)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    ckpt["config"].data.version = config.data.version
    ckpt["config"].data.path = config.data.path
    ckpt["config"].data.split = config.data.split
    ckpt["config"].data.transform.add_retrieved_feat = True
    ckpt["config"].data.transform.top_k_sim = max(args.top_k, 1)

    _, subsets = get_decomp_dataset(config=ckpt["config"].data, transform=None)
    data = subsets["test"][args.data_id]
    reference_arms = []
    for arm_group in data.sim_arms:
        saved_group = []
        for arm in arm_group[: args.top_k]:
            saved_group.append({"arm": arm.clone(), "z_score": 0.0})
        reference_arms.append(saved_group)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(reference_arms, out)
    print(f"saved={out} num_arms={len(reference_arms)} top_k={args.top_k}")


if __name__ == "__main__":
    main()
