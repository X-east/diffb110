"""Validate a saved DiffSBDD trajectory shard with a real PPO optimizer step.

Loads one frozen-policy DiffSBDD shard, assigns mock advantages, runs a single
``EDMActor.update_policy`` pass, and asserts the metrics are finite and the
model parameters actually moved.  This proves the stored ``latents`` /
``logps`` / ``timesteps`` / ``nodesxsample`` tensors are consumable by the
same PPO code path as online RL rollouts.

Usage:
  bash -c 'source /home/md666/miniforge3/etc/profile.d/conda.sh && conda activate dif' && \
  python scripts/validate_diffsbdd_trajectory_shard.py \
    --shard runs/diffsbdd_collect/<run>/shards/rank_00/shard_0000.pt \
    --checkpoint ./DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt \
    --pocket-pdb ./data/8GJL.pdb --ref-ligand "C:1101" \
    --num-graphs 2 --device cuda:0 --output /tmp/diffsbdd_shard_validate.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ddpo.integration.diffsbdd_model import DiffSBDDModel  # noqa: E402
from ddpo.runtime.actor import EDMActor  # noqa: E402
from ddpo.runtime.protocol import DataProto, TensorDict  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pocket-pdb", type=Path, required=True)
    parser.add_argument("--ref-ligand", default="C:1101")
    parser.add_argument("--dataset-name", default="crossdock")
    parser.add_argument("--num-graphs", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The real DiffSBDD trainability probe requires CUDA.")

    payload = torch.load(args.shard, map_location="cpu", weights_only=False)
    if payload.get("format") != "diffsbdd_frozen_policy_trajectory_v1":
        raise TypeError(f"Unexpected shard format: {payload.get('format')!r}")
    metadata = payload["metadata"]
    num_timesteps = int(metadata["num_timesteps"])

    latents = payload["latents"]
    logps = payload["logps"]
    timesteps = payload["timesteps"]
    nodesxsample = payload["nodesxsample"]
    batch_size = int(latents.shape[0])

    graph_count = min(max(1, int(args.num_graphs)), batch_size)
    idx = list(range(graph_count))
    latents = latents[idx]
    logps = logps[idx]
    timesteps = timesteps[idx]
    nodesxsample = nodesxsample[idx]

    # Mock advantages: normalized linear signal so PPO has a non-trivial target.
    mock = torch.linspace(-1.0, 1.0, steps=graph_count, dtype=torch.float32)
    advantages = (mock - mock.mean()) / mock.std(unbiased=False).clamp_min(1.0e-6)

    data = DataProto(
        batch=TensorDict(
            {
                "latents": latents,
                "logps": logps,
                "timesteps": timesteps,
                "nodesxsample": nodesxsample,
                "advantages": advantages,
            },
            batch_size=[graph_count],
        ),
        meta_info={"max_n_nodes": int(metadata["max_n_nodes"]), "condition": False},
    )

    model = DiffSBDDModel(
        checkpoint_path=str(args.checkpoint),
        pocket_pdb_path=str(args.pocket_pdb),
        ref_ligand_path=str(args.ref_ligand),
        device=str(device),
        dataset_name=str(args.dataset_name),
    )
    model.eval()

    config = {
        "train": {
            "learning_rate": 1.0e-6,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_weight_decay": 0.0,
            "adam_epsilon": 1.0e-8,
            "clip_range": 0.2,
            "max_grad_norm": 1.0,
            "ppo_epochs": 1,
            "train_micro_batch_size": max(1, graph_count),
            "gradient_reference_batch_size": max(1, graph_count),
        },
        "model": {"time_step": num_timesteps},
    }
    actor = EDMActor(model, config)
    before = [p.detach().clone() for p in model.parameters() if p.requires_grad]
    metrics = actor.update_policy(data)
    squared_delta = 0.0
    for previous, current in zip(before, (p for p in model.parameters() if p.requires_grad)):
        squared_delta += float((current.detach() - previous).float().square().sum().item())
    parameter_delta_l2 = math.sqrt(squared_delta)

    # EDMActor emits a subset of metrics; require the ones it always produces
    # (PPO_Loss / ClipFrac) to be finite and the model parameters to actually
    # move.  RatioMean / GradNorm are not emitted by this actor variant.
    produced = {
        k: float(v)
        for k, v in metrics.items()
        if isinstance(v, (int, float)) and math.isfinite(float(v))
    }
    finite_metrics = "PPO_Loss" in produced and "ClipFrac" in produced
    status = "passed" if (finite_metrics and parameter_delta_l2 > 0.0) else "failed"

    result = {
        "status": status,
        "shard": str(args.shard.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "format": payload.get("format", ""),
        "num_timesteps": num_timesteps,
        "graphs_tested": graph_count,
        "latent_shape": list(latents.shape),
        "logps_shape": list(logps.shape),
        "timesteps_shape": list(timesteps.shape),
        "finite_metrics": finite_metrics,
        "parameter_delta_l2": parameter_delta_l2,
        "mock_advantages": advantages.tolist(),
        "metrics": {k: float(v) for k, v in metrics.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if status != "passed":
        raise RuntimeError("DiffSBDD trajectory validation probe failed.")


if __name__ == "__main__":
    main()
