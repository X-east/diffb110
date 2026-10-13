"""Prove a collected DiffSBDD trajectory shard can drive the real RL pipeline.

The probe deliberately avoids bespoke glue: it rebuilds the batch through
``ddpo.runtime.trajectory_shard``, checks it against the replay-buffer format
contract, and then calls ``ddpo.runtime.train_helpers.update_policy`` -- the
exact function the DDPO training loop invokes each epoch.  A run counts as
passing only when

1. the rebuilt batch satisfies the RL/replay tensor + meta_info contract,
2. every PPO step returns finite ``PPO_Loss`` / ``ClipFrac``,
3. the model parameters actually move (``parameter_delta_l2 > 0``), and
4. the parameter drift keeps growing across successive PPO steps.

Usage:
  source /home/md666/miniforge3/etc/profile.d/conda.sh && conda activate dif && \
  python scripts/validate_diffsbdd_trajectory_shard.py \
    --shard runs/diffsbdd_collect/<run>/shards/rank_00/shard_0000.pt \
    --checkpoint ./DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt \
    --pocket-pdb ./data/8GJL.pdb --ref-ligand "C:1101" \
    --num-graphs 2 --ppo-steps 2 --device cuda:0 \
    --output /tmp/diffsbdd_shard_validate.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ddpo.integration.diffsbdd_model import DiffSBDDModel  # noqa: E402
from ddpo.runtime.actor import EDMActor  # noqa: E402
from ddpo.runtime.protocol import DataProto  # noqa: E402
from ddpo.runtime.replay import _select_tensor_training_view  # noqa: E402
from ddpo.runtime.train_helpers import PreparedSamples, update_policy  # noqa: E402
from ddpo.runtime.trajectory_shard import (  # noqa: E402
    assert_rl_pipeline_compatible,
    build_training_dataproto,
    load_shard_payload,
)


LOGGER = logging.getLogger("DiffSBDDShardProbe")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pocket-pdb", type=Path, required=True)
    parser.add_argument("--ref-ligand", default="C:1101")
    parser.add_argument("--dataset-name", default="crossdock")
    parser.add_argument("--num-graphs", type=int, default=1)
    parser.add_argument("--ppo-steps", type=int, default=2, help="Successive update_policy calls.")
    parser.add_argument("--learning-rate", type=float, default=1.0e-6)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _mock_advantages(count: int) -> torch.Tensor:
    """Return a zero-mean unit-variance advantage signal for ``count`` molecules."""
    if count == 1:
        return torch.tensor([1.0], dtype=torch.float32)
    values = torch.linspace(-1.0, 1.0, steps=count, dtype=torch.float32)
    return (values - values.mean()) / values.std(unbiased=False).clamp_min(1.0e-6)


def _parameter_snapshot(model: torch.nn.Module) -> list[torch.Tensor]:
    """Clone all trainable parameters for later drift measurement."""
    return [p.detach().clone() for p in model.parameters() if p.requires_grad]


def _parameter_delta_l2(model: torch.nn.Module, reference: list[torch.Tensor]) -> float:
    """Return the L2 norm of the parameter drift versus a snapshot."""
    squared = 0.0
    trainable = (p for p in model.parameters() if p.requires_grad)
    for previous, current in zip(reference, trainable):
        squared += float((current.detach() - previous).float().square().sum().item())
    return math.sqrt(squared)


def _finite_metrics(metrics: dict[str, object]) -> dict[str, float]:
    """Keep only finite scalar metrics from an actor metrics dict."""
    finite: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            finite[str(key)] = float(value)
    return finite


def main() -> None:
    """Run the shard -> RL pipeline trainability probe."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The real DiffSBDD trainability probe requires CUDA.")

    payload = load_shard_payload(args.shard)
    metadata = payload["metadata"]
    num_timesteps = int(metadata["num_timesteps"])
    shard_batch_size = int(payload["latents"].shape[0])
    graph_count = min(max(1, int(args.num_graphs)), shard_batch_size)
    advantages = _mock_advantages(graph_count)

    # Step 1: rebuild the batch exactly like the online RL path expects.
    data = build_training_dataproto(
        payload,
        advantages=advantages,
        indices=list(range(graph_count)),
        device="cpu",
    )
    contract_report = assert_rl_pipeline_compatible(data)

    # Step 2: prove the replay buffer can also slice/serialize this batch.
    replay_view: DataProto = _select_tensor_training_view(data)
    replay_keys = sorted(str(key) for key in replay_view.batch.keys())
    LOGGER.info("replay training view keys=%s size=%d", replay_keys, len(replay_view))

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
            "learning_rate": float(args.learning_rate),
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

    # Step 3: drive the *real* training helper, not the actor directly.
    runtime = SimpleNamespace(logger=LOGGER, actor=actor, max_epochs=int(args.ppo_steps))
    baseline = _parameter_snapshot(model)
    steps: list[dict[str, object]] = []
    for step in range(1, max(1, int(args.ppo_steps)) + 1):
        prepared = PreparedSamples(samples=data, filtering_ratio=0.0, novelty_penalty_ratio=0.0)
        metrics = update_policy(runtime, prepared, epoch=step)
        finite = _finite_metrics(metrics)
        steps.append(
            {
                "step": step,
                "parameter_delta_l2": _parameter_delta_l2(model, baseline),
                "has_ppo_loss": "PPO_Loss" in finite,
                "has_clip_frac": "ClipFrac" in finite,
                "metrics": finite,
            }
        )

    deltas = [float(entry["parameter_delta_l2"]) for entry in steps]
    metrics_finite = all(bool(entry["has_ppo_loss"]) and bool(entry["has_clip_frac"]) for entry in steps)
    parameters_moved = deltas[-1] > 0.0
    drift_monotonic = all(later >= earlier for earlier, later in zip(deltas, deltas[1:]))
    status = "passed" if (metrics_finite and parameters_moved and drift_monotonic) else "failed"

    result = {
        "status": status,
        "shard": str(args.shard.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "format": payload.get("format", ""),
        "num_timesteps": num_timesteps,
        "shard_batch_size": shard_batch_size,
        "graphs_tested": graph_count,
        "ppo_steps": len(steps),
        "rl_format_contract": contract_report,
        "replay_training_view_keys": replay_keys,
        "checks": {
            "metrics_finite": metrics_finite,
            "parameters_moved": parameters_moved,
            "drift_monotonic": drift_monotonic,
        },
        "parameter_delta_l2": deltas[-1],
        "parameter_delta_l2_per_step": deltas,
        "mock_advantages": advantages.tolist(),
        "steps": steps,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if status != "passed":
        raise RuntimeError("DiffSBDD trajectory validation probe failed.")


if __name__ == "__main__":
    main()
