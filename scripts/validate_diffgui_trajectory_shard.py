"""Validate a saved DiffGui trajectory shard with a real PPO optimizer step."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIFFGUI_ROOT = PROJECT_ROOT / "reference project" / "代码" / "DiffGui"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(DIFFGUI_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFGUI_ROOT))

from ddpo.diffgui.runtime.actor import DiffGuiActor, DiffGuiPolicyBatch  # noqa: E402
from ddpo.diffgui.runtime.logprob_adapter import diffgui_forward_step_logprob  # noqa: E402
from scripts.diffgui_docking_batch_sweep import _load_diffgui_model, _resolve_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sample-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-graphs", type=int, default=1)
    parser.add_argument("--graph-micro-batch-size", type=int, default=1)
    parser.add_argument("--num-timesteps", type=int, default=200)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--consistency-only", action="store_true")
    parser.add_argument("--mock-reward-probe", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Load one shard, replace placeholder advantages, and run one PPO update."""
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The real DiffGui trainability probe requires CUDA.")
    payload = torch.load(args.shard, map_location="cpu", weights_only=False)
    batch = payload.get("policy_batch")
    if not isinstance(batch, DiffGuiPolicyBatch):
        raise TypeError("Shard policy_batch is not DiffGuiPolicyBatch.")
    batch.validate()
    graph_count = min(max(1, int(args.num_graphs)), int(batch.advantages.shape[0]))
    if graph_count < int(batch.advantages.shape[0]):
        batch = batch.select_graphs(list(range(graph_count)))
    mock_rewards: torch.Tensor | None = None
    if args.mock_reward_probe:
        if graph_count < 2:
            raise ValueError("--mock-reward-probe requires at least two graphs.")
        mock_rewards = torch.linspace(-1.0, 1.0, steps=graph_count, dtype=torch.float32)
        batch.advantages = (mock_rewards - mock_rewards.mean()) / mock_rewards.std(unbiased=False).clamp_min(1.0e-6)
    else:
        batch.advantages = torch.linspace(1.0, 0.5, steps=graph_count, dtype=torch.float32)

    model, _, _ = _load_diffgui_model(
        sample_config=_resolve_path(args.sample_config),
        checkpoint=_resolve_path(args.checkpoint),
        num_timesteps=int(args.num_timesteps),
        device=device,
    )
    if args.consistency_only:
        result = _run_consistency_probe(
            model=model,
            batch=batch,
            device=device,
            shard=args.shard,
            checkpoint=args.checkpoint,
            payload=payload,
        )
        _write_result(args.output, result)
        return

    actor = DiffGuiActor(
        model,
        {
            "train": {
                "learning_rate": 1.0e-6,
                "adam_beta1": 0.9,
                "adam_beta2": 0.999,
                "adam_weight_decay": 0.0,
                "adam_epsilon": 1.0e-8,
                "clip_range": 0.2,
                "max_grad_norm": 1.0,
                "ppo_epochs": 1,
                "diffgui_logprob_reduction": "mean",
                "precision": "fp32",
                "graph_micro_batch_size": int(args.graph_micro_batch_size),
                "freeze": {"mode": "none"},
            }
        },
    )
    before = [parameter.detach().clone() for parameter in model.parameters() if parameter.requires_grad]
    metrics = actor.update_policy(batch)
    squared_delta = 0.0
    for previous, current in zip(before, (p for p in model.parameters() if p.requires_grad)):
        squared_delta += float((current.detach() - previous).float().square().sum().item())
    parameter_delta_l2 = math.sqrt(squared_delta)
    required_metrics = ("PPO_Loss", "PPO_ApproxKL", "ClipFrac", "RatioMean", "GradNorm")
    finite_metrics = all(math.isfinite(float(metrics[key])) for key in required_metrics)
    result: dict[str, Any] = {
        "status": "passed" if finite_metrics and parameter_delta_l2 > 0.0 else "failed",
        "shard": str(args.shard.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "format": payload.get("format", ""),
        "graphs_tested": graph_count,
        "graph_micro_batch_size": int(args.graph_micro_batch_size),
        "num_timesteps": int(batch.old_logprob.shape[0]),
        "state_steps": int(batch.node_traj.shape[0]),
        "old_logprob_shape": list(batch.old_logprob.shape),
        "finite_metrics": finite_metrics,
        "parameter_delta_l2": parameter_delta_l2,
        "mock_rewards": None if mock_rewards is None else mock_rewards.tolist(),
        "assigned_advantages": batch.advantages.tolist(),
        "metrics": {key: float(value) for key, value in metrics.items()},
    }
    _write_result(args.output, result)


def _run_consistency_probe(
    *,
    model: torch.nn.Module,
    batch: DiffGuiPolicyBatch,
    device: torch.device,
    shard: Path,
    checkpoint: Path,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Recompute old-policy log-probabilities without gradients or batch slicing."""
    model.eval()
    batch = batch.to(device)
    logratios: list[torch.Tensor] = []
    ratios: list[torch.Tensor] = []
    with torch.no_grad():
        for step_index in range(int(batch.old_logprob.shape[0])):
            step = diffgui_forward_step_logprob(
                model,
                protein_node=batch.protein_node,
                protein_pos=batch.protein_pos,
                protein_batch=batch.protein_batch,
                ligand_node_t=batch.node_traj[step_index],
                ligand_pos_t=batch.pos_centered_traj[step_index],
                ligand_batch=batch.ligand_batch,
                ligand_halfedge_t=batch.halfedge_traj[step_index],
                halfedge_index=batch.halfedge_index,
                halfedge_batch=batch.halfedge_batch,
                node_log_state_t=batch.node_log_state_traj[step_index],
                halfedge_log_state_t=batch.halfedge_log_state_traj[step_index],
                ligand_pos_prev=batch.pos_centered_traj[step_index + 1],
                ligand_node_prev=batch.node_traj[step_index + 1],
                ligand_halfedge_prev=batch.halfedge_traj[step_index + 1],
                time_step=batch.timesteps[step_index],
                batch_lab=batch.batch_lab,
                gui_strength=batch.gui_strength,
                reduction="mean",
            )
            logratio = step.logprob_total - batch.old_logprob[step_index]
            logratios.append(logratio.detach().cpu())
            ratios.append(torch.exp(logratio).detach().cpu())
    all_logratios = torch.cat(logratios)
    all_ratios = torch.cat(ratios)
    approx_kl = float((0.5 * all_logratios.square()).mean().item())
    max_abs_logratio = float(all_logratios.abs().max().item())
    finite = bool(torch.isfinite(all_logratios).all().item())
    result = {
        "status": "passed" if finite and approx_kl < 1.0e-6 else "failed",
        "probe": "old_policy_consistency_no_grad",
        "shard": str(shard.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "format": payload.get("format", ""),
        "graphs_tested": int(batch.advantages.shape[0]),
        "num_timesteps": int(batch.old_logprob.shape[0]),
        "state_steps": int(batch.node_traj.shape[0]),
        "finite_logratios": finite,
        "approx_kl": approx_kl,
        "ratio_mean": float(all_ratios.mean().item()),
        "clipfrac_0_2": float((torch.abs(all_ratios - 1.0) > 0.2).float().mean().item()),
        "max_abs_logratio": max_abs_logratio,
    }
    return result


def _write_result(path: Path, result: dict[str, Any]) -> None:
    """Persist a probe result and fail the command when validation fails."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] != "passed":
        raise RuntimeError("DiffGui trajectory validation probe failed.")


if __name__ == "__main__":
    main()
