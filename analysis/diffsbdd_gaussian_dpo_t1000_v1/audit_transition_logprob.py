"""Audit Gaussian transition log-probability parity against the DiffSBDD adapter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ddpo.diffusion_dpo.logprob import gaussian_transition_score
from ddpo.diffusion_dpo.scorer import DiffSBDDGaussianTransitionScorer
from ddpo.diffusion_dpo.trajectory import TrajectoryStore, collate_transitions


def main() -> None:
    """Compare simplified adapter mode and the new pure Gaussian function."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pocket", type=Path, required=True)
    parser.add_argument("--ref-ligand", default="C:1101")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--step", type=int, default=500)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.manifest, compression="gzip", low_memory=False)
    frame = frame[frame["extreme_label"].isin(["good", "hard_bad"])].head(args.rows)
    scorer = DiffSBDDGaussianTransitionScorer(
        str(args.checkpoint),
        str(args.pocket),
        args.ref_ligand,
        device=args.device,
        num_timesteps=1000,
    )
    store = TrajectoryStore(expected_timesteps=1000)
    transitions = [
        store.sample(row.trajectory_shard, int(row.mol_index)).transition(args.step)
        for row in frame.itertuples(index=False)
    ]
    batch = collate_transitions(transitions, device=args.device)
    stats = scorer.reverse_statistics(batch["z_t"], batch["t"], batch["node_mask"])
    ours = gaussian_transition_score(
        batch["z_s"],
        stats.mean,
        stats.sigma,
        batch["node_mask"],
        reduction="mean_per_atom",
        include_normal_constant=False,
        include_sigma_term=False,
    ).total
    adapter = scorer.model.compute_log_p_zs_given_zt(
        batch["z_s"], stats.mean, stats.sigma, batch["node_mask"]
    )
    exact = gaussian_transition_score(
        batch["z_s"],
        stats.mean,
        stats.sigma,
        batch["node_mask"],
        reduction="sum",
        include_normal_constant=True,
        include_sigma_term=True,
    ).total
    delta = (ours - adapter).detach().cpu()
    payload = {
        "rows": len(transitions),
        "step_index": args.step,
        "timestep_t": int(batch["t"][0].item()),
        "adapter_mode": scorer.model.logprob_mode,
        "max_abs_parity_error": float(delta.abs().max()),
        "mean_abs_parity_error": float(delta.abs().mean()),
        "exact_score_finite": bool(torch.isfinite(exact).all()),
        "exact_score_min": float(exact.min().detach().cpu()),
        "exact_score_max": float(exact.max().detach().cpu()),
        "z_t_finite": bool(torch.isfinite(batch["z_t"]).all()),
        "z_s_finite": bool(torch.isfinite(batch["z_s"]).all()),
        "mean_finite": bool(torch.isfinite(stats.mean).all()),
        "sigma_finite": bool(torch.isfinite(stats.sigma).all()),
        "z_t_abs_max": float(batch["z_t"].abs().max().detach().cpu()),
        "z_s_abs_max": float(batch["z_s"].abs().max().detach().cpu()),
        "mean_abs_max": float(stats.mean.abs().max().detach().cpu()),
        "sigma_abs_max": float(stats.sigma.abs().max().detach().cpu()),
    }
    (args.output_dir / "transition_logprob_parity.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    if payload["max_abs_parity_error"] > 1e-5:
        raise RuntimeError(f"transition log-prob parity failed: {payload}")


if __name__ == "__main__":
    main()
