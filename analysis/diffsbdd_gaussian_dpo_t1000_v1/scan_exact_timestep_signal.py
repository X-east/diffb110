"""Scan exact Gaussian preference signal across non-degenerate T=1000 steps."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ddpo.diffusion_dpo.pairs import build_exact_atom_count_pairs, load_selected_manifest
from ddpo.diffusion_dpo.scorer import DiffSBDDGaussianTransitionScorer
from ddpo.diffusion_dpo.trajectory import TrajectoryStore, collate_transitions


def main() -> None:
    """Evaluate reference pair ordering at fixed valid timesteps."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pocket", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--pair-count", type=int, default=64)
    parser.add_argument("--timesteps", type=int, nargs="+", default=[999, 751, 503, 255, 3])
    parser.add_argument("--batch-pairs", type=int, default=8)
    args = parser.parse_args()
    frame = load_selected_manifest(args.manifest)
    pairs = build_exact_atom_count_pairs(frame, "validation", count=args.pair_count, seed=2)
    scorer = DiffSBDDGaussianTransitionScorer(
        str(args.checkpoint), str(args.pocket), "C:1101", device=args.device, num_timesteps=1000
    )
    store = TrajectoryStore(expected_timesteps=1000)
    valid_timesteps = set(scorer.valid_transition_timesteps())
    rows: list[dict[str, float]] = []
    for timestep in args.timesteps:
        if timestep not in valid_timesteps:
            raise ValueError(f"timestep {timestep} is degenerate under the native schedule")
        margins: list[float] = []
        for start in range(0, len(pairs), args.batch_pairs):
            batch_pairs = pairs[start : start + args.batch_pairs]
            transitions = []
            for pair in batch_pairs:
                transitions.extend(
                    [
                        store.sample(pair.winner_shard, pair.winner_index).transition(1000 - timestep),
                        store.sample(pair.loser_shard, pair.loser_index).transition(1000 - timestep),
                    ]
                )
            batch = collate_transitions(transitions, device=args.device)
            with torch.no_grad():
                score = scorer.score(
                    batch["z_t"], batch["z_s"], batch["t"], batch["node_mask"], reduction="sum"
                ).total.reshape(-1, 2)
            margins.extend((score[:, 0] - score[:, 1]).cpu().tolist())
        values = np.asarray(margins, dtype=float)
        rows.append(
            {
                "timestep_t": timestep,
                "pairs": len(values),
                "pair_accuracy": float(np.mean(values > 0.0)),
                "mean_margin": float(np.mean(values)),
                "std_margin": float(np.std(values)),
                "finite": float(np.isfinite(values).all()),
            }
        )
    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
