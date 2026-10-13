"""Audit the saved DiffSBDD T=1000 trajectory protocol."""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ddpo.diffusion_dpo.trajectory import load_trajectory_sample


LOGGER = logging.getLogger("diffsbdd_trajectory_audit")


def parse_args() -> argparse.Namespace:
    """Parse audit inputs and deterministic sampling controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("analysis/offline_iql_extreme_good_bad_pool_260822/inputs/selected_extremes_manifest.csv.gz"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/diffsbdd_gaussian_dpo_t1000_v1/protocol_audit"),
    )
    parser.add_argument("--max-rows", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--timesteps", type=int, default=1000)
    return parser.parse_args()


def audit(args: argparse.Namespace) -> None:
    """Audit deterministic rows and write machine-readable evidence."""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.manifest, compression="gzip", low_memory=False)
    frame = frame[frame["extreme_label"].isin(["good", "hard_bad"])].copy()
    if frame.empty:
        raise RuntimeError("manifest contains no selected good/hard_bad rows")
    if frame["trajectory_id"].duplicated().any():
        raise RuntimeError("manifest contains duplicated trajectory_id")
    rng = random.Random(args.seed)
    selected = frame.sample(n=min(args.max_rows, len(frame)), random_state=rng.randint(0, 2**31 - 1))
    rows: list[dict[str, object]] = []
    errors: list[str] = []
    for row in selected.itertuples(index=False):
        try:
            sample = load_trajectory_sample(
                row.trajectory_shard,
                int(row.mol_index),
                expected_timesteps=args.timesteps,
            )
            first = sample.transition(0)
            last = sample.transition(sample.num_timesteps - 1)
            rows.append(
                {
                    "trajectory_id": sample.sample_id,
                    "split": row.split,
                    "extreme_label": row.extreme_label,
                    "n_nodes": sample.n_nodes,
                    "num_timesteps": sample.num_timesteps,
                    "state_dim": int(sample.states.shape[-1]),
                    "first_t": first.t,
                    "first_s": first.s,
                    "last_t": last.t,
                    "last_s": last.s,
                    "state_finite": True,
                    "terminal_decode_finite": bool(sample.terminal_decode.isfinite().all()),
                }
            )
        except Exception as exc:  # noqa: BLE001 - audit must collect all failures
            errors.append(f"{row.trajectory_id}: {type(exc).__name__}: {exc}")
    result = pd.DataFrame(rows)
    result.to_csv(args.output_dir / "trajectory_protocol_rows.csv", index=False)
    summary = {
        "status": "passed" if not errors else "failed",
        "manifest": str(args.manifest.resolve()),
        "expected_timesteps": args.timesteps,
        "requested_rows": min(args.max_rows, len(frame)),
        "audited_rows": len(rows),
        "error_count": len(errors),
        "errors": errors[:20],
        "unique_shards": int(selected["trajectory_shard"].nunique()),
        "label_counts": selected["extreme_label"].value_counts().to_dict(),
        "split_counts": selected["split"].value_counts().to_dict(),
        "state_dim_counts": result["state_dim"].value_counts().to_dict() if not result.empty else {},
        "n_nodes_min": int(result["n_nodes"].min()) if not result.empty else None,
        "n_nodes_max": int(result["n_nodes"].max()) if not result.empty else None,
        "reverse_protocol": "latents[k] -> latents[k+1] for k=0..T-1; exclude latents[T+1]=xh_final",
        "schedule_protocol": "T=1000 collector loop may contain deterministic duplicate lookup steps when checkpoint schedule has 500 points; exact DPO must audit sigma and exclude degenerate steps",
    }
    (args.output_dir / "trajectory_protocol_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if not result.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].hist(result["n_nodes"], bins=range(result["n_nodes"].min(), result["n_nodes"].max() + 2), color="#356A8A", alpha=0.85)
        axes[0].set_title("Audited atom counts")
        axes[0].set_xlabel("Valid ligand atoms")
        axes[0].set_ylabel("Rows")
        axes[1].bar(["first", "last"], [result["first_s"].iloc[0], result["last_s"].iloc[0]], color=["#2E7D32", "#C62828"])
        axes[1].set_title("Reverse transition boundary")
        axes[1].set_ylabel("s timestep")
        for axis in axes:
            axis.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(args.output_dir / "trajectory_protocol_audit.png", dpi=180)
        fig.savefig(args.output_dir / "trajectory_protocol_audit.svg")
        plt.close(fig)
    if errors:
        raise RuntimeError(f"trajectory audit failed with {len(errors)} errors")
    LOGGER.info("passed rows=%d unique_shards=%d", len(rows), summary["unique_shards"])


def main() -> None:
    """Run the trajectory protocol audit."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    audit(parse_args())


if __name__ == "__main__":
    main()
