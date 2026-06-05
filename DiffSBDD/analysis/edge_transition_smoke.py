"""Smoke-check categorical halfedge transitions on an edge bundle."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIFFSBDD_ROOT = Path(__file__).resolve().parents[1]
for extra in (PROJECT_ROOT, DIFFSBDD_ROOT):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from edge_diffusion.transitions import EdgeCategoricalTransition, EdgeTransitionConfig
from edge_utils import BOND_TYPE_NAMES


def _to_object_list(values: np.ndarray) -> list[np.ndarray]:
    """Convert an object array from ``np.load`` into a plain list."""
    return [np.asarray(item) for item in values.tolist()]


def _class_counts(edge_type: torch.Tensor, num_classes: int) -> dict[str, int]:
    """Return label counts keyed by readable bond names."""
    counts = Counter(int(x) for x in edge_type.cpu().tolist())
    return {
        BOND_TYPE_NAMES.get(idx, str(idx)): int(counts.get(idx, 0))
        for idx in range(num_classes)
    }


def run_smoke(args: argparse.Namespace) -> Path:
    """Run transition shape and noising checks, then write a Markdown report."""
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    with np.load(args.edge_npz, allow_pickle=True) as data:
        halfedge_arrays = _to_object_list(data["ligand_halfedge_type"])

    max_molecules = min(args.max_molecules, len(halfedge_arrays))
    selected = [torch.as_tensor(arr, dtype=torch.long) for arr in halfedge_arrays[:max_molecules]]
    edge_type = torch.cat(selected, dim=0) if selected else torch.zeros(0, dtype=torch.long)

    transition = EdgeCategoricalTransition(
        EdgeTransitionConfig(
            num_classes=args.num_classes,
            num_timesteps=args.num_timesteps,
        )
    )
    generator = torch.Generator(device=edge_type.device).manual_seed(args.seed)
    timesteps = [0, args.num_timesteps // 2, args.num_timesteps - 1]
    rows = []
    for timestep in timesteps:
        t = torch.full_like(edge_type, timestep)
        noised = transition.q_sample(edge_type, t, generator=generator)
        changed = (noised != edge_type).float().mean().item() if edge_type.numel() else 0.0
        rows.append({
            "timestep": timestep,
            "noise_probability_mean": float(transition.noise_probability(t).float().mean().item())
            if edge_type.numel() else 0.0,
            "changed_fraction": changed,
            "min_label": int(noised.min().item()) if noised.numel() else 0,
            "max_label": int(noised.max().item()) if noised.numel() else 0,
            "class_counts": _class_counts(noised, args.num_classes),
        })

    one_hot = transition.one_hot(edge_type)
    pred_logits = torch.zeros((edge_type.shape[0], args.num_classes), dtype=torch.float32)
    nll = transition.negative_log_likelihood(pred_logits, edge_type)
    summary = {
        "edge_npz": str(args.edge_npz),
        "analyzed_molecules": max_molecules,
        "total_halfedges": int(edge_type.numel()),
        "total_real_bonds": int((edge_type > 0).sum().item()),
        "no_bond_fraction": float((edge_type == 0).float().mean().item()) if edge_type.numel() else 0.0,
        "one_hot_shape": list(one_hot.shape),
        "nll_shape": list(nll.shape),
        "uniform_logits_nll_mean": float(nll.mean().item()) if nll.numel() else 0.0,
        "clean_class_counts": _class_counts(edge_type, args.num_classes),
        "timestep_rows": rows,
    }

    json_path = outdir / "edge_transition_smoke.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    report_path = outdir / "edge_transition_smoke_report.md"
    with report_path.open("w") as handle:
        handle.write("# Edge Transition Smoke Report\n\n")
        handle.write(f"- edge_npz: `{args.edge_npz}`\n")
        handle.write(f"- analyzed_molecules: {summary['analyzed_molecules']}\n")
        handle.write(f"- total_halfedges: {summary['total_halfedges']}\n")
        handle.write(f"- total_real_bonds: {summary['total_real_bonds']}\n")
        handle.write(f"- no_bond_fraction: {summary['no_bond_fraction']:.8f}\n")
        handle.write(f"- one_hot_shape: `{summary['one_hot_shape']}`\n")
        handle.write(f"- nll_shape: `{summary['nll_shape']}`\n")
        handle.write(f"- uniform_logits_nll_mean: {summary['uniform_logits_nll_mean']:.6f}\n\n")
        handle.write("| timestep | noise probability | changed fraction | label range |\n")
        handle.write("|---:|---:|---:|---|\n")
        for row in rows:
            handle.write(
                f"| {row['timestep']} | {row['noise_probability_mean']:.6f} | "
                f"{row['changed_fraction']:.6f} | "
                f"{row['min_label']}..{row['max_label']} |\n"
            )

    return report_path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge-npz", type=Path, required=True)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("analysis") / f"edge_transition_smoke_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument("--max-molecules", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--num-timesteps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    path = run_smoke(parse_args())
    print(f"Wrote edge transition smoke report to {path}")
