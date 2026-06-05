"""Run a decoupled sampler smoke test for the standalone x/h/e edge diffusion branch."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from rdkit import Chem, RDLogger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIFFSBDD_ROOT = Path(__file__).resolve().parents[1]
for extra in (PROJECT_ROOT, DIFFSBDD_ROOT):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from DiffSBDD import constants  # noqa: E402
from DiffSBDD.edge_diffusion import (  # noqa: E402
    EdgeBatchCollator,
    EdgeBundleDataset,
    EdgeCategoricalTransition,
    EdgeDiffusionModel,
    EdgeDiffusionModelConfig,
    EdgeDiffusionSampler,
    EdgeSamplerConfig,
    EdgeTransitionConfig,
)
from DiffSBDD.edge_diffusion.evaluation import (  # noqa: E402
    build_reconstruction_row,
    build_semantic_result,
    summarize_reconstruction,
    write_result_mol,
)
from DiffSBDD.edge_utils import BOND_TYPE_NAMES  # noqa: E402

RDLogger.DisableLog("rdApp.*")


def run_smoke(args: argparse.Namespace) -> Path:
    """Run sampler smoke and write CSV, SDF, JSON, plots, and Markdown report."""
    _set_seed(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    device = _choose_device(args.force_cpu)

    dataset = EdgeBundleDataset(
        edge_npz=args.edge_npz,
        semantic_npz=args.semantic_npz if args.semantic_npz.exists() else None,
        max_molecules=args.max_molecules,
    )
    edge_data = np.load(args.edge_npz, allow_pickle=True)
    semantic_data = (
        np.load(args.semantic_npz, allow_pickle=True)
        if args.use_semantic_reconstruction and args.semantic_npz.exists()
        else None
    )
    dataset_info = constants.dataset_params[args.dataset]
    model, checkpoint_payload = _load_model(args.checkpoint, device)
    transition = EdgeCategoricalTransition(EdgeTransitionConfig(
        num_classes=model.config.num_edge_types,
        num_timesteps=model.config.num_timesteps,
        prior=args.edge_prior or _summary_value(checkpoint_payload, "edge_prior", "uniform"),
    ))
    atom_prior = _make_atom_prior(dataset, model.config.num_atom_types).to(device)
    sampler = EdgeDiffusionSampler(
        model=model,
        transition=transition,
        config=EdgeSamplerConfig(
            mode=args.mode,
            start_timestep=args.start_timestep,
            pos_noise_std=args.pos_noise_std if args.pos_noise_std is not None else _summary_value(
                checkpoint_payload,
                "pos_noise_std",
                0.05,
            ),
            pos_noise_mode=args.pos_noise_mode or _summary_value(
                checkpoint_payload,
                "pos_noise_mode",
                "t_scaled",
            ),
            atom_corruption_prob=(
                args.atom_corruption_prob
                if args.atom_corruption_prob is not None
                else _summary_value(checkpoint_payload, "atom_corruption_prob", 0.05)
            ),
            atom_temperature=args.atom_temperature,
            edge_temperature=args.edge_temperature,
            position_prior_scale=args.position_prior_scale,
        ),
        atom_prior=atom_prior,
    )

    rows: list[dict[str, object]] = []
    strategy_counts: Counter[str] = Counter()
    plain_writer = Chem.SDWriter(str(args.outdir / f"{args.mode}_predicted_edge_plain.sdf"))
    semantic_writer = Chem.SDWriter(str(args.outdir / f"{args.mode}_predicted_edge_semantic.sdf"))
    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed))
    try:
        for global_idx in range(len(dataset)):
            sample_started_at = time.time()
            sample = dataset[global_idx]
            batch = EdgeBatchCollator()([sample]).to(device)
            result = sampler.sample(batch, generator=generator)
            sample_seconds = time.time() - sample_started_at

            pred_pos = result.pred_pos.detach().cpu()
            pred_atom_type = result.pred_atom_type.detach().cpu()
            pred_edge_type = result.pred_edge_type.detach().cpu()
            target_pos = batch.ligand_pos.detach().cpu()
            target_atom_type = batch.ligand_atom_type.detach().cpu()
            target_edge_type = batch.target_edge_type.detach().cpu()
            halfedge_index = batch.halfedge_index.detach().cpu()
            pred_atom_symbols = _decode_atom_symbols(pred_atom_type, dataset_info)
            plain_result = build_semantic_result(
                semantic_data=None,
                global_idx=global_idx,
                atom_symbols=pred_atom_symbols,
                positions=pred_pos,
                halfedge_index=halfedge_index,
                halfedge_type=pred_edge_type,
            )
            semantic_result = build_semantic_result(
                semantic_data=semantic_data,
                global_idx=global_idx,
                atom_symbols=pred_atom_symbols,
                positions=pred_pos,
                halfedge_index=halfedge_index,
                halfedge_type=pred_edge_type,
            )
            row = build_reconstruction_row(
                global_idx=global_idx,
                name=str(sample["name"]),
                atom_symbols=pred_atom_symbols,
                positions=pred_pos,
                halfedge_index=halfedge_index,
                target_edge_type=target_edge_type,
                pred_edge_type=pred_edge_type,
                plain_result=plain_result,
                semantic_result=semantic_result,
                original_smiles=(
                    "" if semantic_data is None
                    else str(semantic_data["original_smiles_no_stereo"][global_idx])
                ),
                min_bond_length=args.min_bond_length,
                max_bond_length=args.max_bond_length,
            )
            row.update({
                "mode": args.mode,
                "start_timestep": int(result.start_timestep),
                "reverse_steps": int(result.steps),
                "sample_seconds": sample_seconds,
                "atom_accuracy": _atom_accuracy(pred_atom_type, target_atom_type),
                "position_rmse": _position_rmse(pred_pos, target_pos),
                "pred_atom_formula": _atom_formula(pred_atom_symbols),
                "target_atom_formula": _atom_formula([
                    str(value) for value in edge_data["ligand_atom_symbol"][global_idx]
                ]),
            })
            rows.append(row)
            strategy_counts[f"plain:{plain_result.repair_strategy}"] += 1
            strategy_counts[f"semantic:{semantic_result.repair_strategy}"] += 1
            write_result_mol(plain_writer, plain_result.mol, row, prefix="plain")
            write_result_mol(semantic_writer, semantic_result.mol, row, prefix="semantic")
    finally:
        plain_writer.close()
        semantic_writer.close()
        edge_data.close()
        if semantic_data is not None:
            semantic_data.close()

    elapsed_seconds = time.time() - started_at
    summary = _build_summary(
        args=args,
        rows=rows,
        reconstruction_summary=summarize_reconstruction(rows, strategy_counts),
        checkpoint_payload=checkpoint_payload,
        model=model,
        transition=transition,
        device=device,
        elapsed_seconds=elapsed_seconds,
        atom_prior=atom_prior.detach().cpu(),
    )
    rows_path = args.outdir / f"{args.mode}_sampler_rows.csv"
    summary_path = args.outdir / f"{args.mode}_sampler_summary.json"
    _write_rows_csv(rows_path, rows)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    plot_path = args.outdir / f"{args.mode}_sampler_metrics.png"
    _plot_rows(plot_path, rows)
    return _write_report(args.outdir, args.mode, summary, rows_path, summary_path, plot_path)


def _load_model(path: Path, device: torch.device) -> tuple[EdgeDiffusionModel, dict[str, Any]]:
    """Load a standalone edge diffusion checkpoint."""
    payload = torch.load(path, map_location=device)
    config_payload = payload.get("config") or payload.get("model_config")
    if not isinstance(config_payload, dict):
        raise KeyError("checkpoint must contain a model config dictionary")
    model = EdgeDiffusionModel(EdgeDiffusionModelConfig(**config_payload)).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


def _build_summary(
    args: argparse.Namespace,
    rows: list[dict[str, object]],
    reconstruction_summary: dict[str, object],
    checkpoint_payload: dict[str, Any],
    model: EdgeDiffusionModel,
    transition: EdgeCategoricalTransition,
    device: torch.device,
    elapsed_seconds: float,
    atom_prior: torch.Tensor,
) -> dict[str, object]:
    """Build a JSON-serializable smoke summary."""
    sample_seconds = [float(row["sample_seconds"]) for row in rows]
    return {
        "task": "edge_diffusion_sampler_smoke",
        "decoupled_from_rl": True,
        "ddpo_imported": False,
        "checkpoint": str(args.checkpoint),
        "edge_npz": str(args.edge_npz),
        "semantic_npz": str(args.semantic_npz) if args.semantic_npz.exists() else "",
        "dataset": args.dataset,
        "mode": args.mode,
        "device": str(device),
        "seed": int(args.seed),
        "molecules": len(rows),
        "elapsed_seconds": elapsed_seconds,
        "seconds_per_molecule": elapsed_seconds / max(1, len(rows)),
        "mean_sampling_seconds": float(np.mean(sample_seconds)) if sample_seconds else 0.0,
        "p95_sampling_seconds": float(np.percentile(sample_seconds, 95)) if sample_seconds else 0.0,
        "start_timestep": int(args.start_timestep),
        "reverse_steps": int(args.start_timestep) + 1,
        "edge_prior": transition.config.prior,
        "pos_noise_std": float(args.pos_noise_std) if args.pos_noise_std is not None else _summary_value(
            checkpoint_payload,
            "pos_noise_std",
            0.05,
        ),
        "pos_noise_mode": args.pos_noise_mode or _summary_value(checkpoint_payload, "pos_noise_mode", "t_scaled"),
        "atom_corruption_prob": (
            float(args.atom_corruption_prob)
            if args.atom_corruption_prob is not None
            else _summary_value(checkpoint_payload, "atom_corruption_prob", 0.05)
        ),
        "atom_temperature": float(args.atom_temperature),
        "edge_temperature": float(args.edge_temperature),
        "position_prior_scale": float(args.position_prior_scale),
        "model_config": vars(model.config),
        "checkpoint_summary": checkpoint_payload.get("summary", {}),
        "atom_prior": {
            str(idx): float(value)
            for idx, value in enumerate(atom_prior.tolist())
        },
        "atom_accuracy_mean": _mean_float(rows, "atom_accuracy"),
        "position_rmse_mean": _mean_float(rows, "position_rmse"),
        "reconstruction": reconstruction_summary,
        "plain_sdf": f"{args.mode}_predicted_edge_plain.sdf",
        "semantic_sdf": f"{args.mode}_predicted_edge_semantic.sdf",
        "rows_csv": f"{args.mode}_sampler_rows.csv",
    }


def _write_report(
    outdir: Path,
    mode: str,
    summary: dict[str, object],
    rows_path: Path,
    summary_path: Path,
    plot_path: Path,
) -> Path:
    """Write a Markdown report for the sampler smoke run."""
    report_path = outdir / f"{mode}_sampler_report.md"
    reconstruction = summary["reconstruction"]
    with report_path.open("w") as handle:
        handle.write("# Edge Diffusion Sampler Smoke\n\n")
        handle.write("## Scope\n\n")
        handle.write(
            "This is an isolated x/h/e edge-diffusion sampler smoke test. "
            "It does not import or modify DDPO/RL modules, rewards, launch configs, "
            "or the normal DiffSBDD reinforcement-learning path.\n\n"
        )
        handle.write("## Configuration\n\n")
        handle.write(f"- mode: `{summary['mode']}`\n")
        handle.write(f"- checkpoint: `{summary['checkpoint']}`\n")
        handle.write(f"- edge_npz: `{summary['edge_npz']}`\n")
        handle.write(f"- semantic_npz: `{summary['semantic_npz']}`\n")
        handle.write(f"- device: `{summary['device']}`\n")
        handle.write(f"- molecules: {summary['molecules']}\n")
        handle.write(f"- start_timestep: {summary['start_timestep']}\n")
        handle.write(f"- reverse_steps: {summary['reverse_steps']}\n")
        handle.write(f"- edge_prior: `{summary['edge_prior']}`\n")
        handle.write(f"- pos_noise_std: {summary['pos_noise_std']}\n")
        handle.write(f"- pos_noise_mode: `{summary['pos_noise_mode']}`\n")
        handle.write(f"- atom_corruption_prob: {summary['atom_corruption_prob']}\n\n")
        if summary["mode"] == "prior":
            handle.write(
                "> `prior` mode preserves atom counts/halfedge topology from the input batch "
                "for smoke testing. It is not a production de novo molecule generator.\n\n"
            )
        else:
            handle.write(
                "> `reconstruct` mode starts from noised real CrossDocked x/h/e states and "
                "tests whether the reverse chain plus predicted-edge decoder can run end to end.\n\n"
            )

        handle.write("## Metrics\n\n")
        handle.write("| metric | value |\n")
        handle.write("|---|---:|\n")
        metric_rows = {
            "elapsed_seconds": summary["elapsed_seconds"],
            "seconds_per_molecule": summary["seconds_per_molecule"],
            "mean_sampling_seconds": summary["mean_sampling_seconds"],
            "p95_sampling_seconds": summary["p95_sampling_seconds"],
            "atom_accuracy_mean": summary["atom_accuracy_mean"],
            "position_rmse_mean": summary["position_rmse_mean"],
            "exact_halfedge_match_rate": reconstruction["exact_halfedge_match_rate"],
            "bond_pair_set_match_rate": reconstruction["bond_pair_set_match_rate"],
            "bond_type_map_match_rate": reconstruction["bond_type_map_match_rate"],
            "mean_halfedge_accuracy": reconstruction["mean_halfedge_accuracy"],
            "mean_bond_pair_precision": reconstruction["mean_bond_pair_precision"],
            "mean_bond_pair_recall": reconstruction["mean_bond_pair_recall"],
            "plain_sanitize_success_rate": reconstruction["plain_sanitize_success_rate"],
            "semantic_sanitize_success_rate": reconstruction["semantic_sanitize_success_rate"],
            "mean_bad_bond_length_fraction": reconstruction["mean_bad_bond_length_fraction"],
            "total_bad_bond_length_count": reconstruction["total_bad_bond_length_count"],
        }
        for key, value in metric_rows.items():
            handle.write(f"| {key} | {_format_float(value)} |\n")

        handle.write("\n## Edge Class Names\n\n")
        handle.write("| id | type |\n")
        handle.write("|---:|---|\n")
        for type_id, type_name in BOND_TYPE_NAMES.items():
            handle.write(f"| {type_id} | {type_name} |\n")

        handle.write("\n## Files\n\n")
        handle.write(f"- [{rows_path.name}]({rows_path.name})\n")
        handle.write(f"- [{summary_path.name}]({summary_path.name})\n")
        handle.write(f"- [{summary['plain_sdf']}]({summary['plain_sdf']})\n")
        handle.write(f"- [{summary['semantic_sdf']}]({summary['semantic_sdf']})\n")
        handle.write(f"- [{plot_path.name}]({plot_path.name})\n\n")
        handle.write(f"![sampler metrics]({plot_path.name})\n")
    return report_path


def _write_rows_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write sampler rows to CSV using the union of row keys."""
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _plot_rows(path: Path, rows: list[dict[str, object]]) -> None:
    """Plot compact per-molecule sampler diagnostics."""
    if not rows:
        return
    xs = list(range(len(rows)))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes[0, 0].plot(xs, [float(row["halfedge_accuracy"]) for row in rows], marker="o")
    axes[0, 0].set_title("Halfedge accuracy")
    axes[0, 1].plot(xs, [float(row["atom_accuracy"]) for row in rows], marker="o")
    axes[0, 1].set_title("Atom accuracy")
    axes[1, 0].plot(xs, [float(row["bad_bond_length_fraction"]) for row in rows], marker="o")
    axes[1, 0].set_title("Bad bond length fraction")
    axes[1, 1].bar(xs, [int(row["plain_sanitize_success"]) for row in rows])
    axes[1, 1].set_ylim(0, 1.1)
    axes[1, 1].set_title("Plain sanitize success")
    for axis in axes.flat:
        axis.set_xlabel("sample")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _make_atom_prior(dataset: EdgeBundleDataset, num_atom_types: int) -> torch.Tensor:
    """Build an empirical atom prior from the exposed dataset subset."""
    counts = torch.ones((num_atom_types,), dtype=torch.float32)
    for idx in range(len(dataset)):
        sample = dataset[idx]
        counts += torch.bincount(sample["ligand_atom_type"], minlength=num_atom_types).float()
    return counts / counts.sum().clamp_min(1e-8)


def _decode_atom_symbols(atom_type: torch.Tensor, dataset_info: dict[str, Any]) -> list[str]:
    """Decode atom type IDs with the configured dataset atom decoder."""
    atom_decoder = dataset_info["atom_decoder"]
    symbols = []
    for atom_idx in atom_type.tolist():
        symbol = str(atom_decoder[int(atom_idx)])
        if "." in symbol:
            symbol = symbol.split(".", maxsplit=1)[0]
        if "_" in symbol:
            symbol = symbol.split("_", maxsplit=1)[0]
        symbols.append(symbol)
    return symbols


def _summary_value(payload: dict[str, Any], key: str, default: Any) -> Any:
    """Read a training summary value from a checkpoint payload."""
    summary = payload.get("summary", {})
    if isinstance(summary, dict) and key in summary:
        return summary[key]
    return default


def _atom_accuracy(pred_atom_type: torch.Tensor, target_atom_type: torch.Tensor) -> float:
    """Return per-molecule atom type accuracy."""
    if pred_atom_type.numel() == 0:
        return 0.0
    return float((pred_atom_type.long() == target_atom_type.long()).float().mean().item())


def _position_rmse(pred_pos: torch.Tensor, target_pos: torch.Tensor) -> float:
    """Return coordinate RMSE against the source bundle positions."""
    if pred_pos.numel() == 0:
        return 0.0
    return math.sqrt(float(((pred_pos - target_pos) ** 2).mean().item()))


def _atom_formula(symbols: list[str]) -> str:
    """Return compact atom-count formula text."""
    counts = Counter(symbols)
    return "".join(f"{symbol}{counts[symbol]}" for symbol in sorted(counts))


def _mean_float(rows: list[dict[str, object]], key: str) -> float:
    """Return mean of numeric row values."""
    if not rows:
        return 0.0
    return float(np.mean([float(row[key]) for row in rows]))


def _format_float(value: object) -> str:
    """Format numeric values for Markdown tables."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "nan"
    return f"{number:.6f}"


def _choose_device(force_cpu: bool) -> torch.device:
    """Choose CUDA when visible unless CPU is forced."""
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _set_seed(seed: int) -> None:
    """Set Python, NumPy, and torch seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("analysis/edge_diffusion_xhe_overfit_20260603_n128_uniform_gpu/edge_diffusion_overfit.pt"),
    )
    parser.add_argument(
        "--edge-npz",
        type=Path,
        default=Path("analysis/edge_data_chain_20260602_repair_n1000/train_edge.npz"),
    )
    parser.add_argument(
        "--semantic-npz",
        type=Path,
        default=Path("analysis/edge_semantic_bundle_20260602_n1000/train_edge_semantic.npz"),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("analysis") / f"edge_diffusion_sampler_smoke_{timestamp}",
    )
    parser.add_argument("--dataset", choices=sorted(constants.dataset_params.keys()), default="crossdock_full")
    parser.add_argument("--mode", choices=["reconstruct", "prior"], default="reconstruct")
    parser.add_argument("--max-molecules", type=int, default=8)
    parser.add_argument("--start-timestep", type=int, default=50)
    parser.add_argument("--edge-prior", choices=["uniform", "absorb", "tomask"], default=None)
    parser.add_argument("--pos-noise-std", type=float, default=None)
    parser.add_argument("--pos-noise-mode", choices=["constant", "t_scaled"], default=None)
    parser.add_argument("--atom-corruption-prob", type=float, default=None)
    parser.add_argument("--atom-temperature", type=float, default=1.0)
    parser.add_argument("--edge-temperature", type=float, default=1.0)
    parser.add_argument("--position-prior-scale", type=float, default=1.0)
    parser.add_argument("--min-bond-length", type=float, default=0.7)
    parser.add_argument("--max-bond-length", type=float, default=2.1)
    parser.add_argument("--seed", type=int, default=260604)
    parser.add_argument("--use-semantic-reconstruction", action="store_true")
    parser.add_argument("--force-cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    report = run_smoke(parse_args())
    print(f"Wrote edge diffusion sampler smoke report to {report}")
