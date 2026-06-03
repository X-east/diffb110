"""Evaluate trained edge denoisers with t-stratified metrics and reconstruction."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIFFSBDD_ROOT = Path(__file__).resolve().parents[1]
for extra in (PROJECT_ROOT, DIFFSBDD_ROOT):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from edge_diffusion import (  # noqa: E402
    EdgeBatchCollator,
    EdgeBundleDataset,
    EdgeCategoricalTransition,
    EdgeDenoiserConfig,
    EdgeDenoiserMLP,
    EdgeDenoiserNodeEdge,
    EdgeReconstructionEvalConfig,
    EdgeTransitionConfig,
    build_oracle_mol,
    build_semantic_oracle_mol,
    evaluate_edge_reconstruction,
)
from edge_utils import BOND_TYPE_NAMES  # noqa: E402

RDLogger.DisableLog("rdApp.*")


def run_evaluation(args: argparse.Namespace) -> Path:
    """Run t-stratified classification and predicted-bond reconstruction."""
    args.outdir.mkdir(parents=True, exist_ok=True)
    device = _choose_device(args.force_cpu)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    summary = _load_training_summary(args, checkpoint)
    max_molecules = args.max_molecules or int(summary.get("max_molecules", 0)) or None
    dataset = EdgeBundleDataset(
        edge_npz=args.edge_npz,
        semantic_npz=args.semantic_npz,
        max_molecules=max_molecules,
    )
    eval_indices = _select_eval_indices(summary, len(dataset), args.eval_split)
    if args.max_eval_molecules is not None:
        eval_indices = eval_indices[:args.max_eval_molecules]
    eval_dataset = Subset(dataset, eval_indices)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=EdgeBatchCollator(),
    )
    model = _build_model(checkpoint, summary).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    transition = EdgeCategoricalTransition(EdgeTransitionConfig(
        num_classes=model.config.num_edge_types,
        num_timesteps=model.config.num_timesteps,
    ))
    timesteps = _parse_timesteps(args.timesteps)
    metrics_rows, per_class_rows, confusion_rows, confusion_by_t = _evaluate_timesteps(
        model=model,
        dataloader=eval_loader,
        transition=transition,
        timesteps=timesteps,
        seed=args.seed,
        device=device,
    )
    _write_csv(args.outdir / "metrics_by_t.csv", metrics_rows, list(metrics_rows[0]))
    _write_csv(args.outdir / "per_class_metrics_by_t.csv", per_class_rows, list(per_class_rows[0]))
    _write_csv(args.outdir / "confusion_matrix_by_t.csv", confusion_rows, list(confusion_rows[0]))
    _plot_metrics_by_t(args.outdir / "metrics_by_t.png", metrics_rows)
    _plot_confusion_grid(args.outdir / "confusion_matrix_by_t.png", confusion_by_t)

    reconstruction_rows, reconstruction_summary = _run_reconstruction(
        model=model,
        dataset=dataset,
        eval_indices=eval_indices,
        edge_npz=args.edge_npz,
        semantic_npz=args.semantic_npz,
        transition=transition,
        timestep=args.reconstruct_timestep,
        limit=args.reconstruct_limit,
        seed=args.seed + 100_000,
        device=device,
        outdir=args.outdir,
        min_bond_length=args.min_bond_length,
        max_bond_length=args.max_bond_length,
    )
    _write_csv(
        args.outdir / "reconstruction_rows.csv",
        reconstruction_rows,
        list(reconstruction_rows[0]) if reconstruction_rows else _reconstruction_fieldnames(),
    )
    full_summary = {
        "checkpoint": str(args.checkpoint),
        "edge_npz": str(args.edge_npz),
        "semantic_npz": "" if args.semantic_npz is None else str(args.semantic_npz),
        "device": str(device),
        "eval_split": args.eval_split,
        "eval_molecules": len(eval_indices),
        "timesteps": timesteps,
        "reconstruct_timestep": args.reconstruct_timestep,
        "classification": _summarize_timestep_metrics(metrics_rows),
        "reconstruction": reconstruction_summary,
    }
    (args.outdir / "edge_denoiser_eval_summary.json").write_text(
        json.dumps(full_summary, indent=2, ensure_ascii=False)
    )
    return _write_report(args.outdir, full_summary)


@torch.no_grad()
def _evaluate_timesteps(
    model: torch.nn.Module,
    dataloader: DataLoader,
    transition: EdgeCategoricalTransition,
    timesteps: list[int],
    seed: int,
    device: torch.device,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict[int, torch.Tensor]]:
    """Evaluate classification metrics at fixed denoising timesteps."""
    metrics_rows: list[dict[str, object]] = []
    per_class_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    confusion_by_t: dict[int, torch.Tensor] = {}
    for timestep in timesteps:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed + int(timestep))
        confusion = torch.zeros(
            (model.config.num_edge_types, model.config.num_edge_types),
            dtype=torch.long,
            device=device,
        )
        total_loss = 0.0
        total_edges = 0
        for batch in dataloader:
            batch = batch.to(device)
            t = torch.full((len(batch.names),), int(timestep), dtype=torch.long, device=device)
            noised_edge_type = transition.q_sample(
                batch.target_edge_type,
                t[batch.edge_batch],
                generator=generator,
            )
            logits = model(batch, noised_edge_type, t)
            loss = F.cross_entropy(logits, batch.target_edge_type.long(), reduction="mean")
            pred = logits.argmax(dim=-1)
            confusion += _batch_confusion(
                target=batch.target_edge_type,
                pred=pred,
                num_classes=model.config.num_edge_types,
            )
            edge_count = int(batch.target_edge_type.numel())
            total_loss += float(loss.detach().item()) * edge_count
            total_edges += edge_count
        confusion_cpu = confusion.detach().cpu()
        confusion_by_t[int(timestep)] = confusion_cpu
        metrics, class_rows = _classification_metrics(confusion_cpu, total_loss / max(1, total_edges))
        metrics_rows.append({"timestep": int(timestep), **metrics})
        for row in class_rows:
            per_class_rows.append({"timestep": int(timestep), **row})
        for target_idx in range(model.config.num_edge_types):
            for pred_idx in range(model.config.num_edge_types):
                confusion_rows.append({
                    "timestep": int(timestep),
                    "target": BOND_TYPE_NAMES[target_idx],
                    "predicted": BOND_TYPE_NAMES[pred_idx],
                    "count": int(confusion_cpu[target_idx, pred_idx].item()),
                })
    return metrics_rows, per_class_rows, confusion_rows, confusion_by_t


def _batch_confusion(target: torch.Tensor, pred: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Return target-by-predicted confusion matrix for one batch."""
    flat = target.long() * num_classes + pred.long()
    counts = torch.bincount(flat, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes)


def _classification_metrics(
    confusion: torch.Tensor,
    eval_loss: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Compute overall and per-class precision, recall, and F1."""
    confusion = confusion.float()
    true_positive = torch.diag(confusion)
    support = confusion.sum(dim=1)
    predicted = confusion.sum(dim=0)
    precision = _safe_divide(true_positive, predicted)
    recall = _safe_divide(true_positive, support)
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)
    accuracy = float(true_positive.sum().item() / max(1.0, confusion.sum().item()))
    real_support = support[1:].sum()
    real_predicted = predicted[1:].sum()
    real_true_positive = true_positive[1:].sum()
    metrics = {
        "eval_loss": float(eval_loss),
        "accuracy": accuracy,
        "balanced_accuracy": float(recall.mean().item()),
        "macro_precision": float(precision.mean().item()),
        "macro_f1": float(f1.mean().item()),
        "real_bond_precision": float(_safe_scalar_divide(real_true_positive, real_predicted)),
        "real_bond_recall": float(_safe_scalar_divide(real_true_positive, real_support)),
        "real_bond_f1": float(_safe_scalar_divide(
            2.0 * real_true_positive,
            real_support + real_predicted,
        )),
    }
    class_rows = []
    for idx in range(confusion.shape[0]):
        class_rows.append({
            "class_id": idx,
            "class_name": BOND_TYPE_NAMES[idx],
            "support": int(support[idx].item()),
            "predicted": int(predicted[idx].item()),
            "true_positive": int(true_positive[idx].item()),
            "precision": float(precision[idx].item()),
            "recall": float(recall[idx].item()),
            "f1": float(f1[idx].item()),
        })
    return metrics, class_rows


def _safe_divide(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    """Return numerator / denominator with zero where denominator is zero."""
    return torch.where(
        denominator > 0,
        numerator / denominator.clamp_min(1e-12),
        torch.zeros_like(numerator),
    )


def _safe_scalar_divide(numerator: torch.Tensor, denominator: torch.Tensor) -> float:
    """Return scalar division with zero for empty denominators."""
    if float(denominator.item()) <= 0:
        return 0.0
    return float((numerator / denominator.clamp_min(1e-12)).item())


@torch.no_grad()
def _run_reconstruction(
    model: torch.nn.Module,
    dataset: EdgeBundleDataset,
    eval_indices: list[int],
    edge_npz: Path,
    semantic_npz: Path | None,
    transition: EdgeCategoricalTransition,
    timestep: int,
    limit: int | None,
    seed: int,
    device: torch.device,
    outdir: Path,
    min_bond_length: float,
    max_bond_length: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Predict bonds for validation molecules and reconstruct RDKit molecules."""
    return evaluate_edge_reconstruction(
        model=model,
        dataset=dataset,
        eval_indices=eval_indices,
        edge_npz=edge_npz,
        semantic_npz=semantic_npz,
        transition=transition,
        config=EdgeReconstructionEvalConfig(
            timestep=timestep,
            limit=limit,
            seed=seed,
            min_bond_length=min_bond_length,
            max_bond_length=max_bond_length,
            write_sdf=True,
            sdf_prefix="predicted_bond",
        ),
        device=device,
        outdir=outdir,
    )


def _build_semantic_result(
    semantic_data: np.lib.npyio.NpzFile | None,
    global_idx: int,
    atom_symbols: list[str],
    positions: torch.Tensor,
    halfedge_index: torch.Tensor,
    halfedge_type: torch.Tensor,
):
    """Build semantic-aware reconstruction when sidecar metadata is available."""
    if semantic_data is None:
        return build_oracle_mol(atom_symbols, positions, halfedge_index, halfedge_type)
    return build_semantic_oracle_mol(
        atom_symbols=atom_symbols,
        positions=positions,
        halfedge_index=halfedge_index,
        halfedge_type=halfedge_type,
        formal_charges=semantic_data["original_formal_charge"][global_idx],
        total_hs=semantic_data["original_total_h"][global_idx],
        explicit_hs=semantic_data["original_explicit_h"][global_idx],
        no_implicit_flags=semantic_data["original_no_implicit"][global_idx],
        aromatic_atom_flags=semantic_data["original_aromatic_atom"][global_idx],
        chiral_tags=semantic_data["original_chiral_tag"][global_idx],
        bond_pairs=semantic_data["original_bond_pair"][global_idx],
        bond_stereo=semantic_data["original_bond_stereo"][global_idx],
    )


def _build_reconstruction_row(
    global_idx: int,
    name: str,
    atom_symbols: list[str],
    positions: torch.Tensor,
    halfedge_index: torch.Tensor,
    target_edge_type: torch.Tensor,
    pred_edge_type: torch.Tensor,
    plain_result,
    semantic_result,
    original_smiles: str,
    min_bond_length: float,
    max_bond_length: float,
) -> dict[str, object]:
    """Build one reconstruction metrics row."""
    bond_compare = _compare_halfedge_bond_maps(halfedge_index, target_edge_type, pred_edge_type)
    length_stats = _predicted_bond_length_stats(
        atom_symbols=atom_symbols,
        positions=positions,
        halfedge_index=halfedge_index,
        pred_edge_type=pred_edge_type,
        min_bond_length=min_bond_length,
        max_bond_length=max_bond_length,
    )
    row = {
        "sample_idx": global_idx,
        "name": name,
        "num_atoms": len(atom_symbols),
        "num_halfedges": int(target_edge_type.numel()),
        "target_bond_count": int((target_edge_type > 0).sum().item()),
        "pred_bond_count": int((pred_edge_type > 0).sum().item()),
        "halfedge_accuracy": float((target_edge_type == pred_edge_type).float().mean().item()),
        "exact_halfedge_match": int(bool(torch.equal(target_edge_type, pred_edge_type))),
        **bond_compare,
        **length_stats,
        "plain_sanitize_success": int(plain_result.sanitize_success),
        "plain_repair_strategy": plain_result.repair_strategy,
        "plain_sanitize_error": plain_result.sanitize_error or "",
        "plain_smiles_no_stereo": _smiles_no_stereo(plain_result.mol),
        "semantic_sanitize_success": int(semantic_result.sanitize_success),
        "semantic_repair_strategy": semantic_result.repair_strategy,
        "semantic_sanitize_error": semantic_result.sanitize_error or "",
        "semantic_smiles_no_stereo": _smiles_no_stereo(semantic_result.mol),
        "original_smiles_no_stereo": original_smiles,
    }
    row["plain_smiles_no_stereo_match"] = int(
        bool(original_smiles) and row["plain_smiles_no_stereo"] == original_smiles
    )
    row["semantic_smiles_no_stereo_match"] = int(
        bool(original_smiles) and row["semantic_smiles_no_stereo"] == original_smiles
    )
    return row


def _compare_halfedge_bond_maps(
    halfedge_index: torch.Tensor,
    target_edge_type: torch.Tensor,
    pred_edge_type: torch.Tensor,
) -> dict[str, object]:
    """Compare target and predicted nonzero halfedge bond pair/type maps."""
    target_map = _halfedge_bond_map(halfedge_index, target_edge_type)
    pred_map = _halfedge_bond_map(halfedge_index, pred_edge_type)
    target_pairs = set(target_map)
    pred_pairs = set(pred_map)
    shared = target_pairs & pred_pairs
    type_mismatch = sum(int(target_map[pair] != pred_map[pair]) for pair in shared)
    true_positive_pairs = len(shared)
    precision = true_positive_pairs / max(1, len(pred_pairs))
    recall = true_positive_pairs / max(1, len(target_pairs))
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "bond_pair_set_match": int(target_pairs == pred_pairs),
        "bond_type_map_match": int(target_map == pred_map),
        "missing_bond_count": len(target_pairs - pred_pairs),
        "extra_bond_count": len(pred_pairs - target_pairs),
        "bond_type_mismatch_count": type_mismatch,
        "bond_pair_precision": precision,
        "bond_pair_recall": recall,
        "bond_pair_f1": f1,
    }


def _halfedge_bond_map(
    halfedge_index: torch.Tensor,
    halfedge_type: torch.Tensor,
) -> dict[tuple[int, int], int]:
    """Return nonzero bond pair/type map from halfedge tensors."""
    mapping: dict[tuple[int, int], int] = {}
    for edge_idx, edge_type_tensor in enumerate(halfedge_type):
        edge_type = int(edge_type_tensor)
        if edge_type <= 0:
            continue
        atom_i = int(halfedge_index[0, edge_idx])
        atom_j = int(halfedge_index[1, edge_idx])
        mapping[tuple(sorted((atom_i, atom_j)))] = edge_type
    return mapping


def _predicted_bond_length_stats(
    atom_symbols: list[str],
    positions: torch.Tensor,
    halfedge_index: torch.Tensor,
    pred_edge_type: torch.Tensor,
    min_bond_length: float,
    max_bond_length: float,
) -> dict[str, object]:
    """Return coarse predicted-bond length diagnostics."""
    distances = []
    bad_count = 0
    for edge_idx, edge_type_tensor in enumerate(pred_edge_type):
        if int(edge_type_tensor) <= 0:
            continue
        atom_i = int(halfedge_index[0, edge_idx])
        atom_j = int(halfedge_index[1, edge_idx])
        distance = float(torch.linalg.norm(positions[atom_i] - positions[atom_j]).item())
        distances.append(distance)
        bad_count += int(distance < min_bond_length or distance > max_bond_length)
    if not distances:
        return {
            "pred_bond_length_mean": 0.0,
            "pred_bond_length_p95": 0.0,
            "bad_bond_length_count": 0,
            "bad_bond_length_fraction": 0.0,
        }
    return {
        "pred_bond_length_mean": float(np.mean(distances)),
        "pred_bond_length_p95": float(np.percentile(distances, 95)),
        "bad_bond_length_count": bad_count,
        "bad_bond_length_fraction": bad_count / len(distances),
    }


def _write_result_mol(writer: Chem.SDWriter, mol: Chem.Mol | None, row: dict[str, object], prefix: str) -> None:
    """Write a reconstructed molecule with metrics as SDF properties."""
    if mol is None:
        return
    if not int(row[f"{prefix}_sanitize_success"]):
        return
    mol.SetProp("_Name", f"{row['sample_idx']}|{row['name']}|{prefix}")
    for key, value in row.items():
        if key == "name":
            continue
        mol.SetProp(str(key), str(value))
    try:
        writer.write(mol)
    except Exception:  # noqa: BLE001 - RDKit SDF writer can fail on rare aromatic states
        return


def _smiles_no_stereo(mol: Chem.Mol | None) -> str:
    """Return canonical no-stereo SMILES for a sanitized molecule."""
    if mol is None:
        return ""
    try:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    except Exception:  # noqa: BLE001 - RDKit can fail for partially repaired mols
        return ""


def _summarize_reconstruction(
    rows: list[dict[str, object]],
    strategy_counts: Counter[str],
) -> dict[str, object]:
    """Summarize reconstruction rows."""
    return {
        "molecules": len(rows),
        "exact_halfedge_match_rate": _mean_int(rows, "exact_halfedge_match"),
        "bond_pair_set_match_rate": _mean_int(rows, "bond_pair_set_match"),
        "bond_type_map_match_rate": _mean_int(rows, "bond_type_map_match"),
        "mean_halfedge_accuracy": _mean_float(rows, "halfedge_accuracy"),
        "mean_bond_pair_precision": _mean_float(rows, "bond_pair_precision"),
        "mean_bond_pair_recall": _mean_float(rows, "bond_pair_recall"),
        "plain_sanitize_success_rate": _mean_int(rows, "plain_sanitize_success"),
        "semantic_sanitize_success_rate": _mean_int(rows, "semantic_sanitize_success"),
        "plain_smiles_no_stereo_match_rate": _mean_int(rows, "plain_smiles_no_stereo_match"),
        "semantic_smiles_no_stereo_match_rate": _mean_int(rows, "semantic_smiles_no_stereo_match"),
        "mean_bad_bond_length_fraction": _mean_float(rows, "bad_bond_length_fraction"),
        "total_bad_bond_length_count": sum(int(row["bad_bond_length_count"]) for row in rows),
        "repair_strategy_counts": dict(sorted(strategy_counts.items())),
    }


def _mean_int(rows: list[dict[str, object]], key: str) -> float:
    """Return mean of integer 0/1 fields."""
    if not rows:
        return 0.0
    return sum(int(row[key]) for row in rows) / len(rows)


def _mean_float(rows: list[dict[str, object]], key: str) -> float:
    """Return mean of numeric fields."""
    if not rows:
        return 0.0
    return float(sum(float(row[key]) for row in rows) / len(rows))


def _summarize_timestep_metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    """Summarize t-stratified metrics."""
    best = max(rows, key=lambda row: float(row["balanced_accuracy"]))
    worst = min(rows, key=lambda row: float(row["balanced_accuracy"]))
    return {
        "best_timestep": int(best["timestep"]),
        "best_balanced_accuracy": float(best["balanced_accuracy"]),
        "best_real_bond_recall": float(best["real_bond_recall"]),
        "worst_timestep": int(worst["timestep"]),
        "worst_balanced_accuracy": float(worst["balanced_accuracy"]),
        "worst_real_bond_recall": float(worst["real_bond_recall"]),
    }


def _build_model(checkpoint: dict[str, object], summary: dict[str, object]) -> torch.nn.Module:
    """Build model from checkpoint config and saved model kind."""
    config = EdgeDenoiserConfig(**checkpoint["config"])
    model_kind = str(summary.get("model_kind", "node_edge"))
    if model_kind == "mlp":
        return EdgeDenoiserMLP(config)
    if model_kind == "node_edge":
        return EdgeDenoiserNodeEdge(config)
    raise ValueError(f"Unsupported model kind in checkpoint: {model_kind}")


def _load_training_summary(args: argparse.Namespace, checkpoint: dict[str, object]) -> dict[str, object]:
    """Load training summary from JSON or checkpoint."""
    if args.summary_json is not None:
        return json.loads(args.summary_json.read_text())
    summary = checkpoint.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("checkpoint does not contain summary; pass --summary-json")
    return summary


def _select_eval_indices(summary: dict[str, object], dataset_length: int, split: str) -> list[int]:
    """Select indices from saved split metadata."""
    split_info = summary.get("split", {})
    if not isinstance(split_info, dict):
        split_info = {}
    if split == "val":
        indices = split_info.get("eval_indices")
    elif split == "train":
        indices = split_info.get("train_indices")
    elif split == "all":
        indices = list(range(dataset_length))
    else:
        raise ValueError(f"Unsupported eval split: {split}")
    if indices is None:
        indices = list(range(dataset_length))
    return [int(idx) for idx in indices if int(idx) < dataset_length]


def _parse_timesteps(raw_timesteps: str) -> list[int]:
    """Parse comma-separated timestep list."""
    timesteps = [int(value.strip()) for value in raw_timesteps.split(",") if value.strip()]
    if not timesteps:
        raise ValueError("timesteps must contain at least one integer")
    return timesteps


def _choose_device(force_cpu: bool) -> torch.device:
    """Choose CUDA when available unless CPU is forced."""
    if force_cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    """Write rows to CSV with stable field order."""
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _plot_metrics_by_t(path: Path, rows: list[dict[str, object]]) -> None:
    """Plot main metrics across timesteps."""
    timesteps = [int(row["timestep"]) for row in rows]
    plt.figure(figsize=(8, 5))
    for key in ["balanced_accuracy", "real_bond_precision", "real_bond_recall", "real_bond_f1"]:
        plt.plot(timesteps, [float(row[key]) for row in rows], marker="o", label=key)
    plt.xlabel("timestep")
    plt.ylabel("metric")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_confusion_grid(path: Path, confusion_by_t: dict[int, torch.Tensor]) -> None:
    """Plot row-normalized confusion matrices for each timestep."""
    timesteps = list(confusion_by_t)
    n_cols = min(4, len(timesteps))
    n_rows = int(np.ceil(len(timesteps) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.2 * n_cols, 3.8 * n_rows),
        constrained_layout=True,
    )
    axes_array = np.asarray(axes).reshape(-1)
    labels = [BOND_TYPE_NAMES[idx] for idx in range(len(BOND_TYPE_NAMES))]
    for axis, timestep in zip(axes_array, timesteps):
        confusion = confusion_by_t[timestep].float()
        normalized = confusion / confusion.sum(dim=1, keepdim=True).clamp_min(1.0)
        image = axis.imshow(normalized.numpy(), vmin=0, vmax=1, cmap="Blues")
        axis.set_title(f"t={timestep}")
        axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        axis.set_yticks(range(len(labels)), labels)
        for row_idx in range(len(labels)):
            for col_idx in range(len(labels)):
                value = normalized[row_idx, col_idx].item()
                if value >= 0.01:
                    axis.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", fontsize=7)
    for axis in axes_array[len(timesteps):]:
        axis.axis("off")
    fig.colorbar(image, ax=axes_array.tolist(), shrink=0.75)
    plt.savefig(path, dpi=180)
    plt.close()


def _write_report(outdir: Path, summary: dict[str, object]) -> Path:
    """Write Markdown report for edge denoiser evaluation."""
    report_path = outdir / "edge_denoiser_eval_report.md"
    classification = summary["classification"]
    reconstruction = summary["reconstruction"]
    with report_path.open("w") as handle:
        handle.write("# Edge Denoiser Evaluation Report\n\n")
        handle.write(f"- checkpoint: `{summary['checkpoint']}`\n")
        handle.write(f"- device: `{summary['device']}`\n")
        handle.write(f"- eval_split: `{summary['eval_split']}`\n")
        handle.write(f"- eval_molecules: {summary['eval_molecules']}\n")
        handle.write(f"- timesteps: {summary['timesteps']}\n")
        handle.write(f"- reconstruct_timestep: {summary['reconstruct_timestep']}\n\n")
        handle.write("## Classification Summary\n\n")
        handle.write("| metric | value |\n|---|---:|\n")
        for key, value in classification.items():
            handle.write(f"| {key} | {value} |\n")
        handle.write("\n![metrics_by_t](metrics_by_t.png)\n\n")
        handle.write("![confusion_matrix_by_t](confusion_matrix_by_t.png)\n\n")
        handle.write("## Reconstruction Summary\n\n")
        handle.write("| metric | value |\n|---|---:|\n")
        for key, value in reconstruction.items():
            if key == "repair_strategy_counts":
                continue
            handle.write(f"| {key} | {value} |\n")
        handle.write("\n## Files\n\n")
        for filename in [
            "metrics_by_t.csv",
            "per_class_metrics_by_t.csv",
            "confusion_matrix_by_t.csv",
            "reconstruction_rows.csv",
            "edge_denoiser_eval_summary.json",
            f"predicted_bond_plain_t{summary['reconstruct_timestep']}.sdf",
            f"predicted_bond_semantic_t{summary['reconstruct_timestep']}.sdf",
        ]:
            handle.write(f"- [{filename}]({filename})\n")
    return report_path


def _reconstruction_fieldnames() -> list[str]:
    """Return fieldnames for empty reconstruction output."""
    return [
        "sample_idx", "name", "num_atoms", "num_halfedges", "target_bond_count",
        "pred_bond_count", "halfedge_accuracy", "exact_halfedge_match",
        "bond_pair_set_match", "bond_type_map_match", "missing_bond_count",
        "extra_bond_count", "bond_type_mismatch_count", "bond_pair_precision",
        "bond_pair_recall", "bond_pair_f1", "pred_bond_length_mean",
        "pred_bond_length_p95", "bad_bond_length_count", "bad_bond_length_fraction",
        "plain_sanitize_success", "plain_repair_strategy", "plain_sanitize_error",
        "plain_smiles_no_stereo", "semantic_sanitize_success",
        "semantic_repair_strategy", "semantic_sanitize_error",
        "semantic_smiles_no_stereo", "original_smiles_no_stereo",
        "plain_smiles_no_stereo_match", "semantic_smiles_no_stereo_match",
    ]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("analysis/edge_denoiser_nodeedge_20260603_n1000_train800_val200_gpu/edge_denoiser_overfit.pt"),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("analysis/edge_denoiser_nodeedge_20260603_n1000_train800_val200_gpu/edge_denoiser_overfit_summary.json"),
    )
    parser.add_argument(
        "--edge-npz",
        type=Path,
        default=Path("analysis/edge_data_chain_20260602_n1000/train_edge.npz"),
    )
    parser.add_argument(
        "--semantic-npz",
        type=Path,
        default=Path("analysis/edge_semantic_bundle_20260602_n1000/train_edge_semantic.npz"),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("analysis") / f"edge_denoiser_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument("--eval-split", choices=["val", "train", "all"], default="val")
    parser.add_argument("--max-molecules", type=int, default=None)
    parser.add_argument("--max-eval-molecules", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--timesteps", default="0,50,100,250,500,750,950,999")
    parser.add_argument("--reconstruct-timestep", type=int, default=500)
    parser.add_argument("--reconstruct-limit", type=int, default=None)
    parser.add_argument("--min-bond-length", type=float, default=0.7)
    parser.add_argument("--max-bond-length", type=float, default=2.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force-cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    report = run_evaluation(parse_args())
    print(f"Wrote edge denoiser evaluation report to {report}")
