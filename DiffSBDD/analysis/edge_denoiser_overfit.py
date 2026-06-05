"""Run a minimal edge denoiser overfit experiment on edge bundles."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIFFSBDD_ROOT = Path(__file__).resolve().parents[1]
for extra in (PROJECT_ROOT, DIFFSBDD_ROOT):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from edge_diffusion import (  # noqa: E402
    EdgeBatch,
    EdgeBatchCollator,
    EdgeBundleDataset,
    EdgeCategoricalTransition,
    EdgeDenoiserConfig,
    EdgeDenoiserMLP,
    EdgeDenoiserNodeEdge,
    EdgeReconstructionEvalConfig,
    EdgeTransitionConfig,
    batch_confusion,
    classification_metrics,
    evaluate_edge_reconstruction,
    make_edge_class_weight,
    reconstruction_metric_defaults,
    reconstruction_summary_for_training,
)
from edge_diffusion.denoiser import (  # noqa: E402
    edge_focal_cross_entropy,
)
from edge_utils import BOND_TYPE_NAMES  # noqa: E402


def run_overfit(args: argparse.Namespace) -> Path:
    """Train a tiny edge denoiser and write reports."""
    _set_seed(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)
    device = _choose_device(args.force_cpu)
    dataset = EdgeBundleDataset(
        edge_npz=args.edge_npz,
        semantic_npz=args.semantic_npz,
        max_molecules=args.max_molecules,
    )
    train_dataset, eval_dataset, split_info = _split_dataset(dataset, args)
    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=EdgeBatchCollator(),
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=EdgeBatchCollator(),
    )
    class_counts = _count_classes(train_dataset, args.num_edge_types)
    eval_class_counts = _count_classes(eval_dataset, args.num_edge_types)
    class_weight = make_edge_class_weight(
        class_counts,
        mode=args.class_weight_mode,
        no_bond_weight=args.no_bond_weight,
    ).to(device)
    model = _build_model(args, _infer_num_atom_types(dataset)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    transition = EdgeCategoricalTransition(EdgeTransitionConfig(
        num_classes=args.num_edge_types,
        num_timesteps=args.num_timesteps,
    ))

    metrics_rows = []
    for epoch in range(1, args.epochs + 1):
        train_loss = _train_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            transition=transition,
            class_weight=class_weight,
            args=args,
            device=device,
        )
        eval_metrics = _evaluate(
            model=model,
            dataloader=eval_loader,
            transition=transition,
            args=args,
            device=device,
            seed=args.seed + epoch,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **eval_metrics,
            **_maybe_evaluate_reconstruction(
                model=model,
                dataset=dataset,
                split_info=split_info,
                args=args,
                transition=transition,
                epoch=epoch,
                device=device,
            ),
        }
        metrics_rows.append(row)

    metrics_path = args.outdir / "edge_denoiser_overfit_metrics.csv"
    _write_metrics_csv(metrics_path, metrics_rows)
    summary = _build_summary(
        args=args,
        device=device,
        dataset=dataset,
        split_info=split_info,
        class_counts=class_counts,
        eval_class_counts=eval_class_counts,
        class_weight=class_weight.detach().cpu(),
        model=model,
        metrics_rows=metrics_rows,
    )
    summary_path = args.outdir / "edge_denoiser_overfit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(model.config),
            "summary": summary,
        },
        args.outdir / "edge_denoiser_overfit.pt",
    )
    _plot_training_curves(args.outdir / "edge_denoiser_overfit_curves.png", metrics_rows)
    return _write_report(args.outdir, summary)


def _train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    transition: EdgeCategoricalTransition,
    class_weight: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> float:
    """Train for one epoch and return mean loss."""
    model.train()
    total_loss = 0.0
    total_edges = 0
    for batch in dataloader:
        batch = batch.to(device)
        batch = _maybe_add_position_noise(batch, args.pos_noise_std)
        t = torch.randint(
            low=0,
            high=args.num_timesteps,
            size=(len(batch.names),),
            device=device,
        )
        noised_edge_type = transition.q_sample(batch.target_edge_type, t[batch.edge_batch])
        logits = model(batch, noised_edge_type, t)
        loss = edge_focal_cross_entropy(
            logits=logits,
            target=batch.target_edge_type,
            class_weight=class_weight,
            gamma=args.focal_gamma,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        edge_count = int(batch.target_edge_type.numel())
        total_loss += float(loss.detach().item()) * edge_count
        total_edges += edge_count
    return total_loss / max(1, total_edges)


def _maybe_add_position_noise(batch: EdgeBatch, pos_noise_std: float) -> EdgeBatch:
    """Add DiffGui-style coordinate noise during training when requested."""
    if pos_noise_std <= 0:
        return batch
    return EdgeBatch(
        ligand_pos=batch.ligand_pos + torch.randn_like(batch.ligand_pos) * float(pos_noise_std),
        ligand_atom_type=batch.ligand_atom_type,
        halfedge_index=batch.halfedge_index,
        target_edge_type=batch.target_edge_type,
        atom_batch=batch.atom_batch,
        edge_batch=batch.edge_batch,
        names=batch.names,
    )


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    transition: EdgeCategoricalTransition,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    """Evaluate with random uniform timesteps and noising."""
    model.eval()
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    confusion = torch.zeros((args.num_edge_types, args.num_edge_types), dtype=torch.long, device=device)
    total_loss = 0.0
    total_edges = 0
    for batch in dataloader:
        batch = batch.to(device)
        t = torch.randint(
            low=0,
            high=args.num_timesteps,
            size=(len(batch.names),),
            device=device,
            generator=generator,
        )
        noised_edge_type = transition.q_sample(
            batch.target_edge_type,
            t[batch.edge_batch],
            generator=generator,
        )
        logits = model(batch, noised_edge_type, t)
        loss = edge_focal_cross_entropy(logits, batch.target_edge_type, gamma=0.0)
        pred = logits.argmax(dim=-1)
        confusion += batch_confusion(
            target=batch.target_edge_type,
            pred=pred,
            num_classes=args.num_edge_types,
        )
        edge_count = int(batch.target_edge_type.numel())
        total_loss += float(loss.detach().item()) * edge_count
        total_edges += edge_count

    return _metrics_from_confusion(confusion, total_loss / max(1, total_edges))


def _maybe_evaluate_reconstruction(
    model: torch.nn.Module,
    dataset: EdgeBundleDataset,
    split_info: dict[str, object],
    args: argparse.Namespace,
    transition: EdgeCategoricalTransition,
    epoch: int,
    device: torch.device,
) -> dict[str, float]:
    """Run optional reconstruction validation on fixed evaluation indices."""
    defaults = reconstruction_metric_defaults()
    if args.reconstruction_every <= 0:
        return defaults
    if epoch % args.reconstruction_every != 0 and epoch != args.epochs:
        return defaults
    eval_indices = [int(idx) for idx in split_info["eval_indices"]]
    _, summary = evaluate_edge_reconstruction(
        model=model,
        dataset=dataset,
        eval_indices=eval_indices,
        edge_npz=args.edge_npz,
        semantic_npz=args.semantic_npz,
        transition=transition,
        config=EdgeReconstructionEvalConfig(
            timestep=args.reconstruction_timestep,
            limit=args.reconstruction_limit,
            seed=args.seed + 100_000 + epoch,
            min_bond_length=args.min_bond_length,
            max_bond_length=args.max_bond_length,
            write_sdf=False,
        ),
        device=device,
    )
    return reconstruction_summary_for_training(summary)


def _split_dataset(
    dataset: EdgeBundleDataset,
    args: argparse.Namespace,
) -> tuple[Dataset, Dataset, dict[str, object]]:
    """Return train/eval datasets and split metadata."""
    num_molecules = len(dataset)
    val_molecules = int(args.val_molecules)
    if val_molecules <= 0:
        indices = list(range(num_molecules))
        return dataset, dataset, {
            "mode": "overfit",
            "train_molecules": num_molecules,
            "eval_molecules": num_molecules,
            "val_molecules": 0,
            "split_seed": None,
            "train_indices": indices,
            "eval_indices": indices,
        }
    if val_molecules >= num_molecules:
        raise ValueError("val_molecules must be smaller than max_molecules")

    generator = torch.Generator()
    generator.manual_seed(int(args.split_seed))
    permutation = torch.randperm(num_molecules, generator=generator).tolist()
    eval_indices = sorted(permutation[:val_molecules])
    train_indices = sorted(permutation[val_molecules:])
    return (
        Subset(dataset, train_indices),
        Subset(dataset, eval_indices),
        {
            "mode": "train_val",
            "train_molecules": len(train_indices),
            "eval_molecules": len(eval_indices),
            "val_molecules": val_molecules,
            "split_seed": int(args.split_seed),
            "train_indices": train_indices,
            "eval_indices": eval_indices,
        },
    )


def _metrics_from_confusion(confusion: torch.Tensor, eval_loss: float) -> dict[str, float]:
    """Compute accuracy, balanced accuracy, and per-class precision/recall/F1."""
    metrics, class_rows = classification_metrics(confusion.detach().cpu(), eval_loss)
    metrics["no_bond_recall"] = float(class_rows[0]["recall"])
    for row in class_rows:
        class_name = str(row["class_name"])
        metrics[f"precision_{class_name}"] = float(row["precision"])
        metrics[f"recall_{class_name}"] = float(row["recall"])
        metrics[f"f1_{class_name}"] = float(row["f1"])
    return metrics


def _count_classes(dataset: Dataset, num_classes: int) -> torch.Tensor:
    """Count target edge classes in the exposed dataset subset."""
    counts = torch.zeros((num_classes,), dtype=torch.long)
    for idx in range(len(dataset)):
        sample = dataset[idx]
        counts += torch.bincount(sample["target_edge_type"], minlength=num_classes)
    return counts


def _infer_num_atom_types(dataset: EdgeBundleDataset) -> int:
    """Infer embedding size from atom type IDs."""
    max_atom_type = 0
    for idx in range(len(dataset)):
        sample = dataset[idx]
        max_atom_type = max(max_atom_type, int(sample["ligand_atom_type"].max().item()))
    return max_atom_type + 1


def _build_model(args: argparse.Namespace, num_atom_types: int) -> torch.nn.Module:
    """Build the requested edge denoiser architecture."""
    config = EdgeDenoiserConfig(
        num_atom_types=num_atom_types,
        num_edge_types=args.num_edge_types,
        num_timesteps=args.num_timesteps,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        distance_basis_dim=args.distance_basis_dim,
        dropout=args.dropout,
    )
    if args.model_kind == "mlp":
        return EdgeDenoiserMLP(config)
    if args.model_kind == "node_edge":
        return EdgeDenoiserNodeEdge(config)
    raise ValueError(f"Unsupported model kind: {args.model_kind}")


def _choose_device(force_cpu: bool) -> torch.device:
    """Choose CUDA when available unless CPU is forced."""
    if force_cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


def _set_seed(seed: int) -> None:
    """Set Python and torch random seeds."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_metrics_csv(path: Path, rows: list[dict[str, float]]) -> None:
    """Write training metrics CSV."""
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _build_summary(
    args: argparse.Namespace,
    device: torch.device,
    dataset: EdgeBundleDataset,
    split_info: dict[str, object],
    class_counts: torch.Tensor,
    eval_class_counts: torch.Tensor,
    class_weight: torch.Tensor,
    model: torch.nn.Module,
    metrics_rows: list[dict[str, float]],
) -> dict[str, object]:
    """Build JSON summary for the overfit run."""
    first = metrics_rows[0]
    final = metrics_rows[-1]
    best = max(metrics_rows, key=lambda row: row["balanced_accuracy"])
    return {
        "edge_npz": str(args.edge_npz),
        "semantic_npz": "" if args.semantic_npz is None else str(args.semantic_npz),
        "device": str(device),
        "max_molecules": len(dataset),
        "split": split_info,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "focal_gamma": args.focal_gamma,
        "class_weight_mode": args.class_weight_mode,
        "no_bond_weight": args.no_bond_weight,
        "pos_noise_std": args.pos_noise_std,
        "num_timesteps": args.num_timesteps,
        "model_kind": args.model_kind,
        "model_config": asdict(model.config),
        "reconstruction_validation": {
            "enabled": args.reconstruction_every > 0,
            "every": args.reconstruction_every,
            "timestep": args.reconstruction_timestep,
            "limit": args.reconstruction_limit,
            "min_bond_length": args.min_bond_length,
            "max_bond_length": args.max_bond_length,
        },
        "class_counts": {
            BOND_TYPE_NAMES[idx]: int(value)
            for idx, value in enumerate(class_counts.tolist())
        },
        "eval_class_counts": {
            BOND_TYPE_NAMES[idx]: int(value)
            for idx, value in enumerate(eval_class_counts.tolist())
        },
        "class_weight": {
            BOND_TYPE_NAMES[idx]: float(value)
            for idx, value in enumerate(class_weight.tolist())
        },
        "first_epoch": first,
        "final_epoch": final,
        "best_balanced_accuracy_epoch": best,
    }


def _plot_training_curves(path: Path, rows: list[dict[str, float]]) -> None:
    """Plot loss and recall curves."""
    epochs = [int(row["epoch"]) for row in rows]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [float(row["train_loss"]) for row in rows], label="train loss")
    plt.plot(epochs, [float(row["eval_loss"]) for row in rows], label="eval loss")
    plt.plot(epochs, [float(row["balanced_accuracy"]) for row in rows], label="balanced acc")
    plt.plot(epochs, [float(row["real_bond_recall"]) for row in rows], label="real bond recall")
    if _has_finite_metric(rows, "recon_semantic_sanitize_success_rate"):
        plt.plot(
            epochs,
            [float(row["recon_semantic_sanitize_success_rate"]) for row in rows],
            marker="o",
            label="recon semantic sanitize",
        )
    if _has_finite_metric(rows, "recon_bond_type_map_match_rate"):
        plt.plot(
            epochs,
            [float(row["recon_bond_type_map_match_rate"]) for row in rows],
            marker="o",
            label="recon bond type match",
        )
    plt.xlabel("epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _has_finite_metric(rows: list[dict[str, float]], key: str) -> bool:
    """Return whether any row contains a finite metric value."""
    return any(math.isfinite(float(row.get(key, float("nan")))) for row in rows)


def _write_report(outdir: Path, summary: dict[str, object]) -> Path:
    """Write Markdown report for the overfit experiment."""
    report_path = outdir / "edge_denoiser_overfit_report.md"
    final = summary["final_epoch"]
    best = summary["best_balanced_accuracy_epoch"]
    title = "Edge Denoiser Train/Val Report"
    if summary["split"]["mode"] == "overfit":
        title = "Edge Denoiser Overfit Report"
    with report_path.open("w") as handle:
        handle.write(f"# {title}\n\n")
        handle.write(f"- edge_npz: `{summary['edge_npz']}`\n")
        handle.write(f"- semantic_npz: `{summary['semantic_npz']}`\n")
        handle.write(f"- device: `{summary['device']}`\n")
        handle.write(f"- max_molecules: {summary['max_molecules']}\n")
        handle.write(f"- split_mode: `{summary['split']['mode']}`\n")
        handle.write(f"- train_molecules: {summary['split']['train_molecules']}\n")
        handle.write(f"- eval_molecules: {summary['split']['eval_molecules']}\n")
        handle.write(f"- epochs: {summary['epochs']}\n")
        handle.write(f"- batch_size: {summary['batch_size']}\n")
        handle.write(f"- model_kind: `{summary['model_kind']}`\n")
        handle.write(f"- focal_gamma: {summary['focal_gamma']}\n")
        handle.write(f"- class_weight_mode: `{summary['class_weight_mode']}`\n")
        handle.write(f"- no_bond_weight: {summary['no_bond_weight']}\n")
        handle.write(f"- pos_noise_std: {summary['pos_noise_std']}\n")
        handle.write(f"- reconstruction_validation: `{summary['reconstruction_validation']['enabled']}`\n")
        handle.write(f"- reconstruction_timestep: {summary['reconstruction_validation']['timestep']}\n")
        handle.write(f"- reconstruction_limit: {summary['reconstruction_validation']['limit']}\n\n")
        handle.write("## Class Counts And Weights\n\n")
        handle.write("| bond type | count | weight |\n")
        handle.write("|---|---:|---:|\n")
        for bond_name, count in summary["class_counts"].items():
            handle.write(f"| {bond_name} | {count} | {summary['class_weight'][bond_name]:.6f} |\n")
        handle.write("\n## Eval Class Counts\n\n")
        handle.write("| bond type | count |\n")
        handle.write("|---|---:|\n")
        for bond_name, count in summary["eval_class_counts"].items():
            handle.write(f"| {bond_name} | {count} |\n")
        handle.write("\n## Final Metrics\n\n")
        handle.write("| metric | value |\n")
        handle.write("|---|---:|\n")
        for metric in [
            "train_loss", "eval_loss", "accuracy", "balanced_accuracy",
            "macro_precision", "macro_f1", "real_bond_precision",
            "real_bond_recall", "real_bond_f1", "no_bond_recall",
            "precision_single", "recall_single", "f1_single",
            "precision_double", "recall_double", "f1_double",
            "precision_triple", "recall_triple", "f1_triple",
            "precision_aromatic", "recall_aromatic", "f1_aromatic",
        ]:
            handle.write(f"| {metric} | {float(final[metric]):.6f} |\n")
        if summary["reconstruction_validation"]["enabled"]:
            handle.write("\n## Final Reconstruction Metrics\n\n")
            handle.write("| metric | value |\n")
            handle.write("|---|---:|\n")
            for metric in [
                "recon_molecules", "recon_exact_halfedge_match_rate",
                "recon_bond_pair_set_match_rate", "recon_bond_type_map_match_rate",
                "recon_plain_sanitize_success_rate", "recon_semantic_sanitize_success_rate",
                "recon_plain_smiles_no_stereo_match_rate",
                "recon_semantic_smiles_no_stereo_match_rate",
                "recon_mean_bad_bond_length_fraction",
            ]:
                value = float(final[metric])
                value_text = "nan" if math.isnan(value) else f"{value:.6f}"
                handle.write(f"| {metric} | {value_text} |\n")
        handle.write("\n## Best Balanced Accuracy Epoch\n\n")
        handle.write(f"- epoch: {int(best['epoch'])}\n")
        handle.write(f"- balanced_accuracy: {float(best['balanced_accuracy']):.6f}\n")
        handle.write(f"- real_bond_recall: {float(best['real_bond_recall']):.6f}\n\n")
        handle.write("## Files\n\n")
        handle.write("- [edge_denoiser_overfit_metrics.csv](edge_denoiser_overfit_metrics.csv)\n")
        handle.write("- [edge_denoiser_overfit_summary.json](edge_denoiser_overfit_summary.json)\n")
        handle.write("- [edge_denoiser_overfit.pt](edge_denoiser_overfit.pt)\n")
        handle.write("\n![curves](edge_denoiser_overfit_curves.png)\n")
    return report_path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser()
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
        default=Path("analysis") / f"edge_denoiser_overfit_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument("--max-molecules", type=int, default=128)
    parser.add_argument("--val-molecules", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--model-kind", choices=["mlp", "node_edge"], default="mlp")
    parser.add_argument("--distance-basis-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num-edge-types", type=int, default=5)
    parser.add_argument("--num-timesteps", type=int, default=1000)
    parser.add_argument("--focal-gamma", type=float, default=1.0)
    parser.add_argument(
        "--class-weight-mode",
        choices=["inverse_sqrt", "diffgui", "none"],
        default="inverse_sqrt",
    )
    parser.add_argument("--no-bond-weight", type=float, default=0.1)
    parser.add_argument("--pos-noise-std", type=float, default=0.0)
    parser.add_argument("--reconstruction-every", type=int, default=0)
    parser.add_argument("--reconstruction-timestep", type=int, default=500)
    parser.add_argument("--reconstruction-limit", type=int, default=32)
    parser.add_argument("--min-bond-length", type=float, default=0.7)
    parser.add_argument("--max-bond-length", type=float, default=2.1)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force-cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    report = run_overfit(parse_args())
    print(f"Wrote edge denoiser overfit report to {report}")
