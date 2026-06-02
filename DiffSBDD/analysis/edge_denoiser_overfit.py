"""Run a minimal edge denoiser overfit experiment on edge bundles."""

from __future__ import annotations

import argparse
import csv
import json
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
from torch.utils.data import DataLoader

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
    EdgeTransitionConfig,
)
from edge_diffusion.denoiser import (  # noqa: E402
    edge_focal_cross_entropy,
    make_inverse_sqrt_class_weight,
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
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=EdgeBatchCollator(),
    )
    eval_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=EdgeBatchCollator(),
    )
    class_counts = _count_classes(dataset, args.num_edge_types)
    class_weight = make_inverse_sqrt_class_weight(class_counts).to(device)
    model = EdgeDenoiserMLP(EdgeDenoiserConfig(
        num_atom_types=_infer_num_atom_types(dataset),
        num_edge_types=args.num_edge_types,
        num_timesteps=args.num_timesteps,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    )).to(device)
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
        }
        metrics_rows.append(row)

    metrics_path = args.outdir / "edge_denoiser_overfit_metrics.csv"
    _write_metrics_csv(metrics_path, metrics_rows)
    summary = _build_summary(
        args=args,
        device=device,
        dataset=dataset,
        class_counts=class_counts,
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
    model: EdgeDenoiserMLP,
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


@torch.no_grad()
def _evaluate(
    model: EdgeDenoiserMLP,
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
        for target_class in range(args.num_edge_types):
            mask = batch.target_edge_type == target_class
            if not bool(mask.any()):
                continue
            pred_counts = torch.bincount(pred[mask], minlength=args.num_edge_types)
            confusion[target_class] += pred_counts
        edge_count = int(batch.target_edge_type.numel())
        total_loss += float(loss.detach().item()) * edge_count
        total_edges += edge_count

    return _metrics_from_confusion(confusion, total_loss / max(1, total_edges))


def _metrics_from_confusion(confusion: torch.Tensor, eval_loss: float) -> dict[str, float]:
    """Compute accuracy, balanced accuracy, and per-class recall."""
    confusion_cpu = confusion.detach().cpu()
    correct = torch.diag(confusion_cpu)
    support = confusion_cpu.sum(dim=1).clamp_min(1)
    recall = correct.float() / support.float()
    total = int(confusion_cpu.sum().item())
    accuracy = float(correct.sum().item() / max(1, total))
    real_support = confusion_cpu[1:].sum().item()
    real_correct = correct[1:].sum().item()
    metrics = {
        "eval_loss": eval_loss,
        "accuracy": accuracy,
        "balanced_accuracy": float(recall.mean().item()),
        "real_bond_recall": float(real_correct / max(1, real_support)),
        "no_bond_recall": float(recall[0].item()),
    }
    for class_idx, value in enumerate(recall.tolist()):
        metrics[f"recall_{BOND_TYPE_NAMES[class_idx]}"] = float(value)
    return metrics


def _count_classes(dataset: EdgeBundleDataset, num_classes: int) -> torch.Tensor:
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
    class_counts: torch.Tensor,
    class_weight: torch.Tensor,
    model: EdgeDenoiserMLP,
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
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "focal_gamma": args.focal_gamma,
        "num_timesteps": args.num_timesteps,
        "model_config": asdict(model.config),
        "class_counts": {
            BOND_TYPE_NAMES[idx]: int(value)
            for idx, value in enumerate(class_counts.tolist())
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
    plt.xlabel("epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _write_report(outdir: Path, summary: dict[str, object]) -> Path:
    """Write Markdown report for the overfit experiment."""
    report_path = outdir / "edge_denoiser_overfit_report.md"
    final = summary["final_epoch"]
    best = summary["best_balanced_accuracy_epoch"]
    with report_path.open("w") as handle:
        handle.write("# Edge Denoiser Overfit Report\n\n")
        handle.write(f"- edge_npz: `{summary['edge_npz']}`\n")
        handle.write(f"- semantic_npz: `{summary['semantic_npz']}`\n")
        handle.write(f"- device: `{summary['device']}`\n")
        handle.write(f"- max_molecules: {summary['max_molecules']}\n")
        handle.write(f"- epochs: {summary['epochs']}\n")
        handle.write(f"- batch_size: {summary['batch_size']}\n")
        handle.write(f"- focal_gamma: {summary['focal_gamma']}\n\n")
        handle.write("## Class Counts And Weights\n\n")
        handle.write("| bond type | count | weight |\n")
        handle.write("|---|---:|---:|\n")
        for bond_name, count in summary["class_counts"].items():
            handle.write(f"| {bond_name} | {count} | {summary['class_weight'][bond_name]:.6f} |\n")
        handle.write("\n## Final Metrics\n\n")
        handle.write("| metric | value |\n")
        handle.write("|---|---:|\n")
        for metric in [
            "train_loss", "eval_loss", "accuracy", "balanced_accuracy",
            "real_bond_recall", "no_bond_recall", "recall_single",
            "recall_double", "recall_triple", "recall_aromatic",
        ]:
            handle.write(f"| {metric} | {float(final[metric]):.6f} |\n")
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
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-edge-types", type=int, default=5)
    parser.add_argument("--num-timesteps", type=int, default=1000)
    parser.add_argument("--focal-gamma", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force-cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    report = run_overfit(parse_args())
    print(f"Wrote edge denoiser overfit report to {report}")
