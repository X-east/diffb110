"""Evaluate DiffSBDD-xhe edge-head checkpoints on fixed ligand halfedges."""

from __future__ import annotations

import argparse
import csv
import json
from argparse import Namespace
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
DIFFSBDD_ROOT = REPO_ROOT / "DiffSBDD"
for import_path in (REPO_ROOT, DIFFSBDD_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from lightning_modules import LigandPocketDDPM  # noqa: E402


EDGE_LABELS = ["no_bond", "single", "double", "triple", "aromatic"]


def _namespaceify(value: Any) -> Any:
    """Convert nested mappings into ``argparse.Namespace`` objects."""
    if isinstance(value, dict):
        return Namespace(**{key: _namespaceify(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_namespaceify(item) for item in value]
    return value


def _load_config(path: Path) -> Namespace:
    """Load a YAML config as a namespace tree."""
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return _namespaceify(config)


def _build_model(args: Namespace) -> LigandPocketDDPM:
    """Instantiate the DiffSBDD Lightning module exactly like ``train.py``."""
    histogram = np.load(Path(args.datadir, "size_distribution.npy")).tolist()
    return LigandPocketDDPM(
        outdir=Path(args.logdir, args.run_name),
        dataset=args.dataset,
        datadir=args.datadir,
        batch_size=args.batch_size,
        lr=args.lr,
        egnn_params=args.egnn_params,
        diffusion_params=args.diffusion_params,
        num_workers=args.num_workers,
        augment_noise=args.augment_noise,
        augment_rotation=args.augment_rotation,
        clip_grad=args.clip_grad,
        eval_epochs=args.eval_epochs,
        eval_params=args.eval_params,
        visualize_sample_epoch=args.visualize_sample_epoch,
        visualize_chain_epoch=args.visualize_chain_epoch,
        auxiliary_loss=args.auxiliary_loss,
        loss_params=args.loss_params,
        mode=args.mode,
        node_histogram=histogram,
        pocket_representation=args.pocket_representation,
        virtual_nodes=args.virtual_nodes,
        diffusion_mode=getattr(args, "diffusion_mode", "xh"),
        bond_diffusion=getattr(args, "bond_diffusion", None),
    )


def _load_state(
    model: LigandPocketDDPM,
    checkpoint_path: Path,
    *,
    strict: bool,
) -> dict[str, int]:
    """Load one checkpoint and return compatibility counts."""
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    state_dict = checkpoint.get("state_dict", checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=strict)
    return {
        "missing": len(incompatible.missing_keys),
        "unexpected": len(incompatible.unexpected_keys),
    }


def _seed_everything(seed: int) -> None:
    """Seed CPU and CUDA RNGs so noised edge labels are comparable."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _split_loader(model: LigandPocketDDPM, split: str, batch_size: int) -> DataLoader:
    """Build a deterministic dataloader for one split."""
    if split == "train":
        dataset = model.train_dataset
    elif split == "val":
        dataset = model.val_dataset
    else:
        raise ValueError(f"unsupported split: {split}")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=dataset.collate_fn,
        pin_memory=True,
    )


def _evaluate_split(
    model: LigandPocketDDPM,
    split: str,
    *,
    batch_size: int,
    max_molecules: int | None,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate edge denoising metrics for one dataset split."""
    loader = _split_loader(model, split, batch_size)
    confusion = torch.zeros((len(EDGE_LABELS), len(EDGE_LABELS)), dtype=torch.long)
    total_loss = 0.0
    total_molecules = 0
    total_edges = 0
    total_real_edges = 0
    correct_edges = 0
    correct_real_edges = 0
    correct_no_bond_edges = 0
    total_no_bond_edges = 0
    binary_tp = 0
    binary_fp = 0
    binary_fn = 0

    model.eval()
    with torch.no_grad():
        for batch in loader:
            ligand, pocket = model.get_ligand_and_pocket(batch)
            output = model.ddpm.edge_forward_smoke(ligand, pocket)
            target = output.target_edge_type.detach().cpu().long()
            pred = output.pred_edge_logits.argmax(dim=-1).detach().cpu().long()
            batch_molecules = int(batch["num_lig_atoms"].shape[0])

            for target_class, pred_class in zip(target.tolist(), pred.tolist()):
                confusion[target_class, pred_class] += 1

            real_mask = target > 0
            no_bond_mask = target == 0
            pred_real_mask = pred > 0
            correct = pred == target

            total_loss += float(output.loss_per_molecule.detach().sum().cpu())
            total_molecules += batch_molecules
            total_edges += int(target.numel())
            total_real_edges += int(real_mask.sum().item())
            correct_edges += int(correct.sum().item())
            correct_real_edges += int((correct & real_mask).sum().item())
            correct_no_bond_edges += int((correct & no_bond_mask).sum().item())
            total_no_bond_edges += int(no_bond_mask.sum().item())
            binary_tp += int((pred_real_mask & real_mask).sum().item())
            binary_fp += int((pred_real_mask & ~real_mask).sum().item())
            binary_fn += int((~pred_real_mask & real_mask).sum().item())

            if max_molecules is not None and total_molecules >= max_molecules:
                break

    binary_precision = binary_tp / max(binary_tp + binary_fp, 1)
    binary_recall = binary_tp / max(binary_tp + binary_fn, 1)
    binary_f1 = (
        2 * binary_precision * binary_recall / max(binary_precision + binary_recall, 1e-12)
    )
    class_support = confusion.sum(dim=1).numpy()
    class_correct = confusion.diag().numpy()
    class_recall = {
        EDGE_LABELS[index]: float(class_correct[index] / class_support[index])
        if class_support[index] > 0
        else None
        for index in range(len(EDGE_LABELS))
    }
    prediction_hist = confusion.sum(dim=0).numpy()
    target_hist = confusion.sum(dim=1).numpy()
    return {
        "split": split,
        "molecules": total_molecules,
        "edges": total_edges,
        "real_edges": total_real_edges,
        "no_bond_edges": total_no_bond_edges,
        "edge_loss_per_mol": total_loss / max(total_molecules, 1),
        "edge_acc": correct_edges / max(total_edges, 1),
        "real_bond_acc": correct_real_edges / max(total_real_edges, 1),
        "no_bond_acc": correct_no_bond_edges / max(total_no_bond_edges, 1),
        "binary_bond_precision": binary_precision,
        "binary_bond_recall": binary_recall,
        "binary_bond_f1": binary_f1,
        "class_recall": class_recall,
        "target_hist": {
            EDGE_LABELS[index]: int(target_hist[index])
            for index in range(len(EDGE_LABELS))
        },
        "prediction_hist": {
            EDGE_LABELS[index]: int(prediction_hist[index])
            for index in range(len(EDGE_LABELS))
        },
        "confusion": confusion.numpy().tolist(),
    }


def _evaluate_model(
    *,
    name: str,
    checkpoint_path: Path,
    strict: bool,
    args: Namespace,
    splits: list[str],
    batch_size: int,
    max_train_molecules: int | None,
    max_val_molecules: int | None,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate one named checkpoint across requested splits."""
    _seed_everything(seed)
    model = _build_model(args)
    load_status = _load_state(model, checkpoint_path, strict=strict)
    model.setup("fit")
    model.to(device)
    split_results = {}
    for split in splits:
        _seed_everything(seed)
        max_molecules = max_train_molecules if split == "train" else max_val_molecules
        split_results[split] = _evaluate_split(
            model,
            split,
            batch_size=batch_size,
            max_molecules=max_molecules,
            device=device,
        )
    return {
        "name": name,
        "checkpoint": str(checkpoint_path),
        "strict": strict,
        "load_status": load_status,
        "splits": split_results,
    }


def _write_summary_csv(results: list[dict[str, Any]], output_path: Path) -> None:
    """Write one row per model and split."""
    fields = [
        "model",
        "split",
        "molecules",
        "edges",
        "real_edges",
        "edge_loss_per_mol",
        "edge_acc",
        "real_bond_acc",
        "no_bond_acc",
        "binary_bond_precision",
        "binary_bond_recall",
        "binary_bond_f1",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model_result in results:
            for split, metrics in model_result["splits"].items():
                row = {"model": model_result["name"], "split": split}
                row.update({field: metrics[field] for field in fields if field in metrics})
                writer.writerow(row)


def _plot_metric_bars(results: list[dict[str, Any]], output_path: Path) -> None:
    """Plot headline edge metrics by model and split."""
    metrics = ["edge_acc", "real_bond_acc", "binary_bond_f1"]
    labels = []
    values = {metric: [] for metric in metrics}
    for model_result in results:
        for split, split_metrics in model_result["splits"].items():
            labels.append(f"{model_result['name']}\n{split}")
            for metric in metrics:
                values[metric].append(split_metrics[metric])

    x_positions = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.6), 4.5))
    for offset, metric in enumerate(metrics):
        ax.bar(x_positions + (offset - 1) * width, values[metric], width, label=metric)
    ax.set_ylim(0, 1)
    ax.set_ylabel("score")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_confusion_matrix(
    confusion: list[list[int]],
    title: str,
    output_path: Path,
) -> None:
    """Plot a row-normalized confusion matrix."""
    matrix = np.asarray(confusion, dtype=np.float64)
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("predicted")
    ax.set_ylabel("target")
    ax.set_xticks(np.arange(len(EDGE_LABELS)))
    ax.set_yticks(np.arange(len(EDGE_LABELS)))
    ax.set_xticklabels(EDGE_LABELS, rotation=35, ha="right")
    ax.set_yticklabels(EDGE_LABELS)
    for row in range(len(EDGE_LABELS)):
        for col in range(len(EDGE_LABELS)):
            ax.text(
                col,
                row,
                f"{normalized[row, col]:.2f}\n({int(matrix[row, col])})",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if normalized[row, col] > 0.55 else "black",
            )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    """Run checkpoint edge-head evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trained-checkpoint", type=Path, required=True)
    parser.add_argument("--trained-name", type=str, default="trained")
    parser.add_argument("--warm-start-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-train-molecules", type=int, default=2000)
    parser.add_argument("--max-val-molecules", type=int, default=None)
    parser.add_argument("--seed", type=int, default=260604)
    parser.add_argument("--device", type=str, default="cuda")
    args_cli = parser.parse_args()

    args = _load_config(args_cli.config)
    args_cli.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        args_cli.device
        if args_cli.device == "cpu" or torch.cuda.is_available()
        else "cpu"
    )

    results = [
        _evaluate_model(
            name="warm_start_random_edge",
            checkpoint_path=args_cli.warm_start_checkpoint,
            strict=False,
            args=args,
            splits=["train", "val"],
            batch_size=args_cli.batch_size,
            max_train_molecules=args_cli.max_train_molecules,
            max_val_molecules=args_cli.max_val_molecules,
            seed=args_cli.seed,
            device=device,
        ),
        _evaluate_model(
            name=args_cli.trained_name,
            checkpoint_path=args_cli.trained_checkpoint,
            strict=True,
            args=args,
            splits=["train", "val"],
            batch_size=args_cli.batch_size,
            max_train_molecules=args_cli.max_train_molecules,
            max_val_molecules=args_cli.max_val_molecules,
            seed=args_cli.seed,
            device=device,
        ),
    ]

    summary_json = args_cli.output_dir / "edge_metrics_summary.json"
    summary_csv = args_cli.output_dir / "edge_metrics_summary.csv"
    summary_plot = args_cli.output_dir / "edge_metric_bars.png"
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
    _write_summary_csv(results, summary_csv)
    _plot_metric_bars(results, summary_plot)

    for model_result in results:
        for split, metrics in model_result["splits"].items():
            filename = f"confusion_{model_result['name']}_{split}.png"
            _plot_confusion_matrix(
                metrics["confusion"],
                title=f"{model_result['name']} / {split}",
                output_path=args_cli.output_dir / filename,
            )

    print(f"wrote {summary_json}")
    print(f"wrote {summary_csv}")
    print(f"wrote {summary_plot}")


if __name__ == "__main__":
    main()
