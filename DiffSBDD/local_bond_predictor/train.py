"""Train and evaluate a local pair-wise ligand bond predictor."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIFFSBDD_ROOT = Path(__file__).resolve().parents[1]
for extra in (PROJECT_ROOT, DIFFSBDD_ROOT):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from edge_utils import BOND_TYPE_NAMES  # noqa: E402
from local_bond_predictor.dataset import (  # noqa: E402
    HalfedgePairDataset,
    PairFeatureConfig,
    collate_pair_batches,
    pair_feature_dim,
)
from local_bond_predictor.metrics import (  # noqa: E402
    confusion_matrix,
    summarize_confusion,
)
from local_bond_predictor.model import PairBondPredictor  # noqa: E402

LOGGER = logging.getLogger(__name__)
SPLITS = ("train", "val", "test")


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    """Write CSV rows using LF newlines."""
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _set_seed(seed: int) -> None:
    """Set deterministic seeds for Python and torch."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(requested_device: str) -> torch.device:
    """Resolve an explicit or automatic torch device."""
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested_device)


def _make_dataset(
    split: str,
    args: argparse.Namespace,
    feature_config: PairFeatureConfig,
    *,
    is_train: bool,
) -> HalfedgePairDataset:
    """Create one split dataset."""
    max_samples = {
        "train": args.train_max_samples,
        "val": args.val_max_samples,
        "test": args.test_max_samples,
    }[split]
    negative_ratio = args.train_negative_ratio if is_train else None
    pos_noise_std = args.pos_noise_std if is_train else 0.0
    return HalfedgePairDataset(
        args.processed_dir / f"{split}.npz",
        max_samples=max_samples,
        sample_seed=args.seed,
        negative_ratio=negative_ratio,
        pos_noise_std=pos_noise_std,
        feature_config=feature_config,
    )


def _make_loader(
    dataset: HalfedgePairDataset,
    args: argparse.Namespace,
    *,
    shuffle: bool,
) -> DataLoader:
    """Create a molecule-level DataLoader."""
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=collate_pair_batches,
    )


def _batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Move the tensors needed by the model to the selected device."""
    return batch["features"].to(device), batch["labels"].to(device)


def _run_train_epoch(
    model: PairBondPredictor,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    max_steps: int | None,
) -> tuple[float, torch.Tensor]:
    """Train for one epoch and return average loss plus confusion matrix."""
    model.train()
    total_loss = 0.0
    total_pairs = 0
    matrix = torch.zeros((args_num_classes(), args_num_classes()), dtype=torch.long)
    for step_idx, batch in enumerate(loader, start=1):
        if max_steps is not None and step_idx > max_steps:
            break
        features, labels = _batch_to_device(batch, device)
        if labels.numel() == 0:
            continue
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        pair_count = int(labels.numel())
        total_loss += float(loss.item()) * pair_count
        total_pairs += pair_count
        predictions = logits.argmax(dim=-1).detach().cpu()
        matrix += confusion_matrix(labels.detach().cpu(), predictions, args_num_classes())
    return total_loss / max(1, total_pairs), matrix


@torch.no_grad()
def _evaluate(
    model: PairBondPredictor,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    *,
    max_steps: int | None = None,
) -> tuple[float, torch.Tensor]:
    """Evaluate one split and return average loss plus confusion matrix."""
    model.eval()
    total_loss = 0.0
    total_pairs = 0
    matrix = torch.zeros((args_num_classes(), args_num_classes()), dtype=torch.long)
    for step_idx, batch in enumerate(loader, start=1):
        if max_steps is not None and step_idx > max_steps:
            break
        features, labels = _batch_to_device(batch, device)
        if labels.numel() == 0:
            continue
        logits = model(features)
        loss = loss_fn(logits, labels)
        pair_count = int(labels.numel())
        total_loss += float(loss.item()) * pair_count
        total_pairs += pair_count
        matrix += confusion_matrix(labels.cpu(), logits.argmax(dim=-1).cpu(), args_num_classes())
    return total_loss / max(1, total_pairs), matrix


def args_num_classes() -> int:
    """Return the fixed number of explicit halfedge classes."""
    return 5


def _write_confusion_matrix(path: Path, matrix: torch.Tensor) -> None:
    """Write a confusion matrix with labelled rows and columns."""
    fieldnames = ["true_label"] + [BOND_TYPE_NAMES[idx] for idx in range(args_num_classes())]
    rows = []
    for class_idx in range(args_num_classes()):
        row: dict[str, object] = {"true_label": BOND_TYPE_NAMES[class_idx]}
        for pred_idx in range(args_num_classes()):
            row[BOND_TYPE_NAMES[pred_idx]] = int(matrix[class_idx, pred_idx].item())
        rows.append(row)
    _write_csv(path, rows, fieldnames)


def _plot_loss_curve(path: Path, history: list[dict[str, object]]) -> None:
    """Plot train and validation loss over epochs."""
    if not history:
        return
    epochs = [int(row["epoch"]) for row in history]
    train_loss = [float(row["train_loss"]) for row in history]
    val_loss = [float(row["val_loss"]) for row in history]
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, train_loss, marker="o", label="train")
    plt.plot(epochs, val_loss, marker="o", label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_metric_bars(path: Path, metric_rows: list[dict[str, object]], metric: str) -> None:
    """Plot one metric by split."""
    labels = [str(row["split"]) for row in metric_rows]
    values = [float(row[metric]) for row in metric_rows]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, values, color="#4C78A8")
    plt.ylabel(metric)
    plt.ylim(0, max(1.0, max(values, default=0.0) * 1.15))
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _write_manifest(
    path: Path,
    args: argparse.Namespace,
    feature_config: PairFeatureConfig,
    input_dim: int,
    device: torch.device,
) -> None:
    """Write model and data configuration for reproducibility."""
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "processed_dir": str(args.processed_dir),
        "feature_config": asdict(feature_config),
        "input_dim": input_dim,
        "num_edge_types": args_num_classes(),
        "bond_type_names": {str(key): value for key, value in BOND_TYPE_NAMES.items()},
        "class_weights": args.class_weights,
        "device": str(device),
        "args": _json_safe(vars(args)),
    }
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


def _json_safe(value: Any) -> Any:
    """Convert common config values into JSON-serializable objects."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_report(
    path: Path,
    args: argparse.Namespace,
    metric_rows: list[dict[str, object]],
    history: list[dict[str, object]],
    elapsed_sec: float,
) -> None:
    """Write a Markdown training report."""
    with path.open("w") as handle:
        handle.write("# Local Bond Predictor V0 Report\n\n")
        handle.write("## Purpose\n\n")
        handle.write(
            "Train a decoupled pair-MLP bond predictor on CrossDocked true halfedge "
            "labels before integrating any reconstruction backend.\n\n"
        )
        handle.write("## Inputs\n\n")
        handle.write(f"- processed_dir: `{args.processed_dir}`\n")
        handle.write(f"- train_max_samples: `{args.train_max_samples}`\n")
        handle.write(f"- val_max_samples: `{args.val_max_samples}`\n")
        handle.write(f"- test_max_samples: `{args.test_max_samples}`\n")
        handle.write(f"- train_negative_ratio: `{args.train_negative_ratio}`\n")
        handle.write(f"- pos_noise_std: `{args.pos_noise_std}`\n")
        handle.write(f"- epochs: `{args.epochs}`\n")
        handle.write(f"- elapsed_sec: `{elapsed_sec:.3f}`\n\n")

        handle.write("## Metrics\n\n")
        handle.write(
            "| split | loss | accuracy | macro_f1 | positive_precision | "
            "positive_recall | positive_f1 | no_bond_recall | aromatic_recall |\n"
        )
        handle.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in metric_rows:
            handle.write(
                f"| {row['split']} | {row['loss']} | {row['accuracy']} | "
                f"{row['macro_f1']} | {row['positive_precision']} | "
                f"{row['positive_recall']} | {row['positive_f1']} | "
                f"{row['no_bond_recall']} | {row['aromatic_recall']} |\n"
            )
        handle.write("\n")

        if history:
            handle.write("## Training History\n\n")
            handle.write("| epoch | train_loss | val_loss | val_macro_f1 | val_positive_recall |\n")
            handle.write("|---:|---:|---:|---:|---:|\n")
            for row in history:
                handle.write(
                    f"| {row['epoch']} | {row['train_loss']} | {row['val_loss']} | "
                    f"{row['val_macro_f1']} | {row['val_positive_recall']} |\n"
                )
            handle.write("\n")

        handle.write("## Files\n\n")
        handle.write("- [metrics.csv](metrics.csv)\n")
        handle.write("- [training_history.csv](training_history.csv)\n")
        handle.write("- [confusion_train.csv](confusion_train.csv)\n")
        handle.write("- [confusion_val.csv](confusion_val.csv)\n")
        handle.write("- [confusion_test.csv](confusion_test.csv)\n")
        handle.write("- [manifest.json](manifest.json)\n")
        handle.write("- [best_model.pt](best_model.pt)\n")
        handle.write("- [last_model.pt](last_model.pt)\n\n")
        handle.write("## Plots\n\n")
        handle.write("![loss curve](loss_curve.png)\n\n")
        handle.write("![macro f1](macro_f1_by_split.png)\n\n")
        handle.write("![positive recall](positive_recall_by_split.png)\n")


def train_and_evaluate(args: argparse.Namespace) -> Path:
    """Train the local bond predictor and write report artifacts."""
    started_at = time.perf_counter()
    _set_seed(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)

    feature_config = PairFeatureConfig(
        rbf_min=args.rbf_min,
        rbf_max=args.rbf_max,
        rbf_count=args.rbf_count,
        rbf_gamma=args.rbf_gamma,
    )
    train_dataset = _make_dataset("train", args, feature_config, is_train=True)
    val_dataset = _make_dataset("val", args, feature_config, is_train=False)
    test_dataset = _make_dataset("test", args, feature_config, is_train=False)
    atom_feature_dim = int(train_dataset.base_dataset[0]["lig_one_hot"].shape[-1])
    input_dim = pair_feature_dim(atom_feature_dim, feature_config)

    loaders = {
        "train": _make_loader(train_dataset, args, shuffle=True),
        "val": _make_loader(val_dataset, args, shuffle=False),
        "test": _make_loader(test_dataset, args, shuffle=False),
    }
    model = PairBondPredictor(
        input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_edge_types=args_num_classes(),
    ).to(device)
    class_weights = torch.tensor(args.class_weights, dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: list[dict[str, object]] = []
    best_val_macro_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_matrix = _run_train_epoch(
            model,
            loaders["train"],
            loss_fn,
            optimizer,
            device,
            max_steps=args.max_train_steps_per_epoch,
        )
        val_loss, val_matrix = _evaluate(
            model,
            loaders["val"],
            loss_fn,
            device,
            max_steps=args.max_eval_steps,
        )
        train_metrics = summarize_confusion(train_matrix)
        val_metrics = summarize_confusion(val_matrix)
        history.append({
            "epoch": epoch,
            "train_loss": f"{train_loss:.8f}",
            "train_macro_f1": f"{train_metrics.macro_f1:.8f}",
            "train_positive_recall": f"{train_metrics.positive_recall:.8f}",
            "val_loss": f"{val_loss:.8f}",
            "val_macro_f1": f"{val_metrics.macro_f1:.8f}",
            "val_positive_recall": f"{val_metrics.positive_recall:.8f}",
        })
        LOGGER.info(
            "epoch=%s train_loss=%.6f val_loss=%.6f val_macro_f1=%.4f val_positive_recall=%.4f",
            epoch,
            train_loss,
            val_loss,
            val_metrics.macro_f1,
            val_metrics.positive_recall,
        )
        if val_metrics.macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_metrics.macro_f1
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_dim": input_dim,
                "feature_config": asdict(feature_config),
                "args": vars(args),
                "epoch": epoch,
                "val_macro_f1": val_metrics.macro_f1,
            }, args.outdir / "best_model.pt")

    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": input_dim,
        "feature_config": asdict(feature_config),
        "args": vars(args),
        "epoch": args.epochs,
    }, args.outdir / "last_model.pt")

    metric_rows = []
    split_matrices: dict[str, torch.Tensor] = {}
    for split in SPLITS:
        loss, matrix = _evaluate(
            model,
            loaders[split],
            loss_fn,
            device,
            max_steps=args.max_eval_steps,
        )
        metrics = summarize_confusion(matrix)
        metric_rows.append(metrics.to_row(split, loss))
        split_matrices[split] = matrix

    _write_csv(args.outdir / "metrics.csv", metric_rows, list(metric_rows[0].keys()))
    if history:
        _write_csv(args.outdir / "training_history.csv", history, list(history[0].keys()))
    for split, matrix in split_matrices.items():
        _write_confusion_matrix(args.outdir / f"confusion_{split}.csv", matrix)
    _write_manifest(args.outdir / "manifest.json", args, feature_config, input_dim, device)
    _plot_loss_curve(args.outdir / "loss_curve.png", history)
    _plot_metric_bars(args.outdir / "macro_f1_by_split.png", metric_rows, "macro_f1")
    _plot_metric_bars(args.outdir / "positive_recall_by_split.png", metric_rows, "positive_recall")
    report_path = args.outdir / "local_bond_predictor_v0_report.md"
    _write_report(report_path, args, metric_rows, history, time.perf_counter() - started_at)
    return report_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/crossdocked_raw/processed_edge_noH_full"),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("analysis") / f"local_bond_predictor_v0_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument("--train-max-samples", type=int, default=5000)
    parser.add_argument("--val-max-samples", type=int, default=300)
    parser.add_argument("--test-max-samples", type=int, default=100)
    parser.add_argument("--train-negative-ratio", type=float, default=4.0)
    parser.add_argument("--pos-noise-std", type=float, default=0.05)
    parser.add_argument("--rbf-min", type=float, default=0.0)
    parser.add_argument("--rbf-max", type=float, default=6.0)
    parser.add_argument("--rbf-count", type=int, default=32)
    parser.add_argument("--rbf-gamma", type=float, default=None)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--class-weights", type=float, nargs=5, default=[0.1, 1.0, 1.0, 1.0, 1.0])
    parser.add_argument("--max-train-steps-per-epoch", type=int, default=None)
    parser.add_argument("--max-eval-steps", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    """Run the local bond predictor training CLI."""
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    report_path = train_and_evaluate(args)
    LOGGER.info("wrote report to %s", report_path)


if __name__ == "__main__":
    main()
