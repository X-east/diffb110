"""Train the D-MPNN ChEMBL good/bad classifier from a dataset CSV."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from rdkit import Chem
from torch.utils.data import DataLoader, Dataset

from chembl_classifier.dmpnn.data import build_dataset_from_config
from chembl_classifier.dmpnn.model import DMPNNClassifier, default_dmpnn_config, mols_to_graph_tensors

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DMPNNExample:
    """One classifier training example."""

    smiles: str
    label: float
    sample_weight: float


class DMPNNDataset(Dataset):
    """RDKit-backed in-memory dataset for D-MPNN training."""

    def __init__(self, rows: pd.DataFrame) -> None:
        self.examples: list[DMPNNExample] = [
            DMPNNExample(
                smiles=str(row["smiles"]),
                label=float(row["label"]),
                sample_weight=float(row.get("sample_weight", 1.0)),
            )
            for _, row in rows.iterrows()
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> DMPNNExample:
        return self.examples[index]


def collate_examples(examples: list[DMPNNExample]) -> dict[str, Any]:
    """Collate examples while keeping RDKit molecule construction localized."""
    mols = [Chem.MolFromSmiles(example.smiles) for example in examples]
    keep = [idx for idx, mol in enumerate(mols) if mol is not None]
    if not keep:
        raise ValueError("D-MPNN batch has no valid RDKit molecules")
    kept_examples = [examples[idx] for idx in keep]
    kept_mols = [mols[idx] for idx in keep]
    return {
        "mols": kept_mols,
        "labels": torch.tensor([item.label for item in kept_examples], dtype=torch.float32),
        "weights": torch.tensor([item.sample_weight for item in kept_examples], dtype=torch.float32),
        "smiles": [item.smiles for item in kept_examples],
    }


def train(config_path: str | Path) -> DMPNNClassifier:
    """Train a D-MPNN classifier from a YAML config."""
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    project_root = config_path.resolve().parents[1]

    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})
    output_cfg = cfg.get("output", {})
    dataset_csv = _resolve_path(project_root, output_cfg.get("dataset_csv"))
    if dataset_csv is None:
        raise ValueError("D-MPNN config requires output.dataset_csv")
    if not dataset_csv.exists() or bool(output_cfg.get("rebuild_dataset", False)):
        build_dataset_from_config(config_path)

    rows = pd.read_csv(dataset_csv)
    train_rows = rows[rows["split"].eq("train")].reset_index(drop=True)
    val_rows = rows[rows["split"].eq("val")].reset_index(drop=True)
    test_rows = rows[rows["split"].eq("test")].reset_index(drop=True)
    if len(train_rows) == 0 or len(val_rows) == 0:
        raise ValueError("D-MPNN dataset must contain non-empty train and val splits")

    seed = int(train_cfg.get("seed", 45))
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(str(train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")))
    logger.info("D-MPNN device: %s", device)

    config = default_dmpnn_config(**model_cfg)
    model = DMPNNClassifier(config).to(device)
    logger.info("D-MPNN parameters: %d", model.n_params)

    train_loader = DataLoader(
        DMPNNDataset(train_rows),
        batch_size=int(train_cfg.get("batch_size", 64)),
        shuffle=True,
        collate_fn=collate_examples,
        num_workers=int(train_cfg.get("num_workers", 0)),
    )
    val_loader = DataLoader(
        DMPNNDataset(val_rows),
        batch_size=int(train_cfg.get("batch_size", 64)),
        shuffle=False,
        collate_fn=collate_examples,
        num_workers=0,
    )
    test_loader = DataLoader(
        DMPNNDataset(test_rows),
        batch_size=int(train_cfg.get("batch_size", 64)),
        shuffle=False,
        collate_fn=collate_examples,
        num_workers=0,
    )

    pos_weight = _pos_weight(train_rows) if bool(train_cfg.get("auto_pos_weight", True)) else None
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device) if pos_weight else None,
        reduction="none",
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-5)),
    )

    output_dir = _resolve_path(project_root, output_cfg.get("output_dir", "chembl_classifier/models/dmpnn_v0_1"))
    model_path = _resolve_path(project_root, output_cfg.get("model_path", "chembl_classifier/models/chembl_dmpnn_v0_1.pt"))
    if output_dir is None or model_path is None:
        raise ValueError("D-MPNN output paths could not be resolved")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.csv"

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    max_epochs = int(train_cfg.get("max_epochs", 80))
    patience = int(train_cfg.get("early_stopping_patience", 12))
    start_time = time.time()

    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_loss",
                "val_accuracy",
                "val_precision",
                "val_recall",
                "val_auc",
                "best_epoch",
            ],
        )
        writer.writeheader()
        for epoch in range(1, max_epochs + 1):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_metrics = evaluate(model, val_loader, criterion, device)
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_epoch = epoch
                patience_counter = 0
                model.save(
                    str(model_path),
                    metadata={
                        "epoch": epoch,
                        "val_metrics": val_metrics,
                        "dataset_csv": str(dataset_csv),
                        "config_path": str(config_path),
                    },
                )
            else:
                patience_counter += 1

            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": f"{train_loss:.6f}",
                    "val_loss": f"{val_metrics['loss']:.6f}",
                    "val_accuracy": f"{val_metrics['accuracy']:.6f}",
                    "val_precision": f"{val_metrics['precision']:.6f}",
                    "val_recall": f"{val_metrics['recall']:.6f}",
                    "val_auc": f"{val_metrics['auc']:.6f}",
                    "best_epoch": best_epoch,
                }
            )
            handle.flush()
            logger.info(
                "Epoch %03d/%03d train_loss=%.4f val_loss=%.4f val_acc=%.4f best=%d",
                epoch,
                max_epochs,
                train_loss,
                val_metrics["loss"],
                val_metrics["accuracy"],
                best_epoch,
            )
            if patience_counter >= patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    best_model = DMPNNClassifier.from_checkpoint(str(model_path), device=device)
    test_metrics = evaluate(best_model, test_loader, criterion, device) if len(test_rows) else {}
    summary = {
        "model_path": str(model_path),
        "dataset_csv": str(dataset_csv),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "test_metrics": test_metrics,
        "elapsed_seconds": time.time() - start_time,
        "model_config": asdict(config),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return best_model


def train_epoch(
    model: DMPNNClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one training epoch."""
    model.train()
    total_loss = 0.0
    n_weight = 0.0
    for batch in loader:
        graph = mols_to_graph_tensors(batch["mols"], device=device)
        labels = batch["labels"].to(device)
        weights = batch["weights"].to(device)
        optimizer.zero_grad()
        logits = model(graph)
        loss_raw = criterion(logits, labels)
        loss = (loss_raw * weights).sum() / torch.clamp(weights.sum(), min=1.0)
        loss.backward()
        optimizer.step()
        total_loss += float((loss_raw.detach() * weights).sum().item())
        n_weight += float(weights.sum().item())
    return total_loss / max(n_weight, 1.0)


@torch.no_grad()
def evaluate(
    model: DMPNNClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate loss and threshold metrics."""
    model.eval()
    total_loss = 0.0
    n_weight = 0.0
    probs_all: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []
    for batch in loader:
        graph = mols_to_graph_tensors(batch["mols"], device=device)
        labels = batch["labels"].to(device)
        weights = batch["weights"].to(device)
        logits = model(graph)
        loss_raw = criterion(logits, labels)
        total_loss += float((loss_raw * weights).sum().item())
        n_weight += float(weights.sum().item())
        probs_all.append(torch.sigmoid(logits).detach().cpu())
        labels_all.append(labels.detach().cpu())

    probs = torch.cat(probs_all) if probs_all else torch.empty(0)
    labels = torch.cat(labels_all) if labels_all else torch.empty(0)
    preds = (probs >= 0.5).float()
    tp = int(((preds == 1) & (labels == 1)).sum().item())
    fp = int(((preds == 1) & (labels == 0)).sum().item())
    fn = int(((preds == 0) & (labels == 1)).sum().item())
    return {
        "loss": total_loss / max(n_weight, 1.0),
        "accuracy": float((preds == labels).float().mean().item()) if labels.numel() else 0.0,
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "auc": _binary_auc(labels.numpy(), probs.numpy()) if labels.numel() else 0.0,
    }


def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute ROC AUC without requiring sklearn."""
    labels = labels.astype(np.float64)
    scores = scores.astype(np.float64)
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.0
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    pos_rank_sum = float(ranks[labels == 1].sum())
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _pos_weight(rows: pd.DataFrame) -> float:
    n_pos = int(rows["label"].sum())
    n_neg = int(len(rows) - n_pos)
    return float(n_neg / max(n_pos, 1))


def _resolve_path(project_root: Path, value: Any) -> Path | None:
    if value is None:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else project_root / path


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Train ChEMBL D-MPNN classifier.")
    parser.add_argument("--config", required=True, help="Path to D-MPNN YAML config.")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    train(args.config)


if __name__ == "__main__":
    main()
