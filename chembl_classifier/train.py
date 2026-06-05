"""
Train the ChEMBL binary classifier: 3-layer MLP (2220→512→128→1).

Loads featurized dataset from chembl_classifier/models/dataset.pt,
trains with BCEWithLogitsLoss, saves best model.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, TensorDataset

# ── Project root ────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from chembl_classifier.model import ChemblClassifier  # noqa: E402

_CONFIG_PATH = _PROJECT_ROOT / "chembl_classifier" / "config.yaml"

logger = logging.getLogger(__name__)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    """Train one epoch.

    Returns:
        Average training loss.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).unsqueeze(1)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate model on a data split.

    Returns:
        Dict with loss, accuracy, precision, recall.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).unsqueeze(1)

        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        total_loss += loss.item()
        n_batches += 1

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        all_preds.append(preds.cpu())
        all_labels.append(y_batch.cpu())

    preds = torch.cat(all_preds).squeeze()
    labels = torch.cat(all_labels).squeeze()

    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()

    return {
        "loss": total_loss / max(n_batches, 1),
        "accuracy": (preds == labels).float().mean().item(),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
    }


def train(config_path: str | None = None) -> ChemblClassifier:
    """Train the classifier end-to-end.

    Args:
        config_path: Path to config YAML.

    Returns:
        Trained ChemblClassifier in eval mode.
    """
    if config_path is None:
        config_path = str(_CONFIG_PATH)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    output_cfg = cfg["output"]

    seed = train_cfg["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Load dataset ──────────────────────────────────────────────────────
    dataset_path = _PROJECT_ROOT / output_cfg["dataset_cache"]
    logger.info(f"Loading dataset from {dataset_path}")
    dataset: dict = torch.load(dataset_path, map_location="cpu", weights_only=False)

    X_train = torch.tensor(dataset["X_train"])
    y_train = torch.tensor(dataset["y_train"])
    X_val = torch.tensor(dataset["X_val"])
    y_val = torch.tensor(dataset["y_val"])
    X_test = torch.tensor(dataset["X_test"])
    y_test = torch.tensor(dataset["y_test"])

    # ── Clean NaN/Inf from features (RDKit descriptors for edge-case molecules) ──
    for name, X in [("train", X_train), ("val", X_val), ("test", X_test)]:
        nan_mask = torch.isnan(X) | torch.isinf(X)
        n_bad = nan_mask.any(dim=1).sum().item()
        if n_bad > 0:
            logger.warning(f"  {name}: {n_bad} rows with NaN/Inf — replacing with 0.0")
            X[nan_mask] = 0.0

    logger.info(
        f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}"
    )

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=train_cfg["batch_size"] * 2,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=train_cfg["batch_size"] * 2,
        shuffle=False,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = ChemblClassifier(
        input_dim=model_cfg["input_dim"],
        hidden_dims=tuple(model_cfg["hidden_dims"]),
        dropout_rates=tuple(model_cfg["dropout_rates"]),
    )
    model.to(device)
    logger.info(f"Model parameters: {model.n_params:,}")

    # ── Loss with class weight ────────────────────────────────────────────
    if train_cfg.get("auto_pos_weight", True):
        n_pos = int(y_train.sum().item())
        n_neg = len(y_train) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)]).to(device)
        logger.info(f"pos_weight = {pos_weight.item():.4f} (n_neg={n_neg}, n_pos={n_pos})")
    else:
        pos_weight = None

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── Optimizer ─────────────────────────────────────────────────────────
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    patience = train_cfg["early_stopping_patience"]
    max_epochs = train_cfg["max_epochs"]

    logger.info(f"Training {max_epochs} epochs (early stop patience={patience})")
    t_start = time.time()

    for epoch in range(1, max_epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            patience_counter = 0
            # Save best model
            model_path = _PROJECT_ROOT / output_cfg["model_dir"] / output_cfg["model_name"]
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(model_path), metadata={
                "val_loss": float(best_val_loss),
                "val_accuracy": float(val_metrics["accuracy"]),
                "epoch": epoch,
            })
        else:
            patience_counter += 1

        logger.info(
            f"Epoch {epoch:3d}/{max_epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"best_epoch={best_epoch}"
        )

        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    t_total = time.time() - t_start
    logger.info(f"Training completed in {t_total:.1f}s ({t_total/60:.1f} min)")

    # ── Final evaluation on test set ──────────────────────────────────────
    logger.info("Loading best model for test evaluation...")
    model = ChemblClassifier.from_checkpoint(
        str(_PROJECT_ROOT / output_cfg["model_dir"] / output_cfg["model_name"]),
        device=str(device),
    )

    test_metrics = evaluate(model, test_loader, criterion, device)
    logger.info(
        f"Test  — loss={test_metrics['loss']:.4f}, "
        f"acc={test_metrics['accuracy']:.4f}, "
        f"prec={test_metrics['precision']:.4f}, "
        f"recall={test_metrics['recall']:.4f}"
    )

    return model


def main() -> None:
    """Entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    train()


if __name__ == "__main__":
    main()
