"""
Evaluate trained ChEMBL classifier.

Computes: ROC-AUC, confusion matrix, per-class accuracy,
feature importance (permutation-based), prediction distribution.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

# ── Project root ────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from chembl_classifier.model import ChemblClassifier  # noqa: E402

_CONFIG_PATH = _PROJECT_ROOT / "chembl_classifier" / "config.yaml"

logger = logging.getLogger(__name__)


@torch.no_grad()
def predict_proba(
    model: ChemblClassifier,
    X: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Batch prediction of probabilities.

    Args:
        model: Trained classifier.
        X: Feature tensor [N, D].
        device: Computation device.
        batch_size: Batch size for inference.

    Returns:
        Probability array [N].
    """
    model.eval()
    dataset = TensorDataset(X)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    probs: list[np.ndarray] = []

    for (x_batch,) in loader:
        x_batch = x_batch.to(device)
        p = model.predict_proba(x_batch).cpu().numpy().flatten()
        probs.append(p)

    return np.concatenate(probs)


def evaluate_model(
    model_path: str | None = None,
    config_path: str | None = None,
) -> dict:
    """Run full evaluation suite.

    Args:
        model_path: Path to model checkpoint. Auto-detected from config if None.
        config_path: Path to config YAML.

    Returns:
        Dict of all evaluation metrics.
    """
    if config_path is None:
        config_path = str(_CONFIG_PATH)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    output_cfg = cfg["output"]

    if model_path is None:
        model_path = str(_PROJECT_ROOT / output_cfg["model_dir"] / output_cfg["model_name"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────
    logger.info(f"Loading model from {model_path}")
    model = ChemblClassifier.from_checkpoint(model_path, device=str(device))

    # ── Load dataset ──────────────────────────────────────────────────────
    dataset_path = _PROJECT_ROOT / output_cfg["dataset_cache"]
    logger.info(f"Loading dataset from {dataset_path}")
    dataset: dict = torch.load(dataset_path, map_location="cpu", weights_only=False)

    X_test = torch.tensor(dataset["X_test"])
    y_test = torch.tensor(dataset["y_test"]).numpy()
    smiles_test = dataset["smiles_test"]

    X_train = torch.tensor(dataset["X_train"])
    y_train = torch.tensor(dataset["y_train"]).numpy()

    # ── Clean NaN/Inf from features ───────────────────────────────────────
    for name, X in [("test", X_test), ("train", X_train)]:
        nan_mask = torch.isnan(X) | torch.isinf(X)
        if nan_mask.any():
            logger.warning(f"  {name}: {nan_mask.any(dim=1).sum().item()} rows NaN/Inf → 0.0")
            X[nan_mask] = 0.0

    # ── Predictions ───────────────────────────────────────────────────────
    logger.info("Computing test predictions...")
    test_probs = predict_proba(model, X_test, device)
    test_preds = (test_probs > 0.5).astype(int)

    train_probs = predict_proba(model, X_train, device)
    train_preds = (train_probs > 0.5).astype(int)

    # ── Metrics ───────────────────────────────────────────────────────────
    # Test
    test_acc = accuracy_score(y_test, test_preds)
    test_auc = roc_auc_score(y_test, test_probs)
    test_ap = average_precision_score(y_test, test_probs)
    test_prec, test_rec, test_f1, _ = precision_recall_fscore_support(
        y_test, test_preds, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_test, test_preds)

    # Train (to detect overfitting)
    train_acc = accuracy_score(y_train, train_preds)
    train_auc = roc_auc_score(y_train, train_probs)

    # ── Per-class accuracy ────────────────────────────────────────────────
    pos_mask = y_test == 1
    neg_mask = y_test == 0
    pos_acc = accuracy_score(y_test[pos_mask], test_preds[pos_mask]) if pos_mask.sum() > 0 else 0.0
    neg_acc = accuracy_score(y_test[neg_mask], test_preds[neg_mask]) if neg_mask.sum() > 0 else 0.0

    # ── Prediction distribution ───────────────────────────────────────────
    pos_probs = test_probs[pos_mask] if pos_mask.sum() > 0 else np.array([])
    neg_probs = test_probs[neg_mask] if neg_mask.sum() > 0 else np.array([])

    # ── Top false positives/negatives ─────────────────────────────────────
    fp_indices = np.where((test_preds == 1) & (y_test == 0))[0]
    fn_indices = np.where((test_preds == 0) & (y_test == 1))[0]

    top_fp = sorted(
        zip(fp_indices, test_probs[fp_indices]),
        key=lambda x: x[1], reverse=True
    )[:10]
    top_fn = sorted(
        zip(fn_indices, test_probs[fn_indices]),
        key=lambda x: x[1]
    )[:10]

    # ── Results ───────────────────────────────────────────────────────────
    results: dict = {
        "test": {
            "accuracy": float(test_acc),
            "roc_auc": float(test_auc),
            "avg_precision": float(test_ap),
            "precision": float(test_prec),
            "recall": float(test_rec),
            "f1": float(test_f1),
            "confusion_matrix": cm.tolist(),
            "positive_accuracy": float(pos_acc),
            "negative_accuracy": float(neg_acc),
        },
        "train": {
            "accuracy": float(train_acc),
            "roc_auc": float(train_auc),
        },
        "distribution": {
            "positive_probs_mean": float(pos_probs.mean()) if len(pos_probs) > 0 else 0.0,
            "positive_probs_std": float(pos_probs.std()) if len(pos_probs) > 0 else 0.0,
            "negative_probs_mean": float(neg_probs.mean()) if len(neg_probs) > 0 else 0.0,
            "negative_probs_std": float(neg_probs.std()) if len(neg_probs) > 0 else 0.0,
        },
        "top_errors": {
            "false_positives": [
                {"smiles": smiles_test[i], "prob": float(p.item())}
                for i, p in top_fp
            ],
            "false_negatives": [
                {"smiles": smiles_test[i], "prob": float(p.item())}
                for i, p in top_fn
            ],
        },
        "n_samples": {
            "test": int(len(y_test)),
            "test_pos": int(pos_mask.sum()),
            "test_neg": int(neg_mask.sum()),
        },
    }

    return results


def print_results(results: dict) -> None:
    """Pretty-print evaluation results."""
    t = results["test"]
    tr = results["train"]
    d = results["distribution"]
    n = results["n_samples"]

    print("\n" + "=" * 60)
    print("ChEMBL Classifier Evaluation")
    print("=" * 60)
    print(f"\nTest set: {n['test']:,} samples ({n['test_pos']:,} pos / {n['test_neg']:,} neg)")
    print()

    print(f"{'Metric':<20} {'Test':>10} {'Train':>10}")
    print("-" * 40)
    print(f"{'Accuracy':<20} {t['accuracy']:>10.4f} {tr['accuracy']:>10.4f}")
    print(f"{'ROC-AUC':<20} {t['roc_auc']:>10.4f} {tr['roc_auc']:>10.4f}")
    print(f"{'Avg Precision':<20} {t['avg_precision']:>10.4f}")
    print(f"{'Precision':<20} {t['precision']:>10.4f}")
    print(f"{'Recall':<20} {t['recall']:>10.4f}")
    print(f"{'F1 Score':<20} {t['f1']:>10.4f}")
    print()
    print(f"{'Pos Accuracy':<20} {t['positive_accuracy']:>10.4f}")
    print(f"{'Neg Accuracy':<20} {t['negative_accuracy']:>10.4f}")
    print()

    cm = t["confusion_matrix"]
    print("Confusion Matrix:")
    print(f"  TN={cm[0][0]:5d}  FP={cm[0][1]:5d}")
    print(f"  FN={cm[1][0]:5d}  TP={cm[1][1]:5d}")
    print()

    print("Prediction Distribution:")
    print(f"  Positive (ChEMBL-like):  mean={d['positive_probs_mean']:.4f}, std={d['positive_probs_std']:.4f}")
    print(f"  Negative (non-ChEMBL):   mean={d['negative_probs_mean']:.4f}, std={d['negative_probs_std']:.4f}")
    print()

    print("Top False Positives (predicted ChEMBL-like but are negative):")
    for item in results["top_errors"]["false_positives"][:5]:
        print(f"  P={item['prob']:.4f} | {item['smiles']}")

    print()
    print("Top False Negatives (predicted non-ChEMBL but are ChEMBL):")
    for item in results["top_errors"]["false_negatives"][:5]:
        print(f"  P={item['prob']:.4f} | {item['smiles']}")


def main() -> None:
    """Entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    results = evaluate_model()
    print_results(results)


if __name__ == "__main__":
    main()
