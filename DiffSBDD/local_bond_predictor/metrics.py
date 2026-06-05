"""Classification metrics for local bond prediction."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BondClassificationMetrics:
    """Summary metrics for a bond-type classifier."""

    accuracy: float
    macro_f1: float
    positive_precision: float
    positive_recall: float
    positive_f1: float
    no_bond_recall: float
    single_recall: float
    double_recall: float
    triple_recall: float
    aromatic_recall: float

    def to_row(self, split: str, loss: float) -> dict[str, object]:
        """Return a CSV-ready metric row."""
        return {
            "split": split,
            "loss": f"{loss:.8f}",
            "accuracy": f"{self.accuracy:.8f}",
            "macro_f1": f"{self.macro_f1:.8f}",
            "positive_precision": f"{self.positive_precision:.8f}",
            "positive_recall": f"{self.positive_recall:.8f}",
            "positive_f1": f"{self.positive_f1:.8f}",
            "no_bond_recall": f"{self.no_bond_recall:.8f}",
            "single_recall": f"{self.single_recall:.8f}",
            "double_recall": f"{self.double_recall:.8f}",
            "triple_recall": f"{self.triple_recall:.8f}",
            "aromatic_recall": f"{self.aromatic_recall:.8f}",
        }


def confusion_matrix(labels: torch.Tensor, predictions: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Compute a confusion matrix with rows=true labels and columns=predicted labels."""
    labels = labels.view(-1).long()
    predictions = predictions.view(-1).long()
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.long)
    valid = (labels >= 0) & (labels < num_classes) & (predictions >= 0) & (predictions < num_classes)
    flat = labels[valid] * num_classes + predictions[valid]
    counts = torch.bincount(flat, minlength=num_classes * num_classes)
    return counts.view(num_classes, num_classes)


def precision_recall_f1(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    """Return precision, recall, and f1 with zero-safe denominators."""
    precision = tp / max(1.0, tp + fp)
    recall = tp / max(1.0, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    return precision, recall, f1


def summarize_confusion(matrix: torch.Tensor) -> BondClassificationMetrics:
    """Summarize a confusion matrix into bond classification metrics."""
    matrix_f = matrix.float()
    total = float(matrix_f.sum().item())
    accuracy = float(matrix_f.diag().sum().item()) / max(1.0, total)

    class_f1_values = []
    class_recalls = []
    for class_idx in range(matrix.shape[0]):
        tp = float(matrix_f[class_idx, class_idx].item())
        fp = float(matrix_f[:, class_idx].sum().item() - tp)
        fn = float(matrix_f[class_idx, :].sum().item() - tp)
        _, recall, f1 = precision_recall_f1(tp, fp, fn)
        class_recalls.append(recall)
        class_f1_values.append(f1)
    macro_f1 = sum(class_f1_values) / max(1, len(class_f1_values))

    positive_true = matrix_f[1:, :].sum()
    positive_pred = matrix_f[:, 1:].sum()
    positive_tp = matrix_f[1:, 1:].sum()
    positive_precision, positive_recall, positive_f1 = precision_recall_f1(
        float(positive_tp.item()),
        float((positive_pred - positive_tp).item()),
        float((positive_true - positive_tp).item()),
    )
    return BondClassificationMetrics(
        accuracy=accuracy,
        macro_f1=macro_f1,
        positive_precision=positive_precision,
        positive_recall=positive_recall,
        positive_f1=positive_f1,
        no_bond_recall=class_recalls[0],
        single_recall=class_recalls[1],
        double_recall=class_recalls[2],
        triple_recall=class_recalls[3],
        aromatic_recall=class_recalls[4],
    )
