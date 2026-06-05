"""Tests for edge denoiser evaluation helpers."""

from __future__ import annotations

import unittest

import torch

from DiffSBDD.edge_diffusion import (
    batch_confusion,
    classification_metrics,
    compare_halfedge_bond_maps,
)


class TestEdgeDenoiserEvaluate(unittest.TestCase):
    """Check core evaluation metric helpers."""

    def test_batch_confusion(self):
        target = torch.tensor([0, 1, 1, 4])
        pred = torch.tensor([0, 1, 2, 4])

        confusion = batch_confusion(target, pred, num_classes=5)

        self.assertEqual(int(confusion[0, 0]), 1)
        self.assertEqual(int(confusion[1, 1]), 1)
        self.assertEqual(int(confusion[1, 2]), 1)
        self.assertEqual(int(confusion[4, 4]), 1)

    def test_classification_metrics(self):
        confusion = torch.tensor([
            [8, 2, 0, 0, 0],
            [1, 7, 2, 0, 0],
            [0, 1, 9, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 9],
        ])

        metrics, class_rows = classification_metrics(confusion, eval_loss=0.5)

        self.assertAlmostEqual(metrics["accuracy"], 33 / 40)
        self.assertEqual(len(class_rows), 5)
        self.assertEqual(class_rows[0]["class_name"], "no_bond")
        self.assertAlmostEqual(class_rows[1]["recall"], 0.7)
        self.assertEqual(class_rows[3]["support"], 0)

    def test_compare_halfedge_bond_maps(self):
        halfedge_index = torch.tensor([[0, 0, 1], [1, 2, 2]])
        target_edge_type = torch.tensor([1, 0, 2])
        pred_edge_type = torch.tensor([1, 1, 2])

        metrics = compare_halfedge_bond_maps(
            halfedge_index,
            target_edge_type,
            pred_edge_type,
        )

        self.assertEqual(metrics["missing_bond_count"], 0)
        self.assertEqual(metrics["extra_bond_count"], 1)
        self.assertAlmostEqual(metrics["bond_pair_precision"], 2 / 3)
        self.assertAlmostEqual(metrics["bond_pair_recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
