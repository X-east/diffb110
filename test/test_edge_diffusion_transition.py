"""Tests for prototype halfedge categorical transitions."""

from __future__ import annotations

import unittest

import torch

from DiffSBDD.edge_diffusion import EdgeCategoricalTransition, EdgeTransitionConfig


class TestEdgeCategoricalTransition(unittest.TestCase):
    """Verify categorical edge noising shape and label constraints."""

    def test_q_sample_preserves_shape_and_range(self):
        transition = EdgeCategoricalTransition(
            EdgeTransitionConfig(num_classes=5, num_timesteps=10)
        )
        edge_type = torch.tensor([0, 1, 2, 3, 4, 0, 1], dtype=torch.long)
        noised = transition.q_sample(edge_type, torch.tensor(5))

        self.assertEqual(noised.shape, edge_type.shape)
        self.assertGreaterEqual(int(noised.min()), 0)
        self.assertLess(int(noised.max()), 5)

    def test_one_hot_and_nll_shapes(self):
        transition = EdgeCategoricalTransition()
        edge_type = torch.tensor([0, 1, 4], dtype=torch.long)
        one_hot = transition.one_hot(edge_type)
        logits = torch.zeros((3, 5), dtype=torch.float32)
        nll = transition.negative_log_likelihood(logits, edge_type)

        self.assertEqual(tuple(one_hot.shape), (3, 5))
        self.assertEqual(tuple(nll.shape), (3,))

    def test_invalid_label_raises(self):
        transition = EdgeCategoricalTransition(EdgeTransitionConfig(num_classes=5))
        with self.assertRaisesRegex(ValueError, "outside configured classes"):
            transition.q_sample(torch.tensor([5], dtype=torch.long), torch.tensor(0))


if __name__ == "__main__":
    unittest.main()
