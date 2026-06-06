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

    def test_add_noise_returns_diffusion_training_triplet(self):
        transition = EdgeCategoricalTransition(
            EdgeTransitionConfig(num_classes=5, num_timesteps=10, prior="absorb")
        )
        edge_type = torch.tensor([0, 1, 2, 4], dtype=torch.long)
        one_hot_t, log_t, log_0 = transition.add_noise(edge_type, torch.tensor(3))

        self.assertEqual(tuple(one_hot_t.shape), (4, 5))
        self.assertEqual(tuple(log_t.shape), (4, 5))
        self.assertEqual(tuple(log_0.shape), (4, 5))
        self.assertTrue(torch.isfinite(log_t).all())

    def test_posterior_shape_and_t0_identity(self):
        transition = EdgeCategoricalTransition(
            EdgeTransitionConfig(num_classes=5, num_timesteps=10, prior="absorb")
        )
        edge_type = torch.tensor([0, 1, 2, 4], dtype=torch.long)
        log_v0 = transition.index_to_log_onehot(edge_type)
        _, log_vt = transition.q_vt_sample(log_v0, torch.tensor(5))
        posterior = transition.q_v_posterior(log_v0, log_vt, torch.tensor(5))
        posterior_t0 = transition.q_v_posterior(log_v0, log_v0, torch.tensor(0))

        self.assertEqual(tuple(posterior.shape), (4, 5))
        self.assertTrue(torch.isfinite(posterior).all())
        self.assertTrue(torch.allclose(posterior_t0, log_v0))

    def test_sample_init_uses_absorb_prior(self):
        transition = EdgeCategoricalTransition(
            EdgeTransitionConfig(num_classes=5, num_timesteps=10, prior="absorb")
        )
        sampled, one_hot, log_sampled = transition.sample_init(64)

        self.assertEqual(tuple(sampled.shape), (64,))
        self.assertEqual(tuple(one_hot.shape), (64, 5))
        self.assertEqual(tuple(log_sampled.shape), (64, 5))
        self.assertGreaterEqual(int(sampled.min()), 0)
        self.assertLess(int(sampled.max()), 5)

    def test_segment_schedule_accepts_diffgui_style_config(self):
        transition = EdgeCategoricalTransition(
            EdgeTransitionConfig(
                num_classes=5,
                num_timesteps=10,
                prior="absorb",
                beta_schedule="segment",
                time_segment=(600, 400),
                segment_diff=(
                    {"scale_start": 0.9999, "scale_end": 0.001, "width": 3},
                    {"scale_start": 0.001, "scale_end": 0.0001, "width": 2},
                ),
            )
        )

        self.assertEqual(tuple(transition.betas.shape), (10,))
        self.assertTrue(torch.isfinite(transition.betas).all())
        self.assertGreater(float(transition.prior[0]), float(transition.prior[1]))


if __name__ == "__main__":
    unittest.main()
