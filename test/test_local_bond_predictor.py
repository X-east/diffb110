"""Tests for the decoupled local bond predictor utilities."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIFFSBDD_ROOT = PROJECT_ROOT / "DiffSBDD"
ANALYSIS_ROOT = DIFFSBDD_ROOT / "analysis"
for extra in (PROJECT_ROOT, DIFFSBDD_ROOT, ANALYSIS_ROOT):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from local_bond_predictor.model import PairBondPredictor  # noqa: E402
from local_bond_predictor.dataset import (  # noqa: E402
    HalfedgePairDataset,
    PairFeatureConfig,
    build_pair_features,
    collate_pair_batches,
    pair_feature_dim,
)
from local_bond_predictor.metrics import confusion_matrix, summarize_confusion  # noqa: E402
from reconstruction_benchmark.methods import LocalBondPredictorMethod  # noqa: E402


def _one_hot(indices: list[int], width: int = 5) -> np.ndarray:
    """Build a compact one-hot feature array."""
    values = np.zeros((len(indices), width), dtype=np.float32)
    for row, col in enumerate(indices):
        values[row, col] = 1.0
    return values


def _write_toy_npz(path: Path) -> None:
    """Write a tiny processed ligand-pocket dataset with true bonds."""
    np.savez(
        path,
        names=np.array(["mol_a", "mol_b"], dtype=object),
        lig_coords=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.2, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.4, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        lig_one_hot=_one_hot([0, 1, 2, 0, 0]),
        lig_mask=np.array([0, 0, 0, 1, 1], dtype=np.float32),
        lig_bond_index=np.array(
            [
                [0, 1, 1, 2, 3, 4],
                [1, 0, 2, 1, 4, 3],
            ],
            dtype=np.int64,
        ),
        lig_bond_type=np.array([1, 1, 2, 2, 1, 1], dtype=np.int64),
        lig_bond_mask=np.array([0, 0, 0, 0, 1, 1], dtype=np.float32),
        num_lig_bonds=np.array([2, 1], dtype=np.int64),
        pocket_coords=np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=np.float32),
        pocket_one_hot=_one_hot([0, 1]),
        pocket_mask=np.array([0, 1], dtype=np.float32),
    )


class TestLocalBondPredictor(unittest.TestCase):
    """Verify local bond predictor data and metrics helpers."""

    def test_pair_feature_shape(self) -> None:
        """Pair features should include two atom one-hot vectors plus distance features."""
        config = PairFeatureConfig(rbf_count=4)
        coords = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=torch.float32)
        atom_features = torch.eye(2, dtype=torch.float32)
        halfedge_index = torch.tensor([[0], [1]], dtype=torch.long)
        features = build_pair_features(coords, atom_features, halfedge_index, config)
        self.assertEqual(features.shape, (1, pair_feature_dim(2, config)))
        self.assertAlmostEqual(float(features[0, 4].item()), 1.5, places=5)

    def test_dataset_negative_sampling_and_collate(self) -> None:
        """Dataset should keep positives and sample configured no-bond negatives."""
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = Path(tmpdir) / "toy.npz"
            _write_toy_npz(npz_path)
            dataset = HalfedgePairDataset(
                npz_path,
                negative_ratio=1.0,
                feature_config=PairFeatureConfig(rbf_count=4),
            )
            sample = dataset[0]
            self.assertEqual(int(sample["num_positive"]), 2)
            self.assertEqual(int(sample["num_pairs"]), 3)
            batch = collate_pair_batches([dataset[0], dataset[1]])
            self.assertEqual(batch["features"].shape[0], int(batch["num_pairs"].sum().item()))
            self.assertTrue((batch["labels"] > 0).sum().item() >= 3)

    def test_confusion_metrics(self) -> None:
        """Metric summaries should report class and positive-bond recall."""
        labels = torch.tensor([0, 0, 1, 2, 4])
        predictions = torch.tensor([0, 1, 1, 2, 1])
        matrix = confusion_matrix(labels, predictions, num_classes=5)
        metrics = summarize_confusion(matrix)
        self.assertGreater(metrics.accuracy, 0.0)
        self.assertGreater(metrics.positive_recall, 0.0)
        self.assertEqual(metrics.triple_recall, 0.0)

    def test_local_predictor_reconstruction_adapter(self) -> None:
        """A saved predictor checkpoint should reconstruct a simple two-atom bond."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "toy_bond_predictor.pt"
            config = PairFeatureConfig(rbf_count=4)
            input_dim = pair_feature_dim(atom_feature_dim=2, config=config)
            model = PairBondPredictor(input_dim, num_layers=1, num_edge_types=5)
            with torch.no_grad():
                linear = model.network[0]
                linear.weight.zero_()
                linear.bias.fill_(-10.0)
                linear.bias[1] = 10.0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": input_dim,
                    "feature_config": {
                        "rbf_min": config.rbf_min,
                        "rbf_max": config.rbf_max,
                        "rbf_count": config.rbf_count,
                        "rbf_gamma": config.rbf_gamma,
                    },
                    "args": {
                        "hidden_dim": 256,
                        "num_layers": 1,
                        "dropout": 0.0,
                    },
                },
                checkpoint_path,
            )
            method = LocalBondPredictorMethod(
                name="local_bond_toy",
                checkpoint=checkpoint_path,
                device="cpu",
            )
            attempt = method.reconstruct(
                coords=torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=torch.float32),
                atom_types=torch.tensor([0, 0], dtype=torch.long),
                dataset_info={"atom_decoder": ["C", "N"]},
            )
            self.assertEqual(attempt.build_error, "")
            self.assertIsNotNone(attempt.raw_mol)
            self.assertEqual(attempt.raw_mol.GetNumBonds(), 1)


if __name__ == "__main__":
    unittest.main()
