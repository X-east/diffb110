"""Tests for generated-cloud edge denoiser reconstruction diagnostics."""

from __future__ import annotations

import sys
import tempfile
import unittest
import math
from pathlib import Path

import torch
from rdkit import Chem
from rdkit.Geometry import Point3D

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIFFSBDD_ROOT = PROJECT_ROOT / "DiffSBDD"
ANALYSIS_ROOT = DIFFSBDD_ROOT / "analysis"
for extra in (PROJECT_ROOT, DIFFSBDD_ROOT, ANALYSIS_ROOT):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from edge_diffusion import EdgeDenoiserConfig, EdgeDenoiserNodeEdge  # noqa: E402
from reconstruction_benchmark.methods import (  # noqa: E402
    EdgeDenoiserReconstructionMethod,
    ReconstructionAttempt,
)
from reconstruction_benchmark.metrics import evaluate_reconstruction_attempt  # noqa: E402


class TestGeneratedReconstructionEdgeAdapter(unittest.TestCase):
    """Check edge-denoiser reconstruction and aromatic diagnostics."""

    def test_edge_denoiser_reconstruction_method_builds_bond(self) -> None:
        """A toy checkpoint with a single-bond bias should reconstruct one C-C bond."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "edge.pt"
            config = EdgeDenoiserConfig(
                num_atom_types=1,
                num_edge_types=5,
                num_timesteps=1000,
                hidden_dim=16,
                num_layers=1,
                distance_basis_dim=8,
            )
            model = EdgeDenoiserNodeEdge(config)
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
                final_linear = model.edge_decoder[-1]
                final_linear.bias.fill_(-10.0)
                final_linear.bias[1] = 10.0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "num_atom_types": config.num_atom_types,
                        "num_edge_types": config.num_edge_types,
                        "num_timesteps": config.num_timesteps,
                        "hidden_dim": config.hidden_dim,
                        "num_layers": config.num_layers,
                        "distance_basis_dim": config.distance_basis_dim,
                        "dropout": config.dropout,
                    },
                    "summary": {"model_kind": "node_edge"},
                },
                checkpoint_path,
            )
            method = EdgeDenoiserReconstructionMethod(
                name="edge_toy",
                checkpoint=checkpoint_path,
                device="cpu",
                init_mode="all_no_bond",
            )
            attempt = method.reconstruct(
                coords=torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=torch.float32),
                atom_types=torch.tensor([0, 0], dtype=torch.long),
                dataset_info={"atom_decoder": ["C"]},
            )
            self.assertEqual(attempt.build_error, "")
            self.assertIsNotNone(attempt.raw_mol)
            self.assertEqual(attempt.raw_mol.GetNumBonds(), 1)

    def test_benzene_like_ring_metric(self) -> None:
        """Sanitized benzene should have one normal benzene-like aromatic ring."""
        mol = Chem.MolFromSmiles("c1ccccc1")
        self.assertIsNotNone(mol)
        conformer = Chem.Conformer(6)
        coords = []
        for atom_idx in range(6):
            angle = 2.0 * math.pi * atom_idx / 6.0
            coords.append([math.cos(angle), math.sin(angle), 0.0])
            conformer.SetAtomPosition(atom_idx, Point3D(*coords[-1]))
        mol.AddConformer(conformer)
        row, sanitized = evaluate_reconstruction_attempt(
            sample_idx=0,
            attempt=ReconstructionAttempt(
                method="benzene",
                raw_mol=mol,
                build_error="",
                seconds=0.0,
            ),
            coords=torch.tensor(coords, dtype=torch.float32),
            atom_symbols=["C"] * 6,
            length_tolerance=0.5,
        )
        self.assertIsNotNone(sanitized)
        self.assertEqual(row["sanitize_success"], 1)
        self.assertEqual(row["aromatic_ring_count"], 1)
        self.assertEqual(row["aromatic_six_ring_count"], 1)
        self.assertEqual(row["benzene_like_ring_count"], 1)
        self.assertEqual(row["abnormal_aromatic_ring_count"], 0)


if __name__ == "__main__":
    unittest.main()
