"""Tests for DiffSBDD ligand bond and halfedge dataset fields."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIFFSBDD_ROOT = PROJECT_ROOT / "DiffSBDD"
for extra in (PROJECT_ROOT, DIFFSBDD_ROOT):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from DiffSBDD.dataset import ProcessedLigandPocketDataset
from DiffSBDD.analysis.edge_data_oracle_report import generate_report


def _one_hot(indices: list[int], width: int = 10) -> np.ndarray:
    """Build a compact one-hot feature array."""
    values = np.zeros((len(indices), width), dtype=np.float32)
    for row, col in enumerate(indices):
        values[row, col] = 1.0
    return values


class TestDiffSBDDHalfedgeDataset(unittest.TestCase):
    """Verify halfedge construction and batch index offsets."""

    def test_halfedge_construction_and_collate_offsets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = Path(tmpdir) / "toy.npz"
            np.savez(
                npz_path,
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
                pocket_coords=np.array(
                    [[0.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
                    dtype=np.float32,
                ),
                pocket_one_hot=_one_hot([0, 1]),
                pocket_mask=np.array([0, 1], dtype=np.float32),
            )

            dataset = ProcessedLigandPocketDataset(npz_path, center=False)
            sample_a = dataset[0]
            sample_b = dataset[1]

            self.assertTrue(torch.equal(
                sample_a["ligand_halfedge_index"],
                torch.tensor([[0, 0, 1], [1, 2, 2]]),
            ))
            self.assertTrue(torch.equal(
                sample_a["ligand_halfedge_type"],
                torch.tensor([1, 0, 2]),
            ))
            self.assertTrue(torch.equal(
                sample_b["ligand_halfedge_index"],
                torch.tensor([[0], [1]]),
            ))
            self.assertTrue(torch.equal(
                sample_b["ligand_halfedge_type"],
                torch.tensor([1]),
            ))

            batch = ProcessedLigandPocketDataset.collate_fn([sample_a, sample_b])
            self.assertTrue(torch.equal(
                batch["ligand_halfedge_index"],
                torch.tensor([[0, 0, 1, 3], [1, 2, 2, 4]]),
            ))
            self.assertTrue(torch.equal(
                batch["ligand_halfedge_type"],
                torch.tensor([1, 0, 2, 1]),
            ))
            self.assertTrue(torch.equal(
                batch["ligand_halfedge_type_batch"],
                torch.tensor([0, 0, 0, 1]),
            ))
            self.assertTrue(torch.equal(
                batch["lig_bond_index"],
                torch.tensor([[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]]),
            ))

            args = type("Args", (), {
                "npz_path": npz_path,
                "dataset": "crossdock",
                "outdir": Path(tmpdir) / "oracle",
                "max_samples": 2,
                "length_tolerance": 0.25,
            })()
            report_path = generate_report(args)
            self.assertTrue(report_path.exists())
            self.assertTrue((args.outdir / "bond_type_counts.csv").exists())
            self.assertTrue((args.outdir / "bond_type_distribution.png").exists())


if __name__ == "__main__":
    unittest.main()
