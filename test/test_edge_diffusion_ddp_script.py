"""Tests for the standalone edge diffusion DDP training entrypoint."""

from __future__ import annotations

import ast
import inspect
import sys
import unittest
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIFFSBDD_ROOT = PROJECT_ROOT / "DiffSBDD"
ANALYSIS_ROOT = DIFFSBDD_ROOT / "analysis"
for extra in (PROJECT_ROOT, DIFFSBDD_ROOT, ANALYSIS_ROOT):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import train_edge_diffusion_ddp as edge_ddp


class TestEdgeDiffusionDdpScript(unittest.TestCase):
    """Verify the DDP config and decoupling boundaries."""

    def test_smoke_config_flattens_to_expected_args(self):
        config_path = PROJECT_ROOT / "launch" / "edge_diffusion_ddp_smoke.yaml"
        config = yaml.safe_load(config_path.read_text())

        args = edge_ddp._args_from_config(config, config_path)

        self.assertEqual(args.max_molecules, 64)
        self.assertEqual(args.val_molecules, 16)
        self.assertEqual(args.batch_size, 4)
        self.assertEqual(args.num_timesteps, 100)
        self.assertEqual(args.edge_prior, "uniform")
        self.assertEqual(args.outdir, Path("analysis/edge_diffusion_ddp_smoke_20260604"))

    def test_reduce_metric_dict_noops_without_ddp(self):
        state = edge_ddp.DistributedState(
            distributed=False,
            rank=0,
            local_rank=0,
            world_size=1,
            device=edge_ddp.torch.device("cpu"),
        )
        metrics = {"total": 1.5, "edge": 0.25}

        reduced = edge_ddp._reduce_metric_dict(metrics, state)

        self.assertEqual(reduced, metrics)

    def test_ddp_script_does_not_import_ddpo(self):
        tree = ast.parse(inspect.getsource(edge_ddp))
        imported_modules: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported_modules.append(node.module)
        self.assertFalse(
            any(module == "ddpo" or module.startswith("ddpo.") for module in imported_modules)
        )


if __name__ == "__main__":
    unittest.main()
