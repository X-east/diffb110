"""Tests for the ChEMBL D-MPNN reward integration."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from rdkit import Chem

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "DiffSBDD"))

sys.modules.setdefault("wandb", type(sys)("wandb"))
sys.modules.setdefault("diffusion", type(sys)("diffusion"))
sys.modules["DiffSBDD.lightning_modules"] = sys.modules.get(
    "DiffSBDD.lightning_modules"
) or type("DiffSBDD.lightning_modules", (), {"LigandPocketDDPM": None})

from chembl_classifier.dmpnn.model import DMPNNClassifier, default_dmpnn_config  # noqa: E402
from ddpo.field_registry import reduce_registered_epoch_aggregates  # noqa: E402
from ddpo.rewards.chembl_dmpnn_reward import ChemblDMPNNReward  # noqa: E402
from ddpo.runtime.protocol import DataProto  # noqa: E402
from ddpo.runtime.recording.run_context import RunContext  # noqa: E402
from ddpo.runtime.recording.writers.metrics_writer import MetricsWriter  # noqa: E402
from ddpo.runtime.train_helpers import PreparedSamples, build_epoch_facts, compute_batch_stats  # noqa: E402


def _identity_sum(value: float, _dist_ctx: object) -> float:
    return float(value)


def _identity_min(value: float, _dist_ctx: object) -> float:
    return float(value)


def _identity_max(value: float, _dist_ctx: object) -> float:
    return float(value)


def _write_tiny_checkpoint(path: Path) -> None:
    config = default_dmpnn_config(
        hidden_size=16,
        depth=2,
        ffn_hidden_size=16,
        ffn_num_layers=2,
        dropout=0.0,
    )
    DMPNNClassifier(config).save(str(path), metadata={"test": True})


def test_chembl_dmpnn_reward_scores_valid_and_penalizes_invalid(tmp_path: Path) -> None:
    """Reward writes finite probability/reward fields and honors invalid flags."""
    checkpoint = tmp_path / "chembl_dmpnn.pt"
    _write_tiny_checkpoint(checkpoint)
    reward = ChemblDMPNNReward(
        model_path=str(checkpoint),
        device="cpu",
        failure_penalty=-2.0,
        batch_size=2,
    )
    data = DataProto.from_dict(
        tensors={
            "x": torch.zeros((3, 4, 3), dtype=torch.float32),
            "nodesxsample": torch.tensor([3, 0, 6], dtype=torch.long),
        },
        non_tensors={
            "rdmols": [Chem.MolFromSmiles("CCO"), None, Chem.MolFromSmiles("c1ccccc1")],
            "rdkit_valid": np.asarray([1, 0, 1], dtype=np.int64),
            "smiles": ["CCO", "", "c1ccccc1"],
        },
    )

    scored = reward.calculate_rewards(data)

    assert "chembl_dmpnn_reward" in scored.batch
    assert "chembl_dmpnn_prob" in scored.batch
    assert torch.isfinite(scored.batch["chembl_dmpnn_reward"]).all()
    assert torch.isfinite(scored.batch["chembl_dmpnn_prob"]).all()
    assert scored.batch["chembl_dmpnn_reward"][1].item() == -2.0


def test_chembl_dmpnn_fields_reach_epoch_summary(tmp_path: Path) -> None:
    """FieldSpec aggregation writes the D-MPNN columns into training_summary.csv."""
    reward = ChemblDMPNNReward(model_path=str(tmp_path / "missing.pt"), device="cpu")
    specs = reward.field_specs()
    assert any(spec.name == "chembl_dmpnn_reward" for spec in specs)
    assert any(spec.name == "avg_chembl_dmpnn_reward" for spec in specs)

    data = SimpleNamespace(
        batch={
            "x": torch.zeros(4, 2, 3),
            "nodesxsample": torch.full((4,), 2, dtype=torch.long),
            "chembl_dmpnn_reward": torch.tensor([0.5, -1.0, 1.0, 0.0]),
            "chembl_dmpnn_prob": torch.tensor([0.7, 0.0, 0.9, 0.4]),
            "reward_total": torch.zeros(4),
            "reward_total_raw": torch.zeros(4),
            "reward_total_clamped": torch.zeros(4),
        },
        non_tensor_batch={
            "rdkit_valid": [1, 0, 1, 1],
            "failure_type": ["success", "sanitize_failed", "success", "success"],
            "smiles": ["CCO", "bad", "CCN", "CCC"],
            "rdmols": [None, None, None, None],
        },
    )
    prepared = PreparedSamples(samples=data, filtering_ratio=0.0, novelty_penalty_ratio=0.0)
    stats = compute_batch_stats(
        prepared,
        config={"reward": {"weights": {"chembl_dmpnn": 0.5}}},
        field_specs=specs,
    )
    reduced = reduce_registered_epoch_aggregates(
        stats.registered_epoch_aggregates or {},
        None,
        reduce_sum_scalar=_identity_sum,
        reduce_min_scalar=_identity_min,
        reduce_max_scalar=_identity_max,
    )
    assert reduced["avg_chembl_dmpnn_reward"] == 0.125
    assert round(reduced["avg_chembl_dmpnn_prob"], 6) == round((0.7 + 0.9 + 0.4) / 3, 6)

    stats.extra_epoch_metrics = reduced
    facts = build_epoch_facts(
        runtime=SimpleNamespace(
            config={"reward": {"weights": {"chembl_dmpnn": 0.5}}},
            max_epochs=1,
            checkpoint_interval=10,
        ),
        epoch=1,
        metrics={"PPO_Loss": 0.25},
        batch_stats=stats,
        prepared=prepared,
    )
    run_context = RunContext(
        run_id="test",
        run_name="test",
        started_at="test",
        output_dir=str(tmp_path),
        logs_dir=str(tmp_path / "logs"),
        epochs_dir=str(tmp_path / "epochs"),
        checkpoints_dir=str(tmp_path / "checkpoints"),
        lightning_logs_dir=str(tmp_path / "lightning_logs"),
        config_snapshot_path=str(tmp_path / "config.yaml"),
        training_summary_csv_path=str(tmp_path / "training_summary.csv"),
        run_manifest_path=str(tmp_path / "run_manifest.json"),
        checkpoint_manifest_path=str(tmp_path / "checkpoint_manifest.csv"),
    )
    writer = MetricsWriter(run_context)
    writer.write_epoch_summary(facts)

    with open(run_context.training_summary_csv_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["avg_chembl_dmpnn_reward"] == "0.125000"
    assert rows[0]["avg_chembl_dmpnn_prob"] == "0.666667"
