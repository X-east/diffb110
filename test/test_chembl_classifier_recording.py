"""Recording-chain tests for ChEMBL classifier reward metrics."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "DiffSBDD"))

sys.modules.setdefault("wandb", type(sys)("wandb"))
sys.modules.setdefault("diffusion", type(sys)("diffusion"))
sys.modules["DiffSBDD.lightning_modules"] = sys.modules.get(
    "DiffSBDD.lightning_modules"
) or type("DiffSBDD.lightning_modules", (), {"LigandPocketDDPM": None})

from ddpo.field_registry import reduce_registered_epoch_aggregates  # noqa: E402
from ddpo.rewards.chembl_classifier_reward import ChemblClassifierReward  # noqa: E402
from ddpo.runtime.recording.run_context import RunContext  # noqa: E402
from ddpo.runtime.recording.writers.metrics_writer import MetricsWriter  # noqa: E402
from ddpo.runtime.train_helpers import (  # noqa: E402
    PreparedSamples,
    compute_batch_stats,
    build_epoch_facts,
)


def _identity_sum(value: float, _dist_ctx: object) -> float:
    return float(value)


def _identity_min(value: float, _dist_ctx: object) -> float:
    return float(value)


def _identity_max(value: float, _dist_ctx: object) -> float:
    return float(value)


def test_chembl_classifier_reward_reaches_epoch_summary(tmp_path: Path) -> None:
    """FieldSpec -> aggregate -> EpochFacts.extra_metrics -> CSV column."""
    reward = ChemblClassifierReward(
        model_path=str(tmp_path / "missing_classifier.pt"),
        device="cpu",
    )
    specs = reward.field_specs()
    assert any(spec.name == "chembl_classifier_reward" for spec in specs)
    assert any(spec.name == "avg_chembl_classifier_reward" for spec in specs)

    data = SimpleNamespace(
        batch={
            "x": torch.zeros(4, 2, 3),
            "nodesxsample": torch.full((4,), 2, dtype=torch.long),
            "chembl_classifier_reward": torch.tensor([1.0, -1.0, 1.0, 1.0]),
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
    prepared = PreparedSamples(
        samples=data,
        filtering_ratio=0.0,
        novelty_penalty_ratio=0.0,
    )
    stats = compute_batch_stats(
        prepared,
        config={"reward": {"weights": {"chembl_classifier": 0.5}}},
        field_specs=specs,
    )
    reduced = reduce_registered_epoch_aggregates(
        stats.registered_epoch_aggregates or {},
        None,
        reduce_sum_scalar=_identity_sum,
        reduce_min_scalar=_identity_min,
        reduce_max_scalar=_identity_max,
    )
    assert reduced["avg_chembl_classifier_reward"] == 0.5

    stats.extra_epoch_metrics = reduced
    runtime = SimpleNamespace(
        config={"reward": {"weights": {"chembl_classifier": 0.5}}},
        max_epochs=1,
        checkpoint_interval=10,
    )
    facts = build_epoch_facts(
        runtime=runtime,
        epoch=1,
        metrics={"PPO_Loss": 0.25},
        batch_stats=stats,
        prepared=prepared,
    )
    assert facts.extra_metrics["avg_chembl_classifier_reward"] == 0.5

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

    with open(run_context.training_summary_csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["avg_chembl_classifier_reward"] == "0.500000"
