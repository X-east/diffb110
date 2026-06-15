"""Training-loop smoke test for the ChEMBL D-MPNN classifier."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from chembl_classifier.dmpnn.train import train


def test_dmpnn_train_writes_checkpoint_and_summary(tmp_path: Path) -> None:
    """A tiny scaffold-split dataset can run one D-MPNN training epoch."""
    project_root = tmp_path
    config_dir = project_root / "chembl_classifier"
    config_dir.mkdir(parents=True)
    dataset_csv = project_root / "dataset.csv"
    rows = pd.DataFrame(
        [
            {"smiles": "CCO", "label": 1, "sample_weight": 1.0, "split": "train"},
            {"smiles": "CCN", "label": 1, "sample_weight": 1.0, "split": "train"},
            {"smiles": "CCCCCCCC", "label": 0, "sample_weight": 1.0, "split": "train"},
            {"smiles": "S1SSSSS1", "label": 0, "sample_weight": 1.0, "split": "train"},
            {"smiles": "c1ccccc1", "label": 1, "sample_weight": 1.0, "split": "val"},
            {"smiles": "c1ccc2ccccc2c1", "label": 0, "sample_weight": 1.0, "split": "val"},
            {"smiles": "CCCl", "label": 1, "sample_weight": 1.0, "split": "test"},
            {"smiles": "CCCCCCCCCC", "label": 0, "sample_weight": 1.0, "split": "test"},
        ]
    )
    rows.to_csv(dataset_csv, index=False)
    config = {
        "model": {
            "hidden_size": 8,
            "depth": 1,
            "ffn_hidden_size": 8,
            "ffn_num_layers": 2,
            "dropout": 0.0,
        },
        "training": {
            "batch_size": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "max_epochs": 1,
            "early_stopping_patience": 2,
            "seed": 1,
            "auto_pos_weight": True,
            "device": "cpu",
        },
        "output": {
            "dataset_csv": "dataset.csv",
            "output_dir": "out",
            "model_path": "out/model.pt",
        },
    }
    config_path = config_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    model = train(config_path)

    assert model.n_params > 0
    assert (project_root / "out" / "model.pt").exists()
    assert (project_root / "out" / "metrics.csv").exists()
    assert (project_root / "out" / "summary.json").exists()
