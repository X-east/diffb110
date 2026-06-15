"""Tests for D-MPNN ChEMBL dataset construction."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from chembl_classifier.dmpnn.data import build_dataset_csv


def test_build_dmpnn_dataset_csv_with_scaffold_splits(tmp_path: Path) -> None:
    """Tiny positive/negative tables become canonical scaffold-split CSV rows."""
    positives = tmp_path / "positives.csv"
    negatives = tmp_path / "negatives.csv"
    output = tmp_path / "dataset.csv"
    positives.write_text(
        "smiles,sample_source\n"
        "CCO,chembl\n"
        "CCN,chembl\n"
        "c1ccccc1,chembl\n",
        encoding="utf-8",
    )
    negatives.write_text(
        "smiles,sample_weight,negative_label\n"
        "S1SSSSS1,2.5,sulfur\n"
        "CCCCCCCC,1.0,alkane\n"
        "c1ccc2ccccc2c1,3.0,pah\n",
        encoding="utf-8",
    )

    result = build_dataset_csv(
        positive_path=positives,
        negative_path=negatives,
        output_csv=output,
        train_fraction=0.5,
        val_fraction=0.25,
        seed=7,
        min_negative_weight=2.0,
    )

    rows = pd.read_csv(output)
    assert result.n_total == len(rows) == 6
    assert {"smiles", "label", "sample_weight", "source", "negative_reason", "scaffold", "split"} <= set(rows.columns)
    assert set(rows["label"]) == {0, 1}
    assert set(rows["split"]) <= {"train", "val", "test"}
    assert rows.loc[rows["label"].eq(0), "sample_weight"].min() >= 2.0
