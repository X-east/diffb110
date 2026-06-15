"""Dataset preparation for the ChEMBL D-MPNN classifier."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DMPNNDatasetBuildResult:
    """Paths and counts from a D-MPNN dataset build."""

    dataset_csv: Path
    n_total: int
    n_train: int
    n_val: int
    n_test: int


def build_dataset_from_config(config_path: str | Path) -> DMPNNDatasetBuildResult:
    """Build a scaffold-split CSV dataset from a YAML config."""
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    project_root = config_path.resolve().parents[1]
    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})
    output_cfg = config.get("output", {})

    positive_path = _resolve_path(project_root, data_cfg.get("explicit_positive_samples"))
    negative_path = _resolve_path(project_root, data_cfg.get("explicit_negative_samples"))
    dataset_csv = _resolve_path(
        project_root,
        output_cfg.get("dataset_csv", "chembl_classifier/models/dmpnn_dataset.csv"),
    )
    if positive_path is None or negative_path is None:
        raise ValueError("D-MPNN data config requires explicit positive and negative sample paths")

    return build_dataset_csv(
        positive_path=positive_path,
        negative_path=negative_path,
        output_csv=dataset_csv,
        train_fraction=float(training_cfg.get("train_split", 0.8)),
        val_fraction=float(training_cfg.get("val_split", 0.1)),
        seed=int(training_cfg.get("seed", 45)),
        min_negative_weight=float(data_cfg.get("min_negative_weight", 1.0)),
    )


def build_dataset_csv(
    *,
    positive_path: Path,
    negative_path: Path,
    output_csv: Path,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    seed: int = 45,
    min_negative_weight: float = 1.0,
) -> DMPNNDatasetBuildResult:
    """Build canonical SMILES rows and scaffold split labels into a CSV file."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("train_fraction + val_fraction must be < 1")

    positives = _load_labeled_smiles(positive_path, label=1, source="positive", min_weight=1.0)
    negatives = _load_labeled_smiles(negative_path, label=0, source="negative", min_weight=min_negative_weight)
    rows = pd.concat([positives, negatives], ignore_index=True)
    rows = rows.drop_duplicates(subset=["smiles", "label"], keep="first").reset_index(drop=True)
    rows["scaffold"] = rows["smiles"].map(scaffold_smiles)
    rows["split"] = assign_scaffold_splits(
        rows,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        seed=seed,
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output_csv, index=False)
    counts = rows["split"].value_counts().to_dict()
    logger.info("Wrote D-MPNN dataset CSV to %s (%d rows)", output_csv, len(rows))
    return DMPNNDatasetBuildResult(
        dataset_csv=output_csv,
        n_total=int(len(rows)),
        n_train=int(counts.get("train", 0)),
        n_val=int(counts.get("val", 0)),
        n_test=int(counts.get("test", 0)),
    )


def assign_scaffold_splits(
    rows: pd.DataFrame,
    *,
    train_fraction: float,
    val_fraction: float,
    seed: int,
) -> pd.Series:
    """Assign train/val/test labels by whole scaffolds."""
    shuffled = rows.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    scaffold_sizes = (
        shuffled.groupby("scaffold", dropna=False)
        .size()
        .sort_values(ascending=False)
        .reset_index(name="size")
    )

    n_total = len(rows)
    train_limit = train_fraction * n_total
    val_limit = (train_fraction + val_fraction) * n_total
    scaffold_to_split: dict[str, str] = {}
    n_seen = 0
    for _, item in scaffold_sizes.iterrows():
        scaffold = str(item["scaffold"])
        size = int(item["size"])
        if n_seen < train_limit:
            split = "train"
        elif n_seen < val_limit:
            split = "val"
        else:
            split = "test"
        scaffold_to_split[scaffold] = split
        n_seen += size
    return rows["scaffold"].astype(str).map(scaffold_to_split).fillna("test")


def scaffold_smiles(smiles: str) -> str:
    """Return Bemis-Murcko scaffold SMILES, falling back to canonical SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:
        scaffold = ""
    return scaffold or Chem.MolToSmiles(mol, canonical=True)


def _load_labeled_smiles(path: Path, *, label: int, source: str, min_weight: float) -> pd.DataFrame:
    df = _read_smiles_table(path)
    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        smiles = canonicalize_smiles(str(row["smiles"]))
        if not smiles:
            continue
        sample_weight = _first_numeric(row, ["sample_weight", "weight"], default=1.0)
        if label == 0:
            sample_weight = max(float(sample_weight), min_weight)
        records.append(
            {
                "smiles": smiles,
                "label": int(label),
                "sample_weight": float(sample_weight),
                "source": str(row.get("sample_source", row.get("source", source))),
                "negative_reason": str(row.get("negative_label", row.get("all_labels", row.get("reason", "")))),
            }
        )
    return pd.DataFrame.from_records(records)


def canonicalize_smiles(smiles: str) -> str:
    """Return canonical RDKit SMILES or an empty string for invalid input."""
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return ""
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return ""


def _read_smiles_table(path: Path) -> pd.DataFrame:
    """Read CSV/TSV/SMI input with a required or inferable smiles column."""
    if not path.exists():
        raise FileNotFoundError(f"SMILES sample file does not exist: {path}")
    df = pd.read_csv(path, sep=None, engine="python")
    if "smiles" not in df.columns:
        first_col = str(df.columns[0])
        df = df.rename(columns={first_col: "smiles"})
    df = df[df["smiles"].notna()].copy()
    df["smiles"] = df["smiles"].astype(str).str.strip()
    df = df[df["smiles"].ne("")]
    return df.reset_index(drop=True)


def _first_numeric(row: pd.Series, keys: list[str], default: float) -> float:
    for key in keys:
        if key not in row or pd.isna(row[key]):
            continue
        try:
            return float(row[key])
        except (TypeError, ValueError):
            continue
    return float(default)


def _resolve_path(project_root: Path, value: Any) -> Path | None:
    if value is None:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else project_root / path


def main() -> None:
    """Command-line entry point for dataset CSV construction."""
    parser = argparse.ArgumentParser(description="Build ChEMBL D-MPNN dataset CSV.")
    parser.add_argument("--config", required=True, help="Path to D-MPNN YAML config.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    result = build_dataset_from_config(args.config)
    print(result)


if __name__ == "__main__":
    main()
