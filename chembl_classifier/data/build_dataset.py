"""
Build training dataset for ChEMBL binary classifier.

Produces a featurized .pt file with (features, labels, smiles) for:
  - Positive (label=1): 100k random ChEMBL molecules.
  - Negative (label=0): synthesized PAHs, garbage molecules, ChEMBL outliers.

Featurization: Morgan count FP (2048) + RDKit 2D descriptors (172) = 2220 dim.
Supports multi-process parallel featurization for large datasets.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import sys
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import yaml
from rdkit import Chem
from rdkit import RDLogger

# ── Project root ────────────────────────────────────────────────────────────
# build_dataset.py → data/ → chembl_classifier/ → project root (3 levels up)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ── Import MolSkill featurizer ──────────────────────────────────────────────
_MOLSKILL_ROOT = _PROJECT_ROOT / "reference project" / "molskill"
if str(_MOLSKILL_ROOT) not in sys.path:
    sys.path.insert(0, str(_MOLSKILL_ROOT))

from molskill.data.featurizers import get_featurizer, Featurizer  # noqa: E402

# ── Config ──────────────────────────────────────────────────────────────────
_CONFIG_PATH = _PROJECT_ROOT / "chembl_classifier" / "config.yaml"

logger = logging.getLogger(__name__)

# Suppress RDKit warnings during bulk processing
RDLogger.logger().setLevel(RDLogger.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# Synthetic Negative SMILES
# ============================================================

# Polycyclic Aromatic Hydrocarbons (PAHs) — all-carbon, the model's collapse mode
_SYNTHETIC_PAH_SMILES: list[str] = [
    # Linear acenes
    "c1ccc2ccccc2c1",                    # Naphthalene C10H8
    "c1ccc2cc3ccccc3cc2c1",              # Anthracene C14H10
    "c1ccc2cc3cc4ccccc4cc3cc2c1",        # Tetracene C18H12
    "c1ccc2cc3cc4cc5ccccc5cc4cc3cc2c1",  # Pentacene C22H14
    # Angular PAHs
    "c1ccc2c(c1)ccc3ccccc32",            # Phenanthrene C14H10
    "c1ccc2c(c1)cc3ccc4ccccc4c3c2",     # Chrysene C18H12
    "c1ccc2c(c1)cc3c4ccccc4ccc3c2",     # Triphenylene C18H12
    # Peri-condensed
    "c1cc2ccc3cccc4ccc1c2c34",           # Pyrene C16H10
    "c1cc2c3cccc4ccc(c1)c2c34",          # Fluoranthene C16H10
    "c1cc2cccc3c4cccc5ccc(c1)c2c3c45",   # Perylene C20H12
    "c1cc2ccc3ccc4ccc5ccc6ccc1c7c2c3c4c5c67",  # Coronene C24H12
    "c1cc2c3c4c5c6c7c8c9c%10c(c1)c%11c2c3c4c5c6c7c8c9c%10%11",  # Ovalene C32H14
    # Multi-phenyl (model's tendency)
    "c1ccc(cc1)c2ccccc2",                # Biphenyl C12H10
    "c1ccc(cc1)c2ccc(cc2)c3ccccc3",      # p-Terphenyl C18H14
    "c1ccc(cc1)c2ccc(cc2)c3ccc(cc3)c4ccccc4",  # p-Quarterphenyl C24H18
    # Extra fused variants
    "c1ccc2c(c1)c3ccccc3c4ccccc24",      # Benzo[a]anthracene
    "c1cc2cc3cc4cc5cc6cc7cc8ccccc8c7c6c5c4c3c2cc1",  # Large acene
    # Cata-condensed with 3-6 rings
    "c1ccc2c(c1)cc3c(c2)cccc3",           # 1H-Phenalene
    "c1ccc2c(c1)cc3c4c(c2)cccc4ccc3",     # Benzo[c]phenanthrene
]

# Garbage molecules: all-sulfur, all-silicon, all-carbon linear, etc.
_SYNTHETIC_GARBAGE_SMILES: list[str] = [
    # All-sulfur
    "S1SSSSS1",                           # S6 cyclohexasulfur
    "S1SSSSSS1",                          # S7
    "S1SSSSSSS1",                         # S8 octasulfur
    "S1SSSSSSSS1",                        # S9
    "S1SSSSSSSSS1",                       # S10
    "S1SSSSSSSSSS1",                      # S12
    "SSSSSSSS",                           # Linear S8
    "SSSSSSSSSSSS",                       # Linear S12
    "S1SSS1",                             # S4 (small ring)
    # All-silicon
    "[SiH2]1[SiH2][SiH2][SiH2][SiH2]1",  # Si5 ring
    "[SiH2]1[SiH2][SiH2][SiH2][SiH2][SiH2]1",  # Si6 ring
    # All-carbon linear (no rings, minimal functionality)
    "CCCCCCCCCCCCCCCCCCCC",               # C20 linear
    "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCC",     # C30 linear
    "CCCCCCCCCC",                         # C10 linear
    # Linear alkenes (no function)
    "C=CC=CC=CC=CC=CC=C",                # Polyene
    "C=CC=CC=CC=CC=CC=CC=CC=C",          # Long polyene
    # Simple non-drug-like
    "C1CCCCC1",                           # Cyclohexane (no heteroatoms)
    "C1CCCCCCCCCCCC1",                    # Large cycloalkane
    "CC(C)(C)C(C)(C)C(C)(C)C",            # Branched alkane
]


def _build_synthetic_negatives(
    n_pah: int = 2000,
    n_garbage: int = 1000,
    seed: int = 42,
) -> list[str]:
    """Generate synthetic negative SMILES via enumeration and perturbation.

    Args:
        n_pah: Target number of PAH negative samples.
        n_garbage: Target number of garbage negative samples.
        seed: Random seed for reproducibility.

    Returns:
        List of SMILES strings.
    """
    rng = np.random.RandomState(seed)
    negatives: list[str] = []

    # ── PAHs: listed SMILES + enumerated C-C coupling variants ────────────
    enumerated_pahs: list[str] = list(_SYNTHETIC_PAH_SMILES)

    # Generate more PAHs by coupling benzene rings
    benzene_units = ["c1ccccc1", "c1ccc2ccccc2c1", "c1ccc2cc3ccccc3cc2c1"]
    for _ in range(n_pah):
        unit1 = rng.choice(benzene_units)
        unit2 = rng.choice(benzene_units)
        # Simple biphenyl-type coupling
        smi = f"{unit1[:-1]}{unit2[1:]}"
        enumerated_pahs.append(smi)

    negatives.extend(enumerated_pahs[:n_pah])

    # ── Garbage ───────────────────────────────────────────────────────────
    garbage: list[str] = list(_SYNTHETIC_GARBAGE_SMILES)

    # Generate more via mutations of existing garbage
    for base in _SYNTHETIC_GARBAGE_SMILES[:5]:
        for extra in range(2, 5):
            # Extend sulfur chains
            garbage.append("S" * (rng.randint(4, 16)))

    negatives.extend(garbage[:n_garbage])

    return negatives


def _build_chembl_outlier_negatives(
    properties_path: str,
    n: int = 1000,
    seed: int = 42,
) -> list[str]:
    """Sample extreme outliers from ChEMBL property distribution as negatives.

    Outliers defined as: LogP > 7 OR TPSA < 15 OR MolWt > 450 OR NumHeteroatoms = 0.

    Args:
        properties_path: Path to chembl_36_properties.csv.gz.
        n: Number of outliers to sample.
        seed: Random seed.

    Returns:
        List of SMILES strings of extreme outlier molecules.
    """
    rng = np.random.RandomState(seed)

    logger.info("Loading ChEMBL properties for outlier detection...")
    df = pd.read_csv(properties_path, compression="gzip")
    logger.info(f"  Loaded {len(df):,} molecules with properties")

    # Define outlier conditions
    df["is_outlier"] = (
        (df.get("mol_log_p", 0) > 7)
        | (df.get("tpsa", 100) < 15)
        | (df.get("mol_wt", 300) > 450)
        | (df.get("num_heteroatoms", 5) == 0)
    )

    outliers = df[df["is_outlier"]]
    logger.info(f"  Found {len(outliers):,} extreme outliers")

    if len(outliers) < n:
        logger.warning(
            f"Only {len(outliers)} outliers available, using all of them "
            f"(requested {n})"
        )
        n = len(outliers)

    sampled = outliers.sample(n=n, random_state=seed)
    return sampled["canonical_smiles"].tolist()


def featurize_smiles(
    smiles_list: Sequence[str],
    featurizer: Featurizer,
    label: int,
    skip_errors: bool = True,
    n_jobs: int = 1,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Featurize a list of SMILES into 2220-dim vectors.

    Args:
        smiles_list: SMILES strings.
        featurizer: MolSkill featurizer instance (morgan_count_rdkit_2d).
        label: Binary label (1=positive, 0=negative).
        skip_errors: If True, skip molecules that fail featurization.
        n_jobs: Number of parallel workers. Use 1 to skip multiprocessing.

    Returns:
        (features [N, 2220], labels [N], valid_smiles [N]).
    """
    if n_jobs <= 1 or len(smiles_list) < n_jobs * 100:
        return _featurize_sequential(smiles_list, featurizer, label, skip_errors)

    return _featurize_parallel(smiles_list, label, skip_errors, n_jobs, verbose)


def _featurize_sequential(
    smiles_list: Sequence[str],
    featurizer: Featurizer,
    label: int,
    skip_errors: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Single-process featurization."""
    features: list[np.ndarray] = []
    valid_smiles: list[str] = []
    n_skipped = 0

    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                n_skipped += 1
                continue
            feat = featurizer.get_feat(mol)
            if feat is None or len(feat) != featurizer.dim():
                n_skipped += 1
                continue
            features.append(feat)
            valid_smiles.append(smi)
        except Exception:
            n_skipped += 1
            if not skip_errors:
                raise

    if n_skipped > 0:
        logger.warning(f"  Skipped {n_skipped}/{len(smiles_list)} molecules")

    X = np.stack(features, axis=0).astype(np.float32)
    y = np.full(len(features), label, dtype=np.float32)
    return X, y, valid_smiles


def _featurize_chunk(args: tuple[list[str], int]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Worker function: featurize a chunk of SMILES in a subprocess."""
    smiles_chunk, label = args
    # Each worker needs its own featurizer (RDKit objects can't be pickled)
    featurizer = get_featurizer("morgan_count_rdkit_2d")
    return _featurize_sequential(smiles_chunk, featurizer, label, skip_errors=True)


def _featurize_parallel(
    smiles_list: Sequence[str],
    label: int,
    skip_errors: bool = True,
    n_jobs: int = 24,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Multi-process featurization using Pool."""
    n_total = len(smiles_list)
    chunk_size = max(100, n_total // n_jobs)
    chunks: list[list[str]] = []
    for i in range(0, n_total, chunk_size):
        chunks.append(list(smiles_list[i : i + chunk_size]))

    if verbose:
        logger.info(
            f"  Parallel featurization: {n_total:,} molecules → "
            f"{len(chunks)} chunks × {n_jobs} workers"
        )

    args_list = [(chunk, label) for chunk in chunks]

    all_X: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    all_smiles: list[str] = []
    n_processed = 0

    with mp.Pool(processes=n_jobs) as pool:
        for X_chunk, y_chunk, s_chunk in pool.imap_unordered(_featurize_chunk, args_list):
            if len(X_chunk) > 0:
                all_X.append(X_chunk)
                all_y.append(y_chunk)
                all_smiles.extend(s_chunk)
            n_processed += len(s_chunk)

    if verbose:
        n_lost = n_total - n_processed
        logger.info(f"  Done: {n_processed:,} valid / {n_total:,} total ({n_lost} skipped)")

    if not all_X:
        return np.empty((0, 2220), dtype=np.float32), np.empty((0,), dtype=np.float32), []

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    return X, y, all_smiles


def build_dataset(config_path: str | None = None) -> dict:
    """Build the full featurized dataset.

    Args:
        config_path: Path to config YAML. Defaults to chembl_classifier/config.yaml.

    Returns:
        Dict with keys: X_train, y_train, X_val, y_val, X_test, y_test,
        smiles_train, smiles_val, smiles_test, all as numpy arrays.
    """
    if config_path is None:
        config_path = str(_CONFIG_PATH)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    seed = train_cfg["seed"]
    n_jobs = int(data_cfg.get("n_jobs", 1))
    rng = np.random.RandomState(seed)

    # ── Featurizer ────────────────────────────────────────────────────────
    logger.info("Initializing featurizer: morgan_count_rdkit_2d")
    featurizer: Featurizer = get_featurizer("morgan_count_rdkit_2d")
    logger.info(f"  Feature dimension: {featurizer.dim()}")

    # ── 1. Positive samples: ChEMBL random 100k ───────────────────────────
    logger.info("Loading ChEMBL molecules for positive samples...")
    chembl_path = _PROJECT_ROOT / data_cfg["chembl_molecules"]
    df_chembl = pd.read_csv(chembl_path, compression="gzip")
    logger.info(f"  Total ChEMBL molecules: {len(df_chembl):,}")

    n_pos = min(train_cfg["n_positives"], len(df_chembl))
    df_pos = df_chembl.sample(n=n_pos, random_state=seed)
    pos_smiles = df_pos["canonical_smiles"].tolist()
    logger.info(f"  Sampled {len(pos_smiles):,} positive molecules")

    logger.info(f"Featurizing {len(pos_smiles):,} positives ({n_jobs} workers)...")
    X_pos, y_pos, smiles_pos = featurize_smiles(
        pos_smiles, featurizer, label=1, n_jobs=n_jobs
    )
    logger.info(f"  Positives: {X_pos.shape[0]:,} valid molecules")

    # ── 2. Negative samples ───────────────────────────────────────────────
    all_X_neg: list[np.ndarray] = []
    all_y_neg: list[np.ndarray] = []
    all_smiles_neg: list[str] = []

    # 2a. Synthetic PAHs
    n_pah = data_cfg["n_synthetic_pah"]
    logger.info(f"Building {n_pah} synthetic PAH negatives...")
    pah_smiles = _build_synthetic_negatives(
        n_pah=n_pah, n_garbage=0, seed=seed
    )
    X_pah, y_pah, s_pah = featurize_smiles(pah_smiles, featurizer, label=0)
    logger.info(f"  PAHs: {X_pah.shape[0]} valid")
    all_X_neg.append(X_pah)
    all_y_neg.append(y_pah)
    all_smiles_neg.extend(s_pah)

    # 2b. Synthetic garbage
    n_garbage = data_cfg["n_synthetic_garbage"]
    logger.info(f"Building {n_garbage} synthetic garbage negatives...")
    garbage_smiles = _build_synthetic_negatives(
        n_pah=0, n_garbage=n_garbage, seed=seed + 1
    )
    X_garb, y_garb, s_garb = featurize_smiles(garbage_smiles, featurizer, label=0)
    logger.info(f"  Garbage: {X_garb.shape[0]} valid")
    all_X_neg.append(X_garb)
    all_y_neg.append(y_garb)
    all_smiles_neg.extend(s_garb)

    # 2c. ChEMBL extreme outliers
    n_outliers = data_cfg["n_chembl_outliers"]
    logger.info(f"Building {n_outliers} ChEMBL outlier negatives...")
    outlier_smiles = _build_chembl_outlier_negatives(
        str(_PROJECT_ROOT / data_cfg["chembl_properties"]),
        n=n_outliers,
        seed=seed,
    )
    X_out, y_out, s_out = featurize_smiles(outlier_smiles, featurizer, label=0)
    logger.info(f"  Outliers: {X_out.shape[0]} valid")
    all_X_neg.append(X_out)
    all_y_neg.append(y_out)
    all_smiles_neg.extend(s_out)

    # ── Combine negatives ─────────────────────────────────────────────────
    X_neg = np.concatenate(all_X_neg, axis=0)
    y_neg = np.concatenate(all_y_neg, axis=0)
    logger.info(f"Total negatives: {X_neg.shape[0]:,}")

    # ── 3. Combine and split ──────────────────────────────────────────────
    X = np.concatenate([X_pos, X_neg], axis=0)
    y = np.concatenate([y_pos, y_neg], axis=0)
    all_smiles = smiles_pos + all_smiles_neg
    logger.info(f"Combined dataset: {X.shape[0]:,} samples, {X.shape[1]} features")

    # Shuffle
    indices = rng.permutation(len(X))
    X = X[indices]
    y = y[indices]
    all_smiles = [all_smiles[i] for i in indices]

    # Train / val / test split
    n_total = len(X)
    n_train = int(n_total * train_cfg["train_split"])
    n_val = int(n_total * train_cfg["val_split"])

    dataset = {
        "X_train": X[:n_train],
        "y_train": y[:n_train],
        "smiles_train": all_smiles[:n_train],
        "X_val": X[n_train : n_train + n_val],
        "y_val": y[n_train : n_train + n_val],
        "smiles_val": all_smiles[n_train : n_train + n_val],
        "X_test": X[n_train + n_val :],
        "y_test": y[n_train + n_val :],
        "smiles_test": all_smiles[n_train + n_val :],
    }

    logger.info(
        f"Split: train={len(dataset['X_train']):,}, "
        f"val={len(dataset['X_val']):,}, "
        f"test={len(dataset['X_test']):,}"
    )
    logger.info(
        f"Class balance — train: pos={int(dataset['y_train'].sum())}, "
        f"neg={n_train - int(dataset['y_train'].sum())}"
    )

    return dataset


def main() -> None:
    """Build and save the dataset."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Load config
    with open(_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    cache_path = _PROJECT_ROOT / cfg["output"]["dataset_cache"]
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Building ChEMBL classifier dataset")
    logger.info("=" * 60)

    dataset = build_dataset(str(_CONFIG_PATH))

    logger.info(f"Saving dataset to {cache_path}")
    torch.save(dataset, cache_path)

    # Print summary
    for split in ["train", "val", "test"]:
        X = dataset[f"X_{split}"]
        y = dataset[f"y_{split}"]
        logger.info(
            f"  {split}: {len(X):,} samples, "
            f"pos={int(y.sum())}, neg={len(y) - int(y.sum())}"
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
