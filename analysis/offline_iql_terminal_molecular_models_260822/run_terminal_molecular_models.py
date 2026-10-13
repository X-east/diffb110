"""Compare endpoint molecular models on the frozen rank-40 binary labels.

This is an exploratory terminal-molecule separability experiment.  It reuses
the project ChEMBL MLP and D-MPNN implementations, preserves the existing
scaffold-held-out split, and never exposes the two label-generating fields to
candidate learned models.  The label-generating score is retained only as an
explicit non-deployable oracle ceiling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import multiprocessing
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from numpy.lib.format import open_memmap
from rdkit import Chem
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
    roc_curve,
)


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUTPUT_DIR = ROOT / "analysis/offline_iql_terminal_molecular_models_260822"
DEFAULT_MANIFEST = (
    ROOT
    / "analysis/offline_iql_binary_rank40_diagnosis_260821"
    / "binary_rank40_manifest.csv.gz"
)
FEATURE_PATH = OUTPUT_DIR / "morgan_count_rdkit_2d_features.npy"
FEATURE_VALID_PATH = OUTPUT_DIR / "morgan_count_rdkit_2d_valid.npy"
FEATURE_META_PATH = OUTPUT_DIR / "morgan_count_rdkit_2d_metadata.json"
PREDICTIONS_PATH = OUTPUT_DIR / "terminal_model_predictions.csv.gz"
METRICS_PATH = OUTPUT_DIR / "terminal_model_metrics.csv"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
LOGGER = logging.getLogger("terminal_molecular_models")
LABEL_COLUMN = "binary_label"
POSITIVE_LABEL = "preferred"
FEATURE_DIM = 2220
MORGAN_DIM = 2048
DESCRIPTOR_DIM = FEATURE_DIM - MORGAN_DIM

_WORKER_FEATURIZER: Any | None = None


@dataclass(frozen=True)
class MLPTrainingConfig:
    """Configuration for the supervised endpoint MLP comparison."""

    batch_size: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 30
    early_stopping_patience: int = 5


def parse_args() -> argparse.Namespace:
    """Parse experiment arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=18)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dmpnn-batch-size", type=int, default=512)
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--skip-pretrained-dmpnn", action="store_true")
    parser.add_argument("--rebuild-features", action="store_true")
    return parser.parse_args()


def configure_logging(output_dir: Path) -> None:
    """Configure durable file and rank-zero terminal logging."""
    output_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    file_handler = logging.FileHandler(output_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, stream_handler],
        force=True,
    )


def sha256_file(path: Path) -> str:
    """Return a SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    """Convert NumPy values and non-finite floats to strict JSON values."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def load_manifest(path: Path) -> pd.DataFrame:
    """Load and validate the frozen binary scaffold manifest."""
    frame = pd.read_csv(path, compression="gzip")
    required = {
        "trajectory_id",
        "canonical_smiles",
        "split",
        LABEL_COLUMN,
        "umap_nearest_chembl_distance",
        "structural_hard_bad",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"manifest is missing columns: {missing}")
    if frame["trajectory_id"].duplicated().any():
        raise ValueError("trajectory_id must be unique")
    if frame["canonical_smiles"].isna().any():
        raise ValueError("all binary diagnostic rows must have canonical SMILES")
    if set(frame["split"].unique()) != {"train", "validation", "test"}:
        raise ValueError("manifest must contain train, validation, and test")
    if set(frame[LABEL_COLUMN].unique()) != {"preferred", "non_preferred"}:
        raise ValueError("unexpected binary labels")
    frame = frame.copy()
    frame["target"] = frame[LABEL_COLUMN].eq(POSITIVE_LABEL).astype(np.int8)
    frame["structural_hard_bad"] = frame["structural_hard_bad"].map(
        lambda value: str(value).strip().lower() in {"1", "true", "yes"}
        if not isinstance(value, (bool, np.bool_))
        else bool(value)
    )
    return frame


def _init_featurizer_worker() -> None:
    """Initialize one MolSkill featurizer per worker process."""
    global _WORKER_FEATURIZER
    molskill_root = ROOT / "reference project/代码/molskill"
    if str(molskill_root) not in sys.path:
        sys.path.insert(0, str(molskill_root))
    from molskill.data.featurizers import get_featurizer

    _WORKER_FEATURIZER = get_featurizer("morgan_count_rdkit_2d")


def _featurize_range(
    payload: tuple[int, list[str], str, str],
) -> tuple[int, int, int]:
    """Featurize one contiguous range directly into shared NumPy memmaps."""
    start, smiles_values, feature_path, valid_path = payload
    if _WORKER_FEATURIZER is None:
        _init_featurizer_worker()
    feature_array = open_memmap(feature_path, mode="r+")
    valid_array = open_memmap(valid_path, mode="r+")
    success = 0
    for offset, smiles in enumerate(smiles_values):
        index = start + offset
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            feature_array[index] = 0.0
            valid_array[index] = False
            continue
        try:
            feature = np.asarray(_WORKER_FEATURIZER.get_feat(mol), dtype=np.float32)
        except Exception:
            feature_array[index] = 0.0
            valid_array[index] = False
            continue
        if feature.shape != (FEATURE_DIM,):
            feature_array[index] = 0.0
            valid_array[index] = False
            continue
        feature_array[index] = np.nan_to_num(
            feature,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        valid_array[index] = True
        success += 1
    feature_array.flush()
    valid_array.flush()
    return start, len(smiles_values), success


def build_feature_cache(
    frame: pd.DataFrame,
    *,
    manifest_path: Path,
    output_dir: Path,
    workers: int,
    rebuild: bool,
) -> tuple[np.memmap, np.memmap]:
    """Build or load the 2220-dimensional endpoint feature cache."""
    feature_path = output_dir / FEATURE_PATH.name
    valid_path = output_dir / FEATURE_VALID_PATH.name
    metadata_path = output_dir / FEATURE_META_PATH.name
    manifest_sha = sha256_file(manifest_path)
    if not rebuild and feature_path.exists() and valid_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("manifest_sha256") == manifest_sha
            and int(metadata.get("rows", -1)) == len(frame)
            and int(metadata.get("feature_dim", -1)) == FEATURE_DIM
        ):
            LOGGER.info("Reusing feature cache: %s", feature_path)
            return open_memmap(feature_path, mode="r"), open_memmap(valid_path, mode="r")

    LOGGER.info(
        "Building morgan_count_rdkit_2d features rows=%d workers=%d",
        len(frame),
        workers,
    )
    feature_array = open_memmap(
        feature_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(frame), FEATURE_DIM),
    )
    valid_array = open_memmap(
        valid_path,
        mode="w+",
        dtype=np.bool_,
        shape=(len(frame),),
    )
    feature_array[:] = 0.0
    valid_array[:] = False
    feature_array.flush()
    valid_array.flush()
    del feature_array, valid_array

    smiles = frame["canonical_smiles"].astype(str).tolist()
    worker_count = max(1, min(int(workers), len(smiles)))
    chunk_size = math.ceil(len(smiles) / worker_count)
    payloads = []
    for start in range(0, len(smiles), chunk_size):
        payloads.append(
            (
                start,
                smiles[start : start + chunk_size],
                str(feature_path),
                str(valid_path),
            )
        )
    completed = 0
    success = 0
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=context,
        initializer=_init_featurizer_worker,
    ) as executor:
        futures = [executor.submit(_featurize_range, payload) for payload in payloads]
        for future in as_completed(futures):
            _, rows, successes = future.result()
            completed += rows
            success += successes
            LOGGER.info(
                "Feature progress rows=%d/%d success=%d",
                completed,
                len(frame),
                success,
            )
    metadata = {
        "feature_name": "morgan_count_rdkit_2d",
        "feature_dim": FEATURE_DIM,
        "morgan_dim": MORGAN_DIM,
        "descriptor_dim": DESCRIPTOR_DIM,
        "rows": len(frame),
        "valid_rows": success,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "workers": worker_count,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return open_memmap(feature_path, mode="r"), open_memmap(valid_path, mode="r")


def resolve_device(requested: str) -> torch.device:
    """Resolve and validate the requested compute device."""
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        probe = torch.ones(8, device=device)
        if float(probe.sum().item()) != 8.0:
            raise RuntimeError("CUDA tensor preflight produced an unexpected result")
    return device


@torch.no_grad()
def score_pretrained_mlp(features: np.ndarray, device: torch.device) -> np.ndarray:
    """Score all molecules with the existing ChEMBL MLP checkpoint."""
    from chembl_classifier.model import ChemblClassifier

    checkpoint = (
        ROOT
        / "chembl_classifier/models/chembl_classifier_selected_v0_3_1_scaffold_strongpoly.pt"
    )
    model = ChemblClassifier.from_checkpoint(str(checkpoint), device=str(device))
    output = np.empty(len(features), dtype=np.float32)
    for start in range(0, len(features), 4096):
        stop = min(start + 4096, len(features))
        batch = torch.from_numpy(
            np.array(features[start:stop], dtype=np.float32, copy=True)
        ).to(device)
        output[start:stop] = model.predict_proba(batch).squeeze(1).cpu().numpy()
    return output


@torch.no_grad()
def score_pretrained_dmpnn(
    smiles: Sequence[str],
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Score all molecules with the existing ChEMBL D-MPNN checkpoint."""
    from chembl_classifier.dmpnn.predictor import DMPNNClassifierPredictor

    checkpoint = (
        ROOT
        / "chembl_classifier/models/chembl_dmpnn_v0_1_scaffold_strongpoly.pt"
    )
    predictor = DMPNNClassifierPredictor.from_checkpoint(
        checkpoint,
        device=device,
        cache_smiles=True,
    )
    output = np.full(len(smiles), np.nan, dtype=np.float32)
    for start in range(0, len(smiles), batch_size):
        stop = min(start + batch_size, len(smiles))
        molecules = [Chem.MolFromSmiles(value) for value in smiles[start:stop]]
        probabilities = predictor.predict_proba_mols(molecules, batch_size=batch_size)
        output[start:stop] = np.asarray(
            [np.nan if value is None else float(value) for value in probabilities],
            dtype=np.float32,
        )
        if start == 0 or stop == len(smiles) or stop % (batch_size * 20) == 0:
            LOGGER.info("D-MPNN inference rows=%d/%d", stop, len(smiles))
    return output


def indices_for_split(frame: pd.DataFrame, split: str) -> np.ndarray:
    """Return integer row indices for one frozen split."""
    return np.flatnonzero(frame["split"].to_numpy() == split)


@torch.no_grad()
def predict_mlp(
    model: nn.Module,
    features: np.ndarray,
    indices: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Predict endpoint probabilities for indexed feature rows."""
    model.eval()
    output = np.empty(len(indices), dtype=np.float32)
    for start in range(0, len(indices), batch_size):
        stop = min(start + batch_size, len(indices))
        batch_indices = indices[start:stop]
        batch = torch.from_numpy(
            np.array(features[batch_indices], dtype=np.float32, copy=True)
        ).to(device)
        output[start:stop] = torch.sigmoid(model(batch).reshape(-1)).cpu().numpy()
    return output


def train_supervised_mlp(
    frame: pd.DataFrame,
    features: np.ndarray,
    *,
    output_dir: Path,
    device: torch.device,
    seed: int,
    config: MLPTrainingConfig,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Train the project MLP architecture on the frozen endpoint labels."""
    from chembl_classifier.model import ChemblClassifier

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    train_indices = indices_for_split(frame, "train")
    validation_indices = indices_for_split(frame, "validation")
    labels = frame["target"].to_numpy(dtype=np.float32)
    model = ChemblClassifier(
        input_dim=FEATURE_DIM,
        hidden_dims=(512, 128),
        dropout_rates=(0.3, 0.2),
    ).to(device)
    positives = float(labels[train_indices].sum())
    negatives = float(len(train_indices) - positives)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([negatives / max(positives, 1.0)], device=device)
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    checkpoint_dir = output_dir / f"supervised_mlp_seed_{seed}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best.pt"
    latest_path = checkpoint_dir / "latest.pt"
    metrics_rows: list[dict[str, Any]] = []
    best_val_auc = float("-inf")
    best_epoch = 0
    patience = 0
    generator = np.random.default_rng(seed)
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        shuffled = generator.permutation(train_indices)
        total_loss = 0.0
        total_rows = 0
        for start in range(0, len(shuffled), config.batch_size):
            batch_indices = shuffled[start : start + config.batch_size]
            batch_features = torch.from_numpy(
                np.array(features[batch_indices], dtype=np.float32, copy=True)
            ).to(device)
            batch_labels = torch.from_numpy(labels[batch_indices]).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features).reshape(-1)
            loss = criterion(logits, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += float(loss.detach().item()) * len(batch_indices)
            total_rows += len(batch_indices)
        validation_scores = predict_mlp(
            model,
            features,
            validation_indices,
            device=device,
            batch_size=config.batch_size * 2,
        )
        validation_auc = roc_auc_score(
            labels[validation_indices], validation_scores
        )
        row = {
            "seed": seed,
            "epoch": epoch,
            "train_loss": total_loss / max(total_rows, 1),
            "validation_auc": float(validation_auc),
            "best_epoch": best_epoch,
        }
        metrics_rows.append(row)
        payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "seed": seed,
            "config": asdict(config),
            "validation_auc": float(validation_auc),
        }
        torch.save(payload, latest_path)
        if validation_auc > best_val_auc:
            best_val_auc = float(validation_auc)
            best_epoch = epoch
            patience = 0
            torch.save(payload, best_path)
        else:
            patience += 1
        LOGGER.info(
            "supervised_mlp seed=%d epoch=%d train_loss=%.5f val_auc=%.5f best_epoch=%d",
            seed,
            epoch,
            row["train_loss"],
            validation_auc,
            best_epoch,
        )
        if patience >= config.early_stopping_patience:
            break
    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    all_indices = np.arange(len(frame), dtype=np.int64)
    predictions = predict_mlp(
        model,
        features,
        all_indices,
        device=device,
        batch_size=config.batch_size * 2,
    )
    pd.DataFrame(metrics_rows).to_csv(
        checkpoint_dir / "metrics.csv", index=False
    )
    return predictions, metrics_rows


def train_morgan_linear(
    frame: pd.DataFrame,
    features: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit a validation-selected linear classifier on Morgan features."""
    train_indices = indices_for_split(frame, "train")
    validation_indices = indices_for_split(frame, "validation")
    labels = frame["target"].to_numpy(dtype=np.int8)
    best_model: SGDClassifier | None = None
    best_alpha = 0.0
    best_auc = float("-inf")
    for alpha in (1e-5, 1e-4, 1e-3):
        model = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=alpha,
            max_iter=100,
            tol=1e-4,
            class_weight="balanced",
            random_state=1,
            average=True,
        )
        model.fit(
            np.asarray(features[train_indices, :MORGAN_DIM]),
            labels[train_indices],
        )
        validation_scores = model.predict_proba(
            np.asarray(features[validation_indices, :MORGAN_DIM])
        )[:, 1]
        validation_auc = roc_auc_score(
            labels[validation_indices], validation_scores
        )
        LOGGER.info(
            "morgan_linear alpha=%g validation_auc=%.5f",
            alpha,
            validation_auc,
        )
        if validation_auc > best_auc:
            best_auc = float(validation_auc)
            best_alpha = alpha
            best_model = model
    if best_model is None:
        raise RuntimeError("Morgan linear model selection failed")
    scores = best_model.predict_proba(
        np.asarray(features[:, :MORGAN_DIM])
    )[:, 1].astype(np.float32)
    return scores, {"alpha": best_alpha, "validation_auc": best_auc}


def train_descriptor_boosting(
    frame: pd.DataFrame,
    features: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit a descriptor-only histogram gradient boosting classifier."""
    train_indices = indices_for_split(frame, "train")
    validation_indices = indices_for_split(frame, "validation")
    labels = frame["target"].to_numpy(dtype=np.int8)
    best_model: HistGradientBoostingClassifier | None = None
    best_leaf_nodes = 0
    best_auc = float("-inf")
    for max_leaf_nodes in (15, 31, 63):
        model = HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=250,
            max_leaf_nodes=max_leaf_nodes,
            l2_regularization=1e-3,
            class_weight="balanced",
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=1,
        )
        model.fit(
            np.asarray(features[train_indices, MORGAN_DIM:]),
            labels[train_indices],
        )
        validation_scores = model.predict_proba(
            np.asarray(features[validation_indices, MORGAN_DIM:])
        )[:, 1]
        validation_auc = roc_auc_score(
            labels[validation_indices], validation_scores
        )
        LOGGER.info(
            "descriptor_boosting leaves=%d validation_auc=%.5f",
            max_leaf_nodes,
            validation_auc,
        )
        if validation_auc > best_auc:
            best_auc = float(validation_auc)
            best_leaf_nodes = max_leaf_nodes
            best_model = model
    if best_model is None:
        raise RuntimeError("descriptor boosting model selection failed")
    scores = best_model.predict_proba(
        np.asarray(features[:, MORGAN_DIM:])
    )[:, 1].astype(np.float32)
    return scores, {
        "max_leaf_nodes": best_leaf_nodes,
        "validation_auc": best_auc,
    }


def select_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """Select a balanced-accuracy threshold using validation only."""
    finite = np.isfinite(scores)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        labels[finite], scores[finite]
    )
    index = int(np.argmax(true_positive_rate - false_positive_rate))
    threshold = float(thresholds[index])
    if not math.isfinite(threshold):
        threshold = float(np.nanmedian(scores[finite]))
    return threshold


def bootstrap_auc_interval(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    seed: int = 1,
    repeats: int = 300,
) -> tuple[float, float]:
    """Return a stratified bootstrap 95% interval for ROC AUC."""
    finite = np.isfinite(scores)
    labels = labels[finite]
    scores = scores[finite]
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    generator = np.random.default_rng(seed)
    values = np.empty(repeats, dtype=np.float64)
    for repeat in range(repeats):
        sampled = np.concatenate(
            [
                generator.choice(positive, size=len(positive), replace=True),
                generator.choice(negative, size=len(negative), replace=True),
            ]
        )
        values[repeat] = roc_auc_score(labels[sampled], scores[sampled])
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def evaluate_scores(
    frame: pd.DataFrame,
    scores: dict[str, np.ndarray],
    model_metadata: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Evaluate every score with one validation-selected threshold."""
    labels = frame["target"].to_numpy(dtype=np.int8)
    validation_indices = indices_for_split(frame, "validation")
    rows: list[dict[str, Any]] = []
    for model_name, model_scores in scores.items():
        threshold = select_threshold(
            labels[validation_indices], model_scores[validation_indices]
        )
        for split in ("train", "validation", "test"):
            indices = indices_for_split(frame, split)
            split_scores = model_scores[indices]
            split_labels = labels[indices]
            finite = np.isfinite(split_scores)
            usable_scores = split_scores[finite]
            usable_labels = split_labels[finite]
            predicted = (usable_scores >= threshold).astype(np.int8)
            auc = roc_auc_score(usable_labels, usable_scores)
            ci_low = float("nan")
            ci_high = float("nan")
            if split == "test":
                ci_low, ci_high = bootstrap_auc_interval(
                    usable_labels, usable_scores
                )
            rows.append(
                {
                    "model": model_name,
                    **model_metadata[model_name],
                    "split": split,
                    "rows": int(len(indices)),
                    "scored_rows": int(finite.sum()),
                    "positives": int(usable_labels.sum()),
                    "threshold_selected_on": "validation",
                    "threshold": threshold,
                    "roc_auc_pairwise": float(auc),
                    "roc_auc_ci_low": ci_low,
                    "roc_auc_ci_high": ci_high,
                    "average_precision": float(
                        average_precision_score(usable_labels, usable_scores)
                    ),
                    "balanced_accuracy": float(
                        balanced_accuracy_score(usable_labels, predicted)
                    ),
                    "f1": float(f1_score(usable_labels, predicted)),
                    "brier": float(
                        brier_score_loss(
                            usable_labels,
                            np.clip(usable_scores, 0.0, 1.0),
                        )
                    )
                    if np.nanmin(usable_scores) >= 0.0
                    and np.nanmax(usable_scores) <= 1.0
                    else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def make_plots(
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    scores: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    """Create presentation-readable comparison and score-separation plots."""
    test = metrics[metrics["split"].eq("test")].sort_values(
        "roc_auc_pairwise", ascending=True
    )
    colors = [
        "#8A8A8A" if bool(value) else "#2F6B9A"
        for value in test["non_deployable_oracle"]
    ]
    figure_height = max(6.5, 0.62 * len(test) + 2.5)
    figure, axis = plt.subplots(figsize=(13, figure_height), constrained_layout=True)
    y = np.arange(len(test))
    values = test["roc_auc_pairwise"].to_numpy()
    low = values - test["roc_auc_ci_low"].to_numpy()
    high = test["roc_auc_ci_high"].to_numpy() - values
    axis.barh(y, values, color=colors, alpha=0.88, edgecolor="#263238")
    axis.errorbar(
        values,
        y,
        xerr=np.vstack([low, high]),
        fmt="none",
        ecolor="#222222",
        capsize=4,
        linewidth=1.5,
    )
    axis.axvline(0.5, color="#555555", linestyle="--", linewidth=1.6)
    axis.axvline(0.65, color="#D99B29", linestyle=":", linewidth=2.0)
    axis.set_yticks(y, test["model"].tolist())
    axis.set_xlim(0.4, 1.02)
    axis.set_xlabel("Test ROC-AUC / preferred>non-preferred pairwise", fontsize=14)
    axis.set_title("Terminal molecular model comparison", fontsize=19)
    axis.tick_params(axis="both", labelsize=12)
    axis.grid(axis="x", alpha=0.20)
    for position, value in zip(y, values):
        axis.text(value + 0.009, position, f"{value:.3f}", va="center", fontsize=11)
    figure.savefig(output_dir / "test_auc_model_comparison.png", dpi=200)
    figure.savefig(output_dir / "test_auc_model_comparison.svg")
    plt.close(figure)

    candidates = test.loc[~test["non_deployable_oracle"].astype(bool)].tail(5)
    model_order = candidates["model"].tolist()[::-1]
    labels = frame["target"].to_numpy(dtype=np.int8)
    test_indices = indices_for_split(frame, "test")
    figure, axes = plt.subplots(
        len(model_order),
        1,
        figsize=(13, max(8, 3.2 * len(model_order))),
        constrained_layout=True,
    )
    if len(model_order) == 1:
        axes = [axes]
    for axis, model_name in zip(axes, model_order):
        model_scores = scores[model_name][test_indices]
        finite = np.isfinite(model_scores)
        ranks = pd.Series(model_scores[finite]).rank(pct=True).to_numpy()
        test_labels = labels[test_indices][finite]
        axis.hist(
            ranks[test_labels == 0],
            bins=35,
            density=True,
            alpha=0.58,
            color="#D99B29",
            label="non-preferred",
        )
        axis.hist(
            ranks[test_labels == 1],
            bins=35,
            density=True,
            alpha=0.58,
            color="#2F6B9A",
            label="preferred",
        )
        auc = float(
            test.loc[test["model"].eq(model_name), "roc_auc_pairwise"].iloc[0]
        )
        axis.set_title(f"{model_name}  |  test AUC={auc:.3f}", fontsize=15)
        axis.set_xlabel("Within-model test score percentile", fontsize=12)
        axis.set_ylabel("Density", fontsize=12)
        axis.tick_params(labelsize=10)
        axis.grid(axis="y", alpha=0.18)
    axes[0].legend(fontsize=11)
    figure.savefig(output_dir / "test_score_distributions.png", dpi=190)
    figure.savefig(output_dir / "test_score_distributions.svg")
    plt.close(figure)

    generalization = metrics[metrics["split"].isin(["validation", "test"])].copy()
    deployable = generalization.loc[
        ~generalization["non_deployable_oracle"].astype(bool)
    ]
    models = (
        deployable[deployable["split"].eq("test")]
        .sort_values("roc_auc_pairwise", ascending=False)["model"]
        .tolist()
    )
    figure, axis = plt.subplots(
        figsize=(max(13, 1.4 * len(models)), 7.5), constrained_layout=True
    )
    x = np.arange(len(models))
    width = 0.36
    for offset, split, color in (
        (-width / 2, "validation", "#2F6B9A"),
        (width / 2, "test", "#D99B29"),
    ):
        values = [
            float(
                deployable[
                    deployable["model"].eq(model)
                    & deployable["split"].eq(split)
                ]["roc_auc_pairwise"].iloc[0]
            )
            for model in models
        ]
        axis.bar(x + offset, values, width=width, color=color, label=split)
    axis.axhline(0.5, color="#555555", linestyle="--", linewidth=1.4)
    axis.axhline(0.65, color="#777777", linestyle=":", linewidth=1.6)
    axis.set_xticks(x, models, rotation=32, ha="right")
    axis.set_ylim(0.4, 1.0)
    axis.set_ylabel("ROC-AUC / pairwise", fontsize=14)
    axis.set_title("Validation-to-test generalization", fontsize=19)
    axis.legend(fontsize=12)
    axis.tick_params(labelsize=11)
    axis.grid(axis="y", alpha=0.18)
    figure.savefig(output_dir / "validation_test_generalization.png", dpi=190)
    figure.savefig(output_dir / "validation_test_generalization.svg")
    plt.close(figure)


def main() -> None:
    """Run the terminal molecular model comparison."""
    args = parse_args()
    output_dir = args.output_dir.resolve()
    configure_logging(output_dir)
    started = time.time()
    LOGGER.info("Starting terminal molecular model comparison")
    LOGGER.info("manifest=%s output_dir=%s", args.manifest, output_dir)
    LOGGER.info("git_status_checked_before_edit=true formal_training=false")
    device = resolve_device(args.device)
    LOGGER.info(
        "device=%s cuda_available=%s cuda_count=%d device_name=%s",
        device,
        torch.cuda.is_available(),
        torch.cuda.device_count(),
        torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
    )
    frame = load_manifest(args.manifest)
    LOGGER.info(
        "rows=%d split_counts=%s label_counts=%s",
        len(frame),
        frame["split"].value_counts().to_dict(),
        frame[LABEL_COLUMN].value_counts().to_dict(),
    )
    features, feature_valid = build_feature_cache(
        frame,
        manifest_path=args.manifest,
        output_dir=output_dir,
        workers=args.workers,
        rebuild=args.rebuild_features,
    )
    if not bool(np.asarray(feature_valid).all()):
        failed = int((~np.asarray(feature_valid)).sum())
        raise RuntimeError(f"feature cache contains {failed} failed rows")

    labels = frame["target"].to_numpy(dtype=np.int8)
    scores: dict[str, np.ndarray] = {}
    metadata: dict[str, dict[str, Any]] = {}

    oracle_name = "oracle: label fields"
    scores[oracle_name] = (
        -frame["umap_nearest_chembl_distance"].to_numpy(dtype=np.float64)
        - 0.03 * frame["structural_hard_bad"].to_numpy(dtype=np.float64)
    )
    metadata[oracle_name] = {
        "model_family": "label-generating oracle",
        "training_scope": "none",
        "feature_scope": "forbidden label fields",
        "non_deployable_oracle": True,
    }

    pretrained_mlp_name = "pretrained ChEMBL MLP (zero-shot)"
    scores[pretrained_mlp_name] = score_pretrained_mlp(features, device)
    metadata[pretrained_mlp_name] = {
        "model_family": "MLP",
        "training_scope": "external ChEMBL-like prior",
        "feature_scope": "Morgan count + RDKit 2D",
        "non_deployable_oracle": False,
    }

    if not args.skip_pretrained_dmpnn:
        pretrained_dmpnn_name = "pretrained ChEMBL D-MPNN (zero-shot)"
        scores[pretrained_dmpnn_name] = score_pretrained_dmpnn(
            frame["canonical_smiles"].astype(str).tolist(),
            device=device,
            batch_size=args.dmpnn_batch_size,
        )
        metadata[pretrained_dmpnn_name] = {
            "model_family": "D-MPNN",
            "training_scope": "external ChEMBL-like prior",
            "feature_scope": "RDKit atom/bond graph",
            "non_deployable_oracle": False,
        }

    morgan_name = "supervised Morgan linear"
    scores[morgan_name], morgan_selection = train_morgan_linear(frame, features)
    metadata[morgan_name] = {
        "model_family": "linear classifier",
        "training_scope": "rank40 train split",
        "feature_scope": "Morgan count",
        "non_deployable_oracle": False,
        "selection": json.dumps(morgan_selection, sort_keys=True),
    }

    descriptor_name = "supervised descriptor boosting"
    scores[descriptor_name], descriptor_selection = train_descriptor_boosting(
        frame, features
    )
    metadata[descriptor_name] = {
        "model_family": "histogram gradient boosting",
        "training_scope": "rank40 train split",
        "feature_scope": "RDKit 2D descriptors only",
        "non_deployable_oracle": False,
        "selection": json.dumps(descriptor_selection, sort_keys=True),
    }

    mlp_config = MLPTrainingConfig(max_epochs=args.max_epochs)
    seeds = [int(value) for value in str(args.seeds).split(",") if value.strip()]
    mlp_seed_names: list[str] = []
    mlp_seed_scores: list[np.ndarray] = []
    for seed in seeds:
        model_name = f"supervised molecular MLP seed={seed}"
        predictions, training_rows = train_supervised_mlp(
            frame,
            features,
            output_dir=output_dir,
            device=device,
            seed=seed,
            config=mlp_config,
        )
        scores[model_name] = predictions
        mlp_seed_names.append(model_name)
        mlp_seed_scores.append(predictions)
        metadata[model_name] = {
            "model_family": "MLP",
            "training_scope": "rank40 train split",
            "feature_scope": "Morgan count + RDKit 2D",
            "non_deployable_oracle": False,
            "selection": json.dumps(
                {
                    "seed": seed,
                    "epochs_run": len(training_rows),
                    "best_validation_auc": max(
                        row["validation_auc"] for row in training_rows
                    ),
                },
                sort_keys=True,
            ),
        }
    if len(mlp_seed_scores) > 1:
        ensemble_name = "supervised molecular MLP ensemble"
        scores[ensemble_name] = np.mean(np.stack(mlp_seed_scores), axis=0)
        metadata[ensemble_name] = {
            "model_family": "MLP ensemble",
            "training_scope": "rank40 train split",
            "feature_scope": "Morgan count + RDKit 2D",
            "non_deployable_oracle": False,
            "selection": json.dumps({"seeds": seeds}, sort_keys=True),
        }

    metrics = evaluate_scores(frame, scores, metadata)
    metrics.to_csv(output_dir / METRICS_PATH.name, index=False)
    prediction_frame = frame[
        ["trajectory_id", "split", LABEL_COLUMN, "target"]
    ].copy()
    for name, values in scores.items():
        prediction_frame[name] = values
    prediction_frame.to_csv(
        output_dir / PREDICTIONS_PATH.name,
        index=False,
        compression="gzip",
    )
    make_plots(frame, metrics, scores, output_dir)

    test_metrics = metrics[metrics["split"].eq("test")].sort_values(
        "roc_auc_pairwise", ascending=False
    )
    deployable = test_metrics.loc[
        ~test_metrics["non_deployable_oracle"].astype(bool)
    ]
    best = deployable.iloc[0].to_dict()
    summary = {
        "status": "complete",
        "formal_training": False,
        "purpose": "terminal molecular model separability before actor training",
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "rows": len(frame),
        "split_counts": frame["split"].value_counts().to_dict(),
        "label_counts": frame[LABEL_COLUMN].value_counts().to_dict(),
        "excluded_candidate_features": [
            "umap_nearest_chembl_distance",
            "structural_hard_bad",
        ],
        "primary_metric": "scaffold-held-out test ROC-AUC, equivalent to binary pairwise ordering",
        "acceptance_reference": {
            "test_auc_min": 0.65,
            "max_train_test_auc_gap": 0.15,
        },
        "best_deployable_model": best,
        "test_metrics": test_metrics.to_dict(orient="records"),
        "device": str(device),
        "seeds": seeds,
        "elapsed_seconds": time.time() - started,
        "outputs": {
            "metrics": str(output_dir / METRICS_PATH.name),
            "predictions": str(output_dir / PREDICTIONS_PATH.name),
            "auc_plot": str(output_dir / "test_auc_model_comparison.png"),
            "distribution_plot": str(output_dir / "test_score_distributions.png"),
            "generalization_plot": str(output_dir / "validation_test_generalization.png"),
        },
    }
    (output_dir / SUMMARY_PATH.name).write_text(
        json.dumps(
            json_safe(summary),
            ensure_ascii=False,
            indent=2,
            default=str,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    LOGGER.info(
        "Completed best_model=%s test_auc=%.5f elapsed=%.1fs",
        best["model"],
        best["roc_auc_pairwise"],
        summary["elapsed_seconds"],
    )


if __name__ == "__main__":
    main()
