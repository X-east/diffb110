from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-diffb110")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from rdkit import Chem


ROOT = Path(__file__).resolve().parents[2]
MOLSKILL_ROOT = ROOT / "reference project" / "molskill"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(MOLSKILL_ROOT) not in sys.path:
    sys.path.insert(0, str(MOLSKILL_ROOT))

from chembl_classifier.model import ChemblClassifier  # noqa: E402
from molskill.data.featurizers import get_featurizer  # noqa: E402


CURATED = ROOT / "analysis/good_bad_classifier_curation_20260608/curated_molecules.csv.gz"
MODEL_V01 = ROOT / "chembl_classifier/models/chembl_classifier_selected_v0_1.pt"
MODEL_V02 = ROOT / "chembl_classifier/models/chembl_classifier_selected_v0_2_hardneg.pt"
OUT_DIR = ROOT / "analysis/classifier_v0_2_hardneg_20260608/curated_pool_v0_1_vs_v0_2"
BATCH_SIZE = 512


def _predict(model_path: Path, X: np.ndarray, device: str) -> np.ndarray:
    model = ChemblClassifier.from_checkpoint(str(model_path), device=device)
    model.eval()
    probs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(X), BATCH_SIZE):
            batch = torch.from_numpy(X[start : start + BATCH_SIZE]).float().to(device)
            probs.append(model.predict_proba(batch).cpu().numpy().reshape(-1))
    return np.concatenate(probs)


def _featurize(smiles: list[str]) -> tuple[np.ndarray, list[int]]:
    featurizer = get_featurizer("morgan_count_rdkit_2d")
    features: list[np.ndarray] = []
    kept: list[int] = []
    for idx, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            feat = featurizer.get_feat(mol)
        except Exception:
            feat = None
        if feat is None:
            continue
        features.append(feat)
        kept.append(idx)
    return np.stack(features, axis=0).astype(np.float32), kept


def _summarize_group(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for keys, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec = {col: val for col, val in zip(group_cols, keys)}
        rec.update(
            {
                "n": int(len(sub)),
                "v01_prob_mean": float(sub["prob_v0_1"].mean()),
                "v02_prob_mean": float(sub["prob_v0_2"].mean()),
                "v01_prob_median": float(sub["prob_v0_1"].median()),
                "v02_prob_median": float(sub["prob_v0_2"].median()),
                "v01_prob_ge_0_5": float((sub["prob_v0_1"] >= 0.5).mean()),
                "v02_prob_ge_0_5": float((sub["prob_v0_2"] >= 0.5).mean()),
                "v01_prob_ge_0_8": float((sub["prob_v0_1"] >= 0.8).mean()),
                "v02_prob_ge_0_8": float((sub["prob_v0_2"] >= 0.8).mean()),
                "prob_ge_0_5_delta": float(
                    (sub["prob_v0_2"] >= 0.5).mean() - (sub["prob_v0_1"] >= 0.5).mean()
                ),
                "prob_mean_delta": float(sub["prob_v0_2"].mean() - sub["prob_v0_1"].mean()),
            }
        )
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def _explode_bad_reasons(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.Series] = []
    for _, row in df.iterrows():
        reasons = [r for r in str(row.get("bad_reasons", "")).split("|") if r and r != "nan"]
        for reason in reasons:
            new = row.copy()
            new["bad_reason"] = reason
            rows.append(new)
    return pd.DataFrame(rows)


def _plot_reason_fp(reason_summary: pd.DataFrame) -> None:
    bad = reason_summary[reason_summary["n"] >= 50].copy()
    bad = bad.sort_values("v01_prob_ge_0_5", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    y = np.arange(len(bad))
    ax.barh(y - 0.18, bad["v01_prob_ge_0_5"] * 100, height=0.36, label="v0.1", color="#F28E2B")
    ax.barh(y + 0.18, bad["v02_prob_ge_0_5"] * 100, height=0.36, label="v0.2 hardneg", color="#4E79A7")
    ax.set_yticks(y)
    ax.set_yticklabels(bad["bad_reason"])
    ax.set_xlabel("P(ChEMBL-like) >= 0.5 (%)")
    ax.set_title("Curated bad-reason false positive rate: v0.1 vs v0.2")
    ax.legend()
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "bad_reason_fp_rate_v0_1_vs_v0_2.png", dpi=180)
    fig.savefig(OUT_DIR / "bad_reason_fp_rate_v0_1_vs_v0_2.svg")


def _plot_label_distribution(label_summary: pd.DataFrame) -> None:
    labels = label_summary["label"].astype(str).tolist()
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - 0.18, label_summary["v01_prob_mean"], width=0.36, label="v0.1", color="#F28E2B")
    ax.bar(x + 0.18, label_summary["v02_prob_mean"], width=0.36, label="v0.2 hardneg", color="#4E79A7")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Mean P(ChEMBL-like)")
    ax.set_title("Curated pool mean probability by label")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "label_mean_probability_v0_1_vs_v0_2.png", dpi=180)
    fig.savefig(OUT_DIR / "label_mean_probability_v0_1_vs_v0_2.svg")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(CURATED)
    keep = df["parse_ok"].eq(1) & df["canonical_smiles"].notna()
    df = df[keep].drop_duplicates("canonical_smiles").reset_index(drop=True)

    X, kept = _featurize(df["canonical_smiles"].tolist())
    df = df.iloc[kept].reset_index(drop=True)
    df["prob_v0_1"] = _predict(MODEL_V01, X, device)
    df["prob_v0_2"] = _predict(MODEL_V02, X, device)
    df["prob_delta_v0_2_minus_v0_1"] = df["prob_v0_2"] - df["prob_v0_1"]
    df.to_csv(OUT_DIR / "curated_pool_v0_1_vs_v0_2_scores.csv.gz", index=False)

    label_summary = _summarize_group(df, ["label"])
    primary_summary = _summarize_group(df, ["label", "primary_reason"])
    bad_reason_summary = _summarize_group(_explode_bad_reasons(df), ["bad_reason"])
    label_summary.to_csv(OUT_DIR / "comparison_by_label.csv", index=False)
    primary_summary.to_csv(OUT_DIR / "comparison_by_primary_reason.csv", index=False)
    bad_reason_summary.to_csv(OUT_DIR / "comparison_by_bad_reason.csv", index=False)

    _plot_reason_fp(bad_reason_summary)
    _plot_label_distribution(label_summary)


if __name__ == "__main__":
    main()
