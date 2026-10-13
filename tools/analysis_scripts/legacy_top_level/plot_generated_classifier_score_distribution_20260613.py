#!/usr/bin/env python
"""Plot classifier scores on generated molecules for PPT discussion."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger


ROOT = Path(__file__).resolve().parents[1]
MOLSKILL_ROOT = ROOT / "reference project" / "代码" / "molskill"
if str(MOLSKILL_ROOT) not in sys.path:
    sys.path.insert(0, str(MOLSKILL_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chembl_classifier.model import ChemblClassifier  # noqa: E402
from molskill.data.featurizers import get_featurizer  # noqa: E402


RDLogger.DisableLog("rdApp.*")

SOURCE_CSV = ROOT / "analysis/classifier_v0_3_scaffold_curation_20260610/new_molecule_classification.csv"
MODEL_PATH = ROOT / "chembl_classifier/models/chembl_classifier_selected_v0_3_1_scaffold_strongpoly.pt"
OUT_DIR = ROOT / "analysis/classifier_generated_score_distribution_20260613"


def _featurize(smiles: list[str]) -> tuple[np.ndarray, list[int]]:
    featurizer = get_featurizer("morgan_count_rdkit_2d")
    features: list[np.ndarray] = []
    valid_indices: list[int] = []
    for idx, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        feat = featurizer.get_feat(mol)
        if feat is None or len(feat) != featurizer.dim():
            continue
        features.append(feat.astype(np.float32, copy=False))
        valid_indices.append(idx)
    if not features:
        return np.empty((0, 2220), dtype=np.float32), []
    return np.stack(features, axis=0).astype(np.float32), valid_indices


@torch.no_grad()
def _predict(model_path: Path, features: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChemblClassifier.from_checkpoint(str(model_path), device=device)
    probs: list[np.ndarray] = []
    for start in range(0, features.shape[0], batch_size):
        batch = torch.as_tensor(features[start : start + batch_size], dtype=torch.float32, device=device)
        probs.append(model.predict_proba(batch).detach().cpu().numpy().reshape(-1))
    return np.concatenate(probs, axis=0).astype(np.float64)


def _summary(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for label, group in scored.groupby("label", sort=False):
        probs = group["p_good_v031"].astype(float)
        rows.append(
            {
                "label": label,
                "n": int(len(group)),
                "mean": float(probs.mean()),
                "median": float(probs.median()),
                "q10": float(probs.quantile(0.10)),
                "q90": float(probs.quantile(0.90)),
                "frac_lt_0p2": float((probs < 0.2).mean()),
                "frac_lt_0p5": float((probs < 0.5).mean()),
                "frac_gt_0p8": float((probs > 0.8).mean()),
            }
        )
    return pd.DataFrame(rows)


def _plot(scored: pd.DataFrame, summary: pd.DataFrame) -> None:
    colors = {"bad": "#c2410c", "ambiguous": "#d97706", "good": "#2563eb"}
    labels = [label for label in ["bad", "ambiguous", "good"] if label in set(scored["label"])]

    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [2.0, 1.25]})

    bins = np.linspace(0.0, 1.0, 41)
    for label in labels:
        values = scored.loc[scored["label"] == label, "p_good_v031"].astype(float)
        axes[0].hist(
            values,
            bins=bins,
            density=True,
            alpha=0.42,
            label=f"{label} (n={len(values):,})",
            color=colors.get(label),
        )
    axes[0].axvline(0.2, color="#444444", linestyle="--", linewidth=1.0)
    axes[0].axvline(0.5, color="#111111", linestyle="-", linewidth=1.0)
    axes[0].axvline(0.8, color="#444444", linestyle="--", linewidth=1.0)
    axes[0].set_xlabel("Classifier p_good on generated molecules")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Generated-molecule score distribution")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25)

    plot_summary = summary.set_index("label").loc[labels].reset_index()
    x = np.arange(len(plot_summary))
    width = 0.24
    axes[1].bar(x - width, plot_summary["frac_lt_0p2"] * 100, width=width, label="p < 0.2", color="#991b1b")
    axes[1].bar(x, plot_summary["frac_lt_0p5"] * 100, width=width, label="p < 0.5", color="#f97316")
    axes[1].bar(x + width, plot_summary["frac_gt_0p8"] * 100, width=width, label="p > 0.8", color="#2563eb")
    axes[1].set_xticks(x, plot_summary["label"])
    axes[1].set_ylabel("Fraction of generated molecules (%)")
    axes[1].set_title("How hard is the classifier?")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle("Classifier v0.3.1 scores on generated molecules", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "generated_molecule_classifier_score_distribution.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / "generated_molecule_classifier_score_distribution.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(SOURCE_CSV, low_memory=False)
    data = data[data["label"].isin(["good", "ambiguous", "bad"])].copy()
    data["smiles"] = data["smiles"].astype(str)

    features, valid_indices = _featurize(data["smiles"].tolist())
    probs = _predict(MODEL_PATH, features)

    scored = data.iloc[valid_indices].copy()
    scored["p_good_v031"] = probs
    scored.to_csv(OUT_DIR / "generated_molecule_classifier_scores_v031.csv", index=False)
    summary = _summary(scored)
    summary.to_csv(OUT_DIR / "generated_molecule_classifier_score_summary.csv", index=False)
    _plot(scored, summary)

    payload = {
        "source_csv": str(SOURCE_CSV.relative_to(ROOT)),
        "model_path": str(MODEL_PATH.relative_to(ROOT)),
        "total_rows": int(len(data)),
        "valid_scored": int(len(scored)),
        "summary": summary.to_dict(orient="records"),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
