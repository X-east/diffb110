#!/usr/bin/env python
"""Score original DiffGui and DiffSBDD generations with classifier v0.3.1."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-diffb110")

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

MODEL_PATH = ROOT / "chembl_classifier/models/chembl_classifier_selected_v0_3_1_scaffold_strongpoly.pt"
DIFFGUI_CSV = ROOT / "analysis/diffgui_classifier_score_20260609/diffgui_500_scores.csv"
DIFFSBDD_CSV = ROOT / "analysis/diffsbdd_8gpu_5000_score_20260608/generated_5000_scores.csv"
OUT_DIR = ROOT / "analysis/diffgui_diffsbdd_classifier_v031_20260615"


def _load_smiles(csv_path: Path, source: str) -> pd.DataFrame:
    """Load valid canonical SMILES from an existing generated-molecule score CSV."""
    data = pd.read_csv(csv_path, low_memory=False)
    valid = data[data["scoring_valid"] == 1].copy()
    valid["canonical_smiles"] = valid["canonical_smiles"].astype(str)
    valid = valid[valid["canonical_smiles"].str.len() > 0].copy()
    valid["source"] = source
    keep_cols = [
        "source",
        "mol_id",
        "source_index",
        "canonical_smiles",
        "heavy_atom_count",
        "carbon_fraction",
        "hetero_atom_count",
        "logp",
        "tpsa",
        "qed",
        "ring_count",
        "aromatic_ring_count",
        "three_member_ring_count",
        "fragment_count",
        "guard_violation_count",
    ]
    return valid[[col for col in keep_cols if col in valid.columns]]


def _featurize(smiles: list[str]) -> tuple[np.ndarray, list[int]]:
    """Convert SMILES to the classifier feature matrix."""
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
def _predict(features: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    """Predict classifier v0.3.1 good probabilities."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChemblClassifier.from_checkpoint(str(MODEL_PATH), device=device)
    probs: list[np.ndarray] = []
    for start in range(0, features.shape[0], batch_size):
        batch = torch.as_tensor(features[start : start + batch_size], dtype=torch.float32, device=device)
        probs.append(model.predict_proba(batch).detach().cpu().numpy().reshape(-1))
    return np.concatenate(probs, axis=0).astype(np.float64)


def _summarize(scored: pd.DataFrame) -> pd.DataFrame:
    """Summarize classifier score distribution by source model."""
    rows: list[dict[str, float | int | str]] = []
    for source, group in scored.groupby("source", sort=False):
        probs = group["p_good_v031"].astype(float)
        rows.append(
            {
                "source": source,
                "n": int(len(group)),
                "mean": float(probs.mean()),
                "median": float(probs.median()),
                "q10": float(probs.quantile(0.10)),
                "q20": float(probs.quantile(0.20)),
                "q80": float(probs.quantile(0.80)),
                "q90": float(probs.quantile(0.90)),
                "frac_lt_0p2": float((probs < 0.2).mean()),
                "frac_lt_0p5": float((probs < 0.5).mean()),
                "frac_gt_0p8": float((probs > 0.8).mean()),
            }
        )
    return pd.DataFrame(rows)


def _plot(scored: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Plot classifier score distributions and high/low-score fractions."""
    colors = {"DiffSBDD": "#3268a8", "DiffGui": "#2f7f5f"}
    order = [name for name in ["DiffSBDD", "DiffGui"] if name in set(scored["source"])]

    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [2.0, 1.15]})

    bins = np.linspace(0.0, 1.0, 41)
    for source in order:
        values = scored.loc[scored["source"] == source, "p_good_v031"].astype(float)
        axes[0].hist(
            values,
            bins=bins,
            density=True,
            alpha=0.48,
            color=colors[source],
            label=f"{source} (n={len(values):,})",
        )
    axes[0].axvline(0.2, color="#555555", linestyle="--", linewidth=1.0)
    axes[0].axvline(0.5, color="#111111", linestyle="-", linewidth=1.0)
    axes[0].axvline(0.8, color="#555555", linestyle="--", linewidth=1.0)
    axes[0].set_xlabel("Classifier v0.3.1 p_good")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Original generated molecules")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis="y", alpha=0.25)

    view = summary.set_index("source").loc[order].reset_index()
    x = np.arange(len(view))
    width = 0.24
    axes[1].bar(x - width, view["frac_lt_0p2"] * 100, width=width, label="p < 0.2", color="#991b1b")
    axes[1].bar(x, view["frac_lt_0p5"] * 100, width=width, label="p < 0.5", color="#f97316")
    axes[1].bar(x + width, view["frac_gt_0p8"] * 100, width=width, label="p > 0.8", color="#2563eb")
    axes[1].set_xticks(x, view["source"])
    axes[1].set_ylabel("Fraction of generated molecules (%)")
    axes[1].set_ylim(0, 100)
    axes[1].set_title("Low/high score fraction")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)

    for container in axes[1].containers:
        axes[1].bar_label(container, fmt="%.0f", padding=2, fontsize=7)

    fig.suptitle("Latest ChEMBL classifier v0.3.1: DiffGui vs DiffSBDD", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "diffgui_vs_diffsbdd_classifier_v031_distribution.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / "diffgui_vs_diffsbdd_classifier_v031_distribution.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Score and plot the latest classifier distribution for baseline generations."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.concat(
        [
            _load_smiles(DIFFSBDD_CSV, "DiffSBDD"),
            _load_smiles(DIFFGUI_CSV, "DiffGui"),
        ],
        ignore_index=True,
    )
    features, valid_indices = _featurize(data["canonical_smiles"].tolist())
    probs = _predict(features)

    scored = data.iloc[valid_indices].copy()
    scored["p_good_v031"] = probs
    summary = _summarize(scored)

    scored.to_csv(OUT_DIR / "diffgui_diffsbdd_classifier_v031_scores.csv", index=False)
    summary.to_csv(OUT_DIR / "diffgui_diffsbdd_classifier_v031_summary.csv", index=False)
    _plot(scored, summary)

    payload = {
        "model_path": str(MODEL_PATH.relative_to(ROOT)),
        "diffgui_csv": str(DIFFGUI_CSV.relative_to(ROOT)),
        "diffsbdd_csv": str(DIFFSBDD_CSV.relative_to(ROOT)),
        "total_loaded": int(len(data)),
        "total_scored": int(len(scored)),
        "summary": summary.to_dict(orient="records"),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
