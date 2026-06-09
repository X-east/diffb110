from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-diffb110")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
HELPER_PATH = (
    ROOT
    / "analysis/classifier_v0_2_hardneg_20260608/compare_v0_1_v0_2_on_curated_pool.py"
)
CURATED = ROOT / "analysis/good_bad_classifier_curation_20260608/curated_molecules.csv.gz"
MODEL_V01 = ROOT / "chembl_classifier/models/chembl_classifier_selected_v0_1.pt"
MODEL_V02 = ROOT / "chembl_classifier/models/chembl_classifier_selected_v0_2_hardneg.pt"
MODEL_V021 = ROOT / "chembl_classifier/models/chembl_classifier_selected_v0_2_1_weighted.pt"
OUT_DIR = ROOT / "analysis/classifier_v0_2_1_weighted_20260608/curated_pool_v0_1_v0_2_v0_2_1"


def _load_helper():
    spec = importlib.util.spec_from_file_location("curated_compare_helper", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load helper from {HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["curated_compare_helper"] = module
    spec.loader.exec_module(module)
    return module


def _summarize(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows: list[dict] = []
    for key, sub in df.groupby(group_col, dropna=False):
        rec = {group_col: key, "n": int(len(sub))}
        for tag in ["v0_1", "v0_2", "v0_2_1"]:
            col = f"prob_{tag}"
            rec[f"{tag}_mean"] = float(sub[col].mean())
            rec[f"{tag}_median"] = float(sub[col].median())
            rec[f"{tag}_ge_0_5"] = float((sub[col] >= 0.5).mean())
            rec[f"{tag}_ge_0_8"] = float((sub[col] >= 0.8).mean())
        rec["v0_2_1_minus_v0_2_ge_0_5"] = rec["v0_2_1_ge_0_5"] - rec["v0_2_ge_0_5"]
        rec["v0_2_1_minus_v0_1_ge_0_5"] = rec["v0_2_1_ge_0_5"] - rec["v0_1_ge_0_5"]
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def _explode_bad_reasons(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.Series] = []
    for _, row in df.iterrows():
        for reason in [r for r in str(row.get("bad_reasons", "")).split("|") if r and r != "nan"]:
            new = row.copy()
            new["bad_reason"] = reason
            rows.append(new)
    return pd.DataFrame(rows)


def _plot_bad_reason(summary: pd.DataFrame) -> None:
    view = summary[summary["n"] >= 50].sort_values("v0_1_ge_0_5", ascending=True)
    y = np.arange(len(view))
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(y - 0.24, view["v0_1_ge_0_5"] * 100, height=0.24, label="v0.1", color="#F28E2B")
    ax.barh(y, view["v0_2_ge_0_5"] * 100, height=0.24, label="v0.2", color="#4E79A7")
    ax.barh(y + 0.24, view["v0_2_1_ge_0_5"] * 100, height=0.24, label="v0.2.1 weighted", color="#59A14F")
    ax.set_yticks(y)
    ax.set_yticklabels(view["bad_reason"])
    ax.set_xlabel("P(ChEMBL-like) >= 0.5 (%)")
    ax.set_title("Curated bad-reason false positive rate")
    ax.legend()
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "bad_reason_fp_rate_v0_1_v0_2_v0_2_1.png", dpi=180)
    fig.savefig(OUT_DIR / "bad_reason_fp_rate_v0_1_v0_2_v0_2_1.svg")


def _plot_label(summary: pd.DataFrame) -> None:
    view = summary.sort_values("label")
    x = np.arange(len(view))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - 0.24, view["v0_1_mean"], width=0.24, label="v0.1", color="#F28E2B")
    ax.bar(x, view["v0_2_mean"], width=0.24, label="v0.2", color="#4E79A7")
    ax.bar(x + 0.24, view["v0_2_1_mean"], width=0.24, label="v0.2.1 weighted", color="#59A14F")
    ax.set_xticks(x)
    ax.set_xticklabels(view["label"], rotation=20, ha="right")
    ax.set_ylabel("Mean P(ChEMBL-like)")
    ax.set_title("Curated pool mean probability by label")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "label_mean_probability_v0_1_v0_2_v0_2_1.png", dpi=180)
    fig.savefig(OUT_DIR / "label_mean_probability_v0_1_v0_2_v0_2_1.svg")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    helper = _load_helper()
    device = "cuda" if helper.torch.cuda.is_available() else "cpu"

    df = pd.read_csv(CURATED)
    df = df[df["parse_ok"].eq(1) & df["canonical_smiles"].notna()]
    df = df.drop_duplicates("canonical_smiles").reset_index(drop=True)
    X, kept = helper._featurize(df["canonical_smiles"].tolist())
    df = df.iloc[kept].reset_index(drop=True)
    df["prob_v0_1"] = helper._predict(MODEL_V01, X, device)
    df["prob_v0_2"] = helper._predict(MODEL_V02, X, device)
    df["prob_v0_2_1"] = helper._predict(MODEL_V021, X, device)
    df.to_csv(OUT_DIR / "curated_pool_three_model_scores.csv.gz", index=False)

    label_summary = _summarize(df, "label")
    bad_reason_summary = _summarize(_explode_bad_reasons(df), "bad_reason")
    primary_summary = _summarize(df, "primary_reason")

    label_summary.to_csv(OUT_DIR / "comparison_by_label.csv", index=False)
    bad_reason_summary.to_csv(OUT_DIR / "comparison_by_bad_reason.csv", index=False)
    primary_summary.to_csv(OUT_DIR / "comparison_by_primary_reason.csv", index=False)
    _plot_bad_reason(bad_reason_summary)
    _plot_label(label_summary)


if __name__ == "__main__":
    main()
