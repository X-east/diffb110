"""Evaluate the v0.1 ChEMBL classifier on the 260407 generated SDF."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-diffb110")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from rdkit import Chem

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MOLSKILL_ROOT = PROJECT_ROOT / "reference project" / "molskill"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(MOLSKILL_ROOT) not in sys.path:
    sys.path.insert(0, str(MOLSKILL_ROOT))

from chembl_classifier.model import ChemblClassifier  # noqa: E402
from molskill.data.featurizers import get_featurizer  # noqa: E402

SDF_PATH = PROJECT_ROOT / "OUT/260407_generate_ligands_10000/260407generated_10000_filtered.sdf"
MODEL_PATH = PROJECT_ROOT / "chembl_classifier/models/chembl_classifier_selected_v0_1.pt"
OUT_DIR = PROJECT_ROOT / "analysis/classifier_test_260407_sdf_20260527"

GOOD_THRESHOLD = 0.5
BAD_THRESHOLD = 0.5
CAP_HIGH = 0.8
CAP_LOW = 0.2
MAX_REWARD = 1.0
PENALTY = -1.0
BATCH_SIZE = 512


def map_probability_to_reward(prob: np.ndarray) -> np.ndarray:
    """Use the same threshold mapping as ChemblClassifierReward."""
    reward = np.zeros_like(prob, dtype=np.float32)

    high_mask = prob >= GOOD_THRESHOLD
    low_mask = prob < BAD_THRESHOLD

    scale_h = max(CAP_HIGH - GOOD_THRESHOLD, 1e-8)
    high_scaled = np.clip((prob[high_mask] - GOOD_THRESHOLD) / scale_h, 0.0, 1.0)
    reward[high_mask] = MAX_REWARD * high_scaled

    scale_l = max(BAD_THRESHOLD - CAP_LOW, 1e-8)
    low_scaled = np.clip((BAD_THRESHOLD - prob[low_mask]) / scale_l, 0.0, 1.0)
    reward[low_mask] = PENALTY * low_scaled
    return reward


def molecule_flags(mol: Chem.Mol) -> dict[str, float | int | bool]:
    """Compute reward-hacking risk descriptors for one molecule."""
    atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
    heavy_atoms = len(atoms)
    carbon_atoms = sum(atom.GetAtomicNum() == 6 for atom in atoms)
    hetero_atoms = sum(atom.GetAtomicNum() not in (1, 6) for atom in atoms)
    hetero_symbols = sorted({atom.GetSymbol() for atom in atoms if atom.GetAtomicNum() not in (1, 6)})

    ring_info = mol.GetRingInfo()
    atom_rings = [tuple(ring) for ring in ring_info.AtomRings()]
    three_member_ring = any(len(ring) == 3 for ring in atom_rings)
    aromatic_rings = 0
    benzene_like_rings = 0
    for ring in atom_rings:
        if all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
            aromatic_rings += 1
            if len(ring) == 6 and all(mol.GetAtomWithIdx(idx).GetAtomicNum() == 6 for idx in ring):
                benzene_like_rings += 1

    fused_pairs = 0
    for i, ring_a in enumerate(atom_rings):
        atoms_a = set(ring_a)
        for ring_b in atom_rings[i + 1 :]:
            if len(atoms_a.intersection(ring_b)) >= 2:
                fused_pairs += 1

    carbon_fraction = carbon_atoms / heavy_atoms if heavy_atoms else 0.0
    return {
        "heavy_atoms": heavy_atoms,
        "carbon_atoms": carbon_atoms,
        "hetero_atoms": hetero_atoms,
        "hetero_types": len(hetero_symbols),
        "hetero_symbols": ",".join(hetero_symbols),
        "carbon_fraction": carbon_fraction,
        "aromatic_rings": aromatic_rings,
        "benzene_like_rings": benzene_like_rings,
        "has_benzene_like_ring": benzene_like_rings > 0,
        "three_member_ring": three_member_ring,
        "ring_count": len(atom_rings),
        "fused_ring_pairs": fused_pairs,
        "ring_complex": fused_pairs >= 2 or len(atom_rings) >= 4,
        "c_ge_090": carbon_fraction >= 0.90,
        "hetero_le_1": hetero_atoms <= 1,
        "co_only_mostly_carbon": (
            carbon_fraction >= 0.85
            and hetero_atoms > 0
            and all(sym in {"O"} for sym in hetero_symbols)
        ),
    }


def summarize(rows: pd.DataFrame) -> dict[str, object]:
    """Build compact JSON summary from per-molecule rows."""
    valid = rows[rows["valid"]].copy()
    if valid.empty:
        return {"n_total": int(len(rows)), "n_valid": 0}

    summary: dict[str, object] = {
        "input_sdf": str(SDF_PATH.relative_to(PROJECT_ROOT)),
        "model_path": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
        "n_total": int(len(rows)),
        "n_valid": int(len(valid)),
        "valid_rate": float(len(valid) / len(rows)),
        "probability": {
            "mean": float(valid["chembl_probability"].mean()),
            "median": float(valid["chembl_probability"].median()),
            "p05": float(valid["chembl_probability"].quantile(0.05)),
            "p25": float(valid["chembl_probability"].quantile(0.25)),
            "p75": float(valid["chembl_probability"].quantile(0.75)),
            "p95": float(valid["chembl_probability"].quantile(0.95)),
        },
        "mapped_reward": {
            "mean": float(valid["mapped_reward"].mean()),
            "median": float(valid["mapped_reward"].median()),
            "p05": float(valid["mapped_reward"].quantile(0.05)),
            "p95": float(valid["mapped_reward"].quantile(0.95)),
        },
        "threshold_rates": {
            "prob_ge_0_5": float((valid["chembl_probability"] >= 0.5).mean()),
            "prob_ge_0_8": float((valid["chembl_probability"] >= 0.8).mean()),
            "prob_lt_0_2": float((valid["chembl_probability"] < 0.2).mean()),
        },
    }

    flag_cols = [
        "c_ge_090",
        "hetero_le_1",
        "co_only_mostly_carbon",
        "three_member_ring",
        "ring_complex",
        "has_benzene_like_ring",
    ]
    summary["risk_flag_rates"] = {
        col: float(valid[col].mean())
        for col in flag_cols
    }
    summary["risk_flag_probability_means"] = {
        col: float(valid.loc[valid[col], "chembl_probability"].mean())
        if bool(valid[col].any())
        else None
        for col in flag_cols
    }
    return summary


def write_plots(rows: pd.DataFrame) -> None:
    """Write diagnostic plots for classifier behavior on generated molecules."""
    valid = rows[rows["valid"]].copy()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(valid["chembl_probability"], bins=50, color="#2563eb", alpha=0.85)
    ax.axvline(0.2, color="#991b1b", linestyle="--", linewidth=1)
    ax.axvline(0.5, color="#111827", linestyle="--", linewidth=1)
    ax.axvline(0.8, color="#166534", linestyle="--", linewidth=1)
    ax.set_xlabel("P(ChEMBL-like)")
    ax.set_ylabel("Molecule count")
    ax.set_title("Classifier probability on 260407 generated SDF")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "classifier_probability_hist.png", dpi=220)
    plt.close(fig)

    risk_cols = [
        "c_ge_090",
        "hetero_le_1",
        "co_only_mostly_carbon",
        "three_member_ring",
        "ring_complex",
        "has_benzene_like_ring",
    ]
    rates = valid[risk_cols].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(rates.index, rates.values * 100, color="#c2410c")
    ax.set_xlabel("Rate among valid molecules (%)")
    ax.set_title("Risk structure rates")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "risk_flag_rates.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        valid["carbon_fraction"],
        valid["chembl_probability"],
        s=8,
        alpha=0.35,
        c=valid["hetero_atoms"],
        cmap="viridis",
    )
    ax.set_xlabel("Carbon fraction")
    ax.set_ylabel("P(ChEMBL-like)")
    ax.set_title("Classifier probability vs carbon fraction")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "probability_vs_carbon_fraction.png", dpi=220)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChemblClassifier.from_checkpoint(str(MODEL_PATH), device=str(device))
    featurizer = get_featurizer("morgan_count_rdkit_2d")

    supplier = Chem.SDMolSupplier(str(SDF_PATH), sanitize=True, removeHs=False)
    rows: list[dict[str, object]] = []
    features: list[np.ndarray] = []
    feature_row_indices: list[int] = []

    for idx, mol in enumerate(supplier):
        if mol is None:
            rows.append({"sdf_index": idx, "valid": False})
            continue

        smiles = Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True)
        row: dict[str, object] = {
            "sdf_index": idx,
            "valid": True,
            "smiles": smiles,
        }
        row.update(molecule_flags(mol))
        rows.append(row)

        try:
            feat = featurizer.get_feat(mol)
        except Exception as exc:
            rows[-1]["valid"] = False
            rows[-1]["featurization_error"] = repr(exc)
            continue
        if feat is None:
            rows[-1]["valid"] = False
            rows[-1]["featurization_error"] = "None feature"
            continue
        features.append(feat)
        feature_row_indices.append(len(rows) - 1)

    if features:
        X = torch.tensor(np.stack(features, axis=0), dtype=torch.float32)
        probs: list[np.ndarray] = []
        model.eval()
        with torch.no_grad():
            for start in range(0, len(X), BATCH_SIZE):
                batch = X[start : start + BATCH_SIZE].to(device)
                prob = model.predict_proba(batch).detach().cpu().numpy().reshape(-1)
                probs.append(prob)
        prob_arr = np.concatenate(probs)
        reward_arr = map_probability_to_reward(prob_arr)
        for row_idx, prob, reward in zip(feature_row_indices, prob_arr, reward_arr):
            rows[row_idx]["chembl_probability"] = float(prob)
            rows[row_idx]["mapped_reward"] = float(reward)

    df = pd.DataFrame(rows)
    df["valid"] = df["valid"].fillna(False).astype(bool)
    for col in ["chembl_probability", "mapped_reward"]:
        if col not in df:
            df[col] = np.nan
    scored = df["chembl_probability"].notna()
    df.loc[~scored, "valid"] = False

    csv_path = OUT_DIR / "260407generated_10000_filtered_classifier_scores.csv.gz"
    df.to_csv(csv_path, index=False)

    summary = summarize(df)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    write_plots(df)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
