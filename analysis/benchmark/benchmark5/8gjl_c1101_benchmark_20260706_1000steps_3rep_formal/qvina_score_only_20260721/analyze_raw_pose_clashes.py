"""Quantify receptor clashes in the raw poses used by QVina score-only."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
MODEL_ORDER = ["DiffGui", "DiffSBDD", "PMDM", "GCDM-SBDD", "TargetDiff"]
COLORS = {
    "DiffGui": "#0F4D92",
    "DiffSBDD": "#42949E",
    "PMDM": "#D69E2E",
    "GCDM-SBDD": "#9A4D8E",
    "TargetDiff": "#B64342",
}


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    }
)


def read_heavy_atom_coordinates(pdbqt_path: Path) -> np.ndarray:
    """Read non-hydrogen atom coordinates from an AutoDock PDBQT file."""
    coordinates = []
    for line in pdbqt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        atom_type = line.split()[-1].upper()
        if atom_type.startswith("H"):
            continue
        coordinates.append(
            [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        )
    return np.asarray(coordinates, dtype=float)


def calculate_contacts(scores: pd.DataFrame, receptor_path: Path) -> pd.DataFrame:
    """Calculate the minimum ligand-receptor heavy-atom distance per pose."""
    receptor_tree = cKDTree(read_heavy_atom_coordinates(receptor_path))
    successful = scores[
        scores["success"] & scores["model"].isin(MODEL_ORDER)
    ].copy()
    successful["minimum_receptor_distance_A"] = [
        receptor_tree.query(read_heavy_atom_coordinates(Path(path)), k=1)[0].min()
        for path in successful["pdbqt_path"]
    ]
    successful["clash_lt_0_8A"] = successful["minimum_receptor_distance_A"] < 0.8
    successful["clash_lt_1_2A"] = successful["minimum_receptor_distance_A"] < 1.2
    successful["contact_lt_1_5A"] = successful["minimum_receptor_distance_A"] < 1.5
    return successful


def summarize_contacts(contacts: pd.DataFrame) -> pd.DataFrame:
    """Summarize raw-pose receptor contacts by model."""
    rows = []
    for model in MODEL_ORDER:
        model_rows = contacts[contacts["model"] == model]
        rows.append(
            {
                "model": model,
                "score_success": len(model_rows),
                "median_minimum_receptor_distance_A": model_rows[
                    "minimum_receptor_distance_A"
                ].median(),
                "rate_clash_lt_0_8A": model_rows["clash_lt_0_8A"].mean(),
                "rate_clash_lt_1_2A": model_rows["clash_lt_1_2A"].mean(),
                "rate_contact_lt_1_5A": model_rows["contact_lt_1_5A"].mean(),
            }
        )
    return pd.DataFrame(rows)


def plot_contacts(contacts: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Plot model-level clash rates and the GCDM distance-score relation."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)

    x = np.arange(len(MODEL_ORDER))
    width = 0.24
    metrics = [
        ("rate_clash_lt_0_8A", "<0.8 A", "#B64342"),
        ("rate_clash_lt_1_2A", "<1.2 A", "#D69E2E"),
        ("rate_contact_lt_1_5A", "<1.5 A", "#42949E"),
    ]
    ordered = summary.set_index("model").loc[MODEL_ORDER]
    for offset, (column, label, color) in zip((-width, 0, width), metrics):
        axes[0].bar(x + offset, 100 * ordered[column], width, label=label, color=color)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(MODEL_ORDER, rotation=22)
    axes[0].set_ylabel("Raw poses with close receptor contact (%)")
    axes[0].legend(title="Minimum heavy-atom distance")
    axes[0].grid(axis="y", color="#D8D8D8", linewidth=0.5, alpha=0.7)
    axes[0].text(-0.13, 1.04, "a", transform=axes[0].transAxes, fontweight="bold", fontsize=10)

    gcdm = contacts[contacts["model"] == "GCDM-SBDD"]
    axes[1].scatter(
        gcdm["minimum_receptor_distance_A"],
        gcdm["affinity_kcal_mol"],
        s=8,
        alpha=0.35,
        color=COLORS["GCDM-SBDD"],
        edgecolors="none",
    )
    axes[1].axvline(1.2, color="#D69E2E", linestyle="--", linewidth=0.9)
    axes[1].axhline(0, color="#888888", linewidth=0.7)
    axes[1].set_xlabel("Minimum receptor distance (A)")
    axes[1].set_ylabel("GCDM-SBDD score-only affinity (kcal/mol)")
    axes[1].set_ylim(-15, 100)
    axes[1].grid(color="#D8D8D8", linewidth=0.5, alpha=0.7)
    axes[1].text(-0.13, 1.04, "b", transform=axes[1].transAxes, fontweight="bold", fontsize=10)

    for suffix in ("png", "svg"):
        fig.savefig(RESULTS_DIR / f"raw_pose_receptor_clashes.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Build durable raw-pose clash evidence from score-only PDBQT files."""
    scores = pd.read_csv(RESULTS_DIR / "per_ligand_scores.csv")
    scores["success"] = scores["success"].astype(bool)
    manifest = json.loads((RESULTS_DIR / "run_manifest.json").read_text())
    contacts = calculate_contacts(scores, Path(manifest["receptor"]))
    summary = summarize_contacts(contacts)
    contacts.to_csv(RESULTS_DIR / "raw_pose_receptor_contacts.csv", index=False)
    summary.to_csv(RESULTS_DIR / "raw_pose_receptor_clash_summary.csv", index=False)
    plot_contacts(contacts, summary)


if __name__ == "__main__":
    main()
