"""Plot the evidence hierarchy for the DiffSBDD E2PO master audit."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

SKILL_SCRIPTS = Path(
    "/home/md666/.codex/plugins/cache/nature-skills/"
    "nature-skills/0.1.0/skills/nature-figure/scripts"
)
sys.path.insert(0, str(SKILL_SCRIPTS))
from audit_panel_alignment import require_matplotlib_panel_alignment


ROOT = Path(__file__).resolve().parent
TRAINING_CSV = ROOT / "training_experiment_ledger.csv"
GENERATION_CSV = ROOT / "generation_evidence_ledger.csv"
OUT = ROOT / "e2po_evidence_overview"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value else np.nan


def main() -> None:
    training = _rows(TRAINING_CSV)
    generation = _rows(GENERATION_CSV)

    timestep_ids = [
        "fixed_gap_e2po_epoch5",
        "strong_docking_x_only",
        "strong_docking_v_only",
        "strong_docking_native_joint",
        "ecfp4_native_joint",
        "structure_x_only_low_budget",
        "structure_x_only_full_lr1e5",
        "structure_xv10_1_full_lr1e5",
        "docking_chembl_50_50_full_lr1e5",
    ]
    by_id = {row["experiment_id"]: row for row in training}
    trows = [by_id[item] for item in timestep_ids]
    labels = [
        "Fixed-gap pilot",
        "Dock x-only",
        "Dock v-only",
        "Dock native-joint",
        "ECFP4 native-joint",
        "Structure x-only low",
        "Structure x-only full",
        "Structure x:v=10:1",
        "Dock+ChEMBL 50:50",
    ]
    palette = {
        "neutral": "#6E7781",
        "x": "#3B82B8",
        "v": "#D37A2C",
        "joint": "#8C6BB1",
        "structure": "#2A9D8F",
        "combined": "#C6535A",
    }
    colors = [
        palette["neutral"],
        palette["x"],
        palette["v"],
        palette["joint"],
        palette["joint"],
        palette["x"],
        palette["structure"],
        palette["combined"],
        palette["combined"],
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.35))
    fig.subplots_adjust(left=0.21, right=0.93, bottom=0.20, top=0.79, wspace=0.38)

    ax = axes[0]
    y = np.arange(len(trows))
    acc = np.array([_float(row, "test_pair_accuracy") for row in trows])
    ax.barh(y, acc - 0.5, left=0.5, color=colors, height=0.66)
    ax.axvline(0.5, color="#333333", linewidth=0.9, linestyle="--")
    ax.set_xlim(0.45, 1.01)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Held-out pair accuracy")
    ax.set_title("Training proxy fit")
    for yi, value in zip(y, acc):
        ax.text(min(value + 0.008, 0.985), yi, f"{value:.3f}", va="center", fontsize=7)

    ax = axes[1]
    timesteps = [50, 125, 250, 375, 500]
    keyrows = [
        ("Structure x-only full", by_id["structure_x_only_full_lr1e5"], palette["structure"], "o"),
        ("Dock+ChEMBL 50:50", by_id["docking_chembl_50_50_full_lr1e5"], palette["combined"], "s"),
        ("Dock v-only", by_id["strong_docking_v_only"], palette["v"], "^"),
    ]
    for label, row, color, marker in keyrows:
        values = [_float(row, f"test_t{step:04d}") for step in timesteps]
        ax.plot(timesteps, values, marker=marker, color=color, linewidth=1.8, label=label)
    ax.axhline(0.5, color="#333333", linewidth=0.9, linestyle="--")
    ax.set_xticks(timesteps)
    ax.set_ylim(0.45, 1.02)
    ax.set_xlabel("Diffusion timestep")
    ax.set_ylabel("Held-out pair accuracy")
    ax.set_title("Signal depends on timestep and head")
    ax.legend(loc="upper center", frameon=False)

    ax = axes[2]
    grow = {row["model"]: row for row in generation}
    selected = ["Original", "Strong_docking_x_only", "Strong_docking_v_only", "Strong_docking_native_joint"]
    display = ["Original", "x-only", "v-only", "native-joint"]
    x = np.arange(len(selected))
    dock_gain = np.array([-(_float(grow[item], "docking_median") - _float(grow["Original"], "docking_median")) for item in selected])
    guard_delta = np.array([100 * (_float(grow[item], "guard_violation_rate") - _float(grow["Original"], "guard_violation_rate")) for item in selected])
    width = 0.36
    ax.bar(x - width / 2, dock_gain, width, color="#4C8C6B", label="Docking gain (-kcal/mol)")
    ax2 = ax.twinx()
    ax2.bar(x + width / 2, guard_delta, width, color="#C6535A", label="Guard change (pp)")
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax2.axhline(0, color="#555555", linewidth=0.8)
    ax.set_xticks(x, display, rotation=18, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Docking improvement")
    ax2.set_ylabel("Guard-violation change (percentage points)")
    ax.set_title("Docking gain is coupled to risk drift")
    handles = [ax.patches[0], ax2.patches[0]]
    ax.legend(handles, ["Docking gain", "Guard change"], loc="upper left", frameon=False)
    ax2.spines["top"].set_visible(False)

    for label, axis in zip("abc", axes):
        axis.text(-0.12, 1.05, label, transform=axis.transAxes, fontsize=11, fontweight="bold", va="top")

    fig.suptitle(
        "DiffSBDD endpoint E2PO: proxy learning is real but does not guarantee generation quality",
        fontsize=12,
        fontweight="bold",
    )
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        json_out=str(OUT.with_suffix(".alignment.json")),
        overlay_svg=str(OUT.with_suffix(".alignment.svg")),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        exclude_axes=[ax2],
        strict=True,
    )
    fig.savefig(OUT.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
