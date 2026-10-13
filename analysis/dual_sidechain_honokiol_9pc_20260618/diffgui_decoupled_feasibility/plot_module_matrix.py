from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "module_matrix.csv"
PNG_PATH = BASE_DIR / "diffgui_dual_r_risk_matrix.png"


def main() -> None:
    rows = []
    with CSV_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row["implementation_risk"] = int(row["implementation_risk"])
            row["scientific_value"] = int(row["scientific_value"])
            rows.append(row)

    phase_order = {phase: idx for idx, phase in enumerate(["P0", "P1", "P2", "P3", "P4", "P5"])}
    colors = {
        "P0": "#4C78A8",
        "P1": "#F58518",
        "P2": "#54A24B",
        "P3": "#B279A2",
        "P4": "#E45756",
        "P5": "#72B7B2",
    }

    fig, ax = plt.subplots(figsize=(9.5, 6.2), dpi=180)
    for row in rows:
        phase = row["phase"]
        x = row["implementation_risk"]
        y = row["scientific_value"]
        jitter = (phase_order[phase] - 2.5) * 0.035
        ax.scatter(
            x + jitter,
            y + jitter,
            s=190,
            color=colors.get(phase, "#666666"),
            edgecolor="white",
            linewidth=1.2,
            alpha=0.95,
            label=phase,
        )
        ax.text(
            x + jitter + 0.05,
            y + jitter + 0.03,
            row["module"],
            fontsize=7.2,
            color="#222222",
        )

    ax.axvspan(4.5, 5.5, color="#F8D7DA", alpha=0.35, zorder=0)
    ax.axhspan(4.5, 5.5, color="#D8F3DC", alpha=0.25, zorder=0)
    ax.set_xlim(0.7, 5.45)
    ax.set_ylim(2.6, 5.45)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_yticks([3, 4, 5])
    ax.set_xlabel("Implementation risk (1 low - 5 high)")
    ax.set_ylabel("Scientific value (1 low - 5 high)")
    ax.set_title("DiffGui dual-R decoupled module risk/value map", pad=12)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)

    handles, labels = ax.get_legend_handles_labels()
    unique = {}
    for handle, label in zip(handles, labels):
        unique.setdefault(label, handle)
    ax.legend(
        [unique[key] for key in sorted(unique, key=phase_order.get)],
        sorted(unique, key=phase_order.get),
        title="Phase",
        loc="lower right",
        frameon=True,
        fontsize=8,
        title_fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(PNG_PATH, bbox_inches="tight")


if __name__ == "__main__":
    main()
