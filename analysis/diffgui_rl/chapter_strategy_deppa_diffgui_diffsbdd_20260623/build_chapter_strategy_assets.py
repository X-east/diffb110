"""Build chapter-planning evidence assets for DiffGui RL simplification."""

from __future__ import annotations

import csv
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
SUMMARY_CSV = ROOT / "analysis/diffgui_rl_major_runs_20260618/major_runs_summary.csv"


PAPER_ROWS = [
    {
        "source": "DiffSBDD",
        "data": "CrossDocked 100k/100 test; Binding MOAD 40,344 train/130 test.",
        "reward_or_main_control": "No RL reward in the base model; later iterative noising/denoising ranks by user oracles.",
        "ring_role": "Reports small and large rings over-represented, but not as the main reward.",
        "molecule_selection": "Figure examples are filtered for QED > 0.5 and similarity to Enamine molecules.",
    },
    {
        "source": "DiffGui",
        "data": "PDBbind for training, CrossDocked for generalization; 10,000 molecules for broad metric comparisons.",
        "reward_or_main_control": "Property guidance uses affinity, QED, SA, LogP, TPSA as conditioning signals.",
        "ring_role": "Explicitly discusses three/four-member, extra-large, fused, and seven-member ring issues.",
        "molecule_selection": "For Fig. 5/6, 100 molecules per target/model, then visualizes best docking among PB-valid.",
    },
    {
        "source": "DePPA",
        "data": "CrossDocked2020 refined to 100,000 train pairs and 100 novel test proteins.",
        "reward_or_main_control": "Gaussian-rank reward: 0.5 Vina + 0.2 QED + 0.2 SA + 0.1 diversity.",
        "ring_role": "No ring reward in the main objective; uses Complete, strain, clash, RMSD for quality checks.",
        "molecule_selection": "Samples 100 ligands per pocket; top-N variant selects explored ligands by Vina/QED/SA z-score.",
    },
    {
        "source": "This chapter",
        "data": "Existing DiffGui RL OUT runs plus one clean DePPA-like run on the RE-CmeB pocket.",
        "reward_or_main_control": "Use simple Vina/QED/SA/diversity reward; keep ring facts as monitoring and final filters.",
        "ring_role": "Treat rings as a diagnostic and candidate-screening constraint, not the headline reward.",
        "molecule_selection": "Report distribution shift, then show top valid candidates after docking and property filters.",
    },
]


PLAN_ROWS = [
    {
        "stage": "Baseline",
        "what_to_run": "Sample the same target with frozen DiffGui, same timesteps and molecule count.",
        "purpose": "Create the no-RL comparison pool for docking, QED, SA, diversity, and ring diagnostics.",
    },
    {
        "stage": "Main RL",
        "what_to_run": "Train DePPA-like DiffGui reward with Vina/QED/SA/diversity for 20-50 epochs.",
        "purpose": "Make a clean chapter story: online RL improves candidate-pool quality.",
    },
    {
        "stage": "Ablation",
        "what_to_run": "Compare no-RL, simple DePPA-like reward, and one existing ring/pathology run.",
        "purpose": "Show complex ring rewards are optional diagnostics rather than the central contribution.",
    },
    {
        "stage": "Selection",
        "what_to_run": "Generate 500-1000 molecules, keep RDKit/PB-valid and docking-success molecules, rank top-N.",
        "purpose": "Match DiffGui/DiffSBDD/DePPA paper practice: the final examples come from a filtered candidate pool.",
    },
]


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _wrap(value: object, width: int = 28) -> str:
    return "\n".join(textwrap.wrap(str(value), width=width, break_long_words=False))


def build_paper_matrix_plot(rows: list[dict[str, str]]) -> None:
    """Render a compact visual table for the chapter strategy comparison."""
    columns = [
        ("source", "Paper / use"),
        ("data", "Data"),
        ("reward_or_main_control", "Reward / control"),
        ("ring_role", "Ring role"),
        ("molecule_selection", "Molecule selection"),
    ]
    cell_text = [[_wrap(row[key]) for key, _label in columns] for row in rows]
    labels = [label for _key, label in columns]

    fig, ax = plt.subplots(figsize=(16, 7.5))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        colLabels=labels,
        cellLoc="left",
        colLoc="left",
        colWidths=[0.11, 0.20, 0.24, 0.20, 0.25],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 3.0)
    header_color = "#2f5f73"
    row_colors = ["#f3f7f6", "#fff8ed", "#f6f1fa", "#f7f7f7"]
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#d0d7de")
        cell.set_linewidth(0.7)
        if row_idx == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor(row_colors[(row_idx - 1) % len(row_colors)])
            if col_idx == 0:
                cell.set_text_props(weight="bold")
    ax.set_title(
        "Paper Practice vs DiffGui RL Chapter Strategy",
        fontsize=16,
        weight="bold",
        pad=18,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "paper_strategy_matrix.png", dpi=220)
    fig.savefig(OUT_DIR / "paper_strategy_matrix.svg")
    plt.close(fig)


def build_next6_metrics() -> pd.DataFrame:
    """Extract the existing DiffGui next6 metrics relevant to chapter framing."""
    df = pd.read_csv(SUMMARY_CSV)
    next6 = df[df["run_alias"].astype(str).str.startswith("DG-NEXT-")].copy()
    keep_cols = [
        "run_alias",
        "label",
        "reward_weights",
        "last_reward_total_mean",
        "delta_reward_total_mean",
        "last_valid_ratio",
        "last_docking_success_ratio_total",
        "last_docking_raw_mean",
        "last_descriptor_mmd_vs_baseline",
        "last_ring_count_mean",
        "last_aromatic_ring_mean",
        "last_three_member_ring_mean",
        "last_PPO_ApproxKL",
    ]
    next6 = next6[keep_cols].sort_values("run_alias")
    next6.to_csv(OUT_DIR / "existing_next6_metrics_for_chapter.csv", index=False)
    return next6


def build_next6_plot(next6: pd.DataFrame) -> None:
    """Plot four existing trade-off metrics from the current DiffGui RL runs."""
    labels = [alias.replace("DG-NEXT-", "N") for alias in next6["run_alias"]]
    metrics = [
        ("last_valid_ratio", "Valid ratio", "higher"),
        ("last_docking_raw_mean", "Docking raw mean", "lower"),
        ("last_descriptor_mmd_vs_baseline", "Descriptor MMD", "lower"),
        ("last_three_member_ring_mean", "Three-member rings", "lower"),
    ]
    colors = ["#2f7f7b", "#b65f40", "#6f5ba7", "#8a7f2f"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for ax, (metric, title, direction), color in zip(axes.ravel(), metrics, colors):
        values = pd.to_numeric(next6[metric], errors="coerce")
        ax.bar(labels, values, color=color, alpha=0.88)
        ax.set_title(f"{title} ({direction} is better)", fontsize=12, weight="bold")
        ax.tick_params(axis="x", rotation=0)
        ax.grid(axis="y", color="#d8dee4", linewidth=0.7, alpha=0.9)
        for idx, value in enumerate(values):
            if pd.notna(value):
                ax.text(idx, value, f"{value:.3g}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Existing DiffGui next6 Runs: Reward Simplicity Trade-offs", fontsize=15, weight="bold")
    fig.savefig(OUT_DIR / "existing_next6_tradeoff_for_chapter.png", dpi=220)
    fig.savefig(OUT_DIR / "existing_next6_tradeoff_for_chapter.svg")
    plt.close(fig)


def main() -> None:
    """Write CSV and figures used by the work record."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(OUT_DIR / "paper_strategy_matrix.csv", PAPER_ROWS)
    _write_csv(OUT_DIR / "recommended_experiment_plan.csv", PLAN_ROWS)
    build_paper_matrix_plot(PAPER_ROWS)
    next6 = build_next6_metrics()
    build_next6_plot(next6)


if __name__ == "__main__":
    main()
