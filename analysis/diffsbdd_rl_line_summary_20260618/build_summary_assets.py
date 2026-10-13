from __future__ import annotations

"""Build figures and compact CSV summaries for the DiffSBDD RL line review."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "analysis" / "diffsbdd_rl_line_summary_20260618"
FORMAL_SUMMARY = ROOT / "OUT" / "7gpu-dock-classifier-v0-1_20260529_133609" / "training_summary.csv"
SINGLE_REWARD_FINAL = (
    ROOT
    / "analysis"
    / "diffsbdd_rl"
    / "diffsbdd_single_reward_matrix_20260613"
    / "final_epoch_comparison.csv"
)
RESET_SUMMARY = (
    ROOT
    / "analysis"
    / "diffsbdd_rl"
    / "diffsbdd_rl_reset_plan_20260611"
    / "selected_run_readable_summary.csv"
)


def _style() -> None:
    """Apply a compact matplotlib style for report figures."""
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 180,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
        }
    )


def build_formal_run_delta() -> pd.DataFrame:
    """Write start/end deltas for the formal dock+classifier run."""
    df = pd.read_csv(FORMAL_SUMMARY)
    cols = [
        "rdkit_valid_ratio",
        "avg_docking",
        "avg_chembl_classifier_reward",
        "carbon_fraction",
        "hetero_atom_mean",
        "tpsa_mean",
        "logp_mean",
        "aromatic_ring_mean",
    ]
    first = df.iloc[0]
    last = df.iloc[-1]
    rows = []
    for col in cols:
        start = float(first[col])
        end = float(last[col])
        rows.append(
            {
                "metric": col,
                "epoch1": start,
                "epoch25": end,
                "delta": end - start,
            }
        )
    delta = pd.DataFrame(rows)
    delta.to_csv(OUT_DIR / "formal_dock_classifier_start_end_delta.csv", index=False)
    return df


def plot_formal_run(df: pd.DataFrame) -> None:
    """Plot descriptor drift for the formal dock+classifier run."""
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0))
    axes = axes.reshape(-1)

    axes[0].plot(df["epoch"], df["rdkit_valid_ratio"], color="#2f6c99", linewidth=2)
    axes[0].set_title("RDKit valid ratio")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Ratio")

    axes[1].plot(df["epoch"], df["avg_docking"], color="#4b8f3a", linewidth=2)
    axes[1].set_title("Docking score")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Avg docking")

    axes[2].plot(df["epoch"], df["carbon_fraction"], label="Carbon fraction", color="#b14d3b", linewidth=2)
    axes[2].plot(df["epoch"], df["hetero_atom_mean"] / 10.0, label="Hetero atoms / 10", color="#6f5aa7", linewidth=2)
    axes[2].set_title("Composition drift")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Scaled value")
    axes[2].legend(frameon=False)

    axes[3].plot(df["epoch"], df["tpsa_mean"], label="TPSA", color="#b57b2a", linewidth=2)
    twin = axes[3].twinx()
    twin.plot(df["epoch"], df["logp_mean"], label="LogP", color="#276c68", linewidth=2)
    axes[3].set_title("Polarity / hydrophobicity drift")
    axes[3].set_xlabel("Epoch")
    axes[3].set_ylabel("TPSA")
    twin.set_ylabel("LogP")
    lines, labels = axes[3].get_legend_handles_labels()
    lines2, labels2 = twin.get_legend_handles_labels()
    axes[3].legend(lines + lines2, labels + labels2, frameon=False, loc="best")

    fig.suptitle("DiffSBDD RL formal run: dock + classifier v0.1", y=0.995)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "formal_dock_classifier_descriptor_drift.png", bbox_inches="tight")
    plt.close(fig)


def build_single_reward_summary() -> pd.DataFrame:
    """Write compact final-epoch metrics for the single-reward matrix."""
    df = pd.read_csv(SINGLE_REWARD_FINAL)
    keep = [
        "reward",
        "rdkit_valid_ratio",
        "fragmented_ratio",
        "avg_docking",
        "carbon_fraction",
        "hetero_atom_mean",
        "tpsa_mean",
        "logp_mean",
        "ring_pathology_flag_mean",
        "descriptor_mmd_vs_baseline",
    ]
    summary = df[keep].copy()
    summary.to_csv(OUT_DIR / "single_reward_final_key_metrics.csv", index=False)
    return summary


def plot_single_reward(summary: pd.DataFrame) -> None:
    """Plot final-epoch comparison bars for the single-reward matrix."""
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.2))
    axes = axes.reshape(-1)
    rewards = summary["reward"].tolist()
    x = range(len(rewards))

    specs = [
        ("rdkit_valid_ratio", "Validity", "#2f6c99"),
        ("avg_docking", "Avg docking", "#4b8f3a"),
        ("ring_pathology_flag_mean", "Ring pathology fraction", "#b14d3b"),
        ("descriptor_mmd_vs_baseline", "Descriptor MMD vs ChEMBL", "#6f5aa7"),
    ]
    for ax, (col, title, color) in zip(axes, specs):
        ax.bar(x, summary[col], color=color, alpha=0.85)
        ax.set_title(title)
        ax.set_xticks(list(x))
        ax.set_xticklabels(rewards, rotation=35, ha="right")
        ax.set_ylabel(col)

    fig.suptitle("DiffSBDD single-reward matrix: final epoch key metrics", y=0.995)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "single_reward_final_comparison.png", bbox_inches="tight")
    plt.close(fig)


def build_legacy_run_summary() -> pd.DataFrame:
    """Write compact start/end metrics for representative legacy runs."""
    df = pd.read_csv(RESET_SUMMARY)
    keep = [
        "run_label",
        "epochs",
        "reward_setup",
        "rdkit_valid_ratio_start",
        "rdkit_valid_ratio_end",
        "avg_docking_start",
        "avg_docking_end",
        "carbon_fraction_start",
        "carbon_fraction_end",
        "tpsa_mean_start",
        "tpsa_mean_end",
        "logp_mean_start",
        "logp_mean_end",
        "diagnosis",
    ]
    summary = df[keep].copy()
    summary.to_csv(OUT_DIR / "legacy_rl_run_start_end_summary.csv", index=False)
    return summary


def plot_legacy_run_summary(summary: pd.DataFrame) -> None:
    """Plot descriptor drift and docking change across representative runs."""
    labels = summary["run_label"].tolist()
    y = range(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2), sharey=True)

    axes[0].barh(y, summary["carbon_fraction_end"] - summary["carbon_fraction_start"], color="#b14d3b", alpha=0.85)
    axes[0].axvline(0, color="#333333", linewidth=0.8)
    axes[0].set_yticks(list(y))
    axes[0].set_yticklabels(labels)
    axes[0].set_title("Carbon fraction delta")
    axes[0].set_xlabel("End - start")

    axes[1].barh(y, summary["avg_docking_end"] - summary["avg_docking_start"], color="#4b8f3a", alpha=0.85)
    axes[1].axvline(0, color="#333333", linewidth=0.8)
    axes[1].set_title("Docking delta")
    axes[1].set_xlabel("End - start")

    fig.suptitle("Representative DiffSBDD RL runs: drift versus docking", y=0.995)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "legacy_run_drift_vs_docking.png", bbox_inches="tight")
    plt.close(fig)


def build_architecture_diagram() -> None:
    """Draw the current DiffSBDD RL runtime flow diagram."""
    fig, ax = plt.subplots(figsize=(12.0, 4.8))
    ax.axis("off")
    steps = [
        ("YAML / launch", "launch/run.sh\nlaunch/*.yaml"),
        ("DDP runtime", "main_diffsbdd_ddp.py\nbuild_runtime_ddp"),
        ("DiffSBDD rollout", "DiffSBDDModel.sample\nlatents/logps/timesteps"),
        ("Molecule semantics", "reconstruct_molecule\nRDKit sanitize/connectivity"),
        ("Reward", "CompositeReward\nStabilityAwareReward"),
        ("PPO update", "EDMActor\nclip + optional KL"),
        ("Records", "training_summary.csv\nmolecules.csv/SDF\ncheckpoints"),
    ]
    xs = [0.06 + i * 0.145 for i in range(len(steps))]
    y = 0.55
    for idx, ((title, body), x) in enumerate(zip(steps, xs)):
        box = plt.Rectangle((x - 0.055, y - 0.16), 0.11, 0.26, fill=True, color="#f4f2ed", ec="#444444", lw=1.0)
        ax.add_patch(box)
        ax.text(x, y + 0.035, title, ha="center", va="center", fontsize=9, weight="bold")
        ax.text(x, y - 0.055, body, ha="center", va="center", fontsize=7.6)
        if idx < len(steps) - 1:
            ax.annotate(
                "",
                xy=(xs[idx + 1] - 0.06, y - 0.03),
                xytext=(x + 0.058, y - 0.03),
                arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "#333333"},
            )
    ax.text(
        0.5,
        0.18,
        "Key engineering points: CPU-staged rollout data; streamed PPO microbatches; local reward, descriptor, molecule and checkpoint records.",
        ha="center",
        fontsize=9,
    )
    fig.savefig(OUT_DIR / "diffsbdd_rl_architecture_flow.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Build all report assets."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _style()
    formal = build_formal_run_delta()
    plot_formal_run(formal)
    single = build_single_reward_summary()
    plot_single_reward(single)
    legacy = build_legacy_run_summary()
    plot_legacy_run_summary(legacy)
    build_architecture_diagram()


if __name__ == "__main__":
    main()
