from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


OLD_RUN = Path("OUT/7gpu-dock-classifier-v0-1_20260529_133609/training_summary.csv")
ANTI_CARBON_RUN = Path(
    "OUT/diffsbdd-8gpu-b64-anticarbon-guard-20260611_20260611_155548/training_summary.csv"
)
OUT_DIR = Path("work_records/26-6/260611/docs_archive_四条线方向请教_20260611/figures")


def _load(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["run_label"] = label
    return df


def _safe_float(value: object) -> float:
    if pd.isna(value):
        return float("nan")
    return float(value)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    old = _load(OLD_RUN, "旧 docking+classifier")
    guarded = _load(ANTI_CARBON_RUN, "最后 anti-carbon guard")

    metrics = [
        ("avg_docking", "Docking score", "lower is better; guard weakened docking"),
        ("carbon_fraction", "Carbon fraction", "avoid carbon drift"),
        ("hetero_atom_mean", "Hetero atom count", "avoid hetero atom collapse"),
        ("tpsa_mean", "TPSA", "avoid polarity collapse"),
        ("logp_mean", "LogP", "avoid hydrophobic drift"),
        ("rdkit_valid_ratio", "RDKit valid ratio", "basic validity"),
    ]
    plot_labels = {
        "旧 docking+classifier": "old docking+classifier",
        "最后 anti-carbon guard": "final anti-carbon guard",
    }
    colors = {
        "旧 docking+classifier": "#4C78A8",
        "最后 anti-carbon guard": "#F58518",
    }

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.6))
    for ax, (column, title, note) in zip(axes.ravel(), metrics):
        for df in (old, guarded):
            label = df["run_label"].iloc[0]
            if column not in df.columns:
                continue
            ax.plot(
                df["epoch"],
                df[column],
                marker="o",
                markersize=3,
                linewidth=2,
                color=colors[label],
                label=plot_labels[label],
            )
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.25)
        ax.text(
            0.02,
            0.96,
            note,
            transform=ax.transAxes,
            va="top",
            fontsize=8,
            color="#555555",
        )
        if column == "avg_docking":
            ax.invert_yaxis()
            ax.text(0.02, 0.08, "inverted axis", transform=ax.transAxes, fontsize=8)
        if column == "carbon_fraction":
            ax.axhline(0.82, color="#B279A2", linestyle="--", linewidth=1)
            ax.text(0.98, 0.84, "guard max 0.82", transform=ax.transAxes, ha="right", fontsize=8)
        if column == "rdkit_valid_ratio":
            ax.set_ylim(0.45, 0.95)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("DiffSBDD RL descriptor drift: old run vs final anti-carbon guard run", y=1.07, fontsize=15, fontweight="bold")
    fig.tight_layout()

    png = OUT_DIR / "diffsbdd_rl_descriptor_trends_with_anticarbon_20260611.png"
    svg = OUT_DIR / "diffsbdd_rl_descriptor_trends_with_anticarbon_20260611.svg"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)

    rows = []
    for label, path, df in [
        ("旧 docking+classifier", OLD_RUN, old),
        ("最后 anti-carbon guard", ANTI_CARBON_RUN, guarded),
    ]:
        first = df.iloc[0]
        last = df.iloc[-1]
        row: dict[str, object] = {
            "run": label,
            "source": str(path),
            "epoch_first": int(first["epoch"]),
            "epoch_last": int(last["epoch"]),
        }
        for column, _, _ in metrics:
            if column not in df.columns:
                continue
            row[f"{column}_first"] = _safe_float(first[column])
            row[f"{column}_last"] = _safe_float(last[column])
            row[f"{column}_delta"] = _safe_float(last[column]) - _safe_float(first[column])
        for column in [
            "avg_reward_raw",
            "reward_mean",
            "avg_chemistry_guard_reward",
            "avg_ligand_efficiency",
            "avg_logp_distribution_reward",
        ]:
            if column not in df.columns:
                continue
            row[f"{column}_first"] = _safe_float(first[column])
            row[f"{column}_last"] = _safe_float(last[column])
            row[f"{column}_delta"] = _safe_float(last[column]) - _safe_float(first[column])
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "diffsbdd_rl_old_vs_anticarbon_summary.csv", index=False)
    print(f"wrote {png}")
    print(f"wrote {svg}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
