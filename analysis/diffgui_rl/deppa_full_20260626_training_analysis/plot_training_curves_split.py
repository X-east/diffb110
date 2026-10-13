"""Split the combined Fig 2.3 training curve into per-reward and monitoring figures.

Why
---
The thesis currently carries a single figure (docs/images/fig2_3_training_curve.png)
that overlays three series with two different y-axes: reward_for_advantage_mean,
docking_raw_mean and the RDKit valid ratio. That compresses every reward component
into one noisy line. This script regenerates the training record as a set of
standalone panels:

  reward panels (one per reward function actually used by the main run):
      02a total reward, 02b docking, 02c QED, 02d SA
  monitoring panels:
      02e generation validity, 02f diversity & drift, 02g PPO stability,
      02h runtime & VRAM, 02i molecular property drift

Data
----
Both inputs are the official per-epoch record of the main DiffGui run
(8 GPU, full finetune, t=200, epochs 1-89); they are cross-checked against
thesis Table 2.3 for epoch 1 before anything is drawn.

    official_epoch_summary_selected.csv   (89 rows, source: run log)
    component_epoch_summary.csv           (89 official_merged rows, per-component)

Reward decomposition used by the main run (thesis eq. 2.2):
    reward_total_scored = (docking_reward + QED + SA_mapped) / 3
verified here to reproduce reward_total_scored_mean for the early epochs exactly
and to correlate at 0.994 over all 89 epochs.

Output: split_figures/*.png (300 dpi) and *.svg, next to this script.

Environment: needs seaborn >= 0.13 (the `dif_umap_compat` conda env here)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "split_figures"

OFFICIAL = HERE / "official_epoch_summary_selected.csv"
COMPONENT = HERE / "component_epoch_summary.csv"

# key epochs reported in thesis Table 2.3
EPOCH_FIRST = 1
EPOCH_LAST = 89

C = {
    "black": "#2c3e50",
    "red": "#c0392b",
    "blue": "#2e86c1",
    "green": "#27ae60",
    "orange": "#e67e22",
    "grey": "#7f8c8d",
    "purple": "#8e44ad",
    "teal": "#16a085",
    "brown": "#a04000",
}


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the per-epoch record and run the sanity checks before plotting."""
    off = pd.read_csv(OFFICIAL).sort_values("epoch").reset_index(drop=True)
    comp = pd.read_csv(COMPONENT)
    comp = comp[comp["source"] == "official_merged"].sort_values("epoch")
    comp = comp.reset_index(drop=True)

    if not np.allclose(off["epoch"].to_numpy(), comp["epoch"].to_numpy()):
        raise SystemExit("epoch index mismatch between the two summaries")

    # thesis Table 2.3, epoch 1
    checks = {
        "reward_for_advantage_mean": (-0.1605, off),
        "reward_total_scored_mean": (0.7724, off),
        "docking_raw_mean": (-9.7812, off),
        "molecule_rdkit_valid_ratio": (0.8906, off),
    }
    print("Cross-check against thesis Table 2.3 (epoch 1):")
    for col, (want, frame) in checks.items():
        got = float(frame.loc[frame["epoch"] == EPOCH_FIRST, col].iloc[0])
        flag = "OK" if abs(got - want) < 5e-4 else "MISMATCH"
        print(f"  {col:<30s} {got:>9.4f}  (table {want:>9.4f})  -> {flag}")

    # eq. 2.2 decomposition
    pred = (
        off["docking_reward_mean"]
        + comp["qed_raw_mean"]
        + comp["deppa_sa_raw_mean"]
    ) / 3.0
    actual = off["reward_total_scored_mean"]
    print(
        f"\nEq. 2.2 check (docking_reward + QED + SA)/3 vs reward_total_scored:\n"
        f"  max abs deviation {float(np.abs(pred - actual).max()):.5f}, "
        f"corr {float(np.corrcoef(pred, actual)[0, 1]):.5f}"
    )
    return off, comp


def key_epochs(off: pd.DataFrame) -> dict[str, int]:
    """Return the epoch indices quoted in thesis Table 2.3."""
    return {
        "best_reward": int(off.loc[off["reward_for_advantage_mean"].idxmax(), "epoch"]),
        "best_docking": int(off.loc[off["docking_raw_mean"].idxmin(), "epoch"]),
    }


def new_fig(figsize=(9.4, 4.4), rows=1, cols=1, **kw):
    fig, axes = plt.subplots(rows, cols, figsize=figsize, dpi=300, **kw)
    return fig, axes


def style(ax, ylabel: str, title: str, xlabel: str = "epoch") -> None:
    """Apply the shared look: soft grid, no top/right spine."""
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11.5, pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.grid(alpha=0.28, linewidth=0.7)
    ax.tick_params(labelsize=9)


def mark_key_epochs(ax, keys: dict[str, int]) -> None:
    """Draw the Table 2.3 reference epochs as thin dashed verticals."""
    styles = {
        "best_reward": ("#8e44ad", "best reward (ep %d)"),
        "best_docking": ("#16a085", "best docking (ep %d)"),
    }
    for name, (color, label) in styles.items():
        ep = keys[name]
        ax.axvline(ep, color=color, linestyle="--", linewidth=1.0, alpha=0.75, zorder=1)
        ax.text(
            ep, ax.get_ylim()[1], label % ep,
            rotation=90, ha="right", va="top", fontsize=7.4, color=color,
        )


def save(fig, stem: str) -> list[Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"{stem}.png"
    svg = OUT / f"{stem}.svg"
    fig.tight_layout()
    fig.savefig(png, dpi=300)
    fig.savefig(svg)
    plt.close(fig)
    print(f"  wrote {png.name}")
    return [png, svg]


# --------------------------------------------------------------------------- #
# reward panels
# --------------------------------------------------------------------------- #
def fig_reward_total(off: pd.DataFrame, keys: dict[str, int]) -> None:
    fig, ax = new_fig()
    ax.plot(off.epoch, off.reward_for_advantage_mean, "-o", ms=3.0, lw=1.1,
            color=C["black"], label="reward (for-advantage)")

    ax2 = ax.twinx()
    ax2.plot(off.epoch, off.reward_total_scored_mean, "-s", ms=2.8, lw=1.1,
             color=C["orange"], label="reward_total_scored (absolute)")
    ax2.set_ylabel("reward_total_scored", fontsize=10, color=C["orange"])
    ax2.tick_params(axis="y", labelsize=9, colors=C["orange"])
    ax2.spines["top"].set_visible(False)

    style(ax, "reward (for-advantage)", "Fig 2.3a  Total reward over the main training (ep1-89)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8.4, loc="lower right", framealpha=0.92, ncol=2)
    mark_key_epochs(ax, keys)
    save(fig, "02a_reward_total")


def _component_panel(off, comp, keys, *, raw_col, raw_frame, raw_label, raw_color,
                     norm_col, norm_frame, norm_label, norm_color,
                     title, stem, invert_note=None) -> None:
    """One reward component: raw property on the left axis, reward on the right."""
    fig, ax = new_fig()
    ax.plot(raw_frame.epoch, raw_frame[raw_col], "-o", ms=3.0, lw=1.2,
            color=raw_color, label=raw_label)

    ax2 = ax.twinx()
    ax2.plot(norm_frame.epoch, norm_frame[norm_col], "-s", ms=2.8, lw=1.1,
             color=norm_color, label=norm_label)
    ax2.set_ylabel(norm_label, fontsize=10, color=norm_color)
    ax2.tick_params(axis="y", labelsize=9, colors=norm_color)
    ax2.spines["top"].set_visible(False)

    style(ax, raw_label, title)
    if invert_note:
        ax.text(0.012, 0.965, invert_note, transform=ax.transAxes,
                fontsize=8.0, color=C["grey"], va="top")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8.4, loc="lower right", framealpha=0.92, ncol=2)
    mark_key_epochs(ax, keys)
    save(fig, stem)


def fig_reward_docking(off, comp, keys) -> None:
    _component_panel(
        off, comp, keys,
        raw_col="docking_raw_mean", raw_frame=off,
        raw_label="UniDock score (kcal/mol)", raw_color=C["red"],
        norm_col="docking_reward_mean", norm_frame=off,
        norm_label="docking reward (normalised)", norm_color=C["blue"],
        title="Fig 2.3b  Docking reward component of the main training (ep1-89)",
        stem="02b_reward_docking",
        invert_note="lower score = stronger predicted binding",
    )


def fig_reward_qed(off, comp, keys) -> None:
    """QED enters eq. 2.2 unmodified, so the raw QED *is* the reward term."""
    fig, ax = new_fig()
    ax.plot(comp.epoch, comp.qed_raw_mean, "-o", ms=3.0, lw=1.35, color=C["green"],
            label="QED (RDKit) = QED reward term, eq. 2.2")
    mean = float(comp["qed_raw_mean"].mean())
    ax.axhline(mean, color=C["grey"], ls=":", lw=1.2,
               label=f"mean over training ({mean:.3f})")
    ax.set_ylim(float(comp["qed_raw_mean"].min()) - 0.012,
                float(comp["qed_raw_mean"].max()) + 0.016)
    style(ax, "QED", "Fig 2.3c  QED reward component of the main training (ep1-89)")
    ax.text(0.012, 0.965, "higher is better; QED is used unmodified, no rescaling",
            transform=ax.transAxes, fontsize=8.0, color=C["grey"], va="top")
    ax.legend(fontsize=8.6, loc="lower right", framealpha=0.92)
    mark_key_epochs(ax, keys)
    save(fig, "02c_reward_qed")


def fig_reward_sa(off, comp, keys) -> None:
    _component_panel(
        off, comp, keys,
        raw_col="sa_raw_mean", raw_frame=comp,
        raw_label="SA score (RDKit, 1-10)", raw_color=C["brown"],
        norm_col="deppa_sa_raw_mean", norm_frame=comp,
        norm_label="SA reward (mapped, higher = easier)", norm_color=C["orange"],
        title="Fig 2.3d  SA reward component of the main training (ep1-89)",
        stem="02d_reward_sa",
        invert_note="lower SA = easier to synthesise; the reward is the mapped value",
    )


# --------------------------------------------------------------------------- #
# monitoring panels
# --------------------------------------------------------------------------- #
def fig_monitor_validity(off, keys) -> None:
    fig, ax = new_fig()
    ax.plot(off.epoch, off.reconstruction_success_ratio, "-o", ms=3.0, lw=1.2,
            color=C["blue"], label="reconstruction success")
    ax.plot(off.epoch, off.molecule_rdkit_valid_ratio, "-^", ms=3.0, lw=1.2,
            color=C["green"], label="RDKit valid")
    ax.plot(off.epoch, off.docking_success_ratio_total, "-s", ms=2.8, lw=1.2,
            color=C["red"], label="docking success")
    ax.set_ylim(0.80, 1.01)
    style(ax, "ratio", "Fig 2.3e  Generation validity over the main training (ep1-89)")
    ax.legend(fontsize=8.6, loc="lower right", framealpha=0.92, ncol=3)
    mark_key_epochs(ax, keys)
    save(fig, "02e_monitor_validity")


def fig_monitor_diversity(off, keys) -> None:
    fig, axes = new_fig(figsize=(10.4, 6.0), rows=2, cols=2)
    panels = [
        ("unique_canonical_smiles", "unique canonical SMILES", C["blue"], None),
        ("unique_scaffolds", "unique scaffolds", C["purple"], None),
        ("duplicate_ratio", "duplicate ratio", C["red"], None),
        ("descriptor_mmd_vs_baseline", "descriptor MMD vs baseline", C["orange"], None),
    ]
    for ax, (col, label, color, _) in zip(axes.ravel(), panels):
        ax.plot(off.epoch, off[col], "-o", ms=2.6, lw=1.1, color=color)
        style(ax, label, "", xlabel="")
        ax.set_title(label, fontsize=10.0, pad=6)
        ax.axvline(keys["best_reward"], color=C["purple"], ls="--", lw=0.9, alpha=0.6)
        ax.axvline(keys["best_docking"], color=C["teal"], ls="--", lw=0.9, alpha=0.6)
    fig.suptitle(
        "Fig 2.3f  Diversity and drift over the main training (ep1-89)",
        fontsize=12.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "02f_monitor_diversity.png", dpi=300)
    fig.savefig(OUT / "02f_monitor_diversity.svg")
    plt.close(fig)
    print("  wrote 02f_monitor_diversity.png")


def fig_monitor_ppo(off, keys) -> None:
    fig, axes = new_fig(figsize=(10.4, 6.0), rows=2, cols=2)
    ax = axes[0, 0]
    ax.plot(off.epoch, off.PPO_Loss, "-o", ms=2.6, lw=1.1, color=C["black"])
    style(ax, "PPO_Loss", "", xlabel="")
    ax.set_title("PPO_Loss", fontsize=10.0, pad=6)

    ax = axes[0, 1]
    ax.plot(off.epoch, off.PPO_ApproxKL, "-o", ms=2.6, lw=1.1, color=C["red"])
    ax.set_yscale("log")
    style(ax, "PPO_ApproxKL (log)", "", xlabel="")
    ax.set_title("PPO_ApproxKL (log scale)", fontsize=10.0, pad=6)

    ax = axes[1, 0]
    ax.plot(off.epoch, off.ClipFrac, "-o", ms=2.6, lw=1.1, color=C["orange"])
    style(ax, "ClipFrac", "", xlabel="epoch")
    ax.set_title("ClipFrac", fontsize=10.0, pad=6)

    ax = axes[1, 1]
    ax.plot(off.epoch, off.RatioMean, "-o", ms=2.6, lw=1.1, color=C["blue"],
            label="RatioMean")
    ax2 = ax.twinx()
    ax2.plot(off.epoch, off.GradNorm, "-s", ms=2.4, lw=1.0, color=C["green"],
             label="GradNorm")
    ax2.set_ylabel("GradNorm", fontsize=9, color=C["green"])
    ax2.tick_params(axis="y", labelsize=8.6, colors=C["green"])
    ax2.spines["top"].set_visible(False)
    style(ax, "RatioMean", "", xlabel="epoch")
    ax.set_title("RatioMean / GradNorm", fontsize=10.0, pad=6)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8.2, loc="best", framealpha=0.9)

    fig.suptitle(
        "Fig 2.3g  PPO stability over the main training (ep1-89)",
        fontsize=12.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "02g_monitor_ppo.png", dpi=300)
    fig.savefig(OUT / "02g_monitor_ppo.svg")
    plt.close(fig)
    print("  wrote 02g_monitor_ppo.png")


def fig_monitor_runtime(off, keys) -> None:
    fig, ax = new_fig()
    ax.plot(off.epoch, off.ppo_seconds, "-o", ms=2.8, lw=1.2, color=C["black"],
            label="PPO update")
    ax.plot(off.epoch, off.rollout_seconds, "-s", ms=2.6, lw=1.2, color=C["blue"],
            label="rollout")
    ax.plot(off.epoch, off.reward_seconds, "-^", ms=2.6, lw=1.2, color=C["orange"],
            label="reward computation")
    ax.set_ylabel("seconds per epoch", fontsize=10)

    ax2 = ax.twinx()
    ax2.plot(off.epoch, off.gpu_monitor_peak_used_mb / 1024.0, "-d", ms=2.6, lw=1.1,
             color=C["grey"], label="GPU peak (GiB, per card)")
    ax2.set_ylabel("GPU peak used (GiB)", fontsize=10, color=C["grey"])
    ax2.tick_params(axis="y", labelsize=9, colors=C["grey"])
    ax2.spines["top"].set_visible(False)

    style(ax, "seconds per epoch", "Fig 2.3h  Runtime and memory of the main training (ep1-89)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8.4, loc="center right", framealpha=0.92, ncol=2)
    save(fig, "02h_monitor_runtime")


def fig_monitor_properties(off, keys) -> None:
    fig, axes = new_fig(figsize=(10.8, 6.4), rows=2, cols=3)
    panels = [
        ("logp_mean", "logP mean", C["blue"]),
        ("tpsa_mean", "TPSA mean", C["teal"]),
        ("ring_count_mean", "ring count mean", C["purple"]),
        ("aromatic_ring_mean", "aromatic ring mean", C["orange"]),
        ("three_member_ring_total", "three-member-ring total", C["red"]),
        ("carbon_fraction", "carbon fraction", C["brown"]),
    ]
    for ax, (col, label, color) in zip(axes.ravel(), panels):
        ax.plot(off.epoch, off[col], "-o", ms=2.6, lw=1.1, color=color)
        style(ax, label, "", xlabel="")
        ax.set_title(label, fontsize=10.0, pad=6)
        ax.axvline(keys["best_reward"], color=C["purple"], ls="--", lw=0.9, alpha=0.6)
        ax.axvline(keys["best_docking"], color=C["teal"], ls="--", lw=0.9, alpha=0.6)
    fig.suptitle(
        "Fig 2.3i  Molecular property drift over the main training (ep1-89)",
        fontsize=12.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "02i_monitor_properties.png", dpi=300)
    fig.savefig(OUT / "02i_monitor_properties.svg")
    plt.close(fig)
    print("  wrote 02i_monitor_properties.png")


# --------------------------------------------------------------------------- #
def main() -> None:
    off, comp = load()
    keys = key_epochs(off)
    print(f"\nKey epochs from Table 2.3: best reward = ep{keys['best_reward']}, "
          f"best docking = ep{keys['best_docking']}")
    print("\nRendering:")
    fig_reward_total(off, keys)
    fig_reward_docking(off, comp, keys)
    fig_reward_qed(off, comp, keys)
    fig_reward_sa(off, comp, keys)
    fig_monitor_validity(off, keys)
    fig_monitor_diversity(off, keys)
    fig_monitor_ppo(off, keys)
    fig_monitor_runtime(off, keys)
    fig_monitor_properties(off, keys)
    print(f"\nAll figures under {OUT}")


if __name__ == "__main__":
    main()
