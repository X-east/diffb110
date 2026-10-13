"""Build reproducible evidence for the DecompDPO-to-DiffSBDD adaptation audit."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent
EXTREME_MANIFEST = (
    PROJECT_ROOT
    / "analysis/offline_iql_extreme_good_bad_pool_260822/inputs/selected_extremes_manifest.csv.gz"
)


def _pairing_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute no-reuse exact-heavy-atom-count preference-pair coverage."""
    rows: list[dict[str, float | int | str]] = []
    for split in ("train", "validation", "test"):
        subset = frame.loc[frame["split"] == split]
        good = Counter(
            subset.loc[
                subset["extreme_label"] == "good", "heavy_atom_count"
            ].astype(int)
        )
        bad = Counter(
            subset.loc[
                subset["extreme_label"] == "hard_bad", "heavy_atom_count"
            ].astype(int)
        )
        exact_pairs = sum(min(good[count], bad[count]) for count in set(good) | set(bad))
        rows.append(
            {
                "split": split,
                "good_rows": sum(good.values()),
                "hard_bad_rows": sum(bad.values()),
                "exact_heavy_atom_pairs_without_reuse": exact_pairs,
                "good_coverage": exact_pairs / sum(good.values()),
                "hard_bad_coverage": exact_pairs / sum(bad.values()),
                "minimum_heavy_atom_count": min(set(good) | set(bad)),
                "maximum_heavy_atom_count": max(set(good) | set(bad)),
            }
        )
    return pd.DataFrame(rows)


def _adaptation_matrix() -> pd.DataFrame:
    """Return the audited component-level portability decisions."""
    return pd.DataFrame(
        [
            {
                "component": "pairwise log-sigmoid objective",
                "decompdpo_role": "current/reference loss difference for high vs low",
                "diffsbdd_native_equivalent": "per-sample normalized epsilon error difference",
                "portability": "borrow",
                "decision": "Reuse the objective form, not DecompDiff tensor code.",
            },
            {
                "component": "frozen reference model",
                "decompdpo_role": "pretrained DecompDiff checkpoint",
                "diffsbdd_native_equivalent": "frozen original DiffSBDD checkpoint",
                "portability": "borrow_with_fix",
                "decision": "Freeze requires_grad and evaluate under no_grad.",
            },
            {
                "component": "terminal generated samples",
                "decompdpo_role": "generated ligand x0 stored in LMDB/PyG",
                "diffsbdd_native_equivalent": "shard x, categorical, nodesxsample",
                "portability": "borrow_with_adapter",
                "decision": "Read native trajectory payload; do not create LMDB/PyG.",
            },
            {
                "component": "random diffusion level",
                "decompdpo_role": "one t sampled from DecompDiff 1000-step training schedule",
                "diffsbdd_native_equivalent": "one t sampled from checkpoint-native 500-step schedule",
                "portability": "borrow_with_adapter",
                "decision": "Use checkpoint ddpm.T, not 1000-step rollout discretization.",
            },
            {
                "component": "decomp_mask and priors",
                "decompdpo_role": "noise centers/stds, arm/scaffold features, LocalDPO",
                "diffsbdd_native_equivalent": "global pocket-conditioned zero-CoM noising",
                "portability": "do_not_borrow",
                "decision": "Preserve DiffSBDD noising and equivariant centering.",
            },
            {
                "component": "categorical atom diffusion",
                "decompdpo_role": "discrete posterior KL",
                "diffsbdd_native_equivalent": "continuous Gaussian noise on one-hot channels",
                "portability": "do_not_borrow",
                "decision": "Use DiffSBDD epsilon error over atom-feature channels.",
            },
            {
                "component": "bond diffusion",
                "decompdpo_role": "explicit categorical fully-connected edge diffusion",
                "diffsbdd_native_equivalent": "absent in current base checkpoint",
                "portability": "do_not_borrow",
                "decision": "No bond term in the first DiffSBDD-DPO prototype.",
            },
            {
                "component": "1001-step trajectory log probability",
                "decompdpo_role": "not used",
                "diffsbdd_native_equivalent": "available through offline policy adapter",
                "portability": "separate_experiment",
                "decision": "Do not present as DecompDPO reproduction.",
            },
        ]
    )


def _plot_pairing_coverage(frame: pd.DataFrame) -> None:
    """Plot exact-heavy-atom-count pair coverage by frozen split."""
    figure, axis = plt.subplots(figsize=(10.5, 6.5), constrained_layout=True)
    positions = range(len(frame))
    width = 0.34
    axis.bar(
        [position - width / 2 for position in positions],
        frame["good_coverage"],
        width,
        label="Good coverage",
        color="#2F6B9A",
    )
    axis.bar(
        [position + width / 2 for position in positions],
        frame["hard_bad_coverage"],
        width,
        label="Hard-bad coverage",
        color="#C96B3C",
    )
    axis.set_xticks(list(positions), frame["split"])
    axis.set_ylim(0.85, 1.0)
    axis.set_ylabel("Coverage by exact heavy-atom pairing", fontsize=14)
    axis.set_xlabel("Frozen split", fontsize=14)
    axis.set_title("Preference-pair feasibility without sample reuse", fontsize=18)
    axis.tick_params(labelsize=12)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=12)
    for position, row in frame.iterrows():
        axis.text(
            position - width / 2,
            row["good_coverage"] + 0.003,
            f'{row["good_coverage"]:.1%}',
            ha="center",
            fontsize=11,
        )
        axis.text(
            position + width / 2,
            row["hard_bad_coverage"] + 0.003,
            f'{row["hard_bad_coverage"]:.1%}',
            ha="center",
            fontsize=11,
        )
    figure.savefig(OUTPUT_DIR / "exact_heavy_atom_pairing_coverage.png", dpi=190)
    figure.savefig(OUTPUT_DIR / "exact_heavy_atom_pairing_coverage.svg")
    plt.close(figure)


def _plot_portability(matrix: pd.DataFrame) -> None:
    """Plot audited portability decision counts."""
    order = ["borrow", "borrow_with_fix", "borrow_with_adapter", "do_not_borrow", "separate_experiment"]
    labels = ["Borrow", "Borrow + fix", "Borrow + adapter", "Do not borrow", "Separate experiment"]
    colors = ["#2F6B9A", "#4F8A70", "#7A78B8", "#C96B3C", "#8A6D3B"]
    counts = matrix["portability"].value_counts()
    values = [int(counts.get(item, 0)) for item in order]
    figure, axis = plt.subplots(figsize=(11.5, 6.5), constrained_layout=True)
    bars = axis.bar(labels, values, color=colors)
    axis.set_ylabel("Audited components", fontsize=14)
    axis.set_title("DecompDPO components are not directly portable as one block", fontsize=18)
    axis.tick_params(axis="x", labelrotation=15, labelsize=11)
    axis.tick_params(axis="y", labelsize=12)
    axis.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.06, str(value), ha="center", fontsize=12)
    axis.set_ylim(0, max(values) + 0.8)
    figure.savefig(OUTPUT_DIR / "decompdpo_component_portability.png", dpi=190)
    figure.savefig(OUTPUT_DIR / "decompdpo_component_portability.svg")
    plt.close(figure)


def main() -> None:
    """Write CSV, JSON, and presentation-readable plots for the audit."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(
        EXTREME_MANIFEST,
        usecols=["split", "extreme_label", "heavy_atom_count"],
    )
    coverage = _pairing_coverage(frame)
    matrix = _adaptation_matrix()
    coverage.to_csv(OUTPUT_DIR / "pairing_coverage.csv", index=False)
    matrix.to_csv(OUTPUT_DIR / "adaptation_matrix.csv", index=False)
    _plot_pairing_coverage(coverage)
    _plot_portability(matrix)
    summary = {
        "reference_project": "/home/md10/data/disk6/DiffB110/reference project/代码/DecompDPO",
        "reference_commit": "cb109f7",
        "diffsbdd_checkpoint_training_steps": 500,
        "collected_rollout_reverse_steps": 1000,
        "collected_rollout_actions_including_final_decode": 1001,
        "checkpoint_loss_type": "l2",
        "checkpoint_normalize_factors": [1, 4],
        "terminal_payload_fields_for_native_x0": ["x", "categorical", "nodesxsample"],
        "recommended_first_method": "DiffSBDD-native endpoint denoising DPO",
        "not_recommended_as_first_method": "1001-step recorded-trajectory log-prob DPO",
        "pairing_coverage": coverage.to_dict(orient="records"),
    }
    (OUTPUT_DIR / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
