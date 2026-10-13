"""Aggregate the three scaffold-held-out geometric-critic probe repeats."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTPUT_DIR = Path("analysis/offline_iql_geometric_critic_test_diagnosis_260821")
RUNS = {
    1: Path("analysis/offline_iql_geometric_critic_state_probe_16_200_260821"),
    2: Path("analysis/offline_iql_geometric_critic_state_probe_16_200_seed2_260821"),
    3: Path("analysis/offline_iql_geometric_critic_state_probe_16_200_seed3_260821"),
}
PAIRS = (
    ("good", "mid", "pairwise_good_gt_mid"),
    ("mid", "bad", "pairwise_mid_gt_bad"),
    ("good", "bad", "pairwise_good_gt_bad"),
)


def main() -> None:
    """Write per-seed best-validation and held-out summaries and a figure."""
    rows: list[dict[str, object]] = []
    for seed, run_dir in RUNS.items():
        manifest = json.loads((run_dir / "probe_manifest.json").read_text(encoding="utf-8"))
        metrics = pd.read_csv(run_dir / "probe_metrics.csv")
        best_update = int(manifest["best_update"])
        for split in ("train", "validation", "test"):
            subset = metrics[(metrics["update"] == best_update) & (metrics["split"] == split)]
            for left, right, column in PAIRS:
                rows.append(
                    {
                        "seed": seed,
                        "best_update": best_update,
                        "split": split,
                        "time_scope": "all_selected_times_mean",
                        "pair": f"{left}>{right}",
                        "pairwise": float(subset[column].mean()),
                    }
                )
                terminal = subset[subset["decision_index"] == 1000]
                rows.append(
                    {
                        "seed": seed,
                        "best_update": best_update,
                        "split": split,
                        "time_scope": "terminal_1000",
                        "pair": f"{left}>{right}",
                        "pairwise": float(terminal[column].iloc[0]),
                    }
                )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_DIR / "multi_seed_probe_summary.csv", index=False)

    test_terminal = summary[
        (summary["split"] == "test") & (summary["time_scope"] == "terminal_1000")
    ]
    (OUTPUT_DIR / "multi_seed_probe_summary.json").write_text(
        json.dumps(
            {
                "seeds": sorted(RUNS),
                "terminal_test_rows": test_terminal.to_dict(orient="records"),
                "terminal_test_mean_by_pair": test_terminal.groupby("pair")["pairwise"].mean().to_dict(),
                "terminal_test_std_by_pair": test_terminal.groupby("pair")["pairwise"].std(ddof=1).to_dict(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    figure, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    colors = {"good>mid": "#2F6B9A", "mid>bad": "#D99B29", "good>bad": "#C96B3C"}
    for axis, time_scope, title in zip(
        axes,
        ("terminal_1000", "all_selected_times_mean"),
        ("Test at terminal decision", "Test mean across selected decisions"),
    ):
        subset = summary[
            (summary["split"] == "test") & (summary["time_scope"] == time_scope)
        ]
        positions = np.arange(len(RUNS))
        width = 0.24
        for index, pair in enumerate(colors):
            values = [
                float(subset[(subset["seed"] == seed) & (subset["pair"] == pair)]["pairwise"].iloc[0])
                for seed in sorted(RUNS)
            ]
            axis.bar(
                positions + (index - 1) * width,
                values,
                width=width,
                color=colors[pair],
                label=pair,
            )
        axis.axhline(0.5, color="#555555", linestyle="--", linewidth=1.6, label="random = 0.5")
        axis.set_xticks(positions, [f"seed {seed}" for seed in sorted(RUNS)])
        axis.set_ylim(0.2, 0.82)
        axis.set_ylabel("Pairwise ordering probability", fontsize=13)
        axis.set_title(title, fontsize=16)
        axis.grid(axis="y", alpha=0.22)
        axis.tick_params(labelsize=11)
    axes[0].legend(fontsize=10, ncol=2)
    figure.suptitle("Scaffold-held-out probe repeats: the good–mid gap remains unresolved", fontsize=19)
    figure.savefig(OUTPUT_DIR / "multi_seed_probe_stability.png", dpi=190)
    figure.savefig(OUTPUT_DIR / "multi_seed_probe_stability.svg")
    plt.close(figure)


if __name__ == "__main__":
    main()
