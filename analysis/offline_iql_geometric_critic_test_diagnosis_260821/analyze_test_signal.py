"""Diagnose why the scaffold-held-out geometric critic misses test labels.

This analysis compares the existing geometric-critic probe with the terminal
label-generating descriptors.  It does not train an actor or alter the frozen
trajectory data.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier


MANIFEST = Path("analysis/offline_iql_scaffold_split_260821/scaffold_split_manifest.csv.gz")
FACTS = Path(
    "analysis/offline_iql_100k_reasonableness_schemes_260819/"
    "expanded_reasonableness_facts.csv.gz"
)
PROBE_METRICS = Path(
    "analysis/offline_iql_geometric_critic_state_probe_16_200_260821/probe_metrics.csv"
)
OUTPUT_DIR = Path("analysis/offline_iql_geometric_critic_test_diagnosis_260821")
LABELS = ("good", "mid", "bad")
PAIRS = (("good", "mid"), ("mid", "bad"), ("good", "bad"))
FEATURES = (
    "umap_nearest_chembl_distance",
    "heavy_atom_count",
    "hetero_atom_count",
    "carbon_fraction",
    "qed",
    "sa_raw",
    "logp",
    "tpsa",
    "hba",
    "hbd",
    "mol_wt",
    "total_ring_count",
    "aromatic_ring_count",
    "fused_ring_count",
    "complex_ring_count",
    "brenk_count",
    "fraction_csp3",
    "aromatic_atom_fraction",
    "rotatable_bond_count",
    "three_member_ring_count",
)


def pairwise_ordering(frame: pd.DataFrame, score_column: str) -> dict[str, float]:
    """Return strict pairwise ordering probabilities for the three labels."""
    result: dict[str, float] = {}
    for left, right in PAIRS:
        left_values = frame.loc[frame["quality_label"] == left, score_column].to_numpy()
        right_values = frame.loc[frame["quality_label"] == right, score_column].to_numpy()
        if len(left_values) == 0 or len(right_values) == 0:
            result[f"{left}_gt_{right}"] = float("nan")
            continue
        # Count strict wins with a sorted right-hand sample.  This is the
        # Mann--Whitney pairwise probability without materializing an
        # O(n_left * n_right) matrix for the full 67k-row manifest.
        right_sorted = np.sort(right_values)
        strict_wins = np.searchsorted(right_sorted, left_values, side="left")
        result[f"{left}_gt_{right}"] = float(strict_wins.mean() / len(right_sorted))
    result["pairwise_mean"] = float(np.mean(list(result.values())))
    return result


def choose_train_direction(frame: pd.DataFrame, feature: str) -> tuple[float, float]:
    """Choose a feature sign on train only and return sign and train score."""
    candidates = frame.loc[frame["split"] == "train"].copy()
    candidates["feature_value"] = pd.to_numeric(candidates[feature], errors="coerce")
    candidates = candidates[np.isfinite(candidates["feature_value"])]
    scores: list[tuple[float, float]] = []
    for sign in (-1.0, 1.0):
        candidates["score"] = sign * candidates["feature_value"]
        ordering = pairwise_ordering(candidates, "score")
        scores.append((ordering["pairwise_mean"], sign))
    best_score, best_sign = max(scores, key=lambda item: item[0])
    return best_sign, best_score


def parse_bool(value: object) -> bool:
    """Parse booleans from pandas bools, integers, or CSV strings."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def load_frame() -> pd.DataFrame:
    """Load the scaffold split and terminal descriptor facts."""
    manifest = pd.read_csv(MANIFEST, compression="gzip")
    # The frozen split already owns the distance and partition columns.  Read
    # the remaining descriptor facts from the source table to avoid pandas
    # suffixing the controlling distance column during the join.
    facts_columns = [
        "trajectory_id",
        *[feature for feature in FEATURES if feature != "umap_nearest_chembl_distance"],
    ]
    facts = pd.read_csv(FACTS, compression="gzip", usecols=facts_columns)
    frame = manifest.merge(facts, on="trajectory_id", how="inner", validate="one_to_one")
    if len(frame) != len(manifest):
        raise RuntimeError("descriptor join changed the scaffold manifest row count")
    frame["structural_hard_bad"] = frame["structural_hard_bad"].map(parse_bool)
    frame["quality_label"] = frame["quality_label"].astype(str)
    if set(frame["quality_label"].unique()) != set(LABELS):
        raise RuntimeError("unexpected quality labels in scaffold manifest")
    for feature in FEATURES:
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
    return frame


def build_summaries(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Build label summaries and train-direction feature baselines."""
    summary_rows: list[dict[str, object]] = []
    for split in ("train", "validation", "test"):
        for label in LABELS:
            subset = frame[(frame["split"] == split) & (frame["quality_label"] == label)]
            row: dict[str, object] = {
                "split": split,
                "quality_label": label,
                "rows": len(subset),
                "hard_bad_rate": float(subset["structural_hard_bad"].mean()),
            }
            for feature in FEATURES:
                values = subset[feature].dropna()
                row[f"{feature}_mean"] = float(values.mean())
                row[f"{feature}_median"] = float(values.median())
            summary_rows.append(row)
    label_summary = pd.DataFrame(summary_rows)

    baseline_rows: list[dict[str, object]] = []
    for feature in FEATURES:
        sign, train_mean = choose_train_direction(frame, feature)
        for split in ("train", "validation", "test"):
            subset = frame.loc[frame["split"] == split].copy()
            subset["score"] = sign * subset[feature]
            ordering = pairwise_ordering(subset, "score")
            baseline_rows.append(
                {
                    "baseline": feature,
                    "direction_selected_on": "train",
                    "sign": sign,
                    "split": split,
                    "train_pairwise_mean": train_mean,
                    **ordering,
                }
            )

    # A transparent baseline using the two fields that define the composite
    # label: distance and the structural-hard-bad flag.  The penalty is tuned
    # on train only; test is never used to choose it.
    distance = "umap_nearest_chembl_distance"
    train = frame.loc[frame["split"] == "train"].copy()
    penalty_grid = np.linspace(0.0, 0.20, 401)
    scored_penalties: list[tuple[float, float]] = []
    for penalty in penalty_grid:
        train["score"] = -train[distance] - penalty * train["structural_hard_bad"].astype(float)
        scored_penalties.append((pairwise_ordering(train, "score")["pairwise_mean"], penalty))
    best_train_mean, best_penalty = max(scored_penalties, key=lambda item: item[0])
    for split in ("train", "validation", "test"):
        subset = frame.loc[frame["split"] == split].copy()
        subset["score"] = -subset[distance] - best_penalty * subset["structural_hard_bad"].astype(float)
        ordering = pairwise_ordering(subset, "score")
        baseline_rows.append(
            {
                "baseline": "distance_plus_structural_hard_bad",
                "direction_selected_on": "train",
                "sign": -1.0,
                "hard_bad_penalty": best_penalty,
                "split": split,
                "train_pairwise_mean": best_train_mean,
                **ordering,
            }
        )
    baseline_summary = pd.DataFrame(baseline_rows)

    # A shallow tree shows how readily the labels can be recovered from the
    # actual label-generating fields. It is a diagnostic baseline, not a
    # recommendation for the IQL critic.
    tree_features = [distance, "structural_hard_bad"]
    train_x = frame.loc[frame["split"] == "train", tree_features].astype(float)
    train_y = frame.loc[frame["split"] == "train", "quality_label"]
    tree = DecisionTreeClassifier(max_depth=4, random_state=1)
    tree.fit(train_x, train_y)
    tree_rows: list[dict[str, object]] = []
    for split in ("train", "validation", "test"):
        subset = frame.loc[frame["split"] == split]
        predicted = tree.predict(subset[tree_features].astype(float))
        probabilities = tree.predict_proba(subset[tree_features].astype(float))
        class_to_column = {label: index for index, label in enumerate(tree.classes_)}
        expected_score = probabilities[:, class_to_column["good"]] - probabilities[:, class_to_column["bad"]]
        scored = subset[["quality_label"]].copy()
        scored["score"] = expected_score
        tree_ordering = pairwise_ordering(scored, "score")
        tree_rows.append(
            {
                "baseline": "depth4_tree_distance_plus_hard_bad",
                "split": split,
                "accuracy": float(accuracy_score(subset["quality_label"], predicted)),
                "macro_f1": float(f1_score(subset["quality_label"], predicted, average="macro")),
                **tree_ordering,
            }
        )
    tree_summary = pd.DataFrame(tree_rows)

    metrics: dict[str, object] = {
        "rows": int(len(frame)),
        "split_counts": frame["split"].value_counts().to_dict(),
        "label_counts": frame["quality_label"].value_counts().to_dict(),
        "hard_bad_by_split": {
            split: int(frame.loc[frame["split"] == split, "structural_hard_bad"].sum())
            for split in ("train", "validation", "test")
        },
        "best_distance_plus_hard_bad_penalty": float(best_penalty),
        "tree_metrics": tree_summary.to_dict(orient="records"),
    }
    return label_summary, baseline_summary, {"overview": metrics}


def write_probe_comparison(frame: pd.DataFrame, baseline_summary: pd.DataFrame) -> pd.DataFrame:
    """Combine test-time geometric-critic metrics with descriptor baselines."""
    probe = pd.read_csv(PROBE_METRICS)
    best_update = 190
    model = probe[(probe["update"] == best_update) & (probe["split"] == "test")].copy()
    model_rows: list[dict[str, object]] = []
    for row in model.itertuples(index=False):
        for pair in PAIRS:
            column = f"pairwise_{pair[0]}_gt_{pair[1]}"
            model_rows.append(
                {
                    "source": "geometric_critic",
                    "baseline": "current geometric critic",
                    "decision_index": int(row.decision_index),
                    "pair": f"{pair[0]}>{pair[1]}",
                    "pairwise": float(getattr(row, column)),
                }
            )
    terminal_baseline_rows: list[dict[str, object]] = []
    for baseline in ("umap_nearest_chembl_distance", "distance_plus_structural_hard_bad", "heavy_atom_count"):
        subset = baseline_summary[
            (baseline_summary["baseline"] == baseline)
            & (baseline_summary["split"] == "test")
        ]
        for row in subset.itertuples(index=False):
            for pair in PAIRS:
                pair_name = f"{pair[0]}_gt_{pair[1]}"
                terminal_baseline_rows.append(
                    {
                        "source": "terminal_descriptor",
                        "baseline": str(baseline),
                        "decision_index": 1000,
                        "pair": f"{pair[0]}>{pair[1]}",
                        "pairwise": float(getattr(row, pair_name)),
                    }
                )
    return pd.DataFrame(model_rows + terminal_baseline_rows)


def make_figure(frame: pd.DataFrame, comparison: pd.DataFrame) -> None:
    """Export a readable four-panel diagnostic figure."""
    colors = {"good": "#2F6B9A", "mid": "#D99B29", "bad": "#C96B3C"}
    figure, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)

    test = frame.loc[frame["split"] == "test"]
    distance_values = [
        test.loc[test["quality_label"] == label, "umap_nearest_chembl_distance"].to_numpy()
        for label in LABELS
    ]
    box = axes[0, 0].boxplot(
        distance_values,
        labels=["good", "mid", "bad"],
        patch_artist=True,
        showfliers=False,
    )
    for patch, label in zip(box["boxes"], LABELS):
        patch.set_facecolor(colors[label])
        patch.set_alpha(0.72)
    axes[0, 0].axhline(0.01, color="#555555", linestyle="--", linewidth=1.8, label="0.01 threshold")
    axes[0, 0].axhline(0.03, color="#777777", linestyle=":", linewidth=2.0, label="0.03 threshold")
    axes[0, 0].set_title("Test: label-generating ChEMBL distance", fontsize=16)
    axes[0, 0].set_ylabel("Nearest ChEMBL UMAP distance", fontsize=13)
    axes[0, 0].legend(fontsize=10)

    hard_rates = test.groupby("quality_label")["structural_hard_bad"].mean().reindex(LABELS)
    bars = axes[0, 1].bar(LABELS, hard_rates.to_numpy(), color=[colors[label] for label in LABELS], alpha=0.82)
    axes[0, 1].set_ylim(0, max(0.4, float(hard_rates.max()) * 1.35))
    axes[0, 1].set_title("Test: structural-hard-bad mixture", fontsize=16)
    axes[0, 1].set_ylabel("Fraction marked structural hard bad", fontsize=13)
    for bar, value in zip(bars, hard_rates.to_numpy()):
        axes[0, 1].text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.1%}", ha="center", fontsize=12)

    model = comparison[comparison["source"] == "geometric_critic"].copy()
    for pair, color in zip(("good>mid", "mid>bad", "good>bad"), ("#2F6B9A", "#D99B29", "#C96B3C")):
        subset = model[model["pair"] == pair].sort_values("decision_index")
        axes[1, 0].plot(subset["decision_index"], subset["pairwise"], marker="o", linewidth=2.3, color=color, label=pair)
    axes[1, 0].axhline(0.5, color="#555555", linestyle="--", linewidth=1.6, label="random = 0.5")
    axes[1, 0].set_title("Geometric critic: scaffold-held-out test", fontsize=16)
    axes[1, 0].set_xlabel("Decision index", fontsize=13)
    axes[1, 0].set_ylabel("Pairwise ordering probability", fontsize=13)
    axes[1, 0].set_ylim(0.2, 0.82)
    axes[1, 0].legend(fontsize=10, ncol=2)

    terminal = comparison[comparison["decision_index"] == 1000].copy()
    order = ["umap_nearest_chembl_distance", "distance_plus_structural_hard_bad", "heavy_atom_count", "current geometric critic"]
    labels = ["-UMAP distance", "-distance - hard-bad", "node-count proxy", "critic @1000"]
    x = np.arange(len(order))
    width = 0.24
    pair_colors = {"good>mid": "#2F6B9A", "mid>bad": "#D99B29", "good>bad": "#C96B3C"}
    for index, pair in enumerate(("good>mid", "mid>bad", "good>bad")):
        values: list[float] = []
        for baseline in order:
            row = terminal[(terminal["baseline"] == baseline) & (terminal["pair"] == pair)]
            values.append(float(row["pairwise"].iloc[0]))
        axes[1, 1].bar(x + (index - 1) * width, values, width=width, color=pair_colors[pair], label=pair)
    axes[1, 1].axhline(0.5, color="#555555", linestyle="--", linewidth=1.6)
    axes[1, 1].set_xticks(x, labels, rotation=18, ha="right")
    axes[1, 1].set_ylim(0.2, 1.02)
    axes[1, 1].set_title("Terminal test: raw-label signal vs critic", fontsize=16)
    axes[1, 1].set_ylabel("Pairwise ordering probability", fontsize=13)
    axes[1, 1].legend(fontsize=10)

    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.22)
        axis.tick_params(labelsize=11)
    figure.suptitle("Why the geometric critic does not generalize across scaffolds", fontsize=20)
    figure.savefig(OUTPUT_DIR / "test_signal_vs_geometric_critic.png", dpi=190)
    figure.savefig(OUTPUT_DIR / "test_signal_vs_geometric_critic.svg")
    plt.close(figure)


def main() -> None:
    """Run the test-signal diagnosis and write tables, JSON, and plots."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_frame()
    label_summary, baseline_summary, metadata = build_summaries(frame)
    comparison = write_probe_comparison(frame, baseline_summary)
    label_summary.to_csv(OUTPUT_DIR / "label_descriptor_summary.csv", index=False)
    baseline_summary.to_csv(OUTPUT_DIR / "pairwise_descriptor_baselines.csv", index=False)
    comparison.to_csv(OUTPUT_DIR / "critic_vs_baselines.csv", index=False)
    make_figure(frame, comparison)
    (OUTPUT_DIR / "diagnosis_summary.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
