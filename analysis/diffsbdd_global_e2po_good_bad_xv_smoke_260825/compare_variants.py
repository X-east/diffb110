"""Compare the completed x/v/native-joint Global E2PO smoke runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
VARIANTS = ("x_only", "v_only", "native_joint")
TIMESTEPS = (50, 125, 250, 375, 500)


def read_run(variant: str) -> tuple[dict[str, str], dict[str, object]]:
    """Read the one-row metrics file and the run manifest."""
    run_dir = ROOT / variant
    with (run_dir / "metrics" / "metrics.csv").open(newline="") as handle:
        metrics = next(csv.DictReader(handle))
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    return metrics, manifest


def build_summary() -> list[dict[str, str]]:
    """Build scalar and timestep summary rows."""
    rows: list[dict[str, str]] = []
    for variant in VARIANTS:
        metrics, manifest = read_run(variant)
        validation = [float(metrics[f"validation_pair_accuracy_t{t:04d}"]) for t in TIMESTEPS]
        test = [float(metrics[f"test_pair_accuracy_t{t:04d}"]) for t in TIMESTEPS]
        rows.append(
            {
                "variant": variant,
                "x_weight": str(manifest["x_weight"]),
                "v_weight": str(manifest["v_weight"]),
                "train_dpo_loss": metrics["train_dpo_loss"],
                "train_relative_margin": metrics["train_relative_margin"],
                "train_gradient_norm_mean": metrics["train_gradient_norm_mean"],
                "validation_pair_accuracy": metrics["validation_pair_accuracy"],
                "validation_mean_margin": metrics["validation_mean_margin"],
                "test_pair_accuracy": metrics["test_pair_accuracy"],
                "test_mean_margin": metrics["test_mean_margin"],
                "train_eval_pair_accuracy": metrics["train_eval_pair_accuracy"],
                "elapsed_seconds": metrics["elapsed_seconds"],
                "validation_timestep_min": f"{min(validation):.9f}",
                "validation_timestep_max": f"{max(validation):.9f}",
                "validation_timestep_std": f"{(sum((x - sum(validation) / len(validation)) ** 2 for x in validation) / len(validation)) ** 0.5:.9f}",
                "test_timestep_min": f"{min(test):.9f}",
                "test_timestep_max": f"{max(test):.9f}",
                "test_timestep_std": f"{(sum((x - sum(test) / len(test)) ** 2 for x in test) / len(test)) ** 0.5:.9f}",
            }
        )
    return rows


def write_csv(rows: list[dict[str, str]]) -> None:
    """Write scalar and timestep CSV evidence."""
    with (ROOT / "variant_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with (ROOT / "timestep_accuracy.csv").open("w", newline="") as handle:
        fields = ["variant", "split", "timestep", "pair_accuracy"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for variant in VARIANTS:
            metrics, _manifest = read_run(variant)
            for split in ("validation", "test"):
                for timestep in TIMESTEPS:
                    writer.writerow(
                        {
                            "variant": variant,
                            "split": split,
                            "timestep": timestep,
                            "pair_accuracy": metrics[f"{split}_pair_accuracy_t{timestep:04d}"],
                        }
                    )


def plot_scalar_metrics(rows: list[dict[str, str]]) -> None:
    """Plot validation/test accuracy and relative margin."""
    labels = [row["variant"] for row in rows]
    validation = [float(row["validation_pair_accuracy"]) for row in rows]
    test = [float(row["test_pair_accuracy"]) for row in rows]
    train_margin = [float(row["train_relative_margin"]) for row in rows]
    val_margin = [float(row["validation_mean_margin"]) for row in rows]
    positions = list(range(len(labels)))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    width = 0.34
    axes[0].bar([x - width / 2 for x in positions], validation, width, label="validation", color="#2166ac")
    axes[0].bar([x + width / 2 for x in positions], test, width, label="test", color="#67a9cf")
    axes[0].axhline(0.5, linestyle="--", linewidth=1.3, color="#555555")
    axes[0].set_xticks(positions, labels)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Pair accuracy")
    axes[0].set_title("Overall pair accuracy")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar([x - width / 2 for x in positions], train_margin, width, label="train relative", color="#b2182b")
    axes[1].bar([x + width / 2 for x in positions], val_margin, width, label="validation mean", color="#ef8a62")
    axes[1].axhline(0.0, linewidth=1.0, color="#555555")
    axes[1].set_xticks(positions, labels)
    axes[1].set_ylabel("Score margin")
    axes[1].set_title("Current-reference margin")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("DiffSBDD Global E2PO smoke comparison")
    fig.tight_layout()
    fig.savefig(ROOT / "variant_scalar_metrics.png", dpi=180)
    plt.close(fig)


def plot_timestep_accuracy() -> None:
    """Plot validation and test accuracy for every configured timestep."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    colors = {"x_only": "#d73027", "v_only": "#1a9850", "native_joint": "#4575b4"}
    for axis, split in zip(axes, ("validation", "test")):
        for variant in VARIANTS:
            metrics, _manifest = read_run(variant)
            values = [float(metrics[f"{split}_pair_accuracy_t{t:04d}"]) for t in TIMESTEPS]
            axis.plot(TIMESTEPS, values, marker="o", linewidth=2.3, label=variant, color=colors[variant])
        axis.axhline(0.5, linestyle="--", linewidth=1.3, color="#555555", label="random = 0.5")
        axis.set_title(f"{split.capitalize()} by timestep")
        axis.set_xlabel("Diffusion timestep t (grid = 500)")
        axis.set_xticks(TIMESTEPS)
        axis.set_ylim(0, 1.05)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Pair accuracy")
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle("Time-dependent preference fitting")
    fig.tight_layout()
    fig.savefig(ROOT / "timestep_accuracy_comparison.png", dpi=180)
    plt.close(fig)


def main() -> None:
    """Generate comparison evidence for all completed variants."""
    rows = build_summary()
    write_csv(rows)
    plot_scalar_metrics(rows)
    plot_timestep_accuracy()
    print(ROOT / "variant_summary.csv")
    print(ROOT / "timestep_accuracy.csv")
    print(ROOT / "variant_scalar_metrics.png")
    print(ROOT / "timestep_accuracy_comparison.png")


if __name__ == "__main__":
    main()
