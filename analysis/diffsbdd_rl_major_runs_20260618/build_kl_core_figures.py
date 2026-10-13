from __future__ import annotations

"""Build focused figures for the four core QED/SA/docking KL controls."""

import csv
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/diffb110_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = ROOT / "analysis" / "diffsbdd_rl_major_runs_20260618"

CORE_RUNS = [
    ("KL-01", "OUT/qed-sa-docking-8gjl-8gpu-kl-0_20260511_153246"),
    ("KL-02", "OUT/qed-sa-docking-8gjl-8gpu-kl-0p01_20260511_162333"),
    ("KL-03", "OUT/qed-sa-docking-8gjl-8gpu-kl-0p02_20260511_171944"),
    ("KL-04", "OUT/qed-sa-docking-8gjl-8gpu-kl-0p05_20260511_181521"),
]

COLORS = {
    "KL-01": "#2f6f9f",
    "KL-02": "#3b8f5a",
    "KL-03": "#c28f2c",
    "KL-04": "#b05c46",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data if isinstance(data, dict) else {}


def _float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _composite_norm(row: dict[str, str]) -> float | None:
    qed = _float(row, "avg_qed_norm")
    sa = _float(row, "avg_sa_norm")
    docking = _float(row, "avg_docking_norm")
    if qed is None or sa is None or docking is None:
        return None
    return 0.25 * qed + 0.25 * sa + 0.5 * docking


def _format_float(value: float | None, digits: int = 6) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def collect_epoch_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for alias, rel_dir in CORE_RUNS:
        run_dir = ROOT / rel_dir
        config = _read_yaml(run_dir / "config.yaml")
        train_cfg = config.get("train", {}) if isinstance(config.get("train"), dict) else {}
        kl_coef = float(train_cfg.get("kl_coef", 0.0))
        for row in _read_csv(run_dir / "training_summary.csv"):
            epoch = int(float(row["epoch"]))
            rows.append(
                {
                    "alias": alias,
                    "run_dir": rel_dir,
                    "kl_coef": kl_coef,
                    "epoch": epoch,
                    "rdkit_valid_ratio": _float(row, "rdkit_valid_ratio"),
                    "avg_qed": _float(row, "avg_qed"),
                    "avg_sa": _float(row, "avg_sa"),
                    "avg_docking": _float(row, "avg_docking"),
                    "avg_qed_norm": _float(row, "avg_qed_norm"),
                    "avg_sa_norm": _float(row, "avg_sa_norm"),
                    "avg_docking_norm": _float(row, "avg_docking_norm"),
                    "avg_kl": _float(row, "avg_kl"),
                    "avg_ppo_loss": _float(row, "avg_ppo_loss"),
                    "composite_norm_proxy": _composite_norm(row),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_summary_rows(epoch_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    for alias, rel_dir in CORE_RUNS:
        rows = [row for row in epoch_rows if row["alias"] == alias]
        rows.sort(key=lambda row: row["epoch"])
        final = rows[-1]
        best_composite = max(rows, key=lambda row: row["composite_norm_proxy"] if row["composite_norm_proxy"] is not None else -1e9)
        best_valid = max(rows, key=lambda row: row["rdkit_valid_ratio"] if row["rdkit_valid_ratio"] is not None else -1e9)
        best_docking = min(rows, key=lambda row: row["avg_docking"] if row["avg_docking"] is not None else 1e9)
        summary_rows.append(
            {
                "alias": alias,
                "run_dir": rel_dir,
                "kl_coef": final["kl_coef"],
                "epochs_recorded": len(rows),
                "final_epoch": final["epoch"],
                "final_valid": final["rdkit_valid_ratio"],
                "final_qed": final["avg_qed"],
                "final_sa": final["avg_sa"],
                "final_docking": final["avg_docking"],
                "final_composite_norm_proxy": final["composite_norm_proxy"],
                "best_composite_epoch": best_composite["epoch"],
                "best_composite_norm_proxy": best_composite["composite_norm_proxy"],
                "valid_at_best_composite": best_composite["rdkit_valid_ratio"],
                "docking_at_best_composite": best_composite["avg_docking"],
                "best_valid_epoch": best_valid["epoch"],
                "best_valid": best_valid["rdkit_valid_ratio"],
                "best_docking_epoch": best_docking["epoch"],
                "best_docking": best_docking["avg_docking"],
            }
        )
    return summary_rows


def _group_rows(epoch_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in epoch_rows:
        grouped.setdefault(str(row["alias"]), []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: row["epoch"])
    return grouped


def plot_training_curves(epoch_rows: list[dict[str, Any]]) -> None:
    grouped = _group_rows(epoch_rows)
    panels = [
        ("composite_norm_proxy", "Composite norm proxy", "higher is better; axis clipped"),
        ("rdkit_valid_ratio", "RDKit valid ratio", "higher is better"),
        ("avg_docking", "Average docking", "lower is better; axis clipped"),
        ("avg_qed", "Average QED", "higher is better"),
        ("avg_sa", "Average SA", "higher is better"),
        ("avg_kl", "Average KL", "diagnostic"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(11.5, 9.0), sharex=True)
    for ax, (metric, title, subtitle) in zip(axes.ravel(), panels):
        for alias, rows in grouped.items():
            xs = [row["epoch"] for row in rows]
            ys = [row[metric] for row in rows]
            ax.plot(xs, ys, label=f"{alias}  KL={rows[0]['kl_coef']:g}", color=COLORS[alias], linewidth=1.8)
        ax.set_title(f"{title}\n{subtitle}", fontsize=10)
        ax.grid(True, alpha=0.25)
        if metric == "composite_norm_proxy":
            ax.set_ylim(-0.3, 0.65)
        if metric == "avg_docking":
            ax.set_ylim(-9.0, -3.0)
    axes[-1, 0].set_xlabel("Epoch")
    axes[-1, 1].set_xlabel("Epoch")
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.suptitle("Core QED/SA/docking KL controls: training curves", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for suffix in ("png", "svg"):
        fig.savefig(ANALYSIS_DIR / f"kl_core_training_curves.{suffix}", dpi=200)
    plt.close(fig)


def plot_final_heatmap(summary_rows: list[dict[str, Any]]) -> None:
    metrics = [
        ("final_valid", "final valid"),
        ("final_qed", "final QED"),
        ("final_sa", "final SA"),
        ("final_docking", "final docking"),
        ("final_composite_norm_proxy", "final comp."),
        ("best_composite_norm_proxy", "best comp."),
    ]
    aliases = [str(row["alias"]) for row in summary_rows]
    values = np.array([[float(row[key]) for key, _ in metrics] for row in summary_rows], dtype=float)
    normalized = values.copy()
    for index, (key, _) in enumerate(metrics):
        column = values[:, index]
        if key == "final_docking":
            column = -column
        min_value = float(np.min(column))
        max_value = float(np.max(column))
        if abs(max_value - min_value) < 1e-12:
            normalized[:, index] = 0.5
        else:
            normalized[:, index] = (column - min_value) / (max_value - min_value)
    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    im = ax.imshow(normalized, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([label for _, label in metrics], rotation=30, ha="right")
    ax.set_yticks(range(len(aliases)))
    ax.set_yticklabels([f"{row['alias']}  KL={row['kl_coef']:g}" for row in summary_rows])
    for row_idx, row in enumerate(summary_rows):
        for col_idx, (key, _) in enumerate(metrics):
            value = float(row[key])
            text = f"{value:.3f}"
            ax.text(col_idx, row_idx, text, ha="center", va="center", fontsize=8, color="white" if normalized[row_idx, col_idx] < 0.45 else "black")
    ax.set_title("Core KL controls: normalized final/best metric heatmap")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("relative score (higher is better)")
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(ANALYSIS_DIR / f"kl_core_final_metric_heatmap.{suffix}", dpi=200)
    plt.close(fig)


def plot_final_vs_best(summary_rows: list[dict[str, Any]]) -> None:
    x = [float(row["kl_coef"]) for row in summary_rows]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8))

    axes[0].plot(x, [row["final_composite_norm_proxy"] for row in summary_rows], marker="o", label="final", color="#2f6f9f")
    axes[0].plot(x, [row["best_composite_norm_proxy"] for row in summary_rows], marker="s", label="best epoch", color="#b05c46")
    axes[0].set_title("Composite norm proxy")
    axes[0].set_ylabel("higher is better")
    axes[0].legend(fontsize=8)

    axes[1].plot(x, [row["final_valid"] for row in summary_rows], marker="o", label="final", color="#2f6f9f")
    axes[1].plot(x, [row["best_valid"] for row in summary_rows], marker="s", label="best valid", color="#b05c46")
    axes[1].set_title("RDKit valid ratio")
    axes[1].set_ylabel("higher is better")

    axes[2].plot(x, [row["final_docking"] for row in summary_rows], marker="o", label="final", color="#2f6f9f")
    axes[2].plot(x, [row["best_docking"] for row in summary_rows], marker="s", label="best docking", color="#b05c46")
    axes[2].set_title("Average docking")
    axes[2].set_ylabel("lower is better")

    for ax in axes:
        ax.set_xlabel("kl_coef")
        ax.set_xticks(x)
        ax.grid(True, alpha=0.25)
    fig.suptitle("Core KL controls: final checkpoint vs best observed epoch", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for suffix in ("png", "svg"):
        fig.savefig(ANALYSIS_DIR / f"kl_core_final_vs_best.{suffix}", dpi=200)
    plt.close(fig)


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    epoch_rows = collect_epoch_rows()
    summary_rows = build_summary_rows(epoch_rows)

    write_csv(ANALYSIS_DIR / "kl_core_epoch_metrics.csv", epoch_rows)
    write_csv(ANALYSIS_DIR / "kl_core_summary.csv", summary_rows)
    plot_training_curves(epoch_rows)
    plot_final_heatmap(summary_rows)
    plot_final_vs_best(summary_rows)

    print(f"Wrote {len(epoch_rows)} epoch rows and {len(summary_rows)} core KL summary rows.")
    for row in summary_rows:
        print(
            f"{row['alias']} KL={row['kl_coef']:g} "
            f"final_valid={_format_float(row['final_valid'], 3)} "
            f"final_docking={_format_float(row['final_docking'], 3)} "
            f"best_comp_epoch={row['best_composite_epoch']}"
        )


if __name__ == "__main__":
    main()
