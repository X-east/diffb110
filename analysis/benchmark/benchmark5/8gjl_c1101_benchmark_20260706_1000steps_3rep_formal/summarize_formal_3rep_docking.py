from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean, median, stdev


MODEL_ORDER = ["DiffGui", "DiffSBDD", "PMDM", "GCDM-SBDD", "TargetDiff"]
REPEAT_ORDER = ["rep1", "rep2", "rep3"]


def _slug(model: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in model).strip("_")


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return ordered[int(pos)]
    return ordered[low] * (high - pos) + ordered[high] * (pos - low)


def _mean_std(values: list[float]) -> tuple[float, float]:
    clean_values = [value for value in values if not math.isnan(value)]
    if not clean_values:
        return math.nan, math.nan
    return mean(clean_values), stdev(clean_values) if len(clean_values) > 1 else 0.0


def _read_scores(csv_path: Path) -> tuple[list[float], int, int]:
    scores: list[float] = []
    total = 0
    success = 0
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            total += 1
            if str(row.get("success", "")).lower() not in {"true", "1"}:
                continue
            success += 1
            value = row.get("score", "")
            if value not in {"", None}:
                scores.append(float(value))
    return scores, total, success


def _summarize_scores(repeat: str, model: str, csv_path: Path) -> dict[str, object]:
    scores, total, success = _read_scores(csv_path)
    ordered = sorted(scores)
    best_1_count = max(1, int(0.01 * len(scores))) if scores else 0
    best_5_count = max(1, int(0.05 * len(scores))) if scores else 0
    return {
        "repeat": repeat,
        "model": model,
        "docking_csv": str(csv_path),
        "total_ligands": total,
        "success_count": success,
        "score_count": len(scores),
        "success_rate": f"{success / total:.6f}" if total else "0.000000",
        "score_mean": f"{mean(scores):.6f}" if scores else "",
        "score_median": f"{median(scores):.6f}" if scores else "",
        "score_best_min": f"{min(scores):.6f}" if scores else "",
        "score_worst_max": f"{max(scores):.6f}" if scores else "",
        "score_p05": f"{_quantile(scores, 0.05):.6f}" if scores else "",
        "score_p25": f"{_quantile(scores, 0.25):.6f}" if scores else "",
        "score_p75": f"{_quantile(scores, 0.75):.6f}" if scores else "",
        "score_p95": f"{_quantile(scores, 0.95):.6f}" if scores else "",
        "best_1pct_mean": f"{mean(ordered[:best_1_count]):.6f}" if scores else "",
        "best_5pct_mean": f"{mean(ordered[:best_5_count]):.6f}" if scores else "",
        "rate_le_-8": f"{sum(score <= -8.0 for score in scores) / len(scores):.6f}" if scores else "0.000000",
        "rate_le_-10": f"{sum(score <= -10.0 for score in scores) / len(scores):.6f}" if scores else "0.000000",
        "rate_le_-12": f"{sum(score <= -12.0 for score in scores) / len(scores):.6f}" if scores else "0.000000",
        "positive_score_count": sum(score > 0.0 for score in scores),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["model"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _float(row: dict[str, object], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in {"", None} else math.nan


def build_repeat_rows(exp: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    missing: list[Path] = []
    for repeat in REPEAT_ORDER:
        for model in MODEL_ORDER:
            csv_path = exp / "docking" / repeat / _slug(model) / "docking_results.csv"
            if not csv_path.exists():
                missing.append(csv_path)
                continue
            rows.append(_summarize_scores(repeat, model, csv_path))
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing docking result CSVs:\n{missing_text}")
    return rows


def build_model_rows(repeat_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    metric_keys = [
        "success_rate",
        "score_mean",
        "score_median",
        "score_best_min",
        "score_p05",
        "best_1pct_mean",
        "best_5pct_mean",
        "rate_le_-8",
        "rate_le_-10",
        "rate_le_-12",
    ]
    rows: list[dict[str, object]] = []
    for model in MODEL_ORDER:
        model_rows = [row for row in repeat_rows if row["model"] == model]
        out: dict[str, object] = {
            "model": model,
            "repeats": len(model_rows),
            "total_ligands_mean": f"{mean(int(row['total_ligands']) for row in model_rows):.3f}",
            "success_count_mean": f"{mean(int(row['success_count']) for row in model_rows):.3f}",
        }
        for key in metric_keys:
            values = [_float(row, key) for row in model_rows]
            metric_mean, metric_std = _mean_std(values)
            out[f"{key}_mean"] = f"{metric_mean:.6f}" if not math.isnan(metric_mean) else ""
            out[f"{key}_std"] = f"{metric_std:.6f}" if not math.isnan(metric_std) else ""
        rows.append(out)
    return rows


def write_plots(exp: Path, model_rows: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
        }
    )

    colors = {
        "DiffGui": "#4C78A8",
        "DiffSBDD": "#F58518",
        "PMDM": "#54A24B",
        "GCDM-SBDD": "#E45756",
        "TargetDiff": "#72B7B2",
    }
    plot_specs = [
        ("score_median_mean", "score_median_std", "Docking score median", "docking_score_median_3rep.png"),
        ("score_mean_mean", "score_mean_std", "Docking score mean", "docking_score_mean_3rep.png"),
        ("best_5pct_mean_mean", "best_5pct_mean_std", "Best 5% mean docking score", "docking_best5_mean_3rep.png"),
        ("rate_le_-10_mean", "rate_le_-10_std", "Fraction score <= -10", "docking_rate_le_-10_3rep.png"),
    ]
    for value_key, err_key, ylabel, filename in plot_specs:
        fig, ax = plt.subplots(figsize=(9.5, 5.8))
        values = [float(row[value_key]) if row[value_key] != "" else math.nan for row in model_rows]
        errs = [float(row[err_key]) if row[err_key] != "" else 0.0 for row in model_rows]
        ax.bar(range(len(MODEL_ORDER)), values, yerr=errs, capsize=4, color=[colors[m] for m in MODEL_ORDER])
        ax.set_xticks(range(len(MODEL_ORDER)))
        ax.set_xticklabels(MODEL_ORDER, rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(exp / filename, dpi=200)
        plt.close(fig)


def write_report(exp: Path, repeat_rows: list[dict[str, object]], model_rows: list[dict[str, object]]) -> None:
    summary_headers = [
        "model",
        "success_count_mean",
        "score_mean_mean",
        "score_median_mean",
        "best_5pct_mean_mean",
        "rate_le_-10_mean",
    ]
    repeat_headers = ["repeat", "model", "total_ligands", "success_count", "score_mean", "score_median", "best_5pct_mean", "rate_le_-10"]
    lines = [
        "# 8GJL/C1101 1000-step 3-repeat docking summary",
        "",
        "Docking was run on the sanitize-valid SDF for each model and repeat using the existing UniDock fast protocol.",
        "",
        "![median](docking_score_median_3rep.png)",
        "",
        "![best5](docking_best5_mean_3rep.png)",
        "",
        "## Mean over repeats",
        "",
        "| " + " | ".join(summary_headers) + " |",
        "| " + " | ".join(["---"] * len(summary_headers)) + " |",
    ]
    for row in model_rows:
        lines.append("| " + " | ".join(str(row[key]) for key in summary_headers) + " |")
    lines.extend(
        [
            "",
            "## Per-repeat docking",
            "",
            "| " + " | ".join(repeat_headers) + " |",
            "| " + " | ".join(["---"] * len(repeat_headers)) + " |",
        ]
    )
    for row in repeat_rows:
        lines.append("| " + " | ".join(str(row[key]) for key in repeat_headers) + " |")
    (exp / "docking_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=Path, required=True)
    args = parser.parse_args()
    exp = args.exp.resolve()
    repeat_rows = build_repeat_rows(exp)
    model_rows = build_model_rows(repeat_rows)
    _write_csv(exp / "formal_3rep_docking_per_repeat.csv", repeat_rows)
    _write_csv(exp / "formal_3rep_docking_model_summary.csv", model_rows)
    write_plots(exp, model_rows)
    write_report(exp, repeat_rows, model_rows)
    print(f"wrote docking summary under {exp}")


if __name__ == "__main__":
    main()
