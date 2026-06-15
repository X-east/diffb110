"""Summarize per-model docking CSV files into one table and simple plots."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median

import matplotlib.pyplot as plt


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


def _read_scores(path: Path) -> tuple[list[float], int]:
    scores: list[float] = []
    total = 0
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            total += 1
            if str(row.get("success", "")).lower() not in {"true", "1"}:
                continue
            raw = row.get("score", "")
            if raw in ("", None):
                continue
            scores.append(float(raw))
    return scores, total


def _summarize_one(model: str, csv_path: Path) -> dict[str, object]:
    scores, total = _read_scores(csv_path)
    best_1_count = max(1, int(0.01 * len(scores))) if scores else 0
    best_5_count = max(1, int(0.05 * len(scores))) if scores else 0
    sorted_scores = sorted(scores)
    row: dict[str, object] = {
        "model": model,
        "docking_csv": str(csv_path),
        "total_ligands": total,
        "success_count": len(scores),
        "success_rate": len(scores) / total if total else 0.0,
        "score_mean": mean(scores) if scores else None,
        "score_median": median(scores) if scores else None,
        "score_best_min": min(scores) if scores else None,
        "score_worst_max": max(scores) if scores else None,
        "score_p05": _quantile(scores, 0.05),
        "score_p25": _quantile(scores, 0.25),
        "score_p75": _quantile(scores, 0.75),
        "score_p95": _quantile(scores, 0.95),
        "best_1pct_mean": mean(sorted_scores[:best_1_count]) if scores else None,
        "best_5pct_mean": mean(sorted_scores[:best_5_count]) if scores else None,
        "rate_le_-8": sum(score <= -8.0 for score in scores) / len(scores) if scores else 0.0,
        "rate_le_-10": sum(score <= -10.0 for score in scores) / len(scores) if scores else 0.0,
        "rate_le_-12": sum(score <= -12.0 for score in scores) / len(scores) if scores else 0.0,
        "positive_score_count": sum(score > 0.0 for score in scores),
    }
    return row


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows: list[dict[str, object]], key: str, ylabel: str, path: Path) -> None:
    labels = [str(row["model"]) for row in rows]
    values = [float(row[key]) if row[key] is not None else 0.0 for row in rows]
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.7), 4.5))
    ax.bar(labels, values, color="#4C78A8")
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--result", action="append", required=True, help="Model=/path/docking_results.csv")
    args = parser.parse_args()

    rows = []
    for spec in args.result:
        if "=" not in spec:
            raise ValueError(f"Expected Model=csv, got {spec}")
        model, raw_path = spec.split("=", 1)
        rows.append(_summarize_one(model, Path(raw_path)))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, args.out_dir / "docking_model_summary.csv")
    (args.out_dir / "docking_model_summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _plot(rows, "score_median", "Docking score median (kcal/mol)", args.out_dir / "docking_score_median.png")
    _plot(rows, "best_5pct_mean", "Best 5% mean docking score", args.out_dir / "docking_best5_mean.png")
    _plot(rows, "rate_le_-10", "Fraction with score <= -10", args.out_dir / "docking_rate_le_-10.png")
    print(json.dumps({"models": len(rows), "out_dir": str(args.out_dir)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
