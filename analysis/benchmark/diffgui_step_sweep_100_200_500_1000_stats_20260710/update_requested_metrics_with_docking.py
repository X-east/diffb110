"""Update the requested DiffGui step table with docking statistics."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
STATS_DIR = SCRIPT_PATH.parent
DOCK_ROOT = STATS_DIR / "docking_100_200_500"
FORMAL_1000 = (
    SCRIPT_PATH.parents[3]
    / "analysis/benchmark/benchmark5/8gjl_c1101_benchmark_20260706_1000steps_3rep_formal/formal_3rep_docking_per_repeat.csv"
)
REQUESTED_CSV = STATS_DIR / "diffgui_step_requested_metrics_table.csv"
REQUESTED_MD = STATS_DIR / "diffgui_step_requested_metrics_table.md"
REPORT_MD = STATS_DIR / "report.md"
PER_REPEAT_CSV = STATS_DIR / "diffgui_step_docking_per_repeat.csv"
SUMMARY_CSV = STATS_DIR / "diffgui_step_docking_summary.csv"


def _format_mean_sd(values: list[float]) -> str:
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean:.2f} ± {sd:.2f}"


def _read_new_docking() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for step in (100, 200, 500):
        for repeat in (1, 2, 3):
            matches = sorted(DOCK_ROOT.glob(f"{step}steps_rep{repeat}_*/diffgui/run_summary.json"))
            if len(matches) != 1:
                raise FileNotFoundError(f"Expected one run_summary for {step} rep{repeat}, found {len(matches)}")
            summary_path = matches[0]
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "diffusion_steps": step,
                    "repeat": repeat,
                    "task": summary_path.parent.parent.name,
                    "input_ligands": int(summary["total_ligands"]),
                    "success_count": int(summary["success_count"]),
                    "success_rate": float(summary["success_rate"]),
                    "score_mean": float(summary["score_mean"]),
                    "score_median": float(summary["score_median"]),
                    "score_min_best": float(summary["score_min_best"]),
                    "score_max_worst": float(summary["score_max_worst"]),
                    "seconds": float(summary["seconds"]),
                    "summary_path": str(summary_path),
                }
            )
    return rows


def _read_formal_1000_docking() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with FORMAL_1000.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            model = row.get("model", "")
            if model.lower() != "diffgui":
                continue
            repeat_raw = row.get("repeat", "").replace("rep", "")
            rows.append(
                {
                    "diffusion_steps": 1000,
                    "repeat": int(repeat_raw),
                    "task": f"1000steps_rep{repeat_raw}",
                    "input_ligands": int(float(row["total_ligands"])),
                    "success_count": int(float(row["success_count"])),
                    "success_rate": float(row["success_rate"]),
                    "score_mean": float(row["score_mean"]),
                    "score_median": float(row["score_median"]),
                    "score_min_best": float(row.get("score_min_best") or row["score_best_min"]),
                    "score_max_worst": float(row.get("score_max_worst") or row["score_worst_max"]),
                    "seconds": "",
                    "summary_path": str(FORMAL_1000),
                }
            )
    if len(rows) != 3:
        raise RuntimeError(f"Expected three formal 1000-step DiffGui docking rows, found {len(rows)}")
    return sorted(rows, key=lambda item: int(item["repeat"]))


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summary_rows(per_repeat: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for step in (100, 200, 500, 1000):
        step_rows = [row for row in per_repeat if row["diffusion_steps"] == step]
        values = [float(row["score_mean"]) for row in step_rows]
        rows.append(
            {
                "diffusion_steps": step,
                "repeat_count": len(values),
                "score_mean_mean": statistics.fmean(values),
                "score_mean_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
                "score_mean_formatted": _format_mean_sd(values),
                "success_count_total": sum(int(row["success_count"]) for row in step_rows),
                "input_ligands_total": sum(int(row["input_ligands"]) for row in step_rows),
            }
        )
    return rows


def _read_requested_table() -> list[dict[str, str]]:
    with REQUESTED_CSV.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_requested_table(rows: list[dict[str, str]]) -> None:
    fieldnames = list(rows[0].keys())
    with REQUESTED_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# DiffGui requested metrics table",
        "",
        "Each value is mean ± sample SD across three repeats. Docking has been computed for all four step settings; PB-valid for 100/200/500 is still pending.",
        "",
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join(["---"] * len(fieldnames)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[field] for field in fieldnames) + " |")
    REQUESTED_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _update_report() -> None:
    requested_text = REQUESTED_MD.read_text(encoding="utf-8").splitlines()
    requested_section = ["## Requested Metrics Table", ""] + requested_text[2:]
    report = REPORT_MD.read_text(encoding="utf-8")
    marker = "## Requested Metrics Table"
    before, sep, _after = report.partition(marker)
    if not sep:
        raise RuntimeError(f"Could not find {marker} in {REPORT_MD}")
    REPORT_MD.write_text(before.rstrip() + "\n" + "\n".join(requested_section) + "\n", encoding="utf-8")


def main() -> None:
    per_repeat = _read_new_docking() + _read_formal_1000_docking()
    per_repeat = sorted(per_repeat, key=lambda item: (int(item["diffusion_steps"]), int(item["repeat"])))
    _write_rows(PER_REPEAT_CSV, per_repeat)

    summary_rows = _summary_rows(per_repeat)
    _write_rows(SUMMARY_CSV, summary_rows)
    docking_by_step = {str(row["diffusion_steps"]): row["score_mean_formatted"] for row in summary_rows}

    requested_rows = _read_requested_table()
    for row in requested_rows:
        row["Docking"] = docking_by_step[row["diffusion_steps"]]
    _write_requested_table(requested_rows)
    _update_report()

    print(json.dumps({"updated": str(REQUESTED_CSV), "docking": docking_by_step}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
