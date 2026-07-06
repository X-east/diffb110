from __future__ import annotations

import csv
import math
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "analysis/benchmark/8gjl_c1101_benchmark_20260706_1000steps_3rep_formal"
ATTEMPTS = 1000

MODEL_ORDER = ["DiffGui", "DiffSBDD", "PMDM", "GCDM-SBDD", "TargetDiff"]

RUNS = [
    {
        "repeat": "rep1",
        "seed_rule": "seed=gpu_id",
        "default_model_summary": ROOT
        / "analysis/benchmark/8gjl_c1101_benchmark_20260630_all_1000steps/all_1000step_collection/tables/model_summary.csv",
        "model_overrides": {
            "DiffGui": ROOT
            / "analysis/benchmark/8gjl_c1101_benchmark_20260701_diffgui_rep1_1000steps_seed1_7/evaluation/raw/model_summary.csv",
        },
        "source_note": "rep1 uses explicit DiffGui 1000-step rerun; other models use all_1000steps collection",
    },
    {
        "repeat": "rep2",
        "seed_rule": "seed=gpu_id+7",
        "default_model_summary": ROOT
        / "analysis/benchmark/8gjl_c1101_benchmark_20260701_all_rep2_1000steps_seed8_14/evaluation/raw_all_models/model_summary.csv",
        "model_overrides": {},
        "source_note": "all five models from rep2 1000-step run",
    },
    {
        "repeat": "rep3",
        "seed_rule": "seed=gpu_id+14",
        "default_model_summary": ROOT
        / "analysis/benchmark/8gjl_c1101_benchmark_20260701_all_rep3_1000steps_seed15_21/evaluation/raw_all_models/model_summary.csv",
        "model_overrides": {},
        "source_note": "all five models from rep3 1000-step run",
    },
]


def _read_model_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["model"]: row for row in rows}


def _float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value != "" else math.nan


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["model"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mean_std(values: list[float]) -> tuple[float, float]:
    return mean(values), stdev(values) if len(values) > 1 else 0.0


def build_per_repeat_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run in RUNS:
        default_rows = _read_model_rows(run["default_model_summary"])
        override_rows = {
            model: _read_model_rows(path)[model]
            for model, path in run["model_overrides"].items()
        }
        for model in MODEL_ORDER:
            source_row = override_rows.get(model, default_rows[model])
            sdf_written = int(float(source_row["sdf_records"]))
            rdkit_readable = int(float(source_row["rdkit_readable"]))
            sanitize_valid = int(float(source_row["sanitize_valid"]))
            unique_smiles = int(float(source_row["unique_smiles"]))
            rows.append(
                {
                    "repeat": run["repeat"],
                    "model": model,
                    "diffusion_steps": 1000,
                    "attempts": ATTEMPTS,
                    "seed_rule": run["seed_rule"],
                    "sdf_written": sdf_written,
                    "rdkit_readable": rdkit_readable,
                    "sanitize_valid": sanitize_valid,
                    "unique_smiles": unique_smiles,
                    "written_per_attempt": f"{sdf_written / ATTEMPTS:.6f}",
                    "sanitize_valid_per_attempt": f"{sanitize_valid / ATTEMPTS:.6f}",
                    "unique_per_attempt": f"{unique_smiles / ATTEMPTS:.6f}",
                    "qed_mean": f"{_float(source_row, 'qed_mean'):.6f}",
                    "logp_mean": f"{_float(source_row, 'logp_mean'):.6f}",
                    "tpsa_mean": f"{_float(source_row, 'tpsa_mean'):.6f}",
                    "ring_count_mean": f"{_float(source_row, 'ring_count_mean'):.6f}",
                    "aromatic_ring_count_mean": f"{_float(source_row, 'aromatic_ring_count_mean'):.6f}",
                    "three_member_ring_count_mean": f"{_float(source_row, 'three_member_ring_count_mean'):.6f}",
                    "carbon_fraction_mean": f"{_float(source_row, 'carbon_fraction_mean'):.6f}",
                    "source_model_summary": str(
                        run["model_overrides"].get(model, run["default_model_summary"])
                    ),
                    "source_sdf": source_row["sdf_path"],
                    "source_note": run["source_note"],
                }
            )
    return rows


def build_model_summary(per_repeat_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model in MODEL_ORDER:
        model_rows = [row for row in per_repeat_rows if row["model"] == model]
        sdf_counts = [int(row["sdf_written"]) for row in model_rows]
        sanitize_counts = [int(row["sanitize_valid"]) for row in model_rows]
        unique_counts = [int(row["unique_smiles"]) for row in model_rows]
        written_rates = [float(row["written_per_attempt"]) for row in model_rows]
        sanitize_rates = [float(row["sanitize_valid_per_attempt"]) for row in model_rows]
        unique_rates = [float(row["unique_per_attempt"]) for row in model_rows]
        written_mean, written_std = _mean_std(written_rates)
        sanitize_mean, sanitize_std = _mean_std(sanitize_rates)
        unique_mean, unique_std = _mean_std(unique_rates)
        rows.append(
            {
                "model": model,
                "repeats": len(model_rows),
                "diffusion_steps": 1000,
                "attempts_per_repeat": ATTEMPTS,
                "sdf_written_mean": f"{mean(sdf_counts):.3f}",
                "sdf_written_std": f"{stdev(sdf_counts):.3f}",
                "sanitize_valid_mean": f"{mean(sanitize_counts):.3f}",
                "sanitize_valid_std": f"{stdev(sanitize_counts):.3f}",
                "unique_smiles_mean": f"{mean(unique_counts):.3f}",
                "unique_smiles_std": f"{stdev(unique_counts):.3f}",
                "written_per_attempt_mean": f"{written_mean:.6f}",
                "written_per_attempt_std": f"{written_std:.6f}",
                "sanitize_valid_per_attempt_mean": f"{sanitize_mean:.6f}",
                "sanitize_valid_per_attempt_std": f"{sanitize_std:.6f}",
                "unique_per_attempt_mean": f"{unique_mean:.6f}",
                "unique_per_attempt_std": f"{unique_std:.6f}",
                "sanitize_valid_min": min(sanitize_counts),
                "sanitize_valid_max": max(sanitize_counts),
            }
        )
    return rows


def write_plots(per_repeat_rows: list[dict[str, object]], summary_rows: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    colors = {
        "DiffGui": "#4C78A8",
        "DiffSBDD": "#F58518",
        "PMDM": "#54A24B",
        "GCDM-SBDD": "#E45756",
        "TargetDiff": "#72B7B2",
    }

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    xs = range(len(MODEL_ORDER))
    means = [float(row["sanitize_valid_per_attempt_mean"]) for row in summary_rows]
    stds = [float(row["sanitize_valid_per_attempt_std"]) for row in summary_rows]
    ax.bar(xs, means, yerr=stds, capsize=4, color=[colors[m] for m in MODEL_ORDER])
    ax.set_xticks(list(xs))
    ax.set_xticklabels(MODEL_ORDER, rotation=25, ha="right")
    ax.set_ylabel("sanitize-valid / 1000 attempts")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sanitize_valid_rate_3rep.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for model in MODEL_ORDER:
        rows = [row for row in per_repeat_rows if row["model"] == model]
        ax.plot(
            [row["repeat"] for row in rows],
            [float(row["sanitize_valid_per_attempt"]) for row in rows],
            marker="o",
            label=model,
            color=colors[model],
        )
    ax.set_ylabel("sanitize-valid / 1000 attempts")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sanitize_valid_rate_by_repeat.png", dpi=200)
    plt.close(fig)


def write_report(per_repeat_rows: list[dict[str, object]], summary_rows: list[dict[str, object]]) -> None:
    summary_headers = [
        "model",
        "sdf_written_mean",
        "sanitize_valid_mean",
        "unique_smiles_mean",
        "sanitize_valid_per_attempt_mean",
        "sanitize_valid_per_attempt_std",
    ]
    repeat_headers = [
        "repeat",
        "model",
        "sdf_written",
        "sanitize_valid",
        "unique_smiles",
        "sanitize_valid_per_attempt",
    ]
    lines = [
        "# 8GJL/C1101 1000-step 3-repeat formal summary",
        "",
        "Protocol: 1000 diffusion steps, 1000 attempts per model per repeat. DiffGui rep1 is replaced by the explicit 1000-step rerun from `8gjl_c1101_benchmark_20260701_diffgui_rep1_1000steps_seed1_7`.",
        "",
        "![sanitize-valid mean](sanitize_valid_rate_3rep.png)",
        "",
        "![sanitize-valid by repeat](sanitize_valid_rate_by_repeat.png)",
        "",
        "## Mean over repeats",
        "",
        "| " + " | ".join(summary_headers) + " |",
        "| " + " | ".join(["---"] * len(summary_headers)) + " |",
    ]
    for row in summary_rows:
        lines.append("| " + " | ".join(str(row[key]) for key in summary_headers) + " |")

    lines.extend(
        [
            "",
            "## Per-repeat counts",
            "",
            "| " + " | ".join(repeat_headers) + " |",
            "| " + " | ".join(["---"] * len(repeat_headers)) + " |",
        ]
    )
    for row in per_repeat_rows:
        lines.append("| " + " | ".join(str(row[key]) for key in repeat_headers) + " |")

    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `formal_3rep_per_repeat_counts.csv`",
            "- `formal_3rep_model_summary.csv`",
            "- `formal_3rep_sources.csv`",
        ]
    )
    (OUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_sources(per_repeat_rows: list[dict[str, object]]) -> None:
    fields = ["repeat", "model", "source_model_summary", "source_sdf", "source_note"]
    source_rows = [{key: row[key] for key in fields} for row in per_repeat_rows]
    _write_csv(OUT_DIR / "formal_3rep_sources.csv", source_rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_repeat_rows = build_per_repeat_rows()
    summary_rows = build_model_summary(per_repeat_rows)
    _write_csv(OUT_DIR / "formal_3rep_per_repeat_counts.csv", per_repeat_rows)
    _write_csv(OUT_DIR / "formal_3rep_model_summary.csv", summary_rows)
    write_sources(per_repeat_rows)
    write_plots(per_repeat_rows, summary_rows)
    write_report(per_repeat_rows, summary_rows)
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
