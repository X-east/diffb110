from __future__ import annotations

import csv
import math
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "analysis/benchmark/8gjl_c1101_benchmark_20260706_1000steps_3rep_formal"
ATTEMPTS = 1000

MODEL_ORDER = ["DiffGui", "DiffSBDD", "PMDM", "GCDM-SBDD", "TargetDiff"]

COUNT_KEYS = {
    "sdf_records",
    "rdkit_readable",
    "sanitize_valid",
    "unique_smiles",
}

SELECTED_PROPERTY_KEYS = [
    "qed_mean",
    "logp_mean",
    "tpsa_mean",
    "mw_mean",
    "heavy_atoms_mean",
    "hetero_atoms_mean",
    "carbon_fraction_mean",
    "ring_count_mean",
    "aromatic_ring_count_mean",
    "three_member_ring_count_mean",
    "bad_bond_length_fraction_mean",
    "num_fragments_mean",
    "largest_fragment_atom_frac_mean",
    "rotatable_bonds_mean",
]

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


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _mean_std(values: list[float]) -> tuple[float, float]:
    return mean(values), stdev(values) if len(values) > 1 else 0.0


def _is_float(value: str) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return value != ""


def _property_keys(rows: list[dict[str, object]]) -> list[str]:
    skip = {
        "repeat",
        "model",
        "diffusion_steps",
        "attempts",
        "seed_rule",
        "source_model_summary",
        "source_sdf",
        "source_note",
        *COUNT_KEYS,
    }
    keys: list[str] = []
    for key in rows[0]:
        if key in skip or key.endswith("_per_attempt"):
            continue
        values = [str(row.get(key, "")) for row in rows]
        if all(_is_float(value) for value in values):
            keys.append(key)
    return keys


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
            numeric_properties = {
                key: f"{float(value):.6f}"
                for key, value in source_row.items()
                if key not in {"model", "sdf_path", *COUNT_KEYS} and _is_float(value)
            }
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
                    **numeric_properties,
                    "source_model_summary": str(
                        run["model_overrides"].get(model, run["default_model_summary"])
                    ),
                    "source_sdf": source_row["sdf_path"],
                    "source_note": run["source_note"],
                }
            )
    return rows


def build_property_tables(per_repeat_rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    keys = _property_keys(per_repeat_rows)
    per_repeat_long: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for row in per_repeat_rows:
        for key in keys:
            per_repeat_long.append(
                {
                    "repeat": row["repeat"],
                    "model": row["model"],
                    "metric": key,
                    "value": row[key],
                }
            )
    for model in MODEL_ORDER:
        model_rows = [row for row in per_repeat_rows if row["model"] == model]
        for key in keys:
            values = [float(row[key]) for row in model_rows]
            metric_mean, metric_std = _mean_std(values)
            summary_rows.append(
                {
                    "model": model,
                    "metric": key,
                    "mean": f"{metric_mean:.6f}",
                    "std": f"{metric_std:.6f}",
                    "min": f"{min(values):.6f}",
                    "max": f"{max(values):.6f}",
                    "n_repeats": len(values),
                }
            )
    return per_repeat_long, summary_rows


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


def write_plots(
    per_repeat_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    property_summary_rows: list[dict[str, object]],
) -> None:
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

    selected = [key for key in SELECTED_PROPERTY_KEYS if any(row["metric"] == key for row in property_summary_rows)]
    cols = 2
    rows_n = math.ceil(len(selected) / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(11.0, 3.0 * rows_n))
    flat_axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    summary_by_model_metric = {
        (row["model"], row["metric"]): row for row in property_summary_rows
    }
    for ax, metric in zip(flat_axes, selected):
        values = [float(summary_by_model_metric[(model, metric)]["mean"]) for model in MODEL_ORDER]
        stds = [float(summary_by_model_metric[(model, metric)]["std"]) for model in MODEL_ORDER]
        ax.bar(range(len(MODEL_ORDER)), values, yerr=stds, capsize=3, color=[colors[m] for m in MODEL_ORDER])
        ax.set_title(metric)
        ax.set_xticks(range(len(MODEL_ORDER)))
        ax.set_xticklabels(MODEL_ORDER, rotation=30, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    for ax in flat_axes[len(selected):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "selected_property_summary_3rep.png", dpi=200)
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
        "![selected properties](selected_property_summary_3rep.png)",
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

    docking_summary_path = OUT_DIR / "formal_3rep_docking_model_summary.csv"
    docking_repeat_path = OUT_DIR / "formal_3rep_docking_per_repeat.csv"
    docking_summary_rows = _read_csv_rows(docking_summary_path)
    docking_repeat_rows = _read_csv_rows(docking_repeat_path)
    if docking_summary_rows and docking_repeat_rows:
        docking_summary_headers = [
            "model",
            "success_count_mean",
            "score_mean_mean",
            "score_median_mean",
            "best_5pct_mean_mean",
            "rate_le_-10_mean",
        ]
        docking_repeat_headers = [
            "repeat",
            "model",
            "total_ligands",
            "success_count",
            "score_mean",
            "score_median",
            "best_5pct_mean",
            "rate_le_-10",
        ]
        lines.extend(
            [
                "",
                "## Docking",
                "",
                "Docking was run separately for every repeat/model sanitize-valid SDF with the existing UniDock fast protocol.",
                "",
                "![docking median](docking_score_median_3rep.png)",
                "",
                "![docking best5](docking_best5_mean_3rep.png)",
                "",
                "### Docking mean over repeats",
                "",
                "| " + " | ".join(docking_summary_headers) + " |",
                "| " + " | ".join(["---"] * len(docking_summary_headers)) + " |",
            ]
        )
        for row in docking_summary_rows:
            lines.append("| " + " | ".join(str(row[key]) for key in docking_summary_headers) + " |")
        lines.extend(
            [
                "",
                "### Per-repeat docking",
                "",
                "| " + " | ".join(docking_repeat_headers) + " |",
                "| " + " | ".join(["---"] * len(docking_repeat_headers)) + " |",
            ]
        )
        for row in docking_repeat_rows:
            lines.append("| " + " | ".join(str(row[key]) for key in docking_repeat_headers) + " |")

    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `formal_3rep_per_repeat_counts.csv`",
            "- `formal_3rep_model_summary.csv`",
            "- `formal_3rep_property_per_repeat_long.csv`",
            "- `formal_3rep_property_summary_long.csv`",
            "- `formal_3rep_docking_per_repeat.csv`",
            "- `formal_3rep_docking_model_summary.csv`",
            "- `docking_summary.md`",
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
    property_per_repeat_rows, property_summary_rows = build_property_tables(per_repeat_rows)
    _write_csv(OUT_DIR / "formal_3rep_per_repeat_counts.csv", per_repeat_rows)
    _write_csv(OUT_DIR / "formal_3rep_model_summary.csv", summary_rows)
    _write_csv(OUT_DIR / "formal_3rep_property_per_repeat_long.csv", property_per_repeat_rows)
    _write_csv(OUT_DIR / "formal_3rep_property_summary_long.csv", property_summary_rows)
    write_sources(per_repeat_rows)
    write_plots(per_repeat_rows, summary_rows, property_summary_rows)
    write_report(per_repeat_rows, summary_rows)
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
