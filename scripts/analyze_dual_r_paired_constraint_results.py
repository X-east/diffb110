"""Aggregate paired atom-type/root-distance dual-R constraint experiments."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--experiment-dir", required=True, action="append", type=Path)
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--nontrivial-min-atoms", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    """Aggregate experiment outputs and write report artifacts."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    labels = args.labels or [path.name for path in args.experiment_dir]
    if len(labels) != len(args.experiment_dir):
        raise ValueError("--labels length must match --experiment-dir count")

    summary_rows: list[dict[str, Any]] = []
    success_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    for label, experiment_dir in zip(labels, args.experiment_dir):
        summary_rows.extend(_load_experiment_summary(label, experiment_dir))
        success_rows.extend(_load_success_rows(label, experiment_dir))
        failure_rows.extend(_load_failure_rows(label, experiment_dir))

    _annotate_nontrivial_success(summary_rows, success_rows, args.nontrivial_min_atoms)
    _write_csv(args.out_dir / "paired_constraint_summary.csv", summary_rows)
    _write_csv(args.out_dir / "paired_constraint_success_details.csv", success_rows)
    _write_csv(args.out_dir / "paired_constraint_failure_counts.csv", failure_rows)
    _plot_success_rates(args.out_dir / "paired_constraint_success_rates.png", summary_rows)
    _plot_extractable_rates(args.out_dir / "paired_constraint_extractable_rates.png", summary_rows)
    _plot_success_branch_sizes(args.out_dir / "paired_constraint_success_branch_sizes.png", success_rows)
    _plot_failure_counts(args.out_dir / "paired_constraint_failure_counts.png", failure_rows)
    _write_report(
        args.out_dir / "report.md",
        summary_rows=summary_rows,
        success_rows=success_rows,
        failure_rows=failure_rows,
        nontrivial_min_atoms=int(args.nontrivial_min_atoms),
    )
    print(f"wrote {args.out_dir}")
    return 0


def _load_experiment_summary(label: str, experiment_dir: Path) -> list[dict[str, Any]]:
    """Load one experiment_summary.csv."""
    path = experiment_dir / "experiment_summary.csv"
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            normalized = dict(row)
            normalized["experiment"] = label
            for key in (
                "atom_sync_strength",
                "distance_strength",
                "max_distance_step",
                "dual_r_success_rate",
            ):
                normalized[key] = float(normalized[key])
            for key in (
                "total",
                "mapping_rdkit_valid",
                "core_found",
                "dual_r_success",
                "left_extractable",
                "right_extractable",
                "atom_symbols_match",
                "graph_signature_match",
            ):
                normalized[key] = int(normalized[key])
            rows.append(normalized)
    return rows


def _load_success_rows(label: str, experiment_dir: Path) -> list[dict[str, Any]]:
    """Load dual-success candidate details."""
    rows = []
    for setting_dir in sorted(path for path in experiment_dir.iterdir() if path.is_dir()):
        audit_csv = setting_dir / "audit" / "generated_dual_r_candidates.csv"
        if not audit_csv.exists():
            continue
        with audit_csv.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if int(row.get("dual_success", "0") or 0) != 1:
                    continue
                rows.append({
                    "experiment": label,
                    "setting": setting_dir.name,
                    "molecule_index": int(row["molecule_index"]),
                    "left_atom_count": int(row["left_atom_count"]),
                    "right_atom_count": int(row["right_atom_count"]),
                    "left_symbols": row["left_atom_symbols"],
                    "right_symbols": row["right_atom_symbols"],
                    "left_signature": row["left_signature"],
                    "right_signature": row["right_signature"],
                    "left_attachment": row["left_attachment_bond_type"],
                    "right_attachment": row["right_attachment_bond_type"],
                    "candidate_name": row["candidate_name"],
                })
    return rows


def _load_failure_rows(label: str, experiment_dir: Path) -> list[dict[str, Any]]:
    """Load failure-count rows from experiment summary."""
    rows = []
    summary_path = experiment_dir / "experiment_summary.csv"
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            failure_counts = json.loads(row["failure_counts"])
            for failure_type, count in failure_counts.items():
                rows.append({
                    "experiment": label,
                    "setting": row["setting"],
                    "failure_type": failure_type,
                    "count": int(count),
                })
    return rows


def _annotate_nontrivial_success(
    summary_rows: list[dict[str, Any]],
    success_rows: list[dict[str, Any]],
    min_atoms: int,
) -> None:
    """Add nontrivial success counts to summary rows."""
    counter: Counter[tuple[str, str]] = Counter()
    for row in success_rows:
        if int(row["left_atom_count"]) >= min_atoms and int(row["right_atom_count"]) >= min_atoms:
            counter[(row["experiment"], row["setting"])] += 1
    for row in summary_rows:
        nontrivial = int(counter[(row["experiment"], row["setting"])])
        row["nontrivial_dual_r_success"] = nontrivial
        row["nontrivial_success_rate"] = (
            0.0 if int(row["total"]) == 0 else round(nontrivial / int(row["total"]), 6)
        )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to CSV, including an empty marker if needed."""
    if not rows:
        path.with_suffix(path.suffix + ".none.txt").write_text("no rows\n", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_success_rates(path: Path, rows: list[dict[str, Any]]) -> None:
    """Plot dual-R success and nontrivial success rates."""
    labels = [_row_label(row) for row in rows]
    dual = [float(row["dual_r_success_rate"]) for row in rows]
    nontrivial = [float(row["nontrivial_success_rate"]) for row in rows]
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    x = list(range(len(labels)))
    ax.bar([idx - 0.18 for idx in x], dual, width=0.36, label="audit success")
    ax.bar([idx + 0.18 for idx in x], nontrivial, width=0.36, label="nontrivial success")
    ax.set_ylabel("Rate among audited molecules")
    ax.set_ylim(0, 1.0)
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_title("Dual-R paired constraint success rates")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_extractable_rates(path: Path, rows: list[dict[str, Any]]) -> None:
    """Plot left/right extractable rates."""
    labels = [_row_label(row) for row in rows]
    left = [_rate(row["left_extractable"], row["total"]) for row in rows]
    right = [_rate(row["right_extractable"], row["total"]) for row in rows]
    x = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    ax.bar([idx - 0.18 for idx in x], left, width=0.36, label="left extractable")
    ax.bar([idx + 0.18 for idx in x], right, width=0.36, label="right extractable")
    ax.set_ylabel("Rate among audited molecules")
    ax.set_ylim(0, 1.0)
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_title("Attachment extractability")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_success_branch_sizes(path: Path, rows: list[dict[str, Any]]) -> None:
    """Plot branch atom counts for successful molecules."""
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    if rows:
        labels = [f"{row['experiment']}/{row['setting']}/m{row['molecule_index']}" for row in rows]
        counts = [int(row["left_atom_count"]) for row in rows]
        ax.bar(range(len(labels)), counts)
        ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
        ax.set_ylabel("R atom count")
    else:
        ax.text(0.5, 0.5, "No dual-R successes", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title("Branch size of dual-R successes")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_failure_counts(path: Path, rows: list[dict[str, Any]]) -> None:
    """Plot stacked failure counts."""
    if not rows:
        return
    labels = sorted({(row["experiment"], row["setting"]) for row in rows})
    failure_types = sorted({row["failure_type"] for row in rows})
    counts = {(row["experiment"], row["setting"], row["failure_type"]): int(row["count"]) for row in rows}
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    bottoms = [0] * len(labels)
    x = list(range(len(labels)))
    for failure_type in failure_types:
        values = [counts.get((experiment, setting, failure_type), 0) for experiment, setting in labels]
        ax.bar(x, values, bottom=bottoms, label=failure_type)
        bottoms = [base + value for base, value in zip(bottoms, values)]
    ax.set_xticks(x, [f"{experiment}\n{setting}" for experiment, setting in labels], rotation=25, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Failure type counts")
    ax.legend(frameon=False, fontsize=8, ncols=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_report(
    path: Path,
    *,
    summary_rows: list[dict[str, Any]],
    success_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
    nontrivial_min_atoms: int,
) -> None:
    """Write Markdown report with embedded figures."""
    text = f"""# DiffSBDD paired atom-type + root-distance constraint test

## Summary

This experiment tests the only remaining DiffSBDD route after passing on edge/halfedge integration:

```text
paired atom slots
R1_i / R2_i atom-feature synchronization
R1 root and R2 root coordinates remain free but are softly nudged toward C0/C11 attachment distance
no edge diffusion, no explicit bond constraints
```

Two node-count settings were tested: `add_n_nodes=4, pair_count=2` and `add_n_nodes=6, pair_count=3`.

{_summary_table(summary_rows)}

![Dual-R success rates](paired_constraint_success_rates.png)

![Attachment extractability](paired_constraint_extractable_rates.png)

## Successful Candidate Details

The audit-success molecules are listed below. `nontrivial success` means both R branches contain at least `{nontrivial_min_atoms}` atoms.

{_success_table(success_rows, nontrivial_min_atoms)}

![Success branch sizes](paired_constraint_success_branch_sizes.png)

## Failure Modes

{_failure_table(failure_rows)}

![Failure counts](paired_constraint_failure_counts.png)

## Interpretation

- Atom-type synchronization alone did **not** produce dual-R success in either add4 or add6.
- Adding the root-distance nudge produced audit successes: `4/7` for add4 and `2/10` for add6 among molecules that survived reconstruction/audit.
- All audit-success molecules in this run are single-atom R groups (`O`, `F`, or `Cl`). Therefore, the current prototype proves that paired atom/root constraints can force simple symmetric substituents, but it does **not** yet solve meaningful multi-atom sidechain generation.
- Larger add6 generation did not improve meaningful sidechain success. It increased cases where one branch expands into unintended core-connected structures or atom symbols mismatch.
- Since no edge constraints are used, bond and ring consistency are still left to RDKit/OpenBabel reconstruction and post-hoc audit. This is the central remaining limitation.

## Data Files

- [aggregate summary](paired_constraint_summary.csv)
- [success details](paired_constraint_success_details.csv)
- [failure counts](paired_constraint_failure_counts.csv)
"""
    path.write_text(text, encoding="utf-8")


def _summary_table(rows: list[dict[str, Any]]) -> str:
    """Return Markdown summary table."""
    lines = [
        "| experiment | setting | total | left ext | right ext | atom match | graph match | audit success | nontrivial success |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['experiment']} | {row['setting']} | {row['total']} | "
            f"{row['left_extractable']} | {row['right_extractable']} | "
            f"{row['atom_symbols_match']} | {row['graph_signature_match']} | "
            f"{row['dual_r_success']} | {row['nontrivial_dual_r_success']} |"
        )
    return "\n".join(lines)


def _success_table(rows: list[dict[str, Any]], min_atoms: int) -> str:
    """Return Markdown success table."""
    if not rows:
        return "No audit-success molecules."
    lines = [
        "| experiment | setting | mol | left atoms | right atoms | signature | nontrivial |",
        "|---|---|---:|---:|---:|---|---:|",
    ]
    for row in rows:
        nontrivial = int(row["left_atom_count"]) >= min_atoms and int(row["right_atom_count"]) >= min_atoms
        lines.append(
            f"| {row['experiment']} | {row['setting']} | {row['molecule_index']} | "
            f"{row['left_atom_count']} | {row['right_atom_count']} | "
            f"`{row['left_signature']}` | {int(nontrivial)} |"
        )
    return "\n".join(lines)


def _failure_table(rows: list[dict[str, Any]]) -> str:
    """Return Markdown failure table."""
    if not rows:
        return "No failures recorded."
    lines = [
        "| experiment | setting | failure type | count |",
        "|---|---|---|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['experiment']} | {row['setting']} | {row['failure_type']} | {row['count']} |")
    return "\n".join(lines)


def _row_label(row: dict[str, Any]) -> str:
    """Return plot label."""
    return f"{row['experiment']}/{row['setting']}"


def _rate(count: int, total: int) -> float:
    """Return count / total with zero guard."""
    total = int(total)
    return 0.0 if total == 0 else float(count) / float(total)


if __name__ == "__main__":
    raise SystemExit(main())
