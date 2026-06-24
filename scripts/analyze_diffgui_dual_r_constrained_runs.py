"""Aggregate DiffGui dual-R constrained sampling runs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--run", required=True, action="append", type=Path, help="Run directory with reconstruction_audit.csv.")
    parser.add_argument("--label", action="append", default=None, help="Label for the preceding --run.")
    return parser.parse_args()


def main() -> int:
    """Aggregate run CSV files and write report artifacts."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    labels = args.label or [path.name for path in args.run]
    if len(labels) != len(args.run):
        raise ValueError("--label count must match --run count.")

    summary_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    smiles_rows: list[dict[str, Any]] = []
    example_mols: list[tuple[str, Chem.Mol]] = []
    for label, run_dir in zip(labels, args.run):
        rows = _read_csv(run_dir / "reconstruction_audit.csv")
        summary_rows.append(_summarize_run(label, run_dir, rows))
        failure_rows.extend(_count_rows(label, rows, "strict_failure_type", "failure_type"))
        smiles_rows.extend(_count_rows(label, rows, "smiles", "smiles"))
        example_mols.extend(_example_mols(label, run_dir, rows))

    _write_csv(args.out_dir / "diffgui_dual_r_run_summary.csv", summary_rows)
    _write_csv(args.out_dir / "diffgui_dual_r_failure_counts.csv", failure_rows)
    _write_csv(args.out_dir / "diffgui_dual_r_smiles_counts.csv", smiles_rows)
    _plot_run_summary(args.out_dir / "diffgui_dual_r_run_summary.png", summary_rows)
    _plot_failure_counts(args.out_dir / "diffgui_dual_r_failure_counts.png", failure_rows)
    _draw_examples(args.out_dir / "diffgui_dual_r_examples.png", example_mols)
    _write_report(
        args.out_dir / "diffgui_dual_r_run_comparison.md",
        summary_rows=summary_rows,
        failure_rows=failure_rows,
        smiles_rows=smiles_rows,
        has_examples=bool(example_mols),
    )
    print(f"wrote {args.out_dir}")
    return 0


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read audit CSV rows."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_summary_json(run_dir: Path) -> dict[str, Any]:
    """Read optional run summary JSON."""
    path = run_dir / "summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _summarize_run(label: str, run_dir: Path, rows: list[dict[str, str]]) -> dict[str, Any]:
    """Summarize one constrained sampling run."""
    summary = _read_summary_json(run_dir)
    total = len(rows)
    recon_ok = sum(1 for row in rows if row.get("reconstruction_status") == "ok")
    dual_success = sum(_to_int(row.get("dual_success")) for row in rows)
    strict_success = sum(_to_int(row.get("strict_dual_success")) for row in rows)
    nontrivial = sum(_to_int(row.get("nontrivial_sidechain_pair")) for row in rows)
    disconnected = sum(_to_int(row.get("disconnected_r_slot_count")) for row in rows)
    raw_disconnected = sum(_to_int(row.get("raw_disconnected_r_slot_count")) for row in rows)
    non_target_core = sum(_to_int(row.get("non_target_core_bond_count")) for row in rows)
    return {
        "label": label,
        "run_dir": str(run_dir),
        "force_r_chain": int(bool(summary.get("force_r_chain", False))),
        "drop_disconnected_r_slots": int(bool(summary.get("drop_disconnected_r_slots", False))),
        "total": total,
        "reconstruction_ok": recon_ok,
        "dual_success": dual_success,
        "strict_dual_success": strict_success,
        "nontrivial_sidechain_pair": nontrivial,
        "strict_success_rate": _rate(strict_success, total),
        "nontrivial_rate": _rate(nontrivial, total),
        "disconnected_r_slot_total": disconnected,
        "raw_disconnected_r_slot_total": raw_disconnected,
        "non_target_core_bond_total": non_target_core,
    }


def _count_rows(label: str, rows: list[dict[str, str]], source_key: str, target_key: str) -> list[dict[str, Any]]:
    """Count a string field in one run."""
    counter = Counter(_row_value(row, source_key) for row in rows)
    return [
        {"label": label, target_key: key, "count": int(value)}
        for key, value in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def _example_mols(label: str, run_dir: Path, rows: list[dict[str, str]]) -> list[tuple[str, Chem.Mol]]:
    """Select representative molecules from a run SDF."""
    sdf_path = run_dir / "generated.sdf"
    if not sdf_path.exists():
        return []
    mols = [mol for mol in Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False) if mol is not None]
    selected_indices: list[int] = []
    for desired_label in ("nontrivial_sidechain_pair", "trivial_sidechain_pair", "disconnected_r_slots"):
        for row in rows:
            if row.get("quality_label") == desired_label and int(row.get("mol_index", -1)) not in selected_indices:
                selected_indices.append(int(row["mol_index"]))
                break
    examples: list[tuple[str, Chem.Mol]] = []
    for mol_index in selected_indices[:3]:
        if 0 <= mol_index < len(mols):
            mol = Chem.Mol(mols[mol_index])
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                pass
            legend = f"{label} m{mol_index}"
            examples.append((legend, mol))
    return examples


def _row_value(row: dict[str, str], source_key: str) -> str:
    """Return a countable row value, inferring old strict labels when needed."""
    value = row.get(source_key, "")
    if value:
        return value
    if source_key != "strict_failure_type":
        return "missing"
    if row.get("reconstruction_status") != "ok":
        return "reconstruction_failed"
    if _to_int(row.get("dual_success")) == 0:
        failure = row.get("audit_failure_type") or row.get("failure_type") or "formal_dual_failure"
        return f"formal_{failure}"
    if _to_int(row.get("non_target_core_bond_count")) > 0:
        return "non_target_core_bond"
    if _to_int(row.get("disconnected_r_slot_count")) > 0:
        return "disconnected_r_slots"
    if _to_int(row.get("strict_dual_success")) == 1:
        return "ok"
    return "strict_unknown"


def _plot_run_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    """Plot success counts per run."""
    labels = [str(row["label"]) for row in rows]
    total = [int(row["total"]) for row in rows]
    strict = [int(row["strict_dual_success"]) for row in rows]
    nontrivial = [int(row["nontrivial_sidechain_pair"]) for row in rows]
    x = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.bar([idx - 0.22 for idx in x], total, width=0.22, label="total", color="#4C78A8")
    ax.bar(x, strict, width=0.22, label="strict success", color="#54A24B")
    ax.bar([idx + 0.22 for idx in x], nontrivial, width=0.22, label="nontrivial", color="#E45756")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("molecule count")
    ax.set_title("DiffGui dual-R constrained run comparison")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_failure_counts(path: Path, rows: list[dict[str, Any]]) -> None:
    """Plot strict failure counts."""
    if not rows:
        return
    labels = sorted({str(row["label"]) for row in rows})
    failure_types = sorted({str(row["failure_type"]) for row in rows})
    count_lookup = {(str(row["label"]), str(row["failure_type"])): int(row["count"]) for row in rows}
    x = list(range(len(labels)))
    bottoms = [0] * len(labels)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    colors = ["#4C78A8", "#F58518", "#E45756", "#72B7B2", "#B279A2"]
    for idx, failure_type in enumerate(failure_types):
        values = [count_lookup.get((label, failure_type), 0) for label in labels]
        ax.bar(x, values, bottom=bottoms, label=failure_type, color=colors[idx % len(colors)])
        bottoms = [base + value for base, value in zip(bottoms, values)]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("molecule count")
    ax.set_title("Strict failure modes")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _draw_examples(path: Path, examples: list[tuple[str, Chem.Mol]]) -> None:
    """Draw representative molecules."""
    if not examples:
        return
    legends, mols = zip(*examples)
    image = Draw.MolsToGridImage(
        list(mols),
        molsPerRow=3,
        subImgSize=(360, 260),
        legends=list(legends),
        useSVG=False,
    )
    image.save(str(path))


def _write_report(
    path: Path,
    *,
    summary_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
    smiles_rows: list[dict[str, Any]],
    has_examples: bool,
) -> None:
    """Write a Markdown comparison report."""
    lines = [
        "# DiffGui Dual-R Constrained Run Comparison",
        "",
        "## Summary",
        "",
        "| run | force chain | drop disconnected | total | strict success | strict rate | nontrivial | nontrivial rate | disconnected R slots | raw disconnected R slots |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['label']} | {row['force_r_chain']} | {row['drop_disconnected_r_slots']} | "
            f"{row['total']} | {row['strict_dual_success']} | "
            f"{float(row['strict_success_rate']):.3f} | {row['nontrivial_sidechain_pair']} | "
            f"{float(row['nontrivial_rate']):.3f} | {row['disconnected_r_slot_total']} | "
            f"{row['raw_disconnected_r_slot_total']} |"
        )
    lines.extend(
        [
            "",
            "![Run summary](diffgui_dual_r_run_summary.png)",
            "",
            "![Failure counts](diffgui_dual_r_failure_counts.png)",
            "",
        ]
    )
    if has_examples:
        lines.extend(["![Representative molecules](diffgui_dual_r_examples.png)", ""])
    lines.extend(
        [
            "## Failure Counts",
            "",
            "| run | failure type | count |",
            "|---|---|---:|",
        ]
    )
    for row in failure_rows:
        lines.append(f"| {row['label']} | {row['failure_type']} | {row['count']} |")
    lines.extend(
        [
            "",
            "## SMILES Counts",
            "",
            "| run | count | SMILES |",
            "|---|---:|---|",
        ]
    )
    for row in smiles_rows:
        lines.append(f"| {row['label']} | {row['count']} | `{row['smiles']}` |")
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- [diffgui_dual_r_run_summary.csv](diffgui_dual_r_run_summary.csv)",
            "- [diffgui_dual_r_failure_counts.csv](diffgui_dual_r_failure_counts.csv)",
            "- [diffgui_dual_r_smiles_counts.csv](diffgui_dual_r_smiles_counts.csv)",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write CSV rows."""
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _to_int(value: str | None) -> int:
    """Parse a CSV integer value."""
    if value in (None, ""):
        return 0
    return int(float(value))


def _rate(numerator: int, denominator: int) -> float:
    """Return a stable rounded rate."""
    return 0.0 if denominator == 0 else round(float(numerator) / float(denominator), 6)


if __name__ == "__main__":
    raise SystemExit(main())
