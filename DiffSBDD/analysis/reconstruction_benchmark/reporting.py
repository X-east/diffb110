"""CSV, plot, SDF, and Markdown writers for reconstruction benchmarks."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from rdkit import Chem

PER_MOLECULE_FIELDS = [
    "sample_idx",
    "method",
    "num_atoms",
    "sanitize_success",
    "error",
    "num_bonds",
    "checked_bond_lengths",
    "bad_bond_lengths",
    "bad_bond_length_rate",
    "mean_bond_length_abs_error",
    "chemistry_problem_count",
    "valence_problem_count",
    "num_fragments",
    "largest_fragment_atom_frac",
    "ring_count",
    "aromatic_ring_count",
    "three_member_ring_count",
    "fused_ring_pair_count",
    "carbon_fraction",
    "hetero_atom_count",
    "halogen_count",
    "formal_charge_abs_sum",
    "drug_metrics_available",
    "qed",
    "sa",
    "logp",
    "tpsa",
    "hba",
    "hbd",
    "rotatable_bonds",
    "mol_weight",
    "lipinski",
    "reconstruction_seconds",
    "smiles",
]

SUMMARY_NUMERIC_FIELDS = [
    "sanitize_success",
    "num_atoms",
    "num_bonds",
    "checked_bond_lengths",
    "bad_bond_lengths",
    "bad_bond_length_rate",
    "mean_bond_length_abs_error",
    "chemistry_problem_count",
    "valence_problem_count",
    "num_fragments",
    "largest_fragment_atom_frac",
    "ring_count",
    "aromatic_ring_count",
    "three_member_ring_count",
    "fused_ring_pair_count",
    "carbon_fraction",
    "hetero_atom_count",
    "halogen_count",
    "formal_charge_abs_sum",
    "drug_metrics_available",
    "qed",
    "sa",
    "logp",
    "tpsa",
    "hba",
    "hbd",
    "rotatable_bonds",
    "mol_weight",
    "lipinski",
    "reconstruction_seconds",
]

EXTERNAL_RESOURCE_FIELDS = [
    "name",
    "kind",
    "checkpoint",
    "checkpoint_exists",
    "config",
    "config_exists",
    "source",
    "source_exists",
    "status",
]


def _format_float(value: object) -> object:
    """Format floats for stable CSV output."""
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    """Write rows to CSV with stable column order."""
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: list[dict[str, object]], methods: list[str]) -> list[dict[str, object]]:
    """Aggregate per-molecule rows by reconstruction method."""
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)

    summary_rows = []
    for method in methods:
        method_rows = grouped[method]
        row: dict[str, object] = {"method": method, "samples": len(method_rows)}
        for field in SUMMARY_NUMERIC_FIELDS:
            values = [float(item[field]) for item in method_rows if item[field] != ""]
            row[f"mean_{field}"] = sum(values) / len(values) if values else 0.0
        summary_rows.append(row)
    return summary_rows


def _plot_summary(outdir: Path, summary_rows: list[dict[str, object]]) -> None:
    """Write summary bar plots."""
    methods = [str(row["method"]) for row in summary_rows]
    plot_specs = [
        ("mean_sanitize_success", "sanitize success", "sanitize_success_by_method.png"),
        ("mean_bad_bond_length_rate", "bad bond length rate", "bad_bond_length_by_method.png"),
        ("mean_num_fragments", "fragments", "fragment_count_by_method.png"),
        ("mean_largest_fragment_atom_frac", "largest fragment fraction", "largest_fragment_by_method.png"),
        ("mean_aromatic_ring_count", "aromatic rings", "aromatic_ring_by_method.png"),
        ("mean_three_member_ring_count", "three-member rings", "three_member_ring_by_method.png"),
        ("mean_qed", "QED", "qed_by_method.png"),
        ("mean_sa", "SA", "sa_by_method.png"),
        ("mean_tpsa", "TPSA", "tpsa_by_method.png"),
        ("mean_logp", "LogP", "logp_by_method.png"),
        ("mean_reconstruction_seconds", "seconds / molecule", "time_by_method.png"),
    ]
    palette = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2", "#FF9DA6"]
    for field, ylabel, filename in plot_specs:
        values = [float(row[field]) for row in summary_rows]
        plt.figure(figsize=(7, 4))
        plt.bar(methods, values, color=palette[:len(methods)])
        plt.ylabel(ylabel)
        plt.xticks(rotation=15, ha="right")
        lower = min(0.0, min(values) * 1.15) if values else 0.0
        upper = max(values) * 1.15 if values else 1.0
        plt.ylim(lower, max(1.0, upper))
        plt.tight_layout()
        plt.savefig(outdir / filename, dpi=180)
        plt.close()


def _write_sdf_outputs(outdir: Path, molecules_by_method: dict[str, list[Chem.Mol]]) -> None:
    """Write one SDF containing sanitized molecules per reconstruction method."""
    for method, mols in molecules_by_method.items():
        writer = Chem.SDWriter(str(outdir / f"{method}_sanitized.sdf"))
        try:
            for mol in mols:
                writer.write(mol)
        finally:
            writer.close()


def _write_markdown_report(
    *,
    report_path: Path,
    metadata: dict[str, object],
    summary_rows: list[dict[str, object]],
    external_resources: list[dict[str, object]],
) -> None:
    """Write a compact rendered benchmark report."""
    with report_path.open("w") as handle:
        handle.write("# Generated Reconstruction Benchmark Report\n\n")
        for key, value in metadata.items():
            handle.write(f"- {key}: `{value}`\n")
        handle.write("\n")
        handle.write(
            "| method | sanitize | bad_bond_length | fragments | largest_frag | aromatic | QED | SA | TPSA | sec/mol |\n"
        )
        handle.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in summary_rows:
            handle.write(
                f"| {row['method']} "
                f"| {float(row['mean_sanitize_success']):.3f} "
                f"| {float(row['mean_bad_bond_length_rate']):.3f} "
                f"| {float(row['mean_num_fragments']):.3f} "
                f"| {float(row['mean_largest_fragment_atom_frac']):.3f} "
                f"| {float(row['mean_aromatic_ring_count']):.3f} "
                f"| {float(row['mean_qed']):.3f} "
                f"| {float(row['mean_sa']):.3f} "
                f"| {float(row['mean_tpsa']):.3f} "
                f"| {float(row['mean_reconstruction_seconds']):.4f} |\n"
            )

        if external_resources:
            handle.write("\n## External Bond Resources\n\n")
            handle.write("| name | checkpoint | config | source | status |\n")
            handle.write("|---|---:|---:|---:|---|\n")
            for row in external_resources:
                handle.write(
                    f"| {row['name']} "
                    f"| {row['checkpoint_exists']} "
                    f"| {row['config_exists']} "
                    f"| {row['source_exists']} "
                    f"| {row['status']} |\n"
                )

        handle.write("\n![sanitize success](sanitize_success_by_method.png)\n\n")
        handle.write("![bad bond length](bad_bond_length_by_method.png)\n\n")
        handle.write("![fragment count](fragment_count_by_method.png)\n\n")
        handle.write("![largest fragment](largest_fragment_by_method.png)\n\n")
        handle.write("![aromatic rings](aromatic_ring_by_method.png)\n\n")
        handle.write("![QED](qed_by_method.png)\n\n")
        handle.write("![SA](sa_by_method.png)\n\n")
        handle.write("![TPSA](tpsa_by_method.png)\n\n")
        handle.write("![time](time_by_method.png)\n")


def write_benchmark_outputs(
    *,
    outdir: Path,
    rows: list[dict[str, object]],
    method_names: list[str],
    molecules_by_method: dict[str, list[Chem.Mol]],
    metadata: dict[str, object],
    external_resources: list[dict[str, object]],
) -> Path:
    """Write all benchmark artifacts and return the Markdown report path."""
    formatted_rows = [{key: _format_float(value) for key, value in row.items()} for row in rows]
    _write_csv(outdir / "generated_reconstruction_per_molecule.csv", formatted_rows, PER_MOLECULE_FIELDS)

    summary_rows = summarize_rows(formatted_rows, method_names)
    summary_fields = ["method", "samples"] + [
        key
        for key in summary_rows[0]
        if key not in {"method", "samples"}
    ]
    _write_csv(
        outdir / "generated_reconstruction_summary.csv",
        [{key: _format_float(value) for key, value in row.items()} for row in summary_rows],
        summary_fields,
    )
    if external_resources:
        _write_csv(outdir / "external_bond_resources.csv", external_resources, EXTERNAL_RESOURCE_FIELDS)
    _plot_summary(outdir, summary_rows)
    _write_sdf_outputs(outdir, molecules_by_method)

    report_path = outdir / "generated_reconstruction_benchmark_report.md"
    _write_markdown_report(
        report_path=report_path,
        metadata=metadata,
        summary_rows=summary_rows,
        external_resources=external_resources,
    )
    return report_path
