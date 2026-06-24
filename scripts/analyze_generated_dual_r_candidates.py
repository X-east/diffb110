"""Analyze generated SDF molecules for honokiol dual-R candidate consistency."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from rdkit import Chem

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ddpo.constraints.dual_sidechain import (  # noqa: E402
    audit_dual_sidechain,
    build_core_mapping_query,
    build_dual_substitution_spec_from_phenolic_sites,
    find_phenolic_oh_sites,
    map_core_to_target,
    summarize_audit_results,
)
from ddpo.constraints.dual_sidechain.audit import DualSidechainAuditResult  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdf", required=True, type=Path, help="Generated or candidate SDF to analyze.")
    parser.add_argument(
        "--site-reference-sdf",
        required=True,
        type=Path,
        help="Reference honokiol SDF used to define the two phenolic OH replacement sites.",
    )
    parser.add_argument("--site-reference-index", type=int, default=0, help="Reference molecule index.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for reports and tables.")
    parser.add_argument("--left-site-index", type=int, default=0, help="Phenolic OH site index for left R site.")
    parser.add_argument("--right-site-index", type=int, default=1, help="Phenolic OH site index for right R site.")
    parser.add_argument(
        "--max-branch-atoms",
        type=int,
        default=None,
        help="Optional maximum R branch size before marking candidate branches too large.",
    )
    parser.add_argument(
        "--max-core-matches",
        type=int,
        default=1000,
        help="Maximum substructure matches kept per molecule.",
    )
    parser.add_argument(
        "--compute-mcs",
        action="store_true",
        help="Compute lightweight MCS diagnostics for molecules where the reference core is not found.",
    )
    parser.add_argument("--mcs-timeout", type=int, default=2, help="MCS timeout in seconds per failed molecule.")
    parser.add_argument(
        "--write-success-sdf",
        action="store_true",
        help="Write successful dual-R candidates to successful_dual_r_candidates.sdf.",
    )
    parser.add_argument(
        "--write-failure-sdf",
        action="store_true",
        help="Write failed dual-R candidates to failed_dual_r_candidates.sdf with audit properties.",
    )
    parser.add_argument(
        "--write-audited-sdf",
        action="store_true",
        help="Write all readable candidates to audited_dual_r_candidates.sdf with audit properties.",
    )
    return parser.parse_args()


def main() -> int:
    """Run generated candidate mapping and dual-R audit."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    reference_mol = _read_one_sdf(args.site_reference_sdf, int(args.site_reference_index))
    reference_sites = find_phenolic_oh_sites(reference_mol)
    reference_spec = build_dual_substitution_spec_from_phenolic_sites(
        reference_sites,
        left_index=int(args.left_site_index),
        right_index=int(args.right_site_index),
        max_branch_atoms=args.max_branch_atoms,
    )
    core_query = build_core_mapping_query(reference_mol, reference_spec)
    target_mols = _read_sdf(args.sdf)

    rows: list[dict[str, Any]] = []
    audit_results: list[DualSidechainAuditResult] = []
    success_mols: list[Chem.Mol] = []
    failure_mols: list[Chem.Mol] = []
    audited_mols: list[Chem.Mol] = []
    mapping_results = []
    for molecule_index, mol in enumerate(target_mols):
        mapping = map_core_to_target(
            mol,
            core_query,
            molecule_index=molecule_index,
            max_matches=int(args.max_core_matches),
            compute_mcs=bool(args.compute_mcs),
            mcs_timeout=int(args.mcs_timeout),
        )
        mapping_results.append(mapping)
        if mapping.core_found and mapping.target_spec is not None:
            audit_result = audit_dual_sidechain(mol, mapping.target_spec, molecule_index=molecule_index)
        else:
            audit_result = _mapping_failed_audit_result(molecule_index, mapping.failure_type)
        audit_results.append(audit_result)
        row = {}
        row.update(mapping.to_dict())
        row.update(audit_result.to_dict())
        row["candidate_status"] = _candidate_status(mapping, audit_result)
        row["candidate_name"] = _mol_name(mol, molecule_index)
        rows.append(row)
        if mol is not None:
            annotated_mol = _annotate_candidate_mol(mol, row, molecule_index)
            if args.write_audited_sdf:
                audited_mols.append(Chem.Mol(annotated_mol))
            if audit_result.dual_success:
                if args.write_success_sdf:
                    success_mols.append(Chem.Mol(annotated_mol))
            elif args.write_failure_sdf:
                failure_mols.append(Chem.Mol(annotated_mol))

    summary = _summarize(mapping_results, audit_results, rows)

    _write_csv(args.out_dir / "generated_dual_r_candidates.csv", rows)
    _write_json(
        args.out_dir / "summary.json",
        {
            "input_sdf": str(args.sdf),
            "site_reference_sdf": str(args.site_reference_sdf),
            "site_reference_index": int(args.site_reference_index),
            "reference_sites": [site.to_dict() for site in reference_sites],
            "reference_spec": reference_spec.to_dict(),
            "core_query": core_query.to_dict(),
            "summary": summary,
        },
    )
    _plot_status_counts(args.out_dir / "candidate_status_counts.png", rows)
    _plot_failure_counts(args.out_dir / "failure_type_counts.png", rows)
    _plot_core_mapping(args.out_dir / "core_mapping_summary.png", summary)
    success_sdf_path = args.out_dir / "successful_dual_r_candidates.sdf"
    success_empty_marker_path = args.out_dir / "successful_dual_r_candidates.none.txt"
    if args.write_success_sdf:
        _write_success_sdf_or_marker(success_sdf_path, success_empty_marker_path, success_mols)
    failure_sdf_path = args.out_dir / "failed_dual_r_candidates.sdf"
    failure_empty_marker_path = args.out_dir / "failed_dual_r_candidates.none.txt"
    if args.write_failure_sdf:
        _write_failure_sdf_or_marker(failure_sdf_path, failure_empty_marker_path, failure_mols)
    audited_sdf_path = args.out_dir / "audited_dual_r_candidates.sdf"
    audited_empty_marker_path = args.out_dir / "audited_dual_r_candidates.none.txt"
    if args.write_audited_sdf:
        _write_audited_sdf_or_marker(audited_sdf_path, audited_empty_marker_path, audited_mols)
    _write_report(args.out_dir / "report.md", args, reference_sites, reference_spec, core_query, summary)

    print(f"wrote {args.out_dir}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _read_sdf(path: Path) -> list[Chem.Mol | None]:
    """Read molecules from SDF while preserving invalid entries as much as RDKit allows."""
    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
    mols = [mol for mol in supplier]
    for mol in mols:
        if mol is None:
            continue
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            pass
    return mols


def _read_one_sdf(path: Path, index: int) -> Chem.Mol:
    """Read one valid reference molecule."""
    mols = [mol for mol in _read_sdf(path) if mol is not None]
    if not mols:
        raise ValueError(f"No molecules read from {path}")
    if index < 0 or index >= len(mols):
        raise ValueError(f"Reference index {index} is out of range for {len(mols)} molecules")
    mol = Chem.Mol(mols[index])
    Chem.SanitizeMol(mol)
    return mol


def _mapping_failed_audit_result(molecule_index: int, mapping_failure_type: str) -> DualSidechainAuditResult:
    """Return an audit-shaped result when core mapping failed before branch extraction."""
    from ddpo.constraints.dual_sidechain.audit import SidechainBranchFacts

    left = SidechainBranchFacts(
        label="R1",
        core_atom_idx=-1,
        root_atom_idx=None,
        extractable=False,
        failure_type=mapping_failure_type,
    )
    right = SidechainBranchFacts(
        label="R2",
        core_atom_idx=-1,
        root_atom_idx=None,
        extractable=False,
        failure_type=mapping_failure_type,
    )
    return DualSidechainAuditResult(
        molecule_index=molecule_index,
        rdkit_valid=False,
        left=left,
        right=right,
        failure_type=f"mapping_{mapping_failure_type}",
        errors=(mapping_failure_type,),
    )


def _candidate_status(mapping, audit_result: DualSidechainAuditResult) -> str:
    """Return a single high-level status label for one candidate."""
    if not mapping.rdkit_valid:
        return "invalid_mol"
    if not mapping.core_found:
        return "core_missing"
    if audit_result.dual_success:
        return "dual_r_success"
    return "dual_r_failed"


def _mol_name(mol: Chem.Mol | None, molecule_index: int) -> str:
    """Return a stable display name for one molecule."""
    if mol is not None and mol.HasProp("_Name") and mol.GetProp("_Name"):
        return mol.GetProp("_Name")
    return f"mol_{molecule_index}"


def _summarize(mapping_results, audit_results: list[DualSidechainAuditResult], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize mapping and dual-R audit results."""
    total = len(rows)
    if total == 0:
        return {"total": 0}
    mapping_failures = Counter(mapping.failure_type for mapping in mapping_results)
    statuses = Counter(row["candidate_status"] for row in rows)
    audit_summary = summarize_audit_results(audit_results)
    return {
        "total": total,
        "mapping_rdkit_valid": sum(int(mapping.rdkit_valid) for mapping in mapping_results),
        "mapping_rdkit_valid_rate": _rate(sum(int(mapping.rdkit_valid) for mapping in mapping_results), total),
        "core_found": sum(int(mapping.core_found) for mapping in mapping_results),
        "core_found_rate": _rate(sum(int(mapping.core_found) for mapping in mapping_results), total),
        "dual_r_success": sum(int(result.dual_success) for result in audit_results),
        "dual_r_success_rate": _rate(sum(int(result.dual_success) for result in audit_results), total),
        "candidate_status_counts": dict(statuses),
        "mapping_failure_counts": dict(mapping_failures),
        "audit_failure_counts": audit_summary.get("failure_counts", {}),
        "audit_summary": audit_summary,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write flat candidate rows to CSV."""
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON file."""
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_sdf(path: Path, mols: list[Chem.Mol]) -> None:
    """Write selected molecules to SDF."""
    writer = Chem.SDWriter(str(path))
    try:
        for mol in mols:
            writer.write(mol)
    finally:
        writer.close()


def _annotate_candidate_mol(mol: Chem.Mol, row: dict[str, Any], molecule_index: int) -> Chem.Mol:
    """Return one molecule annotated with mapping and dual-R audit fields."""
    annotated = Chem.Mol(mol)
    candidate_name = str(row.get("candidate_name") or _mol_name(mol, molecule_index))
    status = str(row.get("candidate_status") or "")
    failure_type = str(row.get("failure_type") or "")
    annotated.SetProp("_Name", f"{candidate_name}|{status}|{failure_type}")
    annotated.SetProp("dual_r_molecule_index", str(molecule_index))

    for key in _candidate_sdf_property_keys():
        value = row.get(key)
        if value is None:
            continue
        text = str(value)
        if text == "":
            continue
        annotated.SetProp(f"dual_r_{key}", text)
    return annotated


def _candidate_sdf_property_keys() -> tuple[str, ...]:
    """Return row fields worth embedding as SDF properties."""
    return (
        "candidate_name",
        "candidate_status",
        "failure_type",
        "errors",
        "mapping_rdkit_valid",
        "core_found",
        "mapping_failure_type",
        "mapping_errors",
        "core_match_count",
        "selected_match_index",
        "target_left_core_idx",
        "target_right_core_idx",
        "mcs_num_atoms",
        "mcs_num_bonds",
        "mcs_atom_fraction",
        "dual_success",
        "left_extractable",
        "left_failure_type",
        "left_core_atom_idx",
        "left_root_atom_idx",
        "left_atom_count",
        "left_atom_indices",
        "left_atom_symbols",
        "left_attachment_bond_type",
        "left_attachment_is_aromatic",
        "left_ring_count",
        "left_aromatic_ring_count",
        "left_three_member_ring_count",
        "left_max_fused_ring_system_size",
        "left_signature",
        "right_extractable",
        "right_failure_type",
        "right_core_atom_idx",
        "right_root_atom_idx",
        "right_atom_count",
        "right_atom_indices",
        "right_atom_symbols",
        "right_attachment_bond_type",
        "right_attachment_is_aromatic",
        "right_ring_count",
        "right_aromatic_ring_count",
        "right_three_member_ring_count",
        "right_max_fused_ring_system_size",
        "right_signature",
    )


def _write_success_sdf_or_marker(sdf_path: Path, marker_path: Path, mols: list[Chem.Mol]) -> None:
    """Write successful molecules or an explicit empty-result marker."""
    if not mols:
        if sdf_path.exists():
            sdf_path.unlink()
        marker_path.write_text("No successful dual-R candidates were found.\n", encoding="utf-8")
        return
    if marker_path.exists():
        marker_path.unlink()
    _write_sdf(sdf_path, mols)


def _write_failure_sdf_or_marker(sdf_path: Path, marker_path: Path, mols: list[Chem.Mol]) -> None:
    """Write failed molecules or an explicit empty-result marker."""
    if not mols:
        if sdf_path.exists():
            sdf_path.unlink()
        marker_path.write_text("No failed dual-R candidates were found.\n", encoding="utf-8")
        return
    if marker_path.exists():
        marker_path.unlink()
    _write_sdf(sdf_path, mols)


def _write_audited_sdf_or_marker(sdf_path: Path, marker_path: Path, mols: list[Chem.Mol]) -> None:
    """Write all annotated molecules or an explicit empty-result marker."""
    if not mols:
        if sdf_path.exists():
            sdf_path.unlink()
        marker_path.write_text("No readable dual-R candidates were found.\n", encoding="utf-8")
        return
    if marker_path.exists():
        marker_path.unlink()
    _write_sdf(sdf_path, mols)


def _plot_status_counts(path: Path, rows: list[dict[str, Any]]) -> None:
    """Plot high-level candidate status counts."""
    counts = Counter(row["candidate_status"] for row in rows)
    labels = list(counts.keys()) or ["none"]
    values = [counts[label] for label in labels] or [0]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(labels, values, color="#4c78a8")
    ax.set_ylabel("molecule count")
    ax.set_title("Generated dual-R candidate status")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_failure_counts(path: Path, rows: list[dict[str, Any]]) -> None:
    """Plot detailed audit failure counts."""
    counts = Counter(row["failure_type"] for row in rows)
    labels = list(counts.keys()) or ["none"]
    values = [counts[label] for label in labels] or [0]
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    ax.bar(labels, values, color="#d95f59")
    ax.set_ylabel("molecule count")
    ax.set_title("Dual-R audit failure types")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_core_mapping(path: Path, summary: dict[str, Any]) -> None:
    """Plot core mapping and final success counts."""
    labels = ["total", "rdkit valid", "core found", "dual-R success"]
    values = [
        int(summary.get("total", 0)),
        int(summary.get("mapping_rdkit_valid", 0)),
        int(summary.get("core_found", 0)),
        int(summary.get("dual_r_success", 0)),
    ]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    bars = ax.bar(labels, values, color=["#8f8f8f", "#2f9e8f", "#4c78a8", "#5f5aa2"])
    ax.set_ylabel("molecule count")
    ax.set_title("Core mapping to final dual-R success")
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            str(int(bar.get_height())),
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_report(
    path: Path,
    args: argparse.Namespace,
    reference_sites,
    reference_spec,
    core_query,
    summary: dict[str, Any],
) -> None:
    """Write a Markdown report with inline figures."""
    lines = [
        "# Generated Dual-R Candidate Audit",
        "",
        f"- Input SDF: `{args.sdf}`",
        f"- Site reference SDF: `{args.site_reference_sdf}`",
        f"- Site reference index: `{int(args.site_reference_index)}`",
        f"- Compute MCS diagnostics: `{bool(args.compute_mcs)}`",
        "",
        "## Reference Sites",
        "",
        "| site | aromatic core atom | leaving O | leaving H |",
        "|---:|---:|---:|---:|",
    ]
    for index, site in enumerate(reference_sites):
        lines.append(
            f"| {index} | {site.core_atom_idx} | {site.oxygen_atom_idx} | "
            f"{'' if site.hydrogen_atom_idx is None else site.hydrogen_atom_idx} |"
        )
    lines.extend(
        [
            "",
            "## Reference Spec",
            "",
            "```json",
            json.dumps(reference_spec.to_dict(), ensure_ascii=False, indent=2),
            "```",
            "",
            "## Core Query",
            "",
            "```json",
            json.dumps(core_query.to_dict(), ensure_ascii=False, indent=2),
            "```",
            "",
            "## Summary",
            "",
            "```json",
            json.dumps(summary, ensure_ascii=False, indent=2),
            "```",
            "",
            "![Core mapping summary](core_mapping_summary.png)",
            "",
            "![Candidate status counts](candidate_status_counts.png)",
            "",
            "![Failure type counts](failure_type_counts.png)",
            "",
            "## Output Files",
            "",
            "- `generated_dual_r_candidates.csv`",
            "- `summary.json`",
            "- `core_mapping_summary.png`",
            "- `candidate_status_counts.png`",
            "- `failure_type_counts.png`",
        ]
    )
    if args.write_success_sdf:
        lines.append("- `successful_dual_r_candidates.sdf` or `successful_dual_r_candidates.none.txt`")
    if args.write_failure_sdf:
        lines.append("- `failed_dual_r_candidates.sdf` or `failed_dual_r_candidates.none.txt`")
    if args.write_audited_sdf:
        lines.append("- `audited_dual_r_candidates.sdf` or `audited_dual_r_candidates.none.txt`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _rate(count: int, total: int) -> float:
    """Return a rounded fraction."""
    if total <= 0:
        return 0.0
    return round(float(count) / float(total), 6)


if __name__ == "__main__":
    raise SystemExit(main())
