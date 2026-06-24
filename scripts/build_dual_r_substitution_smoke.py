"""Build dual-R phenolic replacement smoke controls for a reference SDF."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from rdkit import Chem

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ddpo.constraints.dual_sidechain import (  # noqa: E402
    available_r_group_templates,
    build_dual_substitution_spec_from_phenolic_sites,
    find_phenolic_oh_sites,
    substitute_dual_r_groups,
)


DEFAULT_CONTROLS = (
    {
        "control_name": "positive_methyl",
        "left_r_group": "methyl",
        "right_r_group": "methyl",
        "attach_left": True,
        "attach_right": True,
        "expected_dual_success": True,
    },
    {
        "control_name": "positive_ethyl",
        "left_r_group": "ethyl",
        "right_r_group": "ethyl",
        "attach_left": True,
        "attach_right": True,
        "expected_dual_success": True,
    },
    {
        "control_name": "positive_propyl",
        "left_r_group": "propyl",
        "right_r_group": "propyl",
        "attach_left": True,
        "attach_right": True,
        "expected_dual_success": True,
    },
    {
        "control_name": "positive_allyl",
        "left_r_group": "allyl",
        "right_r_group": "allyl",
        "attach_left": True,
        "attach_right": True,
        "expected_dual_success": True,
    },
    {
        "control_name": "positive_methoxy",
        "left_r_group": "methoxy",
        "right_r_group": "methoxy",
        "attach_left": True,
        "attach_right": True,
        "expected_dual_success": True,
    },
    {
        "control_name": "negative_left_only_methyl",
        "left_r_group": "methyl",
        "right_r_group": None,
        "attach_left": True,
        "attach_right": False,
        "expected_dual_success": False,
    },
    {
        "control_name": "negative_right_only_methyl",
        "left_r_group": None,
        "right_r_group": "methyl",
        "attach_left": False,
        "attach_right": True,
        "expected_dual_success": False,
    },
    {
        "control_name": "negative_methyl_vs_ethyl",
        "left_r_group": "methyl",
        "right_r_group": "ethyl",
        "attach_left": True,
        "attach_right": True,
        "expected_dual_success": False,
    },
    {
        "control_name": "negative_ethyl_vs_methoxy",
        "left_r_group": "ethyl",
        "right_r_group": "methoxy",
        "attach_left": True,
        "attach_right": True,
        "expected_dual_success": False,
    },
    {
        "control_name": "negative_no_replacement",
        "left_r_group": None,
        "right_r_group": None,
        "attach_left": False,
        "attach_right": False,
        "expected_dual_success": False,
    },
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdf", required=True, type=Path, help="Reference SDF containing phenolic OH sites.")
    parser.add_argument("--molecule-index", type=int, default=0, help="Reference molecule index in --sdf.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for smoke controls.")
    parser.add_argument("--output-sdf", default="dual_r_controls.sdf", help="Output SDF filename.")
    parser.add_argument("--manifest-csv", default="dual_r_controls_manifest.csv", help="Manifest CSV filename.")
    parser.add_argument("--left-site-index", type=int, default=0, help="Phenolic OH site index for left R site.")
    parser.add_argument("--right-site-index", type=int, default=1, help="Phenolic OH site index for right R site.")
    parser.add_argument(
        "--max-branch-atoms",
        type=int,
        default=None,
        help="Optional branch-size limit stored in the generated substitution spec.",
    )
    return parser.parse_args()


def main() -> int:
    """Build default positive and negative substitution controls."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    reference_mol = _read_reference_mol(args.sdf, int(args.molecule_index))
    sites = find_phenolic_oh_sites(reference_mol)
    spec = build_dual_substitution_spec_from_phenolic_sites(
        sites,
        left_index=int(args.left_site_index),
        right_index=int(args.right_site_index),
        max_branch_atoms=args.max_branch_atoms,
    )

    output_sdf = args.out_dir / args.output_sdf
    manifest_csv = args.out_dir / args.manifest_csv
    rows = _write_controls(output_sdf, reference_mol, spec)

    _write_manifest(manifest_csv, rows)
    _write_summary_json(args.out_dir / "construction_summary.json", args, sites, spec, rows)
    _plot_expected_counts(args.out_dir / "expected_control_counts.png", rows)
    _write_report(args.out_dir / "report.md", args, sites, spec, rows)

    summary = {
        "output_sdf": str(output_sdf),
        "manifest_csv": str(manifest_csv),
        "total_controls": len(rows),
        "expected_success": sum(int(row["expected_dual_success"]) for row in rows),
        "expected_failure": sum(int(not row["expected_dual_success"]) for row in rows),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _read_reference_mol(path: Path, molecule_index: int) -> Chem.Mol:
    """Read one reference molecule from an SDF file."""
    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
    mols = [mol for mol in supplier if mol is not None]
    if not mols:
        raise ValueError(f"No molecules read from {path}")
    if molecule_index < 0 or molecule_index >= len(mols):
        raise ValueError(f"molecule index {molecule_index} out of range for {len(mols)} molecules")
    mol = Chem.Mol(mols[molecule_index])
    Chem.SanitizeMol(mol)
    return mol


def _write_controls(
    output_sdf: Path,
    reference_mol: Chem.Mol,
    spec,
) -> list[dict[str, Any]]:
    """Write all default controls to SDF and return manifest rows."""
    rows: list[dict[str, Any]] = []
    writer = Chem.SDWriter(str(output_sdf))
    try:
        for index, control in enumerate(DEFAULT_CONTROLS):
            result = substitute_dual_r_groups(
                reference_mol,
                spec,
                left_r_group=control["left_r_group"],
                right_r_group=control["right_r_group"],
                attach_left=bool(control["attach_left"]),
                attach_right=bool(control["attach_right"]),
            )
            mol = result.mol
            control_name = str(control["control_name"])
            mol.SetProp("_Name", control_name)
            mol.SetProp("control_name", control_name)
            mol.SetProp("expected_dual_success", str(int(bool(control["expected_dual_success"]))))
            mol.SetProp("left_r_group", str(control["left_r_group"] or ""))
            mol.SetProp("right_r_group", str(control["right_r_group"] or ""))
            mol.SetProp("attach_left", str(int(bool(control["attach_left"]))))
            mol.SetProp("attach_right", str(int(bool(control["attach_right"]))))
            writer.write(mol)

            row = {
                "control_index": index,
                "control_name": control_name,
                "expected_dual_success": int(bool(control["expected_dual_success"])),
                "canonical_smiles": Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True),
            }
            row.update(result.to_manifest_row())
            rows.append(row)
    finally:
        writer.close()
    return rows


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write construction manifest CSV."""
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_json(
    path: Path,
    args: argparse.Namespace,
    sites,
    spec,
    rows: list[dict[str, Any]],
) -> None:
    """Write JSON construction summary."""
    payload = {
        "input_sdf": str(args.sdf),
        "molecule_index": int(args.molecule_index),
        "available_r_group_templates": list(available_r_group_templates()),
        "phenolic_oh_sites": [site.to_dict() for site in sites],
        "dual_substitution_spec": spec.to_dict(),
        "controls": rows,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _plot_expected_counts(path: Path, rows: list[dict[str, Any]]) -> None:
    """Plot expected positive/negative control counts."""
    positive = sum(int(row["expected_dual_success"]) for row in rows)
    negative = len(rows) - positive
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    bars = ax.bar(["expected success", "expected failure"], [positive, negative], color=["#2f9e8f", "#d95f59"])
    ax.set_ylabel("control molecules")
    ax.set_title("Dual-R smoke control design")
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
    sites,
    spec,
    rows: list[dict[str, Any]],
) -> None:
    """Write a Markdown report for the constructed smoke controls."""
    lines = [
        "# Dual-R Substitution Smoke Controls",
        "",
        f"- Reference SDF: `{args.sdf}`",
        f"- Molecule index: `{int(args.molecule_index)}`",
        f"- Output SDF: `{args.output_sdf}`",
        f"- Manifest: `{args.manifest_csv}`",
        "",
        "## Replacement Sites",
        "",
        "| site | aromatic core atom | leaving O | leaving H |",
        "|---:|---:|---:|---:|",
    ]
    for index, site in enumerate(sites):
        lines.append(
            f"| {index} | {site.core_atom_idx} | {site.oxygen_atom_idx} | "
            f"{'' if site.hydrogen_atom_idx is None else site.hydrogen_atom_idx} |"
        )
    lines.extend(
        [
            "",
            "## Dual Substitution Spec",
            "",
            "```json",
            json.dumps(spec.to_dict(), ensure_ascii=False, indent=2),
            "```",
            "",
            "## Control Matrix",
            "",
            "| control | left R | right R | expected success | heavy atoms |",
            "|---|---|---|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['control_name']} | {row['left_r_group']} | {row['right_r_group']} | "
            f"{row['expected_dual_success']} | {row['num_heavy_atoms']} |"
        )
    lines.extend(
        [
            "",
            "![Expected control counts](expected_control_counts.png)",
            "",
            "## Output Files",
            "",
            f"- `{args.output_sdf}`",
            f"- `{args.manifest_csv}`",
            "- `construction_summary.json`",
            "- `expected_control_counts.png`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
