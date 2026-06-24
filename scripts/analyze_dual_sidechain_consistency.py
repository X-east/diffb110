"""Analyze dual aromatic-sidechain consistency for generated or docked SDF files."""

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
    DualSubstitutionSpec,
    audit_dual_sidechain,
    build_dual_substitution_spec_from_phenolic_sites,
    find_phenolic_oh_sites,
    summarize_audit_results,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdf", required=True, type=Path, help="Input SDF with generated or docked ligands.")
    parser.add_argument(
        "--site-reference-sdf",
        type=Path,
        default=None,
        help="Optional reference SDF used only to detect the original phenolic OH replacement sites.",
    )
    parser.add_argument(
        "--site-reference-index",
        type=int,
        default=0,
        help="Molecule index within --site-reference-sdf used for site detection.",
    )
    parser.add_argument("--protein", type=Path, default=None, help="Optional protein PDB for nearest-residue context.")
    parser.add_argument(
        "--include-hetatm",
        action="store_true",
        help="Include HETATM records when computing nearest protein context. Default uses ATOM records only.",
    )
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for audit artifacts.")
    parser.add_argument(
        "--mode",
        choices=("auto_phenolic_oh", "manual_core"),
        default="auto_phenolic_oh",
        help="How to define the two substituted aromatic core atoms.",
    )
    parser.add_argument("--left-core-idx", type=int, default=None, help="Manual left aromatic core atom index.")
    parser.add_argument("--right-core-idx", type=int, default=None, help="Manual right aromatic core atom index.")
    parser.add_argument("--left-leaving-idx", type=int, default=None, help="Optional left leaving atom index.")
    parser.add_argument("--right-leaving-idx", type=int, default=None, help="Optional right leaving atom index.")
    parser.add_argument("--left-site-index", type=int, default=0, help="Phenolic OH site index for left R site.")
    parser.add_argument("--right-site-index", type=int, default=1, help="Phenolic OH site index for right R site.")
    parser.add_argument(
        "--ignore-reference-oh",
        action="store_true",
        help="Ignore detected OH leaving atoms when extracting branches. Use for generated replacement molecules.",
    )
    parser.add_argument(
        "--target-preserves-reference-atom-indices",
        action="store_true",
        help=(
            "When --site-reference-sdf is used, keep original leaving atom indices during target audit. "
            "Use only if the target SDF preserves reference atom numbering."
        ),
    )
    parser.add_argument(
        "--max-branch-atoms",
        type=int,
        default=None,
        help="Optional maximum branch atom count before marking a branch too large.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the audit workflow."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    mols = _read_sdf(args.sdf)
    if not mols:
        raise ValueError(f"No molecules read from {args.sdf}")

    reference_mol = _reference_mol_for_sites(args, mols)
    Chem.SanitizeMol(reference_mol)
    sites = find_phenolic_oh_sites(reference_mol)
    site_spec = _build_spec(args, sites)
    audit_spec = _audit_spec_for_target(args, site_spec)

    results = [
        audit_dual_sidechain(
            mol,
            audit_spec,
            molecule_index=idx,
            ignore_leaving_atoms=bool(args.ignore_reference_oh),
        )
        for idx, mol in enumerate(mols)
    ]
    summary = summarize_audit_results(results)
    protein_context = _protein_context(args.protein, reference_mol, site_spec, include_hetatm=bool(args.include_hetatm))

    _write_sites(args.out_dir / "detected_sites.json", sites, site_spec, audit_spec, protein_context)
    _write_audit_csv(args.out_dir / "dual_sidechain_audit.csv", results)
    _write_summary(args.out_dir / "summary.json", summary, protein_context, args)
    _plot_failure_modes(args.out_dir / "failure_modes.png", results)
    _plot_site_context(args.out_dir / "site_context_distances.png", protein_context)
    _write_report(args.out_dir / "report.md", args, sites, site_spec, audit_spec, summary, protein_context)

    print(f"wrote {args.out_dir}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _read_sdf(path: Path) -> list[Chem.Mol]:
    """Read all molecules from an SDF file."""
    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
    mols = [mol for mol in supplier if mol is not None]
    for mol in mols:
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            # Keep invalid molecules for explicit audit failure accounting.
            pass
    return mols


def _reference_mol_for_sites(args: argparse.Namespace, input_mols: list[Chem.Mol]) -> Chem.Mol:
    """Return the molecule used to define original replacement sites."""
    if args.site_reference_sdf is None:
        return Chem.Mol(input_mols[0])
    reference_mols = _read_sdf(args.site_reference_sdf)
    if not reference_mols:
        raise ValueError(f"No molecules read from site reference SDF: {args.site_reference_sdf}")
    index = int(args.site_reference_index)
    if index < 0 or index >= len(reference_mols):
        raise ValueError(
            f"--site-reference-index {index} is out of range for {len(reference_mols)} reference molecules"
        )
    return Chem.Mol(reference_mols[index])


def _build_spec(args: argparse.Namespace, sites) -> DualSubstitutionSpec:
    """Build substitution spec from arguments."""
    if args.mode == "auto_phenolic_oh":
        return build_dual_substitution_spec_from_phenolic_sites(
            sites,
            left_index=int(args.left_site_index),
            right_index=int(args.right_site_index),
            max_branch_atoms=args.max_branch_atoms,
        )
    if args.left_core_idx is None or args.right_core_idx is None:
        raise ValueError("manual_core mode requires --left-core-idx and --right-core-idx")
    return DualSubstitutionSpec(
        left_core_idx=int(args.left_core_idx),
        right_core_idx=int(args.right_core_idx),
        left_leaving_atom_idx=args.left_leaving_idx,
        right_leaving_atom_idx=args.right_leaving_idx,
        max_branch_atoms=args.max_branch_atoms,
    )


def _audit_spec_for_target(
    args: argparse.Namespace,
    site_spec: DualSubstitutionSpec,
) -> DualSubstitutionSpec:
    """Return the target-molecule audit spec derived from the reference site spec."""
    if (
        args.site_reference_sdf is None
        or not args.ignore_reference_oh
        or args.target_preserves_reference_atom_indices
    ):
        return site_spec
    return DualSubstitutionSpec(
        left_core_idx=site_spec.left_core_idx,
        right_core_idx=site_spec.right_core_idx,
        left_leaving_atom_idx=None,
        right_leaving_atom_idx=None,
        left_label=site_spec.left_label,
        right_label=site_spec.right_label,
        max_branch_atoms=site_spec.max_branch_atoms,
    )


def _write_sites(
    path: Path,
    sites,
    site_spec: DualSubstitutionSpec,
    audit_spec: DualSubstitutionSpec,
    protein_context: dict[str, Any],
) -> None:
    """Write detected site metadata."""
    payload = {
        "phenolic_oh_sites": [site.to_dict() for site in sites],
        "site_reference_spec": site_spec.to_dict(),
        "target_audit_spec": audit_spec.to_dict(),
        "protein_context": protein_context,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_audit_csv(path: Path, results) -> None:
    """Write per-molecule audit rows."""
    rows = [result.to_dict() for result in results]
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(
    path: Path,
    summary: dict[str, Any],
    protein_context: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    """Write JSON summary."""
    payload = {
        "input_sdf": str(args.sdf),
        "site_reference_sdf": None if args.site_reference_sdf is None else str(args.site_reference_sdf),
        "site_reference_index": int(args.site_reference_index),
        "protein": None if args.protein is None else str(args.protein),
        "ignore_reference_oh": bool(args.ignore_reference_oh),
        "target_preserves_reference_atom_indices": bool(args.target_preserves_reference_atom_indices),
        "summary": summary,
        "protein_context": protein_context,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _plot_failure_modes(path: Path, results) -> None:
    """Plot failure mode counts."""
    counts = Counter(result.failure_type for result in results)
    labels = list(counts.keys()) or ["none"]
    values = [counts[label] for label in labels] or [0]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(labels, values, color="#4c78a8")
    ax.set_ylabel("molecule count")
    ax.set_title("Dual-sidechain audit failure modes")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_site_context(path: Path, protein_context: dict[str, Any]) -> None:
    """Plot nearest protein distances for the two R sites when available."""
    if not protein_context:
        return
    labels = []
    values = []
    for key in ("left_core", "right_core", "left_root", "right_root"):
        item = protein_context.get(key)
        if not item:
            continue
        labels.append(key)
        values.append(float(item.get("nearest_distance", 0.0)))
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(labels, values, color="#2f9e8f")
    ax.set_ylabel("nearest protein heavy atom distance (A)")
    ax.set_title("Protein context around substitution sites")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _protein_context(
    protein_path: Path | None,
    mol: Chem.Mol,
    spec: DualSubstitutionSpec,
    *,
    include_hetatm: bool = False,
) -> dict[str, Any]:
    """Return nearest protein atom context around core/root positions."""
    if protein_path is None:
        return {}
    atoms = _read_protein_atoms(protein_path, include_hetatm=include_hetatm)
    if not atoms or mol.GetNumConformers() == 0:
        return {}
    conf = mol.GetConformer()
    context = {}
    index_items = {
        "left_core": spec.left_core_idx,
        "right_core": spec.right_core_idx,
        "left_root": spec.left_leaving_atom_idx,
        "right_root": spec.right_leaving_atom_idx,
    }
    for label, atom_idx in index_items.items():
        if atom_idx is None or atom_idx < 0 or atom_idx >= mol.GetNumAtoms():
            continue
        pos = conf.GetAtomPosition(int(atom_idx))
        context[label] = _nearest_protein_atom((pos.x, pos.y, pos.z), atoms)
    return context


def _read_protein_atoms(protein_path: Path, *, include_hetatm: bool = False) -> list[dict[str, Any]]:
    """Read protein heavy atoms from a PDB file without adding a BioPython dependency."""
    atoms = []
    with protein_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("HETATM") and not include_hetatm:
                continue
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            element = line[76:78].strip() or line[12:16].strip()[0]
            if element.upper() == "H":
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            atoms.append(
                {
                    "atom_name": line[12:16].strip(),
                    "resname": line[17:20].strip(),
                    "chain": line[21].strip(),
                    "resi": line[22:26].strip(),
                    "element": element,
                    "coord": (x, y, z),
                }
            )
    return atoms


def _nearest_protein_atom(coord: tuple[float, float, float], atoms: list[dict[str, Any]]) -> dict[str, Any]:
    """Find nearest protein heavy atom to one ligand coordinate."""
    best_atom = None
    best_distance = float("inf")
    x, y, z = coord
    for atom in atoms:
        ax, ay, az = atom["coord"]
        distance = ((x - ax) ** 2 + (y - ay) ** 2 + (z - az) ** 2) ** 0.5
        if distance < best_distance:
            best_distance = distance
            best_atom = atom
    if best_atom is None:
        return {}
    return {
        "nearest_distance": round(best_distance, 4),
        "atom_name": best_atom["atom_name"],
        "resname": best_atom["resname"],
        "chain": best_atom["chain"],
        "resi": best_atom["resi"],
        "element": best_atom["element"],
    }


def _write_report(
    path: Path,
    args: argparse.Namespace,
    sites,
    site_spec: DualSubstitutionSpec,
    audit_spec: DualSubstitutionSpec,
    summary: dict[str, Any],
    protein_context: dict[str, Any],
) -> None:
    """Write a Markdown report with inline plots."""
    lines = [
        "# Dual Sidechain Consistency Audit",
        "",
        f"- Input SDF: `{args.sdf}`",
        f"- Site reference SDF: `{args.site_reference_sdf}`"
        if args.site_reference_sdf
        else "- Site reference SDF: input first molecule",
        f"- Site reference index: `{int(args.site_reference_index)}`",
        f"- Protein: `{args.protein}`" if args.protein else "- Protein: not provided",
        f"- Mode: `{args.mode}`",
        f"- Ignore reference OH leaving atoms: `{bool(args.ignore_reference_oh)}`",
        f"- Target preserves reference atom indices: `{bool(args.target_preserves_reference_atom_indices)}`",
        f"- Include HETATM in protein context: `{bool(args.include_hetatm)}`",
        "",
        "## Detected Phenolic OH Sites",
        "",
        "| site | aromatic core atom | oxygen leaving atom | hydrogen atom |",
        "|---:|---:|---:|---:|",
    ]
    for idx, site in enumerate(sites):
        lines.append(
            f"| {idx} | {site.core_atom_idx} | {site.oxygen_atom_idx} | "
            f"{'' if site.hydrogen_atom_idx is None else site.hydrogen_atom_idx} |"
        )
    lines.extend([
        "",
        "## Site Reference Spec",
        "",
        "```json",
        json.dumps(site_spec.to_dict(), ensure_ascii=False, indent=2),
        "```",
        "",
        "## Target Audit Spec",
        "",
        "```json",
        json.dumps(audit_spec.to_dict(), ensure_ascii=False, indent=2),
        "```",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary, ensure_ascii=False, indent=2),
        "```",
        "",
        "![Failure modes](failure_modes.png)",
        "",
    ])
    if protein_context:
        lines.extend([
            "## Protein Context",
            "",
            "```json",
            json.dumps(protein_context, ensure_ascii=False, indent=2),
            "```",
            "",
            "![Protein site context](site_context_distances.png)",
            "",
        ])
    lines.extend([
        "## Output Files",
        "",
        "- `detected_sites.json`",
        "- `dual_sidechain_audit.csv`",
        "- `summary.json`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
