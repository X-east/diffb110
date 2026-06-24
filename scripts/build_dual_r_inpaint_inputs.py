"""Build fixed-core inputs for DiffSBDD dual-R inpainting."""

from __future__ import annotations

import argparse
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
    build_dual_substitution_spec_from_phenolic_sites,
    find_phenolic_oh_sites,
    substitute_dual_r_groups,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdf", required=True, type=Path, help="Reference honokiol SDF.")
    parser.add_argument("--molecule-index", type=int, default=0, help="Reference molecule index.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory.")
    parser.add_argument("--fixed-core-sdf", default="fixed_core_no_oh.sdf", help="Fixed-core SDF filename.")
    parser.add_argument("--left-site-index", type=int, default=0, help="Phenolic OH site index for left R site.")
    parser.add_argument("--right-site-index", type=int, default=1, help="Phenolic OH site index for right R site.")
    return parser.parse_args()


def main() -> int:
    """Build fixed-core SDF and metadata."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    reference_mol = _read_reference_mol(args.sdf, int(args.molecule_index))
    sites = find_phenolic_oh_sites(reference_mol)
    spec = build_dual_substitution_spec_from_phenolic_sites(
        sites,
        left_index=int(args.left_site_index),
        right_index=int(args.right_site_index),
    )
    fixed_result = substitute_dual_r_groups(
        reference_mol,
        spec,
        left_r_group=None,
        right_r_group=None,
        attach_left=False,
        attach_right=False,
        sanitize=True,
        add_hs=False,
    )
    fixed_core = Chem.RemoveHs(fixed_result.mol, sanitize=False)
    Chem.SanitizeMol(fixed_core)
    fixed_core.SetProp("_Name", "honokiol_fixed_core_no_oh")
    fixed_core.SetProp("dual_r_fixed_core_no_oh", "1")

    fixed_core_path = args.out_dir / args.fixed_core_sdf
    _write_sdf(fixed_core_path, fixed_core)

    summary = {
        "input_sdf": str(args.sdf),
        "molecule_index": int(args.molecule_index),
        "fixed_core_sdf": str(fixed_core_path),
        "phenolic_oh_sites": [site.to_dict() for site in sites],
        "dual_substitution_spec": spec.to_dict(),
        "fixed_core_adjusted_spec": fixed_result.adjusted_spec.to_dict(),
        "removed_atom_indices": [int(idx) for idx in fixed_result.removed_atom_indices],
        "reference_atoms": int(reference_mol.GetNumAtoms()),
        "reference_heavy_atoms": int(reference_mol.GetNumHeavyAtoms()),
        "fixed_core_atoms": int(fixed_core.GetNumAtoms()),
        "fixed_core_heavy_atoms": int(fixed_core.GetNumHeavyAtoms()),
    }
    _write_json(args.out_dir / "inpaint_input_summary.json", summary)
    _plot_atom_counts(args.out_dir / "inpaint_input_atom_counts.png", summary)
    _write_report(args.out_dir / "report.md", args, summary)

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


def _write_sdf(path: Path, mol: Chem.Mol) -> None:
    """Write one molecule to SDF."""
    writer = Chem.SDWriter(str(path))
    try:
        writer.write(mol)
    finally:
        writer.close()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON payload."""
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _plot_atom_counts(path: Path, summary: dict[str, Any]) -> None:
    """Plot reference versus fixed-core atom counts."""
    labels = ["reference atoms", "reference heavy", "fixed core heavy", "removed"]
    removed = int(summary["reference_atoms"]) - int(summary["fixed_core_atoms"])
    values = [
        int(summary["reference_atoms"]),
        int(summary["reference_heavy_atoms"]),
        int(summary["fixed_core_heavy_atoms"]),
        removed,
    ]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    bars = ax.bar(labels, values, color=["#8f8f8f", "#4c78a8", "#2f9e8f", "#d95f59"])
    ax.set_ylabel("atom count")
    ax.set_title("DiffSBDD inpaint fixed-core input")
    ax.tick_params(axis="x", rotation=20)
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


def _write_report(path: Path, args: argparse.Namespace, summary: dict[str, Any]) -> None:
    """Write Markdown report for fixed-core input construction."""
    lines = [
        "# Dual-R DiffSBDD Inpaint Inputs",
        "",
        f"- Reference SDF: `{args.sdf}`",
        f"- Molecule index: `{int(args.molecule_index)}`",
        f"- Fixed core SDF: `{args.fixed_core_sdf}`",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary, ensure_ascii=False, indent=2),
        "```",
        "",
        "![Inpaint input atom counts](inpaint_input_atom_counts.png)",
        "",
        "## Intended DiffSBDD Command Pattern",
        "",
        "```bash",
        "python DiffSBDD/inpaint.py DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt \\",
        "  --pdbfile data/4dxd_9pc.pdb \\",
        "  --ref_ligand analysis/honokiol_9pc_docking_20260617/docking/successful_docked_outputs.sdf \\",
        f"  --fix_atoms {args.out_dir / args.fixed_core_sdf} \\",
        "  --center ligand \\",
        "  --add_n_nodes <N> \\",
        "  --outfile <output.sdf>",
        "```",
        "",
        "## Output Files",
        "",
        f"- `{args.fixed_core_sdf}`",
        "- `inpaint_input_summary.json`",
        "- `inpaint_input_atom_counts.png`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
