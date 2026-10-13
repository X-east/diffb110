"""Generate one DiffSBDD sidechain and copy it onto the symmetric anchor.

Pipeline:
1. Run DiffSBDD fixed-core inpaint to generate candidate molecules (core plus
   freely grown sidechains) for the 4dxd / honokiol task.
2. For each candidate, take the sidechain branch attached to one anchor (R1),
   replicate it onto the symmetric anchor (R2) with
   :func:`copy_sidechain_branch_to_anchor`, and keep molecules that pass the
   intramolecular clash check.

This realizes the "generate one side, copy to the other" strategy as a pure
post-processing step, without any in-diffusion symmetry constraint. Protein
clashes are intentionally ignored because the docking backend performs energy
optimization during scoring.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rdkit import Chem

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ddpo.constraints.dual_sidechain import (  # noqa: E402
    copy_sidechain_branch_to_anchor,
    extract_sidechain_branch,
)

DEFAULT_CHECKPOINT = "DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt"
DEFAULT_PDBFILE = "data/4dxd_9pc.pdb"
DEFAULT_REF_LIGAND = (
    "analysis/honokiol_9pc_docking_20260617/docking/successful_docked_outputs.sdf"
)
DEFAULT_FIXED_CORE = (
    "analysis/dual_sidechain_honokiol_9pc_20260617/inpaint_fixed_core_inputs/fixed_core_c0oh_only.sdf"
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, type=Path)
    parser.add_argument("--pdbfile", default=DEFAULT_PDBFILE, type=Path)
    parser.add_argument("--ref_ligand", default=DEFAULT_REF_LIGAND, type=Path)
    parser.add_argument("--fixed_core", default=DEFAULT_FIXED_CORE, type=Path)
    parser.add_argument("--anchor1", type=int, default=0, help="R1 core atom index.")
    parser.add_argument("--anchor2", type=int, default=11, help="R2 core atom index.")
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--add_n_nodes", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=50)
    parser.add_argument("--resamplings", type=int, default=5)
    parser.add_argument("--reconstruction_method", default="distance")
    parser.add_argument("--clash_fraction", type=float, default=0.70)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument(
        "--inpaint_sdf",
        type=Path,
        default=None,
        help="Skip DiffSBDD generation and post-process this existing SDF instead.",
    )
    return parser.parse_args()


def run_inpaint(args: argparse.Namespace, inpaint_out: Path) -> None:
    """Run DiffSBDD fixed-core inpaint as a subprocess (matches prior working cmd)."""
    cmd = (
        f"CUDA_VISIBLE_DEVICES={args.gpu} MPLCONFIGDIR=/tmp/matplotlib-diffb110 "
        f"{sys.executable} DiffSBDD/inpaint.py {args.checkpoint} "
        f"--pdbfile {args.pdbfile} --ref_ligand {args.ref_ligand} "
        f"--fix_atoms {args.fixed_core} --center ligand "
        f"--add_n_nodes {args.add_n_nodes} --n_samples {args.n_samples} "
        f"--timesteps {args.timesteps} --resamplings {args.resamplings} "
        f"--reconstruction_method {args.reconstruction_method} "
        f"--sanitize --outfile {inpaint_out}"
    )
    print(f"[inpaint] running:\n{cmd}\n", flush=True)
    subprocess.run(cmd, shell=True, cwd=str(PROJECT_ROOT), check=True)


def _read_molecules(path: Path) -> list[Chem.Mol]:
    """Read all molecules from an SDF file."""
    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
    return [mol for mol in supplier if mol is not None]


def postprocess(
    mols: list[Chem.Mol],
    *,
    anchor1: int,
    anchor2: int,
    clash_fraction: float,
) -> list[dict[str, Any]]:
    """Copy one sidechain onto the other anchor for each candidate molecule."""
    rows: list[dict[str, Any]] = []
    for idx, mol in enumerate(mols):
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            rows.append(
                {
                    "molecule_index": idx,
                    "status": "sanitize_failed_input",
                    "source_side": "",
                    "r1_atom_count": 0,
                    "r2_atom_count": 0,
                    "clash": 0,
                    "min_clash_distance": float("inf"),
                    "attachment_bond_length": float("nan"),
                    "attachment_angle_1_deg": float("nan"),
                    "attachment_angle_2_deg": float("nan"),
                    "attachment_out_of_plane_deg": float("nan"),
                    "canonical_smiles": "",
                }
            )
            continue

        left_ok = extract_sidechain_branch(
            mol, core_atom_idx=anchor1, label="R1"
        ).extractable
        right_ok = extract_sidechain_branch(
            mol, core_atom_idx=anchor2, label="R2"
        ).extractable
        source = "R1" if left_ok else ("R2" if right_ok else "")

        if source == "R1":
            result = copy_sidechain_branch_to_anchor(
                mol,
                anchor1_idx=anchor1,
                anchor2_idx=anchor2,
                clash_fraction=clash_fraction,
            )
        elif source == "R2":
            result = copy_sidechain_branch_to_anchor(
                mol,
                anchor1_idx=anchor2,
                anchor2_idx=anchor1,
                clash_fraction=clash_fraction,
            )
        else:
            result = copy_sidechain_branch_to_anchor(
                mol,
                anchor1_idx=anchor1,
                anchor2_idx=anchor2,
                clash_fraction=clash_fraction,
            )

        smiles = ""
        if result.mol is not None:
            try:
                smiles = Chem.MolToSmiles(result.mol)
            except Exception:
                smiles = ""
        rows.append(
            {
                "molecule_index": idx,
                "status": result.status,
                "source_side": source,
                "r1_atom_count": result.r1_atom_count,
                "r2_atom_count": result.r2_atom_count,
                "clash": int(result.clash),
                "min_clash_distance": round(float(result.min_clash_distance), 4),
                "attachment_bond_length": round(float(result.attachment_bond_length), 6),
                "attachment_angle_1_deg": round(float(result.attachment_angle_1_deg), 6),
                "attachment_angle_2_deg": round(float(result.attachment_angle_2_deg), 6),
                "attachment_out_of_plane_deg": round(
                    float(result.attachment_out_of_plane_deg), 6
                ),
                "canonical_smiles": smiles,
            }
        )
    return rows


def main() -> int:
    """Generate and symmetrize dual-sidechain candidates."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    inpaint_out = args.out_dir / "inpaint_raw.sdf"

    if args.inpaint_sdf is not None:
        print(f"[postprocess] using existing inpaint SDF: {args.inpaint_sdf}", flush=True)
        inpaint_out = args.inpaint_sdf
    else:
        run_inpaint(args, inpaint_out)

    mols = _read_molecules(inpaint_out)
    print(f"[postprocess] read {len(mols)} inpaint molecules", flush=True)

    rows = postprocess(
        mols,
        anchor1=args.anchor1,
        anchor2=args.anchor2,
        clash_fraction=args.clash_fraction,
    )

    success = [r for r in rows if r["status"] == "success"]
    clashes = [r for r in rows if r["status"] == "clash"]
    print(
        f"[postprocess] success={len(success)} clash={len(clashes)} "
        f"other={len(rows) - len(success) - len(clashes)}",
        flush=True,
    )

    # Write symmetric candidate SDF (success only).
    writer = Chem.SDWriter(str(args.out_dir / "symmetric_candidates.sdf"))
    try:
        for mol_idx, row in zip(range(len(mols)), rows):
            if row["status"] != "success":
                continue
            out_mol = mols[mol_idx]
            result = copy_sidechain_branch_to_anchor(
                out_mol,
                anchor1_idx=args.anchor1 if row["source_side"] == "R1" else args.anchor2,
                anchor2_idx=args.anchor2 if row["source_side"] == "R1" else args.anchor1,
                clash_fraction=args.clash_fraction,
            )
            if result.mol is not None:
                result.mol.SetProp("_Name", f"dual_sidechain_{mol_idx}")
                result.mol.SetProp("source_side", row["source_side"])
                writer.write(result.mol)
    finally:
        writer.close()

    # Write CSV summary.
    csv_path = args.out_dir / "generation_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer_csv = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else [])
        writer_csv.writeheader()
        for row in rows:
            writer_csv.writerow(row)

    summary = {
        "inpaint_sdf": str(inpaint_out),
        "n_input": len(mols),
        "n_success": len(success),
        "n_clash": len(clashes),
        "n_other": len(rows) - len(success) - len(clashes),
        "anchor1": args.anchor1,
        "anchor2": args.anchor2,
        "clash_fraction": args.clash_fraction,
        "add_n_nodes": args.add_n_nodes,
        "n_samples": args.n_samples,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    _write_report(args.out_dir / "report.md", args, summary, rows)
    print(f"[done] wrote {len(success)} symmetric candidates to {args.out_dir}", flush=True)
    return 0


def _write_report(
    path: Path,
    args: argparse.Namespace,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    """Write a Markdown report for the generation run."""
    lines = [
        "# DiffSBDD Dual-Sidechain Generation (generate-one-copy-other)",
        "",
        f"- Input inpaint SDF: `{summary['inpaint_sdf']}`",
        f"- Anchor1 (R1) core idx: `{args.anchor1}`",
        f"- Anchor2 (R2) core idx: `{args.anchor2}`",
        f"- add_n_nodes: `{args.add_n_nodes}`  n_samples: `{args.n_samples}`",
        f"- clash_fraction: `{summary['clash_fraction']}`",
        "",
        "## Summary",
        "",
        f"- input molecules: {summary['n_input']}",
        f"- symmetric success: {summary['n_success']}",
        f"- clash (rejected): {summary['n_clash']}",
        f"- other (no extractable sidechain): {summary['n_other']}",
        "",
        "## Per-molecule",
        "",
        "| idx | source | status | R1 atoms | R2 atoms | clash | min_dist | bond A | angles deg | oop deg | SMILES |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['molecule_index']} | {row['source_side']} | {row['status']} "
            f"| {row['r1_atom_count']} | {row['r2_atom_count']} "
            f"| {row['clash']} | {row['min_clash_distance']} "
            f"| {row['attachment_bond_length']} "
            f"| {row['attachment_angle_1_deg']}/{row['attachment_angle_2_deg']} "
            f"| {row['attachment_out_of_plane_deg']} | {row['canonical_smiles']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
