"""Post-process DiffSBDD fixed-core outputs with explicit dual-R attachments."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem, Geometry


DEFAULT_ALLOWED_ROOT_ELEMENTS = ("C", "N", "O", "S")
DEFAULT_TARGET_BOND_LENGTHS = {
    "C": 1.50,
    "N": 1.40,
    "O": 1.36,
    "S": 1.76,
}


@dataclass(frozen=True)
class AttachmentAttempt:
    """Bookkeeping for one molecule post-processing attempt."""

    molecule_index: int
    success: bool
    failure_type: str
    left_root_idx: int | None = None
    right_root_idx: int | None = None
    left_distance_before: float | None = None
    right_distance_before: float | None = None
    left_distance_after: float | None = None
    right_distance_after: float | None = None
    left_root_symbol: str = ""
    right_root_symbol: str = ""
    sanitize_success: bool = False
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return flat manifest fields."""
        return {
            "molecule_index": self.molecule_index,
            "success": int(self.success),
            "failure_type": self.failure_type,
            "left_root_idx": "" if self.left_root_idx is None else self.left_root_idx,
            "right_root_idx": "" if self.right_root_idx is None else self.right_root_idx,
            "left_root_symbol": self.left_root_symbol,
            "right_root_symbol": self.right_root_symbol,
            "left_distance_before": _format_float(self.left_distance_before),
            "right_distance_before": _format_float(self.right_distance_before),
            "left_distance_after": _format_float(self.left_distance_after),
            "right_distance_after": _format_float(self.right_distance_after),
            "sanitize_success": int(self.sanitize_success),
            "errors": ";".join(self.errors),
        }


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdf", required=True, type=Path, help="Input generated fixed-core SDF.")
    parser.add_argument("--fixed-core-sdf", required=True, type=Path, help="Fixed-core SDF used for inpainting.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory.")
    parser.add_argument("--out-sdf-name", default=None, help="Output SDF filename.")
    parser.add_argument("--left-core-idx", type=int, default=0, help="Left attachment core atom index.")
    parser.add_argument("--right-core-idx", type=int, default=11, help="Right attachment core atom index.")
    parser.add_argument(
        "--mode",
        choices=("strict", "snap"),
        default="strict",
        help="strict only accepts near atoms; snap translates selected generated components before bonding.",
    )
    parser.add_argument("--max-attach-distance", type=float, default=1.9, help="Strict attachment distance cutoff.")
    parser.add_argument("--max-snap-distance", type=float, default=5.5, help="Maximum pre-snap root distance.")
    parser.add_argument(
        "--allowed-root-elements",
        default=",".join(DEFAULT_ALLOWED_ROOT_ELEMENTS),
        help="Comma-separated root atom symbols allowed for C0/C11 attachment.",
    )
    parser.add_argument(
        "--allow-same-generated-component",
        action="store_true",
        help="Allow both roots to belong to the same generated component.",
    )
    return parser.parse_args()


def main() -> int:
    """Run attachment-aware post-processing."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    allowed_root_elements = tuple(item.strip() for item in args.allowed_root_elements.split(",") if item.strip())
    fixed_core = _read_one_mol(args.fixed_core_sdf)
    n_fixed = fixed_core.GetNumAtoms()
    mols = _read_mols(args.sdf)

    processed_mols: list[Chem.Mol] = []
    attempts: list[AttachmentAttempt] = []
    for molecule_index, mol in enumerate(mols):
        if mol is None:
            attempts.append(AttachmentAttempt(molecule_index, False, "mol_none"))
            continue
        processed, attempt = postprocess_one_mol(
            mol,
            molecule_index=molecule_index,
            n_fixed=n_fixed,
            left_core_idx=int(args.left_core_idx),
            right_core_idx=int(args.right_core_idx),
            mode=str(args.mode),
            max_attach_distance=float(args.max_attach_distance),
            max_snap_distance=float(args.max_snap_distance),
            allowed_root_elements=allowed_root_elements,
            allow_same_generated_component=bool(args.allow_same_generated_component),
        )
        attempts.append(attempt)
        if processed is not None and attempt.success:
            processed_mols.append(processed)

    out_sdf_name = args.out_sdf_name or f"attachment_{args.mode}.sdf"
    out_sdf = args.out_dir / out_sdf_name
    _write_sdf(out_sdf, processed_mols)
    _write_csv(args.out_dir / f"attachment_{args.mode}_manifest.csv", [attempt.to_dict() for attempt in attempts])
    summary = _summarize_attempts(args, attempts, out_sdf, n_fixed, allowed_root_elements)
    _write_json(args.out_dir / f"attachment_{args.mode}_summary.json", summary)
    _write_report(args.out_dir / f"attachment_{args.mode}_report.md", summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def postprocess_one_mol(
    mol: Chem.Mol,
    *,
    molecule_index: int,
    n_fixed: int,
    left_core_idx: int,
    right_core_idx: int,
    mode: str,
    max_attach_distance: float,
    max_snap_distance: float,
    allowed_root_elements: tuple[str, ...],
    allow_same_generated_component: bool,
) -> tuple[Chem.Mol | None, AttachmentAttempt]:
    """Try to create explicit C0/C11 attachments for one generated molecule."""
    errors: list[str] = []
    if mol.GetNumConformers() == 0:
        return None, AttachmentAttempt(molecule_index, False, "missing_conformer")
    if mol.GetNumAtoms() <= n_fixed:
        return None, AttachmentAttempt(molecule_index, False, "no_generated_atoms")
    if left_core_idx >= n_fixed or right_core_idx >= n_fixed:
        return None, AttachmentAttempt(molecule_index, False, "core_index_out_of_fixed_range")

    working = Chem.Mol(mol)
    generated_indices = tuple(range(n_fixed, working.GetNumAtoms()))
    component_by_atom = _generated_component_by_atom(working, generated_indices)
    left_root, left_distance = _select_root(
        working,
        core_idx=left_core_idx,
        generated_indices=generated_indices,
        allowed_root_elements=allowed_root_elements,
    )
    if left_root is None or left_distance is None:
        return None, AttachmentAttempt(molecule_index, False, "left_no_allowed_root")
    right_root, right_distance = _select_root(
        working,
        core_idx=right_core_idx,
        generated_indices=generated_indices,
        allowed_root_elements=allowed_root_elements,
        exclude_atoms={left_root},
        exclude_components=(
            set()
            if allow_same_generated_component
            else {component_by_atom[left_root]}
        ),
        component_by_atom=component_by_atom,
    )
    if right_root is None or right_distance is None:
        return None, AttachmentAttempt(
            molecule_index,
            False,
            "right_no_allowed_distinct_root",
            left_root_idx=left_root,
            left_distance_before=left_distance,
            left_root_symbol=working.GetAtomWithIdx(left_root).GetSymbol(),
        )

    if mode == "strict" and (left_distance > max_attach_distance or right_distance > max_attach_distance):
        return None, AttachmentAttempt(
            molecule_index,
            False,
            "root_distance_above_strict_cutoff",
            left_root_idx=left_root,
            right_root_idx=right_root,
            left_distance_before=left_distance,
            right_distance_before=right_distance,
            left_root_symbol=working.GetAtomWithIdx(left_root).GetSymbol(),
            right_root_symbol=working.GetAtomWithIdx(right_root).GetSymbol(),
        )
    if mode == "snap" and (left_distance > max_snap_distance or right_distance > max_snap_distance):
        return None, AttachmentAttempt(
            molecule_index,
            False,
            "root_distance_above_snap_cutoff",
            left_root_idx=left_root,
            right_root_idx=right_root,
            left_distance_before=left_distance,
            right_distance_before=right_distance,
            left_root_symbol=working.GetAtomWithIdx(left_root).GetSymbol(),
            right_root_symbol=working.GetAtomWithIdx(right_root).GetSymbol(),
        )

    editable = Chem.RWMol(working)
    if mode == "snap":
        _snap_component_to_bond_length(
            editable,
            core_idx=left_core_idx,
            root_idx=left_root,
            component_atoms=_component_atoms(component_by_atom, left_root),
        )
        _snap_component_to_bond_length(
            editable,
            core_idx=right_core_idx,
            root_idx=right_root,
            component_atoms=_component_atoms(component_by_atom, right_root),
        )

    _add_single_bond_if_missing(editable, left_core_idx, left_root)
    _add_single_bond_if_missing(editable, right_core_idx, right_root)
    processed = editable.GetMol()
    _annotate_attempt_properties(
        processed,
        molecule_index=molecule_index,
        mode=mode,
        left_root=left_root,
        right_root=right_root,
        left_distance_before=left_distance,
        right_distance_before=right_distance,
    )
    left_after = _distance(processed, left_core_idx, left_root)
    right_after = _distance(processed, right_core_idx, right_root)
    sanitize_success = True
    try:
        Chem.SanitizeMol(processed)
    except Exception as exc:  # noqa: BLE001 - capture RDKit sanitize reason in manifest.
        sanitize_success = False
        errors.append(f"sanitize:{type(exc).__name__}:{exc}")

    attempt = AttachmentAttempt(
        molecule_index=molecule_index,
        success=sanitize_success,
        failure_type="success" if sanitize_success else "sanitize_failed",
        left_root_idx=left_root,
        right_root_idx=right_root,
        left_distance_before=left_distance,
        right_distance_before=right_distance,
        left_distance_after=left_after,
        right_distance_after=right_after,
        left_root_symbol=processed.GetAtomWithIdx(left_root).GetSymbol(),
        right_root_symbol=processed.GetAtomWithIdx(right_root).GetSymbol(),
        sanitize_success=sanitize_success,
        errors=tuple(errors),
    )
    return processed if sanitize_success else processed, attempt


def _read_one_mol(path: Path) -> Chem.Mol:
    """Read the first molecule from an SDF."""
    mols = _read_mols(path)
    valid = [mol for mol in mols if mol is not None]
    if not valid:
        raise ValueError(f"No molecule read from {path}")
    return valid[0]


def _read_mols(path: Path) -> list[Chem.Mol | None]:
    """Read molecules without dropping readable but unsanitized records."""
    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
    return [mol for mol in supplier]


def _write_sdf(path: Path, mols: list[Chem.Mol]) -> None:
    """Write molecules to SDF."""
    if not mols:
        if path.exists():
            path.unlink()
        marker_path = path.with_suffix(".none.txt")
        marker_path.write_text("No post-processed molecules passed the requested filters.\n", encoding="utf-8")
        return
    marker_path = path.with_suffix(".none.txt")
    if marker_path.exists():
        marker_path.unlink()
    writer = Chem.SDWriter(str(path))
    try:
        for mol in mols:
            writer.write(mol)
    finally:
        writer.close()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write manifest rows."""
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


def _summarize_attempts(
    args: argparse.Namespace,
    attempts: list[AttachmentAttempt],
    out_sdf: Path,
    n_fixed: int,
    allowed_root_elements: tuple[str, ...],
) -> dict[str, Any]:
    """Summarize post-processing attempts."""
    total = len(attempts)
    success_count = sum(int(attempt.success) for attempt in attempts)
    failure_counts = Counter(attempt.failure_type for attempt in attempts)
    return {
        "input_sdf": str(args.sdf),
        "output_sdf": str(out_sdf),
        "mode": args.mode,
        "n_fixed": int(n_fixed),
        "left_core_idx": int(args.left_core_idx),
        "right_core_idx": int(args.right_core_idx),
        "allowed_root_elements": list(allowed_root_elements),
        "max_attach_distance": float(args.max_attach_distance),
        "max_snap_distance": float(args.max_snap_distance),
        "total": total,
        "postprocess_success": success_count,
        "postprocess_success_rate": _rate(success_count, total),
        "failure_counts": dict(failure_counts),
    }


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    """Write a compact Markdown report."""
    lines = [
        "# Dual-R Attachment Post-processing",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Output Files",
        "",
        f"- `{Path(summary['output_sdf']).name}`",
        f"- `attachment_{summary['mode']}_manifest.csv`",
        f"- `attachment_{summary['mode']}_summary.json`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _select_root(
    mol: Chem.Mol,
    *,
    core_idx: int,
    generated_indices: tuple[int, ...],
    allowed_root_elements: tuple[str, ...],
    exclude_atoms: set[int] | None = None,
    exclude_components: set[int] | None = None,
    component_by_atom: dict[int, int] | None = None,
) -> tuple[int | None, float | None]:
    """Select nearest allowed generated root atom for one core site."""
    exclude_atoms = exclude_atoms or set()
    exclude_components = exclude_components or set()
    candidates: list[tuple[float, int]] = []
    for atom_idx in generated_indices:
        if atom_idx in exclude_atoms:
            continue
        if component_by_atom is not None and component_by_atom[atom_idx] in exclude_components:
            continue
        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.GetSymbol() not in allowed_root_elements:
            continue
        candidates.append((_distance(mol, core_idx, atom_idx), atom_idx))
    if not candidates:
        return None, None
    distance, atom_idx = min(candidates, key=lambda item: (item[0], item[1]))
    return atom_idx, distance


def _generated_component_by_atom(mol: Chem.Mol, generated_indices: tuple[int, ...]) -> dict[int, int]:
    """Return generated-only connected component id for each generated atom."""
    generated_set = set(generated_indices)
    adjacency = {idx: set() for idx in generated_indices}
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        if begin in generated_set and end in generated_set:
            adjacency[begin].add(end)
            adjacency[end].add(begin)

    component_by_atom: dict[int, int] = {}
    component_id = 0
    for atom_idx in generated_indices:
        if atom_idx in component_by_atom:
            continue
        stack = [atom_idx]
        while stack:
            current = stack.pop()
            if current in component_by_atom:
                continue
            component_by_atom[current] = component_id
            stack.extend(adjacency[current] - set(component_by_atom))
        component_id += 1
    return component_by_atom


def _component_atoms(component_by_atom: dict[int, int], root_idx: int) -> tuple[int, ...]:
    """Return all atoms in the generated component containing root_idx."""
    component_id = component_by_atom[root_idx]
    return tuple(sorted(atom_idx for atom_idx, current_id in component_by_atom.items() if current_id == component_id))


def _snap_component_to_bond_length(
    mol: Chem.RWMol,
    *,
    core_idx: int,
    root_idx: int,
    component_atoms: tuple[int, ...],
) -> None:
    """Translate a generated component so its root reaches a target bond length."""
    conf = mol.GetConformer()
    core_pos = np.array(conf.GetAtomPosition(core_idx))
    root_pos = np.array(conf.GetAtomPosition(root_idx))
    direction = root_pos - core_pos
    distance = float(np.linalg.norm(direction))
    if not math.isfinite(distance) or distance < 1.0e-6:
        direction = np.array([1.0, 0.0, 0.0])
    else:
        direction = direction / distance
    root_symbol = mol.GetAtomWithIdx(root_idx).GetSymbol()
    target_length = DEFAULT_TARGET_BOND_LENGTHS.get(root_symbol, 1.50)
    target_root_pos = core_pos + direction * target_length
    shift = target_root_pos - root_pos
    for atom_idx in component_atoms:
        old_pos = np.array(conf.GetAtomPosition(atom_idx))
        new_pos = old_pos + shift
        conf.SetAtomPosition(atom_idx, Geometry.Point3D(float(new_pos[0]), float(new_pos[1]), float(new_pos[2])))


def _add_single_bond_if_missing(mol: Chem.RWMol, begin_idx: int, end_idx: int) -> None:
    """Add a single bond when the two atoms are not already bonded."""
    if mol.GetBondBetweenAtoms(begin_idx, end_idx) is None:
        mol.AddBond(begin_idx, end_idx, Chem.BondType.SINGLE)


def _annotate_attempt_properties(
    mol: Chem.Mol,
    *,
    molecule_index: int,
    mode: str,
    left_root: int,
    right_root: int,
    left_distance_before: float,
    right_distance_before: float,
) -> None:
    """Annotate one post-processed molecule with attachment metadata."""
    mol.SetProp("_Name", f"postprocess_{mode}_mol_{molecule_index}")
    mol.SetProp("dual_r_postprocess_mode", mode)
    mol.SetProp("dual_r_postprocess_molecule_index", str(molecule_index))
    mol.SetProp("dual_r_postprocess_left_root_idx", str(left_root))
    mol.SetProp("dual_r_postprocess_right_root_idx", str(right_root))
    mol.SetProp("dual_r_postprocess_left_distance_before", f"{left_distance_before:.4f}")
    mol.SetProp("dual_r_postprocess_right_distance_before", f"{right_distance_before:.4f}")


def _distance(mol: Chem.Mol, atom_a: int, atom_b: int) -> float:
    """Return conformer distance between two atoms."""
    conf = mol.GetConformer()
    pos_a = np.array(conf.GetAtomPosition(atom_a))
    pos_b = np.array(conf.GetAtomPosition(atom_b))
    return float(np.linalg.norm(pos_a - pos_b))


def _format_float(value: float | None) -> str:
    """Format optional floats for CSV output."""
    return "" if value is None else f"{float(value):.4f}"


def _rate(count: int, total: int) -> float:
    """Return rounded fraction."""
    if total <= 0:
        return 0.0
    return round(float(count) / float(total), 6)


if __name__ == "__main__":
    raise SystemExit(main())
