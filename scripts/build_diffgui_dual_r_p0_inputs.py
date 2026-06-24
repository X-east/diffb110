"""Build DiffGui dual-R fixed-core and constraint input artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from rdkit import Chem

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ddpo.constraints.dual_sidechain import (  # noqa: E402
    build_dual_substitution_spec_from_phenolic_sites,
    find_phenolic_oh_sites,
    substitute_dual_r_groups,
)
from ddpo.diffgui.constraints import (  # noqa: E402
    DiffGuiDualRSlotLayout,
    build_diffgui_dual_r_constraint_spec,
)
from ddpo.diffgui.runtime.rollout import make_diffgui_ligand_placeholder  # noqa: E402


LOGGER = logging.getLogger("build_diffgui_dual_r_p0_inputs")

DIFFGUI_BOND_TYPES = {
    Chem.BondType.UNSPECIFIED: 0,
    Chem.BondType.SINGLE: 1,
    Chem.BondType.DOUBLE: 2,
    Chem.BondType.TRIPLE: 3,
    Chem.BondType.AROMATIC: 4,
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdf", required=True, type=Path, help="Docked honokiol SDF.")
    parser.add_argument("--protein", required=True, type=Path, help="Protein PDB used by later DiffGui sampling.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory.")
    parser.add_argument("--molecule-index", type=int, default=0, help="Molecule index in the SDF.")
    parser.add_argument("--left-site-index", type=int, default=0, help="Phenolic OH site index for R1.")
    parser.add_argument("--right-site-index", type=int, default=1, help="Phenolic OH site index for R2.")
    parser.add_argument("--r-slots-per-side", type=int, default=3, help="Number of generated R slots per side.")
    parser.add_argument("--num-node-types", type=int, default=16, help="DiffGui node categorical dimension.")
    parser.add_argument("--num-edge-types", type=int, default=5, help="DiffGui halfedge categorical dimension.")
    parser.add_argument("--attachment-bond-type", type=int, default=1, help="Forced attachment bond type id.")
    parser.add_argument("--force-r-chain", action="store_true", help="Force consecutive R slots on each side to single bonds.")
    parser.add_argument("--r-chain-bond-type", type=int, default=1, help="Bond type id used by --force-r-chain.")
    parser.add_argument(
        "--allowed-node-types",
        default="",
        help="Optional comma-separated allowed DiffGui node type ids for R slots. Empty means all.",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> int:
    """Build all P0 input artifacts."""
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level)),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("building DiffGui dual-R P0 inputs out_dir=%s", args.out_dir)

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
    fixed_core, heavy_old_to_new = _remove_hs_with_mapping(fixed_result.mol)
    Chem.SanitizeMol(fixed_core)
    fixed_core.SetProp("_Name", "honokiol_fixed_core_for_diffgui_dual_r")
    fixed_core.SetProp("dual_r_fixed_core", "1")

    left_anchor = heavy_old_to_new[fixed_result.adjusted_spec.left_core_idx]
    right_anchor = heavy_old_to_new[fixed_result.adjusted_spec.right_core_idx]
    layout = DiffGuiDualRSlotLayout.from_core(
        core_size=fixed_core.GetNumAtoms(),
        left_anchor_idx=left_anchor,
        right_anchor_idx=right_anchor,
        r_slots_per_side=int(args.r_slots_per_side),
    )
    ligand_batch, halfedge_index, halfedge_batch = make_diffgui_ligand_placeholder([layout.total_nodes])
    del ligand_batch, halfedge_batch

    core_bond_types = _core_bond_types(fixed_core)
    constraint_spec = build_diffgui_dual_r_constraint_spec(
        layout=layout,
        halfedge_index=halfedge_index,
        num_node_types=int(args.num_node_types),
        num_edge_types=int(args.num_edge_types),
        core_bond_types=core_bond_types,
        attachment_bond_type=int(args.attachment_bond_type),
        allowed_node_types=_parse_allowed_node_types(args.allowed_node_types),
        require_r_chain_edges=bool(args.force_r_chain),
        r_chain_bond_type=int(args.r_chain_bond_type),
    )

    _write_sdf(args.out_dir / "core_fragment.sdf", fixed_core)
    _write_json(
        args.out_dir / "site_spec.json",
        _site_payload(
            args=args,
            reference_mol=reference_mol,
            fixed_core=fixed_core,
            sites=sites,
            original_spec=spec,
            adjusted_spec=fixed_result.adjusted_spec,
            heavy_old_to_new=heavy_old_to_new,
            removed_atom_indices=fixed_result.removed_atom_indices,
        ),
    )
    _write_json(args.out_dir / "slot_layout.json", layout.to_dict())
    _write_json(args.out_dir / "constraint_spec.json", constraint_spec.to_dict())
    node_rows = _node_role_rows(layout)
    halfedge_rows = _halfedge_role_rows(layout, halfedge_index, core_bond_types, constraint_spec)
    _write_csv(args.out_dir / "node_role_table.csv", node_rows)
    _write_csv(args.out_dir / "halfedge_role_table.csv", halfedge_rows)
    _plot_halfedge_roles(args.out_dir / "halfedge_role_counts.png", halfedge_rows)
    _write_report(
        args.out_dir / "report.md",
        args=args,
        fixed_core=fixed_core,
        layout=layout,
        node_rows=node_rows,
        halfedge_rows=halfedge_rows,
    )
    LOGGER.info("P0 inputs completed core_atoms=%d total_slots=%d", fixed_core.GetNumAtoms(), layout.total_nodes)
    return 0


def _read_reference_mol(path: Path, molecule_index: int) -> Chem.Mol:
    """Read and sanitize one molecule from SDF."""
    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
    mols = [mol for mol in supplier if mol is not None]
    if not mols:
        raise ValueError(f"No molecules were read from {path}.")
    if molecule_index < 0 or molecule_index >= len(mols):
        raise ValueError(f"molecule-index {molecule_index} out of range for {len(mols)} molecules.")
    mol = Chem.Mol(mols[molecule_index])
    Chem.SanitizeMol(mol)
    if mol.GetNumConformers() == 0:
        raise ValueError("Reference molecule must contain 3D coordinates.")
    return mol


def _remove_hs_with_mapping(mol: Chem.Mol) -> tuple[Chem.Mol, dict[int, int]]:
    """Remove hydrogens and return old heavy-atom index to new index mapping."""
    heavy_old_to_new: dict[int, int] = {}
    next_idx = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        heavy_old_to_new[atom.GetIdx()] = next_idx
        next_idx += 1
    heavy = Chem.RemoveHs(mol, sanitize=False)
    return heavy, heavy_old_to_new


def _core_bond_types(mol: Chem.Mol) -> dict[tuple[int, int], int]:
    """Return DiffGui bond type ids for all fixed-core bonds."""
    values: dict[tuple[int, int], int] = {}
    for bond in mol.GetBonds():
        begin = int(bond.GetBeginAtomIdx())
        end = int(bond.GetEndAtomIdx())
        key = (begin, end) if begin < end else (end, begin)
        try:
            values[key] = int(DIFFGUI_BOND_TYPES[bond.GetBondType()])
        except KeyError as exc:
            raise ValueError(f"Unsupported RDKit bond type: {bond.GetBondType()}") from exc
    return values


def _parse_allowed_node_types(raw: str) -> tuple[int, ...]:
    """Parse optional comma-separated allowed node type ids."""
    if not raw.strip():
        return ()
    values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if len(set(values)) != len(values):
        raise ValueError("allowed-node-types must not contain duplicates.")
    return values


def _site_payload(
    *,
    args: argparse.Namespace,
    reference_mol: Chem.Mol,
    fixed_core: Chem.Mol,
    sites,
    original_spec,
    adjusted_spec,
    heavy_old_to_new: dict[int, int],
    removed_atom_indices: tuple[int, ...],
) -> dict[str, Any]:
    """Build JSON metadata for site and core preparation."""
    return {
        "input_sdf": str(args.sdf),
        "protein_pdb": str(args.protein),
        "molecule_index": int(args.molecule_index),
        "phenolic_oh_sites": [site.to_dict() for site in sites],
        "original_dual_substitution_spec": original_spec.to_dict(),
        "adjusted_spec_after_leaving_group_removal": adjusted_spec.to_dict(),
        "heavy_old_to_new_atom_indices": {str(key): int(value) for key, value in heavy_old_to_new.items()},
        "removed_atom_indices": [int(value) for value in removed_atom_indices],
        "heavy_adjusted_left_anchor_idx": int(heavy_old_to_new[adjusted_spec.left_core_idx]),
        "heavy_adjusted_right_anchor_idx": int(heavy_old_to_new[adjusted_spec.right_core_idx]),
        "reference_atoms": int(reference_mol.GetNumAtoms()),
        "reference_heavy_atoms": int(reference_mol.GetNumHeavyAtoms()),
        "fixed_core_atoms": int(fixed_core.GetNumAtoms()),
        "fixed_core_bonds": int(fixed_core.GetNumBonds()),
    }


def _node_role_rows(layout: DiffGuiDualRSlotLayout) -> list[dict[str, Any]]:
    """Return node role table rows."""
    rows: list[dict[str, Any]] = []
    for idx in range(layout.total_nodes):
        if idx < layout.core_size:
            role = "core_anchor_left" if idx == layout.left_anchor_idx else "core"
            role = "core_anchor_right" if idx == layout.right_anchor_idx else role
        elif idx in layout.left_r_indices:
            role = "left_r_root" if idx == layout.left_root_idx else "left_r"
        else:
            role = "right_r_root" if idx == layout.right_root_idx else "right_r"
        pair_idx = ""
        if idx in layout.left_r_indices:
            pair_idx = layout.right_r_indices[layout.left_r_indices.index(idx)]
        elif idx in layout.right_r_indices:
            pair_idx = layout.left_r_indices[layout.right_r_indices.index(idx)]
        rows.append({"node_idx": idx, "role": role, "paired_node_idx": pair_idx})
    return rows


def _halfedge_role_rows(
    layout: DiffGuiDualRSlotLayout,
    halfedge_index: torch.Tensor,
    core_bond_types: dict[tuple[int, int], int],
    constraint_spec,
) -> list[dict[str, Any]]:
    """Return halfedge role rows for inspection and later prompt construction."""
    required = dict(constraint_spec.halfedge_map.required_edges)
    forbidden = set(constraint_spec.halfedge_map.forbidden_edges)
    paired: dict[int, int] = {}
    for left_edge, right_edge in constraint_spec.halfedge_map.paired_edges:
        paired[int(left_edge)] = int(right_edge)
        paired[int(right_edge)] = int(left_edge)
    rows = []
    for edge_id in range(int(halfedge_index.shape[1])):
        a = int(halfedge_index[0, edge_id])
        b = int(halfedge_index[1, edge_id])
        pair = (a, b) if a < b else (b, a)
        role = "free"
        forced_type: int | str = ""
        paired_edge_id: int | str = ""
        if edge_id in required:
            forced_type = required[edge_id]
            if pair == tuple(sorted((layout.left_anchor_idx, layout.left_root_idx))):
                role = "required_attachment_left"
            elif pair == tuple(sorted((layout.right_anchor_idx, layout.right_root_idx))):
                role = "required_attachment_right"
            elif pair in core_bond_types:
                role = "required_core_bond"
            elif _is_same_side_r_edge(layout, a, b):
                role = "required_r_chain"
            else:
                role = "required"
        elif edge_id in forbidden:
            forced_type = 0
            if _is_core_core_edge(layout, a, b):
                role = "forbidden_core_nonbond"
            elif _is_core_r_edge(layout, a, b):
                role = "forbidden_core_new"
            elif _is_cross_side_edge(layout, a, b):
                role = "forbidden_cross_side"
            else:
                role = "forbidden"
        elif edge_id in paired:
            role = "paired_internal"
            paired_edge_id = paired[edge_id]
        rows.append(
            {
                "edge_id": edge_id,
                "node_i": a,
                "node_j": b,
                "role": role,
                "forced_type": forced_type,
                "paired_edge_id": paired_edge_id,
            }
        )
    return rows


def _is_core_r_edge(layout: DiffGuiDualRSlotLayout, a: int, b: int) -> bool:
    """Return whether an edge connects the fixed core and an R slot."""
    core = set(layout.core_indices())
    r_slots = set(layout.r_indices())
    return (a in core and b in r_slots) or (b in core and a in r_slots)


def _is_core_core_edge(layout: DiffGuiDualRSlotLayout, a: int, b: int) -> bool:
    """Return whether an edge connects two fixed-core slots."""
    core = set(layout.core_indices())
    return a in core and b in core


def _is_cross_side_edge(layout: DiffGuiDualRSlotLayout, a: int, b: int) -> bool:
    """Return whether an edge connects left and right R slots."""
    left = set(layout.left_r_indices)
    right = set(layout.right_r_indices)
    return (a in left and b in right) or (b in left and a in right)


def _is_same_side_r_edge(layout: DiffGuiDualRSlotLayout, a: int, b: int) -> bool:
    """Return whether an edge connects two R slots on the same side."""
    left = set(layout.left_r_indices)
    right = set(layout.right_r_indices)
    return (a in left and b in left) or (a in right and b in right)


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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write CSV rows."""
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_halfedge_roles(path: Path, rows: list[dict[str, Any]]) -> None:
    """Plot halfedge role counts."""
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row["role"])] = counts.get(str(row["role"]), 0) + 1
    labels = sorted(counts)
    values = [counts[label] for label in labels]
    fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=180)
    bars = ax.bar(labels, values, color="#4C78A8")
    ax.set_ylabel("halfedges")
    ax.set_title("DiffGui dual-R P0 halfedge roles")
    ax.tick_params(axis="x", rotation=25)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), str(int(bar.get_height())), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _write_report(
    path: Path,
    *,
    args: argparse.Namespace,
    fixed_core: Chem.Mol,
    layout: DiffGuiDualRSlotLayout,
    node_rows: list[dict[str, Any]],
    halfedge_rows: list[dict[str, Any]],
) -> None:
    """Write a Markdown report for the generated P0 inputs."""
    role_counts: dict[str, int] = {}
    for row in halfedge_rows:
        role_counts[str(row["role"])] = role_counts.get(str(row["role"]), 0) + 1
    lines = [
        "# DiffGui Dual-R P0 Inputs",
        "",
        f"- Input SDF: `{args.sdf}`",
        f"- Protein PDB: `{args.protein}`",
        f"- R slots per side: `{int(args.r_slots_per_side)}`",
        f"- Force R chain: `{bool(args.force_r_chain)}`",
        f"- Fixed core atoms: `{fixed_core.GetNumAtoms()}`",
        f"- Total DiffGui ligand slots: `{layout.total_nodes}`",
        "",
        "## Slot Summary",
        "",
        "| item | value |",
        "|---|---:|",
        f"| core atoms | {layout.core_size} |",
        f"| left anchor | {layout.left_anchor_idx} |",
        f"| right anchor | {layout.right_anchor_idx} |",
        f"| left R slots | {' '.join(str(value) for value in layout.left_r_indices)} |",
        f"| right R slots | {' '.join(str(value) for value in layout.right_r_indices)} |",
        f"| node role rows | {len(node_rows)} |",
        f"| halfedge rows | {len(halfedge_rows)} |",
        "",
        "## Halfedge Role Counts",
        "",
        "| role | count |",
        "|---|---:|",
    ]
    for role, count in sorted(role_counts.items()):
        lines.append(f"| {role} | {count} |")
    lines.extend(
        [
            "",
            "![Halfedge role counts](halfedge_role_counts.png)",
            "",
            "## Output Files",
            "",
            "- [core_fragment.sdf](core_fragment.sdf)",
            "- [site_spec.json](site_spec.json)",
            "- [slot_layout.json](slot_layout.json)",
            "- [constraint_spec.json](constraint_spec.json)",
            "- [node_role_table.csv](node_role_table.csv)",
            "- [halfedge_role_table.csv](halfedge_role_table.csv)",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
