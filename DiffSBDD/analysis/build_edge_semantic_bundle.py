"""Build original-SDF semantic metadata for edge diffusion data bundles."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from rdkit import Chem, RDLogger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIFFSBDD_ROOT = Path(__file__).resolve().parents[1]
for extra in (PROJECT_ROOT, DIFFSBDD_ROOT):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import constants
from analysis.oracle_original_sdf_audit import ligand_sdf_relpath_from_name
from dataset import ProcessedLigandPocketDataset
from edge_utils import BOND_TYPE_NAMES, BOND_TYPE_TO_ID

RDLogger.DisableLog("rdApp.*")


@dataclass(frozen=True)
class OriginalMoleculeSemantics:
    """Original SDF-derived atom, bond, and SMILES semantic fields."""

    smiles_iso: str
    smiles_no_stereo: str
    atom_symbols: np.ndarray
    formal_charges: np.ndarray
    total_hs: np.ndarray
    explicit_hs: np.ndarray
    implicit_hs: np.ndarray
    no_implicit_flags: np.ndarray
    aromatic_atom_flags: np.ndarray
    chiral_tags: np.ndarray
    hybridizations: np.ndarray
    bond_pairs: np.ndarray
    bond_types: np.ndarray
    aromatic_bond_flags: np.ndarray
    bond_stereo: np.ndarray

    @property
    def num_atoms(self) -> int:
        """Return atom count."""
        return int(len(self.atom_symbols))

    @property
    def num_bonds(self) -> int:
        """Return bond count."""
        return int(len(self.bond_types))

    @property
    def total_formal_charge(self) -> int:
        """Return molecule total formal charge."""
        return int(self.formal_charges.sum())


def extract_original_semantics(mol: Chem.Mol) -> OriginalMoleculeSemantics:
    """Extract SDF-derived atom, bond, charge, H, aromatic, and SMILES fields."""
    atom_symbols = []
    formal_charges = []
    total_hs = []
    explicit_hs = []
    implicit_hs = []
    no_implicit_flags = []
    aromatic_atom_flags = []
    chiral_tags = []
    hybridizations = []
    for atom in mol.GetAtoms():
        atom_symbols.append(atom.GetSymbol())
        formal_charges.append(atom.GetFormalCharge())
        total_hs.append(atom.GetTotalNumHs())
        explicit_hs.append(atom.GetNumExplicitHs())
        implicit_hs.append(atom.GetNumImplicitHs())
        no_implicit_flags.append(int(atom.GetNoImplicit()))
        aromatic_atom_flags.append(int(atom.GetIsAromatic()))
        chiral_tags.append(int(atom.GetChiralTag()))
        hybridizations.append(int(atom.GetHybridization()))

    bond_rows = []
    bond_types = []
    aromatic_bond_flags = []
    bond_stereo = []
    for bond in mol.GetBonds():
        begin = int(bond.GetBeginAtomIdx())
        end = int(bond.GetEndAtomIdx())
        bond_rows.append(tuple(sorted((begin, end))))
        bond_types.append(BOND_TYPE_TO_ID.get(bond.GetBondType(), -1))
        aromatic_bond_flags.append(int(bond.GetIsAromatic()))
        bond_stereo.append(int(bond.GetStereo()))

    if bond_rows:
        order = np.argsort([row[0] * mol.GetNumAtoms() + row[1] for row in bond_rows])
        bond_pairs = np.asarray(bond_rows, dtype=np.int64)[order]
        bond_type_array = np.asarray(bond_types, dtype=np.int64)[order]
        aromatic_bond_array = np.asarray(aromatic_bond_flags, dtype=np.int64)[order]
        bond_stereo_array = np.asarray(bond_stereo, dtype=np.int64)[order]
    else:
        bond_pairs = np.zeros((0, 2), dtype=np.int64)
        bond_type_array = np.zeros((0,), dtype=np.int64)
        aromatic_bond_array = np.zeros((0,), dtype=np.int64)
        bond_stereo_array = np.zeros((0,), dtype=np.int64)

    return OriginalMoleculeSemantics(
        smiles_iso=Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True),
        smiles_no_stereo=Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False),
        atom_symbols=np.asarray(atom_symbols, dtype=object),
        formal_charges=np.asarray(formal_charges, dtype=np.int64),
        total_hs=np.asarray(total_hs, dtype=np.int64),
        explicit_hs=np.asarray(explicit_hs, dtype=np.int64),
        implicit_hs=np.asarray(implicit_hs, dtype=np.int64),
        no_implicit_flags=np.asarray(no_implicit_flags, dtype=np.int64),
        aromatic_atom_flags=np.asarray(aromatic_atom_flags, dtype=np.int64),
        chiral_tags=np.asarray(chiral_tags, dtype=np.int64),
        hybridizations=np.asarray(hybridizations, dtype=np.int64),
        bond_pairs=bond_pairs,
        bond_types=bond_type_array,
        aromatic_bond_flags=aromatic_bond_array,
        bond_stereo=bond_stereo_array,
    )


def build_edge_semantic_bundle(args: argparse.Namespace) -> Path:
    """Build a semantic sidecar bundle and report for edge diffusion data."""
    dataset_info = constants.dataset_params[args.dataset]
    atom_decoder = dataset_info["atom_decoder"]
    dataset = ProcessedLigandPocketDataset(args.npz_path, center=False)
    args.outdir.mkdir(parents=True, exist_ok=True)
    limit = len(dataset) if args.max_samples is None else min(args.max_samples, len(dataset))

    names: list[str] = []
    original_sdf_paths: list[str] = []
    load_success: list[int] = []
    load_errors: list[str] = []
    smiles_iso: list[str] = []
    smiles_no_stereo: list[str] = []
    atom_symbols: list[np.ndarray] = []
    formal_charges: list[np.ndarray] = []
    total_hs: list[np.ndarray] = []
    explicit_hs: list[np.ndarray] = []
    implicit_hs: list[np.ndarray] = []
    no_implicit_flags: list[np.ndarray] = []
    aromatic_atom_flags: list[np.ndarray] = []
    chiral_tags: list[np.ndarray] = []
    hybridizations: list[np.ndarray] = []
    bond_pairs: list[np.ndarray] = []
    bond_types: list[np.ndarray] = []
    aromatic_bond_flags: list[np.ndarray] = []
    bond_stereo: list[np.ndarray] = []
    summary_rows: list[dict[str, object]] = []

    load_error_counts: Counter[str] = Counter()
    total_charge_counts: Counter[int] = Counter()
    match_counts: Counter[str] = Counter()
    processed_bond_type_counts: Counter[int] = Counter()
    original_bond_type_counts: Counter[int] = Counter()

    for sample_idx in range(limit):
        sample = dataset[sample_idx]
        name = str(sample["names"])
        names.append(name)
        original_mol, original_path, load_error = _load_original_from_sample(
            name=name,
            raw_ligand_root=args.raw_ligand_root,
        )
        original_sdf_paths.append("" if original_path is None else str(original_path))
        load_success.append(int(original_mol is not None))
        load_errors.append(load_error or "")
        load_error_counts[load_error or "success"] += 1

        processed_symbols = _processed_atom_symbols(sample, atom_decoder)
        processed_bond_map = _processed_halfedge_bond_map(sample)
        processed_bond_type_counts.update(processed_bond_map.values())

        if original_mol is None:
            semantics = _empty_semantics()
            comparison = _empty_comparison(processed_symbols, processed_bond_map)
        else:
            semantics = extract_original_semantics(original_mol)
            original_bond_map = _bond_map_from_semantics(semantics)
            original_bond_type_counts.update(original_bond_map.values())
            comparison = _compare_semantics(
                processed_symbols=processed_symbols,
                processed_bond_map=processed_bond_map,
                semantics=semantics,
                original_bond_map=original_bond_map,
            )

        smiles_iso.append(semantics.smiles_iso)
        smiles_no_stereo.append(semantics.smiles_no_stereo)
        atom_symbols.append(semantics.atom_symbols)
        formal_charges.append(semantics.formal_charges)
        total_hs.append(semantics.total_hs)
        explicit_hs.append(semantics.explicit_hs)
        implicit_hs.append(semantics.implicit_hs)
        no_implicit_flags.append(semantics.no_implicit_flags)
        aromatic_atom_flags.append(semantics.aromatic_atom_flags)
        chiral_tags.append(semantics.chiral_tags)
        hybridizations.append(semantics.hybridizations)
        bond_pairs.append(semantics.bond_pairs)
        bond_types.append(semantics.bond_types)
        aromatic_bond_flags.append(semantics.aromatic_bond_flags)
        bond_stereo.append(semantics.bond_stereo)
        total_charge_counts[semantics.total_formal_charge] += 1
        for key, value in comparison.items():
            if key.endswith("_match"):
                match_counts[key] += int(value)

        summary_rows.append({
            "sample_idx": sample_idx,
            "name": name,
            "original_sdf": "" if original_path is None else str(original_path),
            "original_load_success": int(original_mol is not None),
            "original_load_error": load_error or "",
            "processed_num_atoms": len(processed_symbols),
            "original_num_atoms": semantics.num_atoms,
            "processed_num_bonds": len(processed_bond_map),
            "original_num_bonds": semantics.num_bonds,
            "original_total_formal_charge": semantics.total_formal_charge,
            "original_nonzero_charge_atoms": int(np.count_nonzero(semantics.formal_charges)),
            "original_aromatic_atom_count": int(semantics.aromatic_atom_flags.sum()),
            "original_aromatic_bond_count": int(semantics.aromatic_bond_flags.sum()),
            "original_total_h_count": int(semantics.total_hs.sum()),
            "original_smiles_iso": semantics.smiles_iso,
            "original_smiles_no_stereo": semantics.smiles_no_stereo,
            **comparison,
        })

    bundle_path = args.outdir / args.semantic_npz_name
    np.savez_compressed(
        bundle_path,
        schema_version=np.array("edge_semantic_bundle_v1"),
        created_at=np.array(datetime.now().isoformat(timespec="seconds")),
        source_npz=np.array(str(args.npz_path)),
        raw_ligand_root=np.array(str(args.raw_ligand_root)),
        dataset=np.array(args.dataset),
        names=_object_array(names),
        original_sdf=_object_array(original_sdf_paths),
        original_load_success=np.asarray(load_success, dtype=np.int8),
        original_load_error=_object_array(load_errors),
        original_smiles_iso=_object_array(smiles_iso),
        original_smiles_no_stereo=_object_array(smiles_no_stereo),
        original_atom_symbol=_object_array(atom_symbols),
        original_formal_charge=_object_array(formal_charges),
        original_total_h=_object_array(total_hs),
        original_explicit_h=_object_array(explicit_hs),
        original_implicit_h=_object_array(implicit_hs),
        original_no_implicit=_object_array(no_implicit_flags),
        original_aromatic_atom=_object_array(aromatic_atom_flags),
        original_chiral_tag=_object_array(chiral_tags),
        original_hybridization=_object_array(hybridizations),
        original_bond_pair=_object_array(bond_pairs),
        original_bond_type=_object_array(bond_types),
        original_aromatic_bond=_object_array(aromatic_bond_flags),
        original_bond_stereo=_object_array(bond_stereo),
    )

    _write_summary_csv(args.outdir / "edge_semantic_molecule_summary.csv", summary_rows)
    summary = _build_manifest(
        args=args,
        bundle_path=bundle_path,
        limit=limit,
        load_error_counts=load_error_counts,
        total_charge_counts=total_charge_counts,
        match_counts=match_counts,
        processed_bond_type_counts=processed_bond_type_counts,
        original_bond_type_counts=original_bond_type_counts,
    )
    (args.outdir / "edge_semantic_manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    _plot_match_rates(args.outdir / "edge_semantic_match_rates.png", summary)
    _plot_charge_distribution(args.outdir / "edge_semantic_total_charge_distribution.png", total_charge_counts)
    return _write_report(args.outdir, summary)


def _load_original_from_sample(
    name: str,
    raw_ligand_root: Path,
) -> tuple[Chem.Mol | None, Path | None, str | None]:
    """Load the original ligand SDF aligned to a processed sample name."""
    try:
        relpath = ligand_sdf_relpath_from_name(name)
    except ValueError as exc:
        return None, None, type(exc).__name__
    original_path = raw_ligand_root / relpath
    if not original_path.exists():
        return None, original_path, "missing_sdf"
    mol = Chem.MolFromMolFile(str(original_path), sanitize=False, removeHs=False)
    if mol is None:
        return None, original_path, "mol_from_file_failed"
    try:
        Chem.SanitizeMol(mol)
        mol = Chem.RemoveHs(mol, sanitize=True)
        Chem.SanitizeMol(mol)
    except Exception as exc:  # noqa: BLE001 - report RDKit error class
        return None, original_path, type(exc).__name__
    return mol, original_path, None


def _processed_atom_symbols(sample: dict[str, object], atom_decoder: list[str]) -> list[str]:
    """Return processed atom symbols in ligand atom order."""
    atom_type = sample["lig_one_hot"].argmax(dim=-1).cpu().tolist()
    return [atom_decoder[int(atom_idx)] for atom_idx in atom_type]


def _processed_halfedge_bond_map(sample: dict[str, object]) -> dict[tuple[int, int], int]:
    """Return processed true halfedge bonds as pair -> bond type."""
    halfedge_index = sample["ligand_halfedge_index"]
    halfedge_type = sample["ligand_halfedge_type"]
    bond_map: dict[tuple[int, int], int] = {}
    for edge_pos, edge_type_tensor in enumerate(halfedge_type):
        edge_type = int(edge_type_tensor)
        if edge_type <= 0:
            continue
        atom_i = int(halfedge_index[0, edge_pos])
        atom_j = int(halfedge_index[1, edge_pos])
        bond_map[tuple(sorted((atom_i, atom_j)))] = edge_type
    return bond_map


def _bond_map_from_semantics(semantics: OriginalMoleculeSemantics) -> dict[tuple[int, int], int]:
    """Return original bond pair -> bond type from semantic fields."""
    return {
        tuple(map(int, pair)): int(bond_type)
        for pair, bond_type in zip(semantics.bond_pairs.tolist(), semantics.bond_types.tolist())
    }


def _compare_semantics(
    processed_symbols: list[str],
    processed_bond_map: dict[tuple[int, int], int],
    semantics: OriginalMoleculeSemantics,
    original_bond_map: dict[tuple[int, int], int],
) -> dict[str, int]:
    """Compare processed atom/bond labels against original SDF semantics."""
    original_symbols = [str(symbol) for symbol in semantics.atom_symbols.tolist()]
    processed_pairs = set(processed_bond_map)
    original_pairs = set(original_bond_map)
    shared_pairs = processed_pairs & original_pairs
    bond_type_mismatch_count = sum(
        int(processed_bond_map[pair] != original_bond_map[pair])
        for pair in shared_pairs
    )
    return {
        "atom_count_match": int(len(processed_symbols) == semantics.num_atoms),
        "atom_symbol_match": int(processed_symbols == original_symbols),
        "bond_count_match": int(len(processed_bond_map) == semantics.num_bonds),
        "bond_pair_set_match": int(processed_pairs == original_pairs),
        "bond_type_map_match": int(processed_bond_map == original_bond_map),
        "missing_bond_count": len(original_pairs - processed_pairs),
        "extra_bond_count": len(processed_pairs - original_pairs),
        "bond_type_mismatch_count": bond_type_mismatch_count,
    }


def _empty_comparison(
    processed_symbols: list[str],
    processed_bond_map: dict[tuple[int, int], int],
) -> dict[str, int]:
    """Return comparison fields for failed original SDF loads."""
    return {
        "atom_count_match": 0,
        "atom_symbol_match": 0,
        "bond_count_match": 0,
        "bond_pair_set_match": 0,
        "bond_type_map_match": 0,
        "missing_bond_count": 0,
        "extra_bond_count": len(processed_bond_map),
        "bond_type_mismatch_count": 0,
    }


def _empty_semantics() -> OriginalMoleculeSemantics:
    """Return empty semantic fields for failed original SDF loads."""
    return OriginalMoleculeSemantics(
        smiles_iso="",
        smiles_no_stereo="",
        atom_symbols=np.asarray([], dtype=object),
        formal_charges=np.zeros((0,), dtype=np.int64),
        total_hs=np.zeros((0,), dtype=np.int64),
        explicit_hs=np.zeros((0,), dtype=np.int64),
        implicit_hs=np.zeros((0,), dtype=np.int64),
        no_implicit_flags=np.zeros((0,), dtype=np.int64),
        aromatic_atom_flags=np.zeros((0,), dtype=np.int64),
        chiral_tags=np.zeros((0,), dtype=np.int64),
        hybridizations=np.zeros((0,), dtype=np.int64),
        bond_pairs=np.zeros((0, 2), dtype=np.int64),
        bond_types=np.zeros((0,), dtype=np.int64),
        aromatic_bond_flags=np.zeros((0,), dtype=np.int64),
        bond_stereo=np.zeros((0,), dtype=np.int64),
    )


def _object_array(values: list[object]) -> np.ndarray:
    """Create an object array without NumPy trying to stack variable-length values."""
    array = np.empty(len(values), dtype=object)
    for idx, value in enumerate(values):
        array[idx] = value
    return array


def _write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write per-molecule semantic summary rows."""
    fieldnames = [
        "sample_idx", "name", "original_sdf", "original_load_success",
        "original_load_error", "processed_num_atoms", "original_num_atoms",
        "processed_num_bonds", "original_num_bonds", "original_total_formal_charge",
        "original_nonzero_charge_atoms", "original_aromatic_atom_count",
        "original_aromatic_bond_count", "original_total_h_count",
        "original_smiles_iso", "original_smiles_no_stereo",
        "atom_count_match", "atom_symbol_match", "bond_count_match",
        "bond_pair_set_match", "bond_type_map_match", "missing_bond_count",
        "extra_bond_count", "bond_type_mismatch_count",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _build_manifest(
    args: argparse.Namespace,
    bundle_path: Path,
    limit: int,
    load_error_counts: Counter[str],
    total_charge_counts: Counter[int],
    match_counts: Counter[str],
    processed_bond_type_counts: Counter[int],
    original_bond_type_counts: Counter[int],
) -> dict[str, object]:
    """Build JSON-serializable run summary."""
    return {
        "semantic_npz": str(bundle_path),
        "source_npz": str(args.npz_path),
        "raw_ligand_root": str(args.raw_ligand_root),
        "dataset": args.dataset,
        "samples": limit,
        "load_error_counts": dict(sorted(load_error_counts.items())),
        "total_charge_counts": {str(key): value for key, value in sorted(total_charge_counts.items())},
        "match_counts": {
            key: {
                "count": int(match_counts.get(key, 0)),
                "denominator": limit,
                "rate": float(match_counts.get(key, 0) / max(1, limit)),
            }
            for key in [
                "atom_count_match",
                "atom_symbol_match",
                "bond_count_match",
                "bond_pair_set_match",
                "bond_type_map_match",
            ]
        },
        "processed_bond_type_counts": _bond_count_dict(processed_bond_type_counts),
        "original_bond_type_counts": _bond_count_dict(original_bond_type_counts),
    }


def _bond_count_dict(counts: Counter[int]) -> dict[str, int]:
    """Return bond type counts with readable names."""
    return {
        BOND_TYPE_NAMES.get(int(key), f"unknown_{key}"): int(value)
        for key, value in sorted(counts.items())
    }


def _plot_match_rates(path: Path, summary: dict[str, object]) -> None:
    """Plot semantic match rates."""
    match_counts = summary["match_counts"]
    labels = list(match_counts)
    values = [float(match_counts[label]["rate"]) for label in labels]
    plt.figure(figsize=(8, 4.5))
    plt.bar(labels, values, color="#4C78A8")
    plt.ylim(0, 1.05)
    plt.ylabel("match rate")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_charge_distribution(path: Path, total_charge_counts: Counter[int]) -> None:
    """Plot original total formal charge distribution."""
    keys = sorted(total_charge_counts)
    values = [total_charge_counts[key] for key in keys]
    plt.figure(figsize=(6, 4))
    plt.bar([str(key) for key in keys], values, color="#59A14F")
    plt.xlabel("total formal charge")
    plt.ylabel("molecule count")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _write_report(outdir: Path, summary: dict[str, object]) -> Path:
    """Write a Markdown report for the semantic edge bundle."""
    report_path = outdir / "edge_semantic_bundle_report.md"
    with report_path.open("w") as handle:
        handle.write("# Edge Semantic Bundle Report\n\n")
        handle.write(f"- semantic_npz: `{Path(summary['semantic_npz']).name}`\n")
        handle.write(f"- source_npz: `{summary['source_npz']}`\n")
        handle.write(f"- raw_ligand_root: `{summary['raw_ligand_root']}`\n")
        handle.write(f"- dataset: `{summary['dataset']}`\n")
        handle.write(f"- samples: {summary['samples']}\n\n")
        handle.write("## Original SDF Load Status\n\n")
        handle.write("| status | count |\n")
        handle.write("|---|---:|\n")
        for status, count in summary["load_error_counts"].items():
            handle.write(f"| {status} | {count} |\n")
        handle.write("\n## Semantic Match Rates\n\n")
        handle.write("| metric | count | denominator | rate |\n")
        handle.write("|---|---:|---:|---:|\n")
        for metric, row in summary["match_counts"].items():
            handle.write(
                f"| {metric} | {row['count']} | {row['denominator']} | {row['rate']:.8f} |\n"
            )
        handle.write("\n## Total Formal Charge Counts\n\n")
        handle.write("| total charge | count |\n")
        handle.write("|---|---:|\n")
        for charge, count in summary["total_charge_counts"].items():
            handle.write(f"| {charge} | {count} |\n")
        handle.write("\n## Bond Type Counts\n\n")
        handle.write("| bond type | processed | original SDF |\n")
        handle.write("|---|---:|---:|\n")
        processed_counts = summary["processed_bond_type_counts"]
        original_counts = summary["original_bond_type_counts"]
        for bond_name in sorted(set(processed_counts) | set(original_counts)):
            handle.write(
                f"| {bond_name} | {processed_counts.get(bond_name, 0)} | "
                f"{original_counts.get(bond_name, 0)} |\n"
            )
        handle.write("\n## Files\n\n")
        handle.write("- [train_edge_semantic.npz](train_edge_semantic.npz)\n")
        handle.write("- [edge_semantic_molecule_summary.csv](edge_semantic_molecule_summary.csv)\n")
        handle.write("- [edge_semantic_manifest.json](edge_semantic_manifest.json)\n")
        handle.write("\n![semantic match rates](edge_semantic_match_rates.png)\n\n")
        handle.write("![total charge distribution](edge_semantic_total_charge_distribution.png)\n")
    return report_path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--npz-path",
        type=Path,
        default=Path("data/crossdocked_raw/processed_edge_noH_full/train.npz"),
    )
    parser.add_argument(
        "--raw-ligand-root",
        type=Path,
        default=Path("data/crossdocked_raw/crossdocked_pocket10"),
    )
    parser.add_argument("--dataset", choices=sorted(constants.dataset_params.keys()), default="crossdock_full")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("analysis")
        / f"edge_semantic_bundle_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument("--semantic-npz-name", default="train_edge_semantic.npz")
    parser.add_argument("--max-samples", type=int, default=1000)
    return parser.parse_args()


if __name__ == "__main__":
    report = build_edge_semantic_bundle(parse_args())
    print(f"Wrote edge semantic bundle report to {report}")
