"""Write paired original and repaired-oracle SDF records for inspection."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
from rdkit import Chem, RDLogger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIFFSBDD_ROOT = Path(__file__).resolve().parents[1]
for extra in (PROJECT_ROOT, DIFFSBDD_ROOT):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import constants
from dataset import ProcessedLigandPocketDataset
from edge_diffusion.oracle_reconstruction import OracleReconstructionResult, build_oracle_mol

from DiffSBDD.analysis.oracle_original_sdf_audit import (
    _atom_symbols,
    _compare_bond_maps,
    _count_vector_differences,
    _empty_comparison_fields,
    _load_original_sdf,
    _molecule_facts,
    ligand_sdf_relpath_from_name,
)

RDLogger.DisableLog("rdApp.*")

PAIR_SDF_NAME = "original_vs_repaired_oracle_pairs.sdf"
MISMATCH_SDF_NAME = "original_vs_repaired_oracle_smiles_mismatches.sdf"
MANIFEST_CSV_NAME = "original_vs_repaired_oracle_pair_manifest.csv"
SUMMARY_JSON_NAME = "original_vs_repaired_oracle_pair_summary.json"
REPORT_MD_NAME = "original_vs_repaired_oracle_pair_report.md"


@dataclass(frozen=True)
class PairSdfResult:
    """Paths and counts produced by paired SDF export."""

    report_path: Path
    all_pairs_sdf: Path
    mismatch_pairs_sdf: Path
    manifest_csv: Path
    summary_json: Path
    samples: int
    all_pair_records: int
    mismatch_pair_records: int


def format_pair_id(sample_idx: int) -> str:
    """Return a stable per-sample pair identifier."""
    return f"sample_{sample_idx:06d}"


def make_record_name(pair_id: str, variant: str, original_sdf: str) -> str:
    """Return the SDF molecule title used for one variant record."""
    stem = Path(original_sdf).stem if original_sdf else "unknown_ligand"
    safe_stem = _safe_sdf_title_part(stem)
    return f"{pair_id}_{variant}_{safe_stem}"


def _safe_sdf_title_part(value: str, max_length: int = 96) -> str:
    """Return an SDF title-safe token for broad external viewer compatibility."""
    safe_value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return (safe_value or "unknown")[:max_length]


def build_sdf_properties(
    row: dict[str, object],
    variant: str,
    args: argparse.Namespace,
) -> dict[str, object]:
    """Build SDF properties shared by original and repaired-oracle records."""
    if variant not in {"original", "repaired_oracle"}:
        raise ValueError(f"Unsupported variant: {variant}")

    other_variant = "repaired_oracle" if variant == "original" else "original"
    this_prefix = "original" if variant == "original" else "oracle"
    other_prefix = "oracle" if variant == "original" else "original"
    smiles_mismatch = int(row.get("smiles_no_stereo_match", 0)) == 0
    semantic_warning = (
        "no_stereo_smiles_mismatch_or_failed_load"
        if smiles_mismatch
        else "no_stereo_smiles_match"
    )

    properties: dict[str, object] = {
        "pair_id": row["pair_id"],
        "sample_idx": row["sample_idx"],
        "variant": variant,
        "counterpart_variant": other_variant,
        "record_name": row[f"{variant}_record_name"],
        "counterpart_record_name": row[f"{other_variant}_record_name"],
        "processed_name": row["name"],
        "original_sdf": row["original_sdf"],
        "npz_path": str(args.npz_path),
        "dataset": args.dataset,
        "repair_strategy": row["repair_strategy"],
        "initial_sanitize_error": row["initial_sanitize_error"],
        "oracle_sanitize_error": row["oracle_sanitize_error"],
        "oracle_sanitize_success": row["oracle_sanitize_success"],
        "original_load_success": row["original_load_success"],
        "original_load_error": row["original_load_error"],
        "visual_priority": "mismatch" if smiles_mismatch else "match",
        "semantic_warning": semantic_warning,
        "this_smiles_iso": row[f"{this_prefix}_smiles_iso"],
        "counterpart_smiles_iso": row[f"{other_prefix}_smiles_iso"],
        "this_smiles_no_stereo": row[f"{this_prefix}_smiles_no_stereo"],
        "counterpart_smiles_no_stereo": row[f"{other_prefix}_smiles_no_stereo"],
        "this_num_atoms": row[f"num_atoms_{this_prefix}"],
        "counterpart_num_atoms": row[f"num_atoms_{other_prefix}"],
        "this_num_bonds": row[f"num_bonds_{this_prefix}"],
        "counterpart_num_bonds": row[f"num_bonds_{other_prefix}"],
        "this_total_formal_charge": row[f"total_formal_charge_{this_prefix}"],
        "counterpart_total_formal_charge": row[f"total_formal_charge_{other_prefix}"],
        "this_aromatic_atom_count": row[f"aromatic_atom_count_{this_prefix}"],
        "counterpart_aromatic_atom_count": row[f"aromatic_atom_count_{other_prefix}"],
        "this_aromatic_bond_count": row[f"aromatic_bond_count_{this_prefix}"],
        "counterpart_aromatic_bond_count": row[f"aromatic_bond_count_{other_prefix}"],
    }

    for key in [
        "smiles_iso_match",
        "smiles_no_stereo_match",
        "num_atoms_match",
        "num_bonds_match",
        "total_formal_charge_match",
        "formal_charge_vector_match",
        "formal_charge_diff_count",
        "aromatic_atom_count_match",
        "aromatic_atom_vector_match",
        "aromatic_atom_diff_count",
        "aromatic_bond_count_match",
        "bond_pair_set_match",
        "bond_type_map_match",
        "missing_bond_count",
        "extra_bond_count",
        "bond_type_mismatch_count",
    ]:
        properties[f"comparison_{key}"] = row[key]
    return properties


def run_pair_sdf_export(args: argparse.Namespace) -> PairSdfResult:
    """Write paired SDF records and a manifest for visual comparison."""
    dataset_info = constants.dataset_params[args.dataset]
    dataset = ProcessedLigandPocketDataset(args.npz_path, center=False)
    args.outdir.mkdir(parents=True, exist_ok=True)
    limit = len(dataset) if args.max_samples is None else min(args.max_samples, len(dataset))

    all_pairs_sdf = args.outdir / PAIR_SDF_NAME
    mismatch_pairs_sdf = args.outdir / MISMATCH_SDF_NAME
    manifest_csv = args.outdir / MANIFEST_CSV_NAME
    summary_json = args.outdir / SUMMARY_JSON_NAME

    rows: list[dict[str, object]] = []
    repair_strategy_counts: Counter[str] = Counter()
    all_export_mode_counts: Counter[str] = Counter()
    mismatch_export_mode_counts: Counter[str] = Counter()
    mismatch_pair_count = 0
    all_record_count = 0
    mismatch_record_count = 0

    with all_pairs_sdf.open("w") as all_handle, mismatch_pairs_sdf.open("w") as mismatch_handle:
        for sample_idx in range(limit):
            row, original_mol, oracle_result = _build_pair_row(args, dataset_info, dataset, sample_idx)
            rows.append(row)
            repair_strategy_counts[str(row["repair_strategy"])] += 1

            if original_mol is None or oracle_result.mol is None:
                continue

            original_props = build_sdf_properties(row, "original", args)
            oracle_props = build_sdf_properties(row, "repaired_oracle", args)
            original_export_mode = _write_sdf_record(
                all_handle,
                original_mol,
                str(row["original_record_name"]),
                original_props,
                mol_block_mode=args.mol_block_mode,
                preserve_stereo_in_molblock=args.preserve_stereo_in_molblock,
            )
            oracle_export_mode = _write_sdf_record(
                all_handle,
                oracle_result.mol,
                str(row["repaired_oracle_record_name"]),
                oracle_props,
                mol_block_mode=args.mol_block_mode,
                preserve_stereo_in_molblock=args.preserve_stereo_in_molblock,
            )
            all_export_mode_counts[original_export_mode] += 1
            all_export_mode_counts[oracle_export_mode] += 1
            all_record_count += 2

            if int(row["smiles_no_stereo_match"]) == 0:
                mismatch_pair_count += 1
                original_mismatch_export_mode = _write_sdf_record(
                    mismatch_handle,
                    original_mol,
                    str(row["original_record_name"]),
                    original_props,
                    mol_block_mode=args.mol_block_mode,
                    preserve_stereo_in_molblock=args.preserve_stereo_in_molblock,
                )
                oracle_mismatch_export_mode = _write_sdf_record(
                    mismatch_handle,
                    oracle_result.mol,
                    str(row["repaired_oracle_record_name"]),
                    oracle_props,
                    mol_block_mode=args.mol_block_mode,
                    preserve_stereo_in_molblock=args.preserve_stereo_in_molblock,
                )
                mismatch_export_mode_counts[original_mismatch_export_mode] += 1
                mismatch_export_mode_counts[oracle_mismatch_export_mode] += 1
                mismatch_record_count += 2

    _write_manifest(manifest_csv, rows)
    summary = {
        "npz_path": str(args.npz_path),
        "raw_ligand_root": str(args.raw_ligand_root),
        "dataset": args.dataset,
        "mol_block_mode": args.mol_block_mode,
        "preserve_stereo_in_molblock": args.preserve_stereo_in_molblock,
        "samples": limit,
        "all_pairs_sdf": str(all_pairs_sdf),
        "all_pair_records": all_record_count,
        "mismatch_pairs_sdf": str(mismatch_pairs_sdf),
        "mismatch_pairs": mismatch_pair_count,
        "mismatch_pair_records": mismatch_record_count,
        "manifest_csv": str(manifest_csv),
        "repair_strategy_counts": dict(sorted(repair_strategy_counts.items())),
        "all_pairs_export_mode_counts": dict(sorted(all_export_mode_counts.items())),
        "mismatch_export_mode_counts": dict(sorted(mismatch_export_mode_counts.items())),
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    report_path = _write_report(args, summary)
    return PairSdfResult(
        report_path=report_path,
        all_pairs_sdf=all_pairs_sdf,
        mismatch_pairs_sdf=mismatch_pairs_sdf,
        manifest_csv=manifest_csv,
        summary_json=summary_json,
        samples=limit,
        all_pair_records=all_record_count,
        mismatch_pair_records=mismatch_record_count,
    )


def _build_pair_row(
    args: argparse.Namespace,
    dataset_info: dict[str, object],
    dataset: ProcessedLigandPocketDataset,
    sample_idx: int,
) -> tuple[dict[str, object], Chem.Mol | None, OracleReconstructionResult]:
    """Build one manifest row plus original/oracle molecule objects."""
    sample = dataset[sample_idx]
    name = str(sample["names"])
    pair_id = format_pair_id(sample_idx)
    row: dict[str, object] = {
        "pair_id": pair_id,
        "sample_idx": sample_idx,
        "name": name,
    }

    try:
        ligand_relpath = ligand_sdf_relpath_from_name(name)
        original_path = args.raw_ligand_root / ligand_relpath
        row["original_sdf"] = str(original_path)
    except ValueError as exc:
        original_path = None
        row["original_sdf"] = ""
        row["original_load_error"] = type(exc).__name__

    row["original_record_name"] = make_record_name(pair_id, "original", str(row["original_sdf"]))
    row["repaired_oracle_record_name"] = make_record_name(
        pair_id,
        "repaired_oracle",
        str(row["original_sdf"]),
    )

    atom_symbols = _atom_symbols(sample["lig_one_hot"], dataset_info["atom_decoder"])
    oracle_result = build_oracle_mol(
        atom_symbols=atom_symbols,
        positions=sample["lig_coords"].float(),
        halfedge_index=sample["ligand_halfedge_index"],
        halfedge_type=sample["ligand_halfedge_type"],
    )
    row["oracle_sanitize_success"] = int(oracle_result.sanitize_success)
    row["initial_sanitize_error"] = oracle_result.initial_sanitize_error or ""
    row["oracle_sanitize_error"] = oracle_result.sanitize_error or ""
    row["repair_strategy"] = oracle_result.repair_strategy

    original_mol = None
    original_error = row.get("original_load_error", "")
    if original_path is not None:
        original_mol, original_error = _load_original_sdf(original_path)
    row["original_load_success"] = int(original_mol is not None)
    row["original_load_error"] = original_error or ""

    if original_mol is None or oracle_result.mol is None or not oracle_result.sanitize_success:
        row.update(_empty_comparison_fields())
        return row, original_mol, oracle_result

    original = _molecule_facts(original_mol)
    oracle = _molecule_facts(oracle_result.mol)
    bond_comparison = _compare_bond_maps(original.bond_map, oracle.bond_map)
    formal_charge_diff_count = _count_vector_differences(
        original.formal_charges,
        oracle.formal_charges,
    )
    aromatic_atom_diff_count = _count_vector_differences(
        original.aromatic_atom_flags,
        oracle.aromatic_atom_flags,
    )
    row.update(
        {
            "original_smiles_iso": original.smiles_iso,
            "oracle_smiles_iso": oracle.smiles_iso,
            "original_smiles_no_stereo": original.smiles_no_stereo,
            "oracle_smiles_no_stereo": oracle.smiles_no_stereo,
            "smiles_iso_match": int(original.smiles_iso == oracle.smiles_iso),
            "smiles_no_stereo_match": int(original.smiles_no_stereo == oracle.smiles_no_stereo),
            "num_atoms_original": original.num_atoms,
            "num_atoms_oracle": oracle.num_atoms,
            "num_atoms_match": int(original.num_atoms == oracle.num_atoms),
            "num_bonds_original": original.num_bonds,
            "num_bonds_oracle": oracle.num_bonds,
            "num_bonds_match": int(original.num_bonds == oracle.num_bonds),
            "total_formal_charge_original": original.total_formal_charge,
            "total_formal_charge_oracle": oracle.total_formal_charge,
            "total_formal_charge_match": int(
                original.total_formal_charge == oracle.total_formal_charge
            ),
            "formal_charge_vector_match": int(original.formal_charges == oracle.formal_charges),
            "formal_charge_diff_count": formal_charge_diff_count,
            "aromatic_atom_count_original": original.aromatic_atom_count,
            "aromatic_atom_count_oracle": oracle.aromatic_atom_count,
            "aromatic_atom_count_match": int(
                original.aromatic_atom_count == oracle.aromatic_atom_count
            ),
            "aromatic_atom_vector_match": int(
                original.aromatic_atom_flags == oracle.aromatic_atom_flags
            ),
            "aromatic_atom_diff_count": aromatic_atom_diff_count,
            "aromatic_bond_count_original": original.aromatic_bond_count,
            "aromatic_bond_count_oracle": oracle.aromatic_bond_count,
            "aromatic_bond_count_match": int(
                original.aromatic_bond_count == oracle.aromatic_bond_count
            ),
            **bond_comparison,
        }
    )
    return row, original_mol, oracle_result


def _write_sdf_record(
    handle,
    mol: Chem.Mol,
    name: str,
    properties: dict[str, object],
    mol_block_mode: str,
    preserve_stereo_in_molblock: bool,
) -> str:
    """Write one SDF record with a stable molecule title and properties."""
    mol_to_write = Chem.Mol(mol)
    mol_to_write.SetProp("_Name", name)
    stereo_display = "preserved" if preserve_stereo_in_molblock else "removed_for_display"
    if not preserve_stereo_in_molblock:
        Chem.RemoveStereochemistry(mol_to_write)
    mol_block, export_mode, export_warning = _mol_to_mol_block(
        mol_to_write,
        mode=mol_block_mode,
    )
    record_properties = {
        **properties,
        "molblock_export_mode": export_mode,
        "molblock_export_warning": export_warning,
        "molblock_stereo_display": stereo_display,
    }
    handle.write(mol_block)
    for key, value in record_properties.items():
        handle.write(f">  <{key}>\n{_format_sdf_value(value)}\n\n")
    handle.write("$$$$\n")
    return export_mode


def _mol_to_mol_block(mol: Chem.Mol, mode: str) -> tuple[str, str, str]:
    """Return a MolBlock plus export diagnostics for the selected writer mode."""
    if mode == "rdkit_aromatic":
        return _ensure_mol_block_newline(_mol_to_rdkit_aromatic_block(mol)), "rdkit_aromatic", ""
    if mode != "maestro":
        raise ValueError(f"Unsupported mol block mode: {mode}")

    try:
        return _ensure_mol_block_newline(Chem.MolToMolBlock(mol, kekulize=True)), "kekulized", ""
    except Exception as exc:  # noqa: BLE001 - RDKit exposes several kekulization exceptions
        fallback = _copy_with_aromatic_bonds_as_single(mol)
        warning = f"{type(exc).__name__}: aromatic bonds exported as single for display"
        return _ensure_mol_block_newline(Chem.MolToMolBlock(fallback, kekulize=False)), (
            "aromatic_to_single_fallback"
        ), warning


def _mol_to_rdkit_aromatic_block(mol: Chem.Mol) -> str:
    """Return the old RDKit-oriented aromatic MolBlock representation."""
    try:
        return Chem.MolToMolBlock(mol, kekulize=False)
    except TypeError:
        return Chem.MolToMolBlock(mol)


def _copy_with_aromatic_bonds_as_single(mol: Chem.Mol) -> Chem.Mol:
    """Return a display-only copy with aromatic flags cleared and bonds set single."""
    candidate = Chem.RWMol(Chem.Mol(mol))
    for bond in candidate.GetBonds():
        if bond.GetIsAromatic() or bond.GetBondType() == Chem.rdchem.BondType.AROMATIC:
            bond.SetBondType(Chem.rdchem.BondType.SINGLE)
            bond.SetIsAromatic(False)
    for atom in candidate.GetAtoms():
        atom.SetIsAromatic(False)
    return candidate.GetMol()


def _ensure_mol_block_newline(mol_block: str) -> str:
    """Ensure MolBlock text ends with one newline before properties."""
    if not mol_block.endswith("\n"):
        mol_block += "\n"
    return mol_block


def _format_sdf_value(value: object) -> str:
    """Format values as single SDF property strings."""
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).replace("\r", " ").replace("\n", " ")


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    """Write the paired SDF manifest with stable columns."""
    fieldnames = [
        "pair_id",
        "sample_idx",
        "name",
        "original_sdf",
        "original_record_name",
        "repaired_oracle_record_name",
        "original_load_success",
        "original_load_error",
        "oracle_sanitize_success",
        "initial_sanitize_error",
        "oracle_sanitize_error",
        "repair_strategy",
        "original_smiles_iso",
        "oracle_smiles_iso",
        "original_smiles_no_stereo",
        "oracle_smiles_no_stereo",
        "smiles_iso_match",
        "smiles_no_stereo_match",
        "num_atoms_original",
        "num_atoms_oracle",
        "num_atoms_match",
        "num_bonds_original",
        "num_bonds_oracle",
        "num_bonds_match",
        "total_formal_charge_original",
        "total_formal_charge_oracle",
        "total_formal_charge_match",
        "formal_charge_vector_match",
        "formal_charge_diff_count",
        "aromatic_atom_count_original",
        "aromatic_atom_count_oracle",
        "aromatic_atom_count_match",
        "aromatic_atom_vector_match",
        "aromatic_atom_diff_count",
        "aromatic_bond_count_original",
        "aromatic_bond_count_oracle",
        "aromatic_bond_count_match",
        "bond_pair_set_match",
        "bond_type_map_match",
        "missing_bond_count",
        "extra_bond_count",
        "bond_type_mismatch_count",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_report(args: argparse.Namespace, summary: dict[str, object]) -> Path:
    """Write a Markdown report describing the paired SDF outputs."""
    report_path = args.outdir / REPORT_MD_NAME
    mismatch_pairs = int(summary["mismatch_pairs"])
    samples = int(summary["samples"])
    with report_path.open("w") as handle:
        handle.write("# Original vs Repaired Oracle Paired SDF\n\n")
        handle.write("## Purpose\n\n")
        handle.write(
            "This export writes original ligand SDF records next to repaired oracle records "
            "built from processed coordinates and true halfedge labels. It is meant for "
            "visual inspection of reconstruction/repair differences.\n\n"
        )
        handle.write("## Inputs\n\n")
        handle.write(f"- npz_path: `{args.npz_path}`\n")
        handle.write(f"- raw_ligand_root: `{args.raw_ligand_root}`\n")
        handle.write(f"- dataset: `{args.dataset}`\n")
        handle.write(f"- mol_block_mode: `{args.mol_block_mode}`\n")
        handle.write(f"- preserve_stereo_in_molblock: `{args.preserve_stereo_in_molblock}`\n")
        handle.write(f"- samples: {samples}\n\n")
        handle.write("## Outputs\n\n")
        handle.write(f"- [all paired SDF]({PAIR_SDF_NAME}): {summary['all_pair_records']} records\n")
        handle.write(
            f"- [SMILES mismatch paired SDF]({MISMATCH_SDF_NAME}): "
            f"{summary['mismatch_pair_records']} records from {mismatch_pairs} pairs\n"
        )
        handle.write(f"- [manifest CSV]({MANIFEST_CSV_NAME})\n")
        handle.write(f"- [summary JSON]({SUMMARY_JSON_NAME})\n\n")
        handle.write("## Record Layout\n\n")
        handle.write(
            "Each pair is written as two consecutive records: first `original`, then "
            "`repaired_oracle`. The SDF molecule title uses a Maestro-safe form: "
            "`sample_XXXXXX_variant_ligand_stem`.\n\n"
        )
        handle.write("Important properties on every record include:\n\n")
        handle.write("- `pair_id`, `sample_idx`, `variant`, `counterpart_variant`\n")
        handle.write("- `processed_name`, `original_sdf`, `repair_strategy`\n")
        handle.write("- `this_smiles_*`, `counterpart_smiles_*`\n")
        handle.write("- `comparison_smiles_no_stereo_match`, `comparison_bond_type_map_match`\n")
        handle.write("- `molblock_export_mode`, `molblock_export_warning`\n")
        handle.write("- `molblock_stereo_display`\n")
        handle.write("- `semantic_warning`, `visual_priority`\n\n")
        handle.write("## Repair Strategy Counts\n\n")
        handle.write("| repair_strategy | count |\n")
        handle.write("|---|---:|\n")
        for strategy, count in dict(summary["repair_strategy_counts"]).items():
            handle.write(f"| {strategy} | {count} |\n")
        handle.write("\n## MolBlock Export Mode Counts\n\n")
        handle.write("| export_mode | all paired records | mismatch records |\n")
        handle.write("|---|---:|---:|\n")
        all_counts = dict(summary["all_pairs_export_mode_counts"])
        mismatch_counts = dict(summary["mismatch_export_mode_counts"])
        for export_mode in sorted(set(all_counts) | set(mismatch_counts)):
            handle.write(
                f"| {export_mode} | {all_counts.get(export_mode, 0)} | "
                f"{mismatch_counts.get(export_mode, 0)} |\n"
            )
        handle.write("\n## Interpretation Warning\n\n")
        handle.write(
            "The `repaired_oracle` records are visual comparison objects, not guaranteed "
            "faithful target molecules. Previous audit showed connectivity/atom counts are "
            "aligned, but repaired molecules can change aromatic/bond/charge semantics after "
            "RDKit sanitization. Prioritize the mismatch SDF for manual inspection.\n"
        )
        if args.mol_block_mode == "maestro":
            handle.write(
                "\nThis export uses Maestro-compatible MolBlocks by default: normal records are "
                "Kekulized, and records that RDKit cannot Kekulize are written with aromatic "
                "bonds converted to single bonds for display only.\n"
            )
            if not args.preserve_stereo_in_molblock:
                handle.write(
                    "SDF MolBlock stereochemical wedge/hash annotations are removed for display "
                    "compatibility; isomeric SMILES are preserved in SDF properties.\n"
                )
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
        / f"oracle_original_pair_sdf_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--mol-block-mode",
        choices=["maestro", "rdkit_aromatic"],
        default="maestro",
        help=(
            "maestro writes Kekule/single-display MolBlocks without aromatic bond type 4; "
            "rdkit_aromatic preserves RDKit aromatic bond type 4."
        ),
    )
    parser.add_argument(
        "--preserve-stereo-in-molblock",
        action="store_true",
        help=(
            "Keep SDF MolBlock chiral/wedge annotations. By default they are removed for "
            "Maestro/Open Babel display compatibility while isomeric SMILES stay in properties."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    result = run_pair_sdf_export(parse_args())
    print(f"Wrote paired SDF report to {result.report_path}")
