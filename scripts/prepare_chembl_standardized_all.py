#!/usr/bin/env python3
"""Build an uncropped ChEMBL reference set after official-style standardization.

This utility intentionally performs no molecular-weight, heavy-atom, charge,
element, or drug-likeness filtering.  A molecule is retained only when it can
be parsed, standardized, reduced to its ChEMBL parent, sanitized by RDKit, and
represented by a non-empty canonical isomeric SMILES.  Exact duplicates are
removed after that process.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import multiprocessing as mp
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterator

from chembl_structure_pipeline import standardizer
from rdkit import Chem, RDLogger, rdBase


LOGGER = logging.getLogger("prepare_chembl_standardized_all")
RDLogger.DisableLog("rdApp.*")

DEFAULT_NUM_WORKERS = 30
DEFAULT_CHUNKSIZE = 1_000
OUTPUT_FIELDS = ("chembl_id", "canonical_smiles")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for a standardized ChEMBL build."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite-path", required=True, help="Extracted ChEMBL SQLite database.")
    parser.add_argument("--out-csv", required=True, help="Destination CSV.GZ with unique molecules.")
    parser.add_argument("--summary-json", required=True, help="Destination JSON provenance summary.")
    parser.add_argument("--log-path", required=True, help="Destination durable run log.")
    parser.add_argument(
        "--max-input-records",
        type=int,
        default=0,
        help="Process at most this many raw structure records; 0 means the complete database.",
    )
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--chunksize", type=int, default=DEFAULT_CHUNKSIZE)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def configure_logging(log_path: Path) -> None:
    """Configure matching console and durable file logs."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    LOGGER.addHandler(stream_handler)
    LOGGER.addHandler(file_handler)


def count_input_rows(sqlite_path: Path) -> int:
    """Return the number of ChEMBL rows with a stored canonical SMILES."""
    query = """
        SELECT COUNT(*)
        FROM molecule_dictionary AS md
        JOIN compound_structures AS cs ON md.molregno = cs.molregno
        WHERE cs.canonical_smiles IS NOT NULL
    """
    with sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True) as connection:
        return int(connection.execute(query).fetchone()[0])


def iter_input_rows(sqlite_path: Path, max_records: int) -> Iterator[tuple[str, str]]:
    """Yield raw ChEMBL identifiers and SMILES in a deterministic database order."""
    query = """
        SELECT md.chembl_id, cs.canonical_smiles
        FROM molecule_dictionary AS md
        JOIN compound_structures AS cs ON md.molregno = cs.molregno
        WHERE cs.canonical_smiles IS NOT NULL
        ORDER BY md.molregno
    """
    if max_records > 0:
        query += " LIMIT ?"
        parameters: tuple[int, ...] = (max_records,)
    else:
        parameters = ()
    connection = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        for chembl_id, smiles in connection.execute(query, parameters):
            yield str(chembl_id), str(smiles)
    finally:
        connection.close()


def select_parent_fragment(molecule: Chem.Mol) -> tuple[Chem.Mol | None, bool]:
    """Select the largest organic raw fragment before ChEMBL parent processing.

    ``get_parent_mol`` deliberately retains some excluded multi-fragment entries.
    The project-wide definition of a desalted parent instead begins with the
    largest organic component, mirroring the established local ChEMBL pipeline.
    """
    fragments = Chem.GetMolFrags(molecule, asMols=True, sanitizeFrags=False)
    if not fragments:
        return None, False
    selected = max(
        fragments,
        key=lambda fragment: (
            any(atom.GetAtomicNum() == 6 for atom in fragment.GetAtoms()),
            fragment.GetNumHeavyAtoms(),
            fragment.GetNumAtoms(),
        ),
    )
    return selected, len(fragments) > 1


def standardize_row(row: tuple[str, str]) -> tuple[dict[str, str] | None, str, bool, bool]:
    """Return one parent-standardized, RDKit-valid row or an explicit failure type."""
    chembl_id, smiles = row
    molecule = Chem.MolFromSmiles(smiles, sanitize=False)
    if molecule is None:
        return None, "parse_failed", False, False
    molecule, had_multiple_fragments = select_parent_fragment(molecule)
    if molecule is None:
        return None, "fragment_selection_failed", False, False
    try:
        molecule = standardizer.standardize_mol(molecule)
        parent, excluded = standardizer.get_parent_mol(molecule)
    except Exception:
        return None, "standardize_or_parent_failed", False, had_multiple_fragments
    if parent is None:
        return None, "parent_missing", False, had_multiple_fragments
    try:
        Chem.SanitizeMol(parent)
    except Exception:
        return None, "rdkit_sanitize_failed", bool(excluded), had_multiple_fragments
    try:
        canonical_smiles = Chem.MolToSmiles(
            Chem.RemoveHs(parent), canonical=True, isomericSmiles=True
        )
    except Exception:
        return None, "canonicalize_failed", bool(excluded), had_multiple_fragments
    if not canonical_smiles:
        return None, "canonicalize_failed", bool(excluded), had_multiple_fragments
    return {
        "chembl_id": chembl_id,
        "canonical_smiles": canonical_smiles,
    }, "standardized_parent_sanitize_valid", bool(excluded), had_multiple_fragments


def ensure_output_targets(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    """Validate output targets before starting a potentially long scan."""
    out_csv = Path(args.out_csv).expanduser().resolve()
    summary_json = Path(args.summary_json).expanduser().resolve()
    log_path = Path(args.log_path).expanduser().resolve()
    for path in (out_csv, summary_json):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    return out_csv, summary_json, log_path


def write_summary(
    summary_path: Path,
    args: argparse.Namespace,
    sqlite_path: Path,
    total_input_rows: int,
    counts: Counter[str],
    elapsed_seconds: float,
) -> None:
    """Write a provenance-rich summary that states the absence of property filters."""
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "complete",
        "sqlite_path": str(sqlite_path),
        "input_rows_total": total_input_rows,
        "max_input_records": args.max_input_records,
        "num_workers": args.num_workers,
        "chunksize": args.chunksize,
        "rdkit_version": rdBase.rdkitVersion,
        "standardizer": "chembl_structure_pipeline.standardizer",
        "retention_rule": [
            "parsed by RDKit",
            "standardized with chembl_structure_pipeline",
            "reduced to ChEMBL parent (salts removed)",
            "RDKit sanitized",
            "canonical isomeric SMILES non-empty",
            "canonical isomeric SMILES exact deduplication",
        ],
        "property_filters": {
            "molecular_weight": None,
            "heavy_atom_count": None,
            "formal_charge": None,
            "element_set": None,
            "drug_likeness": None,
        },
        "fragment_parent_rule": (
            "For disconnected raw records, retain the largest organic fragment by "
            "(contains carbon, heavy atoms, atom count) before ChEMBL standardization."
        ),
        "counts": dict(sorted(counts.items())),
        "elapsed_seconds": round(elapsed_seconds, 3),
    }
    temporary_path = summary_path.with_suffix(summary_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(summary_path)


def run(args: argparse.Namespace) -> None:
    """Standardize, validate, and deduplicate ChEMBL records into a CSV.GZ file."""
    sqlite_path = Path(args.sqlite_path).expanduser().resolve()
    if not sqlite_path.is_file():
        raise FileNotFoundError(f"ChEMBL SQLite database not found: {sqlite_path}")
    if args.max_input_records < 0:
        raise ValueError("--max-input-records must be zero (all) or a positive integer")
    if args.num_workers < 1:
        raise ValueError("--num-workers must be at least 1")
    if args.chunksize < 1:
        raise ValueError("--chunksize must be at least 1")

    out_csv, summary_json, _ = ensure_output_targets(args)
    total_input_rows = count_input_rows(sqlite_path)
    planned_rows = min(args.max_input_records, total_input_rows) if args.max_input_records else total_input_rows
    LOGGER.info(
        "Starting standardized-all build: planned_rows=%d total_rows=%d workers=%d output=%s",
        planned_rows,
        total_input_rows,
        args.num_workers,
        out_csv,
    )
    started_at = time.monotonic()
    counts: Counter[str] = Counter()
    seen_smiles: set[str] = set()
    temporary_csv = out_csv.with_suffix(out_csv.suffix + ".tmp")

    with gzip.open(temporary_csv, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        with mp.Pool(processes=args.num_workers) as pool:
            rows = iter_input_rows(sqlite_path, args.max_input_records)
            for index, (record, reason, parent_excluded, had_multiple_fragments) in enumerate(
                pool.imap(standardize_row, rows, chunksize=args.chunksize), start=1
            ):
                counts["input_rows_processed"] += 1
                counts[reason] += 1
                if parent_excluded:
                    counts["parent_excluded_by_pipeline"] += 1
                if had_multiple_fragments:
                    counts["raw_multifragment_records"] += 1
                if record is None:
                    continue
                canonical_smiles = record["canonical_smiles"]
                if canonical_smiles in seen_smiles:
                    counts["duplicate_canonical_smiles"] += 1
                    continue
                seen_smiles.add(canonical_smiles)
                writer.writerow(record)
                counts["unique_output_rows"] += 1
                if index % 100_000 == 0 or index == planned_rows:
                    LOGGER.info(
                        "Progress input=%d/%d standardized=%d unique=%d duplicates=%d",
                        index,
                        planned_rows,
                        counts["standardized_parent_sanitize_valid"],
                        counts["unique_output_rows"],
                        counts["duplicate_canonical_smiles"],
                    )

    temporary_csv.replace(out_csv)
    elapsed_seconds = time.monotonic() - started_at
    write_summary(summary_json, args, sqlite_path, total_input_rows, counts, elapsed_seconds)
    LOGGER.info(
        "Completed standardized-all build: input=%d unique=%d elapsed_seconds=%.1f",
        counts["input_rows_processed"],
        counts["unique_output_rows"],
        elapsed_seconds,
    )


def main() -> int:
    """Run the command-line program and record unexpected errors in its log."""
    args = parse_args()
    log_path = Path(args.log_path).expanduser().resolve()
    configure_logging(log_path)
    LOGGER.info("Command: %s", " ".join(sys.argv))
    try:
        run(args)
    except Exception:
        LOGGER.exception("Standardized-all build failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
