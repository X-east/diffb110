#!/usr/bin/env python3
"""Download ChEMBL SQLite and build cleaned ChEMBL fingerprint caches."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import multiprocessing as mp
import pickle
import random
import shutil
import sqlite3
import subprocess
import time
import urllib.request
from functools import partial
from pathlib import Path
from typing import Iterable

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from tqdm import tqdm

from chembl_structure_pipeline import standardizer

RDLogger.DisableLog("rdApp.*")
LOGGER = logging.getLogger("prepare_chembl")

# ── 默认参数（所有可调值集中在此处）───────────────────────────────
DEFAULT_VERSION = "36"
DEFAULT_BASE_URL = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases"
DEFAULT_OUT_DIR = "data/chembl"
DEFAULT_MAX_MOLECULES = 500_000
DEFAULT_SAMPLE_SEED = 20260515
DEFAULT_RADIUS = 2
DEFAULT_N_BITS = 2048
DEFAULT_FINGERPRINT_TYPES = ["ecfp4", "fcfp4"]
DEFAULT_MIN_MOL_WT = 150.0
DEFAULT_MAX_MOL_WT = 500.0
DEFAULT_MIN_HEAVY_ATOMS = 10
DEFAULT_MAX_HEAVY_ATOMS = 30
DEFAULT_SMOKE_LIMIT = 1_000
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_NUM_WORKERS = 12
CLEAN_POOL_CHUNKSIZE = 100        # 清洗并行时每个 worker 一次领取的任务数
FINGERPRINT_POOL_CHUNKSIZE = 500  # 指纹并行时每个 worker 一次领取的任务数


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chembl-version", default=DEFAULT_VERSION)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--archive-path",
        default=None,
        help="Optional local chembl_<version>_sqlite.tar.gz path. Useful when direct download is blocked.",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--max-molecules",
        type=int,
        default=DEFAULT_MAX_MOLECULES,
        help="Maximum accepted molecules to keep; use 0 or a negative value for all cleaned molecules.",
    )
    parser.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument("--radius", type=int, default=DEFAULT_RADIUS)
    parser.add_argument("--n-bits", type=int, default=DEFAULT_N_BITS)
    parser.add_argument(
        "--fingerprint-types",
        nargs="+",
        choices=["ecfp4", "fcfp4"],
        default=DEFAULT_FINGERPRINT_TYPES,
        help="Fingerprint cache(s) to build. ecfp4 is Morgan radius 2; fcfp4 uses Morgan feature invariants.",
    )
    parser.add_argument(
        "--output-tag",
        default=None,
        help="Optional tag inserted into output filenames, e.g. 'all' -> chembl_36_all_ecfp4_fingerprints.pkl.gz.",
    )
    parser.add_argument(
        "--cleaned-records-path",
        default=None,
        help="Optional cleaned molecule CSV.GZ path to write/read. Defaults to the metadata path for this run.",
    )
    parser.add_argument(
        "--reuse-cleaned-records",
        action="store_true",
        help="Skip SQLite cleaning and reuse --cleaned-records-path when it already exists.",
    )
    parser.add_argument("--min-mol-wt", type=float, default=DEFAULT_MIN_MOL_WT)
    parser.add_argument("--max-mol-wt", type=float, default=DEFAULT_MAX_MOL_WT)
    parser.add_argument("--min-heavy-atoms", type=int, default=DEFAULT_MIN_HEAVY_ATOMS)
    parser.add_argument("--max-heavy-atoms", type=int, default=DEFAULT_MAX_HEAVY_ATOMS)
    parser.add_argument("--allow-charged", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-limit", type=int, default=DEFAULT_SMOKE_LIMIT)
    parser.add_argument("--log-level", default=DEFAULT_LOG_LEVEL, choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="Number of worker processes for parallel computation. 0 = all CPUs (not recommended for large datasets).",
    )
    return parser.parse_args()


def paths_for(out_dir: Path, version: str) -> dict[str, Path]:
    release = f"chembl_{version}"
    return {
        "archive": out_dir / f"{release}_sqlite.tar.gz",
        "extract_dir": out_dir / release,
        "sqlite": out_dir / release / f"{release}_sqlite" / f"{release}.db",
        "fingerprints": out_dir / f"{release}_morgan_fingerprints.pkl.gz",
        "metadata": out_dir / f"{release}_molecules.csv.gz",
        "summary": out_dir / f"{release}_summary.json",
    }


def fingerprint_cache_path(out_dir: Path, version: str, fingerprint_type: str, args: argparse.Namespace) -> Path:
    release = f"chembl_{version}"
    if args.output_tag:
        return out_dir / f"{release}_{args.output_tag}_{fingerprint_type}_fingerprints.pkl.gz"
    if fingerprint_type == "ecfp4" and list(args.fingerprint_types) == ["ecfp4"]:
        return out_dir / f"{release}_morgan_fingerprints.pkl.gz"
    return out_dir / f"{release}_{fingerprint_type}_fingerprints.pkl.gz"


def metadata_path(out_dir: Path, version: str, args: argparse.Namespace) -> Path:
    release = f"chembl_{version}"
    if args.output_tag:
        return out_dir / f"{release}_{args.output_tag}_molecules.csv.gz"
    return out_dir / f"{release}_molecules.csv.gz"


def summary_path(out_dir: Path, version: str, args: argparse.Namespace) -> Path:
    release = f"chembl_{version}"
    if args.output_tag:
        return out_dir / f"{release}_{args.output_tag}_summary.json"
    return out_dir / f"{release}_summary.json"


def cleaned_records_path(out_dir: Path, version: str, args: argparse.Namespace) -> Path:
    if args.cleaned_records_path:
        return Path(args.cleaned_records_path).expanduser().resolve()
    return metadata_path(out_dir, version, args)


def copy_local_archive(archive_path: str, dst: Path) -> None:
    src = Path(archive_path).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"Local ChEMBL archive does not exist: {src}")
    if src == dst.resolve():
        LOGGER.info("Using archive already in output directory: %s", dst)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Copying local archive %s -> %s", src, dst)
    shutil.copy2(src, dst)


def log_manual_download_help(url: str, archive_path: Path, sqlite_path: Path, exc: Exception) -> None:
    LOGGER.error("Could not download ChEMBL archive: %s", exc)
    LOGGER.error("If this host cannot access the ChEMBL FTP/HTTPS endpoint, download manually and resume.")
    LOGGER.error("URL: %s", url)
    LOGGER.error("Expected archive path: %s", archive_path)
    LOGGER.error("After manual download, rerun the same command or pass --archive-path /path/to/chembl_XX_sqlite.tar.gz")
    LOGGER.error("Alternatively, extract it so the SQLite DB exists at: %s", sqlite_path)


def download_file(url: str, dst: Path, force: bool) -> None:
    if dst.exists() and dst.stat().st_size > 0 and not force:
        LOGGER.info("Using existing archive: %s", dst)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    LOGGER.info("Downloading %s", url)
    with urllib.request.urlopen(url, timeout=60) as response, tmp.open("wb") as handle:
        total = int(response.headers.get("Content-Length", "0") or 0)
        with tqdm(total=total, unit="B", unit_scale=True, desc="download") as bar:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
                bar.update(len(chunk))
    tmp.replace(dst)


def extract_archive(archive: Path, extract_dir: Path, sqlite_path: Path, force: bool) -> None:
    if sqlite_path.exists() and not force:
        LOGGER.info("Using existing SQLite DB: %s", sqlite_path)
        return
    if extract_dir.exists() and force:
        shutil.rmtree(extract_dir)
    subprocess.run(["tar", "-xzf", str(archive), "-C", str(extract_dir.parent)], check=True)
    if not sqlite_path.exists():
        raise FileNotFoundError(f"SQLite DB not found after extraction: {sqlite_path}")


def iter_smiles(sqlite_path: Path) -> Iterable[tuple[str, str]]:
    query = """
        SELECT md.chembl_id, cs.canonical_smiles
        FROM molecule_dictionary AS md
        JOIN compound_structures AS cs ON md.molregno = cs.molregno
        WHERE cs.canonical_smiles IS NOT NULL
    """
    conn = sqlite3.connect(str(sqlite_path))
    try:
        yield from ((str(cid), str(smi)) for cid, smi in conn.execute(query))
    finally:
        conn.close()


def sanitize_molecule(smiles: str):
    """Sanitize and standardize a molecule using the ChEMBL Structure Pipeline.

    Steps:
    1. Parse SMILES (lazy, no sanitize).
    2. Get largest fragment (removes salts, solvents).
    3. Apply ChEMBL standardizer (``standardize_mol`` handles valence/aromaticity internally).
    4. Strip salts via ``get_parent_mol``.
    """
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if not fragments:
        return None
    mol = max(fragments, key=lambda item: item.GetNumHeavyAtoms())
    try:
        mol = standardizer.standardize_mol(mol)
        parent_mol, _ = standardizer.get_parent_mol(mol)
        if parent_mol is None:
            parent_mol = mol
    except Exception:
        return None
    return parent_mol


def filter_reason(mol, args: argparse.Namespace) -> str:
    heavy_atoms = mol.GetNumHeavyAtoms()
    mol_wt = float(Descriptors.MolWt(mol))
    if heavy_atoms < args.min_heavy_atoms:
        return "too_few_heavy_atoms"
    if heavy_atoms > args.max_heavy_atoms:
        return "too_many_heavy_atoms"
    if mol_wt < args.min_mol_wt:
        return "mol_wt_too_low"
    if mol_wt > args.max_mol_wt:
        return "mol_wt_too_high"
    if not args.allow_charged and Chem.GetFormalCharge(mol) != 0:
        return "charged"
    if any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
        return "dummy_atom"
    return "ok"


def _sanitize_and_filter(
    filter_args: tuple[float, float, int, int, bool], row: tuple[str, str]
) -> tuple[dict | None, str]:
    """Worker function: sanitize → filter → canonicalize.

    Designed for ``multiprocessing.Pool.imap`` with ``functools.partial``.
    Returns ``(record_dict, "ok")`` on success, ``(None, reason)`` otherwise.
    """
    chembl_id, smiles = row
    min_mol_wt, max_mol_wt, min_heavy_atoms, max_heavy_atoms, allow_charged = filter_args
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None, "parse_failed"
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if not fragments:
        return None, "parse_failed"
    mol = max(fragments, key=lambda item: item.GetNumHeavyAtoms())
    try:
        mol = standardizer.standardize_mol(mol)
        parent_mol, _ = standardizer.get_parent_mol(mol)
        if parent_mol is None:
            parent_mol = mol
    except Exception:
        return None, "standardize_failed"
    heavy_atoms = parent_mol.GetNumHeavyAtoms()
    mol_wt = float(Descriptors.MolWt(parent_mol))
    if heavy_atoms < min_heavy_atoms:
        return None, "too_few_heavy_atoms"
    if heavy_atoms > max_heavy_atoms:
        return None, "too_many_heavy_atoms"
    if mol_wt < min_mol_wt:
        return None, "mol_wt_too_low"
    if mol_wt > max_mol_wt:
        return None, "mol_wt_too_high"
    if not allow_charged and Chem.GetFormalCharge(parent_mol) != 0:
        return None, "charged"
    if any(atom.GetAtomicNum() == 0 for atom in parent_mol.GetAtoms()):
        return None, "dummy_atom"
    Chem.FastFindRings(parent_mol)
    canonical = Chem.MolToSmiles(Chem.RemoveHs(parent_mol), isomericSmiles=False, canonical=True)
    record = {
        "chembl_id": chembl_id,
        "canonical_smiles": canonical,
        "mol_wt": round(mol_wt, 4),
        "heavy_atoms": heavy_atoms,
        "formal_charge": int(Chem.GetFormalCharge(parent_mol)),
        "num_rings": int(rdMolDescriptors.CalcNumRings(parent_mol)),
    }
    return record, "ok"


def build_records(
    sqlite_path: Path, args: argparse.Namespace
) -> tuple[list[dict], dict[str, int]]:
    """Clean and filter molecules from ChEMBL SQLite.

    Parallelizes the CPU-heavy sanitize+filter step via multiprocessing.
    Dedup (canonical SMILES) and reservoir sampling remain serial.
    """
    rng = random.Random(args.sample_seed)
    limit = args.smoke_limit if args.smoke_test else args.max_molecules
    keep_all = limit <= 0

    selected: list[dict] = []
    seen: set[str] = set()
    counts: dict[str, int] = {
        "input_rows": 0, "parse_failed": 0, "standardize_failed": 0,
        "duplicates": 0, "accepted_before_sampling": 0,
    }

    # ── Count total input rows ──────────────────────────────────────────
    conn = sqlite3.connect(str(sqlite_path))
    total_input = conn.execute("""
        SELECT COUNT(*) FROM molecule_dictionary AS md
        JOIN compound_structures AS cs ON md.molregno = cs.molregno
        WHERE cs.canonical_smiles IS NOT NULL
    """).fetchone()[0]
    conn.close()

    # ── Worker setup ────────────────────────────────────────────────────
    filter_args = (
        args.min_mol_wt, args.max_mol_wt,
        args.min_heavy_atoms, args.max_heavy_atoms,
        args.allow_charged,
    )
    worker_fn = partial(_sanitize_and_filter, filter_args)
    num_workers = args.num_workers
    if num_workers == 0:
        num_workers = mp.cpu_count()
    num_workers = max(1, min(num_workers, total_input))

    LOGGER.info("Cleaning %d molecules with %d workers", total_input, num_workers)

    # ── Accept helper (count + dedup + reservoir sample) ────────────────
    def _accept_record(record: dict | None, reason: str) -> None:
        if record is None:
            counts[reason] = counts.get(reason, 0) + 1
            return
        smi = record["canonical_smiles"]
        if smi in seen:
            counts["duplicates"] += 1
            return
        seen.add(smi)
        counts["accepted_before_sampling"] += 1
        if keep_all or len(selected) < limit:
            selected.append(record)
        else:
            idx = rng.randrange(counts["accepted_before_sampling"])
            if idx < limit:
                selected[idx] = record

    # ── Pre-load all rows (close SQLite before any multiprocessing) ──────
    rows = list(iter_smiles(sqlite_path))

    # ── Main loop ───────────────────────────────────────────────────────
    if num_workers <= 1 or args.smoke_test:
        max_smoke = max(args.smoke_limit * 20, args.smoke_limit) if args.smoke_test else total_input
        for i, (chembl_id, smiles) in enumerate(
            tqdm(rows[:max_smoke] if args.smoke_test else rows, desc="clean molecules")
        ):
            counts["input_rows"] = i + 1
            record, reason = _sanitize_and_filter(filter_args, (chembl_id, smiles))
            _accept_record(record, reason)
            if args.smoke_test and i + 1 >= max_smoke:
                break
    else:
        with mp.Pool(num_workers) as pool:
            for i, (record, reason) in enumerate(
                tqdm(
                    pool.imap(worker_fn, rows, chunksize=CLEAN_POOL_CHUNKSIZE),
                    total=total_input,
                    desc=f"clean molecules ({num_workers} workers)",
                )
            ):
                counts["input_rows"] = i + 1
                _accept_record(record, reason)

    counts["selected"] = len(selected)
    return selected, counts


def write_cleaned_records(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0].keys()) if records else ["chembl_id", "canonical_smiles"]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    tmp.replace(path)


def load_cleaned_records(path: Path) -> list[dict]:
    with gzip.open(path, "rt", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_cleaned_records_meta(path: Path, args: argparse.Namespace) -> None:
    """Write a ``.meta.json`` sidecar alongside the cleaned records CSV."""
    meta = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "chembl_version": args.chembl_version,
        "filters": {
            "mol_wt": [args.min_mol_wt, args.max_mol_wt],
            "heavy_atoms": [args.min_heavy_atoms, args.max_heavy_atoms],
            "allow_charged": args.allow_charged,
        },
        "standardizer": "chembl_structure_pipeline (pip)",
        "smoke_test": args.smoke_test,
        "smoke_limit": args.smoke_limit,
        "sample_seed": args.sample_seed,
    }
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def validate_cleaned_records_meta(path: Path, args: argparse.Namespace) -> bool:
    """Validate that the sidecar meta matches current *args*.

    Returns ``True`` if meta is missing (skip validation) or matches.
    Returns ``False`` and logs an error on mismatch.
    """
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    if not meta_path.exists():
        LOGGER.warning("No sidecar meta at %s — reusing without validation", meta_path)
        return True
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    checks = [
        ("chembl_version", meta.get("chembl_version"), args.chembl_version),
        ("filters.mol_wt.low", meta.get("filters", {}).get("mol_wt", [])[:1] or [None], args.min_mol_wt),
        ("filters.mol_wt.high", meta.get("filters", {}).get("mol_wt", [None])[1:2] or [None], args.max_mol_wt),
        ("filters.heavy_atoms.low", meta.get("filters", {}).get("heavy_atoms", [])[:1] or [None], args.min_heavy_atoms),
        ("filters.heavy_atoms.high", meta.get("filters", {}).get("heavy_atoms", [None])[1:2] or [None], args.max_heavy_atoms),
        ("allow_charged", meta.get("filters", {}).get("allow_charged"), args.allow_charged),
        ("smoke_test", meta.get("smoke_test"), args.smoke_test),
        ("sample_seed", meta.get("sample_seed"), args.sample_seed),
    ]
    errors = []
    for name, expected_val, actual_val in checks:
        if str(expected_val) != str(actual_val):
            errors.append(f"  {name}: meta={expected_val!r}, args={actual_val!r}")
    if errors:
        LOGGER.error(
            "Sidecar meta mismatch for %s:\n%s\n"
            "Delete the cleaned CSV + .meta.json, or use a different --output-tag.",
            meta_path, "\n".join(errors),
        )
        return False
    return True


def _compute_multi_fp(
    record: dict, radius: int, n_bits: int, fp_types: list[str]
) -> tuple[dict, dict[str, object] | None]:
    """Worker function: compute all requested fingerprint types for one record."""
    mol = Chem.MolFromSmiles(record["canonical_smiles"])
    if mol is None:
        return record, None
    fps: dict[str, object] = {}
    for ft in fp_types:
        if ft == "ecfp4":
            fps[ft] = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
        elif ft == "fcfp4":
            fps[ft] = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius=radius, nBits=n_bits, useFeatures=True
            )
    return record, fps


def build_fingerprint_caches(
    records: list[dict], args: argparse.Namespace, counts: dict[str, int]
) -> dict[str, tuple[list, list[dict]]]:
    """Build all requested fingerprint caches.

    Uses ``multiprocessing.Pool.imap`` (order-preserving) when ``--num-workers > 1``.
    """
    num_workers = args.num_workers
    if num_workers == 0:
        num_workers = mp.cpu_count()
    num_workers = max(1, min(num_workers, len(records)))

    results: dict[str, tuple[list, list[dict]]] = {ft: ([], []) for ft in args.fingerprint_types}
    parse_failed_key = "fingerprint_parse_failed"
    worker_fn = partial(_compute_multi_fp, radius=args.radius, n_bits=args.n_bits, fp_types=args.fingerprint_types)

    if num_workers <= 1:
        for record in tqdm(records, desc="fingerprints"):
            _, fps = worker_fn(record)
            if fps is None:
                counts[parse_failed_key] = counts.get(parse_failed_key, 0) + 1
                continue
            for ft in args.fingerprint_types:
                results[ft][0].append(fps[ft])
                results[ft][1].append(record)
    else:
        with mp.Pool(num_workers) as pool:
            for record, fps in tqdm(
                pool.imap(worker_fn, records, chunksize=FINGERPRINT_POOL_CHUNKSIZE),
                total=len(records),
                desc=f"fingerprints ({num_workers} workers)",
            ):
                if fps is None:
                    counts[parse_failed_key] = counts.get(parse_failed_key, 0) + 1
                    continue
                for ft in args.fingerprint_types:
                    results[ft][0].append(fps[ft])
                    results[ft][1].append(record)

    counts["fingerprint_total"] = len(results[args.fingerprint_types[0]][0])
    return results


def write_outputs(records: list[dict], paths: dict[str, Path], args: argparse.Namespace, counts: dict[str, int]) -> None:
    out_dir = paths["archive"].parent
    fingerprint_paths: dict[str, str] = {}
    metadata = cleaned_records_path(out_dir, args.chembl_version, args)
    summary_file = summary_path(out_dir, args.chembl_version, args)

    # Single pass: build all requested fingerprint types at once
    all_caches = build_fingerprint_caches(records, args, counts)
    shared_kept: list[dict] | None = None

    for fingerprint_type in args.fingerprint_types:
        fps, kept = all_caches[fingerprint_type]
        fp_path = fingerprint_cache_path(out_dir, args.chembl_version, fingerprint_type, args)
        payload = {
            "fingerprints": fps,
            "records": kept,
            "radius": args.radius,
            "n_bits": args.n_bits,
            "fingerprint_type": fingerprint_type,
            "chembl_version": args.chembl_version,
        }
        with gzip.open(fp_path, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        fingerprint_paths[fingerprint_type] = str(fp_path)
        if shared_kept is None or len(kept) > len(shared_kept):
            shared_kept = kept

    kept_for_metadata = shared_kept or []
    if len(kept_for_metadata) != len(records):
        write_cleaned_records(kept_for_metadata, metadata)
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "chembl_version": args.chembl_version,
        "fingerprints": fingerprint_paths,
        "cleaned_records_path": str(metadata),
        "metadata_path": str(metadata),
        "radius": args.radius,
        "n_bits": args.n_bits,
        "fingerprint_types": list(args.fingerprint_types),
        "output_tag": args.output_tag,
        "counts": counts,
        "filters": {
            "mol_wt": [args.min_mol_wt, args.max_mol_wt],
            "heavy_atoms": [args.min_heavy_atoms, args.max_heavy_atoms],
            "allow_charged": args.allow_charged,
        },
    }
    summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = paths_for(out_dir, args.chembl_version)
    release = f"chembl_{args.chembl_version}"
    url = f"{args.base_url}/{release}/{release}_sqlite.tar.gz"
    if paths["sqlite"].exists() and not args.force_download and not args.force_extract:
        LOGGER.info("Using existing SQLite DB and skipping download: %s", paths["sqlite"])
    else:
        try:
            if args.archive_path:
                copy_local_archive(args.archive_path, paths["archive"])
            else:
                download_file(url, paths["archive"], args.force_download)
        except Exception as exc:
            log_manual_download_help(url, paths["archive"], paths["sqlite"], exc)
            return 2
        extract_archive(paths["archive"], paths["extract_dir"], paths["sqlite"], args.force_extract)
    if args.download_only:
        return 0
    cleaned_path = cleaned_records_path(out_dir, args.chembl_version, args)
    if args.reuse_cleaned_records and cleaned_path.exists():
        if not validate_cleaned_records_meta(cleaned_path, args):
            LOGGER.error("Refusing to reuse cleaned records due to meta mismatch. "
                         "Delete %s(.meta.json) or adjust filter args.", cleaned_path)
            return 1
        LOGGER.info("Reusing cleaned molecule records: %s", cleaned_path)
        records = load_cleaned_records(cleaned_path)
        counts = {
            "input_rows": 0,
            "selected": len(records),
            "reused_cleaned_records": 1,
        }
    else:
        records, counts = build_records(paths["sqlite"], args)
        LOGGER.info("Writing cleaned molecule records: %s", cleaned_path)
        write_cleaned_records(records, cleaned_path)
        write_cleaned_records_meta(cleaned_path, args)
    write_outputs(records, paths, args, counts)
    LOGGER.info("Prepared %d ChEMBL fingerprints", counts.get("selected", 0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
