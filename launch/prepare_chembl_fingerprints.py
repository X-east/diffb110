#!/usr/bin/env python3
"""Download ChEMBL SQLite and build a cleaned Morgan fingerprint cache."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import pickle
import random
import shutil
import sqlite3
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Iterable

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")
LOGGER = logging.getLogger("prepare_chembl")
DEFAULT_VERSION = "35"
DEFAULT_BASE_URL = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chembl-version", default=DEFAULT_VERSION)
    parser.add_argument("--out-dir", default="data/chembl")
    parser.add_argument(
        "--archive-path",
        default=None,
        help="Optional local chembl_<version>_sqlite.tar.gz path. Useful when direct download is blocked.",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--max-molecules", type=int, default=100000)
    parser.add_argument("--sample-seed", type=int, default=20260515)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--n-bits", type=int, default=2048)
    parser.add_argument("--min-mol-wt", type=float, default=150.0)
    parser.add_argument("--max-mol-wt", type=float, default=700.0)
    parser.add_argument("--min-heavy-atoms", type=int, default=10)
    parser.add_argument("--max-heavy-atoms", type=int, default=80)
    parser.add_argument("--allow-charged", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-limit", type=int, default=1000)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
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


def sanitize_largest_fragment(smiles: str):
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if not fragments:
        return None
    mol = max(fragments, key=lambda item: item.GetNumHeavyAtoms())
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return mol


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


def build_records(sqlite_path: Path, args: argparse.Namespace) -> tuple[list[dict], dict[str, int]]:
    rng = random.Random(args.sample_seed)
    limit = args.smoke_limit if args.smoke_test else args.max_molecules
    selected: list[dict] = []
    seen: set[str] = set()
    counts: dict[str, int] = {"input_rows": 0, "parse_failed": 0, "duplicates": 0, "accepted_before_sampling": 0}

    for chembl_id, smiles in tqdm(iter_smiles(sqlite_path), desc="clean molecules"):
        counts["input_rows"] += 1
        mol = sanitize_largest_fragment(smiles)
        if mol is None:
            counts["parse_failed"] += 1
            continue
        reason = filter_reason(mol, args)
        if reason != "ok":
            counts[reason] = counts.get(reason, 0) + 1
            continue
        canonical = Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=False, canonical=True)
        if canonical in seen:
            counts["duplicates"] += 1
            continue
        seen.add(canonical)
        counts["accepted_before_sampling"] += 1
        record = {
            "chembl_id": chembl_id,
            "canonical_smiles": canonical,
            "mol_wt": round(float(Descriptors.MolWt(mol)), 4),
            "heavy_atoms": int(mol.GetNumHeavyAtoms()),
            "formal_charge": int(Chem.GetFormalCharge(mol)),
            "num_rings": int(rdMolDescriptors.CalcNumRings(mol)),
        }
        if limit <= 0 or len(selected) < limit:
            selected.append(record)
        else:
            idx = rng.randrange(counts["accepted_before_sampling"])
            if idx < limit:
                selected[idx] = record
        if args.smoke_test and counts["input_rows"] >= max(args.smoke_limit * 20, args.smoke_limit):
            break
    counts["selected"] = len(selected)
    return selected, counts


def write_outputs(records: list[dict], paths: dict[str, Path], args: argparse.Namespace, counts: dict[str, int]) -> None:
    fps = []
    kept = []
    for record in tqdm(records, desc="fingerprints"):
        mol = Chem.MolFromSmiles(record["canonical_smiles"])
        if mol is None:
            counts["fingerprint_parse_failed"] = counts.get("fingerprint_parse_failed", 0) + 1
            continue
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius=args.radius, nBits=args.n_bits))
        kept.append(record)

    with gzip.open(paths["fingerprints"], "wb") as handle:
        pickle.dump({"fingerprints": fps, "records": kept, "radius": args.radius, "n_bits": args.n_bits}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with gzip.open(paths["metadata"], "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(kept[0].keys()) if kept else ["chembl_id", "canonical_smiles"])
        writer.writeheader()
        writer.writerows(kept)
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "chembl_version": args.chembl_version,
        "fingerprints_path": str(paths["fingerprints"]),
        "metadata_path": str(paths["metadata"]),
        "radius": args.radius,
        "n_bits": args.n_bits,
        "counts": counts,
        "filters": {
            "mol_wt": [args.min_mol_wt, args.max_mol_wt],
            "heavy_atoms": [args.min_heavy_atoms, args.max_heavy_atoms],
            "allow_charged": args.allow_charged,
        },
    }
    paths["summary"].write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


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
    records, counts = build_records(paths["sqlite"], args)
    write_outputs(records, paths, args, counts)
    LOGGER.info("Prepared %d ChEMBL fingerprints", counts.get("selected", 0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
