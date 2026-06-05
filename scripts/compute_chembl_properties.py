#!/usr/bin/env python3
"""Compute per-molecule 2D descriptors for cleaned ChEMBL molecules and aggregate statistics.

Reads ``chembl_36_molecules.csv.gz``, computes ~30 RDKit 2D descriptors per
molecule, and writes:

1. ``chembl_36_properties.csv.gz`` — per-molecule descriptor values.
2. ``chembl_36_properties_stats.json`` — aggregate statistics (mean, median, std,
   percentiles) for each numeric column.

Uses ``multiprocessing.Pool`` for parallel computation.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")
LOGGER = logging.getLogger("compute_chembl_properties")

# ---------------------------------------------------------------------------
# Helper: wrap any RDKit descriptor as (mol) → float
# ---------------------------------------------------------------------------
def _desc_float(func: Callable) -> Callable[[Chem.Mol], float]:
    """Wrap a descriptor function to return float."""
    return lambda m: float(func(m))

# ---------------------------------------------------------------------------
# Descriptor function table
# ---------------------------------------------------------------------------
# Each entry maps column_name → (callable, round_digits | None)
# round_digits controls output precision; None = store full float.

DESCRIPTOR_FUNCTIONS: dict[str, tuple[Callable, int | None]] = {
    # ── Basic (from input CSV, recomputed for consistency) ──────────────
    "heavy_atoms":        (lambda m: m.GetNumHeavyAtoms(), None),
    "formal_charge":      (Chem.GetFormalCharge, None),
    "num_rings":          (rdMolDescriptors.CalcNumRings, None),
    "mol_wt":             (Descriptors.MolWt, 3),
    # ── Lipophilicity & Polarity ────────────────────────────────────────
    "mol_log_p":          (Descriptors.MolLogP, 4),
    "tpsa":               (Descriptors.TPSA, 3),
    # ── H-bonding ───────────────────────────────────────────────────────
    "num_h_acceptors":    (Descriptors.NumHAcceptors, None),
    "num_h_donors":       (Descriptors.NumHDonors, None),
    # ── Flexibility ─────────────────────────────────────────────────────
    "num_rotatable_bonds": (rdMolDescriptors.CalcNumRotatableBonds, None),
    # ── Ring systems ────────────────────────────────────────────────────
    "num_aromatic_rings":            (rdMolDescriptors.CalcNumAromaticRings, None),
    "num_aliphatic_rings":           (rdMolDescriptors.CalcNumAliphaticRings, None),
    "num_saturated_rings":           (rdMolDescriptors.CalcNumSaturatedRings, None),
    "num_aromatic_heterocycles":     (rdMolDescriptors.CalcNumAromaticHeterocycles, None),
    "num_saturated_heterocycles":    (rdMolDescriptors.CalcNumSaturatedHeterocycles, None),
    # ── Carbon saturation ───────────────────────────────────────────────
    "fraction_csp3":      (rdMolDescriptors.CalcFractionCSP3, 4),
    # ── Structural complexity ───────────────────────────────────────────
    "num_spiro_atoms":    (rdMolDescriptors.CalcNumSpiroAtoms, None),
    "num_bridgehead_atoms": (rdMolDescriptors.CalcNumBridgeheadAtoms, None),
    # ── Molecular weight variants ───────────────────────────────────────
    "exact_mol_wt":       (Descriptors.ExactMolWt, 3),
    # ── Electron info ───────────────────────────────────────────────────
    "num_valence_electrons":  (Descriptors.NumValenceElectrons, None),
    "num_radical_electrons":  (Descriptors.NumRadicalElectrons, None),
    # ── NumHeteroatoms (from rdkit.Chem.Descriptors) ────────────────────
    "num_heteroatoms":    (Descriptors.NumHeteroatoms, None),
    # ── Other common 2D topological descriptors ─────────────────────────
    "chi0v":              (_desc_float(Descriptors.Chi0v), 4),
    "chi1v":              (_desc_float(Descriptors.Chi1v), 4),
    "chi0n":              (_desc_float(Descriptors.Chi0n), 4),
    "chi1n":              (_desc_float(Descriptors.Chi1n), 4),
    "kappa1":             (_desc_float(Descriptors.Kappa1), 3),
    "kappa2":             (_desc_float(Descriptors.Kappa2), 3),
    "kappa3":             (_desc_float(Descriptors.Kappa3), 3),
    "hall_kier_alpha":    (_desc_float(Descriptors.HallKierAlpha), 4),
}

# ---------------------------------------------------------------------------
# Custom descriptor helpers
# ---------------------------------------------------------------------------

def _count_carbon(mol: Chem.Mol) -> int:
    """Count carbon atoms."""
    return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6)

def _carbon_fraction(mol: Chem.Mol) -> float:
    """Fraction of heavy atoms that are carbon."""
    heavy = mol.GetNumHeavyAtoms()
    if heavy == 0:
        return 0.0
    n_c = _count_carbon(mol)
    return n_c / heavy

def _num_lipinski_violations(mol: Chem.Mol) -> int:
    """Count Lipinski Rule-of-Five violations (MW>500, LogP>5, HBA>10, HBD>5)."""
    violations = 0
    if float(Descriptors.MolWt(mol)) > 500:
        violations += 1
    if float(Descriptors.MolLogP(mol)) > 5:
        violations += 1
    if int(Descriptors.NumHAcceptors(mol)) > 10:
        violations += 1
    if int(Descriptors.NumHDonors(mol)) > 5:
        violations += 1
    return violations

def _num_lipinski_heavy_violations(mol: Chem.Mol) -> int:
    """Count Lipinski Rule-of-Five heavy violations (MW>600, LogP>7, HBA>15, HBD>10)."""
    violations = 0
    if float(Descriptors.MolWt(mol)) > 600:
        violations += 1
    if float(Descriptors.MolLogP(mol)) > 7:
        violations += 1
    if int(Descriptors.NumHAcceptors(mol)) > 15:
        violations += 1
    if int(Descriptors.NumHDonors(mol)) > 10:
        violations += 1
    return violations

# Custom descriptor table
CUSTOM_DESCRIPTOR_FUNCTIONS: dict[str, tuple[Callable, int | None]] = {
    "num_carbon_atoms":     (_count_carbon, None),
    "carbon_fraction":      (_carbon_fraction, 4),
    "lipinski_violations":  (_num_lipinski_violations, None),
    "lipinski_heavy_violations": (_num_lipinski_heavy_violations, None),
}

# ---------------------------------------------------------------------------
# Column/fieldnames
# ---------------------------------------------------------------------------
BASE_FIELDS = ["chembl_id", "canonical_smiles"]
ALL_DESCRIPTOR_NAMES = list(DESCRIPTOR_FUNCTIONS.keys()) + list(CUSTOM_DESCRIPTOR_FUNCTIONS.keys())
ALL_FIELDS = BASE_FIELDS + ALL_DESCRIPTOR_NAMES


def _compute_properties(
    row: dict[str, str],
) -> dict[str, Any]:
    """Compute all descriptors for one molecule row."""
    smiles = row["canonical_smiles"]
    mol = Chem.MolFromSmiles(smiles)
    result: dict[str, Any] = {
        "chembl_id": row["chembl_id"],
        "canonical_smiles": smiles,
    }
    if mol is None:
        # Invalid molecule — fill with NaN/empty
        for name in ALL_DESCRIPTOR_NAMES:
            result[name] = ""
        return result

    for name, (func, precision) in DESCRIPTOR_FUNCTIONS.items():
        try:
            val = func(mol)
            if precision is not None:
                val = round(float(val), precision)
            result[name] = val
        except Exception:
            result[name] = ""
    for name, (func, precision) in CUSTOM_DESCRIPTOR_FUNCTIONS.items():
        try:
            val = func(mol)
            if precision is not None:
                val = round(float(val), precision)
            result[name] = val
        except Exception:
            result[name] = ""
    return result


def compute_properties(
    records: list[dict[str, str]],
    num_workers: int,
    chunksize: int = 500,
) -> list[dict[str, Any]]:
    """Compute properties for all *records* using multiprocessing."""
    if num_workers <= 1:
        return [_compute_properties(r) for r in tqdm(records, desc="properties")]
    with mp.Pool(num_workers) as pool:
        results = list(
            tqdm(
                pool.imap(_compute_properties, records, chunksize=chunksize),
                total=len(records),
                desc=f"properties ({num_workers} workers)",
            )
        )
    return results


# ---------------------------------------------------------------------------
# Statistics computation
# ---------------------------------------------------------------------------

def _to_float_seq(val) -> list[float]:
    """Convert a column of values to float list, dropping empties."""
    out: list[float] = []
    for v in val:
        try:
            out.append(float(v))
        except (ValueError, TypeError):
            continue
    return out


def compute_statistics(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Compute aggregate statistics for each numeric descriptor column."""
    stats: dict[str, dict[str, float]] = {}
    for name in ALL_DESCRIPTOR_NAMES:
        vals = _to_float_seq(r.get(name) for r in rows)
        if not vals:
            stats[name] = {
                "count": 0, "min": 0, "max": 0, "mean": 0, "median": 0,
                "std": 0, "p5": 0, "p25": 0, "p75": 0, "p95": 0,
            }
            continue
        arr = np.array(vals, dtype=np.float64)
        stats[name] = {
            "count": len(arr),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr, ddof=0)),
            "p5": float(np.percentile(arr, 5)),
            "p25": float(np.percentile(arr, 25)),
            "p75": float(np.percentile(arr, 75)),
            "p95": float(np.percentile(arr, 95)),
        }
    return stats


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_records(path: Path) -> list[dict[str, str]]:
    """Load cleaned molecule CSV into list of dicts."""
    LOGGER.info("Loading cleaned molecules: %s", path)
    with gzip.open(path, "rt", newline="") as f:
        reader = csv.DictReader(f)
        records = list(reader)
    LOGGER.info("Loaded %d molecules", len(records))
    return records


def write_properties_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write per-molecule properties to gzipped CSV."""
    LOGGER.info("Writing properties: %s", path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ALL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)
    LOGGER.info("Wrote %d rows to %s", len(rows), path)


def write_stats_json(stats: dict, path: Path) -> None:
    """Write aggregate statistics to JSON."""
    LOGGER.info("Writing statistics: %s", path)
    # Use the first descriptor's count as the authoritative molecule count
    first_key = next(iter(stats), "")
    mol_count = int(stats[first_key]["count"]) if first_key else 0
    meta = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": str(Path("data/chembl/chembl_36_molecules.csv.gz")),
        "num_molecules": mol_count,
        "num_descriptors": len(stats),
        "descriptor_names": ALL_DESCRIPTOR_NAMES,
        "statistics": stats,
    }
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    LOGGER.info("Wrote %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/chembl/chembl_36_molecules.csv.gz",
                        help="Input cleaned molecule CSV.gz")
    parser.add_argument("--output-csv", default="data/chembl/chembl_36_properties.csv.gz",
                        help="Output per-molecule properties CSV.gz")
    parser.add_argument("--output-stats", default="data/chembl/chembl_36_properties_stats.json",
                        help="Output aggregate statistics JSON")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Number of worker processes. 0 = all CPUs (18 on this machine).")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(levelname)s %(message)s")

    num_workers = args.num_workers
    if num_workers == 0:
        num_workers = mp.cpu_count()
    LOGGER.info("Using %d workers", num_workers)

    input_path = Path(args.input).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()
    output_stats = Path(args.output_stats).expanduser().resolve()

    # Step 1: Load records
    records = load_records(input_path)

    # Step 2: Compute per-molecule properties
    t0 = time.perf_counter()
    results = compute_properties(records, num_workers)
    elapsed = time.perf_counter() - t0
    LOGGER.info("Computed properties for %d molecules in %.1f s (%.1f mol/s)",
                len(results), elapsed, len(results) / max(elapsed, 0.01))

    # Step 3: Write per-molecule CSV
    write_properties_csv(results, output_csv)

    # Step 4: Compute and write statistics
    stats = compute_statistics(results)
    write_stats_json(stats, output_stats)

    # Print summary
    LOGGER.info("Done. Key statistics:")
    for name in ["mol_wt", "mol_log_p", "tpsa", "num_h_acceptors", "num_h_donors",
                 "fraction_csp3", "num_aromatic_rings", "num_rotatable_bonds",
                 "carbon_fraction", "lipinski_violations"]:
        s = stats.get(name)
        if s:
            LOGGER.info("  %s: mean=%.3f, median=%.3f, std=%.3f [%.3f, %.3f]",
                        name, s["mean"], s["median"], s["std"], s["min"], s["max"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
