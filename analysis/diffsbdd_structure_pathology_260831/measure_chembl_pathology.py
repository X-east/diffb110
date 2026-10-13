#!/usr/bin/env python
"""Measure R1-R5 structural pathology rates on the ChEMBL-36 reference library.

This is the control measurement for the DiffSBDD endpoint sidecar: real
drug-like molecules should violate these rules far less often than generated
molecules. If ChEMBL molecules violate a rule at a similar rate to DiffSBDD,
that rule is too strict to be a useful "unreasonable" signal.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from structure_pathology_rules import (
    RULE_FIELDS,
    compute_structure_pathology_from_smiles,
)

LOGGER = logging.getLogger("ChEMBL-Pathology")

MOLECULES_CSV = "data/chembl/chembl_36_molecules.csv.gz"
OUTPUT_JSON = Path(__file__).with_name("chembl_structure_pathology_summary.json")
WORKERS = 18


def _compute_one(smiles: str) -> dict[str, int] | None:
    """Evaluate one SMILES; return per-rule flags or ``None`` if unparseable."""
    facts = compute_structure_pathology_from_smiles(smiles)
    if facts is None:
        return None
    values = facts.as_row()
    return {field: int(values[field]) for field in RULE_FIELDS} | {
        "is_pathological": int(values["is_pathological"]),
        "violation_count": int(values["violation_count"]),
    }


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    LOGGER.info("reading ChEMBL molecules: %s", MOLECULES_CSV)
    frame = pd.read_csv(MOLECULES_CSV, usecols=["canonical_smiles"])
    smiles_list = frame["canonical_smiles"].astype(str).tolist()
    total = len(smiles_list)
    LOGGER.info("ChEMBL molecules loaded: %d", total)

    LOGGER.info("evaluating R1-R5 with %d workers", WORKERS)
    with Pool(processes=WORKERS) as pool:
        results = pool.map(_compute_one, smiles_list, chunksize=2048)

    parse_failures = sum(1 for item in results if item is None)
    valid = [item for item in results if item is not None]
    scored = len(valid)

    hits = Counter()
    for item in valid:
        for field in RULE_FIELDS:
            if item[field]:
                hits[field] += 1
    pathological = sum(1 for item in valid if item["is_pathological"])

    summary = {
        "source": MOLECULES_CSV,
        "total_molecules": total,
        "parse_failures": parse_failures,
        "scored_molecules": scored,
        "rule_hits": {field: hits[field] for field in RULE_FIELDS},
        "rule_rates": {field: (hits[field] / scored if scored else 0.0)
                       for field in RULE_FIELDS},
        "pathological_count": pathological,
        "pathological_rate": (pathological / scored if scored else 0.0),
        "clean_count": scored - pathological,
        "clean_rate": ((scored - pathological) / scored if scored else 0.0),
    }
    OUTPUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    LOGGER.info("wrote %s", OUTPUT_JSON)

    LOGGER.info("=== ChEMBL R1-R5 rates (n=%d) ===", scored)
    for field in RULE_FIELDS:
        LOGGER.info("  %-48s %8d  %6.3f%%", field, hits[field],
                    summary["rule_rates"][field] * 100)
    LOGGER.info("  %-48s %8d  %6.3f%%", "PATHOLOGICAL (any rule)", pathological,
                summary["pathological_rate"] * 100)
    LOGGER.info("  %-48s %8d  %6.3f%%", "CLEAN (no rule hit)",
                scored - pathological, summary["clean_rate"] * 100)


if __name__ == "__main__":
    main()
