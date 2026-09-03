#!/usr/bin/env python
"""Build an R1-R5 structural pathology sidecar for offline RL endpoints.

Reads the episode index, evaluates the user-confirmed structural pathology
rules (R1-R5) for every ``rdkit_valid`` endpoint, and writes:

* ``structure_pathology_sidecar.csv`` -- per-molecule flags and counts
* ``structure_pathology_summary.json`` -- per-rule hit counts and rates

Rules are implemented in ``structure_pathology_rules.py``; this script only
handles IO, parallelism and summarisation.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

from structure_pathology_rules import (
    PATHOLOGY_COLUMNS,
    RULE_FIELDS,
    compute_structure_pathology_from_smiles,
)

LOGGER = logging.getLogger("DiffSBDD-Pathology")

DEFAULT_INDEX = "analysis/offline_iql_full_index_rebuilt_260824_v7/episode_index.jsonl"
DEFAULT_OUTPUT_DIR = "analysis/diffsbdd_structure_pathology_260831"
DEFAULT_WORKERS = 18


def load_valid_smiles(index_path: Path) -> list[tuple[str, str]]:
    """Return ``(sample_id, smiles)`` for every rdkit-valid endpoint."""
    records: list[tuple[str, str]] = []
    with index_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not row.get("rdkit_valid"):
                continue
            smiles = row.get("smiles")
            if not smiles:
                continue
            records.append((str(row.get("sample_id")), str(smiles)))
    return records


def _compute_one(item: tuple[str, str]) -> dict[str, object]:
    """Evaluate one endpoint; unparseable SMILES count as pathological."""
    sample_id, smiles = item
    facts = compute_structure_pathology_from_smiles(smiles)
    if facts is None:
        row: dict[str, object] = {"sample_id": sample_id, "smiles": smiles,
                                  "parse_ok": 0}
        for column in PATHOLOGY_COLUMNS:
            row[column] = 0
        row["is_pathological"] = 1
        row["violation_count"] = 1
        row["is_structurally_pathological"] = 1
        row["structural_violation_count"] = 1
        return row
    row = {"sample_id": sample_id, "smiles": smiles, "parse_ok": 1}
    row.update(facts.as_row())
    return row


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-path", default=DEFAULT_INDEX)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--limit", type=int, default=0,
                        help="Only process the first N endpoints (smoke test).")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    index_path = Path(args.index_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("loading episode index: %s", index_path)
    records = load_valid_smiles(index_path)
    LOGGER.info("rdkit-valid endpoints with SMILES: %d", len(records))
    if args.limit > 0:
        records = records[: args.limit]
        LOGGER.info("smoke limit applied: %d", len(records))

    LOGGER.info("evaluating R1-R5 with %d workers", args.workers)
    with Pool(processes=args.workers) as pool:
        rows = pool.map(_compute_one, records, chunksize=512)
    LOGGER.info("evaluation finished: %d rows", len(rows))

    csv_path = output_dir / "structure_pathology_sidecar.csv"
    fieldnames = ["sample_id", "smiles", "parse_ok", *PATHOLOGY_COLUMNS]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    LOGGER.info("wrote sidecar: %s", csv_path)

    total = len(rows)
    hits = Counter()
    for row in rows:
        for field in RULE_FIELDS:
            if row.get(field):
                hits[field] += 1
    pathological = sum(1 for row in rows if row.get("is_pathological"))
    structural_pathological = sum(
        1 for row in rows if row.get("is_structurally_pathological")
    )
    parse_fail = sum(1 for row in rows if not row.get("parse_ok"))

    summary = {
        "index_path": str(index_path),
        "total_valid_endpoints": total,
        "parse_failures": parse_fail,
        "rule_hits": {field: hits[field] for field in RULE_FIELDS},
        "rule_rates": {
            field: (hits[field] / total if total else 0.0) for field in RULE_FIELDS
        },
        "pathological_count": pathological,
        "pathological_rate": (pathological / total if total else 0.0),
        "clean_count": total - pathological,
        "clean_rate": ((total - pathological) / total if total else 0.0),
        "structural_pathological_count": structural_pathological,
        "structural_pathological_rate": (
            structural_pathological / total if total else 0.0
        ),
        "structural_clean_count": total - structural_pathological,
        "structural_clean_rate": (
            (total - structural_pathological) / total if total else 0.0
        ),
    }
    summary_path = output_dir / "structure_pathology_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    LOGGER.info("wrote summary: %s", summary_path)

    LOGGER.info("=== R1-R5 hit rates (n=%d) ===", total)
    for field in RULE_FIELDS:
        LOGGER.info("  %-48s %7d  %6.3f%%", field, hits[field],
                    summary["rule_rates"][field] * 100)
    LOGGER.info("  %-48s %7d  %6.3f%%", "PATHOLOGICAL (any rule)", pathological,
                summary["pathological_rate"] * 100)
    LOGGER.info("  %-48s %7d  %6.3f%%", "CLEAN (no rule hit)", total - pathological,
                summary["clean_rate"] * 100)


if __name__ == "__main__":
    main()
