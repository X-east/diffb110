#!/usr/bin/env python
"""Recompute R1-R5 structural-pathology rates on the new ChEMBL small-molecule
reference set (built 2026-09-28) and compare it against the previous reference.

Motivation
----------
The thesis reported "ChEMBL structural compliance = 84.72%" on the *old*
processed reference (``data/chembl/chembl_36_molecules.csv.gz``, n = 1,550,090,
mol_wt 150-500 / heavy_atoms 10-30 / no charge). On 2026-09-28 a new
small-molecule-only reference was built (``data/chembl_smallmol_reference/
reference_smallmol.csv.gz``, n = 1,790,085, no size/MW window). Its R1-R5 rates
have not been recomputed, so every number derived from 84.72% is stale.

This script reuses the exact same rule module
(``scripts/offline_rl/structure_pathology_rules.py``) and the same
largest-fragment convention as ``analysis/chembl_deviation_analysis/
compute_chembl_r1r5_full_reference.py`` so the new reference row is directly
comparable to the old one and to the generated-model tables.

It is parameterised (input / output dir / label / workers) instead of hardcoding
paths, so it can be re-run on either the old or the new reference.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "offline_rl"))

from structure_pathology_rules import (  # noqa: E402
    RULE_FIELDS,
    STRUCTURAL_RULE_FIELDS,
    compute_structure_pathology,
)

LOGGER = logging.getLogger("Reference-R1R5")

DEFAULT_INPUT = "data/chembl_smallmol_reference/reference_smallmol.csv.gz"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent
DEFAULT_WORKERS = 18

# Chinese display names, kept identical to the previous reference script so that
# thesis tables built from either source stay consistent.
DISPLAY = {
    "r1_large_ring_violation": "R1 大环 >=7",
    "r2_ring_system_violation": "R2 环系规模 >3",
    "r3_four_ring_violation": "R3 四元环",
    "r4a_three_ring_hetero_violation": "R4a 三元环含杂原子",
    "r4b_three_ring_over_attached_violation": "R4b 三元环过度连接",
    "r4c_three_ring_spiro_or_fused_violation": "R4c 三元环螺环/稠合",
    "r5_carbon_fraction_violation": "R5 碳比例 >0.85",
}
AGGREGATE_DISPLAY = {
    "r4_any_three_ring_violation": "R4 任一子规则",
    "is_structurally_pathological": "R1-R4 任一",
    "is_pathological": "R1-R5 任一",
    "clean_r1_r4": "R1-R4 全清",
    "clean_r1_r5": "R1-R5 全清",
}


def _largest_fragment(mol: Chem.Mol | None) -> tuple[Chem.Mol | None, int]:
    """Return (largest connected fragment, original fragment count).

    Mirrors ``scripts/offline_rl/label_trajectory_shards_batched.py`` and the old
    ChEMBL reference script so the reference and generated sides share one
    convention. The new reference is already salt-stripped, so this is expected
    to be a no-op; the count is reported as evidence.
    """
    if mol is None:
        return None, 0
    try:
        fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    except Exception:
        return None, 0
    if not fragments:
        return None, 0
    largest = max(fragments, key=lambda m: m.GetNumAtoms())
    return largest, len(fragments)


def _compute_one(smiles: str) -> dict | None:
    """Evaluate one SMILES under the processed (largest-fragment) convention."""
    if not smiles:
        return None
    raw = Chem.MolFromSmiles(smiles)
    if raw is None:
        return None
    mol, n_frag = _largest_fragment(raw)
    if mol is None:
        return None
    row = compute_structure_pathology(mol).as_row()
    out = {field: int(row[field]) for field in RULE_FIELDS}
    out["r4_any_three_ring_violation"] = int(row["r4_any_three_ring_violation"])
    out["is_pathological"] = int(row["is_pathological"])
    out["is_structurally_pathological"] = int(row["is_structurally_pathological"])
    out["violation_count"] = int(row["violation_count"])
    out["structural_violation_count"] = int(row["structural_violation_count"])
    out["n_fragments"] = int(n_frag)
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT,
                        help="Reference CSV(.gz) containing a canonical_smiles column")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--label", default="new_smallmol_reference",
                        help="Prefix for output files and the JSON 'label' field")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    return parser.parse_args()


def main() -> None:
    """Run the R1-R5 recomputation and persist JSON + CSV evidence."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{args.label}_r1r5.json"
    out_csv = out_dir / f"{args.label}_r1r5.csv"

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = REPO_ROOT / input_path
    LOGGER.info("reading reference: %s", input_path)
    frame = pd.read_csv(input_path, usecols=["canonical_smiles"])
    smiles_list = frame["canonical_smiles"].astype(str).tolist()
    n_total = len(smiles_list)
    LOGGER.info("total molecules: %d", n_total)

    n_dot = sum(1 for s in smiles_list if "." in s)

    LOGGER.info("evaluating R1-R5 under largest-fragment convention (%d workers)",
                args.workers)
    with Pool(processes=args.workers) as pool:
        results = pool.map(_compute_one, smiles_list, chunksize=2048)

    n_parse_fail = sum(1 for r in results if r is None)
    valid = [r for r in results if r is not None]
    n_scored = len(valid)
    n_frag_reduced = sum(1 for r in valid if r["n_fragments"] > 1)

    hits = Counter()
    for r in valid:
        for field in RULE_FIELDS:
            if r[field]:
                hits[field] += 1

    r4_any = sum(r["r4_any_three_ring_violation"] for r in valid)
    r1_r4_any = sum(r["is_structurally_pathological"] for r in valid)
    r1_r5_any = sum(r["is_pathological"] for r in valid)

    def _rate(n: int) -> float:
        return n / n_scored if n_scored else 0.0

    vc_dist = Counter(r["violation_count"] for r in valid)
    svc_dist = Counter(r["structural_violation_count"] for r in valid)

    cooc: dict[str, dict[str, int]] = {}
    for a in RULE_FIELDS:
        cooc[a] = {b: sum(1 for r in valid if r[a] and r[b]) for b in RULE_FIELDS}

    summary = {
        "label": args.label,
        "source": str(input_path),
        "convention": "largest-fragment reduction (same as generated side); "
                      "structure_pathology_rules R1-R5",
        "total_molecules": n_total,
        "smiles_with_dot": n_dot,
        "parse_failures": n_parse_fail,
        "scored_molecules": n_scored,
        "fragment_reduced_molecules": n_frag_reduced,
        "rule_hits": {f: hits[f] for f in RULE_FIELDS},
        "rule_rates": {f: _rate(hits[f]) for f in RULE_FIELDS},
        "aggregates": {
            "r4_any_three_ring": {"count": r4_any, "rate": _rate(r4_any)},
            "r1_r4_any": {"count": r1_r4_any, "rate": _rate(r1_r4_any)},
            "r1_r5_any": {"count": r1_r5_any, "rate": _rate(r1_r5_any)},
            "r1_r4_clean": {"count": n_scored - r1_r4_any,
                            "rate": _rate(n_scored - r1_r4_any)},
            "r1_r5_clean": {"count": n_scored - r1_r5_any,
                            "rate": _rate(n_scored - r1_r5_any)},
        },
        "violation_count_distribution": {
            str(k): {"count": vc_dist.get(k, 0), "rate": _rate(vc_dist.get(k, 0))}
            for k in range(len(RULE_FIELDS) + 1)
        },
        "structural_violation_count_distribution": {
            str(k): {"count": svc_dist.get(k, 0), "rate": _rate(svc_dist.get(k, 0))}
            for k in range(len(STRUCTURAL_RULE_FIELDS) + 1)
        },
        "cooccurrence_counts": cooc,
    }
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")

    rows = []
    for f in RULE_FIELDS:
        rows.append({"metric": f, "display": DISPLAY[f], "kind": "per_rule",
                     "count": hits[f], "rate": _rate(hits[f])})
    rows.append({"metric": "r4_any_three_ring_violation",
                 "display": AGGREGATE_DISPLAY["r4_any_three_ring_violation"],
                 "kind": "aggregate", "count": r4_any, "rate": _rate(r4_any)})
    rows.append({"metric": "is_structurally_pathological",
                 "display": AGGREGATE_DISPLAY["is_structurally_pathological"],
                 "kind": "aggregate", "count": r1_r4_any, "rate": _rate(r1_r4_any)})
    rows.append({"metric": "is_pathological",
                 "display": AGGREGATE_DISPLAY["is_pathological"],
                 "kind": "aggregate", "count": r1_r5_any, "rate": _rate(r1_r5_any)})
    rows.append({"metric": "clean_r1_r4",
                 "display": AGGREGATE_DISPLAY["clean_r1_r4"], "kind": "aggregate",
                 "count": n_scored - r1_r4_any, "rate": _rate(n_scored - r1_r4_any)})
    rows.append({"metric": "clean_r1_r5",
                 "display": AGGREGATE_DISPLAY["clean_r1_r5"], "kind": "aggregate",
                 "count": n_scored - r1_r5_any, "rate": _rate(n_scored - r1_r5_any)})
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    LOGGER.info("=" * 72)
    LOGGER.info("reference '%s'  n_scored = %d", args.label, n_scored)
    LOGGER.info("  total=%d  parse_fail=%d  smiles_with_dot=%d  fragment_reduced=%d",
                n_total, n_parse_fail, n_dot, n_frag_reduced)
    LOGGER.info("-" * 72)
    for f in RULE_FIELDS:
        LOGGER.info("  %-46s %9d  %7.4f%%", DISPLAY[f], hits[f], _rate(hits[f]) * 100)
    LOGGER.info("  %-46s %9d  %7.4f%%", "R4 任一子规则", r4_any, _rate(r4_any) * 100)
    LOGGER.info("  %-46s %9d  %7.4f%%", "R1-R4 任一", r1_r4_any, _rate(r1_r4_any) * 100)
    LOGGER.info("  %-46s %9d  %7.4f%%", "R1-R5 任一", r1_r5_any, _rate(r1_r5_any) * 100)
    LOGGER.info("  %-46s %9d  %7.4f%%", "R1-R4 全清", n_scored - r1_r4_any,
                _rate(n_scored - r1_r4_any) * 100)
    LOGGER.info("  %-46s %9d  %7.4f%%", "R1-R5 全清", n_scored - r1_r5_any,
                _rate(n_scored - r1_r5_any) * 100)
    LOGGER.info("=" * 72)
    LOGGER.info("wrote %s", out_json)
    LOGGER.info("wrote %s", out_csv)


if __name__ == "__main__":
    main()
