#!/usr/bin/env python
"""Compute the complete R1-R5 reference statistics on the full processed ChEMBL-36 library.

Motivation
----------
``chembl_structure_pathology_summary.json`` reports per-rule rates and the R1-R5
aggregate, but NOT the R1-R4 aggregate that every generated-model table reports
(表 3.2 / 表 3.3). Without it the model-side R1-R4 column has no reference value.
This script recomputes every aggregate in one pass on the same processed ChEMBL
library so the thesis has a single consistent reference row.

Processed-ChEMBL convention
---------------------------
The source file ``data/chembl/chembl_36_molecules.csv.gz`` is the processed
(standardized + filtered) library, per its meta.json:
    standardizer : chembl_structure_pipeline
    filters      : mol_wt 150-500, heavy_atoms 10-30, allow_charged=false

Generated molecules are judged under the "relaxed largest-fragment" convention
(only the largest connected component enters R1-R5). This script applies the
IDENTICAL convention to ChEMBL and reports how many molecules the convention
actually altered, so both sides can be shown to share one pipeline.
"""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "offline_rl"))

from structure_pathology_rules import (  # noqa: E402
    RULE_FIELDS,
    STRUCTURAL_RULE_FIELDS,
    compute_structure_pathology,
)

LOGGER = logging.getLogger("ChEMBL-R1R5-Full")

MOLECULES_CSV = "data/chembl/chembl_36_molecules.csv.gz"
OUT_DIR = Path(__file__).resolve().parent
OUT_JSON = OUT_DIR / "chembl_r1r5_full_reference.json"
OUT_CSV = OUT_DIR / "chembl_r1r5_full_reference.csv"
WORKERS = 18


def _largest_fragment(mol: Chem.Mol | None) -> tuple[Chem.Mol | None, int]:
    """Return (largest connected fragment, original fragment count).

    Mirrors ``scripts/offline_rl/label_trajectory_shards_batched.py::_extract_largest_fragment``
    so the ChEMBL reference and the generated-molecule sidecar share one convention.
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
    facts = compute_structure_pathology(mol)
    row = facts.as_row()
    out = {field: int(row[field]) for field in RULE_FIELDS}
    out["r4_any_three_ring_violation"] = int(row["r4_any_three_ring_violation"])
    out["is_pathological"] = int(row["is_pathological"])
    out["is_structurally_pathological"] = int(row["is_structurally_pathological"])
    out["violation_count"] = int(row["violation_count"])
    out["structural_violation_count"] = int(row["structural_violation_count"])
    out["n_fragments"] = int(n_frag)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    LOGGER.info("reading processed ChEMBL: %s", MOLECULES_CSV)
    frame = pd.read_csv(MOLECULES_CSV, usecols=["canonical_smiles"])
    smiles_list = frame["canonical_smiles"].astype(str).tolist()
    n_total = len(smiles_list)
    LOGGER.info("total molecules: %d", n_total)

    n_dot = sum(1 for s in smiles_list if "." in s)

    LOGGER.info("evaluating R1-R5 under largest-fragment convention (%d workers)", WORKERS)
    with Pool(processes=WORKERS) as pool:
        results = pool.map(_compute_one, smiles_list, chunksize=2048)

    n_parse_fail = sum(1 for r in results if r is None)
    valid = [r for r in results if r is not None]
    n_scored = len(valid)
    n_frag_reduced = sum(1 for r in valid if r["n_fragments"] > 1)

    # ---- per-rule rates ---------------------------------------------------
    hits = Counter()
    for r in valid:
        for field in RULE_FIELDS:
            if r[field]:
                hits[field] += 1

    # ---- aggregates -------------------------------------------------------
    r4_any = sum(r["r4_any_three_ring_violation"] for r in valid)
    r1_r4_any = sum(r["is_structurally_pathological"] for r in valid)
    r1_r5_any = sum(r["is_pathological"] for r in valid)

    def _rate(n: int) -> float:
        return n / n_scored if n_scored else 0.0

    # ---- violation-count distributions ------------------------------------
    vc_dist = Counter(r["violation_count"] for r in valid)          # 0..7  (R1-R5)
    svc_dist = Counter(r["structural_violation_count"] for r in valid)  # 0..6  (R1-R4)

    # ---- pairwise co-occurrence (rules co-firing) -------------------------
    cooc: dict[str, dict[str, int]] = {}
    for a in RULE_FIELDS:
        cooc[a] = {}
        for b in RULE_FIELDS:
            cooc[a][b] = sum(1 for r in valid if r[a] and r[b])

    summary = {
        "source": MOLECULES_CSV,
        "convention": "processed ChEMBL (chembl_structure_pipeline standardised, "
                      "mol_wt 150-500, heavy_atoms 10-30, allow_charged=false) "
                      "+ largest-fragment reduction (same as generated side)",
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
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")

    # ---- flat CSV for direct use in thesis tables --------------------------
    display = {
        "r1_large_ring_violation": "R1 大环 >=7",
        "r2_ring_system_violation": "R2 环系规模 >3",
        "r3_four_ring_violation": "R3 四元环",
        "r4a_three_ring_hetero_violation": "R4a 三元环含杂原子",
        "r4b_three_ring_over_attached_violation": "R4b 三元环过度连接",
        "r4c_three_ring_spiro_or_fused_violation": "R4c 三元环螺环/稠合",
        "r5_carbon_fraction_violation": "R5 碳比例 >0.85",
    }
    rows = []
    for f in RULE_FIELDS:
        rows.append({"metric": f, "display": display[f], "kind": "per_rule",
                     "count": hits[f], "rate": _rate(hits[f])})
    rows.append({"metric": "r4_any_three_ring_violation", "display": "R4 任一子规则",
                 "kind": "aggregate", "count": r4_any, "rate": _rate(r4_any)})
    rows.append({"metric": "is_structurally_pathological", "display": "R1-R4 任一",
                 "kind": "aggregate", "count": r1_r4_any, "rate": _rate(r1_r4_any)})
    rows.append({"metric": "is_pathological", "display": "R1-R5 任一",
                 "kind": "aggregate", "count": r1_r5_any, "rate": _rate(r1_r5_any)})
    rows.append({"metric": "clean_r1_r4", "display": "R1-R4 全清",
                 "kind": "aggregate", "count": n_scored - r1_r4_any,
                 "rate": _rate(n_scored - r1_r4_any)})
    rows.append({"metric": "clean_r1_r5", "display": "R1-R5 全清",
                 "kind": "aggregate", "count": n_scored - r1_r5_any,
                 "rate": _rate(n_scored - r1_r5_any)})
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    # ---- console report ---------------------------------------------------
    LOGGER.info("=" * 72)
    LOGGER.info("ChEMBL-36 (processed, full)  n_scored = %d", n_scored)
    LOGGER.info("  total=%d  parse_fail=%d  smiles_with_dot=%d  fragment_reduced=%d",
                n_total, n_parse_fail, n_dot, n_frag_reduced)
    LOGGER.info("-" * 72)
    for f in RULE_FIELDS:
        LOGGER.info("  %-46s %9d  %7.4f%%", display[f], hits[f], _rate(hits[f]) * 100)
    LOGGER.info("  %-46s %9d  %7.4f%%", "R4 任一子规则", r4_any, _rate(r4_any) * 100)
    LOGGER.info("  %-46s %9d  %7.4f%%", "R1-R4 任一", r1_r4_any, _rate(r1_r4_any) * 100)
    LOGGER.info("  %-46s %9d  %7.4f%%", "R1-R5 任一", r1_r5_any, _rate(r1_r5_any) * 100)
    LOGGER.info("-" * 72)
    LOGGER.info("  违规条数分布 (R1-R5):")
    for k in range(len(RULE_FIELDS) + 1):
        c = vc_dist.get(k, 0)
        LOGGER.info("    %d 条  %9d  %7.4f%%", k, c, _rate(c) * 100)
    LOGGER.info("  违规条数分布 (R1-R4):")
    for k in range(len(STRUCTURAL_RULE_FIELDS) + 1):
        c = svc_dist.get(k, 0)
        LOGGER.info("    %d 条  %9d  %7.4f%%", k, c, _rate(c) * 100)
    LOGGER.info("=" * 72)
    LOGGER.info("wrote %s", OUT_JSON)
    LOGGER.info("wrote %s", OUT_CSV)


if __name__ == "__main__":
    main()
