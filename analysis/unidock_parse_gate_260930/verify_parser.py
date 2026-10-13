#!/usr/bin/env python
"""Validate the canonical Uni-Dock parser against every stored output we have.

Checks, over the real corpus:

1. **coverage**  — does every file contain the ``> <Uni-Dock RESULT>`` block?
   (if not, strict parsing would break existing pipelines)
2. **consistency** — does ``ENERGY == INTER + INTRA`` hold?
3. **impact**    — how many scores change versus the legacy loose ``ENERGY=``
   regex, and in which direction.
"""

from __future__ import annotations

import csv
import glob
import json
import re
from collections import Counter
from pathlib import Path
from statistics import median

from ddpo.utils.unidock_score import parse_unidock_output

ROOT = Path("/home/md666/data/disk6/DiffB110")
OUT_DIR = ROOT / "analysis/unidock_parse_gate_260930"

CORPORA: dict[str, list[str]] = {
    "ab_200k_A_noh": [str(p) for p in sorted(glob.glob(
        str(ROOT / "analysis/unidock_hydrogen_ab_260930/A_noh_seed42/docking/*_out.sdf")))],
    "ab_200k_B_h": [str(p) for p in sorted(glob.glob(
        str(ROOT / "analysis/unidock_hydrogen_ab_260930/B_h_seed42/docking/*_out.sdf")))],
    "ab_200k_C_h_seed0": [str(p) for p in sorted(glob.glob(
        str(ROOT / "analysis/unidock_hydrogen_ab_260930/C_h_seed0/docking/*_out.sdf")))],
    "ab_200k_D_nocoords": [str(p) for p in sorted(glob.glob(
        str(ROOT / "analysis/unidock_hydrogen_ab_260930/D_nocoords/docking/*_out.sdf")))],
    "ab_gen_A_noh": [str(p) for p in sorted(glob.glob(
        str(ROOT / "analysis/unidock_hydrogen_ab_260930/generation/A_noh/batch_outputs/batch_*/*_out.sdf")))],
    "ab_gen_B_h": [str(p) for p in sorted(glob.glob(
        str(ROOT / "analysis/unidock_hydrogen_ab_260930/generation/B_h/batch_outputs/batch_*/*_out.sdf")))],
    "pool_part_0000": [str(p) for p in sorted(glob.glob(
        str(ROOT / "analysis/reference_support_experiment/docking_subset/docking/part_0000/batch_outputs/batch_*/*_out.sdf")))],
    "gen_260902_docking": [str(p) for p in sorted(glob.glob(
        str(ROOT / "analysis/diffsbdd_global_e2po_structure_matched_x_only_lr1e5_full_generation_1000_260902/docking/batch_outputs/batch_*/*_out.sdf")))],
}

# legacy behaviour: first matching pattern wins (loose ENERGY= is second)
LEGACY_PATTERNS = [
    re.compile(r"REMARK\s+VINA RESULT:\s*([-+]?\d+(?:\.\d+)?)"),
    re.compile(r"ENERGY=\s*([-+]?\d+(?:\.\d+)?)"),
]


def legacy_score(path: Path) -> float | None:
    text = path.read_text(encoding="utf-8", errors="ignore")
    for pattern in LEGACY_PATTERNS:
        match = pattern.search(text)
        if match:
            return float(match.group(1))
    return None


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    report: dict[str, dict[str, object]] = {}

    for name, files in CORPORA.items():
        statuses: Counter[str] = Counter()
        reasons: Counter[str] = Counter()
        changed: list[tuple[float, float]] = []
        no_block = 0
        for path_str in files:
            path = Path(path_str)
            result = parse_unidock_output(path)
            statuses[result.status] += 1
            if result.reason:
                reasons[result.reason] += 1
            if not result.block_found:
                no_block += 1
            old = legacy_score(path)
            new = result.energy if result.is_usable else None
            if old is not None and new is not None and abs(old - new) > 1e-6:
                changed.append((old, new))
            rows.append({
                "corpus": name,
                "file": path.name,
                "legacy": "" if old is None else old,
                "energy": "" if result.energy is None else result.energy,
                "status": result.status,
                "reason": result.reason,
                "inter": result.inter,
                "intra": result.intra,
                "unbound": result.unbound,
            })
        report[name] = {
            "n_files": len(files),
            "missing_block": no_block,
            "statuses": dict(statuses),
            "reasons": dict(reasons),
            "n_value_changed": len(changed),
            "max_legacy_value": max((o for o, _ in changed), default=None),
        }
        print(json.dumps({name: report[name]}, ensure_ascii=False))

    with (OUT_DIR / "parser_verification.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    (OUT_DIR / "parser_verification.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    total = sum(v["n_files"] for v in report.values())
    blocks = sum(v["missing_block"] for v in report.values())
    print(f"\nTOTAL files={total}  missing_uni_dock_block={blocks}")

    # pooled summary of statuses
    pooled: Counter[str] = Counter()
    for value in report.values():
        pooled.update(value["statuses"])
    print("pooled statuses:", dict(pooled))

    suspect = [r for r in rows if r["status"] in ("suspect_backend", "parse_error")]
    print(f"suspect/parse_error rows: {len(suspect)} / {len(rows)}")
    if suspect:
        shown = {r["reason"] for r in suspect}
        print("reasons:", shown)


if __name__ == "__main__":
    main()
