#!/usr/bin/env python3
"""Create an immutable candidate training manifest with finite exploratory labels."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--audit", type=Path, required=True)
    ap.add_argument("--run-manifest", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_csv(args.candidates, low_memory=False)
    audit = pd.read_csv(args.audit, low_memory=False)
    if len(candidates) != 419 or len(audit) != 419:
        raise ValueError(f"expected 419 candidates/audit rows, got {len(candidates)}/{len(audit)}")
    merged = candidates.merge(audit, on="trajectory_id", validate="one_to_one", suffixes=("", "_audit"))
    if not merged.shape_ok.astype(bool).all():
        raise ValueError("candidate trajectory audit contains non-conforming shapes")
    checkpoint_sha = json.loads(args.run_manifest.read_text())['summary']['checkpoint_sha256']
    # Exploratory quality score only; it is not a measured activity reward.
    merged["exploratory_candidate_reward"] = (
        merged.qed.clip(0, 1)
        + (1.0 - merged.sa_raw.clip(0, 10) / 10.0)
        + (1.0 - merged.ecfp4_distance.clip(0, 1))
    ) / 3.0
    merged["label_name"] = "rnd53_nearest100_strict_chemistry_candidate"
    merged["reward_is_formal"] = False
    merged["source_checkpoint_sha256"] = checkpoint_sha
    merged["candidate_facts_sha256"] = sha256(args.candidates)
    merged["audit_sha256"] = sha256(args.audit)
    out = merged.sort_values("trajectory_id", kind="mergesort")
    out.to_csv(args.output_dir / "candidate_training_manifest.csv.gz", index=False, compression="gzip")
    summary = {
        "status": "complete",
        "candidate_count": int(len(out)),
        "unique_trajectory_count": int(out.trajectory_id.nunique()),
        "unique_scaffold_count": int(out.murcko_scaffold.nunique()),
        "all_trajectory_shapes_verified": bool(out.shape_ok.astype(bool).all()),
        "reward_is_formal": False,
        "checkpoint_sha256": checkpoint_sha,
        "note": "Exploratory finite candidate score for filtered-BC data preparation only; not the formal full-run reward sidecar.",
    }
    (args.output_dir / "candidate_training_manifest_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
