#!/usr/bin/env python3
"""Audit candidate trajectory IDs against saved DiffSBDD trajectory shards."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
import torch


PATTERN = re.compile(r"^r(?P<rank>\d+)_s(?P<shard>\d+)_m(?P<mol>\d+)$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_csv(args.candidates, low_memory=False)
    rows = []
    for trajectory_id in candidates.trajectory_id.astype(str):
        match = PATTERN.fullmatch(trajectory_id)
        row = {"trajectory_id": trajectory_id, "mapping_parse_ok": bool(match)}
        if not match:
            row["failure_reason"] = "trajectory_id_parse_failed"
            rows.append(row)
            continue
        rank, shard_index, mol_index = (int(match.group(key)) for key in ("rank", "shard", "mol"))
        shard_path = args.run_dir / "shards" / f"rank_{rank:02d}" / f"shard_{shard_index:04d}.pt"
        row.update({"rank": rank, "shard_index": shard_index, "mol_index": mol_index, "shard_path": str(shard_path), "shard_exists": shard_path.is_file()})
        if not shard_path.is_file():
            row["failure_reason"] = "shard_missing"
            rows.append(row)
            continue
        try:
            payload = torch.load(shard_path, map_location="cpu", weights_only=False)
            metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
            latent = payload.get("latents", payload.get("latent", payload.get("trajectory"))) if isinstance(payload, dict) else None
            logps = payload.get("logps") if isinstance(payload, dict) else None
            if latent is None:
                row["failure_reason"] = "latent_missing"
            elif int(latent.shape[0]) <= mol_index:
                row["failure_reason"] = "mol_index_out_of_range"
            else:
                row["trajectory_steps"] = int(latent.shape[1])
                row["latent_nodes"] = int(latent.shape[2])
                row["latent_features"] = int(latent.shape[3])
                row["logps_steps"] = int(logps.shape[1]) if logps is not None and len(logps.shape) > 1 else None
                row["shape_ok"] = row["trajectory_steps"] == 1002 and row["logps_steps"] == 1001
                row["metadata_shard_index"] = metadata.get("shard_index")
                row["failure_reason"] = "" if row["shape_ok"] else "trajectory_shape_mismatch"
            row["sha256"] = sha256(shard_path)
        except Exception as exc:  # pragma: no cover - audit must record per-file errors
            row["failure_reason"] = f"load_error:{type(exc).__name__}:{exc}"
        rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(args.output_dir / "candidate_trajectory_audit.csv", index=False)
    shape_series = result["shape_ok"] if "shape_ok" in result.columns else pd.Series(False, index=result.index)
    exists_series = result["shard_exists"] if "shard_exists" in result.columns else pd.Series(False, index=result.index)
    parse_series = result["mapping_parse_ok"] if "mapping_parse_ok" in result.columns else pd.Series(False, index=result.index)
    failure_series = result["failure_reason"] if "failure_reason" in result.columns else pd.Series("unknown", index=result.index)
    summary = {
        "status": "complete",
        "candidate_rows": int(len(candidates)),
        "unique_candidate_trajectory_ids": int(candidates.trajectory_id.nunique()),
        "parsed": int(parse_series.fillna(False).sum()),
        "shards_existing": int(exists_series.fillna(False).sum()),
        "shape_ok": int(shape_series.fillna(False).sum()),
        "failure_reasons": failure_series.fillna("unknown").value_counts().to_dict(),
        "source_run_dir": str(args.run_dir.resolve()),
    }
    (args.output_dir / "candidate_trajectory_audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
