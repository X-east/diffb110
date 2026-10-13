#!/usr/bin/env python
"""Write attempt-normalized DiffGui benchmark summary."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True, type=Path)
    parser.add_argument("--attempts", required=True, type=int)
    parser.add_argument("--diffusion-steps", type=int)
    parser.add_argument("--source-note")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = args.exp / "filtered_sanitize_valid" / "sanitize_valid_summary.csv"
    rows = [
        row
        for row in csv.DictReader(summary_path.open(newline="", encoding="utf-8"))
        if row.get("model", "").lower() == "diffgui"
    ]
    if len(rows) != 1:
        raise ValueError(f"Expected one DiffGui summary row, got {len(rows)} in {summary_path}")

    row = rows[0]
    manifest_path = args.exp / "run_manifest.csv"
    manifest_rows = [
        manifest_row
        for manifest_row in csv.DictReader(manifest_path.open(newline="", encoding="utf-8"))
        if manifest_row.get("model", "").lower() == "diffgui"
    ]
    seed_offsets = {
        int(manifest_row["seed"]) - int(manifest_row["gpu_id"])
        for manifest_row in manifest_rows
        if manifest_row.get("seed") and manifest_row.get("gpu_id")
    }
    if len(seed_offsets) == 1:
        seed_offset = seed_offsets.pop()
        seed_rule = "seed=gpu_id" if seed_offset == 0 else f"seed=gpu_id+{seed_offset}"
    else:
        seed_rule = "see run_manifest.csv"

    manifest_timesteps = {
        int(manifest_row["timesteps"])
        for manifest_row in manifest_rows
        if manifest_row.get("timesteps")
    }
    if args.diffusion_steps is not None:
        diffusion_steps = args.diffusion_steps
    elif len(manifest_timesteps) == 1:
        diffusion_steps = manifest_timesteps.pop()
    else:
        raise ValueError(
            "Could not infer a single DiffGui diffusion step count from "
            f"{manifest_path}; pass --diffusion-steps explicitly."
        )

    out = args.exp / "attempt_normalized_summary.csv"
    fields = [
        "model",
        "diffusion_steps",
        "attempts",
        "seed_rule",
        "sdf_written",
        "rdkit_readable",
        "sanitize_valid",
        "written_per_attempt",
        "sanitize_valid_per_attempt",
        "source_note",
    ]
    result = {
        "model": "DiffGui",
        "diffusion_steps": diffusion_steps,
        "attempts": args.attempts,
        "seed_rule": seed_rule,
        "sdf_written": int(row["sdf_records"]),
        "rdkit_readable": int(row["rdkit_readable"]),
        "sanitize_valid": int(row["sanitize_valid"]),
        "written_per_attempt": int(row["sdf_records"]) / args.attempts,
        "sanitize_valid_per_attempt": int(row["sanitize_valid"]) / args.attempts,
        "source_note": args.source_note
        or f"DiffGui {diffusion_steps}-step run with explicit sample.num_timesteps",
    }

    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(result)

    print(json.dumps({"attempt_summary": str(out), **result}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
