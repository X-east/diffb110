#!/usr/bin/env python3
"""Apply the frozen label validity gate to an ECFP4 reward sidecar.

The trajectory index and the later molecular-label sidecar disagree for a
small number of Kekulize failures.  This additive step keeps the raw ECFP4
results intact and emits a strict sidecar that excludes any endpoint marked
invalid by the frozen label sidecar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    """Write the strict sidecar, summary, and reward QA plot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-sidecar", type=Path, required=True)
    parser.add_argument("--label-sidecar", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.raw_sidecar, args.label_sidecar):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.raw_sidecar, low_memory=False)
    labels = pd.read_csv(
        args.label_sidecar,
        usecols=["sample_id", "rdkit_valid", "sanitize_success", "failure_reason"],
        low_memory=False,
    )
    merged = raw.merge(labels, on="sample_id", how="left", validate="one_to_one", suffixes=("", "_label"))
    if merged[["rdkit_valid", "sanitize_success", "failure_reason"]].isna().any().any():
        raise ValueError("label sidecar does not cover every raw ECFP4 endpoint")

    strict_success = merged["ecfp4_success"].eq(1) & merged["rdkit_valid"].eq(1) & merged["sanitize_success"].eq(1)
    corrected = int((merged["ecfp4_success"].eq(1) & ~strict_success).sum())
    merged["ecfp4_success"] = strict_success.astype(int)
    merged.loc[~strict_success, ["ecfp4_max_similarity", "ecfp4_top5_mean_similarity"]] = np.nan
    merged["ecfp4_reward"] = merged["ecfp4_top5_mean_similarity"].fillna(-1.0)
    merged["ecfp4_failure_type"] = np.where(strict_success, "success", merged["failure_reason"].astype(str))
    merged["ecfp4_reward_finite"] = np.isfinite(merged["ecfp4_reward"])
    merged["validity_gate"] = "label_sidecar_rdkit_valid_and_sanitize_success"
    output_columns = [
        "sample_id", "split", "rdkit_valid", "failure_type", "ecfp4_failure_type",
        "num_nodes", "shard_path", "shard_index", "smiles", "ecfp4_success",
        "ecfp4_max_similarity", "ecfp4_top5_mean_similarity", "ecfp4_reward",
        "ecfp4_reward_finite", "validity_gate",
    ]
    output = merged[output_columns].copy()
    output_path = args.output_dir / "ecfp4_reward_sidecar_strict.csv"
    output.to_csv(output_path, index=False)

    valid = output.loc[output["ecfp4_success"].eq(1), "ecfp4_reward"]
    summary = {
        "status": "completed",
        "raw_sidecar": str(args.raw_sidecar.resolve()),
        "raw_sidecar_sha256": sha256_file(args.raw_sidecar),
        "label_sidecar": str(args.label_sidecar.resolve()),
        "label_sidecar_sha256": sha256_file(args.label_sidecar),
        "output": str(output_path.resolve()),
        "output_sha256": sha256_file(output_path),
        "endpoints": int(len(output)),
        "strict_valid": int(output["ecfp4_success"].sum()),
        "technical_failures": int((output["ecfp4_success"] == 0).sum()),
        "corrected_from_raw": corrected,
        "reward_name": "ecfp4_top5_mean_similarity",
        "reward_min": float(valid.min()),
        "reward_max": float(valid.max()),
        "reward_mean": float(valid.mean()),
        "reward_median": float(valid.median()),
        "reward_quantiles": {str(q): float(valid.quantile(q)) for q in (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)},
    }
    (args.output_dir / "ecfp4_reward_strict_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].hist(valid.to_numpy(), bins=60, color="#4472c4", alpha=0.9)
    axes[0].set_xlabel("ECFP4 top-5 mean Tanimoto")
    axes[0].set_ylabel("Strict-valid endpoints")
    axes[0].set_title("Strict ECFP4 reward distribution")
    axes[1].boxplot(valid.to_numpy(), vert=True, patch_artist=True, boxprops={"facecolor": "#70ad47"})
    axes[1].set_ylabel("ECFP4 top-5 mean Tanimoto")
    axes[1].set_title("Strict reward summary")
    figure.savefig(args.output_dir / "ecfp4_reward_strict_distribution.png", dpi=220)
    figure.savefig(args.output_dir / "ecfp4_reward_strict_distribution.svg")
    plt.close(figure)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
