#!/usr/bin/env python
"""Build auditable positive/negative training roles from frozen offline labels."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ddpo.offline_rl.curation import (
    QUALITY_CLASSES,
    CurationConfig,
    build_curation_manifest,
    build_curation_split_manifest,
    sha256_file,
    write_curation_manifest,
)
from ddpo.offline_rl.index import load_episode_index, write_split_manifest
from ddpo.offline_rl.reward_sidecar import load_reward_sidecar


LOGGER = logging.getLogger("OfflineCuration")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse required immutable inputs and output location."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-index", required=True, type=Path)
    parser.add_argument("--reward-sidecar", required=True, type=Path)
    parser.add_argument("--label-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--curation-version",
        default="diffsbdd_drift_training_roles_v1",
    )
    parser.add_argument(
        "--split-version",
        default="diffsbdd_drift_scaffold_split_v1",
    )
    parser.add_argument(
        "--split-salt",
        default="diffsbdd-drift-scaffold-split-v1",
    )
    parser.add_argument(
        "--split-fractions",
        nargs=3,
        type=float,
        metavar=("TRAIN", "VALIDATION", "TEST"),
        default=(0.8, 0.1, 0.1),
    )
    return parser.parse_args(argv)


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    """Return a mapping-valued configuration node."""
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a mapping")
    return value


def _load_curation_config(path: Path, version: str) -> CurationConfig:
    """Reuse the exact strict actual-good thresholds from labeling YAML."""
    root = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), name="root")
    labeling = _mapping(root.get("offline_labeling"), name="offline_labeling")
    policy = _mapping(labeling.get("policy"), name="offline_labeling.policy")
    return CurationConfig(
        curation_version=version,
        docking_max=float(policy["actual_good_docking_max"]),
        ligand_efficiency_min=float(
            policy["actual_good_ligand_efficiency_min"]
        ),
        qed_min=float(policy["actual_good_qed_min"]),
        sa_max=float(policy["actual_good_sa_max"]),
        logp_min=float(policy["actual_good_logp_min"]),
        logp_max=float(policy["actual_good_logp_max"]),
        tpsa_min=float(policy["actual_good_tpsa_min"]),
        tpsa_max=float(policy["actual_good_tpsa_max"]),
        max_three_member_rings=int(
            policy["actual_good_max_three_member_rings"]
        ),
        max_fused_rings=int(policy["actual_good_max_fused_rings"]),
        max_complex_rings=int(policy["actual_good_max_complex_rings"]),
        max_brenk=int(policy["actual_good_max_brenk"]),
    )


def _write_plot(path: Path, counts: Mapping[str, int]) -> None:
    """Write a presentation-readable fixed-attempt class-count plot."""
    labels = list(QUALITY_CLASSES)
    values = [counts[label] for label in labels]
    figure, axis = plt.subplots(figsize=(12, 7))
    bars = axis.bar(labels, values, color="#4C78A8")
    axis.set_title("Offline training quality classes", fontsize=20)
    axis.set_ylabel("Episodes", fontsize=16)
    axis.tick_params(axis="x", labelrotation=18, labelsize=13)
    axis.tick_params(axis="y", labelsize=13)
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:,}",
            ha="center",
            va="bottom",
            fontsize=12,
        )
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> int:
    """Build the manifest and denominator-explicit audit artifacts."""
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for path in (args.episode_index, args.reward_sidecar, args.label_config):
        if not path.is_file():
            raise FileNotFoundError(f"required curation input not found: {path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index = load_episode_index(args.episode_index)
    sidecar = load_reward_sidecar(args.reward_sidecar)
    sidecar_sha256 = sha256_file(args.reward_sidecar)
    config = _load_curation_config(args.label_config, args.curation_version)
    manifest = build_curation_manifest(
        index,
        sidecar,
        reward_sidecar_sha256=sidecar_sha256,
        config=config,
    )
    manifest_path = write_curation_manifest(
        args.output_dir / "curation_manifest.jsonl",
        manifest,
    )
    split_manifest = build_curation_split_manifest(
        index,
        manifest,
        split_version=args.split_version,
        salt=args.split_salt,
        fractions=tuple(args.split_fractions),
    )
    split_path = write_split_manifest(
        args.output_dir / "split_manifest.jsonl",
        split_manifest,
    )
    counts = manifest.counts()
    scaffold_counts = Counter(
        row.quality_class
        for row in manifest.rows
        if row.scaffold_key is not None
    )
    unique_scaffolds = {
        quality_class: len(
            {
                row.scaffold_key
                for row in manifest.rows
                if row.quality_class == quality_class
                and row.scaffold_key is not None
            }
        )
        for quality_class in QUALITY_CLASSES
    }
    primary_reason_counts = Counter(row.primary_reason for row in manifest.rows)
    failed_check_counts: Counter[str] = Counter()
    for row in manifest.rows:
        failed_check_counts.update(
            check for check in row.failed_checks.split(";") if check
        )
    attempts = len(manifest.rows)
    split_counts = Counter(
        split_manifest.split_for(row.sample_id) for row in manifest.rows
    )
    split_quality_counts = {
        split: {
            quality_class: sum(
                row.quality_class == quality_class
                and split_manifest.split_for(row.sample_id) == split
                for row in manifest.rows
            )
            for quality_class in QUALITY_CLASSES
        }
        for split in ("train", "validation", "test")
    }
    scaffold_splits: dict[str, set[str]] = {}
    for row in manifest.rows:
        if row.scaffold_key is None:
            continue
        scaffold_splits.setdefault(row.scaffold_key, set()).add(
            split_manifest.split_for(row.sample_id)
        )
    scaffold_overlap_count = sum(
        len(splits) > 1 for splits in scaffold_splits.values()
    )
    summary = {
        "format": "diffsbdd_offline_curation_summary_v1",
        "curation_version": manifest.curation_version,
        "attempts": attempts,
        "episode_index_sha256": index.sha256,
        "episode_index_file_sha256": sha256_file(args.episode_index),
        "reward_sidecar_sha256": sidecar_sha256,
        "label_config_sha256": sha256_file(args.label_config),
        "curation_manifest_sha256": sha256_file(manifest_path),
        "split_manifest_sha256": sha256_file(split_path),
        "split_version": args.split_version,
        "split_fractions": list(args.split_fractions),
        "split_counts": dict(sorted(split_counts.items())),
        "split_quality_counts": split_quality_counts,
        "scaffold_overlap_count": scaffold_overlap_count,
        "counts": counts,
        "rates_fixed_attempts": {
            key: value / attempts for key, value in counts.items()
        },
        "actor_bc_keep": sum(row.actor_bc_keep for row in manifest.rows),
        "critic_keep": sum(row.critic_keep for row in manifest.rows),
        "valid_smiles_by_class": dict(sorted(scaffold_counts.items())),
        "unique_scaffolds_by_class": unique_scaffolds,
        "primary_reason_counts": dict(sorted(primary_reason_counts.items())),
        "failed_check_counts": dict(sorted(failed_check_counts.items())),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "quality_class",
                "count",
                "rate_fixed_attempts",
                "unique_scaffolds",
            ),
        )
        writer.writeheader()
        for quality_class in QUALITY_CLASSES:
            writer.writerow(
                {
                    "quality_class": quality_class,
                    "count": counts[quality_class],
                    "rate_fixed_attempts": counts[quality_class] / attempts,
                    "unique_scaffolds": unique_scaffolds[quality_class],
                }
            )
    _write_plot(args.output_dir / "curation_class_counts.png", counts)
    LOGGER.info(
        "completed attempts=%d positive=%d structural_negative=%d "
        "target_negative=%d property_negative=%d technical_failure=%d",
        attempts,
        counts["positive"],
        counts["structural_negative"],
        counts["target_negative"],
        counts["property_negative"],
        counts["technical_failure"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
