#!/usr/bin/env python3
"""Build explicit AliDiff-style good/bad preference pairs for DiffSBDD.

The original AliDiff data builder creates an ordered ``winner -> loser`` map
inside one pocket/protein group and the trainer consumes one explicit loser
for each winner.  This adaptation deliberately replaces the continuous
nearest-worse rule with two auditable docking pools so that the first pilot
tests a clear preference signal:

* good: ``docking_raw <= --good-max``;
* bad: ``docking_raw >= --bad-min``.

Pairs are built separately inside the frozen train/validation/test split to
avoid endpoint leakage.  Good endpoints are paired in deterministic order
with bad endpoints in the same split; bad endpoints may be reused, matching
the explicit-map behavior of AliDiff when the candidate pools have different
sizes.  This is a good/bad pilot, not an exact reproduction of AliDiff's
continuous adjacent-score rule.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


SAMPLE_ID_RE = re.compile(r"^r(?P<rank>\d+)_s(?P<shard>\d+)_m(?P<mol>\d+)$")
PAIR_FIELDS = [
    "pair_index",
    "split",
    "winner_id",
    "loser_id",
    "winner_docking_raw",
    "loser_docking_raw",
    "reward_winner",
    "reward_loser",
    "reward_gap",
    "winner_shard",
    "loser_shard",
    "winner_index",
    "loser_index",
    "winner_n_nodes",
    "loser_n_nodes",
    "n_nodes",
    "source_rule",
]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL objects and reject malformed rows."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def load_split_manifest(path: Path) -> tuple[dict[str, str], str]:
    """Load assignments and the manifest's canonical episode-index digest."""
    assignments: dict[str, str] = {}
    episode_index_digests: set[str] = set()
    for row in load_jsonl(path):
        sample_id = str(row.get("sample_id", ""))
        split = str(row.get("split", ""))
        episode_index_sha256 = str(row.get("episode_index_sha256", ""))
        if not sample_id or split not in {"train", "validation", "test"}:
            raise ValueError(f"invalid split row in {path}: {row}")
        if len(episode_index_sha256) != 64:
            raise ValueError(f"invalid episode_index_sha256 in {path}: {row}")
        if sample_id in assignments:
            raise ValueError(f"duplicate split assignment for {sample_id}")
        assignments[sample_id] = split
        episode_index_digests.add(episode_index_sha256.lower())
    if len(episode_index_digests) != 1:
        raise ValueError(
            f"split manifest must contain one episode-index digest, found {episode_index_digests}"
        )
    return assignments, next(iter(episode_index_digests))


def load_sidecars(label_dir: Path) -> dict[str, dict[str, str]]:
    """Load all shard sidecars and index rows by unique sample ID."""
    sidecars = sorted(label_dir.glob("shards/rank_*/shard_*/reward_sidecar.csv"))
    if not sidecars:
        raise FileNotFoundError(f"no shard sidecars found under {label_dir}")
    rows: dict[str, dict[str, str]] = {}
    for path in sidecars:
        with path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                sample_id = str(row.get("sample_id", ""))
                if not sample_id:
                    raise ValueError(f"sidecar row without sample_id: {path}")
                if sample_id in rows:
                    raise ValueError(f"duplicate sidecar sample_id: {sample_id}")
                row["_sidecar_path"] = str(path.resolve())
                rows[sample_id] = row
    return rows


def parse_location(row: dict[str, Any]) -> tuple[int, int, int]:
    """Return rank, shard, and molecule index from a frozen episode row."""
    match = SAMPLE_ID_RE.fullmatch(str(row.get("sample_id", "")))
    if match is None:
        raise ValueError(f"sample_id is not in rXX_sXXXX_mYYYY format: {row.get('sample_id')}")
    return tuple(int(match.group(name)) for name in ("rank", "shard", "mol"))


def finite_score(row: dict[str, str]) -> float | None:
    """Return a finite successful docking score, otherwise ``None``."""
    if row.get("docking_success") != "1":
        return None
    try:
        value = float(row.get("docking_raw", ""))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write one stable pair manifest."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAIR_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def build_pairs(
    episode_rows: list[dict[str, Any]],
    sidecars: dict[str, dict[str, str]],
    splits: dict[str, str],
    source_run_dir: Path,
    *,
    good_max: float,
    bad_min: float,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Construct deterministic good/bad pairs independently per split."""
    if good_max >= bad_min:
        raise ValueError("good-max must be lower than bad-min for a clear score gap")
    records: list[dict[str, Any]] = []
    missing_sidecar = 0
    unscored = 0
    for row in episode_rows:
        sample_id = str(row["sample_id"])
        sidecar = sidecars.get(sample_id)
        if sidecar is None:
            missing_sidecar += 1
            continue
        split = splits.get(sample_id)
        if split is None:
            raise ValueError(f"missing split assignment for {sample_id}")
        score = finite_score(sidecar)
        if score is None:
            unscored += 1
            continue
        rank, shard, mol_index = parse_location(row)
        records.append(
            {
                "sample_id": sample_id,
                "split": split,
                "score": score,
                "n_nodes": int(row["num_nodes"]),
                "shard": str((source_run_dir / str(row["shard_path"])).resolve()),
                "mol_index": mol_index,
                "rank": rank,
                "shard_index": shard,
            }
        )
    if missing_sidecar:
        raise ValueError(f"{missing_sidecar} episode rows have no sidecar")
    by_split: dict[str, list[dict[str, Any]]] = {name: [] for name in ("train", "validation", "test")}
    for record in records:
        by_split[record["split"]].append(record)

    outputs: dict[str, list[dict[str, Any]]] = {}
    counts: dict[str, Any] = {
        "indexed_episodes": len(episode_rows),
        "sidecar_rows": len(sidecars),
        "scored_episodes": len(records),
        "unscored_episodes": unscored,
        "good_max": good_max,
        "bad_min": bad_min,
        "source_rule": "good_bad_thresholds_round_robin_within_split_v1",
        "splits": {},
    }
    for split in ("train", "validation", "test"):
        group = by_split[split]
        good = sorted(
            [record for record in group if record["score"] <= good_max],
            key=lambda record: (record["score"], record["sample_id"]),
        )
        bad = sorted(
            [record for record in group if record["score"] >= bad_min],
            key=lambda record: (record["score"], record["sample_id"]),
        )
        if good and not bad:
            raise ValueError(f"split={split} has good endpoints but no bad endpoints")
        pairs: list[dict[str, Any]] = []
        for pair_index, winner in enumerate(good):
            loser = bad[pair_index % len(bad)]
            pairs.append(
                {
                    "pair_index": pair_index,
                    "split": split,
                    "winner_id": winner["sample_id"],
                    "loser_id": loser["sample_id"],
                    "winner_docking_raw": winner["score"],
                    "loser_docking_raw": loser["score"],
                    "reward_winner": -winner["score"],
                    "reward_loser": -loser["score"],
                    "reward_gap": loser["score"] - winner["score"],
                    "winner_shard": winner["shard"],
                    "loser_shard": loser["shard"],
                    "winner_index": winner["mol_index"],
                    "loser_index": loser["mol_index"],
                    "winner_n_nodes": winner["n_nodes"],
                    "loser_n_nodes": loser["n_nodes"],
                    # Legacy field retained for old readers; new readers use
                    # winner_n_nodes and loser_n_nodes when present.
                    "n_nodes": winner["n_nodes"],
                    "source_rule": "good_bad_thresholds_round_robin_within_split_v1",
                }
            )
        outputs[split] = pairs
        counts["splits"][split] = {
            "eligible": len(group),
            "good": len(good),
            "bad": len(bad),
            "pairs": len(pairs),
            "unique_winners": len({pair["winner_id"] for pair in pairs}),
            "unique_losers": len({pair["loser_id"] for pair in pairs}),
            "reward_gap_min": min((pair["reward_gap"] for pair in pairs), default=None),
            "reward_gap_max": max((pair["reward_gap"] for pair in pairs), default=None),
        }
    return outputs, counts


def parse_args() -> argparse.Namespace:
    """Parse pair-builder inputs and auditable thresholds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-index", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--good-max", type=float, default=-8.0)
    parser.add_argument("--bad-min", type=float, default=-5.0)
    return parser.parse_args()


def main() -> None:
    """Build and validate the three split-specific pair manifests."""
    args = parse_args()
    for path in (args.episode_index, args.split_manifest, args.source_run_dir):
        if not path.exists():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    episode_rows = load_jsonl(args.episode_index)
    sidecars = load_sidecars(args.label_dir)
    splits, canonical_episode_index_sha256 = load_split_manifest(args.split_manifest)
    outputs, summary = build_pairs(
        episode_rows,
        sidecars,
        splits,
        args.source_run_dir,
        good_max=args.good_max,
        bad_min=args.bad_min,
    )
    summary["episode_index_sha256"] = canonical_episode_index_sha256
    summary["episode_index_file_sha256"] = sha256_file(args.episode_index)
    summary["split_manifest_sha256"] = sha256_file(args.split_manifest)
    summary["sidecar_files"] = len(list(args.label_dir.glob("shards/rank_*/shard_*/reward_sidecar.csv")))
    summary["status"] = "completed"
    for split, rows in outputs.items():
        write_csv(args.output_dir / f"{split}_pairs.csv", rows)
        summary["splits"][split]["manifest_sha256"] = sha256_file(args.output_dir / f"{split}_pairs.csv")
    (args.output_dir / "pair_manifest_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
