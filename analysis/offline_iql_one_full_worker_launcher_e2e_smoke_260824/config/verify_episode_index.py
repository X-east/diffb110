#!/usr/bin/env python3
"""Verify that a complete episode-index manifest is ready for labeling."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _canonical_index_digest(path: Path) -> tuple[str, int]:
    """Return the index module's canonical digest and JSONL row count."""
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid index JSON at line {line_number}") from exc
            if not isinstance(row, dict) or not str(row.get("sample_id", "")):
                raise ValueError(f"invalid index row at line {line_number}")
            rows.append(row)
    rows.sort(key=lambda row: str(row["sample_id"]))
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest(), len(rows)


def _parse_args() -> argparse.Namespace:
    """Parse expected manifest paths and denominator counts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, default=200_000)
    parser.add_argument("--expected-shards", type=int, default=20_000)
    return parser.parse_args()


def main() -> None:
    """Reject partial or hash-inconsistent indexes."""
    args = _parse_args()
    if not args.index.is_file() or not args.summary.is_file():
        raise FileNotFoundError(
            f"index or summary is missing: index={args.index} summary={args.summary}"
        )
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    if summary.get("status") != "completed":
        raise ValueError(f"index summary status is not completed: {summary.get('status')}")
    if int(summary.get("episodes", -1)) != args.expected_episodes:
        raise ValueError(
            f"episode denominator mismatch: {summary.get('episodes')} != "
            f"{args.expected_episodes}"
        )
    if int(summary.get("shards", -1)) != args.expected_shards:
        raise ValueError(
            f"shard denominator mismatch: {summary.get('shards')} != "
            f"{args.expected_shards}"
        )
    expected_sha256 = str(summary.get("episode_index_sha256", ""))
    actual_sha256, line_count = _canonical_index_digest(args.index)
    if expected_sha256 != actual_sha256:
        raise ValueError(
            f"index SHA256 mismatch: summary={expected_sha256} actual={actual_sha256}"
        )
    if line_count != args.expected_episodes:
        raise ValueError(f"index line count mismatch: {line_count}")
    print(
        json.dumps(
            {
                "status": "verified",
                "index": str(args.index.resolve()),
                "summary": str(args.summary.resolve()),
                "episodes": line_count,
                "shards": int(summary["shards"]),
                "episode_index_sha256": actual_sha256,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
