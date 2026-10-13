#!/usr/bin/env python3
"""Verify that a complete episode-index manifest is ready for labeling."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256_file(path: Path) -> str:
    """Return the SHA256 digest of one index file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    actual_sha256 = _sha256_file(args.index)
    if expected_sha256 != actual_sha256:
        raise ValueError(
            f"index SHA256 mismatch: summary={expected_sha256} actual={actual_sha256}"
        )
    with args.index.open("r", encoding="utf-8") as handle:
        line_count = sum(1 for _ in handle)
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
