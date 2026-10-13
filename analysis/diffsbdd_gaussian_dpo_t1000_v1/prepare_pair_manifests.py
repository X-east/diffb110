"""Create immutable T=1000 Global-DPO pair manifests."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from ddpo.diffusion_dpo.pairs import (
    build_exact_atom_count_pairs,
    load_selected_manifest,
    pairs_to_frame,
)


def sha256(path: Path) -> str:
    """Hash one written pair manifest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """Build train/validation/test pair files with fixed seed semantics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_selected_manifest(args.manifest)
    summary: dict[str, object] = {"seed": args.seed, "source_manifest": str(args.manifest.resolve())}
    for split in ("train", "validation", "test"):
        pairs = build_exact_atom_count_pairs(frame, split, seed=args.seed + (split != "train"))
        output = args.output_dir / f"{split}_pairs.csv"
        pairs_to_frame(pairs).to_csv(output, index=False)
        summary[split] = {"pairs": len(pairs), "sha256": sha256(output)}
    (args.output_dir / "pair_manifest_summary.json").write_text(
        __import__("json").dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
