#!/usr/bin/env python
"""Aggregate DiffGui per-GPU SDF outputs for one property-conditioning group."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _count_sdf_records(path: Path) -> int:
    """Return the number of molecule records in one SDF file.

    DiffGui outputs single-molecule SDF files terminated by ``M  END``
    (not multi-molecule SDF with ``$$$$`` separators).
    """
    text = path.read_text(encoding="utf-8", errors="ignore")
    count = text.count("M  END")
    if count == 0:
        # Fallback: treat any non-empty SDF as 1 molecule
        if text.strip():
            return 1
    return count


def aggregate_sdfs(
    group_dir: Path,
    gpu_ids: list[str],
    output_sdf: Path,
    manifest_csv: Path,
    run_prefix: str | None = None,
) -> int:
    """Concatenate SDF files and write a manifest for traceability."""
    output_sdf.parent.mkdir(parents=True, exist_ok=True)
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str | int]] = []
    total = 0
    with output_sdf.open("w", encoding="utf-8") as out:
        for gpu_id in gpu_ids:
            gpu_dir = group_dir / f"gpu_{gpu_id}"
            if not gpu_dir.is_dir():
                continue
            candidates = [p for p in gpu_dir.iterdir() if p.is_dir()]
            if run_prefix:
                candidates = [p for p in candidates if p.name.startswith(run_prefix)]
            if not candidates:
                continue
            run_dir = max(candidates, key=lambda p: p.stat().st_mtime)
            for sdf_dir in sorted(run_dir.glob("*_SDF")):
                if not sdf_dir.is_dir():
                    continue
                for sdf_file in sorted(sdf_dir.glob("*.sdf")):
                    if sdf_file.name.startswith("traj_"):
                        continue
                    n_mol = _count_sdf_records(sdf_file)
                    if n_mol <= 0:
                        continue
                    out.write(sdf_file.read_text(encoding="utf-8", errors="ignore"))
                    total += n_mol
                    rows.append(
                        {
                            "gpu_id": gpu_id,
                            "run_dir": str(run_dir),
                            "sdf_file": str(sdf_file),
                            "n_mol": n_mol,
                        }
                    )

    with manifest_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["gpu_id", "run_dir", "sdf_file", "n_mol"])
        writer.writeheader()
        writer.writerows(rows)

    return total


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group-dir", required=True, type=Path)
    parser.add_argument("--gpu-ids", nargs="+", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--run-prefix", default=None)
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    total = aggregate_sdfs(
        group_dir=args.group_dir,
        gpu_ids=args.gpu_ids,
        output_sdf=args.out,
        manifest_csv=args.manifest,
        run_prefix=args.run_prefix,
    )
    print(f"[aggregate] group_dir={args.group_dir}")
    print(f"[aggregate] run_prefix={args.run_prefix or '*'}")
    print(f"[aggregate] molecules={total}")
    print(f"[aggregate] sdf={args.out}")
    print(f"[aggregate] manifest={args.manifest}")


if __name__ == "__main__":
    main()
