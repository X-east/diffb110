"""Aggregate model SDF shards into one benchmark SDF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rdkit import Chem


def _iter_sdf_files(paths: list[Path]) -> list[Path]:
    sdf_files: list[Path] = []
    for path in paths:
        if path.is_dir():
            sdf_files.extend(sorted(path.rglob("*.sdf")))
        elif path.exists() and path.suffix.lower() == ".sdf":
            sdf_files.append(path)
    return sdf_files


def aggregate(inputs: list[Path], output: Path, model: str, limit: int | None) -> dict[str, object]:
    """Write up to limit readable molecules from input SDF files."""

    output.parent.mkdir(parents=True, exist_ok=True)
    sdf_files = _iter_sdf_files(inputs)
    writer = Chem.SDWriter(str(output))
    read_records = 0
    written = 0
    write_failed = 0
    per_file: list[dict[str, object]] = []
    for sdf_file in sdf_files:
        supplier = Chem.SDMolSupplier(str(sdf_file), sanitize=False, removeHs=False)
        file_readable = 0
        file_written = 0
        for mol in supplier:
            if mol is None:
                continue
            read_records += 1
            file_readable += 1
            if limit is not None and written >= limit:
                continue
            mol.SetProp("benchmark_model", model)
            mol.SetProp("benchmark_index", str(written))
            mol.SetProp("benchmark_source_sdf", str(sdf_file))
            try:
                writer.write(mol)
            except Exception as exc:
                write_failed += 1
                mol.SetProp("benchmark_write_error", repr(exc))
                continue
            written += 1
            file_written += 1
        per_file.append({"path": str(sdf_file), "readable": file_readable, "written": file_written})
        if limit is not None and written >= limit:
            break
    writer.close()

    summary = {
        "model": model,
        "input_sdf_files": len(sdf_files),
        "readable_records_seen": read_records,
        "written_records": written,
        "write_failed_records": write_failed,
        "limit": limit,
        "output": str(output),
        "per_file": per_file,
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps(aggregate(args.inputs, args.output, args.model, args.limit), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
