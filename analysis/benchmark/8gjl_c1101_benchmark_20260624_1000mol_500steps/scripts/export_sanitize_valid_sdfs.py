"""Export sanitize-valid subsets from benchmark SDF files.

This script keeps only molecules that pass RDKit sanitization and writes one
filtered SDF per model together with a compact summary table.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from rdkit import Chem, RDLogger


RDLogger.DisableLog("rdApp.*")


def _sanitize_copy(mol: Chem.Mol) -> Chem.Mol | None:
    """Return a sanitized copy of `mol`, or `None` if sanitization fails."""

    copy = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(copy)
    except Exception:
        return None
    return copy


def _iter_model_specs(specs: list[str]) -> list[tuple[str, Path]]:
    """Parse `model=/path/to/file.sdf` arguments."""

    parsed: list[tuple[str, Path]] = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Expected model=/path/to/file.sdf, got {spec!r}")
        model, raw_path = spec.split("=", 1)
        parsed.append((model, Path(raw_path)))
    return parsed


def export_valid_sdfs(model_sdfs: list[str], out_dir: Path) -> dict[str, object]:
    """Write sanitize-valid SDF subsets and a summary table."""

    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    failure_examples: dict[str, list[dict[str, object]]] = {}

    for model, sdf_path in _iter_model_specs(model_sdfs):
        output_path = out_dir / f"{model}.sanitize_valid.sdf"
        writer = Chem.SDWriter(str(output_path))
        records = 0
        readable = 0
        valid = 0
        fail = 0
        reasons: Counter[str] = Counter()
        examples: list[dict[str, object]] = []

        supplier = Chem.SDMolSupplier(str(sdf_path), sanitize=False, removeHs=False)
        for idx, mol in enumerate(supplier):
            records += 1
            if mol is None:
                fail += 1
                reasons["supplier_none"] += 1
                if len(examples) < 8:
                    examples.append({"record": idx, "reason": "supplier_none"})
                continue

            readable += 1
            sanitized = _sanitize_copy(mol)
            if sanitized is None:
                fail += 1
                try:
                    tmp = Chem.Mol(mol)
                    Chem.SanitizeMol(tmp)
                except Exception as exc:
                    reason = str(exc).split("\n", 1)[0]
                else:
                    reason = "sanitize_failed"
                reasons[reason] += 1
                if len(examples) < 8:
                    examples.append({"record": idx, "reason": reason})
                continue

            sanitized.SetProp("benchmark_model", model)
            sanitized.SetProp("benchmark_source_sdf", str(sdf_path))
            sanitized.SetProp("benchmark_source_index", str(idx))
            sanitized.SetProp("benchmark_sanitize_valid", "1")
            writer.write(sanitized)
            valid += 1

        writer.close()

        rows.append(
            {
                "model": model,
                "source_sdf": str(sdf_path),
                "output_sdf": str(output_path),
                "sdf_records": records,
                "rdkit_readable": readable,
                "sanitize_valid": valid,
                "sanitize_fail": fail,
                "sanitize_rate": valid / records if records else 0.0,
                "readable_rate": readable / records if records else 0.0,
                "top_failure_reasons": "; ".join(
                    f"{reason} ({count})" for reason, count in reasons.most_common(5)
                ),
            }
        )
        failure_examples[model] = examples

    summary_csv = out_dir / "sanitize_valid_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    summary_json = out_dir / "sanitize_valid_summary.json"
    summary_json.write_text(
        json.dumps({"rows": rows, "failure_examples": failure_examples}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return {
        "out_dir": str(out_dir),
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
        "models": len(rows),
    }


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sdf", action="append", required=True, help="model=/path/to/file.sdf")
    args = parser.parse_args()
    print(json.dumps(export_valid_sdfs(args.sdf, args.out_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
