"""Aggregate one DiffGui multi-GPU sampling batch into one 500-molecule set."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


def _latest_run_dir(gpu_output_dir: Path) -> Path:
    run_dirs = [path for path in gpu_output_dir.iterdir() if path.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No DiffGui run directory found in {gpu_output_dir}")
    return max(run_dirs, key=lambda path: path.stat().st_mtime)


def _find_sdf_dir(run_dir: Path) -> Path:
    candidates = sorted(path for path in run_dir.glob("*_SDF") if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No *_SDF directory found in {run_dir}")
    return candidates[0]


def _load_smiles(run_dir: Path) -> list[str]:
    smiles_path = run_dir / "SMILES.txt"
    if not smiles_path.exists():
        raise FileNotFoundError(f"Missing {smiles_path}")
    return [line.strip() for line in smiles_path.read_text().splitlines() if line.strip()]


def _merge_sdf(records: list[dict[str, str]], output_path: Path) -> int:
    try:
        from rdkit import Chem
    except Exception:
        return 0

    writer = Chem.SDWriter(str(output_path))
    written = 0
    for record in records:
        sdf_path = record.get("aggregated_sdf")
        if not sdf_path:
            continue
        supplier = Chem.SDMolSupplier(str(sdf_path), sanitize=False, removeHs=False)
        mol = next((item for item in supplier if item is not None), None)
        if mol is None:
            continue
        mol.SetProp("aggregated_index", record["aggregated_index"])
        mol.SetProp("gpu_id", record["gpu_id"])
        mol.SetProp("source_index", record["source_index"])
        mol.SetProp("smiles", record["smiles"])
        writer.write(mol)
        written += 1
    writer.close()
    return written


def aggregate(root: Path, steps: int, counts: list[int]) -> dict[str, object]:
    step_output_dir = root / "outputs" / f"steps_{steps}"
    aggregate_dir = root / "aggregated" / f"steps_{steps}"
    sdf_out_dir = aggregate_dir / "sdf"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    sdf_out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, str]] = []
    per_gpu = []
    for gpu_id, target_count in enumerate(counts):
        gpu_output_dir = step_output_dir / f"gpu_{gpu_id}"
        run_dir = _latest_run_dir(gpu_output_dir)
        sdf_dir = _find_sdf_dir(run_dir)
        smiles = _load_smiles(run_dir)
        if len(smiles) < target_count:
            raise RuntimeError(
                f"GPU {gpu_id} for steps={steps} produced {len(smiles)} molecules, "
                f"fewer than requested {target_count}"
            )

        selected = smiles[:target_count]
        copied_sdf = 0
        for source_index, smi in enumerate(selected):
            aggregate_index = len(records)
            source_sdf = sdf_dir / f"{source_index}.sdf"
            dest_sdf = sdf_out_dir / f"{aggregate_index:06d}_gpu{gpu_id}_{source_index:04d}.sdf"
            dest_value = ""
            if source_sdf.exists():
                shutil.copy2(source_sdf, dest_sdf)
                copied_sdf += 1
                dest_value = str(dest_sdf)
            records.append(
                {
                    "aggregated_index": str(aggregate_index),
                    "steps": str(steps),
                    "gpu_id": str(gpu_id),
                    "source_index": str(source_index),
                    "smiles": smi,
                    "source_run_dir": str(run_dir),
                    "source_sdf": str(source_sdf),
                    "aggregated_sdf": dest_value,
                }
            )
        per_gpu.append(
            {
                "gpu_id": gpu_id,
                "target_count": target_count,
                "produced_smiles": len(smiles),
                "selected_count": len(selected),
                "copied_sdf": copied_sdf,
                "run_dir": str(run_dir),
            }
        )

    smiles_out = aggregate_dir / "all_SMILES.txt"
    smiles_out.write_text("\n".join(record["smiles"] for record in records) + "\n")

    manifest_out = aggregate_dir / "manifest.csv"
    with manifest_out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    merged_sdf_out = aggregate_dir / "all_molecules.sdf"
    merged_sdf_count = _merge_sdf(records, merged_sdf_out)

    duplicate_smiles = len(records) - len({record["smiles"] for record in records})
    summary = {
        "steps": steps,
        "target_total": sum(counts),
        "selected_total": len(records),
        "unique_smiles": len({record["smiles"] for record in records}),
        "duplicate_smiles": duplicate_smiles,
        "merged_sdf_count": merged_sdf_count,
        "per_gpu": per_gpu,
        "smiles_path": str(smiles_out),
        "manifest_path": str(manifest_out),
        "merged_sdf_path": str(merged_sdf_out),
    }
    (aggregate_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if len(records) != sum(counts):
        raise RuntimeError(f"Expected {sum(counts)} records, got {len(records)}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--counts", default="84,84,83,83,83,83")
    args = parser.parse_args()

    counts = [int(item) for item in args.counts.split(",") if item]
    summary = aggregate(Path(args.root).resolve(), args.steps, counts)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
