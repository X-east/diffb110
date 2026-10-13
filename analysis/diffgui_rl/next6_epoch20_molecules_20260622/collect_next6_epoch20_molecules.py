#!/usr/bin/env python3
"""Collect labeled DiffGui next6 epoch-20 molecules."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

from rdkit import Chem, RDLogger


RDLogger.DisableLog("rdApp.*")

REPO_ROOT = Path(__file__).resolve().parents[3]
MATRIX_CSV = REPO_ROOT / "launch/archive/generated/diffgui_next6_matrix_20260618/matrix.csv"
OUT_ROOT = REPO_ROOT / "OUT"
OUT_DIR = Path(__file__).resolve().parent
EPOCH = "epoch_0020"


def read_matrix() -> list[dict[str, str]]:
    """Read the six-run matrix metadata."""
    with MATRIX_CSV.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def find_run_dir(run_name: str) -> Path:
    """Find the original 20-epoch run directory for one matrix run."""
    candidates = sorted(OUT_ROOT.glob(f"{run_name}_*"))
    if len(candidates) != 1:
        joined = ", ".join(str(path) for path in candidates) or "<none>"
        raise FileNotFoundError(f"expected one run dir for {run_name}, found {joined}")
    return candidates[0]


def set_label_props(
    mol: Chem.Mol,
    row: dict[str, str],
    run_dir: Path,
    source_sdf: Path,
    source_index: int,
) -> str:
    """Attach collection labels to one molecule and return its collection id."""
    idx = int(row["idx"])
    slug = row["slug"]
    collection_id = f"{idx:02d}_{slug}_{EPOCH}_mol{source_index:04d}"
    mol.SetProp("_Name", collection_id)
    props = {
        "collection_id": collection_id,
        "collection": "diffgui_next6_epoch20",
        "next6_idx": f"{idx:02d}",
        "next6_slug": slug,
        "next6_name": row["name"],
        "next6_weights": row["weights"],
        "next6_purpose": row["purpose"],
        "source_epoch": EPOCH,
        "source_run_dir": str(run_dir.relative_to(REPO_ROOT)),
        "source_sdf": str(source_sdf.relative_to(REPO_ROOT)),
        "source_mol_index": str(source_index),
    }
    for key, value in props.items():
        mol.SetProp(key, value)
    return collection_id


def write_readme(manifest_rows: list[dict[str, object]], total: int) -> None:
    """Write a concise human-readable summary."""
    lines = [
        "# DiffGui next6 epoch-20 molecules",
        "",
        "本目录整理 DiffGui next6 六次 20-epoch 训练的最后一个 epoch 分子。",
        "",
        "口径：使用每个原始 run 的 `epochs/epoch_0020/molecules.sdf`，不包含后续 `continue-to50` 目录。",
        "",
        "公共训练设置：20 epoch，8 GPU，每卡 18 分子，global batch 144，200 diffusion steps，lr=3e-5，PPO epochs=1，freeze last 3 blocks，bf16；这组没有设置 `kl_coef=0.05`。",
        "",
        "## 输出",
        "",
        "- `diffgui_next6_epoch0020_all_molecules_labeled.sdf`：六批合并，分子带清晰标签。",
        "- `*_epoch0020_molecules_labeled.sdf`：每个 run 单独一份。",
        "- `manifest.csv`：每个 run 的来源和数量。",
        "- `molecule_index.csv`：逐分子的 collection id 与来源索引。",
        "- `source_molecules_csv/`：原始 `molecules.csv` 备份。",
        "",
        "## 标签字段",
        "",
        "每个 SDF 分子均写入：`collection_id`、`next6_idx`、`next6_slug`、`next6_name`、`next6_weights`、`next6_purpose`、`source_epoch`、`source_run_dir`、`source_sdf`、`source_mol_index`。",
        "",
        f"总分子数：{total}",
        "",
        "| idx | slug | molecules | source |",
        "|---:|---|---:|---|",
    ]
    for row in manifest_rows:
        lines.append(
            f"| {row['idx']} | {row['slug']} | {row['molecules']} | `{row['source_sdf']}` |"
        )
    lines.append("")
    (OUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Collect labeled SDFs and CSV indexes."""
    matrix_rows = read_matrix()
    combined_path = OUT_DIR / "diffgui_next6_epoch0020_all_molecules_labeled.sdf"
    source_csv_dir = OUT_DIR / "source_molecules_csv"
    source_csv_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    molecule_rows: list[dict[str, object]] = []
    total = 0
    skipped = 0

    with Chem.SDWriter(str(combined_path)) as combined_writer:
        combined_writer.SetKekulize(False)
        for row in matrix_rows:
            run_dir = find_run_dir(row["name"])
            source_sdf = run_dir / "epochs" / EPOCH / "molecules.sdf"
            source_csv = run_dir / "epochs" / EPOCH / "molecules.csv"
            if not source_sdf.exists():
                raise FileNotFoundError(source_sdf)
            idx = int(row["idx"])
            slug = row["slug"]
            per_run_sdf = OUT_DIR / f"{idx:02d}_{slug}_epoch0020_molecules_labeled.sdf"
            copied_csv = source_csv_dir / f"{idx:02d}_{slug}_epoch0020_molecules.csv"
            if source_csv.exists():
                shutil.copy2(source_csv, copied_csv)

            run_count = 0
            run_skipped = 0
            supplier = Chem.SDMolSupplier(str(source_sdf), sanitize=False, removeHs=False)
            with Chem.SDWriter(str(per_run_sdf)) as run_writer:
                run_writer.SetKekulize(False)
                for source_index, mol in enumerate(supplier, start=1):
                    if mol is None:
                        skipped += 1
                        run_skipped += 1
                        continue
                    collection_id = set_label_props(mol, row, run_dir, source_sdf, source_index)
                    run_writer.write(mol)
                    combined_writer.write(mol)
                    run_count += 1
                    total += 1
                    molecule_rows.append(
                        {
                            "collection_id": collection_id,
                            "idx": f"{idx:02d}",
                            "slug": slug,
                            "source_epoch": EPOCH,
                            "source_mol_index": source_index,
                            "source_sdf": str(source_sdf.relative_to(REPO_ROOT)),
                            "run_dir": str(run_dir.relative_to(REPO_ROOT)),
                            "labeled_sdf": per_run_sdf.name,
                        }
                    )
            manifest_rows.append(
                {
                    "idx": f"{idx:02d}",
                    "slug": slug,
                    "name": row["name"],
                    "weights": row["weights"],
                    "purpose": row["purpose"],
                    "run_dir": str(run_dir.relative_to(REPO_ROOT)),
                    "source_sdf": str(source_sdf.relative_to(REPO_ROOT)),
                    "source_csv": str(source_csv.relative_to(REPO_ROOT)) if source_csv.exists() else "",
                    "labeled_sdf": per_run_sdf.name,
                    "copied_csv": str(copied_csv.relative_to(OUT_DIR)) if source_csv.exists() else "",
                    "molecules": run_count,
                    "skipped": run_skipped,
                }
            )

    with (OUT_DIR / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    with (OUT_DIR / "molecule_index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(molecule_rows[0].keys()))
        writer.writeheader()
        writer.writerows(molecule_rows)
    write_readme(manifest_rows, total)
    print(f"[collect] total={total} skipped={skipped} output={OUT_DIR}")


if __name__ == "__main__":
    main()
