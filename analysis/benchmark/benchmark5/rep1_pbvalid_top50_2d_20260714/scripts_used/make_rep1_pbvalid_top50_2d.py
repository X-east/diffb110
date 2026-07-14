"""Select and render rep1 PB-valid docking top 50 for DiffGui and DiffSBDD.

The page layout and molecule depiction reuse ``make_docking_2d_all_pages_v1``.
Selection is restricted to the first formal 1000-attempt repeat (rep1): the
formal docking inputs are already RDKit sanitize-valid, and PoseBusters rows
provide the additional ``pb_valid`` gate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem, rdBase

import make_docking_2d_all_pages_v1 as layout


BASE = Path(__file__).resolve().parent
FORMAL_ROOT = BASE / "8gjl_c1101_benchmark_20260706_1000steps_3rep_formal"
POSEBUSTERS_ROOT = FORMAL_ROOT / "posebusters_queue" / "per_task"
DOCKING_ROOT = FORMAL_ROOT / "docking" / "rep1"
REFERENCE_SDF = (
    BASE.parent
    / "8gjl_c1101_benchmark_20260603_500mol_500steps"
    / "inputs"
    / "8GJL_C1101_ligand.sdf"
)
DEFAULT_OUT = BASE / "rep1_pbvalid_top50_2d_20260714"
TOP_N = 50
ATTEMPTS = 1000
PAGE_SIZE = 25

MODELS = {
    "DiffGui": {
        "slug": "diffgui",
        "posebusters_csv": "rep1_diffgui_posebusters.csv",
    },
    "DiffSBDD": {
        "slug": "diffsbdd",
        "posebusters_csv": "rep1_diffsbdd_posebusters.csv",
    },
}


def _parse_bool(value: str) -> bool:
    """Parse a CSV boolean without accepting ambiguous truthy strings."""
    return value.strip().lower() == "true"


def _font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a size-aware font on Linux or Windows."""
    candidates = [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path(r"C:\Windows\Fonts\arialbd.ttf"),
        Path(r"C:\Windows\Fonts\arial.ttf"),
    ]
    if not bold:
        candidates = [candidates[1], candidates[3], candidates[0], candidates[2]]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _canonical_smiles(mol: Chem.Mol) -> str:
    """Return an isomeric canonical SMILES for traceability."""
    return Chem.MolToSmiles(
        Chem.RemoveHs(Chem.Mol(mol)), canonical=True, isomericSmiles=True
    )


def _load_reference() -> Chem.Mol:
    """Load the crystallographic C1101 ligand used as the page reference."""
    supplier = Chem.SDMolSupplier(str(REFERENCE_SDF), removeHs=False)
    reference = next((mol for mol in supplier if mol is not None), None)
    if reference is None:
        raise ValueError(f"Cannot read reference ligand: {REFERENCE_SDF}")
    Chem.SanitizeMol(reference)
    return reference


def _load_pb_rows(path: Path) -> list[dict[str, Any]]:
    """Load PoseBusters rows and normalize fields used by selection."""
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    normalized: list[dict[str, Any]] = []
    for row in rows:
        normalized.append(
            {
                "source_index": int(row["source_index"]),
                "name": row["name"],
                "docking_score": float(row["docking_score"]),
                "pb_valid": _parse_bool(row["pb_valid"]),
                "posebusters_empty": _parse_bool(row["posebusters_empty"]),
            }
        )
    return normalized


def _select_model(model: str, cfg: dict[str, str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply sanitize-valid and PB-valid gates, then select docking top N."""
    slug = cfg["slug"]
    source_sdf = DOCKING_ROOT / slug / "successful_docked_outputs.sdf"
    pb_csv = POSEBUSTERS_ROOT / cfg["posebusters_csv"]
    supplier = Chem.SDMolSupplier(str(source_sdf), removeHs=False)
    molecules = list(supplier)
    pb_rows = _load_pb_rows(pb_csv)
    if len(molecules) != len(pb_rows):
        raise ValueError(
            f"{model}: SDF/PoseBusters row mismatch: {len(molecules)} != {len(pb_rows)}"
        )

    legal_count = 0
    eligible: list[dict[str, Any]] = []
    for row in pb_rows:
        index = row["source_index"]
        if index < 0 or index >= len(molecules):
            raise IndexError(f"{model}: source_index out of range: {index}")
        mol = molecules[index]
        if mol is None:
            continue
        checked = Chem.Mol(mol)
        try:
            Chem.SanitizeMol(checked)
        except Exception:
            continue
        legal_count += 1
        if not row["pb_valid"] or row["posebusters_empty"]:
            continue
        sdf_score = float(mol.GetProp("DockingScore"))
        if not math.isclose(sdf_score, row["docking_score"], abs_tol=1e-6):
            raise ValueError(
                f"{model}: docking score mismatch at source_index={index}: "
                f"{sdf_score} != {row['docking_score']}"
            )
        eligible.append(
            {
                **row,
                "model": model,
                "slug": slug,
                "mol": mol,
                "source_sdf": str(source_sdf.resolve()),
                "posebusters_csv": str(pb_csv.resolve()),
                "canonical_isomeric_smiles": _canonical_smiles(mol),
            }
        )

    eligible.sort(key=lambda item: (item["docking_score"], item["source_index"]))
    selected = eligible[:TOP_N]
    if len(selected) != TOP_N:
        raise ValueError(f"{model}: only {len(selected)} eligible molecules; need {TOP_N}")
    for rank, item in enumerate(selected, 1):
        item["rank"] = rank
        item["display_name"] = f"mol{rank:03d}"

    stats = {
        "model": model,
        "attempts": ATTEMPTS,
        "sanitize_valid_count": legal_count,
        "sanitize_valid_rate": legal_count / ATTEMPTS,
        "pb_valid_count": len(eligible),
        "pb_valid_rate_among_sanitize_valid": len(eligible) / legal_count,
        "pb_valid_rate_per_attempt": len(eligible) / ATTEMPTS,
        "selected_count": len(selected),
        "best_docking_score": selected[0]["docking_score"],
        "top50_cutoff_docking_score": selected[-1]["docking_score"],
        "source_sdf": str(source_sdf.resolve()),
        "posebusters_csv": str(pb_csv.resolve()),
    }
    return selected, stats


def _write_selected_sdf(path: Path, selected: list[dict[str, Any]]) -> None:
    """Write selected 3D docked poses while adding selection provenance."""
    writer = Chem.SDWriter(str(path))
    try:
        for item in selected:
            mol = Chem.Mol(item["mol"])
            mol.SetProp("_Name", f"{item['model']}_rep1_pbvalid_top{item['rank']:02d}")
            mol.SetProp("SelectionModel", item["model"])
            mol.SetIntProp("SelectionRank", item["rank"])
            mol.SetIntProp("SourceIndex", item["source_index"])
            mol.SetBoolProp("PBValid", True)
            mol.SetDoubleProp("SelectionDockingScore", item["docking_score"])
            writer.write(mol)
    finally:
        writer.close()


def _mapping_row(item: dict[str, Any], page: int, row: int, column: int) -> dict[str, Any]:
    """Build one candidate-to-source mapping row."""
    return {
        "display_name": item["display_name"],
        "role": "candidate",
        "model": item["model"],
        "rank": item["rank"],
        "page": page,
        "row": row,
        "column": column,
        "docking_score": item["docking_score"],
        "pb_valid": True,
        "source_index": item["source_index"],
        "source_name": item["name"],
        "source_sdf": item["source_sdf"],
        "posebusters_csv": item["posebusters_csv"],
        "canonical_isomeric_smiles": item["canonical_isomeric_smiles"],
    }


def _render_model(
    model: str,
    selected: list[dict[str, Any]],
    reference: Chem.Mol,
    out_root: Path,
) -> dict[str, Any]:
    """Render two 5x5 pages using the supplied first-version layout."""
    slug = MODELS[model]["slug"]
    out_dir = out_root / slug
    highres_dir = out_dir / "pages_highres"
    preview_dir = out_dir / "pages_preview"
    highres_dir.mkdir(parents=True)
    preview_dir.mkdir()

    name_font = _font(42)
    score_font = _font(28)
    ref_name_font = _font(34)
    ref_score_font = _font(25, bold=False)
    reference_display, reference_changes = layout.prepare_display_mol(reference)
    reference_image = layout.draw_molecule(reference_display, 1500, 520)
    chemistry_counts: Counter[str] = Counter()
    mapping: list[dict[str, Any]] = []

    for page_num, start in enumerate(range(0, len(selected), PAGE_SIZE), 1):
        entries = selected[start : start + PAGE_SIZE]
        canvas = Image.new("RGB", (layout.CANVAS_W, layout.CANVAS_H), "white")
        draw = ImageDraw.Draw(canvas)
        page_counts: Counter[str] = Counter(reference_changes)

        reference_page = layout.fit(
            reference_image,
            int(layout.CELL_W * 0.78),
            int(layout.TOP_BAND_H * 0.56),
        )
        reference_x0 = layout.CANVAS_W - layout.CELL_W
        reference_x = reference_x0 + (layout.CELL_W - reference_page.width) // 2
        reference_y = 6
        canvas.paste(reference_page, (reference_x, reference_y))
        label_y = min(layout.TOP_BAND_H - 105, reference_y + reference_page.height + 4)
        layout.centered(
            draw,
            (reference_x0, label_y, reference_x0 + layout.CELL_W, label_y + 38),
            "C1101 reference",
            ref_name_font,
        )
        layout.centered(
            draw,
            (reference_x0, label_y + 42, reference_x0 + layout.CELL_W, label_y + 78),
            "Docking score = N/A",
            ref_score_font,
        )

        for local_index, item in enumerate(entries):
            grid_row, grid_column = divmod(local_index, layout.GRID_COLS)
            x0 = grid_column * layout.CELL_W
            y0 = layout.TOP_BAND_H + grid_row * layout.CELL_H
            display_mol, changes = layout.prepare_display_mol(item["mol"])
            page_counts.update(changes)
            molecule_image = layout.fit(
                layout.draw_molecule(display_mol),
                int(layout.CELL_W * 0.94),
                int(layout.CELL_H * 0.67),
            )
            image_x = x0 + (layout.CELL_W - molecule_image.width) // 2
            image_y = y0 + 5
            canvas.paste(molecule_image, (image_x, image_y))
            molecule_label_y = min(
                y0 + layout.CELL_H - 90, image_y + molecule_image.height + 4
            )
            layout.centered(
                draw,
                (x0, molecule_label_y, x0 + layout.CELL_W, molecule_label_y + 42),
                item["display_name"],
                name_font,
            )
            layout.centered(
                draw,
                (x0, molecule_label_y + 43, x0 + layout.CELL_W, molecule_label_y + 80),
                f"Docking score = {item['docking_score']:.3f}",
                score_font,
            )
            mapping.append(
                _mapping_row(item, page_num, grid_row + 1, grid_column + 1)
            )

        chemistry_counts.update(page_counts)
        highres_path = highres_dir / f"{slug}_rep1_pbvalid_top50_page_{page_num:02d}_highres.png"
        preview_path = preview_dir / f"{slug}_rep1_pbvalid_top50_page_{page_num:02d}_preview.png"
        canvas.save(highres_path, dpi=(600, 600))
        canvas.resize((3600, 2025), Image.Resampling.LANCZOS).save(
            preview_path, dpi=(300, 300)
        )
        print(f"{model}: rendered page {page_num}/2", flush=True)

    mapping_path = out_dir / f"{slug}_rep1_pbvalid_top50_mapping.csv"
    with mapping_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(mapping[0]))
        writer.writeheader()
        writer.writerows(mapping)
    selected_sdf = out_dir / f"{slug}_rep1_pbvalid_top50.sdf"
    _write_selected_sdf(selected_sdf, selected)
    return {
        "model": model,
        "page_count": 2,
        "mapping_csv": str(mapping_path),
        "selected_sdf": str(selected_sdf),
        "uncertain_double_bonds_cleared": chemistry_counts[
            "uncertain_double_bonds_cleared"
        ],
        "defined_ez_preserved": chemistry_counts["defined_ez_preserved"],
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write homogeneous dictionaries as UTF-8 CSV."""
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run selection, export traceable SDF/CSV files, and render all pages."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out_root = args.out_dir.resolve()
    if out_root.exists() and any(out_root.iterdir()):
        raise FileExistsError(f"Non-empty output already exists: {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)

    reference = _load_reference()
    reference_out = out_root / "C1101_crystal_reference_original_pose.sdf"
    reference_writer = Chem.SDWriter(str(reference_out))
    reference_writer.write(reference)
    reference_writer.close()

    all_selected: list[dict[str, Any]] = []
    stats_rows: list[dict[str, Any]] = []
    render_results: list[dict[str, Any]] = []
    for model, cfg in MODELS.items():
        selected, stats = _select_model(model, cfg)
        all_selected.extend(selected)
        stats_rows.append(stats)
        render_results.append(_render_model(model, selected, reference, out_root))

    selection_rows = [
        {
            "model": item["model"],
            "rank": item["rank"],
            "display_name": item["display_name"],
            "source_index": item["source_index"],
            "source_name": item["name"],
            "docking_score": item["docking_score"],
            "sanitize_valid": True,
            "pb_valid": True,
            "canonical_isomeric_smiles": item["canonical_isomeric_smiles"],
            "source_sdf": item["source_sdf"],
            "posebusters_csv": item["posebusters_csv"],
        }
        for item in all_selected
    ]
    _write_csv(out_root / "selection_summary.csv", stats_rows)
    _write_csv(out_root / "top50_molecule_selection.csv", selection_rows)

    scripts_dir = out_root / "scripts_used"
    scripts_dir.mkdir()
    shutil.copy2(Path(__file__), scripts_dir / Path(__file__).name)
    shutil.copy2(
        BASE / "make_docking_2d_all_pages_v1.py",
        scripts_dir / "make_docking_2d_all_pages_v1.py",
    )
    shutil.copy2(
        BASE / "README_2D脚本复用说明.md",
        scripts_dir / "README_2D脚本复用说明.md",
    )

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "rdkit_version": rdBase.rdkitVersion,
        "selection": "rep1 -> sanitize-valid -> pb_valid=True -> docking score ascending -> top 50 per model",
        "reference": {
            "source": str(REFERENCE_SDF.resolve()),
            "display": "C1101 crystal reference",
            "docking_score": None,
            "note": "No same-protocol redocking score exists in the formal benchmark; N/A is displayed.",
        },
        "stats": stats_rows,
        "render_results": render_results,
    }
    (out_root / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
