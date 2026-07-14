"""Export editable-PPT molecule assets from the pre-LigPrep original SDFs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops
from rdkit import Chem, rdBase
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D


BASE = Path(__file__).resolve().parent
SELECTION_ROOT = BASE / "rep1_pbvalid_top50_2d_20260714"
SELECTION_CSV = SELECTION_ROOT / "top50_molecule_selection.csv"
DEFAULT_OUT = BASE / "rep1_pbvalid_top50_editable_ppt_20260714"
ARIAL_FONT = Path("/usr/share/fonts/truetype/msttcorefonts/Arial.ttf")

MODEL_SOURCES = {
    "DiffGui": {
        "slug": "diffgui",
        "sdf": BASE
        / "8gjl_c1101_benchmark_20260701_diffgui_rep1_1000steps_seed1_7"
        / "filtered_sanitize_valid"
        / "diffgui.sanitize_valid.sdf",
    },
    "DiffSBDD": {
        "slug": "diffsbdd",
        "sdf": BASE
        / "8gjl_c1101_benchmark_20260630_all_1000steps"
        / "filtered_sanitize_valid"
        / "diffsbdd.sanitize_valid.sdf",
    },
}


def _canonical_smiles(mol: Chem.Mol) -> str:
    """Return canonical isomeric SMILES without display hydrogens."""
    return Chem.MolToSmiles(
        Chem.RemoveHs(Chem.Mol(mol)), canonical=True, isomericSmiles=True
    )


def _orient_long_axis_horizontal(mol: Chem.Mol) -> None:
    """Rotate 2D coordinates so the principal molecular axis is horizontal."""
    if not mol.GetNumConformers():
        return
    conf = mol.GetConformer()
    points: list[tuple[float, float]] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        point = conf.GetAtomPosition(atom.GetIdx())
        points.append((point.x, point.y))
    if len(points) < 2:
        return
    center_x = sum(x for x, _ in points) / len(points)
    center_y = sum(y for _, y in points) / len(points)
    sxx = sum((x - center_x) ** 2 for x, _ in points)
    syy = sum((y - center_y) ** 2 for _, y in points)
    sxy = sum((x - center_x) * (y - center_y) for x, y in points)
    theta = 0.5 * math.atan2(2 * sxy, sxx - syy)
    cosine = math.cos(-theta)
    sine = math.sin(-theta)
    for atom_index in range(mol.GetNumAtoms()):
        point = conf.GetAtomPosition(atom_index)
        x_coord = point.x - center_x
        y_coord = point.y - center_y
        conf.SetAtomPosition(
            atom_index,
            (
                x_coord * cosine - y_coord * sine + center_x,
                x_coord * sine + y_coord * cosine + center_y,
                0.0,
            ),
        )


def _prepare_display_mol(source: Chem.Mol) -> Chem.Mol:
    """Create a 2D depiction from an original pre-LigPrep molecule."""
    mol = Chem.Mol(source)
    Chem.SanitizeMol(mol)
    Chem.AssignStereochemistry(mol, cleanIt=False, force=True)
    try:
        parameters = Chem.RemoveHsParameters()
        for attribute, value in {
            "removeIsotopes": False,
            "removeHydrides": False,
            "removeDefiningBondStereo": False,
            "removeWithWedgedBond": False,
            "removeWithQuery": False,
        }.items():
            if hasattr(parameters, attribute):
                setattr(parameters, attribute, value)
        mol = Chem.RemoveHs(mol, parameters, sanitize=True)
    except Exception:
        mol = Chem.RemoveHs(mol, sanitize=True)
    try:
        rdDepictor.Compute2DCoords(
            mol,
            canonOrient=True,
            clearConfs=True,
            nFlipsPerSample=2,
            nSample=100,
            sampleSeed=0,
        )
    except TypeError:
        rdDepictor.Compute2DCoords(mol, canonOrient=True, clearConfs=True)
    if hasattr(rdDepictor, "StraightenDepiction"):
        try:
            rdDepictor.StraightenDepiction(mol, minimizeRotation=True)
        except TypeError:
            rdDepictor.StraightenDepiction(mol)
    _orient_long_axis_horizontal(mol)
    return mol


def _crop_white(image: Image.Image, padding: int = 16) -> Image.Image:
    """Crop unused white background while retaining a small visual margin."""
    rgb = image.convert("RGB")
    difference = ImageChops.difference(rgb, Image.new("RGB", rgb.size, "white"))
    bounding_box = difference.getbbox()
    if bounding_box is None:
        return rgb
    left, top, right, bottom = bounding_box
    return rgb.crop(
        (
            max(0, left - padding),
            max(0, top - padding),
            min(rgb.width, right + padding),
            min(rgb.height, bottom + padding),
        )
    )


def _draw_molecule(mol: Chem.Mol) -> Image.Image:
    """Render one molecule at high resolution using Arial atom labels."""
    prepared = rdMolDraw2D.PrepareMolForDrawing(
        mol, kekulize=True, wedgeBonds=True, addChiralHs=False
    )
    drawer = rdMolDraw2D.MolDraw2DCairo(1600, 760)
    options = drawer.drawOptions()
    options.padding = 0.012
    options.bondLineWidth = 5.0
    options.fixedFontSize = 48
    options.minFontSize = 38
    options.maxFontSize = 58
    options.multipleBondOffset = 0.20
    if hasattr(options, "additionalAtomLabelPadding"):
        options.additionalAtomLabelPadding = 0.03
    if ARIAL_FONT.exists() and hasattr(options, "fontFile"):
        options.fontFile = str(ARIAL_FONT)
    drawer.DrawMolecule(prepared)
    drawer.FinishDrawing()
    return _crop_white(Image.open(BytesIO(drawer.GetDrawingText())), padding=16)


def _read_selection() -> dict[str, list[dict[str, Any]]]:
    """Read the approved top-50 ranks and docking scores."""
    with SELECTION_CSV.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected: dict[str, list[dict[str, Any]]] = {}
    for model in MODEL_SOURCES:
        model_rows = [row for row in rows if row["model"] == model]
        model_rows.sort(key=lambda row: int(row["rank"]))
        if len(model_rows) != 50:
            raise ValueError(f"{model}: expected 50 selected rows, found {len(model_rows)}")
        selected[model] = model_rows
    return selected


def _write_original_sdf(
    path: Path, selected: list[dict[str, Any]], source_molecules: list[Chem.Mol | None]
) -> None:
    """Write selected original molecules with selection metadata."""
    writer = Chem.SDWriter(str(path))
    try:
        for row in selected:
            source_index = int(row["source_index"])
            source = source_molecules[source_index]
            if source is None:
                raise ValueError(f"Unreadable source molecule at index {source_index}")
            mol = Chem.Mol(source)
            mol.SetProp("_Name", f"{row['model']}_original_top{int(row['rank']):02d}")
            mol.SetProp("SelectionModel", row["model"])
            mol.SetIntProp("SelectionRank", int(row["rank"]))
            mol.SetIntProp("OriginalSourceIndex", source_index)
            mol.SetDoubleProp("DockingScore", float(row["docking_score"]))
            mol.SetBoolProp("PBValid", True)
            writer.write(mol)
    finally:
        writer.close()


def main() -> None:
    """Export original structures, high-resolution PNGs, and provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out_root = args.out_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    assets_root = out_root / "assets"
    assets_root.mkdir(exist_ok=True)

    selected_by_model = _read_selection()
    mapping_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for model, cfg in MODEL_SOURCES.items():
        slug = str(cfg["slug"])
        source_path = Path(cfg["sdf"])
        source_molecules = list(Chem.SDMolSupplier(str(source_path), removeHs=False))
        model_assets = assets_root / slug
        model_assets.mkdir(exist_ok=True)
        selected = selected_by_model[model]
        for row in selected:
            rank = int(row["rank"])
            source_index = int(row["source_index"])
            source = source_molecules[source_index]
            if source is None:
                raise ValueError(f"{model}: unreadable original molecule {source_index}")
            Chem.SanitizeMol(source)
            original_smiles = _canonical_smiles(source)
            if original_smiles != row["canonical_isomeric_smiles"]:
                raise ValueError(
                    f"{model}: original/docked identity mismatch at rank {rank}"
                )
            display_mol = _prepare_display_mol(source)
            image_path = model_assets / f"mol{rank:03d}.png"
            image = _draw_molecule(display_mol)
            image.save(image_path, dpi=(300, 300), optimize=True)
            mapping_rows.append(
                {
                    "model": model,
                    "rank": rank,
                    "display_name": f"mol{rank:03d}",
                    "docking_score": float(row["docking_score"]),
                    "original_source_index": source_index,
                    "original_source_sdf": str(source_path.resolve()),
                    "original_canonical_isomeric_smiles": original_smiles,
                    "asset_png": str(image_path.resolve()),
                    "asset_width_px": image.width,
                    "asset_height_px": image.height,
                }
            )
        original_sdf = out_root / f"{slug}_rep1_pbvalid_top50_original_pre_ligprep.sdf"
        _write_original_sdf(original_sdf, selected, source_molecules)
        summary_rows.append(
            {
                "model": model,
                "source_records": len(source_molecules),
                "selected_records": len(selected),
                "asset_count": len(list(model_assets.glob("mol*.png"))),
                "original_source_sdf": str(source_path.resolve()),
                "selected_original_sdf": str(original_sdf.resolve()),
            }
        )

    mapping_path = out_root / "original_structure_asset_mapping.csv"
    with mapping_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(mapping_rows[0]))
        writer.writeheader()
        writer.writerows(mapping_rows)
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "rdkit_version": rdBase.rdkitVersion,
        "selection_csv": str(SELECTION_CSV.resolve()),
        "identity_check": "100/100 original canonical isomeric SMILES matched",
        "font": "Arial for atom labels",
        "models": summary_rows,
    }
    (out_root / "asset_export_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
