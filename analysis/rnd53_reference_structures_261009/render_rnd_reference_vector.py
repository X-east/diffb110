#!/usr/bin/env python3
"""Render a transparent-background vector (SVG) 2D structure for a chosen RND reference.

Default target: RND-01 (PAbetaN / PABN) from the project RND-family EPI reference set.
Output is SVG with no background rect, Arial text when RDKit emits real <text> nodes,
and a transparent PNG preview when the Cairo backend is available.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Draw import rdMolDraw2D

RDLogger.DisableLog("rdApp.*")

REFERENCE_CSV = Path(
    "/home/md666/data/disk6/DiffB110/analysis/benchmark/benchmark5/"
    "rnd53_umap_overlay_20260817/rnd53_coordinates_chembl100k.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rnd-id", default="RND-01", help="RND reference id, e.g. RND-01 or RND-53")
    parser.add_argument("--legend", default=None, help="Caption drawn under the structure")
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=820)
    parser.add_argument("--font-size", type=float, default=42.0)
    parser.add_argument("--bond-width", type=float, default=4.0)
    parser.add_argument("--font-file", type=Path, default=Path("/home/md666/.fonts/Arial.ttf"))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_molecule(rnd_id: str) -> tuple[Chem.Mol, str, str]:
    frame = pd.read_csv(REFERENCE_CSV)
    row = frame[frame["rnd_id"].eq(rnd_id)]
    if len(row) != 1:
        raise ValueError(f"expected exactly one row for {rnd_id}, found {len(row)}")
    record = row.iloc[0]
    smiles = str(record["canonical_smiles"])
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES for {rnd_id}: {smiles}")
    name = str(record["short_name"])
    if name in ("nan", ""):
        name = str(record["preferred_name"])
    return mol, name, smiles


def configure_options(args: argparse.Namespace) -> rdMolDraw2D.MolDrawOptions:
    options = rdMolDraw2D.MolDrawOptions()
    # No white rect: the exported SVG/PNG keeps a fully transparent background.
    options.clearBackground = False
    options.bondLineWidth = args.bond_width
    options.padding = 0.12
    # This RDKit build exposes the font sizes as integers.
    size = int(round(args.font_size))
    options.minFontSize = size
    options.maxFontSize = size
    options.legendFontSize = int(round(args.font_size * 1.15))
    if hasattr(options, "fixedFontSize"):
        options.fixedFontSize = size
    if hasattr(options, "useBoldAtomSymbols"):
        options.useBoldAtomSymbols = True
    if hasattr(options, "annotationFontScale"):
        options.annotationFontScale = 1.0
    if hasattr(options, "fontFile") and args.font_file.exists():
        options.fontFile = str(args.font_file)
    return options


def force_arial(svg_text: str) -> tuple[str, int, int]:
    """Replace generic font families with Arial and report text/path node counts."""
    updated = re.sub(r"(font-family\s*[:=]\s*['\"]?)[^;'\">]+", r"\1Arial", svg_text, flags=re.IGNORECASE)
    updated = updated.replace("font-family='sans-serif'", "font-family='Arial'")
    text_nodes = len(re.findall(r"<text\b", updated))
    path_nodes = len(re.findall(r"<path\b", updated))
    return updated, text_nodes, path_nodes


def render(args: argparse.Namespace) -> dict[str, object]:
    mol, name, smiles = load_molecule(args.rnd_id)
    legend = args.legend if args.legend is not None else name
    stem = f"{args.rnd_id}_{legend}".replace("β", "beta")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = args.output_dir / f"{stem}_2d.svg"

    drawer = rdMolDraw2D.MolDraw2DSVG(args.width, args.height)
    drawer.SetDrawOptions(configure_options(args))
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol, legend=legend)
    drawer.FinishDrawing()
    svg_text, text_nodes, path_nodes = force_arial(drawer.GetDrawingText())
    if "background" in svg_text.lower() and "rect" in svg_text.lower():
        raise RuntimeError("SVG still contains a background rect; transparency check failed")
    svg_path.write_text(svg_text, encoding="utf-8")

    png_path = None
    try:
        cairo = rdMolDraw2D.MolDraw2DCairo(args.width, args.height)
        cairo.SetDrawOptions(configure_options(args))
        rdMolDraw2D.PrepareAndDrawMolecule(cairo, mol, legend=legend)
        cairo.FinishDrawing()
        png_path = args.output_dir / f"{stem}_2d.png"
        png_path.write_bytes(cairo.GetDrawingText())
    except Exception:  # Cairo backend is optional; SVG is the deliverable.
        png_path = None

    return {
        "rnd_id": args.rnd_id,
        "legend": legend,
        "name": name,
        "canonical_smiles": smiles,
        "svg": str(svg_path),
        "png": str(png_path) if png_path else None,
        "svg_bytes": svg_path.stat().st_size,
        "text_nodes": text_nodes,
        "path_nodes": path_nodes,
        "font_size": args.font_size,
    }


def main() -> None:
    result = render(parse_args())
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
