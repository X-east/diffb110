#!/usr/bin/env python3
"""Render top structures from screening SDFs and EPI nearest-neighbor SDFs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Draw

RDLogger.DisableLog("rdApp.*")


def render_sdf(sdf_path: Path, output: Path, n: int, title: str) -> int:
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
    molecules = [mol for mol in supplier if mol is not None][:n]
    if not molecules:
        return 0
    legends = []
    for index, mol in enumerate(molecules, 1):
        # Ignore generated 3D conformers and lay out every molecule as a clean 2D depiction.
        mol.RemoveAllConformers()
        AllChem.Compute2DCoords(mol)
        name = mol.GetProp("sample_id") if mol.HasProp("sample_id") else mol.GetProp("nearest_epi_id") if mol.HasProp("nearest_epi_id") else "molecule"
        legends.append(f"#{index}  {name}")
    image = Draw.MolsToGridImage(molecules, molsPerRow=5, subImgSize=(420, 360), legends=legends, useSVG=False)
    image.save(output)
    return len(molecules)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-pools", type=Path, required=True)
    parser.add_argument("--filter-pools", type=Path, required=True)
    parser.add_argument("--epi-pools", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"training_pools": [], "chemistry_filter_pools": [], "epi_pools": []}
    for root, key in ((args.training_pools, "training_pools"), (args.filter_pools, "chemistry_filter_pools")):
        for sdf in sorted(root.glob("**/selected_molecules.sdf")):
            rel = sdf.relative_to(root).parent
            out = args.output_dir / key / (str(rel).replace("/", "__") + "_top10.png")
            out.parent.mkdir(parents=True, exist_ok=True)
            count = render_sdf(sdf, out, 10, str(rel))
            manifest[key].append({"source_sdf": str(sdf), "render": str(out), "rendered_count": count, "display_count": 10})
    for sdf in sorted(args.epi_pools.glob("RND-*_nearest100.sdf")):
        out = args.output_dir / "epi_top5" / (sdf.stem + "_top5.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        count = render_sdf(sdf, out, 5, sdf.stem)
        manifest["epi_pools"].append({"source_sdf": str(sdf), "render": str(out), "rendered_count": count, "display_count": 5})
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
