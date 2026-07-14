from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
from collections import Counter
from datetime import datetime
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont
from rdkit import Chem, rdBase
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D


ROOT = Path(r"E:\lishihao\user\lee\杨老师\260713_phoregen_gen_dock\docking_out\deduplicated_best_pose_20260714")
OUT_ROOT = ROOT / "2D_all_pages_v1_20260714"
PAGE_SIZE = 25

SYSTEMS = {
    "8gjl_NMP": {
        "candidate": "8gjl_NMP_experiment_deduplicated_sorted.sdf",
        "reference": "8gjl_NMP_control_redock_best.sdf",
        "reference_name": "NMP reference",
        "expected": 176,
    },
    "8gjl_PAbN": {
        "candidate": "8gjl_PAbN_experiment_deduplicated_sorted.sdf",
        "reference": "8gjl_PAbN_control_redock_best.sdf",
        "reference_name": "PAbN reference",
        "expected": 483,
    },
    "5lq3_CPF": {
        "candidate": "5lq3_CPF_experiment_deduplicated_sorted.sdf",
        "reference": "5lq3_CPF_control_redock_best.sdf",
        "reference_name": "CPF reference",
        "expected": 100,
    },
    "9ksa_covalent": {
        "candidate": "9ksa_covalent_experiment_ligand_deduplicated_sorted.sdf",
        "reference": "9ksa_covalent_control_redock_ligand_best.sdf",
        "reference_name": "CB1 reference",
        "expected": 38,
    },
}

CANVAS_W = 6000
CANVAS_H = 3375
TOP_BAND_H = 410
GRID_COLS = 5
GRID_ROWS = 5
CELL_W = CANVAS_W // GRID_COLS
CELL_H = (CANVAS_H - TOP_BAND_H) // GRID_ROWS


def split_records(text: str) -> list[str]:
    parts = re.split(r"\$\$\$\$\s*(?:\r?\n|$)", text)
    return [p.rstrip("\r\n") + "\n" for p in parts if p.strip()]


def parse_props(raw: str) -> dict[str, str]:
    lines = raw.splitlines()
    props: dict[str, str] = {}
    i = 0
    while i < len(lines):
        m = re.match(r">\s*<([^>]+)>", lines[i])
        if not m:
            i += 1
            continue
        name = m.group(1)
        i += 1
        vals = []
        while i < len(lines) and lines[i].strip() and not lines[i].startswith(">"):
            vals.append(lines[i])
            i += 1
        props[name] = "\n".join(vals).strip()
    return props


def molblock(raw: str) -> str:
    lines = raw.splitlines()
    end = lines.index("M  END")
    return "\n".join(lines[: end + 1]) + "\n"


def mol_from_record(raw: str) -> Chem.Mol:
    block = molblock(raw)
    mol = Chem.MolFromMolBlock(block, sanitize=True, removeHs=False, strictParsing=False)
    if mol is None:
        mol = Chem.MolFromMolBlock(block, sanitize=False, removeHs=False, strictParsing=False)
        if mol is None:
            raise ValueError("RDKit failed to parse mol block")
        Chem.SanitizeMol(mol)
    Chem.AssignStereochemistry(mol, cleanIt=False, force=True)
    return mol


def score_of(props: dict[str, str]) -> float:
    for name in ("Codex_Score", "r_i_docking_score"):
        if name in props:
            return float(props[name])
    raise KeyError("No docking score field")


def prepare_display_mol(source: Chem.Mol) -> tuple[Chem.Mol, dict[str, int]]:
    mol = Chem.Mol(source)
    changes = {"uncertain_double_bonds_cleared": 0, "defined_ez_preserved": 0}
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        uncertain = (
            bond.GetStereo() == Chem.BondStereo.STEREOANY
            or bond.GetBondDir() == Chem.BondDir.EITHERDOUBLE
        )
        if uncertain:
            bond.SetStereo(Chem.BondStereo.STEREONONE)
            bond.SetBondDir(Chem.BondDir.NONE)
            changes["uncertain_double_bonds_cleared"] += 1
        elif bond.GetStereo() in (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ):
            changes["defined_ez_preserved"] += 1

    # Remove only ordinary display hydrogens. Preserve isotopic, hydride,
    # stereo-defining and wedged-bond hydrogens whenever this RDKit version
    # exposes the corresponding controls.
    try:
        params = Chem.RemoveHsParameters()
        for attr, value in {
            "removeIsotopes": False,
            "removeHydrides": False,
            "removeDefiningBondStereo": False,
            "removeWithWedgedBond": False,
            "removeWithQuery": False,
        }.items():
            if hasattr(params, attr):
                setattr(params, attr, value)
        mol = Chem.RemoveHs(mol, params, sanitize=True)
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
    orient_long_axis_horizontal(mol)
    return mol, changes


def orient_long_axis_horizontal(mol: Chem.Mol) -> None:
    if not mol.GetNumConformers():
        return
    conf = mol.GetConformer()
    pts = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        p = conf.GetAtomPosition(atom.GetIdx())
        pts.append((p.x, p.y))
    if len(pts) < 2:
        return
    cx = sum(x for x, _ in pts) / len(pts)
    cy = sum(y for _, y in pts) / len(pts)
    sxx = sum((x - cx) ** 2 for x, _ in pts)
    syy = sum((y - cy) ** 2 for _, y in pts)
    sxy = sum((x - cx) * (y - cy) for x, y in pts)
    theta = 0.5 * math.atan2(2 * sxy, sxx - syy)
    c, s = math.cos(-theta), math.sin(-theta)
    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        x, y = p.x - cx, p.y - cy
        conf.SetAtomPosition(i, (x * c - y * s + cx, x * s + y * c + cy, 0.0))


def crop_nonwhite(img: Image.Image, pad: int = 8) -> Image.Image:
    rgb = img.convert("RGB")
    bbox = ImageChops.difference(rgb, Image.new("RGB", rgb.size, "white")).getbbox()
    if bbox is None:
        return rgb
    return rgb.crop((max(0, bbox[0]-pad), max(0, bbox[1]-pad), min(rgb.width, bbox[2]+pad), min(rgb.height, bbox[3]+pad)))


def draw_molecule(mol: Chem.Mol, width: int = 1800, height: int = 680) -> Image.Image:
    prepared = rdMolDraw2D.PrepareMolForDrawing(mol, kekulize=True, wedgeBonds=True, addChiralHs=False)
    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    opts = drawer.drawOptions()
    opts.padding = 0.003
    opts.bondLineWidth = 4.2
    opts.fixedFontSize = 42
    opts.minFontSize = 34
    opts.maxFontSize = 52
    opts.multipleBondOffset = 0.20
    if hasattr(opts, "additionalAtomLabelPadding"):
        opts.additionalAtomLabelPadding = 0.03
    arial = Path(r"C:\Windows\Fonts\arialbd.ttf")
    if arial.exists() and hasattr(opts, "fontFile"):
        opts.fontFile = str(arial)
    drawer.DrawMolecule(prepared)
    drawer.FinishDrawing()
    return crop_nonwhite(Image.open(BytesIO(drawer.GetDrawingText())), pad=8)


def fit(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    scale = min(max_w / img.width, max_h / img.height)
    return img.resize((max(1, int(img.width*scale)), max(1, int(img.height*scale))), Image.Resampling.LANCZOS)


def get_font(size: int):
    for p in (Path(r"C:\Windows\Fonts\arialbd.ttf"), Path(r"C:\Windows\Fonts\arial.ttf")):
        if p.exists():
            return ImageFont.truetype(str(p), size=size)
    return ImageFont.load_default()


def centered(draw: ImageDraw.ImageDraw, box, text: str, font, fill=(24, 24, 24)):
    x0, y0, x1, y1 = box
    b = draw.textbbox((0, 0), text, font=font)
    x = x0 + (x1-x0-(b[2]-b[0]))/2
    y = y0 + (y1-y0-(b[3]-b[1]))/2
    draw.text((x, y), text, font=font, fill=fill)


def load_records(path: Path) -> list[dict]:
    records = []
    for idx, raw in enumerate(split_records(path.read_text(encoding="utf-8", errors="replace")), 1):
        props = parse_props(raw)
        mol = mol_from_record(raw)
        records.append({
            "raw": raw,
            "props": props,
            "mol": mol,
            "score": score_of(props),
            "record": idx,
            "title": raw.splitlines()[0].strip(),
            "canonical_smiles": Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(mol)), canonical=True, isomericSmiles=True),
        })
    records.sort(key=lambda r: (r["score"], r["record"]))
    return records


def render_system(system: str, cfg: dict) -> dict:
    outdir = OUT_ROOT / system
    outdir.mkdir(parents=True, exist_ok=False)
    highres_dir = outdir / "pages_highres"
    preview_dir = outdir / "pages_preview"
    highres_dir.mkdir()
    preview_dir.mkdir()
    candidates = load_records(ROOT / cfg["candidate"])
    references = load_records(ROOT / cfg["reference"])
    if len(candidates) != cfg["expected"] or len(references) != 1:
        raise ValueError(f"Unexpected counts for {system}: {len(candidates)}, {len(references)}")
    reference = references[0]
    page_count = math.ceil(len(candidates) / PAGE_SIZE)
    name_font = get_font(42)
    score_font = get_font(28)
    ref_name_font = get_font(34)
    ref_score_font = get_font(25)
    total_chemistry_counts = Counter()
    mapping = [{
        "display_name": cfg["reference_name"], "role": "reference", "rank": 0,
        "page": "all", "row": 0, "column": GRID_COLS,
        "docking_score": reference["score"], "source_sdf": cfg["reference"],
        "source_record": reference["record"], "original_title": reference["title"],
        "dedup_key": reference["props"].get("Codex_Dedup_Key", ""),
        "canonical_isomeric_smiles": reference["canonical_smiles"],
    }]
    ref_mol, changes = prepare_display_mol(reference["mol"])
    ref_img_base = draw_molecule(ref_mol, 1500, 520)

    for page_num in range(1, page_count + 1):
        start = (page_num - 1) * PAGE_SIZE
        page = candidates[start : start + PAGE_SIZE]
        page_chemistry_counts = Counter(changes)
        canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), "white")
        draw = ImageDraw.Draw(canvas)

        # First-version layout: compact reference in the upper-right, with the
        # 5x5 candidate grid below it.
        ref_img = fit(ref_img_base, int(CELL_W*0.78), int(TOP_BAND_H*0.56))
        rx0 = CANVAS_W - CELL_W
        rmx = rx0 + (CELL_W-ref_img.width)//2
        rmy = 6
        canvas.paste(ref_img, (rmx, rmy))
        rlabel = min(TOP_BAND_H-105, rmy+ref_img.height+4)
        centered(draw, (rx0, rlabel, rx0+CELL_W, rlabel+38), cfg["reference_name"], ref_name_font)
        centered(draw, (rx0, rlabel+42, rx0+CELL_W, rlabel+78), f"Docking score = {reference['score']:.3f}", ref_score_font)

        for local_i, entry in enumerate(page, 1):
            global_rank = start + local_i
            r, c = divmod(local_i-1, GRID_COLS)
            x0, y0 = c*CELL_W, TOP_BAND_H+r*CELL_H
            display_mol, molecule_changes = prepare_display_mol(entry["mol"])
            page_chemistry_counts.update(molecule_changes)
            mol_img = fit(draw_molecule(display_mol), int(CELL_W*0.94), int(CELL_H*0.67))
            mx, my = x0+(CELL_W-mol_img.width)//2, y0+5
            canvas.paste(mol_img, (mx, my))
            label_y = min(y0+CELL_H-90, my+mol_img.height+4)
            centered(draw, (x0, label_y, x0+CELL_W, label_y+42), f"mol{global_rank:03d}", name_font)
            centered(draw, (x0, label_y+43, x0+CELL_W, label_y+80), f"Docking score = {entry['score']:.3f}", score_font)
            mapping.append({
                "display_name": f"mol{global_rank:03d}", "role": "candidate", "rank": global_rank,
                "page": page_num, "row": r+1, "column": c+1,
                "docking_score": entry["score"], "source_sdf": cfg["candidate"],
                "source_record": entry["record"], "original_title": entry["title"],
                "dedup_key": entry["props"].get("Codex_Dedup_Key", ""),
                "canonical_isomeric_smiles": entry["canonical_smiles"],
            })

        total_chemistry_counts.update(page_chemistry_counts)
        highres_path = highres_dir / f"{system}_page_{page_num:02d}_highres.png"
        preview_path = preview_dir / f"{system}_page_{page_num:02d}_preview.png"
        canvas.save(highres_path, dpi=(600, 600))
        canvas.resize((3600, 2025), Image.Resampling.LANCZOS).save(preview_path, dpi=(300, 300))
        print(f"  {system}: page {page_num}/{page_count} ({len(page)} candidates)", flush=True)

    with (outdir / f"{system}_all_molecules_mapping.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(mapping[0]))
        writer.writeheader()
        writer.writerows(mapping)
    readme = f"""# {system} 全部分页 2D 图

- 候选来源：`{cfg['candidate']}`
- 参考来源：`{cfg['reference']}`
- 候选总数：{len(candidates)}
- 总页数：{page_count}；每页最多 {PAGE_SIZE} 个，按 docking score 升序（越负越好）
- 参考分子：{cfg['reference_name']}，固定右上角
- 所有候选与参考分子均标注 docking score
- 使用用户最终选择的第一版字号、分子大小与间距
- 仅清理不确定双键标记；明确 E/Z 统计保留
- 全部页面清理的不确定双键计数：{total_chemistry_counts['uncertain_double_bonds_cleared']}
- 全部页面保留的明确 E/Z 双键计数：{total_chemistry_counts['defined_ez_preserved']}
- `pages_highres`：6000×3375 高清 PNG
- `pages_preview`：3600×2025 预览 PNG
- 原始 SDF 未修改
"""
    (outdir / "README_全部分页说明.md").write_text(readme, encoding="utf-8")
    return {
        "system": system,
        "candidate_total": len(candidates),
        "page_count": page_count,
        "reference_score": reference["score"],
        "best_candidate_score": candidates[0]["score"],
        "highres_dir": str(highres_dir),
        "preview_dir": str(preview_dir),
        "chemistry_counts": dict(total_chemistry_counts),
    }


def main():
    if OUT_ROOT.exists():
        raise FileExistsError(OUT_ROOT)
    OUT_ROOT.mkdir(parents=False)
    results = []
    for system, cfg in SYSTEMS.items():
        print(f"Rendering {system}...", flush=True)
        results.append(render_system(system, cfg))
    scripts = OUT_ROOT / "scripts_used"
    scripts.mkdir()
    shutil.copy2(Path(__file__), scripts / Path(__file__).name)
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "python": os.sys.executable,
        "rdkit": rdBase.rdkitVersion,
        "layout": "16:9; reference top-right; 5x5 candidate grid; all candidates paginated in groups of 25; first-version sizing",
        "results": results,
    }
    (OUT_ROOT / "preview_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
