#!/usr/bin/env python
"""Top-docking molecule showcase for the two central before/after comparisons.

Figure A : DiffGui RL  epoch 1  vs  epoch 89      (main training run)
Figure B : DiffSBDD   Original vs  E2PO step 4000

Selection rule (stated in the figure caption, no cherry-picking):
    the N molecules with the most negative docking score in each condition.

Every cell is annotated with the docking score, the heavy-atom count (HAC) and
the R1-R4 rules the molecule triggers.  HAC is annotated because ring pathology
is strongly size-correlated (corr(HAC, R1-R4) ~= +0.7 in the DiffGui run);
reporting it keeps the size effect visible instead of hidden.

Three-membered rings are highlighted in red, because that is the substructure
whose enrichment carries the reward-hacking signal.

Outputs
-------
fig_A_diffgui_rl_top_docking.png
fig_B_e2po_top_docking.png
top_docking_showcase.csv        (exact molecules shown, with all annotations)
top_docking_showcase.json       (per-condition statistics for the footer band)
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

RDLogger.DisableLog("rdApp.*")

REPO = Path("/home/md666/data/disk6/DiffB110")
sys.path.insert(0, str(REPO / "scripts" / "offline_rl"))
from structure_pathology_rules import compute_structure_pathology  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
N_TOP = 10

DIFFGUI_RUN = REPO / "runs/diffgui-deppa-full-lr1e5-wd1e4-b8x8-t200-100ep_20260626_165809/epochs"
E2PO_ORIG = (REPO / "analysis/diffsbdd_global_e2po_structure_matched_x_only_generation_1000_260901"
             / "docking_relaxed/docking_results.csv")
E2PO_STEP = (REPO / "OUT/diffsbdd_e2po_mainline_intermediate_gen/step_004000"
             / "docking_relaxed/docking_results.csv")

SCALE = 2                                     # 2x raster: keeps atom labels crisp in print
CELL_W, CELL_H = 230 * SCALE, 250 * SCALE     # 460 x 500
DRAW_H = 185 * SCALE                          # 370
ATOM_FONT = 40                                # RDKit atom-label size, canvas px
ANNOT_FONT = 26                               # annotation text under each cell

RULE_ORDER = ["R1", "R2", "R3", "R4a", "R4b", "R4c", "R5"]


# --------------------------------------------------------------------------- #
# data loading
# --------------------------------------------------------------------------- #
def _facts(smiles: str) -> tuple[dict, Chem.Mol | None]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}, None
    f = compute_structure_pathology(mol)
    flags = {
        "R1": f.r1_large_ring_violation,
        "R2": f.r2_ring_system_violation,
        "R3": f.r3_four_ring_violation,
        "R4a": f.r4a_three_ring_hetero_violation,
        "R4b": f.r4b_three_ring_over_attached_violation,
        "R4c": f.r4c_three_ring_spiro_or_fused_violation,
        "R5": f.r5_carbon_fraction_violation,
    }
    return {
        "flags": flags,
        "hit": [k for k in RULE_ORDER if flags[k]],
        "R1R4": f.is_structurally_pathological,
        "R1R5": f.is_pathological,
        "three": any(len(r) == 3 for r in mol.GetRingInfo().AtomRings()),
        "hac": mol.GetNumHeavyAtoms(),
    }, mol


def load_diffgui(epoch: int) -> pd.DataFrame:
    d = pd.read_csv(DIFFGUI_RUN / f"epoch_{epoch:04d}" / "molecules.csv", low_memory=False)
    d = d[d["canonical_smiles"].apply(
        lambda s: isinstance(s, str) and s.strip() and s.lower() != "nan")]
    rows = []
    for _, r in d.iterrows():
        info, mol = _facts(r["canonical_smiles"])
        if mol is None or pd.isna(r.get("docking_raw")):
            continue
        rows.append({"smiles": r["canonical_smiles"], "dock": float(r["docking_raw"]), **info})
    return pd.DataFrame(rows)


def load_e2po(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, low_memory=False)
    d = d[d["success"].astype(str).str.lower() == "true"]
    rows = []
    for _, r in d.iterrows():
        info, mol = _facts(str(r["input_smiles"]))
        if mol is None:
            continue
        rows.append({"smiles": str(r["input_smiles"]), "dock": float(r["score"]), **info})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# drawing
# --------------------------------------------------------------------------- #
def _font(size: int) -> ImageFont.FreeTypeFont:
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def draw_cell(row: pd.Series, idx: int) -> Image.Image:
    """Render one molecule cell: 2D structure + docking/HAC/rule annotations."""
    mol = Chem.MolFromSmiles(row["smiles"])
    three_atoms = [a for r in mol.GetRingInfo().AtomRings() if len(r) == 3 for a in r]
    drawer = rdMolDraw2D.MolDraw2DCairo(CELL_W, DRAW_H)
    opt = drawer.drawOptions()
    opt.clearBackground = True
    opt.bondLineWidth = 4
    opt.fixedFontSize = ATOM_FONT   # uniform, larger atom letters
    opt.minFontSize = ATOM_FONT
    opt.maxFontSize = ATOM_FONT
    if three_atoms:
        drawer.DrawMolecule(mol, highlightAtoms=three_atoms,
                            highlightAtomColors={a: (1.0, 0.72, 0.72) for a in three_atoms})
    else:
        drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    img = Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGB").resize((CELL_W, DRAW_H))

    canvas = Image.new("RGB", (CELL_W, CELL_H), "white")
    canvas.paste(img, (0, 0))
    d = ImageDraw.Draw(canvas)
    f = _font(ANNOT_FONT)
    y = DRAW_H + 8
    d.text((8, y), f"#{idx}  dock {row['dock']:.2f}", font=f, fill="black")
    d.text((8, y + 30), f"HAC {row['hac']}   {'CLEAN' if not row['hit'] else ''}", font=f,
           fill=(0, 120, 0) if not row["hit"] else "black")
    rules = " ".join(row["hit"]) if row["hit"] else "-"
    d.text((8, y + 60), rules[:30], font=f, fill=(200, 0, 0))
    # 1 px separator
    d.line([(0, CELL_H - 1), (CELL_W, CELL_H - 1)], fill=(200, 200, 200))
    return canvas


def build_figure(conditions: list[tuple[str, pd.DataFrame]], title: str, path: Path,
                 stats: dict) -> None:
    """Compose a 2-row x N_TOP-column grid plus a statistics footer band."""
    header_h, footer_h = 68, 200
    W = CELL_W * N_TOP
    H = header_h + CELL_H * len(conditions) + footer_h
    fig = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(fig)
    d.text((16, 18), title, font=_font(34), fill="black")

    for r, (label, df) in enumerate(conditions):
        y0 = header_h + r * CELL_H
        # row label band
        band = Image.new("RGB", (W, 44), (238, 238, 238))
        fig.paste(band, (0, y0))
        d.text((16, y0 + 8), label, font=_font(28), fill="black")
        for c, (_, row) in enumerate(df.head(N_TOP).iterrows()):
            fig.paste(draw_cell(row, c + 1), (c * CELL_W, y0 + 44))

    # footer statistics
    fy = header_h + CELL_H * len(conditions) + 12
    d.text((16, fy), "condition statistics (top-%d subset)" % N_TOP, font=_font(28), fill="black")
    lines = [
        f"{lab:<26s} n={st['n']:<4d} dock_mean={st['dock']:7.2f}  HAC={st['hac']:5.1f}  "
        f"3ring={st['three']:5.0f}%  R4b={st['R4b']:5.0f}%  R4c={st['R4c']:5.0f}%  "
        f"R1-R4={st['R1R4']:5.0f}%"
        for lab, st in stats.items()
    ]
    for i, ln in enumerate(lines):
        d.text((16, fy + 40 + i * 30), ln, font=_font(24), fill="black")
    fig.save(path)


def summarize(df: pd.DataFrame) -> dict:
    top = df.sort_values("dock").head(N_TOP)
    return {
        "n": int(len(df)),
        "dock": float(top["dock"].mean()),
        "hac": float(top["hac"].mean()),
        "three": float(top["three"].mean() * 100),
        "R4b": float(top["flags"].apply(lambda x: x["R4b"]).mean() * 100),
        "R4c": float(top["flags"].apply(lambda x: x["R4c"]).mean() * 100),
        "R1R4": float(top["R1R4"].mean() * 100),
    }


def main() -> None:
    print("loading DiffGui run ...")
    # sort by docking ascending == most negative first; the figure must show the
    # top-N subset that the footer statistics are computed on.
    ep1 = load_diffgui(1).sort_values("dock").reset_index(drop=True)
    ep89 = load_diffgui(89).sort_values("dock").reset_index(drop=True)
    print("loading E2PO checkpoints ...")
    orig = load_e2po(E2PO_ORIG).sort_values("dock").reset_index(drop=True)
    step = load_e2po(E2PO_STEP).sort_values("dock").reset_index(drop=True)

    a_stats = {"DiffGui ep1": summarize(ep1), "DiffGui ep89": summarize(ep89)}
    b_stats = {"DiffSBDD Original": summarize(orig), "E2PO step4000": summarize(step)}

    build_figure([("DiffGui epoch 1  (reward-RL start)", ep1),
                  ("DiffGui epoch 89 (reward-RL end)", ep89)],
                 "A  DiffGui reward RL, top-%d docking molecules" % N_TOP,
                 OUT_DIR / "fig_A_diffgui_rl_top_docking.png", a_stats)
    build_figure([("DiffSBDD Original  (E2PO start)", orig),
                  ("E2PO step 4000     (recommended stop)", step)],
                 "B  DiffSBDD E2PO, top-%d docking molecules" % N_TOP,
                 OUT_DIR / "fig_B_e2po_top_docking.png", b_stats)

    # exact molecule list for the appendix
    rows = []
    for tag, df in [("diffgui_ep1", ep1), ("diffgui_ep89", ep89),
                    ("e2po_original", orig), ("e2po_step4000", step)]:
        t = df.sort_values("dock").head(N_TOP).reset_index(drop=True)
        for i, r in t.iterrows():
            rows.append({"condition": tag, "rank_by_dock": i + 1, "smiles": r["smiles"],
                         "docking": round(r["dock"], 3), "hac": r["hac"],
                         "rules_hit": " ".join(r["hit"]) if r["hit"] else "-",
                         "R1R4": int(r["R1R4"]), "three_ring": int(r["three"])})
    pd.DataFrame(rows).to_csv(OUT_DIR / "top_docking_showcase.csv", index=False)

    (OUT_DIR / "top_docking_showcase.json").write_text(
        json.dumps({"figure_A": a_stats, "figure_B": b_stats, "n_top": N_TOP},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for tag, st in list(a_stats.items()) + list(b_stats.items()):
        print(f"  {tag:<22s} dock={st['dock']:7.2f} HAC={st['hac']:5.1f} "
              f"3ring={st['three']:5.1f}% R4b={st['R4b']:5.1f}% R4c={st['R4c']:5.1f}% "
              f"R1-R4={st['R1R4']:5.1f}%")
    print("wrote figures + csv + json to", OUT_DIR)


if __name__ == "__main__":
    main()
