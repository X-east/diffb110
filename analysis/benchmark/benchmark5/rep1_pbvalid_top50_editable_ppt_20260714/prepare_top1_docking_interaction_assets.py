"""Prepare PyMOL docking views and Schrodinger 2D interaction diagrams."""

from __future__ import annotations

import csv
import gzip
import json
import os
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageChops
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
OUT = ROOT / "top1_docking_interactions_20260716"
RECEPTOR_PDBQT = REPO / "data" / "Recmeb_receptor.pdbqt"
SCHRODINGER_LID = Path(
    os.environ.get(
        "SCHRODINGER_LID",
        "/opt/schrodinger2021-2/utilities/ligand_interaction_diagram",
    )
)
SCHRODINGER_STRUCTCAT = Path(
    os.environ.get("SCHRODINGER_STRUCTCAT", "/opt/schrodinger2021-2/utilities/structcat")
)
PYMOL = os.environ.get("PYMOL", "pymol")
ARIAL_FONT = Path("/usr/share/fonts/truetype/msttcorefonts/Arial.ttf")

PABN_POSE = (
    REPO
    / "bishe"
    / "RE-out"
    / "unidock_ligprep_duizhao_batch"
    / "recmeb_unidock_vina_balance"
    / "batch_outputs"
    / "batch_0001"
    / "ligand_000008_out.sdf"
)

TOP50_SOURCES = {
    "DiffGui": ROOT
    / ".."
    / "rep1_pbvalid_top50_2d_20260714"
    / "diffgui"
    / "diffgui_rep1_pbvalid_top50.sdf",
    "DiffSBDD": ROOT
    / ".."
    / "rep1_pbvalid_top50_2d_20260714"
    / "diffsbdd"
    / "diffsbdd_rep1_pbvalid_top50.sdf",
}

MODEL_CONFIG = {
    "DiffGui": {
        "slug": "diffgui",
        "source_name": "diffgui_000762",
        "score": -15.466,
        "pose": ROOT
        / ".."
        / "8gjl_c1101_benchmark_20260706_1000steps_3rep_formal"
        / "docking"
        / "rep1"
        / "diffgui"
        / "batch_outputs"
        / "batch_0006"
        / "diffgui_000762_out.sdf",
    },
    "DiffSBDD": {
        "slug": "diffsbdd",
        "source_name": "diffsbdd_000207",
        "score": -13.850,
        "pose": ROOT
        / ".."
        / "8gjl_c1101_benchmark_20260706_1000steps_3rep_formal"
        / "docking"
        / "rep1"
        / "diffsbdd"
        / "batch_outputs"
        / "batch_0002"
        / "diffsbdd_000207_out.sdf",
    },
}


def _first_molecule(path: Path) -> Chem.Mol:
    """Read the first valid molecule from an SDF."""
    for mol in Chem.SDMolSupplier(str(path), removeHs=False):
        if mol is not None:
            return mol
    raise ValueError(f"No readable molecule in {path}")


def _score_from_pose(mol: Chem.Mol) -> float:
    """Read the Uni-Dock score from a docked molecule."""
    tokens = mol.GetProp("Uni-Dock RESULT").splitlines()[0].replace("=", " ").split()
    for token in tokens:
        try:
            return float(token)
        except ValueError:
            continue
    raise ValueError("Uni-Dock RESULT does not contain a numeric energy")


def _protein_pdb_lines(path: Path) -> list[str]:
    """Convert PDBQT atom records to standard PDB records."""
    records: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        element = line.split()[-1]
        if len(element) > 2 or not element[0].isalpha():
            element = line[12:14].strip()[0]
        records.append(f"{line[:66]:<76}{element:>2}")
    if not records:
        raise ValueError(f"No receptor atoms in {path}")
    return records


def _ligand_pdb_lines(
    mol: Chem.Mol, serial_offset: int, residue_number: int = 999
) -> list[str]:
    """Write ligand atoms and bond-order CONECT records with unique serials."""
    with_hydrogens = Chem.AddHs(Chem.Mol(mol), addCoords=True)
    lines: list[str] = []
    for line in Chem.MolToPDBBlock(with_hydrogens).splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            serial = int(line[6:11]) + serial_offset
            line = f"HETATM{serial:5d}{line[11:]}"
            line = f"{line[:17]}LIG X{residue_number:4d}{line[26:]}"
            lines.append(line)
            continue
        if not line.startswith("CONECT"):
            continue
        serials = [int(line[index : index + 5]) for index in range(6, len(line), 5)]
        lines.append("CONECT" + "".join(f"{value + serial_offset:5d}" for value in serials))
    return lines


def _write_complex(path: Path, protein: list[str], ligand: Chem.Mol) -> None:
    """Write one receptor-ligand PDB complex for interaction analysis."""
    maximum_serial = max(int(line[6:11]) for line in protein)
    payload = [
        *protein,
        "TER",
        *_ligand_pdb_lines(ligand, serial_offset=maximum_serial),
        "END",
    ]
    path.write_text("\n".join(payload) + "\n", encoding="ascii")


def _crop_white(image: Image.Image, padding: int = 18) -> Image.Image:
    """Crop white margins around a molecule depiction."""
    rgb = image.convert("RGB")
    diff = ImageChops.difference(rgb, Image.new("RGB", rgb.size, "white"))
    box = diff.getbbox()
    if box is None:
        return rgb
    left, top, right, bottom = box
    return rgb.crop(
        (
            max(0, left - padding),
            max(0, top - padding),
            min(rgb.width, right + padding),
            min(rgb.height, bottom + padding),
        )
    )


def _draw_2d(mol: Chem.Mol, path: Path) -> None:
    """Draw a compact 2D structure with Arial atom labels."""
    display = Chem.RemoveHs(Chem.Mol(mol), sanitize=True)
    rdDepictor.Compute2DCoords(display, canonOrient=True, clearConfs=True)
    drawer = rdMolDraw2D.MolDraw2DCairo(1000, 650)
    options = drawer.drawOptions()
    options.padding = 0.025
    options.bondLineWidth = 3.0
    options.fixedFontSize = 34
    if ARIAL_FONT.exists() and hasattr(options, "fontFile"):
        options.fontFile = str(ARIAL_FONT)
    drawer.DrawMolecule(display)
    drawer.FinishDrawing()
    image = Image.open(__import__("io").BytesIO(drawer.GetDrawingText()))
    _crop_white(image).save(path, dpi=(300, 300), optimize=True)


def _write_pymol_script(
    path: Path,
    receptor_pdb: Path,
    pabn_sdf: Path,
    candidate_sdf: Path,
    png_path: Path,
    session_path: Path,
) -> None:
    """Write a deterministic headless PyMOL rendering script."""
    commands = f"""
reinitialize
load {receptor_pdb}, receptor
load {pabn_sdf}, pabn
load {candidate_sdf}, candidate
hide everything
show cartoon, receptor
color palegreen, receptor
set cartoon_transparency, 0.68, receptor
show sticks, pabn or candidate
color cyan, pabn and elem C
color yellow, candidate and elem C
color red, (pabn or candidate) and elem O
color blue, (pabn or candidate) and elem N
color tv_green, (pabn or candidate) and elem F
color white, (pabn or candidate) and elem H
set stick_radius, 0.22, pabn or candidate
set stick_ball, on
set stick_ball_ratio, 1.35
set sphere_scale, 0.24, pabn or candidate
set ray_opaque_background, off
set antialias, 2
set depth_cue, 0
set specular, 0.30
set ambient, 0.52
set direct, 0.55
bg_color white
orient pabn or candidate
turn z, 32
turn x, -16
turn y, 12
zoom pabn or candidate, 3.5
save {session_path}
ray 1800, 1350
png {png_path}, dpi=300
quit
""".strip()
    path.write_text(commands + "\n", encoding="ascii")


def _write_master_pymol_script(
    path: Path,
    receptor_pdb: Path,
    pabn_sdf: Path,
    diffgui_sdf: Path,
    diffsbdd_sdf: Path,
    session_path: Path,
) -> None:
    """Write a PyMOL session containing receptor, reference, and both Top1 ligands."""
    commands = f"""
reinitialize
load {receptor_pdb}, receptor
load {pabn_sdf}, pabn
load {diffgui_sdf}, diffgui_top1
load {diffsbdd_sdf}, diffsbdd_top1
hide everything
show cartoon, receptor
color palegreen, receptor
set cartoon_transparency, 0.68, receptor
show sticks, pabn or diffgui_top1 or diffsbdd_top1
color cyan, pabn and elem C
color yellow, diffgui_top1 and elem C
color magenta, diffsbdd_top1 and elem C
color red, (pabn or diffgui_top1 or diffsbdd_top1) and elem O
color blue, (pabn or diffgui_top1 or diffsbdd_top1) and elem N
color tv_green, (pabn or diffgui_top1 or diffsbdd_top1) and elem F
color white, (pabn or diffgui_top1 or diffsbdd_top1) and elem H
set stick_radius, 0.22, pabn or diffgui_top1 or diffsbdd_top1
set stick_ball, on
set stick_ball_ratio, 1.35
set ray_opaque_background, off
set antialias, 2
set depth_cue, 0
bg_color white
orient pabn or diffgui_top1 or diffsbdd_top1
turn z, 32
turn x, -16
turn y, 12
zoom pabn or diffgui_top1 or diffsbdd_top1, 3.5
save {session_path}
quit
""".strip()
    path.write_text(commands + "\n", encoding="ascii")


def _run(command: list[str], log_path: Path, env: dict[str, str] | None = None) -> None:
    """Run one external visualization command with a durable log."""
    result = subprocess.run(
        command,
        cwd=OUT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(result.stdout, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(command)}")


def main() -> None:
    """Generate source poses, complexes, PyMOL views, and interaction diagrams."""
    OUT.mkdir(parents=True, exist_ok=True)
    protein = _protein_pdb_lines(RECEPTOR_PDBQT)
    receptor_pdb = OUT / "8GJL_Recmeb_docking_receptor.pdb"
    receptor_pdb.write_text("\n".join([*protein, "END"]) + "\n", encoding="ascii")
    shutil.copy2(RECEPTOR_PDBQT, OUT / RECEPTOR_PDBQT.name)

    pabn = _first_molecule(PABN_POSE)
    pabn_score = _score_from_pose(pabn)
    pabn_copy = OUT / "8GJL_PABN_ligand_000008_docked.sdf"
    shutil.copy2(PABN_POSE, pabn_copy)
    _draw_2d(pabn, OUT / "PABN_2d.png")
    pabn_complex = OUT / "8GJL_PABN_complex.pdb"
    _write_complex(pabn_complex, protein, pabn)
    pabn_interaction = OUT / "8GJL_PABN_2d_interaction.png"
    runtime_dir = OUT / "qt_runtime_pabn"
    runtime_dir.mkdir(exist_ok=True)
    runtime_dir.chmod(0o700)
    pabn_env = os.environ.copy()
    pabn_env.update(
        {"QT_QPA_PLATFORM": "offscreen", "XDG_RUNTIME_DIR": str(runtime_dir)}
    )
    _run(
        [
            str(SCHRODINGER_LID),
            "-i",
            str(pabn_complex),
            "-o",
            str(pabn_interaction),
            "-ligandASL",
            "res.ptype LIG",
            "-showLegend",
        ],
        OUT / "pabn_schrodinger_2d_interaction.log",
        env=pabn_env,
    )

    rows: list[dict[str, object]] = []
    candidate_copies: dict[str, Path] = {}
    for model, config in MODEL_CONFIG.items():
        slug = str(config["slug"])
        source_pose = Path(config["pose"]).resolve()
        candidate = _first_molecule(source_pose)
        actual_score = _score_from_pose(candidate)
        expected_score = float(config["score"])
        if abs(actual_score - expected_score) > 1e-3:
            raise ValueError(f"{model}: score mismatch {actual_score} != {expected_score}")

        candidate_copy = OUT / f"8GJL_{model}_mol001_docked.sdf"
        shutil.copy2(source_pose, candidate_copy)
        candidate_copies[model] = candidate_copy
        _draw_2d(candidate, OUT / f"{model}_mol001_2d.png")
        complex_pdb = OUT / f"8GJL_{model}_mol001_complex.pdb"
        _write_complex(complex_pdb, protein, candidate)

        interaction_prefix = OUT / f"8GJL_{model}_mol001_2d_interaction.png"
        runtime_dir = OUT / f"qt_runtime_{slug}"
        runtime_dir.mkdir(exist_ok=True)
        runtime_dir.chmod(0o700)
        env = os.environ.copy()
        env.update({"QT_QPA_PLATFORM": "offscreen", "XDG_RUNTIME_DIR": str(runtime_dir)})
        _run(
            [
                str(SCHRODINGER_LID),
                "-i",
                str(complex_pdb),
                "-o",
                str(interaction_prefix),
                "-ligandASL",
                "res.ptype LIG",
                "-showLegend",
            ],
            OUT / f"{slug}_schrodinger_2d_interaction.log",
            env=env,
        )

        pymol_script = OUT / f"{slug}_pymol_docking_overlay.pml"
        pymol_png = OUT / f"8GJL_PABN_{model}_mol001_pymol.png"
        pymol_session = OUT / f"8GJL_PABN_{model}_mol001.pse"
        _write_pymol_script(
            pymol_script,
            receptor_pdb,
            pabn_copy,
            candidate_copy,
            pymol_png,
            pymol_session,
        )
        _run(
            [PYMOL, "-cq", str(pymol_script)],
            OUT / f"{slug}_pymol.log",
        )
        rows.append(
            {
                "model": model,
                "display_name": "mol001",
                "source_name": config["source_name"],
                "pb_valid": True,
                "docking_score": actual_score,
                "pabn_docking_score": pabn_score,
                "pabn_complex_pdb": str(pabn_complex.resolve()),
                "pabn_interaction_png": str(pabn_interaction.resolve()),
                "candidate_pose": str(candidate_copy.resolve()),
                "complex_pdb": str(complex_pdb.resolve()),
                "pymol_png": str(pymol_png.resolve()),
                "pymol_session": str(pymol_session.resolve()),
                "interaction_png": str(interaction_prefix.resolve()),
            }
        )

    master_pymol_script = OUT / "8GJL_PABN_DiffGui_DiffSBDD_top1_master.pml"
    master_pymol_session = OUT / "8GJL_PABN_DiffGui_DiffSBDD_top1_master.pse"
    _write_master_pymol_script(
        master_pymol_script,
        receptor_pdb,
        pabn_copy,
        candidate_copies["DiffGui"],
        candidate_copies["DiffSBDD"],
        master_pymol_session,
    )
    _run(
        [PYMOL, "-cq", str(master_pymol_script)],
        OUT / "master_pymol.log",
    )

    with (OUT / "top1_interaction_asset_mapping.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    top50_copies: list[Path] = []
    for model, source_path in TOP50_SOURCES.items():
        molecules = [
            mol
            for mol in Chem.SDMolSupplier(str(source_path.resolve()), removeHs=False)
            if mol is not None
        ]
        if len(molecules) != 50:
            raise ValueError(f"{model}: expected 50 docked Top50 molecules, found {len(molecules)}")
        for expected_rank, mol in enumerate(molecules, start=1):
            if mol.GetIntProp("SelectionRank") != expected_rank:
                raise ValueError(f"{model}: rank order mismatch at {expected_rank}")
            if not mol.GetBoolProp("PBValid"):
                raise ValueError(f"{model}: non-PB-valid molecule at rank {expected_rank}")
        destination = OUT / f"8GJL_{model}_PBvalid_Top50_docked.sdf"
        shutil.copy2(source_path.resolve(), destination)
        top50_copies.append(destination)

    maestro_path = OUT / "8GJL_DiffGui_DiffSBDD_PBvalid_Top50_poseviewer.maegz"
    _run(
        [
            str(SCHRODINGER_STRUCTCAT),
            "-ipdb",
            str(receptor_pdb),
            "-isd",
            str(top50_copies[0]),
            "-isd",
            str(top50_copies[1]),
            "-omae",
            str(maestro_path),
        ],
        OUT / "schrodinger_top50_poseviewer.log",
    )
    maestro_uncompressed = OUT / "8GJL_DiffGui_DiffSBDD_PBvalid_Top50_poseviewer.mae"
    with gzip.open(maestro_path, "rb") as source, maestro_uncompressed.open("wb") as target:
        shutil.copyfileobj(source, target)
    summary = {
        "receptor": str(RECEPTOR_PDBQT.resolve()),
        "reference_ligand": "PAβN ligand_000008",
        "reference_score": pabn_score,
        "pymol": shutil.which(PYMOL),
        "schrodinger_ligand_interaction_diagram": str(SCHRODINGER_LID),
        "schrodinger_poseviewer": str(maestro_path.resolve()),
        "schrodinger_poseviewer_uncompressed": str(maestro_uncompressed.resolve()),
        "schrodinger_poseviewer_entries": 101,
        "master_pymol_session": str(master_pymol_session.resolve()),
        "master_pymol_objects": [
            "receptor",
            "pabn",
            "diffgui_top1",
            "diffsbdd_top1",
        ],
        "models": rows,
    }
    (OUT / "asset_generation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
