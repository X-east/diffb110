#!/usr/bin/env python3
"""Replay and visualize deterministic aromatic attachment construction."""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ddpo.constraints.dual_sidechain.audit import extract_sidechain_branch


OUT_DIR = Path(__file__).resolve().parent
BASE_DIR = ROOT / "analysis/dual_sidechain_diffsbdd_onecopy_20260716"
RAW_SDF = BASE_DIR / "inpaint_raw.sdf"
OLD_SDF = BASE_DIR / "symmetric_candidates.sdf"
OLD_SUMMARY = BASE_DIR / "generation_summary.csv"
NEW_DIR = OUT_DIR / "postprocess_replay"
WORK_RECORD_DIR = ROOT / "work_records/26-7/260718"
LOGGER = logging.getLogger("planar_attachment_validation")


def _setup_logging() -> None:
    """Configure durable file and terminal logging."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    file_handler = logging.FileHandler(OUT_DIR / "validation.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(stream_handler)


def _run_replay() -> None:
    """Run the postprocessor on the original 12-molecule SDF."""
    command = [
        sys.executable,
        str(ROOT / "scripts/generate_dual_sidechain_diffsbdd.py"),
        "--out_dir",
        str(NEW_DIR),
        "--inpaint_sdf",
        str(RAW_SDF),
    ]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
    LOGGER.info("postprocess command=%s", " ".join(command))
    for line in result.stdout.splitlines():
        LOGGER.info("postprocess: %s", line)


def _load_named_molecule(path: Path, name: str) -> Chem.Mol:
    """Load one named molecule from an SDF."""
    for mol in Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False):
        if mol is not None and mol.HasProp("_Name") and mol.GetProp("_Name") == name:
            Chem.SanitizeMol(mol)
            return mol
    raise RuntimeError(f"molecule {name!r} not found in {path}")


def _array(conf: Chem.Conformer, atom_idx: int) -> np.ndarray:
    """Return one coordinate as a NumPy vector."""
    point = conf.GetAtomPosition(atom_idx)
    return np.asarray([point.x, point.y, point.z], dtype=np.float64)


def _angle(left: np.ndarray, right: np.ndarray) -> float:
    """Return an unsigned vector angle in degrees."""
    cosine = np.dot(left, right) / np.linalg.norm(left) / np.linalg.norm(right)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _geometry(mol: Chem.Mol, anchor_idx: int) -> dict[str, float]:
    """Measure the aromatic attachment geometry at one anchor."""
    branch = extract_sidechain_branch(mol, core_atom_idx=anchor_idx, label="R2")
    if not branch.extractable or branch.root_atom_idx is None:
        raise RuntimeError("R2 branch is not extractable")
    conf = mol.GetConformer()
    anchor = _array(conf, anchor_idx)
    root = _array(conf, branch.root_atom_idx)
    branch_set = set(branch.atom_indices)
    neighbors = [
        neighbor.GetIdx()
        for neighbor in mol.GetAtomWithIdx(anchor_idx).GetNeighbors()
        if neighbor.GetIdx() not in branch_set and neighbor.GetIsAromatic()
    ]
    if len(neighbors) != 2:
        raise RuntimeError(f"expected two aromatic neighbors, found {neighbors}")
    root_vector = root - anchor
    first = _array(conf, neighbors[0]) - anchor
    second = _array(conf, neighbors[1]) - anchor
    normal = np.cross(first, second)
    normal /= np.linalg.norm(normal)
    root_unit = root_vector / np.linalg.norm(root_vector)
    return {
        "bond_length": float(np.linalg.norm(root_vector)),
        "angle_1_deg": _angle(root_vector, first),
        "angle_2_deg": _angle(root_vector, second),
        "out_of_plane_deg": float(
            np.degrees(np.arcsin(np.clip(abs(np.dot(root_unit, normal)), 0.0, 1.0)))
        ),
    }


def _write_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write before/after geometry and replay status counts."""
    old_mol = _load_named_molecule(OLD_SDF, "dual_sidechain_9")
    new_mol = _load_named_molecule(NEW_DIR / "symmetric_candidates.sdf", "dual_sidechain_9")
    geometry = pd.DataFrame(
        [
            {"method": "old_translation", **_geometry(old_mol, 11)},
            {"method": "new_planar_rigid", **_geometry(new_mol, 11)},
        ]
    )
    geometry.to_csv(OUT_DIR / "candidate9_attachment_geometry.csv", index=False)

    old_rows = pd.read_csv(OLD_SUMMARY)
    new_rows = pd.read_csv(NEW_DIR / "generation_summary.csv")
    all_statuses = sorted(set(old_rows["status"]) | set(new_rows["status"]))
    old_counts = Counter(old_rows["status"])
    new_counts = Counter(new_rows["status"])
    statuses = pd.DataFrame(
        [
            {
                "status": status,
                "old_count": old_counts.get(status, 0),
                "new_count": new_counts.get(status, 0),
            }
            for status in all_statuses
        ]
    )
    statuses.to_csv(OUT_DIR / "real12_status_comparison.csv", index=False)
    return geometry, statuses


def _plot(geometry: pd.DataFrame, statuses: pd.DataFrame) -> None:
    """Plot candidate-9 geometry and 12-input status changes."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    labels = ["Angle 1", "Angle 2", "Out of plane"]
    columns = ["angle_1_deg", "angle_2_deg", "out_of_plane_deg"]
    x = np.arange(len(labels))
    width = 0.34
    for offset, (_, row) in enumerate(geometry.iterrows()):
        axes[0].bar(
            x + (offset - 0.5) * width,
            [row[column] for column in columns],
            width,
            label=row["method"],
        )
    axes[0].axhline(120.0, color="black", linestyle="--", linewidth=1.0, label="ideal 120 deg")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Degrees")
    axes[0].set_title("Candidate 9 R2 attachment geometry")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis="y", alpha=0.25)

    positions = np.arange(len(statuses))
    axes[1].bar(positions - width / 2, statuses["old_count"], width, label="old translation")
    axes[1].bar(positions + width / 2, statuses["new_count"], width, label="new planar rigid")
    axes[1].set_xticks(positions, statuses["status"], rotation=30, ha="right")
    axes[1].set_ylabel("Molecules (n=12)")
    axes[1].set_title("Real-input postprocess outcomes")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    png_path = OUT_DIR / "planar_attachment_validation.png"
    svg_path = OUT_DIR / "planar_attachment_validation.svg"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    WORK_RECORD_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(png_path, WORK_RECORD_DIR / png_path.name)
    LOGGER.info("wrote plot=%s", png_path)


def main() -> None:
    """Run replay, tables, plots, and validation logging."""
    _setup_logging()
    _run_replay()
    geometry, statuses = _write_tables()
    _plot(geometry, statuses)
    LOGGER.info("candidate9 geometry=%s", geometry.to_dict(orient="records"))
    LOGGER.info("status comparison=%s", statuses.to_dict(orient="records"))


if __name__ == "__main__":
    main()
