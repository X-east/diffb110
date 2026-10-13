#!/usr/bin/env python
"""UniDock ligand-preparation A/B/C experiment (2026-09-30).

Question
--------
AutoDock Vina (and therefore Uni-Dock) needs *polar* hydrogens in the ligand
input to type heavy atoms as hydrogen-bond donors
(https://vina.scripps.edu/manual/ : "polar hydrogens are needed in the input
structures to correctly type heavy atoms as hydrogen bond donors").

The project currently ships two contradictory ligand-preparation behaviours:

  * ① five-model benchmark / 200k endpoint docking / E2PO generation eval
    -> SDF is fed to UniDock *without* explicit hydrogens
  * ③ offline IQL labelling / ④ online DDPO reward
    -> Chem.AddHs(addCoords=True) + Gasteiger before docking

Uni-Dock's own SDF reader (``unidock_tools/utils/read_ligand.py``) uses
``removeHs=False`` and its ligand prep (``modules/ligand_prep/torsion_tree.py``)
never calls ``AddHs``, so an H-free input cannot produce ``HD`` (donor) atoms.

This script quantifies the score difference on the *same* molecules so we can
decide whether the historical ① docking scores must be recomputed.

Arms
----
  A_noh_seed42   : current ① behaviour (input SDF has 0 explicit H)
  B_h_seed42     : Chem.AddHs(mol, addCoords=True), seed 42
  C_h_seed0      : same inputs as B, seed 0 (Uni-Dock default) -> isolates seed

Everything else (receptor, box, scoring, search mode, num_modes, batch) is
identical across arms; the only variable is ligand preparation (and seed for C).

Outputs
-------
  <arm>/inputs/*.sdf            ligands actually docked
  <arm>/docking/*_out.sdf       UniDock poses
  <arm>/unidock.{stdout,stderr}.log
  ab_scores.csv                 per-molecule scores for every arm
  ab_summary.json               paired statistics
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from statistics import mean, median

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = Path("/home/md666/data/disk6/DiffB110")
SRC = (
    ROOT
    / "analysis/reference_support_experiment/docking_subset/docking/part_0000/split_ligands"
)
EXISTING_RESULTS = (
    ROOT
    / "analysis/reference_support_experiment/docking_subset/docking/part_0000/docking_results.csv"
)
OUT_ROOT = ROOT / "analysis/unidock_hydrogen_ab_260930"

UNIDOCK = "/opt/unidock-1.1.3/bin/unidock"
RECEPTOR = str(ROOT / "data/8gjl.pdbqt")
GPU_ID = "0"
N_LIGANDS = 300

BOX = [
    "--center_x", "207.293",
    "--center_y", "212.430",
    "--center_z", "172.771",
    "--size_x", "20.0",
    "--size_y", "20.0",
    "--size_z", "20.0",
]
COMMON = [
    "--scoring", "vina",
    "--search_mode", "fast",
    "--num_modes", "1",
    "--verbosity", "1",
    "--cpu", "0",
    "--max_gpu_memory", "0",
]

PAT_VINA = re.compile(r"REMARK\s+VINA RESULT:\s*([-+]?\d+(?:\.\d+)?)")
PAT_UNIDOCK = re.compile(
    r">\s*<Uni-Dock RESULT>\s*\n(?:.*\n)*?ENERGY=\s*([-+]?\d+(?:\.\d+)?)",
    re.MULTILINE,
)
PAT_ENERGY = re.compile(r"ENERGY=\s*([-+]?\d+(?:\.\d+)?)")

ARMS = {
    "A_noh_seed42": {"prep": "noh", "seed": 42},
    "B_h_seed42": {"prep": "h", "seed": 42},
    "C_h_seed0": {"prep": "h", "seed": 0},
}


def build_inputs() -> list[str]:
    """Materialise the A (no-H) and B (AddHs) ligand sets. Returns stems."""
    src = sorted(SRC.glob("*.sdf"))[:N_LIGANDS]
    for arm in ("A_noh_seed42", "B_h_seed42"):
        (OUT_ROOT / arm / "inputs").mkdir(parents=True, exist_ok=True)
    stems: list[str] = []
    for path in src:
        mol = Chem.MolFromMolFile(str(path), sanitize=False, removeHs=False)
        if mol is None:
            raise RuntimeError(f"failed to read {path}")
        n_h = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 1)
        shutil.copy(path, OUT_ROOT / "A_noh_seed42/inputs" / path.name)
        mol_h = Chem.AddHs(Chem.Mol(mol), addCoords=True)
        Chem.MolToMolFile(mol_h, str(OUT_ROOT / "B_h_seed42/inputs" / path.name))
        if n_h != 0:
            raise RuntimeError(f"unexpected explicit H in source ligand {path.name}")
        stems.append(path.stem)
    print(f"[inputs] {len(stems)} ligands, source explicit_H == 0 confirmed")
    return stems


def run_arm(arm: str, input_arm: str, seed: int) -> int:
    inputs = sorted((OUT_ROOT / input_arm / "inputs").glob("*.sdf"))
    out_dir = OUT_ROOT / arm / "docking"
    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = GPU_ID
    cmd = [
        UNIDOCK,
        "--receptor", RECEPTOR,
        "--gpu_batch", *[str(p) for p in inputs],
        *BOX,
        *COMMON,
        "--seed", str(seed),
        "--dir", str(out_dir),
    ]
    (OUT_ROOT / arm / "unidock_command.txt").write_text(" ".join(cmd) + "\n")
    print(f"[run] {arm}: {len(inputs)} ligands, seed={seed}")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=3600)
    (OUT_ROOT / arm / "unidock.stdout.log").write_text(result.stdout, encoding="utf-8")
    (OUT_ROOT / arm / "unidock.stderr.log").write_text(result.stderr, encoding="utf-8")
    print(f"[run] {arm}: returncode={result.returncode}")
    return result.returncode


def score_of(path: Path) -> float | None:
    text = path.read_text(encoding="utf-8", errors="ignore")
    for pattern in (PAT_VINA, PAT_UNIDOCK, PAT_ENERGY):
        match = pattern.search(text)
        if match:
            return float(match.group(1))
    return None


def collect_scores(arm: str, stems: list[str]) -> dict[str, float]:
    dock_dir = OUT_ROOT / arm / "docking"
    scores: dict[str, float] = {}
    missing = 0
    for stem in stems:
        candidates = [
            dock_dir / f"{stem}_out.sdf",
            dock_dir / f"{stem}.sdf",
            dock_dir / f"{stem}_out.pdbqt",
        ]
        found = next((p for p in candidates if p.exists()), None)
        if found is None:
            missing += 1
            continue
        value = score_of(found)
        if value is None:
            missing += 1
            continue
        scores[stem] = value
    print(f"[parse] {arm}: {len(scores)} scored, {missing} missing")
    return scores


def load_existing() -> dict[str, float]:
    """Scores from the historical 200k docking run (arm A, batch_size 128)."""
    by_index: dict[str, float] = {}
    if not EXISTING_RESULTS.exists():
        return by_index
    with EXISTING_RESULTS.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("success") != "True":
                continue
            by_index[row["ligand_stem"]] = float(row["score"])
    return by_index


def rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(a: list[float], b: list[float]) -> float:
    ra, rb = rankdata(a), rankdata(b)
    n = len(ra)
    ma, mb = mean(ra), mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (
        sum((x - ma) ** 2 for x in ra) ** 0.5
        * sum((y - mb) ** 2 for y in rb) ** 0.5
    )
    return num / den if den else float("nan")


def paired(x: dict[str, float], y: dict[str, float], label: str) -> dict:
    keys = sorted(set(x) & set(y))
    if not keys:
        return {"pair": label, "n": 0}
    xs = [x[k] for k in keys]
    ys = [y[k] for k in keys]
    diffs = [b - a for a, b in zip(xs, ys)]  # >0 means arm B scores worse (less negative)
    absd = sorted(abs(d) for d in diffs)
    return {
        "pair": label,
        "n": len(keys),
        "median_x": median(xs),
        "median_y": median(ys),
        "median_diff": median(diffs),
        "mean_diff": mean(diffs),
        "pct_absdiff_gt_0_5": sum(1 for d in absd if d > 0.5) / len(absd),
        "pct_absdiff_gt_1_0": sum(1 for d in absd if d > 1.0) / len(absd),
        "p95_absdiff": absd[int(0.95 * (len(absd) - 1))],
        "spearman": spearman(xs, ys),
    }


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    stems = build_inputs()

    for arm, cfg in ARMS.items():
        input_arm = "B_h_seed42" if cfg["prep"] == "h" else "A_noh_seed42"
        run_arm(arm, input_arm, cfg["seed"])

    scores = {arm: collect_scores(arm, stems) for arm in ARMS}

    with (OUT_ROOT / "ab_scores.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ligand_stem", *ARMS.keys()])
        for stem in stems:
            writer.writerow(
                [stem, *[scores[arm].get(stem, "") for arm in ARMS]]
            )

    existing = load_existing()
    summaries = [
        paired(scores["A_noh_seed42"], scores["B_h_seed42"], "B_h - A_noh (H effect)"),
        paired(scores["B_h_seed42"], scores["C_h_seed0"], "C_seed0 - B_seed42 (seed effect)"),
        paired(existing, scores["A_noh_seed42"], "harnessA - historicalA (sanity)"),
    ]
    for row in summaries:
        print(json.dumps(row, ensure_ascii=False))

    (OUT_ROOT / "ab_summary.json").write_text(
        json.dumps({"summaries": summaries, "n_ligands": len(stems)}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
