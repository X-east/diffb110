#!/usr/bin/env python
"""Same A/B hydrogen test, but on the DiffSBDD *generation* population.

The first A/B (run_ab.py) used the 200k collection endpoint pool, which was
sampled with a uniform `randint(10, 31)` node distribution.  The question
"is DiffSBDD's ~-7 kcal/mol docking score caused by the missing hydrogens?"
must be answered on the population that actually produced that number:
`analysis/diffsbdd_global_e2po_structure_matched_x_only_lr1e5_full_generation_1000_260902/molecules.sdf`
(1000 fixed attempts, same protocol as the reported -7.937 median).

Arms
----
  A_noh : molecules.sdf as written by the generator (0 explicit H), docked with
          the project's production docker (largest-fragment + UniDock).
  B_h   : the same records with Chem.AddHs(addCoords=True) before docking.

Everything else is identical: same docker script, same receptor, box, scoring
(vina), search_mode (fast), num_modes (1), seed (42), batch size (256).
"""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
from pathlib import Path
from statistics import mean, median

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = Path("/home/md666/data/disk6/DiffB110")
GEN_DIR = (
    ROOT
    / "analysis/diffsbdd_global_e2po_structure_matched_x_only_lr1e5_full_generation_1000_260902"
)
SRC_SDF = GEN_DIR / "molecules.sdf"
DOCKER = (
    ROOT
    / "analysis/benchmark/8gjl_c1101_benchmark_20260624_1000mol_500steps/scripts/dock_sdf_unidock.py"
)
OUT_ROOT = ROOT / "analysis/unidock_hydrogen_ab_260930/generation"
BATCH_SIZE = 256

PAT_VINA = re.compile(r"REMARK\s+VINA RESULT:\s*([-+]?\d+(?:\.\d+)?)")


def write_hydrogenated(src: Path, dst: Path) -> tuple[int, int]:
    """Write a copy of every record in `src` with explicit H added."""
    supplier = Chem.SDMolSupplier(str(src), sanitize=False, removeHs=False)
    writer = Chem.SDWriter(str(dst))
    n_records = 0
    n_zero_h = 0
    for mol in supplier:
        if mol is None:
            continue
        n_h_before = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 1)
        if n_h_before == 0:
            n_zero_h += 1
        mol_h = Chem.AddHs(Chem.Mol(mol), addCoords=True)
        writer.write(mol_h)
        n_records += 1
    writer.close()
    return n_records, n_zero_h


def run_docker(tag: str, input_sdf: Path) -> int:
    out_dir = OUT_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    cmd = [
        "conda", "run", "-n", "dif", "python", str(DOCKER),
        "--model", f"hydrogen_ab_{tag}",
        "--input-sdf", str(input_sdf),
        "--receptor", str(ROOT / "data/8gjl.pdbqt"),
        "--out-dir", str(out_dir),
        "--unidock-bin", "/opt/unidock-1.1.3/bin/unidock",
        "--gpu-id", "0",
        "--batch-size", str(BATCH_SIZE),
    ]
    (OUT_ROOT / f"{tag}.command.txt").write_text(" ".join(cmd) + "\n")
    print(f"[run] {tag}: input={input_sdf.name}")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=3600)
    (OUT_ROOT / f"{tag}.stdout.log").write_text(result.stdout, encoding="utf-8")
    (OUT_ROOT / f"{tag}.stderr.log").write_text(result.stderr, encoding="utf-8")
    print(f"[run] {tag}: returncode={result.returncode}")
    return result.returncode


def load_results(tag: str) -> dict[int, float]:
    path = OUT_ROOT / tag / "docking_results.csv"
    scores: dict[int, float] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("success") != "True":
                continue
            scores[int(row["benchmark_index"])] = float(row["score"])
    return scores


def load_hac(tag: str) -> dict[int, int]:
    """Heavy-atom count per benchmark_index, read from the split ligands."""
    sdf = GEN_DIR / "molecules.sdf" if tag == "A_noh" else OUT_ROOT / "molecules_h.sdf"
    counts: dict[int, int] = {}
    for idx, mol in enumerate(Chem.SDMolSupplier(str(sdf), sanitize=False, removeHs=False)):
        if mol is None:
            continue
        counts[idx] = mol.GetNumHeavyAtoms()
    return counts


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
    ma, mb = mean(ra), mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = sum((x - ma) ** 2 for x in ra) ** 0.5 * sum((y - mb) ** 2 for y in rb) ** 0.5
    return num / den if den else float("nan")


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    dst = OUT_ROOT / "molecules_h.sdf"
    n_records, n_zero_h = write_hydrogenated(SRC_SDF, dst)
    print(f"[prep] records={n_records}, records_with_0_H_before={n_zero_h}")

    run_docker("A_noh", SRC_SDF)
    run_docker("B_h", dst)

    a = load_results("A_noh")
    b = load_results("B_h")
    keys = sorted(set(a) & set(b))
    diffs = [b[k] - a[k] for k in keys]

    hac = load_hac("A_noh")
    diff_by_key = {k: b[k] - a[k] for k in keys}
    per_ha: dict[int, list[float]] = {}
    for k in keys:
        per_ha.setdefault(hac.get(k, 0), []).append(diff_by_key[k])

    absd = sorted(abs(d) for d in diffs)

    def topk(scores: dict[int, float], k: int = 50) -> set[int]:
        return {idx for _, idx in sorted((s, i) for i, s in scores.items())[:k]}

    summary = {
        "population": "diffsbdd_lr1e5_generation_1000_260902",
        "n_paired": len(keys),
        "median_A_noh": median([a[k] for k in keys]),
        "median_B_h": median([b[k] for k in keys]),
        "median_diff_B_minus_A": median(diffs),
        "mean_diff_B_minus_A": mean(diffs),
        "pct_absdiff_gt_0_5": sum(1 for d in absd if d > 0.5) / len(absd),
        "pct_absdiff_gt_1_0": sum(1 for d in absd if d > 1.0) / len(absd),
        "p95_absdiff": absd[int(0.95 * (len(absd) - 1))],
        "spearman": spearman([a[k] for k in keys], [b[k] for k in keys]),
        "top50_overlap": len(topk(a) & topk(b)),
        "mean_diff_by_heavy_atom_count": {
            str(ha): round(mean(v), 3) for ha, v in sorted(per_ha.items()) if len(v) >= 10
        },
        "existing_run_median": -7.937,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    (OUT_ROOT / "ab_generation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (OUT_ROOT / "ab_generation_scores.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["benchmark_index", "heavy_atoms", "A_noh", "B_h", "diff_B_minus_A"])
        for k in keys:
            writer.writerow([k, hac.get(k, ""), a[k], b[k], b[k] - a[k]])


if __name__ == "__main__":
    main()
