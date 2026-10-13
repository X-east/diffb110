#!/usr/bin/env python
"""Dock generated molecules into RE-CmeB with UniDock and summarise scores.

Same scoring setup as the ChEMBL docking reference run (vina scoring, fast
search, same receptor/box) so the distributions are directly comparable.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdf", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--receptor", type=Path, default=Path("data/8gjl.pdbqt"))
    parser.add_argument("--unidock", type=Path, default=Path("/opt/unidock-1.1.3/bin/unidock"))
    parser.add_argument("--gpu", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    lig_dir = args.work_dir / "ligands"
    lig_dir.mkdir(exist_ok=True)
    out_dir = args.work_dir / "outputs"
    out_dir.mkdir(exist_ok=True)  # unidock does not create the output directory

    # 1) extract sanitize-passing molecules as individual ligand SDFs
    supplier = Chem.SDMolSupplier(str(args.sdf), sanitize=False, removeHs=False)
    ligands: list[tuple[str, Chem.Mol]] = []
    n_attempts = n_fail = 0
    for record in supplier:
        n_attempts += 1
        if record is None:
            n_fail += 1
            continue
        mol = Chem.Mol(record)
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            n_fail += 1
            continue
        name = f"gen{len(ligands):04d}"
        mol = Chem.AddHs(mol, addCoords=True)  # unidock force field needs explicit H
        mol.SetProp("_Name", name)
        writer_path = lig_dir / f"{name}.sdf"
        writer = Chem.SDWriter(str(writer_path))
        writer.write(mol)
        writer.close()
        ligands.append((name, mol))
    print(f"attempts={n_attempts} sanitized={len(ligands)} failed={n_fail}")

    # 2) unidock (vina scoring, fast search — same as ChEMBL reference),
    #    with the same sub-batch fallback as dock_worker (random rc=-11
    #    segfaults would otherwise lose the whole batch)
    def run_unidock(names: list[str], batch_dir: Path) -> int:
        batch_dir.mkdir(parents=True, exist_ok=True)
        idx = batch_dir / "ligand_index.txt"
        idx.write_text("\n".join(str((lig_dir / f"{n}.sdf").resolve()) for n in names) + "\n")
        cmd = [
            str(args.unidock), "--receptor", str(args.receptor.resolve()),
            "--ligand_index", str(idx.resolve()),
            "--center_x", "207.293", "--center_y", "212.430", "--center_z", "172.771",
            "--size_x", "20", "--size_y", "20", "--size_z", "20",
            "--scoring", "vina", "--search_mode", "fast",
            "--num_modes", "1", "--seed", str(args.seed),
            "--max_gpu_memory", "0", "--verbosity", "1",
            "--dir", str(batch_dir),
        ]
        return subprocess.run(cmd, capture_output=True, text=True).returncode

    def dock_with_fallback(names: list[str], tag: str) -> None:
        if not names:
            return
        rc = run_unidock(names, out_dir / f"batch_{tag}")
        if rc == 0:
            return
        print(f"batch {tag}: rc={rc}, falling back to sub-batches of 50")
        for k in range(0, len(names), 50):
            sub = names[k : k + 50]
            if run_unidock(sub, out_dir / f"batch_{tag}_sub{k:05d}") != 0:
                print(f"batch {tag} sub {k}: retrying in 5s")
                for j in range(0, len(sub), 5):
                    if run_unidock(sub[j : j + 5], out_dir / f"batch_{tag}_s{k:05d}_{j:05d}") != 0:
                        print(f"batch {tag} sub {k} group {j}: giving up")

    print("running unidock on", len(ligands), "ligands ...")
    all_names = [n for n, _ in ligands]
    dock_with_fallback(all_names, "whole")

    # 3) parse scores (vina REMARK in output pdbqt/sdf)
    scores: dict[str, float] = {}
    for name, _ in ligands:
        for cand in (out_dir / f"{name}_out.sdf", out_dir / f"{name}_out.pdbqt"):
            if cand.exists():
                text = cand.read_text(errors="ignore")
                for line in text.splitlines():
                    if line.startswith("REMARK VINA RESULT:") or "VINA RESULT" in line:
                        scores[name] = float(line.split()[3])
                        break
                if name in scores:
                    break
    rows = [
        {"name": n, "score": scores.get(n)}
        for n, _ in ligands
    ]
    frame = pd.DataFrame(rows)
    frame.to_csv(args.work_dir / "gen_docking_scores.csv", index=False)

    sc = frame["score"].dropna()
    print(f"\ndocking success: {len(sc)}/{len(ligands)} ({len(sc)/len(ligands):.1%})")
    if len(sc):
        print(sc.describe(percentiles=[.05, .25, .5, .75, .95]).round(2).to_string())
        print()
        print("reference medians: ChEMBL-ref -9.16 | DiffSBDD-200k -7.27")
    return


if __name__ == "__main__":
    main()
