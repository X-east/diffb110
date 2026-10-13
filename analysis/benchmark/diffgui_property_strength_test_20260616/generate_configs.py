#!/usr/bin/env python
"""Generate 5 DiffGui property-conditioning test configs for RE-CmeB (8GJL)."""
from __future__ import annotations

import csv
from copy import deepcopy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]  # DiffB110 root
OUT_DIR = Path(__file__).resolve().parent / "configs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE = {
    "model": {
        "checkpoint": "ckpt/trained.pt",
        "target": str(ROOT / "analysis/benchmark/8gjl_c1101_benchmark_20260603_500mol_500steps/inputs/8GJL_C1101_pocket8A.pdb"),
        "ligand": "None",
        "frag": "None",
        "gen_mode": "denovo",
    },
    "bond_predictor": "ckpt/bond_trained.pt",
    "sample": {
        "seed": 20260616,
        "batch_size": 2,
        "num_mols": 84,  # 6 GPU × 84 = 504 before sanitize/filtering
        "save_traj_prob": 0.0,
        "sample": True,
        "sample_method": "priori",
        "mode": "pocket",
        "test_id": 0,
        "add_edge": None,
        "gui_strength": 3.0,
        "guidance": ["uncertainty", 1e-4],
        "num_timesteps": 200,  # override train config's 1000 → 5× faster
    },
    "data": {
        "name": "protein_ligand",
        "path": "data/PDBbind_v2020_pocket10",
        "split": "data/PDBbind_pocket10_split.pt",
        "protein_root": "data/PDBbind_v2020",
        "dataset": "pdbbind",
        "transform": {
            "ligand_atom_mode": "aromatic",
            "random_rot": False,
            "sample": False,
        },
    },
}

# 5 test points:
#  1. zero       — batch_lab=0   (no property conditioning, "unconditional")
#  2. default    — DiffGui defaults (benchmark baseline)
#  3. high_qed   — push drug-likeness
#  4. simple     — push simple / polar
#  5. affinity   — push hydrophobic + high affinity
TESTS = [
    {
        "idx": 1, "slug": "zero",
        "logp": 0.0, "tpsa": 0.0, "sa": 0.0, "qed": 0.0, "aff": 0.0,
        "purpose": "No property conditioning (batch_lab=0). Baseline for all other groups.",
    },
    {
        "idx": 2, "slug": "default",
        "logp": 2.0, "tpsa": 100, "sa": 1.0, "qed": 0.8, "aff": 12.0,
        "purpose": "DiffGui default property targets. Compare with benchmark.",
    },
    {
        "idx": 3, "slug": "high_qed",
        "logp": 2.5, "tpsa": 80, "sa": 1.5, "qed": 0.95, "aff": 10.0,
        "purpose": "Push drug-likeness: high QED, moderate logP, tighter TPSA.",
    },
    {
        "idx": 4, "slug": "simple_polar",
        "logp": 1.0, "tpsa": 130, "sa": 2.5, "qed": 0.7, "aff": 8.0,
        "purpose": "Push simpler + more polar: high SA (simpler), high TPSA, low affinity.",
    },
    {
        "idx": 5, "slug": "affinity_push",
        "logp": 3.5, "tpsa": 50, "sa": 1.0, "qed": 0.6, "aff": 16.0,
        "purpose": "Push hydrophobic + high pocket affinity: high logP, low TPSA, max affinity.",
    },
]

rows = []
for test in TESTS:
    cfg = deepcopy(BASE)
    cfg["model"]["logp"] = test["logp"]
    cfg["model"]["tpsa"] = test["tpsa"]
    cfg["model"]["sa"] = test["sa"]
    cfg["model"]["qed"] = test["qed"]
    cfg["model"]["aff"] = test["aff"]

    yaml_path = OUT_DIR / f"{test['idx']:02d}_{test['slug']}.yml"
    yaml_path.write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True, default_flow_style=False),
        encoding="utf-8",
    )
    rows.append({
        "idx": test["idx"],
        "slug": test["slug"],
        "config": str(yaml_path),
        "logp": test["logp"],
        "tpsa": test["tpsa"],
        "sa": test["sa"],
        "qed": test["qed"],
        "aff": test["aff"],
        "purpose": test["purpose"],
    })

matrix_csv = OUT_DIR / "matrix.csv"
with matrix_csv.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["idx", "slug", "config", "logp", "tpsa", "sa", "qed", "aff", "purpose"])
    w.writeheader()
    w.writerows(rows)

print(f"[generate] wrote {len(TESTS)} configs → {OUT_DIR}")
for r in rows:
    print(f"  {r['idx']:02d}_{r['slug']}: logp={r['logp']} tpsa={r['tpsa']} sa={r['sa']} qed={r['qed']} aff={r['aff']}")
