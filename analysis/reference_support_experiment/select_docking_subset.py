#!/usr/bin/env python
"""Step 2a: sample a docking subset from the profiled 200k pool.

The original shard SDFs are re-opened and the selected molecules are copied by
index, so the *generated 3D conformers* are preserved (docking needs them;
regenerating from SMILES would change the input distribution).

Outputs
-------
docking_subset/sdf/part_XXXX.sdf      ready-to-dock shards (~1000 molecules each)
docking_subset/subset_manifest.csv    molecule_id -> smiles, soft_novel, shard, index
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = Path("/home/md666/data/disk6/DiffB110")
HERE = ROOT / "analysis/reference_support_experiment"
PROFILES = HERE / "candidate_200k_profiles.csv"

FIELDS = [
    "mol_id", "smiles", "soft_novel", "hard_novel_frac", "null_percentile",
    "element_composition", "heavy_atom_count", "tpsa", "molwt", "logp",
    "murcko_scaffold", "n_fragments", "source_shard", "source_index", "part",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40_000)
    ap.add_argument("--part-size", type=int, default=1_000)
    ap.add_argument("--seed", type=int, default=20260914)
    ap.add_argument("--out", default=str(HERE / "docking_subset"))
    args = ap.parse_args()

    rows = list(csv.DictReader(PROFILES.open("r", newline="")))
    print(f"[pool] {len(rows):,} profiled molecules")

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    sel = rows[: args.n]
    print(f"[sample] {len(sel):,} molecules (seed={args.seed})")

    lookup = {(r["shard"], int(r["shard_index"])): r for r in sel}
    by_shard: dict[str, list[int]] = defaultdict(list)
    for r in sel:
        by_shard[r["shard"]].append(int(r["shard_index"]))
    print(f"[sample] spans {len(by_shard):,} source shards")

    out = Path(args.out)
    sdf_dir = out / "sdf"
    sdf_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    writer = None
    part_idx = 0
    written_in_part = 0
    mol_id = 0

    def open_part() -> None:
        nonlocal writer, written_in_part
        path = sdf_dir / f"part_{part_idx:04d}.sdf"
        writer = Chem.SDWriter(str(path))
        written_in_part = 0
        print(f"[write] {path.name}")

    open_part()

    for shard in sorted(by_shard):
        try:
            supplier = Chem.SDMolSupplier(shard, removeHs=False, sanitize=True)
            mols = [m for m in supplier]          # keep None placeholders for index alignment
        except Exception:
            continue
        for i in by_shard[shard]:
            if i >= len(mols):
                continue
            mol = mols[i]
            if mol is None:
                continue
            row = lookup.get((shard, i))
            if row is None:
                continue
            writer.write(mol)
            manifest.append({
                "mol_id": mol_id,
                "smiles": row["smiles"],
                "soft_novel": row["soft_novel"],
                "hard_novel_frac": row["hard_novel_frac"],
                "null_percentile": row["null_percentile"],
                "element_composition": row["element_composition"],
                "heavy_atom_count": row["heavy_atom_count"],
                "tpsa": row["tpsa"],
                "molwt": row["molwt"],
                "logp": row["logp"],
                "murcko_scaffold": row["murcko_scaffold"],
                "n_fragments": row["n_fragments"],
                "source_shard": shard,
                "source_index": i,
                "part": part_idx,
            })
            mol_id += 1
            written_in_part += 1
            if written_in_part >= args.part_size:
                writer.close()
                part_idx += 1
                open_part()

    writer.close()

    with (out / "subset_manifest.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(manifest)

    print(f"[done] {mol_id:,} molecules -> {part_idx + 1} parts")
    print(f"[wrote] {out / 'subset_manifest.csv'}")


if __name__ == "__main__":
    main()
