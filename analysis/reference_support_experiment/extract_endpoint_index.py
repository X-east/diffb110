#!/usr/bin/env python
"""Build the endpoint index: sample_id <-> (.pt shard, mol_index) <-> SDF location.

E2PO reads endpoints from the ``.pt`` trajectory shards (``x``, ``categorical``,
``nodesxsample``, ``sample_ids``), while the realism score and the docking score
were computed on the ``molecules/*.sdf`` outputs.  These two views must be joined.

Verified mapping rule (checked exactly on shard_0000, 7/7):
    SDF record k  ==  k-th entry with rdkit_valid == 1 in ``reconstruction_rows``
    its ``mol_index`` is the index inside the .pt shard (0..9)

Only the small fields of each shard are kept, so the scan is cheap.
"""

from __future__ import annotations

import csv
import glob
import multiprocessing as mp
import re
from pathlib import Path

import torch

ROOT = Path("/home/md666/data/disk6/DiffB110")
HERE = ROOT / "analysis/reference_support_experiment"
OUT = HERE / "endpoint_index.csv"

FIELDS = [
    "sample_id", "pt_shard", "mol_index", "n_nodes",
    "sdf_path", "sdf_index", "smiles_raw",
]

_SDF_RE = re.compile(r"^(.*)/shards/(rank_\d+)/shard_(\d+)\.pt$")


def _sdf_for_pt(pt_path: str) -> str:
    m = _SDF_RE.match(pt_path)
    if not m:
        return ""
    base, rank, shard = m.group(1), m.group(2), m.group(3)
    rank_num = rank.split("_")[1]
    return f"{base}/molecules/{rank}/shard_{shard}/diffsbdd_collect_r{rank_num}_s{shard}.sdf"


def _worker(path: str) -> list[dict]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return []
    rows = payload.get("reconstruction_rows") or []
    nodes = payload.get("nodesxsample")
    sdf_path = _sdf_for_pt(path)
    out: list[dict] = []
    k = 0
    for i, row in enumerate(rows):
        try:
            valid = int(row.get("rdkit_valid", 0)) == 1
        except Exception:
            continue
        if not valid:
            continue
        n_nodes = int(nodes[i].item()) if nodes is not None else 0
        out.append({
            "sample_id": str(row.get("sample_id", "")),
            "pt_shard": path,
            "mol_index": i,
            "n_nodes": n_nodes,
            "sdf_path": sdf_path,
            "sdf_index": k,
            "smiles_raw": str(row.get("smiles", "")),
        })
        k += 1
    return out


def main() -> None:
    shards = sorted(glob.glob(str(ROOT / "runs/diffsbdd_collect/**/shards/**/*.pt"), recursive=True))
    print(f"[scan] {len(shards):,} pt shards")
    rows: list[dict] = []
    with mp.Pool(8) as pool:
        for n, part in enumerate(pool.imap_unordered(_worker, shards, chunksize=32)):
            rows.extend(part)
            if (n + 1) % 4000 == 0:
                print(f"   {n + 1}/{len(shards)}  endpoints={len(rows):,}")
    print(f"[scan] endpoints: {len(rows):,}")

    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"[wrote] {OUT}")


if __name__ == "__main__":
    main()
