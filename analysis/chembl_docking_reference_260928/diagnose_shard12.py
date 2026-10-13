#!/usr/bin/env python
"""Diagnose why gen shard 12 is slow: replay its full molecule range through a
fresh Pool(16) with per-molecule timing and report the slowest completions."""
from __future__ import annotations

import sys
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gen_3d_conformers import _process_one  # noqa: E402

START = int(sys.argv[1]) if len(sys.argv) > 1 else 240000
END = int(sys.argv[2]) if len(sys.argv) > 2 else 260000


def timed(task):
    t0 = time.time()
    row = _process_one(task)
    return task[0], task[1], row["status"], time.time() - t0


def main() -> None:
    df = pd.read_csv(
        "data/chembl_smallmol_reference/reference_smallmol.csv.gz",
        usecols=["chembl_id", "canonical_smiles"],
    )
    chunk = df.iloc[START:END]
    tasks = [
        (START + i, str(c), str(s), 42, 60, 2000)
        for i, (c, s) in enumerate(zip(chunk["chembl_id"], chunk["canonical_smiles"]))
    ]
    print(f"replaying rows [{START},{END}) = {len(tasks)} molecules, Pool(16)")
    t0 = time.time()
    done = 0
    last_idx = None
    with Pool(16) as pool:
        for idx, cid, status, dt in pool.imap_unordered(timed, tasks, chunksize=8):
            done += 1
            last_idx = idx
            if done % 2000 == 0:
                print(f"  {done}/{len(tasks)} done, elapsed={time.time()-t0:.0f}s")
    total = time.time() - t0
    print(f"ALL DONE: {done} molecules in {total:.0f}s "
          f"({done/total:.1f} mol/s wall, last_completed_row={last_idx})")


if __name__ == "__main__":
    main()
