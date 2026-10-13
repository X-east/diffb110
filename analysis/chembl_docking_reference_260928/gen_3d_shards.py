#!/usr/bin/env python
"""Stage 1 (streaming): generate ETKDG 3D shards for the ChEMBL docking reference.

Streams the ChEMBL SMILES in fixed-size chunks (default 20,000), embeds each
chunk with RDKit ETKDG + MMFF94 (multiprocessing, 18 workers by default), and
writes one multi-record SDF shard plus a per-molecule sidecar and a ``.done``
marker per chunk. Chunking bounds memory and makes the stage resumable, and it
lets the Stage-2 docking workers consume shards incrementally (pipeline).

Reuses the tested ``_process_one`` from ``gen_3d_conformers`` so the chirality
preservation check is identical to the pilot.

Chirality policy
----------------
Already-assigned stereocentres must be preserved. Molecules whose assigned CIP
tags or E/Z bonds change after embedding/optimisation are recorded as failures
and excluded from the shard, so a silent flip cannot enter the docking reference.

Usage
-----
    conda run -n dif python gen_3d_shards.py \\
        --input data/chembl_smallmol_reference/reference_smallmol.csv.gz \\
        --out-dir analysis/chembl_docking_reference_260928/full \\
        --shard-size 20000 --num-workers 18 --max-heavy-atoms 60 \\
        --mmff-max-iters 200
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gen_3d_conformers import _process_one  # noqa: E402

LOGGER = logging.getLogger("chembl-3d-shards")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default="data/chembl_smallmol_reference/reference_smallmol.csv.gz"
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--shard-size", type=int, default=20000)
    parser.add_argument("--num-workers", type=int, default=18)
    parser.add_argument("--max-heavy-atoms", type=int, default=60)
    parser.add_argument("--mmff-max-iters", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="Redo existing shards")
    parser.add_argument("--redo-shards", default="",
                        help="Comma-separated shard indices to force-redo even if marked done")
    return parser.parse_args()


def _write_shard(
    results: list[dict],
    *,
    ligand_dir: Path,
    sidecar_path: Path,
) -> tuple[int, int]:
    """Write one shard as per-molecule SDF files; return (written, ok).

    UniDock reads only the first record of a multi-record SDF, so every molecule
    must be its own file before docking. Deletion is blocked in this environment,
    so files are written once here and reused by the docking workers (no temp
    copies, no cleanup).
    """
    ligand_dir.mkdir(parents=True, exist_ok=True)
    ok = [r for r in results if r["status"] == "ok"]
    written = 0
    for row in ok:
        mol = Chem.MolFromMolBlock(str(row["mol_block"]), sanitize=False, removeHs=False)
        if mol is None:
            continue
        mol.SetProp("_Name", str(row["chembl_id"]))
        mol.SetProp("chembl_id", str(row["chembl_id"]))
        mol.SetProp("heavy_atoms", str(row["heavy_atoms"]))
        writer = Chem.SDWriter(str(ligand_dir / f"{row['chembl_id']}.sdf"))
        writer.write(mol)
        writer.close()
        written += 1
    sidecar = pd.DataFrame([{k: v for k, v in r.items() if k != "mol_block"}
                            for r in results])
    sidecar.to_csv(sidecar_path, index=False)
    return written, len(ok)


def main() -> None:
    """Generate 3D shards chunk by chunk and mark each completed shard."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    out_dir = Path(args.out_dir)
    ligand_root = out_dir / "ligands"
    sidecar_dir = out_dir / "sidecars"
    marker_dir = out_dir / "markers"
    for d in (ligand_root, sidecar_dir, marker_dir):
        d.mkdir(parents=True, exist_ok=True)

    RDLogger.DisableLog("rdApp.*")

    frame = pd.read_csv(args.input, usecols=["chembl_id", "canonical_smiles"])
    if args.limit:
        frame = frame.head(args.limit)
    n_total = len(frame)
    n_shards = (n_total + args.shard_size - 1) // args.shard_size
    redo = {int(x) for x in str(args.redo_shards).split(",") if x.strip()}
    LOGGER.info("molecules=%d  shard_size=%d  n_shards=%d  workers=%d  mmff_max_iters=%d",
                n_total, args.shard_size, n_shards, args.num_workers, args.mmff_max_iters)

    summary = {"n_total": n_total, "n_shards": n_shards, "shards": []}
    started = time.time()

    with mp.Pool(processes=args.num_workers) as pool:
        for shard_idx, start in enumerate(range(0, n_total, args.shard_size)):
            ligand_dir = ligand_root / f"shard_{shard_idx:05d}"
            sidecar_path = sidecar_dir / f"shard_{shard_idx:05d}.csv"
            marker_path = marker_dir / f"shard_{shard_idx:05d}.done"
            if marker_path.exists() and not args.force and shard_idx not in redo:
                LOGGER.info("shard %d already done, skipping", shard_idx)
                continue

            chunk = frame.iloc[start:start + args.shard_size]
            tasks = [
                (start + i, str(cid), str(smi), args.seed, args.max_heavy_atoms,
                 args.mmff_max_iters)
                for i, (cid, smi) in enumerate(
                    zip(chunk["chembl_id"], chunk["canonical_smiles"])
                )
            ]
            t0 = time.time()
            t_shard = time.time()
            results = []
            # imap_unordered + heartbeat: pool.map gives zero visibility inside a
            # shard, so a pathological molecule stalls silently (shard 12 once ran
            # 56 min instead of ~8). The heartbeat makes any stall immediately
            # visible as a stopped counter.
            for res in pool.imap_unordered(_process_one, tasks, chunksize=8):
                results.append(res)
                if len(results) % 2000 == 0:
                    LOGGER.info("shard %d heartbeat: %d/%d molecules (%.0fs)",
                                shard_idx, len(results), len(tasks),
                                time.time() - t_shard)
            results.sort(key=lambda r: int(r["row_index"]))
            written, ok = _write_shard(
                results, ligand_dir=ligand_dir, sidecar_path=sidecar_path
            )
            dt = time.time() - t0
            reasons = Counter(str(r["failure_reason"]) for r in results
                              if r["status"] != "ok")
            marker_path.write_text(json.dumps({
                "shard_idx": shard_idx, "start": start, "n": len(tasks),
                "written": written, "ok": ok, "seconds": dt,
                "reasons": dict(reasons), "ligand_dir": str(ligand_dir),
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            summary["shards"].append({
                "shard_idx": shard_idx, "written": written, "ok": ok,
                "seconds": dt, "reasons": dict(reasons),
            })
            done_n = min(start + args.shard_size, n_total)
            elapsed = time.time() - started
            LOGGER.info("shard %d/%d: written=%d ok=%d (%.1fs)  progress=%d/%d  "
                        "elapsed=%.0fs  rate=%.1f mol/s",
                        shard_idx, n_shards - 1, written, ok, dt, done_n, n_total,
                        elapsed, done_n / elapsed if elapsed else 0.0)
            (out_dir / "gen_progress.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8")

    (out_dir / "gen_done.json").write_text(
        json.dumps({"status": "completed", "n_total": n_total,
                    "elapsed_seconds": time.time() - started,
                    "n_shards": n_shards}, indent=2) + "\n",
        encoding="utf-8")
    LOGGER.info("ALL SHARDS DONE: %d molecules, %.0fs", n_total, time.time() - started)


if __name__ == "__main__":
    main()
