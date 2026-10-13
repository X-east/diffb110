#!/usr/bin/env python
"""Stage 1 of the ChEMBL -> RE-CmeB docking reference pipeline.

Generate 3D conformers for ChEMBL SMILES via RDKit ETKDG + MMFF94, with an
explicit **stereochemistry preservation check**.

Why the check matters
---------------------
ChEMBL SMILES are isomeric (24.9% carry tetrahedral ``@``, 12.1% carry ``/``
double-bond stereo). ETKDG embeds with ``enforceChirality=True`` by default, so
already-assigned stereocentres should be preserved exactly. This script verifies
that per molecule and records the result in the sidecar, so a silent flip can
never contaminate the docking reference.

Distinction recorded explicitly:
- ``stereo_centers_input``  : assigned tetrahedral centres read from the SMILES
- ``stereo_flipped``        : assigned centres whose CIP tag changed after 3D
- ``stereo_newly_assigned`` : centres that were unspecified in the input and
  received a configuration from the 3D geometry (expected; NOT a flip)

Usage
-----
    conda run -n dif python analysis/chembl_docking_reference_260928/gen_3d_conformers.py \\
        --input data/chembl_smallmol_reference/reference_smallmol.csv.gz \\
        --output-dir analysis/chembl_docking_reference_260928/pilot_3d \\
        --limit 2000 --max-heavy-atoms 60 --num-workers 18 --shard-size 2000
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import time
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

LOGGER = logging.getLogger("chembl-3d")

DEFAULT_SEED = 42
DEFAULT_WORKERS = 18
DEFAULT_MAX_HEAVY_ATOMS = 60


def _assigned_centers(mol: Chem.Mol) -> dict[int, str]:
    """Return ``{atom_idx: CIP}`` for assigned tetrahedral stereocentres only."""
    return dict(
        Chem.FindMolChiralCenters(
            mol, force=True, includeUnassigned=False, useLegacyImplementation=False
        )
    )


def _ez_bonds(mol: Chem.Mol) -> dict[int, str]:
    """Return ``{bond_idx: stereo}`` for bonds with a specified E/Z stereo."""
    return {
        bond.GetIdx(): str(bond.GetStereo())
        for bond in mol.GetBonds()
        if bond.GetStereo() != Chem.BondStereo.STEREONONE
    }


def _process_one(task: tuple[int, str, str, int, int, int]) -> dict:
    """Embed one SMILES and return its metadata plus mol block text.

    Parameters are packed as a tuple so the function is picklable for a
    multiprocessing pool.
    """
    index, chembl_id, smiles, seed, max_heavy_atoms, mmff_max_iters = task
    row: dict[str, object] = {
        "row_index": index,
        "chembl_id": chembl_id,
        "input_smiles": smiles,
        "status": "ok",
        "failure_reason": "",
        "heavy_atoms": 0,
        "stereo_centers_input": 0,
        "stereo_flipped": 0,
        "stereo_newly_assigned": 0,
        "ez_input": 0,
        "ez_flipped": 0,
        "mol_block": "",
    }

    # One bad molecule (e.g. an unkekulizable ChEMBL entry) must never abort the
    # whole multiprocessing pool: every RDKit failure is reported per molecule.
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            row["status"] = "fail"
            row["failure_reason"] = "smiles_parse_failed"
            return row

        heavy_atoms = int(mol.GetNumHeavyAtoms())
        row["heavy_atoms"] = heavy_atoms
        if heavy_atoms > max_heavy_atoms:
            row["status"] = "skip"
            row["failure_reason"] = "skipped_too_large"
            return row
        if heavy_atoms == 0:
            row["status"] = "skip"
            row["failure_reason"] = "no_heavy_atom"
            return row

        in_centers = _assigned_centers(mol)
        in_ez = _ez_bonds(mol)
        row["stereo_centers_input"] = len(in_centers)
        row["ez_input"] = len(in_ez)

        mol_h = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol_h, randomSeed=seed, useRandomCoords=False) < 0:
            row["status"] = "fail"
            row["failure_reason"] = "embed_failed"
            return row
        if AllChem.MMFFOptimizeMolecule(mol_h, maxIters=int(mmff_max_iters)) < 0:
            row["status"] = "fail"
            row["failure_reason"] = "mmff_failed"
            return row

        mol3d = Chem.Mol(mol_h)
        Chem.AssignStereochemistryFrom3D(mol3d)
        out_centers = _assigned_centers(mol3d)

        flipped = 0
        for idx, cip_in in in_centers.items():
            cip_out = out_centers.get(idx)
            if cip_out is None or cip_out != cip_in:
                flipped += 1
        row["stereo_flipped"] = flipped
        row["stereo_newly_assigned"] = len(set(out_centers) - set(in_centers))

        out_ez = _ez_bonds(mol3d)
        ez_flipped = sum(
            1 for idx, stereo in in_ez.items() if out_ez.get(idx) != stereo
        )
        row["ez_flipped"] = ez_flipped

        if flipped or ez_flipped:
            row["status"] = "fail"
            row["failure_reason"] = f"stereo_changed(atoms={flipped},ez={ez_flipped})"
            return row

        mol3d.SetProp("_Name", str(chembl_id))
        row["mol_block"] = Chem.MolToMolBlock(mol3d)
        return row
    except Exception as exc:  # noqa: BLE001 - a single bad molecule must not kill the pool
        row["status"] = "fail"
        row["failure_reason"] = f"exception:{type(exc).__name__}"
        return row


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default="data/chembl_smallmol_reference/reference_smallmol.csv.gz"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=0, help="0 = no limit")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--max-heavy-atoms", type=int, default=DEFAULT_MAX_HEAVY_ATOMS)
    parser.add_argument(
        "--shard-size",
        type=int,
        default=100000,
        help="Number of molecules per output SDF shard",
    )
    return parser.parse_args()


def main() -> None:
    """Run the 3D conformer generation pass and write shards plus a sidecar."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = _parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = out_dir / "sdf_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    RDLogger.DisableLog("rdApp.*")

    frame = pd.read_csv(args.input, usecols=["chembl_id", "canonical_smiles"])
    if args.limit:
        frame = frame.head(args.limit)
    LOGGER.info("loaded %d molecules from %s", len(frame), args.input)

    tasks = [
        (i, str(cid), str(smi), args.seed, args.max_heavy_atoms, 2000)
        for i, (cid, smi) in enumerate(zip(frame["chembl_id"], frame["canonical_smiles"]))
    ]

    started = time.time()
    results: list[dict] = []
    with mp.Pool(processes=args.num_workers) as pool:
        for done, row in enumerate(
            pool.imap(_process_one, tasks, chunksize=64), start=1
        ):
            results.append(row)
            if done % 500 == 0:
                LOGGER.info("processed %d / %d", done, len(tasks))
    elapsed = time.time() - started

    results.sort(key=lambda r: int(r["row_index"]))

    ok = [r for r in results if r["status"] == "ok"]
    failed = [r for r in results if r["status"] == "fail"]
    skipped = [r for r in results if r["status"] == "skip"]

    writer_index = 0
    written = 0
    writer: Chem.SDWriter | None = None
    for r in ok:
        if writer is None or written % args.shard_size == 0:
            if writer is not None:
                writer.close()
            shard_path = shard_dir / f"shard_{writer_index:05d}.sdf"
            writer = Chem.SDWriter(str(shard_path))
            writer_index += 1
        mol = Chem.MolFromMolBlock(str(r["mol_block"]), sanitize=False, removeHs=False)
        if mol is not None:
            mol.SetProp("_Name", str(r["chembl_id"]))
            mol.SetProp("chembl_id", str(r["chembl_id"]))
            mol.SetProp("heavy_atoms", str(r["heavy_atoms"]))
            writer.write(mol)
            written += 1
    if writer is not None:
        writer.close()

    sidecar = pd.DataFrame(
        [{k: v for k, v in r.items() if k != "mol_block"} for r in results]
    )
    sidecar_path = out_dir / "gen3d_sidecar.csv"
    sidecar.to_csv(sidecar_path, index=False)

    n = len(results)
    n_input_stereo = sum(int(r["stereo_centers_input"]) for r in ok)
    n_flip = sum(int(r["stereo_flipped"]) for r in ok)
    LOGGER.info("=" * 66)
    LOGGER.info("3D generation complete")
    LOGGER.info("  total=%d  ok=%d  fail=%d  skip=%d", n, len(ok), len(failed), len(skipped))
    LOGGER.info("  ok_rate=%.4f  elapsed=%.1fs  mol/s=%.1f",
                len(ok) / n if n else 0.0, elapsed, n / elapsed if elapsed else 0.0)
    LOGGER.info("  sdf_shards=%d  sdf_records=%d", writer_index, written)
    LOGGER.info("  assigned stereocenters (ok mols)=%d  flipped=%d",
                n_input_stereo, n_flip)
    if failed:
        from collections import Counter
        reasons = Counter(str(r["failure_reason"]) for r in failed)
        LOGGER.info("  failure reasons: %s", dict(reasons))
    if skipped:
        from collections import Counter
        reasons = Counter(str(r["failure_reason"]) for r in skipped)
        LOGGER.info("  skip reasons: %s", dict(reasons))
    LOGGER.info("  sidecar=%s", sidecar_path)
    LOGGER.info("=" * 66)


if __name__ == "__main__":
    main()
