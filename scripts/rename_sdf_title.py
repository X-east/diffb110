#!/usr/bin/env python3
"""Rewrite SDF _Name title to include similarity score."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

from rdkit import Chem, RDLogger
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")
LOGGER = logging.getLogger("rename_sdf")

SDF = Path("bishe/RE-out/260407generated_10000_valid_dedup2d_chembl_scored.sdf")
TMP = SDF.with_suffix(".sdf.tmp")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    LOGGER.info("Reading: %s", SDF)

    suppl = Chem.SDMolSupplier(str(SDF), sanitize=False, removeHs=False)
    writer = Chem.SDWriter(str(TMP))
    count = 0

    for mol in tqdm(suppl, desc="molecules"):
        if mol is None:
            continue
        count += 1
        idx = mol.GetIntProp("id")
        score = mol.GetProp("chembl_top10_mean") if mol.HasProp("chembl_top10_mean") else "0.0"
        mol.SetProp("_Name", f"ID{idx}_top10={float(score):.6f}")
        writer.write(mol)

    writer.close()
    shutil.move(str(TMP), str(SDF))
    LOGGER.info("Done: %d molecules -> %s", count, SDF)


if __name__ == "__main__":
    main()
