#!/usr/bin/env python3
"""Add ChEMBL top-10 mean similarity scores as SDF properties."""
from __future__ import annotations

import logging
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from tqdm import tqdm

from ddpo.rewards.chembl_similarity import load_chembl_fingerprint_cache

RDLogger.DisableLog("rdApp.*")
LOGGER = logging.getLogger("annotate_sdf")

SDF_IN = Path("bishe/RE-out/260407generated_10000_valid_dedup2d.sdf")
SDF_OUT = Path("bishe/RE-out/260407generated_10000_valid_dedup2d_chembl_scored.sdf")
CACHE = Path("data/chembl/chembl_36_morgan_fingerprints.pkl.gz")
CHUNK_SIZE = 100000


def compute_top_k(query_fp, reference_fps, max_k=10, chunk_size=100000):
    best = []
    for start in range(0, len(reference_fps), chunk_size):
        sims = AllChem.DataStructs.BulkTanimotoSimilarity(
            query_fp, reference_fps[start : start + chunk_size]
        )
        if not sims:
            continue
        if len(best) < max_k:
            best.extend(float(v) for v in sims)
            best.sort(reverse=True)
            best = best[:max_k]
            continue
        cutoff = best[-1]
        candidates = [float(v) for v in sims if float(v) > cutoff]
        if candidates:
            best.extend(candidates)
            best.sort(reverse=True)
            best = best[:max_k]
    return best


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    LOGGER.info("Loading ChEMBL cache: %s", CACHE)
    cache = load_chembl_fingerprint_cache(str(CACHE))
    LOGGER.info("Loaded %d reference fingerprints", len(cache.fingerprints))

    supplier = Chem.SDMolSupplier(str(SDF_IN), sanitize=False, removeHs=False)
    writer = Chem.SDWriter(str(SDF_OUT))

    total = 0
    scored = 0
    for idx, raw_mol in enumerate(tqdm(supplier, desc="molecules")):
        if raw_mol is None:
            continue
        total += 1

        mol = Chem.Mol(raw_mol)
        try:
            Chem.SanitizeMol(mol)
            clean_mol = Chem.RemoveHs(mol)
        except Exception:
            clean_mol = None

        mol.SetProp("_Name", str(idx + 1))
        mol.SetIntProp("id", idx + 1)

        if clean_mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(
                clean_mol, radius=cache.radius, nBits=cache.n_bits
            )
            sorted_top = compute_top_k(fp, cache.fingerprints, max_k=10, chunk_size=CHUNK_SIZE)
            if sorted_top:
                import numpy as np
                vals = np.array(sorted_top[:10])
                max_sim = float(vals[0])
                top10_mean = float(vals.mean())
                mol.SetDoubleProp("chembl_max", round(max_sim, 6))
                mol.SetDoubleProp("chembl_top10_mean", round(top10_mean, 6))
                mol.SetProp("canonical_smiles", Chem.MolToSmiles(clean_mol, isomericSmiles=False, canonical=True))
                scored += 1

        writer.write(mol)

    writer.close()
    LOGGER.info("Done: %d molecules written to %s (scored: %d)", total, SDF_OUT, scored)


if __name__ == "__main__":
    main()
