"""Molecular descriptor computation for epoch-level monitoring.

This module provides pure functions to compute aggregate molecular
descriptors from a batch of RDKit molecules. It is designed as a
standalone utility — no dependencies on training infrastructure.

All metrics are computed only on valid molecules (``rdkit_valid=True``).
The output is a flat dict suitable for inclusion in ``extra_metrics``
for CSV logging.
"""
from __future__ import annotations

import statistics
import warnings
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from ddpo.rewards.diversity_reward import DiversityReward

# ---------------------------------------------------------------------------
# Fingerprint parameters (match DiversityReward defaults)
# ---------------------------------------------------------------------------
DEFAULT_FINGERPRINT_RADIUS: int = 2
DEFAULT_FINGERPRINT_BITS: int = 2048

# ---------------------------------------------------------------------------
# Metric key names — single source of truth
# ---------------------------------------------------------------------------
METRIC_KEYS: tuple[str, ...] = (
    # LogP
    "logp_mean", "logp_median", "logp_std",
    # TPSA
    "tpsa_mean", "tpsa_median", "tpsa_std",
    # HBA / HBD
    "hba_mean", "hba_std", "hbd_mean", "hbd_std",
    # Composition
    "carbon_fraction",
    "hetero_atom_mean", "hetero_atom_std",
    "aromatic_ring_mean", "aromatic_ring_std",
    # Fraction-based alerts (Lipinski / absorption)
    "fraction_logp_high",
    "fraction_tpsa_low",
    "fraction_hba_zero_or_low",
    # Distribution shift vs. reference
    "descriptor_mmd_vs_baseline",
    # Fingerprint pairwise diversity
    "ecfp4_mean_pairwise_similarity",
    "fcfp4_mean_pairwise_similarity",
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _median(values: list[float]) -> float:
    """Compute median, returning 0.0 for empty input."""
    return float(statistics.median(values)) if values else 0.0


def _std(values: list[float]) -> float:
    """Compute sample standard deviation, returning 0.0 for <2 inputs."""
    if len(values) < 2:
        return 0.0
    return float(statistics.stdev(values))


def _rbf_mmd(x: np.ndarray, y: np.ndarray, sigma: float | None = None) -> float:
    """Unbiased MMD² between two sets of samples using an RBF kernel.

    Uses the median heuristic for sigma if not provided.
    """
    nx, ny = x.shape[0], y.shape[0]
    if nx == 0 or ny == 0:
        return 0.0

    def _kernel(a: np.ndarray, b: np.ndarray, s: float) -> np.ndarray:
        a2 = np.sum(a ** 2, axis=1, keepdims=True)
        b2 = np.sum(b ** 2, axis=1)[np.newaxis, :]
        sq_dists = a2 + b2 - 2.0 * a @ b.T
        return np.exp(-sq_dists / (2.0 * s ** 2))

    if sigma is None:
        # Median heuristic for kernel bandwidth
        combined = np.vstack([x, y])
        dists_sq = (
            np.sum(combined ** 2, axis=1, keepdims=True)
            + np.sum(combined ** 2, axis=1)[np.newaxis, :]
            - 2.0 * combined @ combined.T
        )
        med = float(np.median(np.sqrt(np.maximum(0.0, dists_sq))))
        sigma = max(med, 0.1) if med > 0 else 1.0

    kxx = _kernel(x, x, sigma)
    kyy = _kernel(y, y, sigma)
    kxy = _kernel(x, y, sigma)

    mmd2 = (np.sum(kxx) - np.trace(kxx)) / (nx * (nx - 1)) if nx > 1 else 0.0
    mmd2 += (np.sum(kyy) - np.trace(kyy)) / (ny * (ny - 1)) if ny > 1 else 0.0
    mmd2 -= 2.0 * float(np.mean(kxy))
    return float(max(0.0, mmd2))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_fingerprint_pairwise_similarities(
    rdmols: list[Chem.Mol],
    use_features: bool = False,
) -> list[float]:
    """Compute mean pairwise Tanimoto similarity for a list of RDKit molecules.

    Args:
        rdmols: Valid RDKit Mol objects (non-None).
        use_features: False → ECFP4, True → FCFP4.

    Returns:
        Mean similarity per molecule, or [0.0] * len(rdmols) when <2 molecules.
    """
    if len(rdmols) < 2:
        return [0.0] * len(rdmols)

    fps = DiversityReward.compute_fingerprints(rdmols, use_features=use_features)
    bit_sets = [DiversityReward.fingerprint_on_bits(fp) for fp in fps]
    return DiversityReward.compute_mean_similarities_from_bit_sets(bit_sets)


def compute_molecular_descriptors(
    rdmols: list[Chem.Mol | None],
    valid_mask: list[bool],
    reference_descriptor_vectors: list[list[float]] | None = None,
) -> dict[str, float]:
    """Compute epoch-level aggregate molecular descriptors for monitoring.

    All statistics are computed only over valid molecules (``rdkit_valid=True``).
    Returns a flat dict with all keys from :data:`METRIC_KEYS`.

    Args:
        rdmols: RDKit Mol objects per molecule (None for invalid).
        valid_mask: Per-molecule validity flags, same length as *rdmols*.
        reference_descriptor_vectors: Optional pre-computed 6D descriptor
            vectors from a reference set (e.g., ChEMBL). If None,
            ``descriptor_mmd_vs_baseline`` is 0.0.

    Returns:
        Flat dict with all descriptor metrics.  Returns all-zero when no
        valid molecules are present.
    """
    # Filter to valid molecules
    valid_mols = [
        mol for mol, v in zip(rdmols, valid_mask)
        if v and mol is not None
    ]

    if not valid_mols:
        return {k: 0.0 for k in METRIC_KEYS}

    # Per-molecule descriptor arrays
    logp_vals: list[float] = []
    tpsa_vals: list[float] = []
    hba_vals: list[float] = []
    hbd_vals: list[float] = []
    carbon_frac_vals: list[float] = []
    hetero_vals: list[float] = []
    aromatic_vals: list[float] = []
    descriptor_vectors: list[list[float]] = []

    for mol in valid_mols:
        logp = float(Descriptors.MolLogP(mol))
        tpsa = float(Descriptors.TPSA(mol))
        hba = float(Descriptors.NumHAcceptors(mol))
        hbd = float(Descriptors.NumHDonors(mol))
        heavy = mol.GetNumHeavyAtoms()
        n_carbon = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6)
        carbon_frac = n_carbon / heavy if heavy > 0 else 0.0
        hetero = heavy - n_carbon
        aromatic = float(rdMolDescriptors.CalcNumAromaticRings(mol))

        logp_vals.append(logp)
        tpsa_vals.append(tpsa)
        hba_vals.append(hba)
        hbd_vals.append(hbd)
        carbon_frac_vals.append(carbon_frac)
        hetero_vals.append(float(hetero))
        aromatic_vals.append(aromatic)
        descriptor_vectors.append([logp, tpsa, hba, hbd, carbon_frac, float(hetero)])

    n = len(valid_mols)
    result: dict[str, float] = {}

    # --- LogP ---
    result["logp_mean"] = float(np.mean(logp_vals))
    result["logp_median"] = _median(logp_vals)
    result["logp_std"] = _std(logp_vals)

    # --- TPSA ---
    result["tpsa_mean"] = float(np.mean(tpsa_vals))
    result["tpsa_median"] = _median(tpsa_vals)
    result["tpsa_std"] = _std(tpsa_vals)

    # --- HBA / HBD ---
    result["hba_mean"] = float(np.mean(hba_vals))
    result["hba_std"] = _std(hba_vals)
    result["hbd_mean"] = float(np.mean(hbd_vals))
    result["hbd_std"] = _std(hbd_vals)

    # --- Composition ---
    result["carbon_fraction"] = float(np.mean(carbon_frac_vals))
    result["hetero_atom_mean"] = float(np.mean(hetero_vals))
    result["hetero_atom_std"] = _std(hetero_vals)
    result["aromatic_ring_mean"] = float(np.mean(aromatic_vals))
    result["aromatic_ring_std"] = _std(aromatic_vals)

    # --- Fraction-based alerts ---
    result["fraction_logp_high"] = float(sum(1 for v in logp_vals if v > 5.0)) / n
    result["fraction_tpsa_low"] = float(sum(1 for v in tpsa_vals if v < 40.0)) / n
    result["fraction_hba_zero_or_low"] = float(sum(1 for v in hba_vals if v <= 2)) / n

    # --- MMD vs reference ---
    if reference_descriptor_vectors is not None and len(reference_descriptor_vectors) > 0:
        try:
            cur = np.array(descriptor_vectors, dtype=np.float64)
            ref = np.array(reference_descriptor_vectors, dtype=np.float64)
            result["descriptor_mmd_vs_baseline"] = _rbf_mmd(cur, ref)
        except Exception as exc:
            warnings.warn(f"[MolecularDescriptors] MMD computation failed: {exc}")
            result["descriptor_mmd_vs_baseline"] = 0.0
    else:
        result["descriptor_mmd_vs_baseline"] = 0.0

    # --- ECFP4 / FCFP4 mean pairwise similarity ---
    try:
        ecfp4_sims = compute_fingerprint_pairwise_similarities(valid_mols, use_features=False)
        result["ecfp4_mean_pairwise_similarity"] = float(np.mean(ecfp4_sims)) if ecfp4_sims else 0.0
    except Exception as exc:
        warnings.warn(f"[MolecularDescriptors] ECFP4 similarity failed: {exc}")
        result["ecfp4_mean_pairwise_similarity"] = 0.0

    try:
        fcfp4_sims = compute_fingerprint_pairwise_similarities(valid_mols, use_features=True)
        result["fcfp4_mean_pairwise_similarity"] = float(np.mean(fcfp4_sims)) if fcfp4_sims else 0.0
    except Exception as exc:
        warnings.warn(f"[MolecularDescriptors] FCFP4 similarity failed: {exc}")
        result["fcfp4_mean_pairwise_similarity"] = 0.0

    return result
