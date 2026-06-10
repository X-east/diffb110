"""Molecular descriptor computation for epoch-level monitoring.

This module provides pure functions to compute aggregate molecular
descriptors from a batch of RDKit molecules. It is designed as a
standalone utility — no dependencies on training infrastructure.

All metrics are computed only on valid molecules (``rdkit_valid=True``).
The output is a flat dict suitable for inclusion in ``extra_metrics``
for CSV logging.
"""
from __future__ import annotations

import csv
import gzip
import statistics
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

# ---------------------------------------------------------------------------
# Fingerprint parameters (match DiversityReward defaults)
# ---------------------------------------------------------------------------
DEFAULT_FINGERPRINT_RADIUS: int = 2
DEFAULT_FINGERPRINT_BITS: int = 2048

# ---------------------------------------------------------------------------
# ChEMBL reference vectors (for MMD vs baseline)
# ---------------------------------------------------------------------------
# Column names in the properties CSV that form the 6D descriptor vector.
_REFERENCE_6D_COLUMNS: list[str] = [
    "mol_log_p", "tpsa", "num_h_acceptors", "num_h_donors",
    "num_heteroatoms", "carbon_fraction",
]
# Default path — resolved relative to project root.
_DEFAULT_REFERENCE_CSV: str = "data/chembl/chembl_36_properties.csv.gz"
# Lazy-loaded cache (module-level singleton, `np.ndarray` shape [N, 6]).
_REFERENCE_CACHE: np.ndarray | None = None

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


# ---------------------------------------------------------------------------
# MMD computation (with subsampling for large sets)
# ---------------------------------------------------------------------------
# Maximum samples used for pairwise MMD computation.
# The MMD estimator uses O(N²) memory — this cap prevents OOM with
# large reference sets (e.g., 1.55M ChEMBL molecules).
_MAX_MMD_SAMPLES: int = 5000


def _rbf_mmd(x: np.ndarray, y: np.ndarray, sigma: float | None = None,
             max_samples: int | None = None) -> float:
    """Unbiased MMD² between two sets of samples using an RBF kernel.

    To avoid O(N²) memory blowup, both *x* and *y* are sub-sampled to
    at most *max_samples* (default ``_MAX_MMD_SAMPLES``) when they exceed
    this cap.  The median heuristic for sigma is computed on the
    sub-sampled combined set.

    Args:
        x: Current batch descriptor vectors [N, D].
        y: Reference descriptor vectors [M, D].
        sigma: Optional fixed kernel bandwidth (auto from median if None).
        max_samples: Max points to use from each set for pairwise
            computation (defaults to ``_MAX_MMD_SAMPLES``).

    Returns:
        Unbiased MMD² estimate (≥ 0).
    """
    nx, ny = x.shape[0], y.shape[0]
    if nx == 0 or ny == 0:
        return 0.0

    if max_samples is None:
        max_samples = _MAX_MMD_SAMPLES

    # Sub-sample to keep O(N²) memory manageable
    if nx > max_samples:
        x = x[np.random.default_rng(42).choice(nx, max_samples, replace=False)]
    if ny > max_samples:
        y = y[np.random.default_rng(42).choice(ny, max_samples, replace=False)]

    nx, ny = x.shape[0], y.shape[0]

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
# Reference vector loader
# ---------------------------------------------------------------------------

def load_chembl_6d_vectors(
    csv_path: str | Path | None = None,
    project_root: str | Path | None = None,
) -> np.ndarray:
    """Load the 6D descriptor vectors from a ChEMBL properties CSV.

    Results are cached in a module-level singleton — subsequent calls
    return the cached array instantly.

    Args:
        csv_path: Path to ``chembl_36_properties.csv.gz``.
                  If None, uses ``data/chembl/chembl_36_properties.csv.gz``
                  relative to *project_root*.
        project_root: Absolute project root directory.  If None, walks up
                      from this file's location.

    Returns:
        ``np.ndarray`` of shape ``[N, 6]`` with columns
        ``[LogP, TPSA, HBA, HBD, heteroatoms, carbon_fraction]``.
    """
    global _REFERENCE_CACHE
    if _REFERENCE_CACHE is not None:
        return _REFERENCE_CACHE

    if csv_path is None:
        root = Path(project_root) if project_root else _discover_project_root()
        csv_path = root / _DEFAULT_REFERENCE_CSV

    csv_path = Path(csv_path)
    _REFERENCE_CACHE = _load_6d_from_csv(csv_path)

    _log_once(f"[MolecularDescriptors] Loaded {len(_REFERENCE_CACHE):,} "
              f"reference 6D vectors from {csv_path}")
    return _REFERENCE_CACHE


def _discover_project_root() -> Path:
    """Walk up from this file to find the project root (where ``ddpo/`` lives)."""
    here = Path(__file__).resolve().parent
    for candidate in [here, here.parent, here.parent.parent]:
        if (candidate / "ddpo").is_dir():
            return candidate
    return here.parent  # fallback


_WARNED: set[str] = set()


def _log_once(msg: str) -> None:
    if msg not in _WARNED:
        _WARNED.add(msg)
        print(msg)


def _load_6d_from_csv(csv_path: Path) -> np.ndarray:
    """Read only the required 6 columns from a gzipped CSV with header."""
    if not csv_path.exists():
        warnings.warn(f"[MolecularDescriptors] Reference CSV not found: {csv_path}")
        return np.empty((0, 6), dtype=np.float64)

    try:
        with gzip.open(str(csv_path), "rt") as f:
            reader = csv.DictReader(f)
            vectors: list[list[float]] = []
            for row in reader:
                vectors.append([float(row[col]) for col in _REFERENCE_6D_COLUMNS])
        return np.array(vectors, dtype=np.float64)
    except Exception as exc:
        warnings.warn(f"[MolecularDescriptors] Failed to load reference CSV: {exc}")
        return np.empty((0, 6), dtype=np.float64)


def clear_reference_cache() -> None:
    """Clear the module-level reference cache (mostly for tests)."""
    global _REFERENCE_CACHE
    _REFERENCE_CACHE = None


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

    # Lazy import — avoids dragging the full reward stack on load.
    from ddpo.rewards.diversity_reward import DiversityReward  # noqa: E402

    fps = DiversityReward.compute_fingerprints(rdmols, use_features=use_features)
    bit_sets = [DiversityReward.fingerprint_on_bits(fp) for fp in fps]
    return DiversityReward.compute_mean_similarities_from_bit_sets(bit_sets)


def compute_molecular_descriptors(
    rdmols: list[Chem.Mol | None],
    valid_mask: list[bool],
    reference_descriptor_vectors: np.ndarray | list[list[float]] | None = None,
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
