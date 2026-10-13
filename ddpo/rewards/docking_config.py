"""Docking configuration defaults.

All DEFAULT_* constants for DockingReward live here.
DockingReward imports from this module; no other file should re-define these.

These are defaults for the 8GJJ-C (RE-CmeB) receptor.
Override via YAML config; do not hard-code new values in DockingReward.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

# Primary backend for production training.
# Valid values: 'autodock_gpu' | 'unidock' | 'smina'
# See docking_registry.py for support-level policy.
DEFAULT_BACKEND: str = "autodock_gpu"

# ---------------------------------------------------------------------------
# File paths (relative to project root)
# ---------------------------------------------------------------------------

# Receptor PDBQT file for UniDock / Smina.
DEFAULT_RECEPTOR_FILE: str = "data/8GJJ-C-Cip.pdbqt"

# AutoGrid4 map file (produced by autogrid4).
# Generate with:
#   prepare_receptor4.py -r receptor.pdb -o receptor.pdbqt
#   prepare_gpf4.py -l ligand.pdbqt -r receptor.pdbqt -o receptor.gpf
#   autogrid4 -p receptor.gpf -l receptor.glg
DEFAULT_FLD_FILE: str = "data/8GJJ-C-Cip.maps.fld"

# ---------------------------------------------------------------------------
# GPU / search parameters
# ---------------------------------------------------------------------------

# 0-indexed GPU id (maps to AutoDock-GPU --devnum = gpu_id + 1).
DEFAULT_GPU_ID: int = 0

# UniDock search mode. Options: 'fast' | 'balance' | 'detail'.
DEFAULT_SEARCH_MODE: str = "fast"

# UniDock scoring function. Passed explicitly: the binary happens to default to
# ``vina``, but that default is not a contract and may change across versions.
# See docs/评估口径规范.md §3.1.
DEFAULT_SCORING: str = "vina"

# UniDock RNG seed. Uni-Dock defaults to 0; the benchmark standard is 42 so that
# the reward search and the evaluation search are reproducible and comparable.
DEFAULT_SEED: int = 42

# UniDock runtime controls, passed explicitly for protocol reproducibility.
DEFAULT_VERBOSITY: int = 1
DEFAULT_CPU: int = 0
DEFAULT_MAX_GPU_MEMORY: int = 0

# Subprocess timeout for one Uni-Dock batch call (previously hard-coded to 600 s).
DEFAULT_DOCKING_TIMEOUT_SECONDS: int = 3600

# AutoDock-GPU: number of LGA runs per ligand.
DEFAULT_NRUN: int = 20

# AutoDock-GPU: use adaptive nrun heuristic.
DEFAULT_HEURISTICS: bool = True

# ---------------------------------------------------------------------------
# Reward normalisation
# ---------------------------------------------------------------------------

# reward = docking_score / NORMALIZATION_FACTOR
# Typical scores: -12 (very good) to 0 (bad).
# Dividing by -10 maps this to roughly [0, 1.2].
NORMALIZATION_FACTOR: float = -10.0

# Reward given to molecules that fail to dock or fail canonical build.
# Use -1.0 so failures are distinguishable from 0-score successes.
FAILURE_PENALTY: float = -1.0

# ---------------------------------------------------------------------------
# Molecule preparation
# ---------------------------------------------------------------------------

# Hydrogens are mandatory: Vina needs *polar* hydrogens to type heavy atoms as
# hydrogen-bond donors.  Partial charges are NOT: "AutoDock Vina ignores the
# user-supplied partial charges" (https://vina.scripps.edu/manual/), so the
# Gasteiger step is a no-op for ``--scoring vina``.  Keep it off unless an
# ``ad4`` scoring path is introduced.  See docs/评估口径规范.md §二.4.
SANITIZE_MOLECULES: bool = True
ADD_HYDROGENS: bool = True
ASSIGN_GASTEIGER_CHARGES: bool = False

# ---------------------------------------------------------------------------
# Docking box parameters — 8GJJ-C-Cip (RE-CmeB)
# ---------------------------------------------------------------------------

DEFAULT_CENTER_X: float = 183.0  # X centre (Angstroms)
DEFAULT_CENTER_Y: float = 180.0  # Y centre (Angstroms)
DEFAULT_CENTER_Z: float = 230.0  # Z centre (Angstroms)
DEFAULT_SIZE_X: float = 25.0     # Box size X (Angstroms)
DEFAULT_SIZE_Y: float = 25.0     # Box size Y (Angstroms)
DEFAULT_SIZE_Z: float = 25.0     # Box size Z (Angstroms)
DEFAULT_EXHAUSTIVENESS: int = 8   # Search exhaustiveness

# ---------------------------------------------------------------------------
# Autobox parameters (Smina only)
# ---------------------------------------------------------------------------

# Reference ligand file for autobox; None = use manual box above.
DEFAULT_AUTOBOX_LIGAND: None = None

# Buffer space added around the autobox reference ligand (Angstroms).
DEFAULT_AUTOBOX_ADD: float = 4.0
