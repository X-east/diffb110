"""Runtime compatibility shims for unmodified DiffSMol sampling commands."""

import numpy as np

if not hasattr(np, "int"):
    np.int = int
