"""Runtime compatibility aliases for legacy KGDiff code."""

import numpy as np

if not hasattr(np, "long"):
    np.long = int

if not hasattr(np, "int"):
    np.int = int

if not hasattr(np, "bool"):
    np.bool = bool
