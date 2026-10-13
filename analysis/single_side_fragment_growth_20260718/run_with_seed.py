"""Run a Python entrypoint after setting deterministic random seeds."""

from __future__ import annotations

import random
import runpy
import sys
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    """Seed Python, NumPy, and PyTorch, then execute the requested script."""
    if len(sys.argv) < 3:
        raise SystemExit("usage: run_with_seed.py SEED SCRIPT [ARGS ...]")
    seed = int(sys.argv[1])
    script = Path(sys.argv[2]).resolve()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    sys.path.insert(0, str(script.parent))
    sys.argv = [str(script), *sys.argv[3:]]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
