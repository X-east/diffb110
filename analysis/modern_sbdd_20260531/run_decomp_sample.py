from __future__ import annotations

import argparse
import os
import runpy
import sys
import types
from pathlib import Path


def _inject_local_packages(repo: Path) -> None:
    for name in ("utils", "datasets", "models"):
        module = types.ModuleType(name)
        module.__path__ = [str(repo / name)]
        sys.modules[name] = module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("sample_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    _inject_local_packages(repo)
    sys.argv = [str(repo / "scripts" / "sample_diffusion_decomp_compose.py")] + args.sample_args
    runpy.run_path(str(repo / "scripts" / "sample_diffusion_decomp_compose.py"), run_name="__main__")


if __name__ == "__main__":
    main()
