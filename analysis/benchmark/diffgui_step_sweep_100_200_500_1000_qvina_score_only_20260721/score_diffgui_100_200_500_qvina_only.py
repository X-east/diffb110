"""Run QVina score-only for the DiffGui 100/200/500-step sweeps."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


STEP_LABELS = ["100 steps", "200 steps", "500 steps"]
REPEATS = [
    ("rep1", "seed1_7"),
    ("rep2", "seed8_14"),
    ("rep3", "seed15_21"),
]


def _load_qvina_module(project_root: Path) -> ModuleType:
    """Load the validated five-model QVina score-only implementation."""
    source = (
        project_root
        / "analysis/benchmark/benchmark5"
        / "8gjl_c1101_benchmark_20260706_1000steps_3rep_formal"
        / "qvina_score_only_20260721/score_qvina_only.py"
    )
    spec = importlib.util.spec_from_file_location("validated_qvina_score_only", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load QVina scorer from {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _build_sources(module: ModuleType, project_root: Path) -> list[object]:
    """Build the exact three-repeat inputs used by the four-step analysis."""
    root_100 = project_root / "analysis/benchmark/diffgui_step_sweep_100_1000attempts_20260709"
    root_200_500 = (
        project_root
        / "analysis/benchmark/diffgui_step_sweep_500_200_1000attempts_20260707_run2"
    )
    roots = {100: root_100, 200: root_200_500, 500: root_200_500}
    sources = []
    for steps in (100, 200, 500):
        for repeat_index, (repeat, seed_group) in enumerate(REPEATS, start=1):
            sources.append(
                module.SourceSpec(
                    repeat=repeat,
                    model=f"{steps} steps",
                    sdf_path=(
                        roots[steps]
                        / f"{steps}steps_rep{repeat_index}_{seed_group}"
                        / "filtered_sanitize_valid/diffgui.sanitize_valid.sdf"
                    ),
                    generation_attempts=1000,
                )
            )
    sources.append(
        module.SourceSpec(
            repeat="reference",
            model="C1101",
            sdf_path=(
                project_root
                / "analysis/benchmark/8gjl_c1101_benchmark_20260624_1000mol_500steps"
                / "inputs/8GJL_C1101_ligand.sdf"
            ),
            generation_attempts=1,
            is_reference=True,
        )
    )
    return sources


def main() -> None:
    """Delegate scoring to the validated implementation with step-sweep inputs."""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=script_dir / "results_100_200_500")
    parser.add_argument("--receptor", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=18)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    module = _load_qvina_module(project_root)
    module.MODEL_ORDER = list(STEP_LABELS)
    module.build_source_specs = lambda _benchmark_root, _project_root: _build_sources(
        module, project_root
    )

    delegated_args = [
        str(Path(module.__file__).resolve()),
        "--output-dir",
        str(args.output_dir),
        "--receptor",
        str(args.receptor),
        "--workers",
        str(args.workers),
        "--center-x",
        "207.4519",
        "--center-y",
        "212.5211",
        "--center-z",
        "172.9314",
        "--size",
        "20.0",
        "--exhaustiveness",
        "16",
    ]
    if args.force:
        delegated_args.append("--force")
    sys.argv = delegated_args
    module.main()


if __name__ == "__main__":
    main()
