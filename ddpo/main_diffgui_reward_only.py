"""CLI entrypoint for DiffGui reward-only stage-1 evaluation."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIFFSBDD_ROOT = PROJECT_ROOT / "DiffSBDD"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(DIFFSBDD_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFSBDD_ROOT))

from constants import dataset_params  # noqa: E402
from ddpo.diffgui.runtime.reward_only import run_reward_only_evaluation  # noqa: E402
from ddpo.utils.suppress_warnings import setup_chemistry_logging  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="YAML config path.")
    parser.add_argument("--input-sdf", type=Path, default=None, help="Override DiffGui input SDF.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output directory.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override reward batch size.")
    parser.add_argument("--device", type=str, default=None, help="Override DataProto device.")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    """Load a YAML config file."""
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Apply CLI overrides to the DiffGui reward-only config section."""
    eval_cfg = config.setdefault("diffgui_reward_only", {})
    if args.input_sdf is not None:
        eval_cfg["input_sdf"] = str(args.input_sdf)
    if args.output_dir is not None:
        eval_cfg["output_dir"] = str(args.output_dir)
    if args.batch_size is not None:
        eval_cfg["batch_size"] = int(args.batch_size)
    if args.device is not None:
        eval_cfg["device"] = str(args.device)
    return config


def main() -> None:
    """Run reward-only evaluation and print the summary JSON."""
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    setup_chemistry_logging()
    config = apply_overrides(load_config(args.config), args)
    dataset_name = str(config.get("dataset", "crossdock"))
    if dataset_name not in dataset_params:
        raise KeyError(f"Unknown dataset {dataset_name!r}; available: {sorted(dataset_params)}")
    summary = run_reward_only_evaluation(
        config,
        project_root=PROJECT_ROOT,
        dataset_info=dataset_params[dataset_name],
        logger_obj=logging.getLogger("DiffGuiRewardOnly"),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

