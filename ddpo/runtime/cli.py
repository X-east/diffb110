from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for DiffSBDD training entrypoints."""
    parser = argparse.ArgumentParser(
        description="DiffSBDD + RLPF RL training entrypoint"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Optional output directory override.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed override.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reserved for future checkpoint resume support.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Reserved for future checkpoint resume support.",
    )
    return parser.parse_args()
