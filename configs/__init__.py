"""
Configuration module for DDPO training.

Contains:
- dataset_info: DiffSBDD crossdock dataset parameters
- Training configuration utilities
- Default config file paths
"""

import os
from pathlib import Path

from .dataset_info import DIFFSBDD_DATASET_INFO

# Get the directory containing this file
_CONFIG_DIR = Path(__file__).parent

# Default configuration file path
DEFAULT_CONFIG_PATH = _CONFIG_DIR / "diffsbdd_ddpo.yaml"

__all__ = [
    "DIFFSBDD_DATASET_INFO",
    "DEFAULT_CONFIG_PATH",
]
