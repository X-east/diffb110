"""
Config Python package for DiffB110.

Only keeps dataset_info used by DiffSBDD subproject.
Launch YAMLs and run scripts have moved to launch/.
"""

from .dataset_info import DIFFSBDD_DATASET_INFO

__all__ = [
    "DIFFSBDD_DATASET_INFO",
]
