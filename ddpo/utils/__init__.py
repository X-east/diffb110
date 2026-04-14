"""
Utility functions for DDPO training.

Canonical molecule utilities now live in `ddpo.utils.molecule_semantics`.
Legacy `molecule_converter` placeholders are no longer exposed here.
"""

# Install wandb stub before any DiffSBDD imports
import sys
from . import wandb_stub
if 'wandb' not in sys.modules:
    sys.modules['wandb'] = wandb_stub

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name == "DiffSBDDFilter":
        from .diffsbdd_filter import DiffSBDDFilter
        return DiffSBDDFilter
    elif name == "ValidityFilter":
        from .validity_filter import ValidityFilter
        return ValidityFilter
    elif name == "wandb_stub":
        return wandb_stub
    elif name == "logger":
        from . import logger
        return logger
    elif name == "TrainingLogger":
        from .logger import TrainingLogger
        return TrainingLogger
    elif name == "BatchRecorder":
        from .batch_recorder import BatchRecorder
        return BatchRecorder
    elif name == "annotate_batch_molecule_semantics":
        from .molecule_semantics import annotate_batch_molecule_semantics
        return annotate_batch_molecule_semantics
    elif name == "compute_sample_molecule_semantics":
        from .molecule_semantics import compute_sample_molecule_semantics
        return compute_sample_molecule_semantics
    elif name == "write_rdmol_to_pdbqt":
        from .molecule_semantics import write_rdmol_to_pdbqt
        return write_rdmol_to_pdbqt
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "DiffSBDDFilter",
    "ValidityFilter",
    "wandb_stub",
    "logger",
    "FileLogger",
    "setup_logger",
    "get_log_dir",
    "BatchRecorder",
    "annotate_batch_molecule_semantics",
    "compute_sample_molecule_semantics",
    "write_rdmol_to_pdbqt",
]
