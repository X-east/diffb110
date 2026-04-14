"""Utilities to suppress and route chemistry-library warnings.

Handles both Python-level warnings and noisy third-party stderr output.
"""

from __future__ import annotations

import logging
import os
import warnings
from contextlib import contextmanager
from pathlib import Path

from ddpo.utils.log_control import get_chem_warnings_log_path


def setup_chemistry_logging() -> None:
    """Configure baseline suppression for chemistry-related warnings."""
    logging.getLogger("rdkit").setLevel(logging.ERROR)
    logging.getLogger("rdkit.Chem").setLevel(logging.ERROR)
    logging.getLogger("rdkit.Chem.AllChem").setLevel(logging.ERROR)
    logging.getLogger("openbabel").setLevel(logging.ERROR)

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"

    # Prefer RDKit-native logger controls when available.
    try:
        from rdkit import RDLogger  # pylint: disable=import-outside-toplevel

        RDLogger.DisableLog("rdApp.warning")
        RDLogger.DisableLog("rdApp.info")
    except Exception:
        pass


@contextmanager
def redirect_stderr_to_log(log_path: str | None = None):
    """Redirect process stderr to a file within this context.

    This catches many C/C++ extension writes that bypass Python logging.
    """
    target_path = Path(log_path or get_chem_warnings_log_path())
    target_path.parent.mkdir(parents=True, exist_ok=True)

    with open(target_path, "a", encoding="utf-8") as sink:
        sink.write("\n===== chemistry stderr capture begin =====\n")
        sink.flush()

        saved_fd = os.dup(2)
        try:
            os.dup2(sink.fileno(), 2)
            yield
        finally:
            os.dup2(saved_fd, 2)
            os.close(saved_fd)
            sink.write("\n===== chemistry stderr capture end =====\n")
            sink.flush()


@contextmanager
def suppress_all_warnings():
    """Suppress Python warnings and redirect stderr to chemistry log."""
    old_filters = warnings.filters[:]
    warnings.filterwarnings("ignore")
    try:
        with redirect_stderr_to_log():
            yield
    finally:
        warnings.filters[:] = old_filters
