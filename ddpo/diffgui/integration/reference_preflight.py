"""Preflight checks for loading the local reference DiffGui project."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path


DEFAULT_REFERENCE_ROOT = Path("reference project/DiffGui")
REQUIRED_MODULES = (
    "easydict",
    "torch_geometric",
    "torch_scatter",
    "rdkit",
    "scipy",
)
REQUIRED_FILES = (
    "models/model.py",
    "models/transition.py",
    "utils/transforms.py",
    "utils/sample_utils.py",
    "ckpt/trained.pt",
)


@dataclass(frozen=True)
class DiffGuiReferencePreflight:
    """Dependency and file-system readiness for reference DiffGui loading."""

    reference_root: Path
    module_available: dict[str, bool]
    file_available: dict[str, bool]

    @property
    def missing_modules(self) -> list[str]:
        """Return missing Python modules."""
        return [name for name, available in self.module_available.items() if not available]

    @property
    def missing_files(self) -> list[str]:
        """Return missing required reference files."""
        return [name for name, available in self.file_available.items() if not available]

    @property
    def ready(self) -> bool:
        """Return whether reference DiffGui can be imported and loaded."""
        return not self.missing_modules and not self.missing_files

    def error_message(self) -> str:
        """Build a concise actionable failure message."""
        parts: list[str] = []
        if self.missing_modules:
            parts.append(f"missing modules: {', '.join(self.missing_modules)}")
        if self.missing_files:
            parts.append(f"missing files: {', '.join(self.missing_files)}")
        if not parts:
            return "reference DiffGui preflight passed"
        return f"reference DiffGui preflight failed at {self.reference_root}: " + "; ".join(parts)


def check_reference_diffgui_preflight(
    reference_root: str | Path = DEFAULT_REFERENCE_ROOT,
) -> DiffGuiReferencePreflight:
    """Check local reference DiffGui dependencies without importing its model."""
    root = Path(reference_root)
    module_available = {name: importlib.util.find_spec(name) is not None for name in REQUIRED_MODULES}
    file_available = {name: (root / name).exists() for name in REQUIRED_FILES}
    return DiffGuiReferencePreflight(
        reference_root=root,
        module_available=module_available,
        file_available=file_available,
    )


def require_reference_diffgui_ready(
    reference_root: str | Path = DEFAULT_REFERENCE_ROOT,
) -> DiffGuiReferencePreflight:
    """Return preflight status or raise a clear RuntimeError."""
    status = check_reference_diffgui_preflight(reference_root)
    if not status.ready:
        raise RuntimeError(status.error_message())
    return status
