"""Canonical Uni-Dock output score parsing — single implementation.

Authority: ``docs/评估口径规范.md`` §3.3 / §3.3.1.

Why this module exists
----------------------
The previous implementations searched the whole output file for a loose
``ENERGY=`` pattern.  Two verified facts made that unsafe:

1. **Uni-Dock never writes ``REMARK VINA RESULT:``.**  Measured on 1300+ output
   files: ``'REMARK' in text`` is ``False``.  The loose ``ENERGY=`` pattern was
   therefore the *only* thing that ever matched, so one stray ``ENERGY=`` line
   poisons the whole batch.

2. The ``> <Uni-Dock RESULT>`` block is self-consistent even when Uni-Dock
   fails.  A failed run writes a physically impossible, huge positive energy::

       > <Uni-Dock RESULT>
       ENERGY=110590696.000  LOWER_BOUND=    0.000  UPPER_BOUND=    0.000
       INTER + INTRA:    110590696.000
       INTER:            110590696.000
       INTRA:                   0.000
       UNBOUND:                 0.000

   rc is 0 and the pose is written, so a consistency check alone cannot detect
   it — only the magnitude can.

Policy (SOP §3.3.1)
-------------------
Scores are **never deleted**.  Every score gets one of four statuses:

===============  ==========================================  ==================
status           condition                                   use
===============  ==========================================  ==================
``ok``           ``s <= 0``                                  primary median
``positive_no_binding``  ``0 < s <= SUSPECT_THRESHOLD``      kept, reported separately
``suspect_backend``      ``s > SUSPECT_THRESHOLD`` or        counted, never in mean
                         ``s < HARD_LOWER_BOUND``
``parse_error``  block/fields missing or inconsistent        counted as failure
===============  ==========================================  ==================

There is deliberately **no** ``-30 <= s <= 0`` validity window: measured data
shows legitimate positive (non-binding) results up to +44.9 in the generation
batch and up to +331 in the 200k run, so clamping at 0 would delete real
measurements (survivorship bias) and ``-30`` was never approached
(observed minima: -13.0 and -11.5).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

STATUS_OK = "ok"
STATUS_POSITIVE = "positive_no_binding"
STATUS_SUSPECT = "suspect_backend"
STATUS_PARSE_ERROR = "parse_error"

#: kcal/mol.  Generation batch: real positives end at +44.9, Uni-Dock failures
#: start at +2051 (a >40x gap).  200k batch: positives are continuous to +331.
SUSPECT_THRESHOLD = 100.0

#: Defensive only — never reached in 136k measured scores (min -13.9).
HARD_LOWER_BOUND = -25.0

_BLOCK_HEADER = re.compile(r"^>\s*<Uni-Dock RESULT>\s*$", re.MULTILINE)
_BLOCK_END = re.compile(r"\$\$\$\$")
_ENERGY = re.compile(r"^ENERGY\s*=\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", re.MULTILINE)
_INTER_INTRA = re.compile(r"^INTER\s*\+\s*INTRA\s*:\s*([-+]?\d+(?:\.\d+)?)", re.MULTILINE)
_INTER = re.compile(r"^INTER\s*:\s*([-+]?\d+(?:\.\d+)?)", re.MULTILINE)
_INTRA = re.compile(r"^INTRA\s*:\s*([-+]?\d+(?:\.\d+)?)", re.MULTILINE)
_UNBOUND = re.compile(r"^UNBOUND\s*:\s*([-+]?\d+(?:\.\d+)?)", re.MULTILINE)


@dataclass(frozen=True)
class UniDockScore:
    """Result of parsing one Uni-Dock output file."""

    energy: float | None
    status: str
    inter: float | None = None
    intra: float | None = None
    unbound: float | None = None
    block_found: bool = False
    reason: str = ""

    @property
    def is_ok(self) -> bool:
        """True only for ``s <= 0`` — the primary-median subset."""
        return self.status == STATUS_OK

    @property
    def is_usable(self) -> bool:
        """True when the value is a physically interpretable affinity."""
        return self.status in (STATUS_OK, STATUS_POSITIVE)

    def as_legacy_float(self) -> float | None:
        """Float accessor for legacy callers.

        Returns ``None`` for ``suspect_backend`` / ``parse_error`` so that
        existing ``score is not None`` failure handling keeps working, while
        genuine positive (non-binding) results are still propagated.
        """
        return self.energy if self.is_usable else None


def _num(pattern: re.Pattern[str], body: str) -> float | None:
    match = pattern.search(body)
    if match is None:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _block_body(text: str, header_end: int, max_chars: int = 800) -> str:
    tail = text[header_end:header_end + max_chars]
    stop = _BLOCK_END.search(tail)
    return tail[: stop.start()] if stop else tail


def classify_energy(energy: float) -> tuple[str, str]:
    """Map a parsed energy to (status, reason)."""
    if energy > SUSPECT_THRESHOLD:
        return STATUS_SUSPECT, "energy_above_threshold"
    if energy < HARD_LOWER_BOUND:
        return STATUS_SUSPECT, "energy_below_hard_lower_bound"
    if energy > 0.0:
        return STATUS_POSITIVE, "non_binding_positive_score"
    return STATUS_OK, ""


def parse_unidock_output(path: str | Path) -> UniDockScore:
    """Parse one Uni-Dock output file (SDF or PDBQT) into a :class:`UniDockScore`."""
    target = Path(path)
    try:
        text = target.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:  # noqa: BLE001 - any read failure is a parse error
        return UniDockScore(
            None, STATUS_PARSE_ERROR, block_found=False,
            reason=f"read_error:{type(exc).__name__}",
        )

    header = _BLOCK_HEADER.search(text)
    if header is None:
        return UniDockScore(
            None, STATUS_PARSE_ERROR, block_found=False,
            reason="missing_uni_dock_result_block",
        )

    body = _block_body(text, header.end())
    energy = _num(_ENERGY, body)
    inter_intra = _num(_INTER_INTRA, body)
    inter = _num(_INTER, body)
    intra = _num(_INTRA, body)
    unbound = _num(_UNBOUND, body)

    if energy is None:
        return UniDockScore(
            None, STATUS_PARSE_ERROR, inter, intra, unbound,
            block_found=True, reason="missing_energy_field",
        )
    if inter is None or intra is None:
        return UniDockScore(
            energy, STATUS_PARSE_ERROR, inter, intra, unbound,
            block_found=True, reason="missing_inter_or_intra_field",
        )

    tolerance = max(1e-3, 1e-3 * abs(energy))
    if inter_intra is not None and abs(energy - inter_intra) > tolerance:
        return UniDockScore(
            energy, STATUS_PARSE_ERROR, inter, intra, unbound,
            block_found=True, reason="energy_inconsistent_with_inter_plus_intra",
        )
    if abs(energy - (inter + intra)) > tolerance:
        return UniDockScore(
            energy, STATUS_PARSE_ERROR, inter, intra, unbound,
            block_found=True, reason="energy_inconsistent_with_inter_plus_intra",
        )

    status, reason = classify_energy(energy)
    return UniDockScore(
        energy, status, inter, intra, unbound, block_found=True, reason=reason,
    )


def parse_score_float(path: str | Path) -> float | None:
    """Backward-compatible helper: usable affinity or ``None``."""
    return parse_unidock_output(path).as_legacy_float()


__all__ = [
    "STATUS_OK",
    "STATUS_POSITIVE",
    "STATUS_SUSPECT",
    "STATUS_PARSE_ERROR",
    "SUSPECT_THRESHOLD",
    "HARD_LOWER_BOUND",
    "UniDockScore",
    "classify_energy",
    "parse_unidock_output",
    "parse_score_float",
]
