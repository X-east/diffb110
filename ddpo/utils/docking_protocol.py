"""Docking protocol fingerprint.

Authority: ``docs/评估口径规范.md`` §3.5.

Every docking artefact carries a ``docking_protocol_id`` derived from the
parameters that actually change the numbers.  Two docking results may only be
placed in the same comparison table when their ids match.

The fingerprint covers:

* Uni-Dock binary version
* receptor PDBQT sha256
* box (centre + size)
* scoring function, search mode, ``num_modes``, ``seed``
* ligand preparation (fragment policy, hydrogens + ``addCoords``, charges)
* ligand coordinate frame (``centered_zero`` vs ``receptor_frame``)

The coordinate frame is included because it was measured to differ between the
200k collection line (centre-of-mass zeroed) and the generation-evaluation line
(translated back to the pocket centre) — see SOP §3.0.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

FRAME_CENTERED_ZERO = "centered_zero"
FRAME_RECEPTOR = "receptor_frame"

#: Ligand-preparation contract fixed by SOP §二.
DEFAULT_LIGAND_PREP: dict[str, Any] = {
    "fragment_policy": "largest_fragment",
    "add_hydrogens": True,
    "add_coords": True,
    "charges": "none",
}


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Streaming sha256 of a file; ``""`` when unreadable."""
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            while True:
                block = handle.read(chunk_size)
                if not block:
                    break
                digest.update(block)
    except OSError:
        return ""
    return digest.hexdigest()


def unidock_version(binary: str | Path = "/opt/unidock-1.1.3/bin/unidock") -> str:
    """Best-effort version string of the Uni-Dock binary."""
    try:
        result = subprocess.run(
            [str(binary), "--version"],
            capture_output=True, text=True, timeout=15, check=False,
        )
    except Exception:  # noqa: BLE001 - version probing must never be fatal
        return "unknown"
    text = (result.stdout or result.stderr or "").strip()
    return text.splitlines()[0] if text else "unknown"


def canonical_payload(
    *,
    receptor_pdbqt: str | Path,
    box: Mapping[str, float],
    scoring: str,
    search_mode: str | None,
    num_modes: int,
    seed: int,
    ligand_prep: Mapping[str, Any] | None = None,
    frame: str = FRAME_RECEPTOR,
    unidock_bin: str | Path = "/opt/unidock-1.1.3/bin/unidock",
) -> dict[str, Any]:
    """Deterministic payload that feeds :func:`compute_docking_protocol_id`."""
    prep = dict(DEFAULT_LIGAND_PREP)
    if ligand_prep:
        prep.update(ligand_prep)
    return {
        "unidock_version": unidock_version(unidock_bin),
        "receptor_sha256": sha256_file(receptor_pdbqt),
        "box": {key: box[key] for key in sorted(box)},
        "scoring": scoring,
        "search_mode": search_mode,
        "num_modes": int(num_modes),
        "seed": int(seed),
        "ligand_prep": prep,
        "ligand_frame": frame,
    }


def compute_docking_protocol_id(**kwargs: Any) -> str:
    """Stable 16-hex-char fingerprint of one docking configuration."""
    payload = canonical_payload(**kwargs)
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def describe_protocol(**kwargs: Any) -> dict[str, Any]:
    """Payload plus the derived id, for logging/manifests."""
    payload = canonical_payload(**kwargs)
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    payload["docking_protocol_id"] = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
    return payload


__all__ = [
    "FRAME_CENTERED_ZERO",
    "FRAME_RECEPTOR",
    "DEFAULT_LIGAND_PREP",
    "sha256_file",
    "unidock_version",
    "canonical_payload",
    "compute_docking_protocol_id",
    "describe_protocol",
]
