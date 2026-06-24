"""Build symmetry specs from project YAML dictionaries."""

from __future__ import annotations

from typing import Any

from ddpo.constraints.symmetry.spec import SymmetrySpec


def _as_int_tuple(value: Any, field_name: str) -> tuple[int, ...]:
    """Convert a YAML sequence into an integer tuple."""
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        raise ValueError(f"symmetry.{field_name} must be a sequence of integers, got string")
    try:
        return tuple(int(item) for item in value)
    except TypeError as exc:
        raise ValueError(f"symmetry.{field_name} must be a sequence of integers") from exc


def _as_str_tuple(value: Any, default: tuple[str, ...]) -> tuple[str, ...]:
    """Convert a YAML string/list into a tuple of strings."""
    if value is None:
        return default
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def build_symmetry_spec(
    config: dict[str, Any] | None,
    *,
    n_dims: int = 3,
    atom_nf: int | None = None,
) -> SymmetrySpec | None:
    """Return an enabled :class:`SymmetrySpec` or ``None`` when disabled."""
    cfg = dict((config or {}).get("symmetry", {}) or {})
    if not bool(cfg.get("enabled", False)):
        return None

    logprob_cfg = dict(cfg.get("logprob", {}) or {})
    topology_cfg = dict(cfg.get("topology", {}) or {})
    return SymmetrySpec(
        enabled=True,
        mode=str(cfg.get("mode", "single_sidechain_copy")),
        r1_indices=_as_int_tuple(cfg.get("r1_indices"), "r1_indices"),
        r2_indices=_as_int_tuple(cfg.get("r2_indices"), "r2_indices"),
        copy_channels=_as_str_tuple(cfg.get("copy_channels"), ("h",)),
        coordinate_policy=str(cfg.get("coordinate_policy", "free")),
        copy_schedule=str(cfg.get("copy_schedule", "every_step")),
        topology_mode=str(topology_cfg.get("mode", cfg.get("topology_mode", "audit_only"))),
        n_dims=int(cfg.get("n_dims", n_dims)),
        atom_nf=atom_nf if cfg.get("atom_nf") is None else int(cfg["atom_nf"]),
        normalize_logprob_by_free_dims=bool(logprob_cfg.get("normalize_by_free_dims", True)),
    )

