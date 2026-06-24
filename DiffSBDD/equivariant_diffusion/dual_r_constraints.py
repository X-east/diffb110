"""Lightweight paired-slot constraints for dual-R inpainting prototypes.

Version 2: adds target atom types (not just symmetry) and extends distance
constraints beyond the root pair.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DualRConstraintConfig:
    """Configuration for paired atom-type and root-distance constraints.

    v2 changes:
      - ``target_atom_types``: per-pair expected atom type indices.
        When non-empty, atom features are lerped toward the target one-hot
        instead of toward the mean of R1[i] and R2[i].  An empty tuple
        preserves the legacy symmetric-sync behaviour.
      - ``distance_all_pairs``: when True, apply distance nudges to ALL
        paired atoms (from their respective core), with decaying strength
        for deeper pairs.  Default True.
    """

    enabled: bool = False
    fixed_atom_count: int = 0
    pair_count: int = 0
    left_core_idx: int = 0
    right_core_idx: int = 0
    root_pair_index: int = 0
    target_bond_length: float = 1.50
    atom_sync_strength: float = 1.0
    distance_strength: float = 0.25
    max_distance_step: float = 0.25
    apply_every_n_steps: int = 1
    apply_final: bool = True
    target_atom_types: tuple[int, ...] = ()
    distance_all_pairs: bool = True

    _atom_nf: int = field(default=0, repr=False)
    # cached one-hot targets (populated by build_target_one_hots)
    _target_one_hots: torch.Tensor | object = field(
        default_factory=lambda: torch.empty(0), repr=False,
    )

    def __post_init__(self) -> None:
        """Validate static paired-slot metadata."""
        if self.fixed_atom_count < 0:
            raise ValueError("fixed_atom_count must be non-negative")
        if self.pair_count < 0:
            raise ValueError("pair_count must be non-negative")
        if self.left_core_idx < 0 or self.right_core_idx < 0:
            raise ValueError("core atom indices must be non-negative")
        if self.root_pair_index < 0:
            raise ValueError("root_pair_index must be non-negative")
        if self.root_pair_index >= max(1, self.pair_count):
            raise ValueError("root_pair_index must be smaller than pair_count")
        if self.target_bond_length <= 0:
            raise ValueError("target_bond_length must be positive")
        if not 0 <= self.atom_sync_strength <= 1:
            raise ValueError("atom_sync_strength must be in [0, 1]")
        if not 0 <= self.distance_strength <= 1:
            raise ValueError("distance_strength must be in [0, 1]")
        if self.max_distance_step < 0:
            raise ValueError("max_distance_step must be non-negative")
        if self.apply_every_n_steps < 1:
            raise ValueError("apply_every_n_steps must be >= 1")
        if self.target_atom_types and len(self.target_atom_types) != self.pair_count:
            raise ValueError(
                f"target_atom_types length ({len(self.target_atom_types)}) "
                f"must equal pair_count ({self.pair_count}) or be empty"
            )

    @property
    def active(self) -> bool:
        """Return whether constraints should be applied."""
        return bool(self.enabled and self.pair_count > 0)

    @property
    def has_target_types(self) -> bool:
        """Return whether explicit target atom types are configured."""
        return bool(self.target_atom_types and len(self.target_atom_types) > 0)

    def build_target_one_hots(self, atom_nf: int) -> DualRConstraintConfig:
        """Return a config with cached one-hot target tensors on CPU."""
        if not self.has_target_types or atom_nf <= 0:
            return self
        one_hots = torch.zeros(self.pair_count, atom_nf)
        for i, t in enumerate(self.target_atom_types):
            if 0 <= t < atom_nf:
                one_hots[i, t] = 1.0
        # dataclass is frozen, so use object.__setattr__
        object.__setattr__(self, "_atom_nf", atom_nf)
        object.__setattr__(self, "_target_one_hots", one_hots)
        return self


def should_apply_dual_r_constraints(
    config: DualRConstraintConfig | None,
    *,
    step: int,
) -> bool:
    """Return whether a denoising step should receive dual-R constraints."""
    if config is None or not config.active:
        return False
    return int(step) % int(config.apply_every_n_steps) == 0


def apply_dual_r_latent_constraints(
    z_lig: torch.Tensor,
    lig_mask: torch.Tensor,
    config: DualRConstraintConfig | None,
    *,
    n_dims: int,
    coordinate_scale: float = 1.0,
) -> torch.Tensor:
    """Apply paired atom-type and root-distance constraints to ligand latent state."""
    if config is None or not config.active:
        return z_lig
    constrained = z_lig
    constrained = sync_paired_atom_features(
        constrained,
        lig_mask,
        config,
        n_dims=n_dims,
    )
    constrained = nudge_root_distances(
        constrained,
        lig_mask,
        config,
        n_dims=n_dims,
        coordinate_scale=coordinate_scale,
    )
    return constrained


def sync_paired_atom_features(
    z_lig: torch.Tensor,
    lig_mask: torch.Tensor,
    config: DualRConstraintConfig,
    *,
    n_dims: int,
) -> torch.Tensor:
    """Synchronize h/features for paired R1/R2 generated slots.

    When ``config.target_atom_types`` is non-empty, lerps each atom toward
    the target one-hot.  Otherwise falls back to legacy symmetric mean.
    """
    if config.atom_sync_strength <= 0:
        return z_lig
    out = z_lig.clone()
    strength = float(config.atom_sync_strength)
    use_targets = config.has_target_types and config._atom_nf > 0
    if use_targets:
        targets = config._target_one_hots  # (pair_count, atom_nf)

    for atom_indices in _sample_atom_indices(lig_mask):
        for pair_idx in range(config.pair_count):
            left_local, right_local = _paired_slot_local_indices(config, pair_idx)
            if not _locals_in_sample(atom_indices, left_local, right_local):
                continue
            left_idx = int(atom_indices[left_local].item())
            right_idx = int(atom_indices[right_local].item())

            if use_targets:
                target_h = targets[pair_idx].to(out.device)
                left_h = out[left_idx, n_dims:].clone()
                right_h = out[right_idx, n_dims:].clone()
                out[left_idx, n_dims:] = torch.lerp(left_h, target_h, strength)
                out[right_idx, n_dims:] = torch.lerp(right_h, target_h, strength)
            else:
                left_h = out[left_idx, n_dims:].clone()
                right_h = out[right_idx, n_dims:].clone()
                mean_h = 0.5 * (left_h + right_h)
                out[left_idx, n_dims:] = torch.lerp(left_h, mean_h, strength)
                out[right_idx, n_dims:] = torch.lerp(right_h, mean_h, strength)
    return out


def nudge_root_distances(
    z_lig: torch.Tensor,
    lig_mask: torch.Tensor,
    config: DualRConstraintConfig,
    *,
    n_dims: int,
    coordinate_scale: float = 1.0,
) -> torch.Tensor:
    """Nudge paired atoms toward target distances from their core.

    When ``config.distance_all_pairs`` is True, applies to every pair with
    decaying strength and growing target distance for deeper positions.
    Otherwise only the root pair (pair_index=0) is nudged.
    """
    if config.distance_strength <= 0 or config.max_distance_step <= 0:
        return z_lig
    out = z_lig.clone()
    base_target = float(config.target_bond_length) / float(coordinate_scale)
    base_max_step = float(config.max_distance_step) / float(coordinate_scale)
    base_strength = float(config.distance_strength)

    pair_range = range(config.pair_count) if config.distance_all_pairs else [config.root_pair_index]

    for atom_indices in _sample_atom_indices(lig_mask):
        if not _locals_in_sample(
            atom_indices,
            config.left_core_idx,
            config.right_core_idx,
        ):
            continue
        left_core = int(atom_indices[config.left_core_idx].item())
        right_core = int(atom_indices[config.right_core_idx].item())

        for pair_idx in pair_range:
            left_local, right_local = _paired_slot_local_indices(config, pair_idx)
            if not _locals_in_sample(atom_indices, left_local, right_local):
                continue

            # Decay: deeper pairs get larger target distance (chain length grows)
            # and weaker nudge strength.
            multiplier = float(pair_idx + 1)
            decay = 0.7 ** pair_idx
            strength = base_strength * decay
            target_dist = base_target * multiplier
            max_step = base_max_step * multiplier * decay

            _nudge_one_root(
                out,
                core_idx=left_core,
                root_idx=int(atom_indices[left_local].item()),
                target_distance=target_dist,
                strength=strength,
                max_step=max_step,
                n_dims=n_dims,
            )
            _nudge_one_root(
                out,
                core_idx=right_core,
                root_idx=int(atom_indices[right_local].item()),
                target_distance=target_dist,
                strength=strength,
                max_step=max_step,
                n_dims=n_dims,
            )
    return out


def _nudge_one_root(
    values: torch.Tensor,
    *,
    core_idx: int,
    root_idx: int,
    target_distance: float,
    strength: float,
    max_step: float,
    n_dims: int,
) -> None:
    """Move one root atom toward a target distance from its fixed core."""
    core_pos = values[core_idx, :n_dims]
    root_pos = values[root_idx, :n_dims]
    vector = root_pos - core_pos
    distance = torch.linalg.norm(vector).clamp_min(1e-8)
    direction = vector / distance
    step_length = (target_distance - distance) * float(strength)
    step_length = step_length.clamp(min=-float(max_step), max=float(max_step))
    values[root_idx, :n_dims] = root_pos + direction * step_length


def _paired_slot_local_indices(
    config: DualRConstraintConfig,
    pair_idx: int,
) -> tuple[int, int]:
    """Return local atom indices for one R1/R2 slot pair."""
    left_local = int(config.fixed_atom_count) + int(pair_idx)
    right_local = int(config.fixed_atom_count) + int(config.pair_count) + int(pair_idx)
    return left_local, right_local


def _sample_atom_indices(lig_mask: torch.Tensor) -> list[torch.Tensor]:
    """Return per-sample atom indices preserving original order."""
    sample_ids = torch.unique(lig_mask.detach().cpu(), sorted=True)
    return [torch.nonzero(lig_mask == sample_id.to(lig_mask.device), as_tuple=False).view(-1) for sample_id in sample_ids]


def _locals_in_sample(atom_indices: torch.Tensor, *local_indices: int) -> bool:
    """Return whether all local indices exist in one molecule."""
    n_atoms = int(atom_indices.numel())
    return all(0 <= int(local_idx) < n_atoms for local_idx in local_indices)
