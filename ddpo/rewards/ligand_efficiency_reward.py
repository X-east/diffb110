"""LigandEfficiencyReward: reward based on Ligand Efficiency (LE).

LE = -docking_score / heavy_atom_count

Single responsibility: read *ligand_efficiency_raw* from batch (set by
:class:`DockingReward`), apply min/max clamping, write
``ligand_efficiency_reward``.

This reward depends on :class:`DockingReward` running first.
The raw LE values and per-epoch averages are already tracked via
:class:`DockingReward.field_specs` — this class only produces the
reward signal consumed by :class:`CompositeReward`.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "DiffSBDD"))

from ddpo.rewards.interfaces import BaseReward

if TYPE_CHECKING:
    from ddpo.runtime.protocol import DataProto


DEFAULT_FAILURE_PENALTY: float = -1.0
DEFAULT_MIN_LE: float = 0.0
DEFAULT_MAX_LE: float = 2.0
DEFAULT_SCALE: float = 1.0


class LigandEfficiencyReward(BaseReward):
    """Reward based on Ligand Efficiency: docking score per heavy atom.

    .. math::

        \\text{LE} = -\\frac{\\text{docking\\_score}}{\\text{heavy\\_atom\\_count}}

    Higher LE means stronger binding per atom.  Typical range: 0-1.5
    kcal/mol/atom (docking score = -12 → LE = 12/8 ≈ 1.5).

    This reward **assumes** :class:`DockingReward.calculate_rewards` has
    already populated ``batch['ligand_efficiency_raw']``.  If missing, all
    molecules receive ``failure_penalty``.

    Writes to batch::

        'ligand_efficiency_reward'  [B]  — clamped & scaled LE or failure_penalty
    """

    def __init__(
        self,
        failure_penalty: float = DEFAULT_FAILURE_PENALTY,
        min_le: float = DEFAULT_MIN_LE,
        max_le: float = DEFAULT_MAX_LE,
        scale: float = DEFAULT_SCALE,
    ) -> None:
        """
        Args:
            failure_penalty:
                Reward assigned when ``ligand_efficiency_raw`` is unavailable
                (e.g. docking backend failure or DockingReward not run).
            min_le:
                Lower clamp bound for raw LE values.
            max_le:
                Upper clamp bound for raw LE values.
            scale:
                Multiplier applied **after** clamping.
        """
        self.failure_penalty = failure_penalty
        self.min_le = min_le
        self.max_le = max_le
        self.scale = scale

    # ------------------------------------------------------------------
    # Reward calculation
    # ------------------------------------------------------------------

    def calculate_rewards(self, data: "DataProto") -> "DataProto":
        """Read ``ligand_efficiency_raw``, clamp, and produce reward.

        Expects :meth:`DockingReward.calculate_rewards` to have run first.

        Args:
            data:
                DataProto whose batch must contain
                ``ligand_efficiency_raw`` [B].

        Returns:
            DataProto with ``batch['ligand_efficiency_reward']`` added.
        """
        batch = data.batch
        batch_size = batch["x"].shape[0]
        device = batch["x"].device

        le_raw: torch.Tensor | None = batch.get("ligand_efficiency_raw")

        if le_raw is None:
            warnings.warn(
                "[LigandEfficiencyReward] ligand_efficiency_raw missing from batch. "
                "DockingReward must run before LigandEfficiencyReward. "
                "Returning failure_penalty for all molecules."
            )
            reward = torch.full((batch_size,), self.failure_penalty, device=device)
        else:
            le_clamped = torch.clamp(le_raw, self.min_le, self.max_le)
            reward = le_clamped * self.scale

        data.batch["ligand_efficiency_reward"] = reward
        return data

    def __repr__(self) -> str:
        return (
            f"LigandEfficiencyReward("
            f"failure_penalty={self.failure_penalty}, "
            f"min_le={self.min_le}, max_le={self.max_le}, "
            f"scale={self.scale})"
        )
