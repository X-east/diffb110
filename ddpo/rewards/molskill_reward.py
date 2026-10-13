"""MolSkillReward: reward based on MolSkill human preference model.

Wraps Microsoft Research's MolSkill RankNet (Nature Communications, 2023),
which was trained on human medicinal chemists' pairwise preference judgments.
Bypasses :class:`pytorch_lightning.Trainer` to avoid conflicts with DDP
training — loads the checkpoint and runs pure forward-pass inference.

Usage:
    from ddpo.rewards.molskill_reward import MolSkillReward
    reward = MolSkillReward(failure_penalty=-1.0, clamp_min=-5.0, clamp_max=5.0)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

# ── Add molskill to Python path ──────────────────────────────────────────
_MOLSKILL_ROOT = Path(__file__).resolve().parent.parent.parent / "reference project" / "代码" / "molskill"
if str(_MOLSKILL_ROOT) not in sys.path:
    sys.path.insert(0, str(_MOLSKILL_ROOT))

from molskill.data.featurizers import get_featurizer, Featurizer  # noqa: E402
from molskill.models.ranknet import LitRankNet  # noqa: E402

from ddpo.field_registry import FieldSpec  # noqa: E402
from ddpo.rewards.interfaces import BaseReward  # noqa: E402

if TYPE_CHECKING:
    from ddpo.runtime.protocol import DataProto

logger = logging.getLogger(__name__)

# ── Defaults ─────────────────────────────────────────────────────────────
DEFAULT_FAILURE_PENALTY: float = -1.0
DEFAULT_CLAMP_MIN: float = -5.0
DEFAULT_CLAMP_MAX: float = 5.0
_DEFAULT_CHECKPOINT_PATH: str = str(
    _MOLSKILL_ROOT / "models" / "default" / "checkpoints" / "last.ckpt"
)


class MolSkillReward(BaseReward):
    """Reward based on MolSkill human preference model.

    Loads the pretrained LitRankNet checkpoint and uses it for single-molecule
    scoring.  Features are extracted with the default ``morgan_count_rdkit_2d``
    featurizer (2048-bit Morgan count fingerprints + ~172 RDKit 2D descriptors).

    Inference runs in pure PyTorch forward mode — no Lightning Trainer is
    created, avoiding nesting / DDP conflicts.

    Writes to batch::

        'molskill_raw'      [B]  — raw MolSkill score (unclamped)
        'molskill_reward'   [B]  — clamped score or failure_penalty
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        failure_penalty: float = DEFAULT_FAILURE_PENALTY,
        clamp_min: float = DEFAULT_CLAMP_MIN,
        clamp_max: float = DEFAULT_CLAMP_MAX,
    ) -> None:
        """
        Args:
            checkpoint_path:
                Path to LitRankNet ``.ckpt`` file.  Defaults to the figshare
                checkpoint shipped with the MolSkill reference project.
            failure_penalty:
                Reward assigned to molecules that cannot be featurized
                (``rdmol`` is ``None`` or featurization raises).
            clamp_min:
                Lower bound applied to raw scores before writing
                ``molskill_reward``.
            clamp_max:
                Upper bound applied to raw scores before writing
                ``molskill_reward``.
        """
        self.failure_penalty = failure_penalty
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

        if checkpoint_path is None:
            checkpoint_path = _DEFAULT_CHECKPOINT_PATH

        self._checkpoint_path = checkpoint_path
        self._model: LitRankNet | None = None
        self._device: torch.device | None = None

        # ── Build featurizer (same as MolSkill default) ─────────────────
        self._featurizer: Featurizer = get_featurizer("morgan_count_rdkit_2d")

    # ──────────────────────────────────────────────────────────────────
    #  Lazy-loading (avoids loading model on module import)
    # ──────────────────────────────────────────────────────────────────

    def _ensure_model_loaded(self, device: torch.device) -> None:
        """Lazy-load the MolSkill LitRankNet checkpoint onto *device*."""
        if self._model is not None and self._device == device:
            return

        logger.info(
            "Loading MolSkill checkpoint: %s  (device=%s)",
            self._checkpoint_path,
            device,
        )
        self._model = LitRankNet.load_from_checkpoint(
            checkpoint_path=self._checkpoint_path,
            input_size=self._featurizer.dim(),
            map_location=device,
        )
        self._model.eval()
        self._model.to(device)
        self._device = device

    # ──────────────────────────────────────────────────────────────────
    #  Reward calculation
    # ──────────────────────────────────────────────────────────────────

    def calculate_rewards(self, data: "DataProto") -> "DataProto":
        """Calculate MolSkill scores for a batch.

        Uses pre-built ``rdmols`` from ``data.non_tensor_batch`` (populated by
        :func:`annotate_batch_molecule_semantics` upstream).  Each valid RDKit
        molecule is featurized, and the RankNet forward pass produces a raw
        score which is clamped into ``[clamp_min, clamp_max]``.

        Args:
            data:
                DataProto with ``non_tensor_batch['rdmols']`` and
                ``non_tensor_batch['rdkit_valid']`` set.

        Returns:
            DataProto with ``batch['molskill_raw']`` and
            ``batch['molskill_reward']`` tensors [B] added.
        """
        batch = data.batch
        ntb = data.non_tensor_batch if hasattr(data, "non_tensor_batch") else {}
        device = batch["x"].device
        batch_size = batch["x"].shape[0]

        rdkit_valid = ntb.get("rdkit_valid")
        rdmols = ntb.get("rdmols")

        if rdmols is None or rdkit_valid is None:
            raise ValueError(
                "[MolSkillReward] rdmols/rdkit_valid missing from non_tensor_batch. "
                "annotate_batch_molecule_semantics() must be called before calculate_rewards()."
            )

        # ── Init output tensors ────────────────────────────────────────
        molskill_raw = torch.zeros(batch_size, device=device)
        molskill_reward = torch.full(
            (batch_size,), self.failure_penalty, device=device
        )

        # ── Featurize valid molecules ──────────────────────────────────
        features_list: list[torch.Tensor] = []
        valid_indices: list[int] = []

        for i in range(batch_size):
            if int(rdkit_valid[i]) == 0 or rdmols[i] is None:
                continue

            mol = rdmols[i]
            try:
                feat = torch.from_numpy(
                    self._featurizer.get_feat(mol).astype(np.float32)
                )
                features_list.append(feat)
                valid_indices.append(i)
            except Exception:
                logger.debug("Featurization failed for molecule %d", i, exc_info=True)

        if not features_list:
            data.batch["molskill_raw"] = molskill_raw
            data.batch["molskill_reward"] = molskill_reward
            return data

        # ── Stack & run inference ──────────────────────────────────────
        self._ensure_model_loaded(device)
        assert self._model is not None

        features = torch.stack(features_list).to(device)

        with torch.inference_mode():
            raw_scores: torch.Tensor = self._model.net.score(features).squeeze(-1)

        clamped = torch.clamp(raw_scores, self.clamp_min, self.clamp_max)

        # ── Scatter back ───────────────────────────────────────────────
        for idx, (batch_idx, raw, cl) in enumerate(
            zip(valid_indices, raw_scores, clamped)
        ):
            molskill_raw[batch_idx] = raw
            molskill_reward[batch_idx] = cl

        data.batch["molskill_raw"] = molskill_raw
        data.batch["molskill_reward"] = molskill_reward

        return data

    # ──────────────────────────────────────────────────────────────────
    #  Dynamic field registration (CSV columns)
    # ──────────────────────────────────────────────────────────────────

    def field_specs(self) -> list[FieldSpec]:
        """Declare molecule-level raw/reward fields and epoch-level average."""
        return [
            FieldSpec(
                name="molskill_raw",
                scope="molecule",
                source="batch",
                source_key="molskill_raw",
                reduce="none",
                order=500,
            ),
            FieldSpec(
                name="molskill_reward",
                scope="molecule",
                source="batch",
                source_key="molskill_reward",
                reduce="none",
                order=501,
            ),
            FieldSpec(
                name="avg_molskill_reward",
                scope="epoch",
                source="batch",
                source_key="molskill_reward",
                reduce="generated_mean",
                order=500,
            ),
        ]

    # ──────────────────────────────────────────────────────────────────
    #  Debug representation
    # ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"MolSkillReward("
            f"failure_penalty={self.failure_penalty}, "
            f"clamp=[{self.clamp_min}, {self.clamp_max}])"
        )
