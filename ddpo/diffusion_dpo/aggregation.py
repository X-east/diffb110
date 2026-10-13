"""Pure utilities for aggregating per-transition preference scores."""

from __future__ import annotations

from typing import Literal

import torch


Aggregation = Literal["mean", "sum"]


def aggregate_transition_scores(
    scores: list[torch.Tensor], aggregation: Aggregation
) -> torch.Tensor:
    """Aggregate ``[batch, 2]`` timestep scores without changing pair order."""
    if not scores:
        raise ValueError("at least one timestep score is required")
    if any(score.ndim != 2 or score.shape[1] != 2 for score in scores):
        raise ValueError("each transition score must have shape [batch, 2]")
    stacked = torch.stack(scores, dim=0)
    if aggregation == "mean":
        return stacked.mean(dim=0)
    if aggregation == "sum":
        return stacked.sum(dim=0)
    raise ValueError(f"unsupported score aggregation: {aggregation}")
