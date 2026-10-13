"""DiffSBDD-native Gaussian diffusion preference optimization."""

from .aggregation import aggregate_transition_scores
from .dpo_loss import DPOFacts, compute_dpo_loss
from .logprob import TransitionScore, gaussian_transition_score
from .pairs import (
    PreferencePair,
    build_exact_atom_count_pairs,
    load_pair_manifest,
    load_selected_manifest,
    validate_pair_manifests,
)
from .trajectory import (
    ReverseTransition,
    TrajectoryProtocolError,
    TrajectorySample,
    collate_transitions,
    load_trajectory_sample,
)

__all__ = [
    "DPOFacts",
    "PreferencePair",
    "ReverseTransition",
    "TrajectoryProtocolError",
    "TrajectorySample",
    "TransitionScore",
    "compute_dpo_loss",
    "aggregate_transition_scores",
    "build_exact_atom_count_pairs",
    "collate_transitions",
    "gaussian_transition_score",
    "load_selected_manifest",
    "load_pair_manifest",
    "load_trajectory_sample",
    "validate_pair_manifests",
]
from .aggregation import aggregate_transition_scores
