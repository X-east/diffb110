import random

import numpy as np
import torch


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_seed_for_rank(base_seed: int, rank: int):
    """Set rank-specific random seeds so each GPU generates different molecules.

    Each GPU gets seed = base_seed + rank, ensuring statistically independent
    trajectories across GPUs while remaining fully reproducible given the same
    base_seed and rank.

    Args:
        base_seed: Global base seed from config/CLI (e.g. 42).
        rank: Process rank (0-indexed, matches torch.distributed global_rank).
    """
    rank_seed = base_seed + rank
    random.seed(rank_seed)
    np.random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(rank_seed)
