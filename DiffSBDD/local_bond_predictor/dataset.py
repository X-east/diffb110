"""Datasets and feature builders for local ligand bond prediction."""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

DIFFSBDD_ROOT = Path(__file__).resolve().parents[1]
if str(DIFFSBDD_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFSBDD_ROOT))

from dataset import ProcessedLigandPocketDataset  # noqa: E402


@dataclass(frozen=True)
class PairFeatureConfig:
    """Configuration for atom-pair feature construction."""

    rbf_min: float = 0.0
    rbf_max: float = 6.0
    rbf_count: int = 32
    rbf_gamma: float | None = None

    @property
    def distance_feature_dim(self) -> int:
        """Return the number of distance-derived feature dimensions."""
        return self.rbf_count + 1


def choose_sample_indices(length: int, max_samples: int | None, seed: int) -> list[int]:
    """Choose a deterministic subset of sample indices."""
    if max_samples is None or max_samples >= length:
        return list(range(length))
    rng = random.Random(seed)
    return sorted(rng.sample(range(length), max_samples))


def pair_feature_dim(atom_feature_dim: int, config: PairFeatureConfig) -> int:
    """Return the feature dimension for one atom pair."""
    return 2 * atom_feature_dim + config.distance_feature_dim


def build_pair_features(
    coords: torch.Tensor,
    atom_features: torch.Tensor,
    halfedge_index: torch.Tensor,
    config: PairFeatureConfig,
) -> torch.Tensor:
    """Build atom-pair features from coordinates, atom one-hot vectors, and halfedges."""
    if halfedge_index.numel() == 0:
        feature_dim = pair_feature_dim(atom_features.shape[-1], config)
        return atom_features.new_zeros((0, feature_dim))

    atom_i = halfedge_index[0].long()
    atom_j = halfedge_index[1].long()
    delta = coords[atom_i] - coords[atom_j]
    distances = torch.linalg.norm(delta, dim=-1, keepdim=True)
    centers = torch.linspace(
        config.rbf_min,
        config.rbf_max,
        config.rbf_count,
        dtype=coords.dtype,
        device=coords.device,
    )
    if config.rbf_gamma is None:
        width = max((config.rbf_max - config.rbf_min) / max(1, config.rbf_count - 1), 1e-6)
        gamma = 1.0 / (width * width)
    else:
        gamma = config.rbf_gamma
    rbf = torch.exp(-gamma * (distances - centers.view(1, -1)).pow(2))
    return torch.cat([atom_features[atom_i], atom_features[atom_j], distances, rbf], dim=-1)


class HalfedgePairDataset(Dataset):
    """Molecule-level dataset that emits sampled halfedge pair features and labels."""

    def __init__(
        self,
        npz_path: Path,
        *,
        max_samples: int | None = None,
        sample_seed: int = 42,
        negative_ratio: float | None = None,
        pos_noise_std: float = 0.0,
        feature_config: PairFeatureConfig | None = None,
    ) -> None:
        """Load one processed DiffSBDD split for pair-wise bond prediction."""
        self.base_dataset = ProcessedLigandPocketDataset(npz_path, center=False)
        self.indices = choose_sample_indices(len(self.base_dataset), max_samples, sample_seed)
        self.sample_seed = sample_seed
        self.negative_ratio = negative_ratio
        self.pos_noise_std = pos_noise_std
        self.feature_config = PairFeatureConfig() if feature_config is None else feature_config

    def __len__(self) -> int:
        """Return the number of selected molecules."""
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        """Return pair features and halfedge labels for one molecule."""
        base_index = int(self.indices[item])
        sample = self.base_dataset[base_index]
        coords = sample["lig_coords"].float()
        atom_features = sample["lig_one_hot"].float()
        halfedge_index = sample["ligand_halfedge_index"].long()
        halfedge_type = sample["ligand_halfedge_type"].long()

        generator = torch.Generator()
        generator.manual_seed(self.sample_seed + base_index)
        if self.pos_noise_std > 0:
            noise = torch.randn(coords.shape, generator=generator, dtype=coords.dtype)
            coords = coords + noise * self.pos_noise_std

        selected = self._select_halfedges(halfedge_type, generator)
        selected_halfedge_index = halfedge_index[:, selected]
        labels = halfedge_type[selected]
        features = build_pair_features(
            coords,
            atom_features,
            selected_halfedge_index,
            self.feature_config,
        )
        return {
            "features": features,
            "labels": labels,
            "source_index": torch.tensor(base_index, dtype=torch.long),
            "num_pairs": torch.tensor(len(labels), dtype=torch.long),
            "num_positive": torch.tensor(int((labels > 0).sum().item()), dtype=torch.long),
        }

    def _select_halfedges(self, halfedge_type: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        """Select all positive edges and a configurable number of no-bond negatives."""
        all_indices = torch.arange(len(halfedge_type), dtype=torch.long)
        if self.negative_ratio is None:
            return all_indices

        positive = all_indices[halfedge_type > 0]
        negative = all_indices[halfedge_type == 0]
        if len(negative) == 0:
            return positive
        if len(positive) == 0:
            max_negative = min(len(negative), max(1, int(math.ceil(self.negative_ratio))))
        else:
            max_negative = min(len(negative), int(math.ceil(len(positive) * self.negative_ratio)))
        neg_perm = torch.randperm(len(negative), generator=generator)[:max_negative]
        selected = torch.cat([positive, negative[neg_perm]])
        order = torch.randperm(len(selected), generator=generator)
        return selected[order]


def collate_pair_batches(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Collate molecule-level pair feature batches into one tensor batch."""
    return {
        "features": torch.cat([item["features"] for item in batch], dim=0),
        "labels": torch.cat([item["labels"] for item in batch], dim=0),
        "source_index": torch.stack([item["source_index"] for item in batch]),
        "num_pairs": torch.stack([item["num_pairs"] for item in batch]),
        "num_positive": torch.stack([item["num_positive"] for item in batch]),
    }
