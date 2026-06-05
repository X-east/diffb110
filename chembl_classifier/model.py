"""
ChEMBL Classifier: 3-layer MLP binary classifier.

Distinguishes ChEMBL-like molecules from anomalous molecules
(carbon collapse, all-sulfur garbage, extreme PAHs, etc.).

Architecture: 2220 → 512 → 128 → 1 (Sigmoid)
Input: 2048 Morgan count FP (radius=2) + 172 RDKit 2D descriptors
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ChemblClassifier(nn.Module):
    """3-layer MLP with BatchNorm + ReLU + Dropout for binary classification.

    Args:
        input_dim: Feature dimension (default 2220 = 2048 FP + 172 RDKit desc).
        hidden_dims: List of hidden layer dimensions.
        dropout_rates: Dropout rates for each hidden layer (same length as hidden_dims).
    """

    def __init__(
        self,
        input_dim: int = 2220,
        hidden_dims: tuple[int, ...] = (512, 128),
        dropout_rates: tuple[float, ...] = (0.3, 0.2),
    ) -> None:
        super().__init__()

        if len(hidden_dims) != len(dropout_rates):
            raise ValueError(
                f"hidden_dims ({len(hidden_dims)}) and dropout_rates "
                f"({len(dropout_rates)}) must have same length"
            )

        layers: list[nn.Module] = []
        in_dim = input_dim

        for i, (out_dim, drop_rate) in enumerate(zip(hidden_dims, dropout_rates)):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU(inplace=True))
            if drop_rate > 0:
                layers.append(nn.Dropout(drop_rate))
            in_dim = out_dim

        # Output layer: 1 logit (no Sigmoid — handled by BCEWithLogitsLoss)
        layers.append(nn.Linear(in_dim, 1))

        self._net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass, returns raw logits.

        Args:
            x: Feature tensor [B, input_dim].

        Returns:
            Logits tensor [B, 1].
        """
        return self._net(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return P(ChEMBL-like) in [0, 1].

        Args:
            x: Feature tensor [B, input_dim].

        Returns:
            Probability tensor [B, 1].
        """
        self.eval()
        with torch.no_grad():
            logits = self._net(x)
            return torch.sigmoid(logits)

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cpu") -> "ChemblClassifier":
        """Load model from checkpoint.

        Args:
            path: Path to .pt checkpoint file.
            device: Device to load model on.

        Returns:
            Loaded ChemblClassifier in eval mode.
        """
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = cls(
            input_dim=checkpoint.get("input_dim", 2220),
            hidden_dims=checkpoint.get("hidden_dims", (512, 128)),
            dropout_rates=checkpoint.get("dropout_rates", (0.3, 0.2)),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        return model

    def save(self, path: str, metadata: dict | None = None) -> None:
        """Save model checkpoint with metadata.

        Args:
            path: Output path.
            metadata: Optional extra data to save alongside weights.
        """
        checkpoint: dict = {
            "model_state_dict": self.state_dict(),
            "input_dim": 2220,
            "hidden_dims": (512, 128),
            "dropout_rates": (0.3, 0.2),
        }
        if metadata:
            checkpoint.update(metadata)
        torch.save(checkpoint, path)

    @property
    def n_params(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
