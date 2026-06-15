"""Lightweight Chemprop-style directed message passing classifier.

The implementation follows the D-MPNN shape used by Chemprop: messages live on
directed bonds, each update aggregates incoming neighbor-bond messages while
excluding the reverse edge, then molecule embeddings are read out from atom
states and passed through a feed-forward classifier.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

import torch
import torch.nn as nn


COMMON_ATOMIC_NUMBERS: tuple[int, ...] = (
    1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53,
)
DEGREE_VALUES: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
FORMAL_CHARGE_VALUES: tuple[int, ...] = (-2, -1, 0, 1, 2)
TOTAL_H_VALUES: tuple[int, ...] = (0, 1, 2, 3, 4)
HYBRIDIZATION_VALUES: tuple[str, ...] = ("SP", "SP2", "SP3", "SP3D", "SP3D2")
BOND_STEREO_VALUES: tuple[str, ...] = ("STEREONONE", "STEREOANY", "STEREOZ", "STEREOE")


@dataclass(frozen=True)
class DMPNNConfig:
    """Serializable architecture configuration for DMPNNClassifier."""

    atom_fdim: int
    bond_fdim: int
    hidden_size: int = 300
    depth: int = 3
    ffn_hidden_size: int = 300
    ffn_num_layers: int = 2
    dropout: float = 0.1
    aggregation: str = "mean"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DMPNNConfig":
        """Build config from a checkpoint or YAML dictionary."""
        valid_keys = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in payload.items() if key in valid_keys})

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML/checkpoint-friendly dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class MolGraphTensor:
    """Tensorized molecule graph used by the D-MPNN forward pass."""

    atom_features: torch.Tensor
    bond_features: torch.Tensor
    a2b: torch.Tensor
    b2a: torch.Tensor
    b2revb: torch.Tensor
    atom_scope: list[tuple[int, int]]


class DMPNNClassifier(nn.Module):
    """Directed message passing binary classifier for molecular graphs."""

    def __init__(self, config: DMPNNConfig) -> None:
        super().__init__()
        if config.depth < 1:
            raise ValueError("DMPNNConfig.depth must be >= 1")
        if config.aggregation not in {"mean", "sum"}:
            raise ValueError("DMPNNConfig.aggregation must be 'mean' or 'sum'")

        self.config = config
        self.W_i = nn.Linear(config.atom_fdim + config.bond_fdim, config.hidden_size, bias=False)
        self.W_h = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.W_o = nn.Linear(config.atom_fdim + config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = nn.ReLU()
        self.ffn = self._build_ffn(config)

    def forward(self, graph: MolGraphTensor) -> torch.Tensor:
        """Return one logit per molecule in the batched graph."""
        if graph.atom_features.numel() == 0:
            raise ValueError("DMPNNClassifier received an empty atom feature tensor")

        if graph.bond_features.shape[0] == 0:
            atom_messages = graph.atom_features.new_zeros(
                (graph.atom_features.shape[0], self.config.hidden_size)
            )
        else:
            source_atom_features = graph.atom_features.index_select(0, graph.b2a)
            message_input = torch.cat([source_atom_features, graph.bond_features], dim=1)
            message = self.activation(self.W_i(message_input))

            for _ in range(self.config.depth - 1):
                atom_message = _index_select_nd(message, graph.a2b).sum(dim=1)
                rev_message = message.index_select(0, graph.b2revb)
                message = self.activation(message + self.W_h(atom_message.index_select(0, graph.b2a) - rev_message))
                message = self.dropout(message)

            atom_messages = _index_select_nd(message, graph.a2b).sum(dim=1)

        atom_hidden = self.activation(self.W_o(torch.cat([graph.atom_features, atom_messages], dim=1)))
        atom_hidden = self.dropout(atom_hidden)
        mol_vecs = _readout(atom_hidden, graph.atom_scope, self.config.aggregation)
        return self.ffn(mol_vecs).squeeze(1)

    def save(self, path: str, metadata: dict[str, Any] | None = None) -> None:
        """Save model weights and architecture config."""
        payload: dict[str, Any] = {
            "model_state_dict": self.state_dict(),
            "model_config": self.config.to_dict(),
            "model_type": "chembl_dmpnn",
        }
        if metadata:
            payload.update(metadata)
        torch.save(payload, path)

    @classmethod
    def from_checkpoint(cls, path: str, device: str | torch.device = "cpu") -> "DMPNNClassifier":
        """Load a DMPNNClassifier from a checkpoint."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config_payload = checkpoint.get("model_config") or checkpoint.get("config")
        if not isinstance(config_payload, dict):
            raise ValueError(f"D-MPNN checkpoint lacks model_config: {path}")
        model = cls(DMPNNConfig.from_dict(config_payload))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        return model

    @property
    def n_params(self) -> int:
        """Total number of trainable parameters."""
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    @staticmethod
    def _build_ffn(config: DMPNNConfig) -> nn.Sequential:
        layers: list[nn.Module] = []
        in_dim = config.hidden_size
        for _ in range(max(config.ffn_num_layers - 1, 0)):
            layers.append(nn.Linear(in_dim, config.ffn_hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout))
            in_dim = config.ffn_hidden_size
        layers.append(nn.Linear(in_dim, 1))
        return nn.Sequential(*layers)


def atom_feature_dim() -> int:
    """Return the fixed atom feature dimension."""
    return (
        len(COMMON_ATOMIC_NUMBERS) + 1
        + len(DEGREE_VALUES) + 1
        + len(FORMAL_CHARGE_VALUES) + 1
        + len(TOTAL_H_VALUES) + 1
        + len(HYBRIDIZATION_VALUES) + 1
        + 3
    )


def bond_feature_dim() -> int:
    """Return the fixed bond feature dimension."""
    return 4 + 2 + len(BOND_STEREO_VALUES) + 1


def atom_features(atom: Any) -> list[float]:
    """Build atom features from an RDKit atom."""
    hybridization = str(atom.GetHybridization())
    return (
        _one_hot_with_unknown(atom.GetAtomicNum(), COMMON_ATOMIC_NUMBERS)
        + _one_hot_with_unknown(atom.GetTotalDegree(), DEGREE_VALUES)
        + _one_hot_with_unknown(atom.GetFormalCharge(), FORMAL_CHARGE_VALUES)
        + _one_hot_with_unknown(atom.GetTotalNumHs(), TOTAL_H_VALUES)
        + _one_hot_with_unknown(hybridization, HYBRIDIZATION_VALUES)
        + [
            float(atom.GetIsAromatic()),
            float(atom.IsInRing()),
            float(atom.GetMass() * 0.01),
        ]
    )


def bond_features(bond: Any) -> list[float]:
    """Build bond features from an RDKit bond."""
    bond_type = bond.GetBondType()
    stereo = str(bond.GetStereo())
    return [
        float(str(bond_type) == "SINGLE"),
        float(str(bond_type) == "DOUBLE"),
        float(str(bond_type) == "TRIPLE"),
        float(str(bond_type) == "AROMATIC"),
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
        *_one_hot_with_unknown(stereo, BOND_STEREO_VALUES),
    ]


def mols_to_graph_tensors(mols: list[Any], device: torch.device | str | None = None) -> MolGraphTensor:
    """Convert a list of RDKit molecules into one batched graph tensor."""
    atom_rows: list[list[float]] = []
    bond_rows: list[list[float]] = []
    a2b_lists: list[list[int]] = []
    b2a: list[int] = []
    b2revb: list[int] = []
    atom_scope: list[tuple[int, int]] = []

    for mol in mols:
        start = len(atom_rows)
        atoms = list(mol.GetAtoms())
        atom_scope.append((start, len(atoms)))
        for atom in atoms:
            atom_rows.append(atom_features(atom))
            a2b_lists.append([])

        for bond in mol.GetBonds():
            begin = start + bond.GetBeginAtomIdx()
            end = start + bond.GetEndAtomIdx()
            features = bond_features(bond)
            forward_idx = len(bond_rows)
            backward_idx = forward_idx + 1
            bond_rows.append(features)
            bond_rows.append(features)
            b2a.extend([begin, end])
            b2revb.extend([backward_idx, forward_idx])
            a2b_lists[end].append(forward_idx)
            a2b_lists[begin].append(backward_idx)

    target_device = torch.device(device) if device is not None else torch.device("cpu")
    atom_tensor = torch.tensor(atom_rows, dtype=torch.float32, device=target_device)
    if bond_rows:
        bond_tensor = torch.tensor(bond_rows, dtype=torch.float32, device=target_device)
        b2a_tensor = torch.tensor(b2a, dtype=torch.long, device=target_device)
        b2revb_tensor = torch.tensor(b2revb, dtype=torch.long, device=target_device)
    else:
        bond_tensor = torch.empty((0, bond_feature_dim()), dtype=torch.float32, device=target_device)
        b2a_tensor = torch.empty((0,), dtype=torch.long, device=target_device)
        b2revb_tensor = torch.empty((0,), dtype=torch.long, device=target_device)

    max_bonds = max([len(indices) for indices in a2b_lists] + [0])
    a2b = torch.full((len(a2b_lists), max_bonds), -1, dtype=torch.long, device=target_device)
    for atom_idx, incoming in enumerate(a2b_lists):
        if incoming:
            a2b[atom_idx, : len(incoming)] = torch.tensor(incoming, dtype=torch.long, device=target_device)

    return MolGraphTensor(
        atom_features=atom_tensor,
        bond_features=bond_tensor,
        a2b=a2b,
        b2a=b2a_tensor,
        b2revb=b2revb_tensor,
        atom_scope=atom_scope,
    )


def default_dmpnn_config(**overrides: Any) -> DMPNNConfig:
    """Return the default D-MPNN architecture config with optional overrides."""
    payload: dict[str, Any] = {
        "atom_fdim": atom_feature_dim(),
        "bond_fdim": bond_feature_dim(),
        "hidden_size": 300,
        "depth": 3,
        "ffn_hidden_size": 300,
        "ffn_num_layers": 2,
        "dropout": 0.1,
        "aggregation": "mean",
    }
    payload.update(overrides)
    return DMPNNConfig.from_dict(payload)


def _one_hot_with_unknown(value: Any, choices: tuple[Any, ...]) -> list[float]:
    output = [float(value == choice) for choice in choices]
    output.append(float(value not in choices))
    return output


def _index_select_nd(source: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Index source with -1 padding mapped to all-zero rows."""
    if index.numel() == 0:
        return source.new_zeros((*index.shape, source.shape[-1]))
    padded = torch.cat([source.new_zeros((1, source.shape[-1])), source], dim=0)
    safe_index = torch.clamp(index + 1, min=0)
    return padded.index_select(0, safe_index.reshape(-1)).reshape(*index.shape, source.shape[-1])


def _readout(atom_hidden: torch.Tensor, atom_scope: list[tuple[int, int]], aggregation: str) -> torch.Tensor:
    """Aggregate atom embeddings into molecule embeddings."""
    mol_vecs: list[torch.Tensor] = []
    hidden_size = atom_hidden.shape[1]
    for start, length in atom_scope:
        if length <= 0:
            mol_vecs.append(atom_hidden.new_zeros(hidden_size))
            continue
        atoms = atom_hidden.narrow(0, start, length)
        if aggregation == "sum":
            mol_vecs.append(atoms.sum(dim=0))
        else:
            mol_vecs.append(atoms.mean(dim=0))
    return torch.stack(mol_vecs, dim=0)
