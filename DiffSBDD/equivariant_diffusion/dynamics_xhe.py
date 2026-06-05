"""Optional x/h/e dynamics extensions for DiffSBDD.

This module keeps the original :class:`EGNNDynamics` path intact and adds a
separate x/h/e-capable variant that can decode chemical halfedge bond logits
only when explicitly requested.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
import torch.nn as nn

from equivariant_diffusion.dynamics import EGNNDynamics
from equivariant_diffusion.en_diffusion import EnVariationalDiffusion


remove_mean_batch = EnVariationalDiffusion.remove_mean_batch


@dataclass(frozen=True)
class XHEEdgeHeadConfig:
    """Configuration for the optional chemical edge decoder."""

    num_edge_types: int = 5
    num_timesteps: int = 1000
    hidden_dim: int = 128
    distance_basis_dim: int = 32
    dropout: float = 0.0
    use_edge_message_passing: bool = False
    edge_message_dim: int = 32
    edge_message_scale: float = 1.0


class EGNNDynamicsXHE(EGNNDynamics):
    """EGNN dynamics with an optional DiffGui-style chemical edge decoder."""

    supports_xhe_edge_logits = True

    def __init__(
        self,
        *args,
        xhe_edge_head_config: XHEEdgeHeadConfig | None = None,
        **kwargs,
    ) -> None:
        """Initialize the original EGNN dynamics and optional edge head."""
        positional_joint_nf = args[3] if len(args) > 3 else 16
        self._xhe_joint_nf = int(kwargs.get("joint_nf", positional_joint_nf))
        config = xhe_edge_head_config or XHEEdgeHeadConfig()
        if config.use_edge_message_passing and kwargs.get("edge_embedding_dim") is None:
            kwargs["edge_embedding_dim"] = int(config.edge_message_dim)
        super().__init__(*args, **kwargs)
        self.xhe_edge_head_config = config
        if config.num_edge_types <= 1:
            raise ValueError("num_edge_types must be greater than 1")
        if config.distance_basis_dim < 4:
            raise ValueError("distance_basis_dim must be at least 4")
        if config.edge_message_dim <= 0:
            raise ValueError("edge_message_dim must be positive")
        self.chemical_edge_embedding = nn.Embedding(
            config.num_edge_types,
            self._xhe_joint_nf,
        )
        self.chemical_message_embedding = (
            nn.Embedding(config.num_edge_types, self.edge_nf)
            if config.use_edge_message_passing and self.edge_nf > 0
            else None
        )
        self.edge_time_mlp = _make_mlp(
            input_dim=4,
            output_dim=self._xhe_joint_nf,
            hidden_dim=config.hidden_dim,
            num_layers=2,
            dropout=config.dropout,
        )
        self.edge_distance_expansion = GaussianDistanceExpansion(
            num_basis=config.distance_basis_dim,
            start=0.0,
            stop=6.0,
        )
        self.chemical_edge_decoder = _make_mlp(
            input_dim=self._xhe_joint_nf * 4 + config.distance_basis_dim + 1,
            output_dim=config.num_edge_types,
            hidden_dim=config.hidden_dim,
            num_layers=3,
            dropout=config.dropout,
        )

    def forward(
        self,
        xh_atoms: torch.Tensor,
        xh_residues: torch.Tensor,
        t: torch.Tensor,
        mask_atoms: torch.Tensor,
        mask_residues: torch.Tensor,
        *,
        halfedge_index: torch.Tensor | None = None,
        noised_edge_type: torch.Tensor | None = None,
        edge_batch: torch.Tensor | None = None,
        return_edge_logits: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run DiffSBDD dynamics and optionally decode halfedge bond logits."""
        x_atoms = xh_atoms[:, :self.n_dims].clone()
        h_atoms = xh_atoms[:, self.n_dims:].clone()

        x_residues = xh_residues[:, :self.n_dims].clone()
        h_residues = xh_residues[:, self.n_dims:].clone()

        h_atoms = self.atom_encoder(h_atoms)
        h_residues = self.residue_encoder(h_residues)

        x = torch.cat((x_atoms, x_residues), dim=0)
        h = torch.cat((h_atoms, h_residues), dim=0)
        mask = torch.cat([mask_atoms, mask_residues])

        if self.condition_time:
            if isinstance(t, (int, float)):
                h_time = torch.empty_like(h[:, 0:1]).fill_(t)
            elif np.prod(t.size()) == 1:
                h_time = torch.empty_like(h[:, 0:1]).fill_(t.item())
            else:
                h_time = t[mask]
                if h_time.dim() > 1:
                    h_time = h_time.reshape(h_time.shape[0], -1)[:, :1]
                else:
                    h_time = h_time.unsqueeze(-1)
            h = torch.cat([h, h_time], dim=1)

        edges = self.get_edges(mask_atoms, mask_residues, x_atoms, x_residues)
        assert torch.all(mask[edges[0]] == mask[edges[1]])

        edge_types = self._graph_edge_types(edges, len(mask_atoms)) if self.edge_nf > 0 else None
        edge_types = self._add_chemical_edge_messages(
            edge_attr=edge_types,
            graph_edges=edges,
            num_ligand_atoms=len(mask_atoms),
            halfedge_index=halfedge_index,
            noised_edge_type=noised_edge_type,
        )

        if self.mode == "egnn_dynamics":
            update_coords_mask = None if self.update_pocket_coords else torch.cat((
                torch.ones_like(mask_atoms),
                torch.zeros_like(mask_residues),
            )).unsqueeze(1)
            h_final, x_final = self.egnn(
                h,
                x,
                edges,
                update_coords_mask=update_coords_mask,
                batch_mask=mask,
                edge_attr=edge_types,
            )
            vel = x_final - x
        elif self.mode == "gnn_dynamics":
            xh = torch.cat([x, h], dim=1)
            output = self.gnn(xh, edges, node_mask=None, edge_attr=edge_types)
            vel = output[:, :3]
            h_final = output[:, 3:]
        else:
            raise ValueError(f"Wrong mode {self.mode}")

        if self.condition_time:
            h_final = h_final[:, :-1]

        h_final_atoms = h_final[:len(mask_atoms)]
        edge_logits = None
        if return_edge_logits:
            edge_logits = self._predict_edge_logits(
                node_h=h_final_atoms,
                ligand_pos=x_atoms,
                t=t,
                halfedge_index=halfedge_index,
                noised_edge_type=noised_edge_type,
                edge_batch=edge_batch,
            )

        h_final_residues = h_final[len(mask_atoms):]
        h_final_atoms = self.atom_decoder(h_final_atoms)
        h_final_residues = self.residue_decoder(h_final_residues)

        if torch.any(torch.isnan(vel)):
            if self.training:
                vel[torch.isnan(vel)] = 0.0
            else:
                raise ValueError("NaN detected in EGNN output")

        if self.update_pocket_coords:
            vel = remove_mean_batch(vel, mask)

        out_atoms = torch.cat([vel[:len(mask_atoms)], h_final_atoms], dim=-1)
        out_residues = torch.cat([vel[len(mask_atoms):], h_final_residues], dim=-1)
        if return_edge_logits:
            if edge_logits is None:
                raise RuntimeError("edge_logits were not computed")
            return out_atoms, out_residues, edge_logits
        return out_atoms, out_residues

    def _graph_edge_types(self, edges: torch.Tensor, num_ligand_atoms: int) -> torch.Tensor:
        """Return original DiffSBDD graph-edge embeddings."""
        edge_types = torch.zeros(edges.size(1), dtype=torch.long, device=edges.device)
        edge_types[(edges[0] < num_ligand_atoms) & (edges[1] < num_ligand_atoms)] = 1
        edge_types[(edges[0] >= num_ligand_atoms) & (edges[1] >= num_ligand_atoms)] = 2
        return self.edge_embedding(edge_types)

    def _add_chemical_edge_messages(
        self,
        *,
        edge_attr: torch.Tensor | None,
        graph_edges: torch.Tensor,
        num_ligand_atoms: int,
        halfedge_index: torch.Tensor | None,
        noised_edge_type: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Inject noised ligand halfedge labels into EGNN graph-edge features."""
        config = self.xhe_edge_head_config
        if (
            not config.use_edge_message_passing
            or self.chemical_message_embedding is None
            or halfedge_index is None
            or noised_edge_type is None
        ):
            return edge_attr
        if self.edge_nf <= 0:
            return edge_attr
        if edge_attr is None:
            edge_attr = graph_edges.new_zeros((graph_edges.shape[1], self.edge_nf), dtype=torch.float32)
        if halfedge_index.numel() == 0 or graph_edges.numel() == 0:
            return edge_attr

        graph_src, graph_dst = graph_edges.long()
        ligand_graph_mask = (
            (graph_src < num_ligand_atoms)
            & (graph_dst < num_ligand_atoms)
            & (graph_src != graph_dst)
        )
        if not bool(ligand_graph_mask.any()):
            return edge_attr

        graph_u = torch.minimum(graph_src[ligand_graph_mask], graph_dst[ligand_graph_mask])
        graph_v = torch.maximum(graph_src[ligand_graph_mask], graph_dst[ligand_graph_mask])
        graph_keys = graph_u * int(num_ligand_atoms) + graph_v

        half_src, half_dst = halfedge_index.long()
        half_u = torch.minimum(half_src, half_dst)
        half_v = torch.maximum(half_src, half_dst)
        half_keys = half_u * int(num_ligand_atoms) + half_v
        order = torch.argsort(half_keys)
        sorted_keys = half_keys.index_select(0, order)
        positions = torch.searchsorted(sorted_keys, graph_keys)
        valid = positions < sorted_keys.numel()
        if bool(valid.any()):
            matched = sorted_keys.index_select(0, positions[valid]) == graph_keys[valid]
            valid_indices = valid.nonzero(as_tuple=False).flatten()[matched]
            matched_positions = positions[valid][matched]
            graph_edge_type = torch.zeros(
                graph_keys.shape[0],
                dtype=torch.long,
                device=graph_edges.device,
            )
            graph_edge_type[valid_indices] = noised_edge_type.long().index_select(
                0,
                order.index_select(0, matched_positions),
            )
        else:
            graph_edge_type = torch.zeros(
                graph_keys.shape[0],
                dtype=torch.long,
                device=graph_edges.device,
            )

        chemical_attr = self.chemical_message_embedding(graph_edge_type)
        updated = edge_attr.clone()
        updated[ligand_graph_mask] = updated[ligand_graph_mask] + float(config.edge_message_scale) * chemical_attr
        return updated

    def _predict_edge_logits(
        self,
        *,
        node_h: torch.Tensor,
        ligand_pos: torch.Tensor,
        t: torch.Tensor,
        halfedge_index: torch.Tensor | None,
        noised_edge_type: torch.Tensor | None,
        edge_batch: torch.Tensor | None,
    ) -> torch.Tensor:
        """Decode clean bond-type logits for ligand halfedges."""
        if halfedge_index is None or noised_edge_type is None or edge_batch is None:
            raise ValueError(
                "halfedge_index, noised_edge_type, and edge_batch are required "
                "when return_edge_logits=True"
            )
        if halfedge_index.ndim != 2 or halfedge_index.shape[0] != 2:
            raise ValueError("halfedge_index must have shape [2, num_halfedges]")
        num_halfedges = int(halfedge_index.shape[1])
        if noised_edge_type.numel() != num_halfedges:
            raise ValueError("noised_edge_type length must match halfedge_index")
        if edge_batch.numel() != num_halfedges:
            raise ValueError("edge_batch length must match halfedge_index")
        if num_halfedges == 0:
            return ligand_pos.new_zeros((0, self.xhe_edge_head_config.num_edge_types))

        halfedge_index = halfedge_index.long()
        src, dst = halfedge_index
        distances = torch.linalg.norm(ligand_pos[src] - ligand_pos[dst], dim=-1, keepdim=True)
        distance_features = self.edge_distance_expansion(distances.squeeze(-1))
        edge_features = self.chemical_edge_embedding(noised_edge_type.long())
        time_features = self.edge_time_mlp(
            _time_features(t, edge_batch.long(), self.xhe_edge_head_config.num_timesteps)
        )
        return self.chemical_edge_decoder(torch.cat([
            node_h[src],
            node_h[dst],
            edge_features,
            time_features,
            distance_features,
            distances,
        ], dim=-1))


def _time_features(t: torch.Tensor, batch_index: torch.Tensor, num_timesteps: int) -> torch.Tensor:
    """Return simple sinusoidal timestep features for atom or edge batches."""
    if batch_index.numel() == 0:
        return t.new_zeros((0, 4), dtype=torch.float32)
    if t.ndim == 0 or t.numel() == 1:
        t_per_item = t.reshape(()).expand_as(batch_index)
    else:
        t_per_item = t.reshape(t.shape[0], -1)[:, 0].index_select(0, batch_index.long())
    t_float = t_per_item.float()
    if bool((t_float > 1.0).any()):
        t_float = t_float / float(max(1, num_timesteps - 1))
    return torch.stack([
        t_float,
        torch.sin(math.pi * t_float),
        torch.cos(math.pi * t_float),
        torch.ones_like(t_float),
    ], dim=-1)


def _make_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float = 0.0,
) -> nn.Sequential:
    """Build a compact SiLU MLP."""
    if num_layers < 1:
        raise ValueError("num_layers must be at least 1")
    layers: list[nn.Module] = []
    current_dim = input_dim
    for _ in range(num_layers - 1):
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(nn.SiLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


class GaussianDistanceExpansion(nn.Module):
    """Expand scalar distances into fixed Gaussian basis features."""

    def __init__(self, num_basis: int, start: float, stop: float) -> None:
        """Initialize Gaussian centers and width."""
        super().__init__()
        centers = torch.linspace(start, stop, num_basis)
        self.register_buffer("centers", centers)
        self.coefficient = -0.5 / ((centers[1] - centers[0]).item() ** 2)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """Return Gaussian basis features for distance tensor ``[num_edges]``."""
        return torch.exp(self.coefficient * (distances.unsqueeze(-1) - self.centers) ** 2)
