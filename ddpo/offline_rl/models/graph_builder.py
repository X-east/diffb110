"""Sparse radius-graph construction for geometric offline-RL critics."""

from __future__ import annotations

from dataclasses import dataclass

import torch


POCKET_ROLE = 0
STATE_ROLE = 1
ACTION_ROLE = 2
IDENTITY_EDGE_TYPE = 9
NUM_EDGE_TYPES = 10


@dataclass(frozen=True)
class SparseGeometricGraph:
    """Sparse directed edges over a padded batch of geometric nodes."""

    edge_index: torch.Tensor
    edge_distance: torch.Tensor
    edge_type: torch.Tensor

    def __post_init__(self) -> None:
        """Validate edge tensor shapes."""
        if self.edge_index.ndim != 2 or int(self.edge_index.shape[0]) != 2:
            raise ValueError("edge_index must have shape [2,E]")
        edge_count = int(self.edge_index.shape[1])
        if tuple(self.edge_distance.shape) != (edge_count,):
            raise ValueError("edge_distance must have shape [E]")
        if tuple(self.edge_type.shape) != (edge_count,):
            raise ValueError("edge_type must have shape [E]")
        if self.edge_index.dtype != torch.long:
            raise TypeError("edge_index must use torch.long")
        if self.edge_type.dtype != torch.long:
            raise TypeError("edge_type must use torch.long")
        if edge_count and (
            bool(torch.any(self.edge_type < 0))
            or bool(torch.any(self.edge_type >= NUM_EDGE_TYPES))
        ):
            raise ValueError("edge_type contains an unsupported value")


class RadiusGraphBuilder:
    """Build role-aware sparse radius graphs without external graph packages."""

    def __init__(
        self,
        *,
        pocket_cutoff: float,
        ligand_cutoff: float,
        interaction_cutoff: float,
        max_neighbors: int | None,
    ) -> None:
        """Initialize graph cutoffs and the optional neighbor cap."""
        for name, value in (
            ("pocket_cutoff", pocket_cutoff),
            ("ligand_cutoff", ligand_cutoff),
            ("interaction_cutoff", interaction_cutoff),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if max_neighbors is not None and max_neighbors <= 0:
            raise ValueError("max_neighbors must be positive or None")
        self.pocket_cutoff = float(pocket_cutoff)
        self.ligand_cutoff = float(ligand_cutoff)
        self.interaction_cutoff = float(interaction_cutoff)
        self.max_neighbors = max_neighbors

    def build(
        self,
        coordinates: torch.Tensor,
        node_mask: torch.Tensor,
        node_roles: torch.Tensor,
        *,
        identity_pairs: torch.Tensor | None = None,
    ) -> SparseGeometricGraph:
        """Build directed radius edges and optional state-action identity edges."""
        if coordinates.ndim != 3 or int(coordinates.shape[-1]) != 3:
            raise ValueError("coordinates must have shape [B,M,3]")
        batch_size, nodes_per_graph, _ = coordinates.shape
        if tuple(node_mask.shape) != (batch_size, nodes_per_graph, 1):
            raise ValueError("node_mask must have shape [B,M,1]")
        if tuple(node_roles.shape) != (nodes_per_graph,):
            raise ValueError("node_roles must have shape [M]")
        if node_roles.dtype != torch.long:
            raise TypeError("node_roles must use torch.long")
        if identity_pairs is not None:
            if identity_pairs.ndim != 2 or int(identity_pairs.shape[-1]) != 2:
                raise ValueError("identity_pairs must have shape [K,2]")
            if identity_pairs.dtype != torch.long:
                raise TypeError("identity_pairs must use torch.long")

        edge_sources: list[torch.Tensor] = []
        edge_targets: list[torch.Tensor] = []
        edge_distances: list[torch.Tensor] = []
        edge_types: list[torch.Tensor] = []
        for batch_index in range(batch_size):
            valid_local = torch.nonzero(
                node_mask[batch_index, :, 0] != 0,
                as_tuple=False,
            ).flatten()
            local_coordinates = coordinates[batch_index, valid_local]
            local_roles = node_roles[valid_local]
            distance_matrix = torch.cdist(
                local_coordinates,
                local_coordinates,
            )
            source_local, target_local = self._radius_edges(
                distance_matrix,
                local_roles,
            )
            if int(source_local.numel()) > 0:
                source_nodes = valid_local[source_local]
                target_nodes = valid_local[target_local]
                global_offset = batch_index * nodes_per_graph
                edge_sources.append(source_nodes + global_offset)
                edge_targets.append(target_nodes + global_offset)
                edge_distances.append(distance_matrix[target_local, source_local])
                edge_types.append(
                    node_roles[source_nodes] * 3 + node_roles[target_nodes]
                )

            if identity_pairs is not None:
                pair_valid = torch.logical_and(
                    node_mask[batch_index, identity_pairs[:, 0], 0] != 0,
                    node_mask[batch_index, identity_pairs[:, 1], 0] != 0,
                )
                pairs = identity_pairs[pair_valid]
                if int(pairs.shape[0]) > 0:
                    identity_source = torch.cat([pairs[:, 0], pairs[:, 1]])
                    identity_target = torch.cat([pairs[:, 1], pairs[:, 0]])
                    global_offset = batch_index * nodes_per_graph
                    edge_sources.append(identity_source + global_offset)
                    edge_targets.append(identity_target + global_offset)
                    edge_distances.append(
                        torch.linalg.vector_norm(
                            coordinates[batch_index, identity_source]
                            - coordinates[batch_index, identity_target],
                            dim=-1,
                        )
                    )
                    edge_types.append(
                        torch.full(
                            (int(identity_source.numel()),),
                            IDENTITY_EDGE_TYPE,
                            dtype=torch.long,
                            device=coordinates.device,
                        )
                    )

        if not edge_sources:
            empty_index = torch.empty(
                (2, 0),
                dtype=torch.long,
                device=coordinates.device,
            )
            return SparseGeometricGraph(
                edge_index=empty_index,
                edge_distance=coordinates.new_empty((0,)),
                edge_type=torch.empty(
                    (0,),
                    dtype=torch.long,
                    device=coordinates.device,
                ),
            )
        return SparseGeometricGraph(
            edge_index=torch.stack(
                [torch.cat(edge_sources), torch.cat(edge_targets)],
                dim=0,
            ),
            edge_distance=torch.cat(edge_distances),
            edge_type=torch.cat(edge_types),
        )

    def _radius_edges(
        self,
        distances: torch.Tensor,
        roles: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return local directed source and target indices."""
        node_count = int(distances.shape[0])
        if node_count <= 1:
            empty = torch.empty(
                (0,),
                dtype=torch.long,
                device=distances.device,
            )
            return empty, empty

        source_roles = roles.unsqueeze(0).expand(node_count, -1)
        target_roles = roles.unsqueeze(1).expand(-1, node_count)
        pocket_pair = torch.logical_and(
            source_roles == POCKET_ROLE,
            target_roles == POCKET_ROLE,
        )
        pocket_interaction = torch.logical_xor(
            source_roles == POCKET_ROLE,
            target_roles == POCKET_ROLE,
        )
        cutoffs = distances.new_full(
            distances.shape,
            self.ligand_cutoff,
        )
        cutoffs = torch.where(
            pocket_pair,
            cutoffs.new_tensor(self.pocket_cutoff),
            cutoffs,
        )
        cutoffs = torch.where(
            pocket_interaction,
            cutoffs.new_tensor(self.interaction_cutoff),
            cutoffs,
        )
        candidate = distances <= cutoffs
        candidate.fill_diagonal_(False)

        if self.max_neighbors is None:
            target, source = torch.where(candidate)
            return source, target

        capped = distances.masked_fill(~candidate, float("inf"))
        neighbor_count = min(self.max_neighbors, node_count - 1)
        nearest_distance, nearest_source = torch.topk(
            capped,
            k=neighbor_count,
            dim=1,
            largest=False,
        )
        keep = torch.isfinite(nearest_distance)
        target = (
            torch.arange(node_count, device=distances.device)
            .unsqueeze(1)
            .expand_as(nearest_source)
        )
        return nearest_source[keep], target[keep]
