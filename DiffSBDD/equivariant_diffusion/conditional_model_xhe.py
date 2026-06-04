"""Optional x/h/e conditional diffusion smoke path for DiffSBDD."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import torch

from equivariant_diffusion.conditional_model import ConditionalDDPM

try:
    from DiffSBDD.edge_diffusion.transitions import EdgeCategoricalTransition, EdgeTransitionConfig
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from DiffSBDD.edge_diffusion.transitions import EdgeCategoricalTransition, EdgeTransitionConfig


@dataclass(frozen=True)
class BondDiffusionConfig:
    """Configuration for the optional ligand halfedge diffusion path."""

    enabled: bool = False
    num_edge_types: int = 5
    edge_prior: str = "uniform"
    beta_schedule: str = "linear"
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    edge_hidden_dim: int = 128
    distance_basis_dim: int = 32
    edge_head_dropout: float = 0.0

    @classmethod
    def from_config(
        cls,
        config: Any,
        *,
        default_enabled: bool = False,
    ) -> BondDiffusionConfig:
        """Create settings from ``dict`` / ``Namespace`` / dataclass-like input."""
        return cls(
            enabled=bool(_config_get(config, "enabled", default_enabled)),
            num_edge_types=int(_config_get(config, "num_edge_types", cls.num_edge_types)),
            edge_prior=str(_config_get(config, "edge_prior", cls.edge_prior)),
            beta_schedule=str(_config_get(config, "beta_schedule", cls.beta_schedule)),
            beta_start=float(_config_get(config, "beta_start", cls.beta_start)),
            beta_end=float(_config_get(config, "beta_end", cls.beta_end)),
            edge_hidden_dim=int(_config_get(config, "edge_hidden_dim", cls.edge_hidden_dim)),
            distance_basis_dim=int(_config_get(config, "distance_basis_dim", cls.distance_basis_dim)),
            edge_head_dropout=float(_config_get(config, "edge_head_dropout", cls.edge_head_dropout)),
        )


@dataclass(frozen=True)
class XHEEdgeSmokeOutput:
    """Output of the edge-forward smoke pass."""

    pred_edge_logits: torch.Tensor
    noised_edge_type: torch.Tensor
    target_edge_type: torch.Tensor
    loss: torch.Tensor
    accuracy: torch.Tensor


class ConditionalDDPMXHE(ConditionalDDPM):
    """Conditional DiffSBDD model with optional halfedge smoke diagnostics."""

    def __init__(
        self,
        *args,
        bond_diffusion_config: BondDiffusionConfig | None = None,
        **kwargs,
    ) -> None:
        """Initialize the original conditional DDPM and edge transition."""
        timesteps = int(kwargs.get("timesteps", 1000))
        super().__init__(*args, **kwargs)
        self.bond_diffusion_config = bond_diffusion_config or BondDiffusionConfig(enabled=True)
        self.edge_transition = EdgeCategoricalTransition(
            EdgeTransitionConfig(
                num_classes=self.bond_diffusion_config.num_edge_types,
                num_timesteps=timesteps,
                prior=self.bond_diffusion_config.edge_prior,
                beta_schedule=self.bond_diffusion_config.beta_schedule,
                beta_start=self.bond_diffusion_config.beta_start,
                beta_end=self.bond_diffusion_config.beta_end,
            )
        )

    def forward(self, ligand: dict[str, torch.Tensor], pocket: dict[str, torch.Tensor], return_info: bool = False):
        """Run the old x/h loss and attach edge smoke metrics to ``info``."""
        edge_ligand, edge_pocket = _clone_ligand_pocket(ligand, pocket)
        old_terms = super().forward(ligand, pocket, return_info=True)
        loss_terms = old_terms[:-1]
        info = dict(old_terms[-1])

        edge_output = self.edge_forward_smoke(edge_ligand, edge_pocket)
        info["edge_smoke_loss"] = edge_output.loss
        info["edge_smoke_acc"] = edge_output.accuracy
        info["edge_smoke_logits_abs"] = edge_output.pred_edge_logits.abs().mean() \
            if edge_output.pred_edge_logits.numel() > 0 else edge_output.loss.detach()

        return (*loss_terms, info) if return_info else loss_terms

    def edge_forward_smoke(
        self,
        ligand: dict[str, torch.Tensor],
        pocket: dict[str, torch.Tensor],
    ) -> XHEEdgeSmokeOutput:
        """Noise true halfedge labels and predict clean bond-type logits."""
        _validate_ligand_halfedges(ligand)
        if not getattr(self.dynamics, "supports_xhe_edge_logits", False):
            raise TypeError("ConditionalDDPMXHE requires dynamics with xhe edge-logit support")

        ligand, pocket = _clone_ligand_pocket(ligand, pocket)
        ligand, pocket = self.normalize(ligand, pocket)
        batch_size = int(ligand["size"].shape[0])
        t_int = torch.randint(
            0,
            self.T + 1,
            size=(batch_size, 1),
            device=ligand["x"].device,
        ).float()
        t = t_int / self.T
        gamma_t = self.inflate_batch_array(self.gamma(t), ligand["x"])

        xh0_lig = torch.cat([ligand["x"], ligand["one_hot"]], dim=1)
        xh0_pocket = torch.cat([pocket["x"], pocket["one_hot"]], dim=1)
        xh0_lig[:, :self.n_dims], xh0_pocket[:, :self.n_dims] = self.remove_mean_batch(
            xh0_lig[:, :self.n_dims],
            xh0_pocket[:, :self.n_dims],
            ligand["mask"],
            pocket["mask"],
        )
        z_t_lig, xh_pocket, _ = self.noised_representation(
            xh0_lig,
            xh0_pocket,
            ligand["mask"],
            pocket["mask"],
            gamma_t,
        )

        target_edge_type = ligand["halfedge_type"].long()
        edge_batch = ligand["edge_batch"].long()
        log_v0 = self.edge_transition.index_to_log_onehot(target_edge_type)
        noised_edge_type, _ = self.edge_transition.q_vt_sample(
            log_v0,
            t_int.reshape(-1).long(),
            edge_batch,
        )

        _, _, pred_edge_logits = self.dynamics(
            z_t_lig,
            xh_pocket,
            t,
            ligand["mask"],
            pocket["mask"],
            halfedge_index=ligand["halfedge_index"],
            noised_edge_type=noised_edge_type,
            edge_batch=edge_batch,
            return_edge_logits=True,
        )
        if target_edge_type.numel() == 0:
            loss = pred_edge_logits.sum() * 0.0
            accuracy = loss.detach()
        else:
            loss = self.edge_transition.negative_log_likelihood(
                pred_edge_logits,
                target_edge_type,
            ).mean()
            accuracy = (pred_edge_logits.argmax(dim=-1) == target_edge_type).float().mean()
        return XHEEdgeSmokeOutput(
            pred_edge_logits=pred_edge_logits,
            noised_edge_type=noised_edge_type,
            target_edge_type=target_edge_type,
            loss=loss,
            accuracy=accuracy,
        )


def _config_get(config: Any, key: str, default: Any) -> Any:
    """Return one value from a config object without assuming its concrete type."""
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _clone_ligand_pocket(
    ligand: dict[str, torch.Tensor],
    pocket: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Clone tensor values so normalization does not mutate the caller's dicts."""
    return _clone_tensor_dict(ligand), _clone_tensor_dict(pocket)


def _clone_tensor_dict(values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Clone tensors and shallow-copy non-tensor values."""
    return {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in values.items()
    }


def _validate_ligand_halfedges(ligand: dict[str, torch.Tensor]) -> None:
    """Validate xhe ligand fields before running edge noising."""
    required = ("halfedge_index", "halfedge_type", "edge_batch")
    missing = [key for key in required if key not in ligand]
    if missing:
        raise ValueError(f"xhe diffusion requires ligand fields: {', '.join(missing)}")
    halfedge_index = ligand["halfedge_index"]
    halfedge_type = ligand["halfedge_type"]
    edge_batch = ligand["edge_batch"]
    if halfedge_index.ndim != 2 or halfedge_index.shape[0] != 2:
        raise ValueError("halfedge_index must have shape [2, num_halfedges]")
    num_halfedges = int(halfedge_index.shape[1])
    if halfedge_type.numel() != num_halfedges:
        raise ValueError("halfedge_type length must match halfedge_index")
    if edge_batch.numel() != num_halfedges:
        raise ValueError("edge_batch length must match halfedge_index")
