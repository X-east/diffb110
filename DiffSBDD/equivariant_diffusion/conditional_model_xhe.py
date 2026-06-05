"""Optional x/h/e conditional diffusion smoke path for DiffSBDD."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn.functional as F
from torch_scatter import scatter_mean

from equivariant_diffusion.conditional_model import ConditionalDDPM

try:
    from DiffSBDD.edge_diffusion.transitions import (
        EdgeCategoricalTransition,
        EdgeTransitionConfig,
    )
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from DiffSBDD.edge_diffusion.transitions import (
        EdgeCategoricalTransition,
        EdgeTransitionConfig,
    )


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
    edge_loss_weight: float = 1.0
    edge_class_weight_mode: str = "diffgui"
    no_bond_weight: float = 0.1
    edge_focal_gamma: float = 0.0

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
            edge_loss_weight=float(_config_get(config, "edge_loss_weight", cls.edge_loss_weight)),
            edge_class_weight_mode=str(_config_get(
                config,
                "edge_class_weight_mode",
                cls.edge_class_weight_mode,
            )),
            no_bond_weight=float(_config_get(config, "no_bond_weight", cls.no_bond_weight)),
            edge_focal_gamma=float(_config_get(config, "edge_focal_gamma", cls.edge_focal_gamma)),
        )


@dataclass(frozen=True)
class XHEEdgeSmokeOutput:
    """Output of the edge-forward smoke pass."""

    pred_edge_logits: torch.Tensor
    noised_edge_type: torch.Tensor
    target_edge_type: torch.Tensor
    loss: torch.Tensor
    loss_per_molecule: torch.Tensor
    accuracy: torch.Tensor
    real_bond_accuracy: torch.Tensor


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
        """Run one denoiser pass that predicts x/h noise and clean edge logits."""
        _validate_ligand_halfedges(ligand)
        if not getattr(self.dynamics, "supports_xhe_edge_logits", False):
            raise TypeError("ConditionalDDPMXHE requires dynamics with xhe edge-logit support")

        ligand, pocket = _clone_ligand_pocket(ligand, pocket)
        ligand, pocket = self.normalize(ligand, pocket)

        delta_log_px = self.delta_log_px(ligand["size"])

        # Match ConditionalDDPM.forward: evaluation avoids t=0 here and handles
        # the low-variance t=0 estimator in a second no-grad dynamics call.
        lowest_t = 0 if self.training else 1
        t_int = torch.randint(
            lowest_t,
            self.T + 1,
            size=(ligand["size"].size(0), 1),
            device=ligand["x"].device,
        ).float()
        s_int = t_int - 1
        t_is_zero = (t_int == 0).float()
        t_is_not_zero = 1 - t_is_zero
        s = s_int / self.T
        t = t_int / self.T

        gamma_s = self.inflate_batch_array(self.gamma(s), ligand["x"])
        gamma_t = self.inflate_batch_array(self.gamma(t), ligand["x"])

        xh0_lig = torch.cat([ligand["x"], ligand["one_hot"]], dim=1)
        xh0_pocket = torch.cat([pocket["x"], pocket["one_hot"]], dim=1)
        xh0_lig[:, :self.n_dims], xh0_pocket[:, :self.n_dims] = \
            self.remove_mean_batch(
                xh0_lig[:, :self.n_dims],
                xh0_pocket[:, :self.n_dims],
                ligand["mask"],
                pocket["mask"],
            )

        z_t_lig, xh_pocket, eps_t_lig = self.noised_representation(
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

        net_out_lig, _, pred_edge_logits = self.dynamics(
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
        edge_output = self._edge_output_from_logits(
            pred_edge_logits=pred_edge_logits,
            noised_edge_type=noised_edge_type,
            target_edge_type=target_edge_type,
            edge_batch=edge_batch,
            batch_size=int(ligand["size"].shape[0]),
        )

        xh_lig_hat = self.xh_given_zt_and_epsilon(
            z_t_lig,
            net_out_lig,
            gamma_t,
            ligand["mask"],
        )

        squared_error = (eps_t_lig - net_out_lig) ** 2
        if self.vnode_idx is not None:
            squared_error[ligand["one_hot"][:, self.vnode_idx].bool(), :self.n_dims] = 0
        error_t_lig = self.sum_except_batch(squared_error, ligand["mask"])

        SNR_weight = (1 - self.SNR(gamma_s - gamma_t)).squeeze(1)
        assert error_t_lig.size() == SNR_weight.size()

        neg_log_constants = -self.log_constants_p_x_given_z0(
            n_nodes=ligand["size"],
            device=error_t_lig.device,
        )
        kl_prior = self.kl_prior(xh0_lig, ligand["mask"], ligand["size"])

        if self.training:
            log_p_x_given_z0_without_constants_ligand, log_ph_given_z0 = \
                self.log_pxh_given_z0_without_constants(
                    ligand,
                    z_t_lig,
                    eps_t_lig,
                    net_out_lig,
                    gamma_t,
                )
            loss_0_x_ligand = -log_p_x_given_z0_without_constants_ligand * \
                t_is_zero.squeeze()
            loss_0_h = -log_ph_given_z0 * t_is_zero.squeeze()
            error_t_lig = error_t_lig * t_is_not_zero.squeeze()
        else:
            t_zeros = torch.zeros_like(s)
            gamma_0 = self.inflate_batch_array(self.gamma(t_zeros), ligand["x"])
            z_0_lig, xh_pocket, eps_0_lig = self.noised_representation(
                xh0_lig,
                xh0_pocket,
                ligand["mask"],
                pocket["mask"],
                gamma_0,
            )
            net_out_0_lig, _ = self.dynamics(
                z_0_lig,
                xh_pocket,
                t_zeros,
                ligand["mask"],
                pocket["mask"],
            )
            log_p_x_given_z0_without_constants_ligand, log_ph_given_z0 = \
                self.log_pxh_given_z0_without_constants(
                    ligand,
                    z_0_lig,
                    eps_0_lig,
                    net_out_0_lig,
                    gamma_0,
                )
            loss_0_x_ligand = -log_p_x_given_z0_without_constants_ligand
            loss_0_h = -log_ph_given_z0

        log_pN = self.log_pN(ligand["size"], pocket["size"])
        info = {
            "eps_hat_lig_x": scatter_mean(
                net_out_lig[:, :self.n_dims].abs().mean(1),
                ligand["mask"],
                dim=0,
            ).mean(),
            "eps_hat_lig_h": scatter_mean(
                net_out_lig[:, self.n_dims:].abs().mean(1),
                ligand["mask"],
                dim=0,
            ).mean(),
            "_edge_loss_batch": edge_output.loss_per_molecule,
            "edge_smoke_loss": edge_output.loss.detach(),
            "edge_smoke_acc": edge_output.accuracy.detach(),
            "edge_smoke_real_bond_acc": edge_output.real_bond_accuracy.detach(),
            "edge_smoke_logits_abs": edge_output.pred_edge_logits.detach().abs().mean()
            if edge_output.pred_edge_logits.numel() > 0 else edge_output.loss.detach(),
        }
        loss_terms = (
            delta_log_px,
            error_t_lig,
            torch.tensor(0.0),
            SNR_weight,
            loss_0_x_ligand,
            torch.tensor(0.0),
            loss_0_h,
            neg_log_constants,
            kl_prior,
            log_pN,
            t_int.squeeze(),
            xh_lig_hat,
        )
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
        return self._edge_output_from_logits(
            pred_edge_logits=pred_edge_logits,
            noised_edge_type=noised_edge_type,
            target_edge_type=target_edge_type,
            edge_batch=edge_batch,
            batch_size=batch_size,
        )

    def _edge_output_from_logits(
        self,
        *,
        pred_edge_logits: torch.Tensor,
        noised_edge_type: torch.Tensor,
        target_edge_type: torch.Tensor,
        edge_batch: torch.Tensor,
        batch_size: int,
    ) -> XHEEdgeSmokeOutput:
        """Compute edge loss and metrics from one denoiser edge-logit output."""
        if target_edge_type.numel() == 0:
            loss_per_molecule = pred_edge_logits.new_zeros(batch_size)
            loss = loss_per_molecule.sum()
            accuracy = loss.detach()
            real_bond_accuracy = loss.detach()
        else:
            per_edge_loss = _edge_classification_loss(
                pred_edge_logits,
                target_edge_type,
                self.bond_diffusion_config,
            )
            loss_per_molecule = scatter_mean(
                per_edge_loss,
                edge_batch,
                dim=0,
                dim_size=batch_size,
            )
            loss = loss_per_molecule.mean()
            pred_edge_type = pred_edge_logits.argmax(dim=-1)
            accuracy = (pred_edge_type == target_edge_type).float().mean()
            real_bond_mask = target_edge_type > 0
            real_bond_accuracy = (pred_edge_type[real_bond_mask] == target_edge_type[real_bond_mask]).float().mean() \
                if bool(real_bond_mask.any()) else loss.detach() * 0.0
        return XHEEdgeSmokeOutput(
            pred_edge_logits=pred_edge_logits,
            noised_edge_type=noised_edge_type,
            target_edge_type=target_edge_type,
            loss=loss,
            loss_per_molecule=loss_per_molecule,
            accuracy=accuracy,
            real_bond_accuracy=real_bond_accuracy,
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


def _edge_classification_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    config: BondDiffusionConfig,
) -> torch.Tensor:
    """Return per-halfedge weighted CE/focal loss for imbalanced bond labels."""
    if logits.ndim != 2:
        raise ValueError("pred_edge_logits must have shape [num_halfedges, num_edge_types]")
    nll = F.cross_entropy(logits, target.long(), reduction="none")
    if config.edge_focal_gamma > 0:
        prob = torch.exp(-nll)
        nll = ((1.0 - prob).clamp_min(0.0) ** config.edge_focal_gamma) * nll
    class_weight = _edge_class_weight(config, logits.device, logits.dtype)
    if class_weight is not None:
        nll = class_weight[target.long()] * nll
    return nll


def _edge_class_weight(
    config: BondDiffusionConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """Return class weights borrowed from the existing edge-denoiser strategy."""
    mode = config.edge_class_weight_mode
    if mode == "none":
        return None
    if mode == "diffgui":
        weights = torch.ones(config.num_edge_types, device=device, dtype=dtype)
        weights[0] = float(config.no_bond_weight)
        return weights
    raise ValueError(f"Unsupported edge_class_weight_mode: {mode}")
