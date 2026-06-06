"""Optional x/h/e conditional diffusion smoke path for DiffSBDD."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_mean

from equivariant_diffusion.conditional_model import ConditionalDDPM
import utils
from generation_refinement import apply_atom_score_bias, apply_local_atom_score_mask

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
    edge_prior: str = "absorb"
    beta_schedule: str = "segment"
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    scale_start: float = 0.9999
    scale_end: float = 0.0001
    width: float = 3.0
    time_segment: tuple[int, ...] | None = None
    segment_diff: tuple[dict[str, float], ...] | None = None
    edge_hidden_dim: int = 128
    distance_basis_dim: int = 32
    edge_head_dropout: float = 0.0
    edge_loss_mode: str = "diffusion_kl"
    edge_loss_scale: float = 100.0
    edge_loss_weight: float = 1.0
    edge_class_weight_mode: str = "none"
    no_bond_weight: float = 1.0
    edge_focal_gamma: float = 0.0
    use_edge_message_passing: bool = True
    edge_message_dim: int = 32
    edge_message_scale: float = 1.0
    bond_length_loss_weight: float = 1.0

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
            scale_start=float(_config_get(config, "scale_start", cls.scale_start)),
            scale_end=float(_config_get(config, "scale_end", cls.scale_end)),
            width=float(_config_get(config, "width", cls.width)),
            time_segment=_optional_int_tuple(_config_get(
                config,
                "time_segment",
                cls.time_segment,
            )),
            segment_diff=_optional_segment_diff(_config_get(
                config,
                "segment_diff",
                cls.segment_diff,
            )),
            edge_hidden_dim=int(_config_get(config, "edge_hidden_dim", cls.edge_hidden_dim)),
            distance_basis_dim=int(_config_get(config, "distance_basis_dim", cls.distance_basis_dim)),
            edge_head_dropout=float(_config_get(config, "edge_head_dropout", cls.edge_head_dropout)),
            edge_loss_mode=str(_config_get(config, "edge_loss_mode", cls.edge_loss_mode)),
            edge_loss_scale=float(_config_get(config, "edge_loss_scale", cls.edge_loss_scale)),
            edge_loss_weight=float(_config_get(config, "edge_loss_weight", cls.edge_loss_weight)),
            edge_class_weight_mode=str(_config_get(
                config,
                "edge_class_weight_mode",
                cls.edge_class_weight_mode,
            )),
            no_bond_weight=float(_config_get(config, "no_bond_weight", cls.no_bond_weight)),
            edge_focal_gamma=float(_config_get(config, "edge_focal_gamma", cls.edge_focal_gamma)),
            use_edge_message_passing=bool(_config_get(
                config,
                "use_edge_message_passing",
                cls.use_edge_message_passing,
            )),
            edge_message_dim=int(_config_get(config, "edge_message_dim", cls.edge_message_dim)),
            edge_message_scale=float(_config_get(config, "edge_message_scale", cls.edge_message_scale)),
            bond_length_loss_weight=float(_config_get(
                config,
                "bond_length_loss_weight",
                cls.bond_length_loss_weight,
            )),
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
                scale_start=self.bond_diffusion_config.scale_start,
                scale_end=self.bond_diffusion_config.scale_end,
                width=self.bond_diffusion_config.width,
                time_segment=self.bond_diffusion_config.time_segment,
                segment_diff=self.bond_diffusion_config.segment_diff,
            )
        )

    def sample_p_zs_given_zt_xhe(
        self,
        s: torch.Tensor,
        t: torch.Tensor,
        zt_lig: torch.Tensor,
        xh0_pocket: torch.Tensor,
        ligand_mask: torch.Tensor,
        pocket_mask: torch.Tensor,
        *,
        halfedge_index: torch.Tensor,
        edge_type_t: torch.Tensor,
        edge_batch: torch.Tensor,
        edge_temperature: float = 1.0,
        generator: torch.Generator | None = None,
        fix_noise: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample joint x/h/e state z_s, e_s from z_t, e_t."""
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = \
            self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, zt_lig)

        sigma_s = self.sigma(gamma_s, target_tensor=zt_lig)
        sigma_t = self.sigma(gamma_t, target_tensor=zt_lig)

        eps_t_lig, _, pred_edge_logits = self.dynamics(
            zt_lig,
            xh0_pocket,
            t,
            ligand_mask,
            pocket_mask,
            halfedge_index=halfedge_index,
            noised_edge_type=edge_type_t,
            edge_batch=edge_batch,
            return_edge_logits=True,
        )

        mu_lig = zt_lig / alpha_t_given_s[ligand_mask] - \
            (sigma2_t_given_s / alpha_t_given_s / sigma_t)[ligand_mask] * eps_t_lig
        sigma = sigma_t_given_s * sigma_s / sigma_t
        zs_lig, xh0_pocket = self.sample_normal_zero_com(
            mu_lig,
            xh0_pocket,
            sigma,
            ligand_mask,
            pocket_mask,
            fix_noise,
        )
        edge_type_s = self._sample_edge_posterior(
            pred_edge_logits=pred_edge_logits,
            edge_type_t=edge_type_t,
            t=t,
            edge_batch=edge_batch,
            edge_temperature=edge_temperature,
            generator=generator,
        )

        self.assert_mean_zero_with_mask(zt_lig[:, :self.n_dims], ligand_mask)
        return zs_lig, xh0_pocket, edge_type_s

    def sample_p_xh_given_z0_xhe(
        self,
        z0_lig: torch.Tensor,
        xh0_pocket: torch.Tensor,
        lig_mask: torch.Tensor,
        pocket_mask: torch.Tensor,
        batch_size: int,
        *,
        halfedge_index: torch.Tensor,
        edge_type_0: torch.Tensor,
        edge_batch: torch.Tensor,
        edge_temperature: float = 1.0,
        generator: torch.Generator | None = None,
        fix_noise: bool = False,
        atom_score_bias: torch.Tensor | None = None,
        atom_temperature: float = 1.0,
        atom_local_mask_config: Any | None = None,
        atom_decoder: list[str] | None = None,
        atom_score_diagnostics: Any | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample final x/h and predicted edge labels from z_0 and e_0."""
        t_zeros = torch.zeros(size=(batch_size, 1), device=z0_lig.device)
        gamma_0 = self.gamma(t_zeros)
        sigma_x = self.SNR(-0.5 * gamma_0)
        net_out_lig, _, pred_edge_logits = self.dynamics(
            z0_lig,
            xh0_pocket,
            t_zeros,
            lig_mask,
            pocket_mask,
            halfedge_index=halfedge_index,
            noised_edge_type=edge_type_0,
            edge_batch=edge_batch,
            return_edge_logits=True,
        )

        mu_x_lig = self.compute_x_pred(net_out_lig, z0_lig, gamma_0, lig_mask)
        xh_lig, xh0_pocket = self.sample_normal_zero_com(
            mu_x_lig,
            xh0_pocket,
            sigma_x,
            lig_mask,
            pocket_mask,
            fix_noise,
        )

        x_lig, h_lig = self.unnormalize(
            xh_lig[:, :self.n_dims],
            z0_lig[:, self.n_dims:],
        )
        x_pocket, h_pocket = self.unnormalize(
            xh0_pocket[:, :self.n_dims],
            xh0_pocket[:, self.n_dims:],
        )

        raw_h_lig = h_lig
        h_lig = apply_atom_score_bias(
            h_lig,
            atom_score_bias,
            temperature=atom_temperature,
        )
        if atom_local_mask_config is not None:
            if getattr(atom_local_mask_config, "enabled", False) and atom_decoder is None:
                raise ValueError("atom_decoder is required when atom_local_mask_config is enabled.")
            h_lig = apply_local_atom_score_mask(
                h_lig,
                x_lig,
                lig_mask,
                atom_decoder if atom_decoder is not None else [],
                atom_local_mask_config,
            )
        if atom_score_diagnostics is not None:
            atom_score_diagnostics.update(raw_h_lig, h_lig)
        h_lig = F.one_hot(torch.argmax(h_lig, dim=1), self.atom_nf)
        edge_type = _sample_logits(
            pred_edge_logits,
            temperature=edge_temperature,
            generator=generator,
        )
        return x_lig, h_lig, x_pocket, h_pocket, edge_type

    @torch.no_grad()
    def sample_given_pocket(
        self,
        pocket: dict[str, torch.Tensor],
        num_nodes_lig: torch.Tensor,
        return_frames: int = 1,
        timesteps: int | None = None,
        atom_score_bias: torch.Tensor | None = None,
        atom_temperature: float = 1.0,
        atom_local_mask_config: Any | None = None,
        atom_decoder: list[str] | None = None,
        atom_score_diagnostics: Any | None = None,
        *,
        return_edges: bool = False,
        edge_temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ):
        """Draw joint x/h/e samples while preserving the old return contract."""
        timesteps = self.T if timesteps is None else timesteps
        assert 0 < return_frames <= timesteps
        assert timesteps % return_frames == 0

        n_samples = len(pocket["size"])
        device = pocket["x"].device
        generator = generator or torch.Generator(device=device)

        _, pocket = self.normalize(pocket=pocket)
        xh0_pocket = torch.cat([pocket["x"], pocket["one_hot"]], dim=1)

        lig_mask = utils.num_nodes_to_batch_mask(
            n_samples,
            num_nodes_lig,
            device,
        )
        halfedge_index, edge_batch, num_halfedges = _build_sample_halfedges(
            num_nodes_lig,
            device=device,
        )
        edge_type_t, _, _ = self.edge_transition.sample_init(
            int(halfedge_index.shape[1]),
            device=device,
            generator=generator,
        )

        mu_lig_x = scatter_mean(pocket["x"], pocket["mask"], dim=0)
        mu_lig_h = torch.zeros((n_samples, self.atom_nf), device=device)
        mu_lig = torch.cat((mu_lig_x, mu_lig_h), dim=1)[lig_mask]
        sigma = torch.ones_like(pocket["size"]).unsqueeze(1)

        z_lig, xh_pocket = self.sample_normal_zero_com(
            mu_lig,
            xh0_pocket,
            sigma,
            lig_mask,
            pocket["mask"],
        )

        self.assert_mean_zero_with_mask(z_lig[:, :self.n_dims], lig_mask)

        out_lig = torch.zeros((return_frames,) + z_lig.size(), device=z_lig.device)
        out_pocket = torch.zeros((return_frames,) + xh_pocket.size(), device=device)

        for s_step in reversed(range(0, timesteps)):
            s_array = torch.full((n_samples, 1), fill_value=s_step, device=z_lig.device)
            t_array = s_array + 1
            s_array = s_array / timesteps
            t_array = t_array / timesteps

            z_lig, xh_pocket, edge_type_t = self.sample_p_zs_given_zt_xhe(
                s_array,
                t_array,
                z_lig,
                xh_pocket,
                lig_mask,
                pocket["mask"],
                halfedge_index=halfedge_index,
                edge_type_t=edge_type_t,
                edge_batch=edge_batch,
                edge_temperature=edge_temperature,
                generator=generator,
            )

            if (s_step * return_frames) % timesteps == 0:
                idx = (s_step * return_frames) // timesteps
                out_lig[idx], out_pocket[idx] = self.unnormalize_z(z_lig, xh_pocket)

        x_lig, h_lig, x_pocket, h_pocket, edge_type = self.sample_p_xh_given_z0_xhe(
            z_lig,
            xh_pocket,
            lig_mask,
            pocket["mask"],
            n_samples,
            halfedge_index=halfedge_index,
            edge_type_0=edge_type_t,
            edge_batch=edge_batch,
            edge_temperature=edge_temperature,
            generator=generator,
            atom_score_bias=atom_score_bias,
            atom_temperature=atom_temperature,
            atom_local_mask_config=atom_local_mask_config,
            atom_decoder=atom_decoder,
            atom_score_diagnostics=atom_score_diagnostics,
        )

        self.assert_mean_zero_with_mask(x_lig, lig_mask)

        if return_frames == 1:
            max_cog = scatter_add(x_lig, lig_mask, dim=0).abs().max().item()
            if max_cog > 5e-2:
                print(f"Warning CoG drift with error {max_cog:.3f}. Projecting the positions down.")
                x_lig, x_pocket = self.remove_mean_batch(
                    x_lig,
                    x_pocket,
                    lig_mask,
                    pocket["mask"],
                )

        out_lig[0] = torch.cat([x_lig, h_lig], dim=1)
        out_pocket[0] = torch.cat([x_pocket, h_pocket], dim=1)

        result = (
            out_lig.squeeze(0),
            out_pocket.squeeze(0),
            lig_mask,
            pocket["mask"],
        )
        if not return_edges:
            return result
        edge_bundle = {
            "halfedge_index": halfedge_index.detach(),
            "halfedge_type": edge_type.detach(),
            "edge_batch": edge_batch.detach(),
            "num_halfedges": num_halfedges.detach(),
        }
        return (*result, edge_bundle)

    def _sample_edge_posterior(
        self,
        *,
        pred_edge_logits: torch.Tensor,
        edge_type_t: torch.Tensor,
        t: torch.Tensor,
        edge_batch: torch.Tensor,
        edge_temperature: float,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """Sample e_{t-1} from the discrete posterior using predicted e_0 logits."""
        if edge_type_t.numel() == 0:
            return edge_type_t
        t_int = _normalized_t_to_edge_timestep(t, self.T)
        log_v0 = _log_softmax_with_temperature(pred_edge_logits, edge_temperature)
        log_vt = self.edge_transition.index_to_log_onehot(edge_type_t)
        posterior = self.edge_transition.q_v_posterior(
            log_v0,
            log_vt,
            t_int,
            batch=edge_batch.long(),
            v0_prob=True,
        )
        return self.edge_transition.log_sample_categorical(posterior, generator=generator)

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
        noised_edge_type, log_vt = self.edge_transition.q_vt_sample(
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
            log_vt=log_vt,
            target_edge_type=target_edge_type,
            log_v0=log_v0,
            t_int=t_int.reshape(-1).long(),
            edge_batch=edge_batch,
            batch_size=int(ligand["size"].shape[0]),
        )

        xh_lig_hat = self.xh_given_zt_and_epsilon(
            z_t_lig,
            net_out_lig,
            gamma_t,
            ligand["mask"],
        )
        bond_geometry_loss_batch = self._bond_length_loss_per_molecule(
            pred_x=xh_lig_hat[:, :self.n_dims],
            target_x=ligand["x"],
            halfedge_index=ligand["halfedge_index"],
            target_edge_type=target_edge_type,
            edge_batch=edge_batch,
            batch_size=int(ligand["size"].shape[0]),
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
            edge_type_0 = target_edge_type.clone()
            net_out_0_lig, _, _ = self.dynamics(
                z_0_lig,
                xh_pocket,
                t_zeros,
                ligand["mask"],
                pocket["mask"],
                halfedge_index=ligand["halfedge_index"],
                noised_edge_type=edge_type_0,
                edge_batch=edge_batch,
                return_edge_logits=True,
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
            "_bond_geometry_loss_batch": bond_geometry_loss_batch,
            "edge_smoke_loss": edge_output.loss.detach(),
            "edge_diffusion_loss": edge_output.loss.detach(),
            "edge_smoke_acc": edge_output.accuracy.detach(),
            "edge_smoke_real_bond_acc": edge_output.real_bond_accuracy.detach(),
            "bond_length_loss": bond_geometry_loss_batch.detach().mean(),
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
        noised_edge_type, log_vt = self.edge_transition.q_vt_sample(
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
            log_vt=log_vt,
            target_edge_type=target_edge_type,
            log_v0=log_v0,
            t_int=t_int.reshape(-1).long(),
            edge_batch=edge_batch,
            batch_size=batch_size,
        )

    def _edge_output_from_logits(
        self,
        *,
        pred_edge_logits: torch.Tensor,
        noised_edge_type: torch.Tensor,
        log_vt: torch.Tensor,
        target_edge_type: torch.Tensor,
        log_v0: torch.Tensor,
        t_int: torch.Tensor,
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
            per_edge_loss = self._edge_loss_per_halfedge(
                pred_edge_logits=pred_edge_logits,
                noised_edge_type=noised_edge_type,
                log_vt=log_vt,
                target_edge_type=target_edge_type,
                log_v0=log_v0,
                t_int=t_int,
                edge_batch=edge_batch,
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

    def _edge_loss_per_halfedge(
        self,
        *,
        pred_edge_logits: torch.Tensor,
        noised_edge_type: torch.Tensor,
        log_vt: torch.Tensor,
        target_edge_type: torch.Tensor,
        log_v0: torch.Tensor,
        t_int: torch.Tensor,
        edge_batch: torch.Tensor,
    ) -> torch.Tensor:
        """Return DiffGui-style edge diffusion loss or explicit legacy CE."""
        del noised_edge_type
        mode = self.bond_diffusion_config.edge_loss_mode
        if mode == "diffusion_kl":
            log_edge_recon = F.log_softmax(pred_edge_logits, dim=-1)
            log_edge_post_true = self.edge_transition.q_v_posterior(
                log_v0,
                log_vt,
                t_int,
                batch=edge_batch.long(),
                v0_prob=True,
            )
            log_edge_post_pred = self.edge_transition.q_v_posterior(
                log_edge_recon,
                log_vt,
                t_int,
                batch=edge_batch.long(),
                v0_prob=True,
            )
            per_edge_loss = self.edge_transition.compute_v_Lt(
                log_edge_post_true,
                log_edge_post_pred,
                log_v0,
                t=t_int,
                batch=edge_batch.long(),
            )
        elif mode == "ce":
            per_edge_loss = _edge_classification_loss(
                pred_edge_logits,
                target_edge_type,
                self.bond_diffusion_config,
            )
        else:
            raise ValueError("Unsupported edge_loss_mode. Expected diffusion_kl or ce.")
        return per_edge_loss * float(self.bond_diffusion_config.edge_loss_scale)

    def _bond_length_loss_per_molecule(
        self,
        *,
        pred_x: torch.Tensor,
        target_x: torch.Tensor,
        halfedge_index: torch.Tensor,
        target_edge_type: torch.Tensor,
        edge_batch: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Return smooth L1 x0 bond-length loss for true ligand bonds."""
        if (
            self.bond_diffusion_config.bond_length_loss_weight <= 0
            or target_edge_type.numel() == 0
        ):
            return pred_x.new_zeros(batch_size)
        bond_mask = target_edge_type.long() > 0
        if not bool(bond_mask.any()):
            return pred_x.new_zeros(batch_size)
        bond_index = halfedge_index[:, bond_mask].long()
        src, dst = bond_index
        pred_distance = torch.linalg.norm(pred_x[src] - pred_x[dst], dim=-1)
        target_distance = torch.linalg.norm(target_x[src] - target_x[dst], dim=-1)
        per_bond_loss = F.smooth_l1_loss(
            pred_distance,
            target_distance.detach(),
            reduction="none",
        )
        return scatter_mean(
            per_bond_loss,
            edge_batch[bond_mask].long(),
            dim=0,
            dim_size=batch_size,
        )


def _optional_int_tuple(value: Any) -> tuple[int, ...] | None:
    """Return ``value`` as an int tuple, preserving ``None``."""
    if value is None:
        return None
    if torch.is_tensor(value):
        value = value.detach().cpu().reshape(-1).tolist()
    return tuple(int(item) for item in value)


def _optional_segment_diff(value: Any) -> tuple[dict[str, float], ...] | None:
    """Return DiffGui-style segment params as immutable dictionaries."""
    if value is None:
        return None
    segment_params = []
    for item in value:
        if isinstance(item, dict):
            getter = item.get
        else:
            getter = lambda key, default=None, item=item: getattr(item, key, default)
        segment_params.append({
            "scale_start": float(getter("scale_start", 0.9999)),
            "scale_end": float(getter("scale_end", 0.0001)),
            "width": float(getter("width", 3.0)),
        })
    return tuple(segment_params)


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


def _log_softmax_with_temperature(
    logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Return log probabilities, using argmax when temperature is non-positive."""
    if float(temperature) <= 0:
        hard = F.one_hot(logits.argmax(dim=-1), logits.shape[-1]).to(dtype=logits.dtype)
        return torch.log(hard.clamp_min(1e-30))
    return F.log_softmax(logits / float(temperature), dim=-1)


def _build_sample_halfedges(
    num_nodes_lig: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build concatenated upper-triangle halfedges for generated ligand sizes."""
    edge_indices = []
    edge_batches = []
    edge_counts = []
    node_offset = 0
    for batch_idx, num_nodes_tensor in enumerate(num_nodes_lig.detach().cpu().long().tolist()):
        num_nodes = int(num_nodes_tensor)
        if num_nodes < 0:
            raise ValueError("num_nodes_lig must be non-negative")
        local_edges = torch.triu_indices(num_nodes, num_nodes, offset=1, device=device)
        edge_count = int(local_edges.shape[1])
        if edge_count > 0:
            edge_indices.append(local_edges + int(node_offset))
            edge_batches.append(torch.full((edge_count,), batch_idx, dtype=torch.long, device=device))
        edge_counts.append(edge_count)
        node_offset += num_nodes
    if edge_indices:
        halfedge_index = torch.cat(edge_indices, dim=1)
        edge_batch = torch.cat(edge_batches, dim=0)
    else:
        halfedge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        edge_batch = torch.zeros((0,), dtype=torch.long, device=device)
    return (
        halfedge_index,
        edge_batch,
        torch.tensor(edge_counts, dtype=torch.long, device=device),
    )


def _normalized_t_to_edge_timestep(t: torch.Tensor, total_steps: int) -> torch.Tensor:
    """Convert normalized continuous diffusion t values to edge transition ids."""
    if t.ndim == 0:
        t_flat = t.reshape(1)
    else:
        t_flat = t.reshape(t.shape[0], -1)[:, 0]
    return torch.round(t_flat.float() * float(total_steps)).long().clamp(0, total_steps - 1)


def _sample_logits(
    logits: torch.Tensor,
    *,
    temperature: float,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Sample class labels from logits, falling back to argmax at zero temperature."""
    if logits.numel() == 0:
        return logits.new_zeros((logits.shape[0],), dtype=torch.long)
    if temperature <= 1e-8:
        return logits.argmax(dim=-1)
    probabilities = torch.softmax(logits / float(temperature), dim=-1)
    return torch.multinomial(probabilities, num_samples=1, generator=generator).squeeze(-1)
