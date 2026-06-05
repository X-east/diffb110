"""Standalone sampler for optional ligand x/h/e edge diffusion experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from DiffSBDD.edge_diffusion.dataset import EdgeBatch
from DiffSBDD.edge_diffusion.model import EdgeDiffusionModel
from DiffSBDD.edge_diffusion.schema import EdgeDiffusionInputs, EdgeDiffusionOutput
from DiffSBDD.edge_diffusion.transitions import EdgeCategoricalTransition


EdgeSamplerMode = Literal["reconstruct", "prior"]


@dataclass(frozen=True)
class EdgeSamplerConfig:
    """Configuration for a decoupled x/h/e sampler smoke run."""

    mode: EdgeSamplerMode = "reconstruct"
    start_timestep: int = 50
    pos_noise_std: float = 0.05
    pos_noise_mode: Literal["constant", "t_scaled"] = "t_scaled"
    atom_corruption_prob: float = 0.05
    atom_temperature: float = 1.0
    edge_temperature: float = 1.0
    position_prior_scale: float = 1.0


@dataclass(frozen=True)
class EdgeSamplerResult:
    """Final clean-state prediction and lightweight sampler diagnostics."""

    final_output: EdgeDiffusionOutput
    pred_pos: torch.Tensor
    pred_atom_type: torch.Tensor
    pred_edge_type: torch.Tensor
    initial_inputs: EdgeDiffusionInputs
    final_inputs: EdgeDiffusionInputs
    mode: EdgeSamplerMode
    start_timestep: int
    steps: int


class EdgeDiffusionSampler:
    """Run a small reverse chain for the standalone edge diffusion branch.

    This class intentionally depends only on `DiffSBDD.edge_diffusion` objects.
    It does not import DDPO, rewards, rollout code, or regular DiffSBDD RL
    launch helpers, so the edge branch can be smoke-tested without touching the
    existing reinforcement-learning pipeline.
    """

    def __init__(
        self,
        model: EdgeDiffusionModel,
        transition: EdgeCategoricalTransition,
        config: EdgeSamplerConfig | None = None,
        *,
        atom_prior: torch.Tensor | None = None,
    ) -> None:
        """Initialize the sampler with a denoiser, edge transition, and priors."""
        self.model = model
        self.transition = transition
        self.config = config or EdgeSamplerConfig()
        self.atom_prior = None if atom_prior is None else _normalize_prior(atom_prior)
        self._validate_config()

    @torch.no_grad()
    def sample(
        self,
        batch: EdgeBatch,
        *,
        generator: torch.Generator | None = None,
    ) -> EdgeSamplerResult:
        """Sample clean x/h/e predictions for a collated edge batch."""
        if not batch.names:
            raise ValueError("batch must contain at least one molecule")
        device = batch.ligand_pos.device
        generator = _ensure_generator(generator, device)
        start_timestep = self._clamped_start_timestep()

        initial_inputs = self._make_initial_inputs(
            batch=batch,
            timestep=start_timestep,
            generator=generator,
        )
        current_pos = initial_inputs.ligand_pos_t
        current_atom_type = initial_inputs.ligand_atom_type_t
        current_edge_type = initial_inputs.ligand_edge_type_t

        was_training = bool(self.model.training)
        self.model.eval()
        final_output: EdgeDiffusionOutput | None = None
        try:
            for step in range(start_timestep, -1, -1):
                t = torch.full(
                    (len(batch.names),),
                    step,
                    dtype=torch.long,
                    device=device,
                )
                inputs = EdgeDiffusionInputs(
                    ligand_pos_t=current_pos,
                    ligand_atom_type_t=current_atom_type,
                    ligand_edge_type_t=current_edge_type,
                    t=t,
                )
                output = self.model(batch, inputs)
                final_output = output
                if step == 0:
                    current_pos = output.pred_pos
                    current_atom_type = output.pred_atom_type
                    current_edge_type = output.pred_edge_type
                    break
                current_pos = self._reverse_position_step(
                    pred_pos=output.pred_pos,
                    next_timestep=step - 1,
                    batch=batch,
                    generator=generator,
                )
                current_atom_type = self._reverse_atom_step(
                    pred_atom_logits=output.pred_atom_logits,
                    next_timestep=step - 1,
                    batch=batch,
                    generator=generator,
                )
                current_edge_type = self._reverse_edge_step(
                    pred_edge_logits=output.pred_edge_logits,
                    current_edge_type=current_edge_type,
                    current_timestep=step,
                    batch=batch,
                    generator=generator,
                )
        finally:
            if was_training:
                self.model.train()

        if final_output is None:
            raise RuntimeError("sampler loop did not produce model output")
        final_inputs = EdgeDiffusionInputs(
            ligand_pos_t=current_pos,
            ligand_atom_type_t=current_atom_type,
            ligand_edge_type_t=current_edge_type,
            t=torch.zeros((len(batch.names),), dtype=torch.long, device=device),
        )
        return EdgeSamplerResult(
            final_output=final_output,
            pred_pos=current_pos.detach(),
            pred_atom_type=current_atom_type.detach(),
            pred_edge_type=current_edge_type.detach(),
            initial_inputs=initial_inputs,
            final_inputs=final_inputs,
            mode=self.config.mode,
            start_timestep=start_timestep,
            steps=start_timestep + 1,
        )

    def _make_initial_inputs(
        self,
        batch: EdgeBatch,
        timestep: int,
        generator: torch.Generator,
    ) -> EdgeDiffusionInputs:
        """Create x_t, h_t, and e_t for reconstruct or prior smoke mode."""
        device = batch.ligand_pos.device
        t = torch.full((len(batch.names),), timestep, dtype=torch.long, device=device)
        if self.config.mode == "reconstruct":
            edge_type_t = self.transition.q_sample(
                batch.target_edge_type,
                t[batch.edge_batch],
                generator=generator,
            )
            pos_t = self._noise_positions(batch.ligand_pos, t, batch.atom_batch, generator)
            atom_type_t = self._noise_atom_types(batch.ligand_atom_type, t, batch.atom_batch, generator)
        elif self.config.mode == "prior":
            edge_type_t, _, _ = self.transition.sample_init(
                int(batch.target_edge_type.numel()),
                device=device,
                generator=generator,
            )
            pos_t = self._sample_position_prior(batch, generator)
            atom_type_t = self._sample_atom_prior(
                int(batch.ligand_atom_type.numel()),
                device=device,
                generator=generator,
            )
        else:
            raise ValueError(f"Unsupported sampler mode: {self.config.mode!r}")
        return EdgeDiffusionInputs(
            ligand_pos_t=pos_t,
            ligand_atom_type_t=atom_type_t,
            ligand_edge_type_t=edge_type_t,
            t=t,
        )

    def _reverse_position_step(
        self,
        pred_pos: torch.Tensor,
        next_timestep: int,
        batch: EdgeBatch,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """Approximate x_{t-1} by re-noising predicted clean coordinates."""
        if next_timestep <= 0 or self.config.pos_noise_std <= 0:
            return pred_pos
        t = torch.full(
            (len(batch.names),),
            int(next_timestep),
            dtype=torch.long,
            device=pred_pos.device,
        )
        return self._noise_positions(pred_pos, t, batch.atom_batch, generator)

    def _reverse_atom_step(
        self,
        pred_atom_logits: torch.Tensor,
        next_timestep: int,
        batch: EdgeBatch,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """Sample h_0 from logits and apply the simple atom corruption used in training."""
        atom_type = _sample_logits(
            pred_atom_logits,
            temperature=float(self.config.atom_temperature),
            generator=generator,
        )
        if next_timestep <= 0:
            return atom_type
        t = torch.full(
            (len(batch.names),),
            int(next_timestep),
            dtype=torch.long,
            device=pred_atom_logits.device,
        )
        return self._noise_atom_types(atom_type, t, batch.atom_batch, generator)

    def _reverse_edge_step(
        self,
        pred_edge_logits: torch.Tensor,
        current_edge_type: torch.Tensor,
        current_timestep: int,
        batch: EdgeBatch,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """Sample e_{t-1} from q(e_{t-1} | e_t, predicted e_0)."""
        if current_timestep <= 0:
            return pred_edge_logits.argmax(dim=-1)
        log_v0 = F.log_softmax(
            pred_edge_logits / float(self.config.edge_temperature),
            dim=-1,
        )
        log_vt = self.transition.index_to_log_onehot(current_edge_type)
        t = torch.full(
            (len(batch.names),),
            int(current_timestep),
            dtype=torch.long,
            device=pred_edge_logits.device,
        )
        posterior = self.transition.q_v_posterior(
            log_v0,
            log_vt,
            t,
            batch=batch.edge_batch,
            v0_prob=True,
        )
        return self.transition.log_sample_categorical(posterior, generator=generator)

    def _noise_positions(
        self,
        pos: torch.Tensor,
        t: torch.Tensor,
        atom_batch: torch.Tensor,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """Add the same simple Gaussian coordinate corruption used by training."""
        if self.config.pos_noise_std <= 0:
            return pos
        noise = torch.randn(pos.shape, device=pos.device, dtype=pos.dtype, generator=generator)
        if self.config.pos_noise_mode == "constant":
            scale = torch.full(
                (pos.shape[0], 1),
                float(self.config.pos_noise_std),
                device=pos.device,
                dtype=pos.dtype,
            )
        else:
            t_per_atom = t[atom_batch.long()].float() / float(max(1, self.transition.num_timesteps - 1))
            scale = float(self.config.pos_noise_std) * t_per_atom.unsqueeze(-1)
        return pos + noise * scale

    def _noise_atom_types(
        self,
        atom_type: torch.Tensor,
        t: torch.Tensor,
        atom_batch: torch.Tensor,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """Apply the simple timestep-scaled atom corruption used by training."""
        if self.config.atom_corruption_prob <= 0:
            return atom_type
        t_per_atom = t[atom_batch.long()].float() / float(max(1, self.transition.num_timesteps - 1))
        corruption_prob = (float(self.config.atom_corruption_prob) * t_per_atom).clamp(max=1.0)
        random_values = torch.rand(atom_type.shape, device=atom_type.device, generator=generator)
        corrupt_mask = random_values < corruption_prob
        random_atom_type = self._sample_atom_prior(
            int(atom_type.numel()),
            device=atom_type.device,
            generator=generator,
        )
        return torch.where(corrupt_mask, random_atom_type, atom_type)

    def _sample_position_prior(
        self,
        batch: EdgeBatch,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """Sample prior positions per molecule while preserving atom counts."""
        pos = torch.empty_like(batch.ligand_pos)
        for molecule_idx in range(len(batch.names)):
            mask = batch.atom_batch == molecule_idx
            true_pos = batch.ligand_pos[mask]
            if true_pos.numel() == 0:
                continue
            center = true_pos.mean(dim=0, keepdim=True)
            radius = torch.linalg.norm(true_pos - center, dim=-1).std().clamp_min(1.0)
            noise = torch.randn(true_pos.shape, device=true_pos.device, dtype=true_pos.dtype, generator=generator)
            pos[mask] = center + noise * radius * float(self.config.position_prior_scale)
        return pos

    def _sample_atom_prior(
        self,
        n: int,
        *,
        device: torch.device,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """Sample atom labels from the configured or uniform prior."""
        if n < 0:
            raise ValueError("n must be non-negative")
        if self.atom_prior is None:
            logits = torch.zeros((n, self.model.config.num_atom_types), device=device)
        else:
            prior = self.atom_prior.to(device=device)
            logits = torch.log(prior.clamp_min(1e-30)).unsqueeze(0).repeat(n, 1)
        return _sample_logits(logits, temperature=1.0, generator=generator)

    def _clamped_start_timestep(self) -> int:
        """Return the requested start timestep clamped to transition bounds."""
        return int(max(0, min(int(self.config.start_timestep), self.transition.num_timesteps - 1)))

    def _validate_config(self) -> None:
        """Validate sampler configuration values."""
        if self.config.mode not in {"reconstruct", "prior"}:
            raise ValueError("mode must be 'reconstruct' or 'prior'")
        if self.config.pos_noise_mode not in {"constant", "t_scaled"}:
            raise ValueError("pos_noise_mode must be 'constant' or 't_scaled'")
        if self.config.pos_noise_std < 0:
            raise ValueError("pos_noise_std must be non-negative")
        if self.config.atom_corruption_prob < 0:
            raise ValueError("atom_corruption_prob must be non-negative")
        if self.config.atom_temperature <= 0:
            raise ValueError("atom_temperature must be positive")
        if self.config.edge_temperature <= 0:
            raise ValueError("edge_temperature must be positive")
        if self.config.position_prior_scale <= 0:
            raise ValueError("position_prior_scale must be positive")


def _sample_logits(
    logits: torch.Tensor,
    *,
    temperature: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample class labels from logits, using argmax at very low temperature."""
    if logits.numel() == 0:
        return logits.new_zeros((logits.shape[0],), dtype=torch.long)
    if temperature <= 1e-8:
        return logits.argmax(dim=-1)
    probs = torch.softmax(logits / float(temperature), dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


def _normalize_prior(prior: torch.Tensor) -> torch.Tensor:
    """Return a normalized one-dimensional probability vector."""
    if prior.ndim != 1:
        raise ValueError("atom_prior must be one-dimensional")
    if (prior < 0).any() or float(prior.sum().item()) <= 0:
        raise ValueError("atom_prior must be non-negative and have positive mass")
    return (prior.float() / prior.float().sum()).detach()


def _ensure_generator(
    generator: torch.Generator | None,
    device: torch.device,
) -> torch.Generator:
    """Create a generator on the sampler device when one is not provided."""
    if generator is not None:
        return generator
    return torch.Generator(device=device)
