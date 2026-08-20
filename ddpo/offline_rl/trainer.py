"""Decoupled BC/AWR actor updates and Implicit Q-Learning train steps."""

from __future__ import annotations

import copy
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn
from torch.nn import functional as F

from .critic_adapter import (
    PocketContext,
    build_geometric_critic_input,
    build_next_geometric_critic_input,
)
from .losses import (
    QLossType,
    compute_actor_loss,
    compute_advantage,
    compute_advantage_weight,
    compute_q_loss,
    compute_td_target,
    compute_value_loss,
    effective_sample_size,
)
from .policy_adapter import DiffSBDDPolicyAdapter
from .protocol import DenoisingTransitionBatch
from .target import polyak_update


ActorObjective = Literal["bc", "awr", "iql"]


def _unwrap_module(module: nn.Module) -> nn.Module:
    """Return a DDP-wrapped module's trainable implementation when present."""
    wrapped = getattr(module, "module", None)
    return wrapped if isinstance(wrapped, nn.Module) else module


def _validate_optimizer_coverage(
    module: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    component: str,
) -> None:
    """Require an optimizer to cover every and only trainable parameter."""
    expected = {
        id(parameter)
        for parameter in module.parameters()
        if parameter.requires_grad
    }
    actual = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if expected != actual:
        raise ValueError(
            f"{component} optimizer parameter coverage mismatch: "
            f"missing={len(expected - actual)} extra={len(actual - expected)}"
        )


@dataclass(frozen=True)
class ActorUpdateConfig:
    """Configuration for BC, terminal-return AWR, or IQL actor updates."""

    objective: ActorObjective = "iql"
    temperature: float = 3.0
    max_weight: float = 100.0
    max_grad_norm: float = 10.0
    awr_reward_mean: float | None = None
    awr_reward_std: float | None = None

    def __post_init__(self) -> None:
        """Validate actor objective and numerical controls."""
        if self.objective not in ("bc", "awr", "iql"):
            raise ValueError(f"unsupported actor objective {self.objective!r}")
        for name, value in (
            ("temperature", self.temperature),
            ("max_weight", self.max_weight),
            ("max_grad_norm", self.max_grad_norm),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.objective == "awr":
            if self.awr_reward_mean is None or self.awr_reward_std is None:
                raise ValueError(
                    "AWR requires fixed train-split reward mean and std"
                )
            if not math.isfinite(self.awr_reward_mean):
                raise ValueError("awr_reward_mean must be finite")
            if (
                not math.isfinite(self.awr_reward_std)
                or self.awr_reward_std <= 0
            ):
                raise ValueError("awr_reward_std must be finite and positive")


@dataclass(frozen=True)
class ActorStepMetrics:
    """Scalar diagnostics from one actor optimizer update."""

    loss: float
    log_prob_mean: float
    weight_mean: float
    weight_max: float
    effective_sample_size: float
    clipped_fraction: float
    grad_norm: float


@dataclass(frozen=True)
class IQLTrainerConfig:
    """Fixed numerical choices for one IQL update stream."""

    num_diffusion_steps: int
    discount: float = 1.0
    expectile: float = 0.7
    target_tau: float = 0.005
    q_loss_type: QLossType = "mse"
    critic_max_grad_norm: float = 10.0
    actor_warmup_steps: int = 0
    actor_update_interval: int = 1

    def __post_init__(self) -> None:
        """Validate IQL hyperparameters."""
        if self.num_diffusion_steps <= 0:
            raise ValueError("num_diffusion_steps must be positive")
        if not 0.0 <= self.discount <= 1.0:
            raise ValueError("discount must lie in [0,1]")
        if not 0.0 < self.expectile < 1.0:
            raise ValueError("expectile must lie in (0,1)")
        if not 0.0 <= self.target_tau <= 1.0:
            raise ValueError("target_tau must lie in [0,1]")
        if self.q_loss_type not in ("mse", "smooth_l1"):
            raise ValueError("q_loss_type must be mse or smooth_l1")
        if (
            not math.isfinite(self.critic_max_grad_norm)
            or self.critic_max_grad_norm <= 0
        ):
            raise ValueError("critic_max_grad_norm must be finite and positive")
        if self.actor_warmup_steps < 0:
            raise ValueError("actor_warmup_steps must be non-negative")
        if self.actor_update_interval <= 0:
            raise ValueError("actor_update_interval must be positive")


@dataclass(frozen=True)
class IQLStepMetrics:
    """Scalar diagnostics from one complete V/Q/target/optional actor step."""

    global_step: int
    value_loss: float
    q_loss: float
    q1_loss: float
    q2_loss: float
    value_grad_norm: float
    q_grad_norm: float
    q1_mean: float
    q2_mean: float
    value_mean: float
    td_target_mean: float
    td_target_std: float
    td_target_min: float
    td_target_max: float
    advantage_mean: float
    advantage_std: float
    q_disagreement_mean: float
    actor_updated: bool
    actor: ActorStepMetrics | None
    gradient_loss_scale: float = 1.0


def _optimizer_step(
    *,
    loss: torch.Tensor,
    module: nn.Module,
    optimizer: torch.optim.Optimizer,
    max_grad_norm: float,
    component: str,
    gradient_loss_scale: float = 1.0,
) -> float:
    """Run one finite checked backward/clip/step and return pre-clip norm.

    ``gradient_loss_scale`` affects only the backward objective. It lets a DDP
    coordinator correct unequal rank ownership while keeping reported losses in
    the original per-batch units.
    """
    if loss.ndim != 0 or not bool(torch.isfinite(loss)):
        raise FloatingPointError(f"{component} loss must be a finite scalar")
    if (
        not math.isfinite(gradient_loss_scale)
        or gradient_loss_scale <= 0.0
    ):
        raise ValueError("gradient_loss_scale must be finite and positive")
    optimizer.zero_grad(set_to_none=True)
    (loss * gradient_loss_scale).backward()
    gradients = [
        parameter.grad
        for parameter in module.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not gradients:
        raise RuntimeError(f"{component} update produced no trainable gradients")
    if any(not bool(torch.isfinite(gradient).all()) for gradient in gradients):
        raise FloatingPointError(f"{component} gradients contain NaN or Inf")
    grad_norm = torch.nn.utils.clip_grad_norm_(
        module.parameters(),
        max_norm=max_grad_norm,
        error_if_nonfinite=True,
    )
    optimizer.step()
    return float(grad_norm.detach().cpu())


class OfflineActorUpdater:
    """Apply one policy NLL update under BC, AWR, or supplied IQL advantage."""

    def __init__(
        self,
        *,
        policy_adapter: DiffSBDDPolicyAdapter,
        optimizer: torch.optim.Optimizer,
        config: ActorUpdateConfig,
    ) -> None:
        """Initialize the actor update boundary."""
        self.policy_adapter = policy_adapter
        self.optimizer = optimizer
        self.config = config
        self.updates = 0
        _validate_optimizer_coverage(
            self.policy,
            self.optimizer,
            component="actor",
        )

    @property
    def policy(self) -> nn.Module:
        """Return the trainable policy module."""
        return self.policy_adapter.policy

    def step(
        self,
        transitions: DenoisingTransitionBatch,
        *,
        iql_advantage: torch.Tensor | None = None,
        gradient_loss_scale: float = 1.0,
    ) -> ActorStepMetrics:
        """Update the actor once with detached objective-specific weights."""
        score: torch.Tensor | None
        if self.config.objective == "bc":
            if iql_advantage is not None:
                raise ValueError("BC actor update does not accept IQL advantage")
            score = None
            weight = transitions.state_latent.new_ones(
                transitions.batch_size
            )
        elif self.config.objective == "awr":
            if iql_advantage is not None:
                raise ValueError("AWR actor update does not accept IQL advantage")
            if transitions.episode_return is None:
                raise ValueError("AWR requires episode_return from a reward sidecar")
            assert self.config.awr_reward_mean is not None
            assert self.config.awr_reward_std is not None
            score = (
                transitions.episode_return - self.config.awr_reward_mean
            ) / self.config.awr_reward_std
            weight = compute_advantage_weight(
                score,
                self.config.temperature,
                self.config.max_weight,
            )
        else:
            if iql_advantage is None:
                raise ValueError("IQL actor update requires detached advantage")
            if tuple(iql_advantage.shape) != (transitions.batch_size,):
                raise ValueError("IQL advantage must have shape [B]")
            score = iql_advantage.detach()
            weight = compute_advantage_weight(
                score,
                self.config.temperature,
                self.config.max_weight,
            )

        output = self.policy_adapter.log_prob(transitions)
        loss = compute_actor_loss(output.log_prob, weight)
        grad_norm = _optimizer_step(
            loss=loss,
            module=self.policy,
            optimizer=self.optimizer,
            max_grad_norm=self.config.max_grad_norm,
            component="actor",
            gradient_loss_scale=gradient_loss_scale,
        )
        clipped_fraction = 0.0
        if score is not None:
            clip_threshold = math.log(self.config.max_weight)
            clipped_fraction = float(
                (
                    score.detach() * self.config.temperature
                    > clip_threshold
                )
                .float()
                .mean()
                .cpu()
            )
        self.updates += 1
        return ActorStepMetrics(
            loss=float(loss.detach().cpu()),
            log_prob_mean=float(output.log_prob.detach().mean().cpu()),
            weight_mean=float(weight.mean().cpu()),
            weight_max=float(weight.max().cpu()),
            effective_sample_size=float(effective_sample_size(weight).cpu()),
            clipped_fraction=clipped_fraction,
            grad_norm=grad_norm,
        )

    def state_dict(self) -> dict[str, Any]:
        """Return actor module, optimizer, config, and update counter."""
        return {
            "format": "diffsbdd_offline_actor_updater_v2",
            "config": asdict(self.config),
            "policy": _unwrap_module(self.policy).state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "updates": self.updates,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore an actor updater with exact objective configuration."""
        if state.get("format") not in {
            "diffsbdd_offline_actor_updater_v1",
            "diffsbdd_offline_actor_updater_v2",
        }:
            raise ValueError("unsupported actor updater checkpoint format")
        if state.get("config") != asdict(self.config):
            raise ValueError("actor updater checkpoint config mismatch")
        _unwrap_module(self.policy).load_state_dict(state["policy"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.updates = int(state["updates"])


class OfflineIQLTrainer:
    """Execute fixed critic updates plus BC, AWR, or IQL actor updates."""

    def __init__(
        self,
        *,
        online_q: nn.Module,
        value: nn.Module,
        q_optimizer: torch.optim.Optimizer,
        value_optimizer: torch.optim.Optimizer,
        actor_updater: OfflineActorUpdater,
        pocket: PocketContext,
        config: IQLTrainerConfig,
    ) -> None:
        """Initialize online/target critics, optimizers, actor, and pocket."""
        self.online_q = online_q
        self._online_q_core = _unwrap_module(online_q)
        self.target_q = copy.deepcopy(self._online_q_core)
        self.target_q.requires_grad_(False)
        self.target_q.eval()
        self.value = value
        self._value_core = _unwrap_module(value)
        self.q_optimizer = q_optimizer
        self.value_optimizer = value_optimizer
        self.actor_updater = actor_updater
        self.pocket = pocket
        self.config = config
        self.global_step = 0
        _validate_optimizer_coverage(
            self._online_q_core,
            self.q_optimizer,
            component="twin_q",
        )
        _validate_optimizer_coverage(
            self._value_core,
            self.value_optimizer,
            component="value",
        )

    def step(
        self,
        transitions: DenoisingTransitionBatch,
        *,
        gradient_loss_scale: float = 1.0,
    ) -> IQLStepMetrics:
        """Run one finite checked IQL update on a scored transition batch."""
        if transitions.reward is None:
            raise ValueError("IQL requires sparse reward from a reward sidecar")
        if (
            not math.isfinite(gradient_loss_scale)
            or gradient_loss_scale <= 0.0
        ):
            raise ValueError("gradient_loss_scale must be finite and positive")
        inputs = build_geometric_critic_input(
            transitions,
            self.pocket,
            num_diffusion_steps=self.config.num_diffusion_steps,
        )
        next_inputs = build_next_geometric_critic_input(
            transitions,
            self.pocket,
            num_diffusion_steps=self.config.num_diffusion_steps,
        )

        with torch.no_grad():
            pre_update_q1, pre_update_q2 = self.target_q(inputs)
        value_prediction = self.value(inputs)
        value_loss = compute_value_loss(
            pre_update_q1,
            pre_update_q2,
            value_prediction,
            self.config.expectile,
        )
        value_grad_norm = _optimizer_step(
            loss=value_loss,
            module=self.value,
            optimizer=self.value_optimizer,
            max_grad_norm=self.config.critic_max_grad_norm,
            component="value",
            gradient_loss_scale=gradient_loss_scale,
        )

        with torch.no_grad():
            next_value = self.value(next_inputs)
            td_target = compute_td_target(
                transitions.reward,
                transitions.done,
                next_value,
                self.config.discount,
            )
        q1, q2 = self.online_q(inputs)
        q_loss = compute_q_loss(
            q1,
            q2,
            td_target,
            self.config.q_loss_type,
        )
        if self.config.q_loss_type == "mse":
            q1_loss_tensor = F.mse_loss(q1, td_target)
            q2_loss_tensor = F.mse_loss(q2, td_target)
        else:
            q1_loss_tensor = F.smooth_l1_loss(q1, td_target)
            q2_loss_tensor = F.smooth_l1_loss(q2, td_target)
        q_grad_norm = _optimizer_step(
            loss=q_loss,
            module=self._online_q_core,
            optimizer=self.q_optimizer,
            max_grad_norm=self.config.critic_max_grad_norm,
            component="twin_q",
            gradient_loss_scale=gradient_loss_scale,
        )
        polyak_update(
            self._online_q_core,
            self.target_q,
            self.config.target_tau,
        )

        with torch.no_grad():
            updated_value = self.value(inputs)
            advantage = compute_advantage(
                pre_update_q1,
                pre_update_q2,
                updated_value,
            )
        actor_due = (
            self.global_step >= self.config.actor_warmup_steps
            and (
                self.global_step - self.config.actor_warmup_steps
            )
            % self.config.actor_update_interval
            == 0
        )
        actor_metrics: ActorStepMetrics | None = None
        if actor_due:
            actor_metrics = self.actor_updater.step(
                transitions,
                iql_advantage=(
                    advantage
                    if self.actor_updater.config.objective == "iql"
                    else None
                ),
                gradient_loss_scale=gradient_loss_scale,
            )
        metrics = IQLStepMetrics(
            global_step=self.global_step,
            value_loss=float(value_loss.detach().cpu()),
            q_loss=float(q_loss.detach().cpu()),
            q1_loss=float(q1_loss_tensor.detach().cpu()),
            q2_loss=float(q2_loss_tensor.detach().cpu()),
            value_grad_norm=value_grad_norm,
            q_grad_norm=q_grad_norm,
            q1_mean=float(q1.detach().mean().cpu()),
            q2_mean=float(q2.detach().mean().cpu()),
            value_mean=float(updated_value.mean().cpu()),
            td_target_mean=float(td_target.mean().cpu()),
            td_target_std=float(td_target.std(unbiased=False).cpu()),
            td_target_min=float(td_target.min().cpu()),
            td_target_max=float(td_target.max().cpu()),
            advantage_mean=float(advantage.mean().cpu()),
            advantage_std=float(advantage.std(unbiased=False).cpu()),
            q_disagreement_mean=float(
                torch.abs(q1.detach() - q2.detach()).mean().cpu()
            ),
            actor_updated=actor_metrics is not None,
            actor=actor_metrics,
            gradient_loss_scale=gradient_loss_scale,
        )
        self.global_step += 1
        return metrics

    def state_dict(self) -> dict[str, Any]:
        """Return a complete resumable trainer checkpoint payload."""
        rng_state: dict[str, Any] = {
            "torch_cpu": torch.get_rng_state().clone(),
        }
        if torch.cuda.is_available():
            rng_state["torch_cuda_all"] = [
                state.clone() for state in torch.cuda.get_rng_state_all()
            ]
        return {
            "format": "diffsbdd_offline_iql_trainer_v2",
            "config": asdict(self.config),
            "global_step": self.global_step,
            "online_q": self._online_q_core.state_dict(),
            "target_q": self.target_q.state_dict(),
            "value": self._value_core.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "value_optimizer": self.value_optimizer.state_dict(),
            "actor_updater": self.actor_updater.state_dict(),
            "module_training": {
                "online_q": self._online_q_core.training,
                "value": self._value_core.training,
                "policy": _unwrap_module(self.actor_updater.policy).training,
            },
            "rng_state": rng_state,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore exact model/optimizer state and update counters."""
        if state.get("format") not in {
            "diffsbdd_offline_iql_trainer_v1",
            "diffsbdd_offline_iql_trainer_v2",
        }:
            raise ValueError("unsupported IQL trainer checkpoint format")
        if state.get("config") != asdict(self.config):
            raise ValueError("IQL trainer checkpoint config mismatch")
        self._online_q_core.load_state_dict(state["online_q"], strict=True)
        self.target_q.load_state_dict(state["target_q"], strict=True)
        self._value_core.load_state_dict(state["value"], strict=True)
        self.q_optimizer.load_state_dict(state["q_optimizer"])
        self.value_optimizer.load_state_dict(state["value_optimizer"])
        self.actor_updater.load_state_dict(state["actor_updater"])
        self.global_step = int(state["global_step"])
        module_training = state["module_training"]
        self._online_q_core.train(bool(module_training["online_q"]))
        self._value_core.train(bool(module_training["value"]))
        _unwrap_module(self.actor_updater.policy).train(
            bool(module_training["policy"])
        )
        rng_state = state["rng_state"]
        torch.set_rng_state(rng_state["torch_cpu"])
        if torch.cuda.is_available() and "torch_cuda_all" in rng_state:
            torch.cuda.set_rng_state_all(rng_state["torch_cuda_all"])
        self.target_q.requires_grad_(False)
        self.target_q.eval()

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        epoch: int,
        sampler_state: dict[str, Any],
        extra: dict[str, Any] | None = None,
    ) -> Path:
        """Atomically save one epoch-bound resumable local checkpoint."""
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "trainer": self.state_dict(),
            "sampler": sampler_state,
            "epoch": epoch,
            "extra": {} if extra is None else dict(extra),
        }
        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
        try:
            with temporary_path.open("wb") as handle:
                torch.save(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            temporary_path.replace(output_path)
        except Exception as exc:
            temporary_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"failed to save IQL checkpoint {output_path}"
            ) from exc
        return output_path
