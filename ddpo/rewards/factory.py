"""Reward factory for the Lightning DDPO mainline.

Goal:
- make the effective reward structure derivable from config
- keep entrypoint assembly thin
- preserve current reward semantics unless config explicitly changes them
"""

from __future__ import annotations

import logging
from typing import Dict

from ddpo.rewards.composite import CompositeReward
from ddpo.rewards.diversity_reward import DiversityReward
from ddpo.rewards.qed_reward import QEDReward
from ddpo.rewards.sa_reward import SAReward
from ddpo.rewards.docking_factory import build_docking_reward
from ddpo.rewards.stability import StabilityAwareReward



def _get_enabled_rewards(config: dict) -> list[str]:
    reward_cfg = config.get("reward", {})
    explicit_enabled = reward_cfg.get("enabled")
    if explicit_enabled is not None:
        return list(explicit_enabled)

    weights = reward_cfg.get("weights", {})
    enabled = [name for name, value in weights.items() if float(value) > 0.0]
    if enabled:
        return enabled

    return ["qed", "sa"]


def build_rewarder(config: dict, dataset_info: dict, logger: logging.Logger):
    """Build the effective rewarder for the Lightning mainline.

    Current policy:
    - derive enabled components from `reward.enabled` when present
    - otherwise infer from positive `reward.weights`
    - preserve current defaults for DruglikenessReward parameters
    - only instantiate DockingReward when explicitly enabled
    """
    reward_cfg = config.get("reward", {})
    enabled = _get_enabled_rewards(config)
    weights_cfg = reward_cfg.get("weights", {})

    rewards_dict: Dict[str, object] = {}
    weights_dict: Dict[str, float] = {}

    if "qed" in enabled:
        qed_cfg = reward_cfg.get("qed", {})
        rewards_dict["qed"] = QEDReward(
            failure_penalty=float(qed_cfg.get("failure_penalty", -1.0)),
            dataset_info=dataset_info,
        )
        weights_dict["qed"] = float(weights_cfg.get("qed", 1.0))

    if "sa" in enabled:
        sa_cfg = reward_cfg.get("sa", {})
        rewards_dict["sa"] = SAReward(
            failure_penalty=float(sa_cfg.get("failure_penalty", -1.0)),
            dataset_info=dataset_info,
        )
        weights_dict["sa"] = float(weights_cfg.get("sa", 1.0))

    if "docking" in enabled:
        docking_reward = build_docking_reward(config, logger)
        docking_reward.dataset_info = dataset_info
        rewards_dict["docking"] = docking_reward
        weights_dict["docking"] = float(weights_cfg.get("docking", 1.0))

    if "diversity" in enabled:
        div_cfg = reward_cfg.get("diversity", {})
        rewards_dict["diversity"] = DiversityReward(
            failure_penalty=float(div_cfg.get("failure_penalty", -1.0)),
            global_batch=bool(div_cfg.get("global_batch", True)),
            mode=str(div_cfg.get("mode", "hinge_penalty")),
            target_similarity=float(div_cfg.get("target_similarity", 0.5)),
            penalty_scale=float(div_cfg.get("penalty_scale", 1.0)),
        )
        weights_dict["diversity"] = float(weights_cfg.get("diversity", 1.0))

    if 'connectivity' in enabled:
        try:
            from ddpo.rewards.connectivity_reward import ConnectivityReward
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Reward config enables 'connectivity', but ddpo.rewards.connectivity_reward "
                "is missing. Remove 'connectivity' from reward.enabled / reward.weights, "
                "or restore the reward module."
            ) from exc
        conn_cfg = reward_cfg.get('connectivity', {})
        rewards_dict['connectivity'] = ConnectivityReward(
            fragment_penalty=float(conn_cfg.get('fragment_penalty', -1.0)),
        )
        weights_dict['connectivity'] = float(weights_cfg.get('connectivity', 1.0))

    if not rewards_dict:
        raise ValueError("No reward components enabled for Lightning training")

    logger.info("Enabled reward components: %s", enabled)
    logger.info("Configured reward weights: %s", weights_dict)

    composite_reward = CompositeReward(
        rewards=rewards_dict,
        weights=weights_dict,
        normalize_weights=True,
    )
    logger.info("CompositeReward normalized weights: %s", composite_reward.get_weights())

    rewarder = StabilityAwareReward(composite_reward, dataset_info)
    logger.info("StabilityAwareReward wrapper applied")
    return rewarder
