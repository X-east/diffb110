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
from ddpo.rewards.component_normalizer import ComponentNormalizer
from ddpo.rewards.chembl_prior_reward import ChEMBLPriorReward
from ddpo.rewards.diversity_reward import DiversityReward
from ddpo.rewards.logp_distribution_reward import LogPDistributionReward
from ddpo.rewards.tpsa_distribution_reward import TPSADistributionReward
from ddpo.rewards.qed_reward import QEDReward
from ddpo.rewards.sa_reward import SAReward
from ddpo.rewards.docking_factory import build_docking_reward
from ddpo.rewards.ligand_efficiency_reward import LigandEfficiencyReward
from ddpo.rewards.molskill_reward import MolSkillReward
from ddpo.rewards.chembl_classifier_reward import ChemblClassifierReward
from ddpo.rewards.chembl_dmpnn_reward import ChemblDMPNNReward
from ddpo.rewards.aromatic_scaffold_reward import AromaticScaffoldReward
from ddpo.rewards.chemistry_guard_reward import ChemistryGuardReward
from ddpo.rewards.ring_complexity_reward import RingComplexityReward
from ddpo.rewards.ring_pathology_reward import RingPathologyReward
from ddpo.rewards.stability import StabilityAwareReward



def get_enabled_rewards(config: dict) -> list[str]:
    """推断启用的奖励组件列表。

    唯一来源是 `reward.weights`：weight > 0 视为启用，weight == 0 视为禁用。
    如果 weights 为空或全零则回退到 ["qed", "sa"]。

    `reward.monitor_components` 可额外启用只做记录的组件；这些组件的
    weight 仍保持配置值（通常为 0），因此不会改变 PPO 优化目标。

    特殊规则：ligand_efficiency 依赖 docking 提供的 ``ligand_efficiency_raw``，
    因此只要 LE 启用，docking 也自动启用（但 weight 保持用户配置的原值，可能为 0）。
    """
    reward_cfg = config.get("reward", {})
    weights = reward_cfg.get("weights", {})
    enabled = [name for name, value in weights.items() if float(value) > 0.0]
    for name in reward_cfg.get("monitor_components", []):
        if name not in enabled:
            enabled.append(str(name))
    # LE 依赖 docking 产生数据，自动启用 docking 作为数据提供者
    if "ligand_efficiency" in enabled and "docking" not in enabled:
        enabled.append("docking")
    if bool(reward_cfg.get("chemistry_guard", {}).get("enabled", False)) and "chemistry_guard" not in enabled:
        enabled.append("chemistry_guard")
    return enabled if enabled else ["qed", "sa"]


def build_rewarder(config: dict, dataset_info: dict, logger: logging.Logger):
    """Build the effective rewarder for the Lightning mainline.

    Current policy:
    - derive enabled components from positive `reward.weights`
    - preserve current defaults for DruglikenessReward parameters
    - only instantiate DockingReward when explicitly enabled
    """
    reward_cfg = config.get("reward", {})
    enabled = get_enabled_rewards(config)
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

    if "ligand_efficiency" in enabled:
        le_cfg = reward_cfg.get("ligand_efficiency", {})
        rewards_dict["ligand_efficiency"] = LigandEfficiencyReward(
            failure_penalty=float(le_cfg.get("failure_penalty", -1.0)),
            min_le=float(le_cfg.get("min_le", 0.0)),
            max_le=float(le_cfg.get("max_le", 2.0)),
            scale=float(le_cfg.get("scale", 1.0)),
        )
        weights_dict["ligand_efficiency"] = float(weights_cfg.get("ligand_efficiency", 0.1))

    if "diversity_ecfp4" in enabled:
        div_cfg = reward_cfg.get("diversity_ecfp4", {})
        rewards_dict["diversity_ecfp4"] = DiversityReward(
            failure_penalty=float(div_cfg.get("failure_penalty", -1.0)),
            global_batch=bool(div_cfg.get("global_batch", True)),
            mode=str(div_cfg.get("mode", "hinge_penalty")),
            target_similarity=float(div_cfg.get("target_similarity", 0.5)),
            penalty_scale=float(div_cfg.get("penalty_scale", 1.0)),
            use_features=False,
            reward_prefix="diversity_ecfp4",
        )
        weights_dict["diversity_ecfp4"] = float(weights_cfg.get("diversity_ecfp4", 1.0))

    if "diversity_fcfp4" in enabled:
        div_cfg = reward_cfg.get("diversity_fcfp4", {})
        rewards_dict["diversity_fcfp4"] = DiversityReward(
            failure_penalty=float(div_cfg.get("failure_penalty", -1.0)),
            global_batch=bool(div_cfg.get("global_batch", True)),
            mode=str(div_cfg.get("mode", "hinge_penalty")),
            target_similarity=float(div_cfg.get("target_similarity", 0.5)),
            penalty_scale=float(div_cfg.get("penalty_scale", 1.0)),
            use_features=True,
            reward_prefix="diversity_fcfp4",
        )
        weights_dict["diversity_fcfp4"] = float(weights_cfg.get("diversity_fcfp4", 1.0))

    # 向后兼容：旧名 chembl_prior → 单 ECFP4（仅当新名未启用时）
    if "chembl_prior" in enabled and "chembl_prior_ecfp4" not in enabled:
        chembl_cfg = reward_cfg.get("chembl_prior", {})
        fingerprint_cache = chembl_cfg.get("fingerprint_cache")
        if not fingerprint_cache:
            raise ValueError(
                "Reward config enables 'chembl_prior', but "
                "reward.chembl_prior.fingerprint_cache is missing."
            )
        rewards_dict["chembl_prior"] = ChEMBLPriorReward(
            fingerprint_cache=str(fingerprint_cache),
            failure_penalty=float(chembl_cfg.get("failure_penalty", -1.0)),
            top_k=int(chembl_cfg.get("top_k", 5)),
            chunk_size=int(chembl_cfg.get("chunk_size", 100000)),
            low_similarity_threshold=float(chembl_cfg.get("low_similarity_threshold", 0.25)),
            low_similarity_weight=float(chembl_cfg.get("low_similarity_weight", 4.0)),
            max_reward_similarity=float(chembl_cfg.get("max_reward_similarity", 0.50)),
            high_similarity_threshold=float(chembl_cfg.get("high_similarity_threshold", 0.85)),
            high_similarity_weight=float(chembl_cfg.get("high_similarity_weight", 0.1)),
        )
        weights_dict["chembl_prior"] = float(weights_cfg.get("chembl_prior", 1.0))

    if "chembl_prior_ecfp4" in enabled:
        chembl_cfg = reward_cfg.get("chembl_prior_ecfp4", {})
        fingerprint_cache = chembl_cfg.get("fingerprint_cache")
        if not fingerprint_cache:
            raise ValueError(
                "Reward config enables 'chembl_prior_ecfp4', but "
                "reward.chembl_prior_ecfp4.fingerprint_cache is missing."
            )
        rewards_dict["chembl_prior_ecfp4"] = ChEMBLPriorReward(
            fingerprint_cache=str(fingerprint_cache),
            reward_prefix="chembl_prior_ecfp4",
            failure_penalty=float(chembl_cfg.get("failure_penalty", -1.0)),
            top_k=int(chembl_cfg.get("top_k", 5)),
            chunk_size=int(chembl_cfg.get("chunk_size", 100000)),
            low_similarity_threshold=float(chembl_cfg.get("low_similarity_threshold", 0.25)),
            low_similarity_weight=float(chembl_cfg.get("low_similarity_weight", 4.0)),
            max_reward_similarity=float(chembl_cfg.get("max_reward_similarity", 0.50)),
            high_similarity_threshold=float(chembl_cfg.get("high_similarity_threshold", 0.85)),
            high_similarity_weight=float(chembl_cfg.get("high_similarity_weight", 0.1)),
        )
        weights_dict["chembl_prior_ecfp4"] = float(weights_cfg.get("chembl_prior_ecfp4", 1.0))

    if "chembl_prior_fcfp4" in enabled:
        chembl_cfg = reward_cfg.get("chembl_prior_fcfp4", {})
        fingerprint_cache = chembl_cfg.get("fingerprint_cache")
        if not fingerprint_cache:
            raise ValueError(
                "Reward config enables 'chembl_prior_fcfp4', but "
                "reward.chembl_prior_fcfp4.fingerprint_cache is missing."
            )
        rewards_dict["chembl_prior_fcfp4"] = ChEMBLPriorReward(
            fingerprint_cache=str(fingerprint_cache),
            reward_prefix="chembl_prior_fcfp4",
            use_features=True,
            failure_penalty=float(chembl_cfg.get("failure_penalty", -1.0)),
            top_k=int(chembl_cfg.get("top_k", 5)),
            chunk_size=int(chembl_cfg.get("chunk_size", 100000)),
            low_similarity_threshold=float(chembl_cfg.get("low_similarity_threshold", 0.25)),
            low_similarity_weight=float(chembl_cfg.get("low_similarity_weight", 4.0)),
            max_reward_similarity=float(chembl_cfg.get("max_reward_similarity", 0.50)),
            high_similarity_threshold=float(chembl_cfg.get("high_similarity_threshold", 0.85)),
            high_similarity_weight=float(chembl_cfg.get("high_similarity_weight", 0.1)),
        )
        weights_dict["chembl_prior_fcfp4"] = float(weights_cfg.get("chembl_prior_fcfp4", 1.0))

    if "logp_distribution" in enabled:
        lp_cfg = reward_cfg.get("logp_distribution", {})
        logp_bins_raw = lp_cfg.get("logp_bins", None)
        logp_bins = list(logp_bins_raw) if logp_bins_raw is not None else None
        rewards_dict["logp_distribution"] = LogPDistributionReward(
            chembl_csv=str(lp_cfg.get("chembl_csv", "data/chembl/chembl_36_properties.csv.gz")),
            logp_bins=logp_bins,
            failure_penalty=float(lp_cfg.get("failure_penalty", -1.0)),
            min_reward=float(lp_cfg.get("min_reward", -2.0)),
            max_reward=float(lp_cfg.get("max_reward", 2.0)),
            scale=float(lp_cfg.get("scale", 12.0)),
            ema_alpha=float(lp_cfg.get("ema_alpha", 0.0)),
            reward_prefix=str(lp_cfg.get("reward_prefix", "logp_distribution")),
        )
        weights_dict["logp_distribution"] = float(weights_cfg.get("logp_distribution", 0.1))

    if "tpsa_distribution" in enabled:
        tpsa_cfg = reward_cfg.get("tpsa_distribution", {})
        tpsa_bins_raw = tpsa_cfg.get("tpsa_bins", None)
        tpsa_bins = list(tpsa_bins_raw) if tpsa_bins_raw is not None else None
        rewards_dict["tpsa_distribution"] = TPSADistributionReward(
            chembl_csv=str(tpsa_cfg.get("chembl_csv", "data/chembl/chembl_36_properties.csv.gz")),
            tpsa_bins=tpsa_bins,
            failure_penalty=float(tpsa_cfg.get("failure_penalty", -1.0)),
            min_reward=float(tpsa_cfg.get("min_reward", -2.0)),
            max_reward=float(tpsa_cfg.get("max_reward", 2.0)),
            scale=float(tpsa_cfg.get("scale", 12.0)),
            ema_alpha=float(tpsa_cfg.get("ema_alpha", 0.0)),
            reward_prefix=str(tpsa_cfg.get("reward_prefix", "tpsa_distribution")),
        )
        weights_dict["tpsa_distribution"] = float(weights_cfg.get("tpsa_distribution", 0.1))

    if 'connectivity' in enabled:
        try:
            from ddpo.rewards.connectivity_reward import ConnectivityReward
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Reward config enables 'connectivity', but ddpo.rewards.connectivity_reward "
                "is missing. Remove 'connectivity' from reward.weights, "
                "or restore the reward module."
            ) from exc
        conn_cfg = reward_cfg.get('connectivity', {})
        rewards_dict['connectivity'] = ConnectivityReward(
            fragment_penalty=float(conn_cfg.get('fragment_penalty', -1.0)),
        )
        weights_dict['connectivity'] = float(weights_cfg.get('connectivity', 1.0))

    if "molskill" in enabled:
        ms_cfg = reward_cfg.get("molskill", {})
        rewards_dict["molskill"] = MolSkillReward(
            checkpoint_path=ms_cfg.get("checkpoint_path", None),
            failure_penalty=float(ms_cfg.get("failure_penalty", -1.0)),
            clamp_min=float(ms_cfg.get("clamp_min", -5.0)),
            clamp_max=float(ms_cfg.get("clamp_max", 5.0)),
        )
        weights_dict["molskill"] = float(weights_cfg.get("molskill", 1.0))

    if "chembl_classifier" in enabled:
        cc_cfg = reward_cfg.get("chembl_classifier", {})
        rewards_dict["chembl_classifier"] = ChemblClassifierReward(
            model_path=cc_cfg.get("model_path", None),
            failure_penalty=float(cc_cfg.get("failure_penalty", -1.0)),
            good_threshold=float(cc_cfg.get("good_threshold", 0.5)),
            bad_threshold=cc_cfg.get("bad_threshold", None),
            cap_high=float(cc_cfg.get("cap_high", 0.8)),
            cap_low=float(cc_cfg.get("cap_low", 0.2)),
            max_reward=float(cc_cfg.get("max_reward", 1.0)),
            penalty=float(cc_cfg.get("penalty", -1.0)),
            device=cc_cfg.get("device", None),
        )
        weights_dict["chembl_classifier"] = float(weights_cfg.get("chembl_classifier", 1.0))

    if "chembl_dmpnn" in enabled:
        dmpnn_cfg = reward_cfg.get("chembl_dmpnn", {})
        rewards_dict["chembl_dmpnn"] = ChemblDMPNNReward(
            model_path=dmpnn_cfg.get("model_path", None),
            failure_penalty=float(dmpnn_cfg.get("failure_penalty", -1.0)),
            good_threshold=float(dmpnn_cfg.get("good_threshold", 0.5)),
            bad_threshold=dmpnn_cfg.get("bad_threshold", None),
            cap_high=float(dmpnn_cfg.get("cap_high", 0.8)),
            cap_low=float(dmpnn_cfg.get("cap_low", 0.2)),
            max_reward=float(dmpnn_cfg.get("max_reward", 1.0)),
            penalty=float(dmpnn_cfg.get("penalty", -1.0)),
            device=dmpnn_cfg.get("device", None),
            batch_size=int(dmpnn_cfg.get("batch_size", 128)),
            cache_smiles=bool(dmpnn_cfg.get("cache_smiles", True)),
        )
        weights_dict["chembl_dmpnn"] = float(weights_cfg.get("chembl_dmpnn", 1.0))

    if "chemistry_guard" in enabled:
        guard_cfg = reward_cfg.get("chemistry_guard", {})
        rewards_dict["chemistry_guard"] = ChemistryGuardReward(
            failure_penalty=float(guard_cfg.get("failure_penalty", -1.0)),
            base_reward=float(guard_cfg.get("base_reward", 1.0)),
            violation_penalty=float(guard_cfg.get("violation_penalty", 0.2)),
            min_reward=float(guard_cfg.get("min_reward", -1.0)),
            max_reward=float(guard_cfg.get("max_reward", 1.0)),
            max_carbon_fraction=float(guard_cfg.get("max_carbon_fraction", 0.85)),
            min_hetero_atom_fraction=float(guard_cfg.get("min_hetero_atom_fraction", 0.0)),
            max_three_member_ring_count=int(guard_cfg.get("max_three_member_ring_count", 0)),
            max_fragment_count=int(guard_cfg.get("max_fragment_count", 1)),
            max_bad_bond_length_fraction=float(guard_cfg.get("max_bad_bond_length_fraction", 0.05)),
            max_halogen_bad_position_count=int(guard_cfg.get("max_halogen_bad_position_count", 0)),
            bond_length_relative_tolerance=float(guard_cfg.get("bond_length_relative_tolerance", 0.30)),
            bond_length_absolute_tolerance=float(guard_cfg.get("bond_length_absolute_tolerance", 0.20)),
            max_halogen_neighbor_heavy_degree=guard_cfg.get("max_halogen_neighbor_heavy_degree", None),
        )
        weights_dict["chemistry_guard"] = float(weights_cfg.get("chemistry_guard", 0.0))

    if "ring_complexity" in enabled:
        ring_cfg = reward_cfg.get("ring_complexity", {})
        rewards_dict["ring_complexity"] = RingComplexityReward(
            failure_penalty=float(ring_cfg.get("failure_penalty", -1.0)),
            base_reward=float(ring_cfg.get("base_reward", 1.0)),
            min_reward=float(ring_cfg.get("min_reward", -1.0)),
            max_reward=float(ring_cfg.get("max_reward", 1.0)),
            target_ring_count=int(ring_cfg.get("target_ring_count", 4)),
            target_aromatic_ring_count=int(ring_cfg.get("target_aromatic_ring_count", 2)),
            target_fused_ring_system_size=int(ring_cfg.get("target_fused_ring_system_size", 2)),
            max_three_member_ring_count=int(ring_cfg.get("max_three_member_ring_count", 0)),
            max_spiro_atom_count=int(ring_cfg.get("max_spiro_atom_count", 1)),
            max_bridgehead_atom_count=int(ring_cfg.get("max_bridgehead_atom_count", 2)),
            ring_penalty=float(ring_cfg.get("ring_penalty", 0.18)),
            aromatic_penalty=float(ring_cfg.get("aromatic_penalty", 0.12)),
            fused_penalty=float(ring_cfg.get("fused_penalty", 0.20)),
            three_member_penalty=float(ring_cfg.get("three_member_penalty", 0.35)),
            spiro_penalty=float(ring_cfg.get("spiro_penalty", 0.10)),
            bridgehead_penalty=float(ring_cfg.get("bridgehead_penalty", 0.08)),
        )
        weights_dict["ring_complexity"] = float(weights_cfg.get("ring_complexity", 0.0))

    if "ring_pathology" in enabled:
        pathology_cfg = reward_cfg.get("ring_pathology", {})
        rewards_dict["ring_pathology"] = RingPathologyReward(
            failure_penalty=float(pathology_cfg.get("failure_penalty", -1.0)),
            base_reward=float(pathology_cfg.get("base_reward", 1.0)),
            min_reward=float(pathology_cfg.get("min_reward", -1.0)),
            max_reward=float(pathology_cfg.get("max_reward", 1.0)),
            max_allowed_ring_count=int(pathology_cfg.get("max_allowed_ring_count", 3)),
            min_large_ring_size=int(pathology_cfg.get("min_large_ring_size", 7)),
            max_allowed_three_member_ring_count=int(pathology_cfg.get("max_allowed_three_member_ring_count", 1)),
            ring_count_penalty=float(pathology_cfg.get("ring_count_penalty", 0.25)),
            large_ring_penalty=float(pathology_cfg.get("large_ring_penalty", 0.45)),
            large_ring_fused_six_penalty=float(pathology_cfg.get("large_ring_fused_six_penalty", 0.65)),
            extra_three_member_penalty=float(pathology_cfg.get("extra_three_member_penalty", 0.45)),
            three_member_fused_or_spiro_penalty=float(
                pathology_cfg.get("three_member_fused_or_spiro_penalty", 0.65)
            ),
        )
        weights_dict["ring_pathology"] = float(weights_cfg.get("ring_pathology", 0.0))

    if "aromatic_scaffold" in enabled:
        aromatic_cfg = reward_cfg.get("aromatic_scaffold", {})
        rewards_dict["aromatic_scaffold"] = AromaticScaffoldReward(
            failure_penalty=float(aromatic_cfg.get("failure_penalty", -1.0)),
            base_reward=float(aromatic_cfg.get("base_reward", 0.0)),
            min_reward=float(aromatic_cfg.get("min_reward", -1.0)),
            max_reward=float(aromatic_cfg.get("max_reward", 1.0)),
            exact_single_benzene_reward=float(aromatic_cfg.get("exact_single_benzene_reward", 1.0)),
            missing_benzene_penalty=float(aromatic_cfg.get("missing_benzene_penalty", 0.25)),
            max_benzene_ring_count=int(aromatic_cfg.get("max_benzene_ring_count", 2)),
            extra_benzene_penalty=float(aromatic_cfg.get("extra_benzene_penalty", 0.25)),
            max_heterocycle_ring_count=int(aromatic_cfg.get("max_heterocycle_ring_count", 1)),
            multi_heterocycle_penalty=float(aromatic_cfg.get("multi_heterocycle_penalty", 0.55)),
            max_fused_ring_system_size=int(aromatic_cfg.get("max_fused_ring_system_size", 3)),
            fused_penalty=float(aromatic_cfg.get("fused_penalty", 0.30)),
            max_three_member_ring_count=int(aromatic_cfg.get("max_three_member_ring_count", 0)),
            three_member_penalty=float(aromatic_cfg.get("three_member_penalty", 0.45)),
            max_total_halogen_count=int(aromatic_cfg.get("max_total_halogen_count", 2)),
            total_halogen_penalty=float(aromatic_cfg.get("total_halogen_penalty", 0.20)),
            max_aromatic_halogen_count=int(aromatic_cfg.get("max_aromatic_halogen_count", 2)),
            aromatic_halogen_penalty=float(aromatic_cfg.get("aromatic_halogen_penalty", 0.20)),
        )
        weights_dict["aromatic_scaffold"] = float(weights_cfg.get("aromatic_scaffold", 0.0))

    if not rewards_dict:
        raise ValueError("No reward components enabled for Lightning training")

    # ── Per-component normalisation ──
    norm_cfg = reward_cfg.get("per_component_normalization", {})
    component_normalizer = None
    if bool(norm_cfg.get("enabled", False)):
        component_normalizer = ComponentNormalizer(
            component_names=list(rewards_dict.keys()),
            momentum=float(norm_cfg.get("momentum", 0.01)),
            epsilon=float(norm_cfg.get("epsilon", 1e-6)),
        )
        logger.info(
            "Per-component normalisation ENABLED (momentum=%.3f, components=%d)",
            component_normalizer._normalizers[list(rewards_dict.keys())[0]].momentum,
            len(rewards_dict),
        )

    logger.info("Enabled reward components: %s", enabled)
    logger.info("Configured reward weights: %s", weights_dict)

    composite_reward = CompositeReward(
        rewards=rewards_dict,
        weights=weights_dict,
        normalize_weights=True,
        component_normalizer=component_normalizer,
    )
    logger.info("CompositeReward normalized weights: %s", composite_reward.get_weights())

    rewarder = StabilityAwareReward(composite_reward, dataset_info)
    logger.info("StabilityAwareReward wrapper applied")
    return rewarder
