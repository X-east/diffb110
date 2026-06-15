# analysis 剩余内容分类整理记录

- 执行时间：2026-06-15T09:39:46
- 移动目录：103
- 移动顶层文件：55
- 创建兼容软链接：158
- 跳过/需复查：0

## 目标结构

```text
analysis/
  benchmark/
  classifier/
  diffgui_rl/
  diffsbdd_rl/
  diffsbdd_small_mod/
  diffsbdd_xhe/
  preprocessing/
  reconstruction/
  _archive/
  _index/
  _loose/
tools/analysis_scripts/legacy_top_level/
```

## 说明

- 本次没有删除任何内容。
- 所有被移动的顶层目录/文件，旧位置都保留软链接，旧报告和旧命令仍可打开。
- `analysis/_index/analysis_catalog.csv` 会按分类后的真实目录重新生成。

## 分类目录

| 类别 | 是否处理 |
|---|---|
| benchmark | 是 |
| classifier | 是 |
| diffgui_rl | 是 |
| diffsbdd_rl | 是 |
| diffsbdd_small_mod | 是 |
| diffsbdd_xhe | 是 |
| misc | 是 |
| preprocessing | 是 |
| reconstruction | 是 |
| smoke_or_debug | 是 |

## 移动目录清单

| 旧路径 | 新路径 | 类别 | 大小 |
|---|---|---|---:|
| `analysis/5xzr_8j6_benchmark_20260611` | `analysis/benchmark/5xzr_8j6_benchmark_20260611` | benchmark | 70.8M |
| `analysis/8gjl_c1101_benchmark_20260603_500mol_500steps` | `analysis/benchmark/8gjl_c1101_benchmark_20260603_500mol_500steps` | benchmark | 60.7M |
| `analysis/aromatic_benzene_diagnosis_20260527` | `analysis/diffsbdd_small_mod/aromatic_benzene_diagnosis_20260527` | diffsbdd_small_mod | 7.5M |
| `analysis/atom_score_diagnostics_20260602_n1000_gpu` | `analysis/diffsbdd_small_mod/atom_score_diagnostics_20260602_n1000_gpu` | diffsbdd_small_mod | 4.0M |
| `analysis/bond_predictor_diagnostics_20260602_n300` | `analysis/reconstruction/bond_predictor_diagnostics_20260602_n300` | reconstruction | 6.4M |
| `analysis/chembl_full_analysis` | `analysis/classifier/chembl_full_analysis` | classifier | 4.1M |
| `analysis/chembl_sim_full` | `analysis/classifier/chembl_sim_full` | classifier | 146.6K |
| `analysis/classifier_failure_review_20260608` | `analysis/classifier/classifier_failure_review_20260608` | classifier | 254.3K |
| `analysis/classifier_generated_score_distribution_20260613` | `analysis/classifier/classifier_generated_score_distribution_20260613` | classifier | 16.3M |
| `analysis/classifier_named_sdfs_20260609` | `analysis/classifier/classifier_named_sdfs_20260609` | classifier | 19.9M |
| `analysis/classifier_test_260407_sdf_20260527` | `analysis/classifier/classifier_test_260407_sdf_20260527` | classifier | 1.0M |
| `analysis/classifier_v0_2_1_weighted_20260608` | `analysis/classifier/classifier_v0_2_1_weighted_20260608` | classifier | 32.5M |
| `analysis/classifier_v0_2_hardneg_20260608` | `analysis/classifier/classifier_v0_2_hardneg_20260608` | classifier | 32.1M |
| `analysis/classifier_v0_3_1_training_20260610` | `analysis/classifier/classifier_v0_3_1_training_20260610` | classifier | 11.0K |
| `analysis/classifier_v0_3_scaffold_curation_20260610` | `analysis/classifier/classifier_v0_3_scaffold_curation_20260610` | classifier | 1.9G |
| `analysis/crossdock_preprocess_20260601` | `analysis/preprocessing/crossdock_preprocess_20260601` | preprocessing | 10.4M |
| `analysis/dataset` | `analysis/preprocessing/dataset` | preprocessing | 36B |
| `analysis/diffgui_200step_batch_sweep_20260609` | `analysis/diffgui_rl/diffgui_200step_batch_sweep_20260609` | diffgui_rl | 446.4K |
| `analysis/diffgui_200step_timestep_fix_20260609` | `analysis/diffgui_rl/diffgui_200step_timestep_fix_20260609` | diffgui_rl | 631.0K |
| `analysis/diffgui_6gpu_500x3_20260531_085146` | `analysis/diffgui_rl/diffgui_6gpu_500x3_20260531_085146` | diffgui_rl | 43.2M |
| `analysis/diffgui_baseline_classifier_v02_20260611` | `analysis/classifier/diffgui_baseline_classifier_v02_20260611` | classifier | 1.3M |
| `analysis/diffgui_baseline_classifier_v03_20260611` | `analysis/classifier/diffgui_baseline_classifier_v03_20260611` | classifier | 1.4M |
| `analysis/diffgui_big_runs_summary_20260611` | `analysis/diffgui_rl/diffgui_big_runs_summary_20260611` | diffgui_rl | 2.8M |
| `analysis/diffgui_classifier_score_20260609` | `analysis/classifier/diffgui_classifier_score_20260609` | classifier | 2.0M |
| `analysis/diffgui_diffsbdd_classifier_v031_20260615` | `analysis/classifier/diffgui_diffsbdd_classifier_v031_20260615` | classifier | 871.3K |
| `analysis/diffgui_docking_batch_sweep_20260609` | `analysis/diffgui_rl/diffgui_docking_batch_sweep_20260609` | diffgui_rl | 307.8K |
| `analysis/diffgui_docking_batch_sweep_20260609_last1_b24_b32` | `analysis/diffgui_rl/diffgui_docking_batch_sweep_20260609_last1_b24_b32` | diffgui_rl | 452.6K |
| `analysis/diffgui_docking_batch_sweep_20260609_logiccheck_bs2` | `analysis/diffgui_rl/diffgui_docking_batch_sweep_20260609_logiccheck_bs2` | diffgui_rl | 95.8K |
| `analysis/diffgui_docking_batch_sweep_20260609_preflight_bs2` | `analysis/diffgui_rl/diffgui_docking_batch_sweep_20260609_preflight_bs2` | diffgui_rl | 100.8K |
| `analysis/diffgui_reward_only_stage1_20260609` | `analysis/diffgui_rl/diffgui_reward_only_stage1_20260609` | diffgui_rl | 1.8M |
| `analysis/diffgui_ring_last3_lr3e5_training_plots_20260611` | `analysis/diffgui_rl/diffgui_ring_last3_lr3e5_training_plots_20260611` | diffgui_rl | 170.5M |
| `analysis/diffgui_timestep_investigation_20260609` | `analysis/diffgui_rl/diffgui_timestep_investigation_20260609` | diffgui_rl | 105.7K |
| `analysis/diffsbdd_8gpu_5000_score_20260608` | `analysis/diffsbdd_rl/diffsbdd_8gpu_5000_score_20260608` | diffsbdd_rl | 46.4M |
| `analysis/diffsbdd_rl_reset_plan_20260611` | `analysis/diffsbdd_rl/diffsbdd_rl_reset_plan_20260611` | diffsbdd_rl | 1.5M |
| `analysis/diffsbdd_single_reward_matrix_20260613` | `analysis/diffsbdd_rl/diffsbdd_single_reward_matrix_20260613` | diffsbdd_rl | 45.9M |
| `analysis/diffsbdd_xhe_diffgui_scratch_20260606_longtest` | `analysis/diffsbdd_xhe/diffsbdd_xhe_diffgui_scratch_20260606_longtest` | diffsbdd_xhe | 866.8M |
| `analysis/diffsbdd_xhe_joint_20260605_20k_6gpu` | `analysis/diffsbdd_xhe/diffsbdd_xhe_joint_20260605_20k_6gpu` | diffsbdd_xhe | 41.3M |
| `analysis/diffsbdd_xhe_predicted_reconstruction_20260605_val` | `analysis/diffsbdd_xhe/diffsbdd_xhe_predicted_reconstruction_20260605_val` | diffsbdd_xhe | 11.6M |
| `analysis/diffsbdd_xhe_supervised_20260604_20k_6gpu` | `analysis/diffsbdd_xhe/diffsbdd_xhe_supervised_20260604_20k_6gpu` | diffsbdd_xhe | 915.8M |
| `analysis/edge_data_chain_20260602_n1000` | `analysis/diffsbdd_xhe/edge_data_chain_20260602_n1000` | diffsbdd_xhe | 7.5M |
| `analysis/edge_data_chain_20260602_repair_n1000` | `analysis/diffsbdd_xhe/edge_data_chain_20260602_repair_n1000` | diffsbdd_xhe | 7.6M |
| `analysis/edge_data_chain_20260604_repair_n10000` | `analysis/diffsbdd_xhe/edge_data_chain_20260604_repair_n10000` | diffsbdd_xhe | 75.8M |
| `analysis/edge_data_oracle_crossdock_20260601` | `analysis/diffsbdd_xhe/edge_data_oracle_crossdock_20260601` | diffsbdd_xhe | 6.1M |
| `analysis/edge_denoiser_ablation_20260603_A_inverse_clean` | `analysis/diffsbdd_xhe/edge_denoiser_ablation_20260603_A_inverse_clean` | diffsbdd_xhe | 855.3K |
| `analysis/edge_denoiser_ablation_20260603_B_diffgui_noise005` | `analysis/diffsbdd_xhe/edge_denoiser_ablation_20260603_B_diffgui_noise005` | diffsbdd_xhe | 858.6K |
| `analysis/edge_denoiser_ablation_20260603_C_diffgui_noise005_focal2` | `analysis/diffsbdd_xhe/edge_denoiser_ablation_20260603_C_diffgui_noise005_focal2` | diffsbdd_xhe | 859.0K |
| `analysis/edge_denoiser_ablation_20260603_C_diffgui_noise005_focal2_gpu` | `analysis/diffsbdd_xhe/edge_denoiser_ablation_20260603_C_diffgui_noise005_focal2_gpu` | diffsbdd_xhe | 858.6K |
| `analysis/edge_denoiser_ablation_20260603_D_diffgui_noise010` | `analysis/diffsbdd_xhe/edge_denoiser_ablation_20260603_D_diffgui_noise010` | diffsbdd_xhe | 857.7K |
| `analysis/edge_denoiser_ablation_20260603_logs` | `analysis/diffsbdd_xhe/edge_denoiser_ablation_20260603_logs` | diffsbdd_xhe | 390B |
| `analysis/edge_denoiser_ablation_eval_20260603_B_diffgui_noise005` | `analysis/diffsbdd_xhe/edge_denoiser_ablation_eval_20260603_B_diffgui_noise005` | diffsbdd_xhe | 1.5M |
| `analysis/edge_denoiser_ablation_eval_20260603_C_diffgui_noise005_focal2` | `analysis/diffsbdd_xhe/edge_denoiser_ablation_eval_20260603_C_diffgui_noise005_focal2` | diffsbdd_xhe | 1.5M |
| `analysis/edge_denoiser_ablation_eval_20260603_C_diffgui_noise005_focal2_gpu` | `analysis/diffsbdd_xhe/edge_denoiser_ablation_eval_20260603_C_diffgui_noise005_focal2_gpu` | diffsbdd_xhe | 1.5M |
| `analysis/edge_denoiser_ablation_eval_20260603_D_diffgui_noise010` | `analysis/diffsbdd_xhe/edge_denoiser_ablation_eval_20260603_D_diffgui_noise010` | diffsbdd_xhe | 1.5M |
| `analysis/edge_denoiser_ablation_summary_20260603` | `analysis/diffsbdd_xhe/edge_denoiser_ablation_summary_20260603` | diffsbdd_xhe | 109.1K |
| `analysis/edge_denoiser_eval_20260603_n1000_val_tgrid_reconstruct_gpu` | `analysis/diffsbdd_xhe/edge_denoiser_eval_20260603_n1000_val_tgrid_reconstruct_gpu` | diffsbdd_xhe | 253.7K |
| `analysis/edge_denoiser_eval_20260603_n1000_val_tgrid_reconstruct_gpu_v2` | `analysis/diffsbdd_xhe/edge_denoiser_eval_20260603_n1000_val_tgrid_reconstruct_gpu_v2` | diffsbdd_xhe | 1.3M |
| `analysis/edge_denoiser_eval_20260603_n1000_val_tgrid_reconstruct_gpu_v3` | `analysis/diffsbdd_xhe/edge_denoiser_eval_20260603_n1000_val_tgrid_reconstruct_gpu_v3` | diffsbdd_xhe | 1.3M |
| `analysis/edge_denoiser_nodeedge_20260602_n32_gpu` | `analysis/diffsbdd_xhe/edge_denoiser_nodeedge_20260602_n32_gpu` | diffsbdd_xhe | 837.0K |
| `analysis/edge_denoiser_nodeedge_20260603_n1000_train800_val200_gpu` | `analysis/diffsbdd_xhe/edge_denoiser_nodeedge_20260603_n1000_train800_val200_gpu` | diffsbdd_xhe | 807.3K |
| `analysis/edge_denoiser_overfit_20260602_n128_gpu` | `analysis/diffsbdd_xhe/edge_denoiser_overfit_20260602_n128_gpu` | diffsbdd_xhe | 499.3K |
| `analysis/edge_denoiser_overfit_20260602_n32_gpu` | `analysis/diffsbdd_xhe/edge_denoiser_overfit_20260602_n32_gpu` | diffsbdd_xhe | 1.8M |
| `analysis/edge_diffusion_ddp_20260604_10k_6gpu_test3` | `analysis/diffsbdd_xhe/edge_diffusion_ddp_20260604_10k_6gpu_test3` | diffsbdd_xhe | 5.9M |
| `analysis/edge_diffusion_xhe_overfit_20260603_n128_gpu` | `analysis/diffsbdd_xhe/edge_diffusion_xhe_overfit_20260603_n128_gpu` | diffsbdd_xhe | 3.1M |
| `analysis/edge_diffusion_xhe_overfit_20260603_n128_uniform_gpu` | `analysis/diffsbdd_xhe/edge_diffusion_xhe_overfit_20260603_n128_uniform_gpu` | diffsbdd_xhe | 3.2M |
| `analysis/edge_semantic_bundle_20260602_n1000` | `analysis/diffsbdd_xhe/edge_semantic_bundle_20260602_n1000` | diffsbdd_xhe | 886.1K |
| `analysis/edge_semantic_bundle_20260604_n10000` | `analysis/diffsbdd_xhe/edge_semantic_bundle_20260604_n10000` | diffsbdd_xhe | 7.9M |
| `analysis/final_classifier_dataset_selection_20260527` | `analysis/classifier/final_classifier_dataset_selection_20260527` | classifier | 32.2M |
| `analysis/final_refine_ablation_20260602_n100` | `analysis/diffsbdd_small_mod/final_refine_ablation_20260602_n100` | diffsbdd_small_mod | 920.4K |
| `analysis/final_refine_ablation_20260602_n1000_gpu` | `analysis/diffsbdd_small_mod/final_refine_ablation_20260602_n1000_gpu` | diffsbdd_small_mod | 7.1M |
| `analysis/formal_xh_xhe_comparison_20260605_full_8gpu` | `analysis/diffsbdd_xhe/formal_xh_xhe_comparison_20260605_full_8gpu` | diffsbdd_xhe | 457.6M |
| `analysis/generated_edge_denoiser_offmanifold_20260603_n500_t100_uniform_t999` | `analysis/diffsbdd_xhe/generated_edge_denoiser_offmanifold_20260603_n500_t100_uniform_t999` | diffsbdd_xhe | 2.5M |
| `analysis/generated_reconstruction_ablation_20260601_n50` | `analysis/reconstruction/generated_reconstruction_ablation_20260601_n50` | reconstruction | 205.6K |
| `analysis/generated_reconstruction_ablation_20260601_n500` | `analysis/reconstruction/generated_reconstruction_ablation_20260601_n500` | reconstruction | 1.0M |
| `analysis/generated_reconstruction_benchmark_20260601_external_n100` | `analysis/benchmark/generated_reconstruction_benchmark_20260601_external_n100` | benchmark | 9.3M |
| `analysis/generated_reconstruction_benchmark_20260602_local_bond_v0_external_n100` | `analysis/benchmark/generated_reconstruction_benchmark_20260602_local_bond_v0_external_n100` | benchmark | 9.3M |
| `analysis/generated_reconstruction_benchmark_20260602_local_bond_v0_n500` | `analysis/benchmark/generated_reconstruction_benchmark_20260602_local_bond_v0_n500` | benchmark | 1.4M |
| `analysis/good_bad_classifier_curation_20260608` | `analysis/classifier/good_bad_classifier_curation_20260608` | classifier | 26.0M |
| `analysis/good_pool_audit_20260604_173430` | `analysis/diffsbdd_rl/good_pool_audit_20260604_173430` | diffsbdd_rl | 0B |
| `analysis/good_pool_audit_20260604_174051` | `analysis/diffsbdd_rl/good_pool_audit_20260604_174051` | diffsbdd_rl | 414.6K |
| `analysis/hetero_composition_diagnosis_20260527` | `analysis/diffsbdd_small_mod/hetero_composition_diagnosis_20260527` | diffsbdd_small_mod | 8.4M |
| `analysis/kl_constraint_anomaly_20260604` | `analysis/diffsbdd_rl/kl_constraint_anomaly_20260604` | diffsbdd_rl | 1013.5K |
| `analysis/kl_diagnostic_20260604_kl0p02_25ep` | `analysis/diffsbdd_rl/kl_diagnostic_20260604_kl0p02_25ep` | diffsbdd_rl | 427.1K |
| `analysis/kl_reference_comparison_20260604` | `analysis/diffsbdd_rl/kl_reference_comparison_20260604` | diffsbdd_rl | 16.0K |
| `analysis/local_atom_mask_ablation_20260602_n1000_gpu` | `analysis/diffsbdd_small_mod/local_atom_mask_ablation_20260602_n1000_gpu` | diffsbdd_small_mod | 4.0M |
| `analysis/local_bond_predictor_data_20260601` | `analysis/diffsbdd_small_mod/local_bond_predictor_data_20260601` | diffsbdd_small_mod | 15.3M |
| `analysis/local_bond_predictor_v0_20260602_small` | `analysis/diffsbdd_small_mod/local_bond_predictor_v0_20260602_small` | diffsbdd_small_mod | 727.7K |
| `analysis/logprob_mode_experiment_20260604` | `analysis/diffsbdd_rl/logprob_mode_experiment_20260604` | diffsbdd_rl | 1.0M |
| `analysis/model_strategy_20260531` | `analysis/benchmark/model_strategy_20260531` | benchmark | 4.6M |
| `analysis/modern_sbdd_20260531` | `analysis/benchmark/modern_sbdd_20260531` | benchmark | 1.8G |
| `analysis/oracle_original_sdf_audit_20260602_n1000` | `analysis/reconstruction/oracle_original_sdf_audit_20260602_n1000` | reconstruction | 19.4M |
| `analysis/reconstruction_ablation_crossdock_20260601` | `analysis/preprocessing/reconstruction_ablation_crossdock_20260601` | preprocessing | 4.2M |
| `analysis/reference_project_generation_10mol_20260530_224503` | `analysis/benchmark/reference_project_generation_10mol_20260530_224503` | benchmark | 34.4M |
| `analysis/reference_project_generation_20260530_173617` | `analysis/benchmark/reference_project_generation_20260530_173617` | benchmark | 539.0M |
| `analysis/reference_project_supplement_20260531` | `analysis/benchmark/reference_project_supplement_20260531` | benchmark | 438.7K |
| `analysis/refinement_ablation_20260603_n1000_t100` | `analysis/diffsbdd_small_mod/refinement_ablation_20260603_n1000_t100` | diffsbdd_small_mod | 12.4M |
| `analysis/reward_hacking_diagnosis_20260527` | `analysis/diffsbdd_rl/reward_hacking_diagnosis_20260527` | diffsbdd_rl | 14.0M |
| `analysis/reward_loophole_risk_register_20260527` | `analysis/diffsbdd_rl/reward_loophole_risk_register_20260527` | diffsbdd_rl | 213.9K |
| `analysis/rl_data_sufficiency_20260608` | `analysis/diffsbdd_rl/rl_data_sufficiency_20260608` | diffsbdd_rl | 175.5K |
| `analysis/rl_high_carbon_sparse_reward_20260604` | `analysis/diffsbdd_rl/rl_high_carbon_sparse_reward_20260604` | diffsbdd_rl | 2.5M |
| `analysis/rl_next_steps_20260608` | `analysis/diffsbdd_rl/rl_next_steps_20260608` | diffsbdd_rl | 96.9K |
| `analysis/semantic_oracle_reconstruction_20260602_n1000` | `analysis/reconstruction/semantic_oracle_reconstruction_20260602_n1000` | reconstruction | 648.1K |
| `analysis/train_analysis` | `analysis/preprocessing/train_analysis` | preprocessing | 75.6K |
| `analysis/xhe_ep37_failure_analysis_20260606` | `analysis/diffsbdd_xhe/xhe_ep37_failure_analysis_20260606` | diffsbdd_xhe | 158.5K |

## 移动文件清单

| 旧路径 | 新路径 | 类别 | 大小 |
|---|---|---|---:|
| `analysis/7gpu_dock_classifier_v0_1_20260528_training_and_classifier_summary.md` | `analysis/_loose/classifier/7gpu_dock_classifier_v0_1_20260528_training_and_classifier_summary.md` | classifier | 12.1K |
| `analysis/_compare_ep14_ep24.py` | `tools/analysis_scripts/legacy_top_level/_compare_ep14_ep24.py` | misc | 5.1K |
| `analysis/_le_debug.py` | `tools/analysis_scripts/legacy_top_level/_le_debug.py` | smoke_or_debug | 1.8K |
| `analysis/_molskill_verify.py` | `tools/analysis_scripts/legacy_top_level/_molskill_verify.py` | misc | 3.1K |
| `analysis/_plot_classifier_only_training.py` | `tools/analysis_scripts/legacy_top_level/_plot_classifier_only_training.py` | classifier | 23.8K |
| `analysis/_verify.py` | `tools/analysis_scripts/legacy_top_level/_verify.py` | misc | 2.2K |
| `analysis/_verify2.py` | `tools/analysis_scripts/legacy_top_level/_verify2.py` | misc | 2.1K |
| `analysis/_verify_logp_direct.py` | `tools/analysis_scripts/legacy_top_level/_verify_logp_direct.py` | misc | 2.8K |
| `analysis/_verify_logp_per_rank.py` | `tools/analysis_scripts/legacy_top_level/_verify_logp_per_rank.py` | misc | 3.4K |
| `analysis/analyze_6gpu_dock.py` | `tools/analysis_scripts/legacy_top_level/analyze_6gpu_dock.py` | misc | 12.2K |
| `analysis/analyze_6gpu_dock_chembl.py` | `tools/analysis_scripts/legacy_top_level/analyze_6gpu_dock_chembl.py` | classifier | 14.7K |
| `analysis/analyze_7gpu_dock_classifier_v0_1_20260528.py` | `tools/analysis_scripts/legacy_top_level/analyze_7gpu_dock_classifier_v0_1_20260528.py` | classifier | 29.8K |
| `analysis/analyze_7gpu_dock_ecfp4_fcfp4.py` | `tools/analysis_scripts/legacy_top_level/analyze_7gpu_dock_ecfp4_fcfp4.py` | misc | 41.4K |
| `analysis/analyze_7gpu_dock_ecfp4_fcfp4_20260526.py` | `tools/analysis_scripts/legacy_top_level/analyze_7gpu_dock_ecfp4_fcfp4_20260526.py` | misc | 22.9K |
| `analysis/analyze_8gpu_chembl_only.py` | `tools/analysis_scripts/legacy_top_level/analyze_8gpu_chembl_only.py` | classifier | 18.5K |
| `analysis/analyze_diffsbdd_single_reward_matrix_20260613.py` | `tools/analysis_scripts/legacy_top_level/analyze_diffsbdd_single_reward_matrix_20260613.py` | diffsbdd_rl | 35.2K |
| `analysis/analyze_fingerprint_exps.py` | `tools/analysis_scripts/legacy_top_level/analyze_fingerprint_exps.py` | misc | 23.9K |
| `analysis/analyze_logp_bins_detail.py` | `tools/analysis_scripts/legacy_top_level/analyze_logp_bins_detail.py` | misc | 16.4K |
| `analysis/analyze_reward_influence.py` | `tools/analysis_scripts/legacy_top_level/analyze_reward_influence.py` | misc | 9.9K |
| `analysis/analyze_reward_influence_v2.py` | `tools/analysis_scripts/legacy_top_level/analyze_reward_influence_v2.py` | misc | 5.5K |
| `analysis/analyze_valid_sdf_properties.py` | `tools/analysis_scripts/legacy_top_level/analyze_valid_sdf_properties.py` | reconstruction | 7.1K |
| `analysis/atom_composition.png` | `analysis/_loose/misc/atom_composition.png` | misc | 80.3K |
| `analysis/baseline_intra_group_similarity.csv` | `analysis/_loose/misc/baseline_intra_group_similarity.csv` | misc | 522.9K |
| `analysis/chembl_full_analysis.py` | `tools/analysis_scripts/legacy_top_level/chembl_full_analysis.py` | classifier | 28.2K |
| `analysis/chembl_sim_analysis.py` | `tools/analysis_scripts/legacy_top_level/chembl_sim_analysis.py` | classifier | 7.2K |
| `analysis/chembl_sim_full_analysis.py` | `tools/analysis_scripts/legacy_top_level/chembl_sim_full_analysis.py` | classifier | 11.7K |
| `analysis/chemical_space_2d.png` | `analysis/_loose/misc/chemical_space_2d.png` | misc | 980.9K |
| `analysis/compare_all_properties.py` | `tools/analysis_scripts/legacy_top_level/compare_all_properties.py` | misc | 9.9K |
| `analysis/compare_molecular_properties.py` | `tools/analysis_scripts/legacy_top_level/compare_molecular_properties.py` | misc | 14.8K |
| `analysis/compute_group_fingerprint_similarity.py` | `tools/analysis_scripts/legacy_top_level/compute_group_fingerprint_similarity.py` | misc | 7.5K |
| `analysis/correlation_heatmap.png` | `analysis/_loose/misc/correlation_heatmap.png` | misc | 253.7K |
| `analysis/dock_diffsbdd_single_reward_matrix_20260613.py` | `tools/analysis_scripts/legacy_top_level/dock_diffsbdd_single_reward_matrix_20260613.py` | diffsbdd_rl | 14.1K |
| `analysis/epoch20_intra_group_similarity.csv` | `analysis/_loose/misc/epoch20_intra_group_similarity.csv` | misc | 30.2K |
| `analysis/epoch20_properties.csv` | `analysis/_loose/misc/epoch20_properties.csv` | misc | 48.2K |
| `analysis/evaluate_classifier_on_base_diffsbdd.py` | `tools/analysis_scripts/legacy_top_level/evaluate_classifier_on_base_diffsbdd.py` | classifier | 23.9K |
| `analysis/formal_training_test_20260529_133444.md` | `analysis/_loose/misc/formal_training_test_20260529_133444.md` | misc | 4.2K |
| `analysis/guacamol.py` | `tools/analysis_scripts/legacy_top_level/guacamol.py` | misc | 13.4K |
| `analysis/laplace_vs_ema_demo.py` | `tools/analysis_scripts/legacy_top_level/laplace_vs_ema_demo.py` | misc | 6.2K |
| `analysis/logp_binning_analysis.py` | `tools/analysis_scripts/legacy_top_level/logp_binning_analysis.py` | misc | 4.1K |
| `analysis/molskill_generated_vs_real.py` | `tools/analysis_scripts/legacy_top_level/molskill_generated_vs_real.py` | misc | 12.8K |
| `analysis/plot_all_properties.py` | `tools/analysis_scripts/legacy_top_level/plot_all_properties.py` | misc | 8.9K |
| `analysis/plot_beautiful_core_properties.py` | `tools/analysis_scripts/legacy_top_level/plot_beautiful_core_properties.py` | misc | 11.7K |
| `analysis/plot_core_properties.py` | `tools/analysis_scripts/legacy_top_level/plot_core_properties.py` | misc | 11.1K |
| `analysis/plot_core_properties_raw_sa.py` | `tools/analysis_scripts/legacy_top_level/plot_core_properties_raw_sa.py` | misc | 11.8K |
| `analysis/plot_core_properties_raw_sa_cmeb.py` | `tools/analysis_scripts/legacy_top_level/plot_core_properties_raw_sa_cmeb.py` | misc | 11.8K |
| `analysis/plot_core_properties_raw_sa_cmeb_2.py` | `tools/analysis_scripts/legacy_top_level/plot_core_properties_raw_sa_cmeb_2.py` | misc | 11.6K |
| `analysis/plot_generated_classifier_score_distribution_20260613.py` | `tools/analysis_scripts/legacy_top_level/plot_generated_classifier_score_distribution_20260613.py` | classifier | 6.4K |
| `analysis/property_comparison.png` | `analysis/_loose/misc/property_comparison.png` | misc | 367.9K |
| `analysis/property_comparison_table.csv` | `analysis/_loose/misc/property_comparison_table.csv` | misc | 710B |
| `analysis/property_comparison_table.png` | `analysis/_loose/misc/property_comparison_table.png` | misc | 247.3K |
| `analysis/property_table.png` | `analysis/_loose/misc/property_table.png` | misc | 202.1K |
| `analysis/score_diffgui_diffsbdd_classifier_v031_20260615.py` | `tools/analysis_scripts/legacy_top_level/score_diffgui_diffsbdd_classifier_v031_20260615.py` | classifier | 8.0K |
| `analysis/simulate_logp_batch.py` | `tools/analysis_scripts/legacy_top_level/simulate_logp_batch.py` | misc | 4.3K |
| `analysis/test_classifier_on_260407_sdf.py` | `tools/analysis_scripts/legacy_top_level/test_classifier_on_260407_sdf.py` | classifier | 10.3K |
| `analysis/test_logp_on_real_mols.py` | `tools/analysis_scripts/legacy_top_level/test_logp_on_real_mols.py` | misc | 5.9K |
