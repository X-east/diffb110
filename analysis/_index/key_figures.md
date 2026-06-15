# analysis 关键图索引

这里只列出每个目录扫描到的前若干张图，方便后续找 PPT/报告素材。

## 手动整理的重点图

- classifier v3 / v0.3.1 关键图合集：[`analysis/classifier/key_figures_v3_20260615/`](../classifier/key_figures_v3_20260615/)
  - 适合讲分类器数据构建、负样本类别、生成分子打分偏严格、DiffGui vs DiffSBDD 原始生成分子打分对比。

| 目录 | 类别 | 示例图 |
|---|---|---|
| `analysis/benchmark/5xzr_8j6_benchmark_20260611` | benchmark | ![](../benchmark/5xzr_8j6_benchmark_20260611/summary/benchmark_docking_scores.png) |
| `analysis/benchmark/8gjl_c1101_benchmark_20260603_500mol_500steps` | benchmark | ![](../benchmark/8gjl_c1101_benchmark_20260603_500mol_500steps/docking/unidock_fast_8gjl_c1101/summary/docking_best5_mean.png) |
| `analysis/benchmark/generated_reconstruction_benchmark_20260601_external_n100` | benchmark | ![](../benchmark/generated_reconstruction_benchmark_20260601_external_n100/sanitize_success_by_method.png) |
| `analysis/benchmark/generated_reconstruction_benchmark_20260602_local_bond_v0_external_n100` | benchmark | ![](../benchmark/generated_reconstruction_benchmark_20260602_local_bond_v0_external_n100/sanitize_success_by_method.png) |
| `analysis/benchmark/generated_reconstruction_benchmark_20260602_local_bond_v0_n500` | benchmark | ![](../benchmark/generated_reconstruction_benchmark_20260602_local_bond_v0_n500/sanitize_success_by_method.png) |
| `analysis/benchmark/model_strategy_20260531` | benchmark | ![](../benchmark/model_strategy_20260531/validity_by_model.png) |
| `analysis/classifier/chembl_full_analysis` | classifier | ![](../classifier/chembl_full_analysis/histogram_num_h_acceptors.png) |
| `analysis/classifier/chembl_sim_full` | classifier | ![](../classifier/chembl_sim_full/hist_ECFP4_full.png) |
| `analysis/classifier/classifier_failure_review_20260608` | classifier | ![](../classifier/classifier_failure_review_20260608/classifier_failure_review.svg) |
| `analysis/classifier/classifier_generated_score_distribution_20260613` | classifier | ![](../classifier/classifier_generated_score_distribution_20260613/generated_molecule_classifier_score_distribution.svg) |
| `analysis/classifier/classifier_test_260407_sdf_20260527` | classifier | ![](../classifier/classifier_test_260407_sdf_20260527/classifier_probability_hist.png) |
| `analysis/classifier/classifier_v0_2_1_weighted_20260608` | classifier | ![](../classifier/classifier_v0_2_1_weighted_20260608/base_diffsbdd_260407_distribution/02_probability_vs_properties.png) |
| `analysis/classifier/classifier_v0_2_hardneg_20260608` | classifier | ![](../classifier/classifier_v0_2_hardneg_20260608/base_diffsbdd_260407_distribution/02_probability_vs_properties.png) |
| `analysis/classifier/classifier_v0_3_scaffold_curation_20260610` | classifier | ![](../classifier/classifier_v0_3_scaffold_curation_20260610/primary_reason_counts.png) |
| `analysis/classifier/diffgui_classifier_score_20260609` | classifier | ![](../classifier/diffgui_classifier_score_20260609/diffgui_vs_diffsbdd_classifier_distribution.png) |
| `analysis/classifier/diffgui_diffsbdd_classifier_v031_20260615` | classifier | ![](../classifier/diffgui_diffsbdd_classifier_v031_20260615/diffgui_vs_diffsbdd_classifier_v031_distribution.svg) |
| `analysis/classifier/final_classifier_dataset_selection_20260527` | classifier | ![](../classifier/final_classifier_dataset_selection_20260527/selected_positive_distribution.png) |
| `analysis/classifier/good_bad_classifier_curation_20260608` | classifier | ![](../classifier/good_bad_classifier_curation_20260608/curation_label_distribution.png) |
| `analysis/diffgui_rl/diffgui_big_runs_summary_20260611` | diffgui_rl | ![](../diffgui_rl/diffgui_big_runs_summary_20260611/03_docking_runtime_trends.svg) |
| `analysis/diffgui_rl/diffgui_docking_batch_sweep_20260609` | diffgui_rl | ![](../diffgui_rl/diffgui_docking_batch_sweep_20260609/batch_sweep_summary.png) |
| `analysis/diffgui_rl/diffgui_docking_batch_sweep_20260609_last1_b24_b32` | diffgui_rl | ![](../diffgui_rl/diffgui_docking_batch_sweep_20260609_last1_b24_b32/batch_sweep_summary.png) |
| `analysis/diffgui_rl/diffgui_docking_batch_sweep_20260609_logiccheck_bs2` | diffgui_rl | ![](../diffgui_rl/diffgui_docking_batch_sweep_20260609_logiccheck_bs2/batch_sweep_summary.png) |
| `analysis/diffgui_rl/diffgui_docking_batch_sweep_20260609_preflight_bs2` | diffgui_rl | ![](../diffgui_rl/diffgui_docking_batch_sweep_20260609_preflight_bs2/batch_sweep_summary.png) |
| `analysis/diffgui_rl/diffgui_ring_last3_lr3e5_training_plots_20260611` | diffgui_rl | ![](../diffgui_rl/diffgui_ring_last3_lr3e5_training_plots_20260611/04_ring_complexity_diagnostics.png) |
| `analysis/diffgui_rl/diffgui_timestep_investigation_20260609` | diffgui_rl | ![](../diffgui_rl/diffgui_timestep_investigation_20260609/timestep_fix_benchmark.png) |
| `analysis/diffsbdd_rl/diffsbdd_8gpu_5000_score_20260608` | diffsbdd_rl | ![](../diffsbdd_rl/diffsbdd_8gpu_5000_score_20260608/score_vs_properties.png) |
| `analysis/diffsbdd_rl/diffsbdd_rl_reset_plan_20260611` | diffsbdd_rl | ![](../diffsbdd_rl/diffsbdd_rl_reset_plan_20260611/selected_run_descriptor_drift.svg) |
| `analysis/diffsbdd_rl/diffsbdd_single_reward_matrix_20260613` | diffsbdd_rl | ![](../diffsbdd_rl/diffsbdd_single_reward_matrix_20260613/final_epoch_bar_comparison.png) |
| `analysis/diffsbdd_rl/good_pool_audit_20260604_174051` | diffsbdd_rl | ![](../diffsbdd_rl/good_pool_audit_20260604_174051/retained_score_vs_carbon.png) |
| `analysis/diffsbdd_rl/kl_constraint_anomaly_20260604` | diffsbdd_rl | ![](../diffsbdd_rl/kl_constraint_anomaly_20260604/kl0p05_epoch86_outlier_hist.png) |
| `analysis/diffsbdd_rl/kl_diagnostic_20260604_kl0p02_25ep` | diffsbdd_rl | ![](../diffsbdd_rl/kl_diagnostic_20260604_kl0p02_25ep/weighted_kl_by_epoch.png) |
| `analysis/diffsbdd_rl/logprob_mode_experiment_20260604` | diffsbdd_rl | ![](../diffsbdd_rl/logprob_mode_experiment_20260604/ppo_logratio_abs_p95_by_epoch.png) |
| `analysis/diffsbdd_rl/reward_loophole_risk_register_20260527` | diffsbdd_rl | ![](../diffsbdd_rl/reward_loophole_risk_register_20260527/top_reward_loopholes.png) |
| `analysis/diffsbdd_rl/rl_data_sufficiency_20260608` | diffsbdd_rl | ![](../diffsbdd_rl/rl_data_sufficiency_20260608/base_diffsbdd_descriptor_histograms.png) |
| `analysis/diffsbdd_rl/rl_high_carbon_sparse_reward_20260604` | diffsbdd_rl | ![](../diffsbdd_rl/rl_high_carbon_sparse_reward_20260604/rl_descriptor_trends.png) |
| `analysis/diffsbdd_rl/rl_next_steps_20260608` | diffsbdd_rl | ![](../diffsbdd_rl/rl_next_steps_20260608/rl_next_step_priority_matrix.png) |
| `analysis/diffsbdd_small_mod/aromatic_benzene_diagnosis_20260527` | diffsbdd_small_mod | ![](../diffsbdd_small_mod/aromatic_benzene_diagnosis_20260527/global_aromatic_benzene_comparison.png) |
| `analysis/diffsbdd_small_mod/atom_score_diagnostics_20260602_n1000_gpu` | diffsbdd_small_mod | ![](../diffsbdd_small_mod/atom_score_diagnostics_20260602_n1000_gpu/baseline/atom_score_top1_fractions.png) |
| `analysis/diffsbdd_small_mod/final_refine_ablation_20260602_n100` | diffsbdd_small_mod | ![](../diffsbdd_small_mod/final_refine_ablation_20260602_n100/element_fractions.png) |
| `analysis/diffsbdd_small_mod/final_refine_ablation_20260602_n1000_gpu` | diffsbdd_small_mod | ![](../diffsbdd_small_mod/final_refine_ablation_20260602_n1000_gpu/element_fractions.png) |
| `analysis/diffsbdd_small_mod/hetero_composition_diagnosis_20260527` | diffsbdd_small_mod | ![](../diffsbdd_small_mod/hetero_composition_diagnosis_20260527/top_reward_co_only_runs.png) |
| `analysis/diffsbdd_small_mod/local_atom_mask_ablation_20260602_n1000_gpu` | diffsbdd_small_mod | ![](../diffsbdd_small_mod/local_atom_mask_ablation_20260602_n1000_gpu/element_fractions.png) |
| `analysis/diffsbdd_small_mod/local_bond_predictor_data_20260601` | diffsbdd_small_mod | ![](../diffsbdd_small_mod/local_bond_predictor_data_20260601/oracle_sanitize_success_by_split.png) |
| `analysis/diffsbdd_small_mod/local_bond_predictor_v0_20260602_small` | diffsbdd_small_mod | ![](../diffsbdd_small_mod/local_bond_predictor_v0_20260602_small/loss_curve.png) |
| `analysis/diffsbdd_small_mod/refinement_ablation_20260603_n1000_t100` | diffsbdd_small_mod | ![](../diffsbdd_small_mod/refinement_ablation_20260603_n1000_t100/openbabel_delta_vs_baseline.png) |
| `analysis/diffsbdd_xhe/diffsbdd_xhe_diffgui_scratch_20260606_longtest` | diffsbdd_xhe | ![](../diffsbdd_xhe/diffsbdd_xhe_diffgui_scratch_20260606_longtest/longtest_loss_components.png) |
| `analysis/diffsbdd_xhe/diffsbdd_xhe_joint_20260605_20k_6gpu` | diffsbdd_xhe | ![](../diffsbdd_xhe/diffsbdd_xhe_joint_20260605_20k_6gpu/eval_edge_metrics_last/confusion_warm_start_random_edge_train.png) |
| `analysis/diffsbdd_xhe/diffsbdd_xhe_predicted_reconstruction_20260605_val` | diffsbdd_xhe | ![](../diffsbdd_xhe/diffsbdd_xhe_predicted_reconstruction_20260605_val/predicted_edge_reconstruction_metrics.png) |
| `analysis/diffsbdd_xhe/diffsbdd_xhe_supervised_20260604_20k_6gpu` | diffsbdd_xhe | ![](../diffsbdd_xhe/diffsbdd_xhe_supervised_20260604_20k_6gpu/combined_edge_metrics_val.png) |
| `analysis/diffsbdd_xhe/edge_data_chain_20260602_n1000` | diffsbdd_xhe | ![](../diffsbdd_xhe/edge_data_chain_20260602_n1000/bond_type_distribution.png) |
| `analysis/diffsbdd_xhe/edge_data_chain_20260602_repair_n1000` | diffsbdd_xhe | ![](../diffsbdd_xhe/edge_data_chain_20260602_repair_n1000/bond_type_distribution.png) |
| `analysis/diffsbdd_xhe/edge_data_chain_20260604_repair_n10000` | diffsbdd_xhe | ![](../diffsbdd_xhe/edge_data_chain_20260604_repair_n10000/bond_type_distribution.png) |
| `analysis/diffsbdd_xhe/edge_data_oracle_crossdock_20260601` | diffsbdd_xhe | ![](../diffsbdd_xhe/edge_data_oracle_crossdock_20260601/bond_type_distribution.png) |
| `analysis/diffsbdd_xhe/edge_denoiser_ablation_20260603_A_inverse_clean` | diffsbdd_xhe | ![](../diffsbdd_xhe/edge_denoiser_ablation_20260603_A_inverse_clean/edge_denoiser_overfit_curves.png) |
| `analysis/diffsbdd_xhe/edge_denoiser_ablation_20260603_B_diffgui_noise005` | diffsbdd_xhe | ![](../diffsbdd_xhe/edge_denoiser_ablation_20260603_B_diffgui_noise005/edge_denoiser_overfit_curves.png) |
| `analysis/diffsbdd_xhe/edge_denoiser_ablation_20260603_C_diffgui_noise005_focal2` | diffsbdd_xhe | ![](../diffsbdd_xhe/edge_denoiser_ablation_20260603_C_diffgui_noise005_focal2/edge_denoiser_overfit_curves.png) |
| `analysis/diffsbdd_xhe/edge_denoiser_ablation_20260603_C_diffgui_noise005_focal2_gpu` | diffsbdd_xhe | ![](../diffsbdd_xhe/edge_denoiser_ablation_20260603_C_diffgui_noise005_focal2_gpu/edge_denoiser_overfit_curves.png) |
| `analysis/diffsbdd_xhe/edge_denoiser_ablation_20260603_D_diffgui_noise010` | diffsbdd_xhe | ![](../diffsbdd_xhe/edge_denoiser_ablation_20260603_D_diffgui_noise010/edge_denoiser_overfit_curves.png) |
| `analysis/diffsbdd_xhe/edge_denoiser_ablation_eval_20260603_B_diffgui_noise005` | diffsbdd_xhe | ![](../diffsbdd_xhe/edge_denoiser_ablation_eval_20260603_B_diffgui_noise005/metrics_by_t.png) |
| `analysis/diffsbdd_xhe/edge_denoiser_ablation_eval_20260603_C_diffgui_noise005_focal2` | diffsbdd_xhe | ![](../diffsbdd_xhe/edge_denoiser_ablation_eval_20260603_C_diffgui_noise005_focal2/metrics_by_t.png) |
| `analysis/diffsbdd_xhe/edge_denoiser_ablation_eval_20260603_C_diffgui_noise005_focal2_gpu` | diffsbdd_xhe | ![](../diffsbdd_xhe/edge_denoiser_ablation_eval_20260603_C_diffgui_noise005_focal2_gpu/metrics_by_t.png) |
| `analysis/diffsbdd_xhe/edge_denoiser_ablation_eval_20260603_D_diffgui_noise010` | diffsbdd_xhe | ![](../diffsbdd_xhe/edge_denoiser_ablation_eval_20260603_D_diffgui_noise010/metrics_by_t.png) |
| `analysis/diffsbdd_xhe/edge_denoiser_ablation_summary_20260603` | diffsbdd_xhe | ![](../diffsbdd_xhe/edge_denoiser_ablation_summary_20260603/train_final_key_metrics.png) |
| `analysis/diffsbdd_xhe/edge_denoiser_eval_20260603_n1000_val_tgrid_reconstruct_gpu` | diffsbdd_xhe | ![](../diffsbdd_xhe/edge_denoiser_eval_20260603_n1000_val_tgrid_reconstruct_gpu/metrics_by_t.png) |
| `analysis/diffsbdd_xhe/edge_denoiser_eval_20260603_n1000_val_tgrid_reconstruct_gpu_v2` | diffsbdd_xhe | ![](../diffsbdd_xhe/edge_denoiser_eval_20260603_n1000_val_tgrid_reconstruct_gpu_v2/metrics_by_t.png) |
| `analysis/diffsbdd_xhe/edge_denoiser_eval_20260603_n1000_val_tgrid_reconstruct_gpu_v3` | diffsbdd_xhe | ![](../diffsbdd_xhe/edge_denoiser_eval_20260603_n1000_val_tgrid_reconstruct_gpu_v3/metrics_by_t.png) |
| `analysis/diffsbdd_xhe/edge_denoiser_nodeedge_20260602_n32_gpu` | diffsbdd_xhe | ![](../diffsbdd_xhe/edge_denoiser_nodeedge_20260602_n32_gpu/edge_denoiser_overfit_curves.png) |
| `analysis/diffsbdd_xhe/edge_denoiser_nodeedge_20260603_n1000_train800_val200_gpu` | diffsbdd_xhe | ![](../diffsbdd_xhe/edge_denoiser_nodeedge_20260603_n1000_train800_val200_gpu/edge_denoiser_overfit_curves.png) |
| `analysis/diffsbdd_xhe/edge_denoiser_overfit_20260602_n128_gpu` | diffsbdd_xhe | ![](../diffsbdd_xhe/edge_denoiser_overfit_20260602_n128_gpu/edge_denoiser_overfit_curves.png) |
| `analysis/diffsbdd_xhe/edge_denoiser_overfit_20260602_n32_gpu` | diffsbdd_xhe | ![](../diffsbdd_xhe/edge_denoiser_overfit_20260602_n32_gpu/edge_denoiser_overfit_curves.png) |
| `analysis/diffsbdd_xhe/edge_diffusion_ddp_20260604_10k_6gpu_test3` | diffsbdd_xhe | ![](../diffsbdd_xhe/edge_diffusion_ddp_20260604_10k_6gpu_test3/edge_diffusion_ddp_curves.png) |
| `analysis/diffsbdd_xhe/edge_diffusion_xhe_overfit_20260603_n128_gpu` | diffsbdd_xhe | ![](../diffsbdd_xhe/edge_diffusion_xhe_overfit_20260603_n128_gpu/edge_diffusion_overfit_curves.png) |
| `analysis/diffsbdd_xhe/edge_diffusion_xhe_overfit_20260603_n128_uniform_gpu` | diffsbdd_xhe | ![](../diffsbdd_xhe/edge_diffusion_xhe_overfit_20260603_n128_uniform_gpu/edge_diffusion_overfit_curves.png) |
| `analysis/diffsbdd_xhe/edge_semantic_bundle_20260602_n1000` | diffsbdd_xhe | ![](../diffsbdd_xhe/edge_semantic_bundle_20260602_n1000/edge_semantic_total_charge_distribution.png) |
| `analysis/diffsbdd_xhe/edge_semantic_bundle_20260604_n10000` | diffsbdd_xhe | ![](../diffsbdd_xhe/edge_semantic_bundle_20260604_n10000/edge_semantic_total_charge_distribution.png) |
| `analysis/diffsbdd_xhe/generated_edge_denoiser_offmanifold_20260603_n500_t100_uniform_t999` | diffsbdd_xhe | ![](../diffsbdd_xhe/generated_edge_denoiser_offmanifold_20260603_n500_t100_uniform_t999/sanitize_success_by_method.png) |
| `analysis/diffsbdd_xhe/xhe_ep37_failure_analysis_20260606` | diffsbdd_xhe | ![](../diffsbdd_xhe/xhe_ep37_failure_analysis_20260606/failure_modes_and_bond_types.png) |
| `analysis/_archive` | internal | ![](../_archive/smoke/edge_denoiser_reconval_smoke_20260603_diffgui_gpu/edge_denoiser_overfit_curves.png) |
| `analysis/_compat_links` | internal | ![](../_compat_links/top_level/atom_composition.png) |
| `analysis/_index` | internal | ![](largest_directories.png) |
| `analysis/_loose` | internal | ![](../_loose/misc/atom_composition.png) |
| `analysis/preprocessing/reconstruction_ablation_crossdock_20260601` | preprocessing | ![](../preprocessing/reconstruction_ablation_crossdock_20260601/sanitize_success_by_method.png) |
| `analysis/reconstruction/bond_predictor_diagnostics_20260602_n300` | reconstruction | ![](../reconstruction/bond_predictor_diagnostics_20260602_n300/generated_bad_bond_length_by_method.png) |
| `analysis/reconstruction/generated_reconstruction_ablation_20260601_n50` | reconstruction | ![](../reconstruction/generated_reconstruction_ablation_20260601_n50/sanitize_success_by_method.png) |
| `analysis/reconstruction/generated_reconstruction_ablation_20260601_n500` | reconstruction | ![](../reconstruction/generated_reconstruction_ablation_20260601_n500/sanitize_success_by_method.png) |
| `analysis/reconstruction/oracle_original_sdf_audit_20260602_n1000` | reconstruction | ![](../reconstruction/oracle_original_sdf_audit_20260602_n1000/oracle_original_sdf_strategy_rates.png) |
| `analysis/reconstruction/semantic_oracle_reconstruction_20260602_n1000` | reconstruction | ![](../reconstruction/semantic_oracle_reconstruction_20260602_n1000/semantic_oracle_repair_strategies.png) |
