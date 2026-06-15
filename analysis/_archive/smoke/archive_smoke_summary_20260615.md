# smoke/debug analysis 目录归档记录

- 执行时间：2026-06-15T09:30:28
- 来源清单：`analysis/_index/analysis_catalog.csv`
- 归档目标：`analysis/_archive/smoke`
- 处理条目：40
- 已移动：40
- 创建旧路径软链接：33
- 需要复查/跳过：0

## 说明

- 本次只归档 `archive_smoke` 目录，不移动正式结果目录。
- 若目录在 `docs/work_records/work_days/launch/tools` 中被文本引用，会在旧位置保留软链接。
- 后续重建 `analysis/_index` 时，索引脚本会跳过顶层软链接，避免重复统计。

## 移动清单

| 旧路径 | 新路径 | 大小 | 软链接 | 引用数 |
|---|---|---:|---:|---:|
| `analysis/bond_predictor_diagnostics_20260602_confusion_smoke` | `analysis/_archive/smoke/bond_predictor_diagnostics_20260602_confusion_smoke` | 432.4K | 否 | 0 |
| `analysis/bond_predictor_diagnostics_20260602_smoke` | `analysis/_archive/smoke/bond_predictor_diagnostics_20260602_smoke` | 643.3K | 否 | 0 |
| `analysis/dataset_selection_dryrun_20260527` | `analysis/_archive/smoke/dataset_selection_dryrun_20260527` | 50.2M | 是 | 2 |
| `analysis/diffgui_100step_multigpu_smoke_20260609` | `analysis/_archive/smoke/diffgui_100step_multigpu_smoke_20260609` | 199.6K | 是 | 1 |
| `analysis/diffgui_100step_multigpu_smoke_20260609_batch32` | `analysis/_archive/smoke/diffgui_100step_multigpu_smoke_20260609_batch32` | 182.7K | 是 | 1 |
| `analysis/diffgui_100step_rollout_smoke_20260609` | `analysis/_archive/smoke/diffgui_100step_rollout_smoke_20260609` | 2.4K | 是 | 2 |
| `analysis/diffgui_100step_rollout_smoke_20260609_1x24` | `analysis/_archive/smoke/diffgui_100step_rollout_smoke_20260609_1x24` | 4.9K | 否 | 0 |
| `analysis/diffgui_100step_rollout_smoke_20260609_amp_2x24` | `analysis/_archive/smoke/diffgui_100step_rollout_smoke_20260609_amp_2x24` | 1.2K | 否 | 0 |
| `analysis/diffgui_100step_rollout_smoke_20260609_amp_micro_2x24` | `analysis/_archive/smoke/diffgui_100step_rollout_smoke_20260609_amp_micro_2x24` | 5.3K | 是 | 1 |
| `analysis/diffgui_100step_rollout_smoke_20260609_freeze_last1_2x24` | `analysis/_archive/smoke/diffgui_100step_rollout_smoke_20260609_freeze_last1_2x24` | 5.7K | 是 | 1 |
| `analysis/diffgui_100step_rollout_smoke_20260609_small` | `analysis/_archive/smoke/diffgui_100step_rollout_smoke_20260609_small` | 15.1K | 是 | 1 |
| `analysis/diffgui_ring_reward_smoke_20260610` | `analysis/_archive/smoke/diffgui_ring_reward_smoke_20260610` | 187.3K | 是 | 1 |
| `analysis/diffsbdd_xhe_joint_20260605_smoke` | `analysis/_archive/smoke/diffsbdd_xhe_joint_20260605_smoke` | 40.6M | 是 | 1 |
| `analysis/diffsbdd_xhe_predicted_reconstruction_20260605_val_smoke_n8` | `analysis/_archive/smoke/diffsbdd_xhe_predicted_reconstruction_20260605_val_smoke_n8` | 379.8K | 是 | 1 |
| `analysis/edge_denoiser_eval_shared_recon_smoke_20260603_gpu` | `analysis/_archive/smoke/edge_denoiser_eval_shared_recon_smoke_20260603_gpu` | 103.4K | 是 | 1 |
| `analysis/edge_denoiser_eval_smoke_20260603` | `analysis/_archive/smoke/edge_denoiser_eval_smoke_20260603` | 112.1K | 否 | 0 |
| `analysis/edge_denoiser_nodeedge_smoke_20260602` | `analysis/_archive/smoke/edge_denoiser_nodeedge_smoke_20260602` | 190.7K | 是 | 1 |
| `analysis/edge_denoiser_nodeedge_smoke_20260602_gpu` | `analysis/_archive/smoke/edge_denoiser_nodeedge_smoke_20260602_gpu` | 190.7K | 是 | 1 |
| `analysis/edge_denoiser_nodeedge_smoke_20260602_gpu_escalated` | `analysis/_archive/smoke/edge_denoiser_nodeedge_smoke_20260602_gpu_escalated` | 191.5K | 是 | 1 |
| `analysis/edge_denoiser_overfit_smoke_20260602` | `analysis/_archive/smoke/edge_denoiser_overfit_smoke_20260602` | 66.5K | 是 | 1 |
| `analysis/edge_denoiser_reconval_smoke_20260603_diffgui_gpu` | `analysis/_archive/smoke/edge_denoiser_reconval_smoke_20260603_diffgui_gpu` | 627.9K | 是 | 1 |
| `analysis/edge_denoiser_reconval_smoke_20260603_inverse_gpu` | `analysis/_archive/smoke/edge_denoiser_reconval_smoke_20260603_inverse_gpu` | 631.0K | 是 | 1 |
| `analysis/edge_denoiser_split_smoke_20260603` | `analysis/_archive/smoke/edge_denoiser_split_smoke_20260603` | 72.7K | 是 | 1 |
| `analysis/edge_diffusion_ddp_smoke_20260604` | `analysis/_archive/smoke/edge_diffusion_ddp_smoke_20260604` | 333.4K | 是 | 2 |
| `analysis/edge_diffusion_sampler_smoke_20260604_10k3ep_prior_t100_n32` | `analysis/_archive/smoke/edge_diffusion_sampler_smoke_20260604_10k3ep_prior_t100_n32` | 349.2K | 是 | 1 |
| `analysis/edge_diffusion_sampler_smoke_20260604_10k3ep_reconstruct_t100_n32` | `analysis/_archive/smoke/edge_diffusion_sampler_smoke_20260604_10k3ep_reconstruct_t100_n32` | 497.8K | 是 | 1 |
| `analysis/edge_diffusion_sampler_smoke_20260604_ddp_ckpt_t10_n4` | `analysis/_archive/smoke/edge_diffusion_sampler_smoke_20260604_ddp_ckpt_t10_n4` | 151.4K | 是 | 1 |
| `analysis/edge_diffusion_sampler_smoke_20260604_prior_t50_n8` | `analysis/_archive/smoke/edge_diffusion_sampler_smoke_20260604_prior_t50_n8` | 176.0K | 是 | 1 |
| `analysis/edge_diffusion_sampler_smoke_20260604_reconstruct_t50_n8` | `analysis/_archive/smoke/edge_diffusion_sampler_smoke_20260604_reconstruct_t50_n8` | 171.3K | 是 | 1 |
| `analysis/edge_diffusion_xhe_overfit_smoke_20260603` | `analysis/_archive/smoke/edge_diffusion_xhe_overfit_smoke_20260603` | 294.6K | 否 | 0 |
| `analysis/edge_transition_smoke_20260603_full_transition` | `analysis/_archive/smoke/edge_transition_smoke_20260603_full_transition` | 1.9K | 是 | 1 |
| `analysis/generated_reconstruction_ablation_20260601_smoke` | `analysis/_archive/smoke/generated_reconstruction_ablation_20260601_smoke` | 120.2K | 是 | 1 |
| `analysis/generated_reconstruction_benchmark_20260601_external_smoke_n4` | `analysis/_archive/smoke/generated_reconstruction_benchmark_20260601_external_smoke_n4` | 753.1K | 是 | 1 |
| `analysis/generated_reconstruction_benchmark_20260601_smoke` | `analysis/_archive/smoke/generated_reconstruction_benchmark_20260601_smoke` | 294.2K | 是 | 1 |
| `analysis/generated_reconstruction_benchmark_20260602_local_bond_v0_smoke_n20` | `analysis/_archive/smoke/generated_reconstruction_benchmark_20260602_local_bond_v0_smoke_n20` | 362.8K | 是 | 1 |
| `analysis/local_bond_predictor_data_20260601_smoke` | `analysis/_archive/smoke/local_bond_predictor_data_20260601_smoke` | 215.4K | 是 | 1 |
| `analysis/local_bond_predictor_v0_20260602_smoke` | `analysis/_archive/smoke/local_bond_predictor_v0_20260602_smoke` | 259.5K | 是 | 1 |
| `analysis/refinement_ablation_smoke_20260603` | `analysis/_archive/smoke/refinement_ablation_smoke_20260603` | 323.5K | 是 | 1 |
| `analysis/refinement_ablation_smoke_20260603_v2` | `analysis/_archive/smoke/refinement_ablation_smoke_20260603_v2` | 354.3K | 否 | 0 |
| `analysis/refinement_ablation_smoke_atomdiag_20260603` | `analysis/_archive/smoke/refinement_ablation_smoke_atomdiag_20260603` | 441.4K | 是 | 6 |
