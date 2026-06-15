# DiffSBDD-xhe Predicted Edge Reconstruction Smoke

## 目的

本测试验证训练后的 DiffSBDD-xhe edge head 是否能在真实 CrossDocked val 点云与真实 atom type 条件下，把 noised halfedge type 预测回可重建分子。这是 teacher-forced reconstruction smoke，不是完整自由采样，也不接入 RL。

## 配置

- checkpoint: `analysis/diffsbdd_xhe_supervised_20260604_20k_6gpu/lightning/diffsbdd-xhe-20k-6gpu-260604-b4-3ep/checkpoints/last.ckpt`
- config: `analysis/diffsbdd_xhe_supervised_20260604_20k_6gpu/diffsbdd_xhe_20k_6gpu_b4_3ep.yaml`
- split: `val`
- device: `cuda`
- molecules: 300
- timesteps: `[0, 10, 50, 100, 250, 500]`
- elapsed_seconds: 285.074919
- seconds_per_molecule_per_timestep: 0.158375

## 汇总

| timestep | pred_sanitize_success_rate | pred_vs_oracle_smiles_match_rate | mean_bond_pair_f1 | mean_halfedge_accuracy | mean_bad_bond_length_fraction | mean_pred_bond_count | mean_target_bond_count | mean_pred_three_member_rings |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000222 | 24.996667 | 24.996667 | 0.033333 |
| 10 | 0.996667 | 0.990000 | 1.000000 | 0.999897 | 0.000222 | 24.996667 | 24.996667 | 0.033333 |
| 50 | 0.823333 | 0.603333 | 0.994595 | 0.996578 | 0.009884 | 25.226667 | 24.996667 | 0.143333 |
| 100 | 0.273333 | 0.036667 | 0.939447 | 0.977616 | 0.107494 | 27.880000 | 24.996667 | 0.496667 |
| 250 | 0.000000 | 0.000000 | 0.534883 | 0.800725 | 0.613027 | 57.193333 | 24.996667 | 0.000000 |
| 500 | 0.726667 | 0.000000 | 0.070306 | 0.776875 | 0.302880 | 10.853333 | 24.996667 | 0.010000 |

## 解释边界

- 如果这里 sanitize/键恢复差，说明 edge head 或 x/h/e 训练还不够；不能直接进入完整 sampler/RL。
- 如果这里表现好，也只证明真实点云上的 edge denoising/reconstruction 可行；还需要完整 e sampler 与生成点云测试，才能判断真实生成质量提升。

## 文件

- [predicted_edge_reconstruction_rows.csv](predicted_edge_reconstruction_rows.csv)
- [predicted_edge_reconstruction_summary.csv](predicted_edge_reconstruction_summary.csv)
- [predicted_edge_reconstruction_summary.json](predicted_edge_reconstruction_summary.json)
- [predicted_edge_reconstruction_confusion.csv](predicted_edge_reconstruction_confusion.csv)
- [predicted_edge_reconstruction_metrics.png](predicted_edge_reconstruction_metrics.png)
- [oracle_edges_t0.sdf](oracle_edges_t0.sdf)
- [predicted_edges_t0.sdf](predicted_edges_t0.sdf)
- [oracle_edges_t10.sdf](oracle_edges_t10.sdf)
- [predicted_edges_t10.sdf](predicted_edges_t10.sdf)
- [oracle_edges_t50.sdf](oracle_edges_t50.sdf)
- [predicted_edges_t50.sdf](predicted_edges_t50.sdf)
- [oracle_edges_t100.sdf](oracle_edges_t100.sdf)
- [predicted_edges_t100.sdf](predicted_edges_t100.sdf)
- [oracle_edges_t250.sdf](oracle_edges_t250.sdf)
- [predicted_edges_t250.sdf](predicted_edges_t250.sdf)
- [oracle_edges_t500.sdf](oracle_edges_t500.sdf)
- [predicted_edges_t500.sdf](predicted_edges_t500.sdf)

![metrics](predicted_edge_reconstruction_metrics.png)
