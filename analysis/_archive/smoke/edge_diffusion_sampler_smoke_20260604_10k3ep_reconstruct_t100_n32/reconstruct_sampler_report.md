# Edge Diffusion Sampler Smoke

## Scope

This is an isolated x/h/e edge-diffusion sampler smoke test. It does not import or modify DDPO/RL modules, rewards, launch configs, or the normal DiffSBDD reinforcement-learning path.

## Configuration

- mode: `reconstruct`
- checkpoint: `analysis/edge_diffusion_ddp_20260604_10k_6gpu_test3/edge_diffusion_ddp_latest.pt`
- edge_npz: `analysis/edge_data_chain_20260604_repair_n10000/train_edge.npz`
- semantic_npz: `analysis/edge_semantic_bundle_20260604_n10000/train_edge_semantic.npz`
- device: `cuda`
- molecules: 32
- start_timestep: 100
- reverse_steps: 101
- edge_prior: `uniform`
- pos_noise_std: 0.05
- pos_noise_mode: `t_scaled`
- atom_corruption_prob: 0.05

> `reconstruct` mode starts from noised real CrossDocked x/h/e states and tests whether the reverse chain plus predicted-edge decoder can run end to end.

## Metrics

| metric | value |
|---|---:|
| elapsed_seconds | 67.874444 |
| seconds_per_molecule | 2.121076 |
| mean_sampling_seconds | 1.184420 |
| p95_sampling_seconds | 1.196100 |
| atom_accuracy_mean | 0.818679 |
| position_rmse_mean | 0.158214 |
| exact_halfedge_match_rate | 0.312500 |
| bond_pair_set_match_rate | 0.968750 |
| bond_type_map_match_rate | 0.312500 |
| mean_halfedge_accuracy | 0.993593 |
| mean_bond_pair_precision | 1.000000 |
| mean_bond_pair_recall | 0.997596 |
| plain_sanitize_success_rate | 0.781250 |
| semantic_sanitize_success_rate | 0.781250 |
| mean_bad_bond_length_fraction | 0.000000 |
| total_bad_bond_length_count | 0.000000 |

## Edge Class Names

| id | type |
|---:|---|
| 0 | no_bond |
| 1 | single |
| 2 | double |
| 3 | triple |
| 4 | aromatic |

## Files

- [reconstruct_sampler_rows.csv](reconstruct_sampler_rows.csv)
- [reconstruct_sampler_summary.json](reconstruct_sampler_summary.json)
- [reconstruct_predicted_edge_plain.sdf](reconstruct_predicted_edge_plain.sdf)
- [reconstruct_predicted_edge_semantic.sdf](reconstruct_predicted_edge_semantic.sdf)
- [reconstruct_sampler_metrics.png](reconstruct_sampler_metrics.png)

![sampler metrics](reconstruct_sampler_metrics.png)
