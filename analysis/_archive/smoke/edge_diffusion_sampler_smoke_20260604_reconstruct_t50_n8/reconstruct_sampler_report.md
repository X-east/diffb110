# Edge Diffusion Sampler Smoke

## Scope

This is an isolated x/h/e edge-diffusion sampler smoke test. It does not import or modify DDPO/RL modules, rewards, launch configs, or the normal DiffSBDD reinforcement-learning path.

## Configuration

- mode: `reconstruct`
- checkpoint: `analysis/edge_diffusion_xhe_overfit_20260603_n128_uniform_gpu/edge_diffusion_overfit.pt`
- edge_npz: `analysis/edge_data_chain_20260602_repair_n1000/train_edge.npz`
- semantic_npz: `analysis/edge_semantic_bundle_20260602_n1000/train_edge_semantic.npz`
- device: `cuda`
- molecules: 8
- start_timestep: 50
- reverse_steps: 51
- edge_prior: `uniform`
- pos_noise_std: 0.05
- pos_noise_mode: `t_scaled`
- atom_corruption_prob: 0.05

> `reconstruct` mode starts from noised real CrossDocked x/h/e states and tests whether the reverse chain plus predicted-edge decoder can run end to end.

## Metrics

| metric | value |
|---|---:|
| elapsed_seconds | 4.112895 |
| seconds_per_molecule | 0.514112 |
| mean_sampling_seconds | 0.399289 |
| p95_sampling_seconds | 0.557124 |
| atom_accuracy_mean | 0.843650 |
| position_rmse_mean | 0.067567 |
| exact_halfedge_match_rate | 0.875000 |
| bond_pair_set_match_rate | 1.000000 |
| bond_type_map_match_rate | 0.875000 |
| mean_halfedge_accuracy | 0.999731 |
| mean_bond_pair_precision | 1.000000 |
| mean_bond_pair_recall | 1.000000 |
| plain_sanitize_success_rate | 0.750000 |
| semantic_sanitize_success_rate | 0.750000 |
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
