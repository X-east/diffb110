# Edge Diffusion Sampler Smoke

## Scope

This is an isolated x/h/e edge-diffusion sampler smoke test. It does not import or modify DDPO/RL modules, rewards, launch configs, or the normal DiffSBDD reinforcement-learning path.

## Configuration

- mode: `reconstruct`
- checkpoint: `analysis/edge_diffusion_ddp_smoke_20260604/edge_diffusion_ddp_latest.pt`
- edge_npz: `analysis/edge_data_chain_20260602_repair_n1000/train_edge.npz`
- semantic_npz: `analysis/edge_semantic_bundle_20260602_n1000/train_edge_semantic.npz`
- device: `cuda`
- molecules: 4
- start_timestep: 10
- reverse_steps: 11
- edge_prior: `uniform`
- pos_noise_std: 0.05
- pos_noise_mode: `t_scaled`
- atom_corruption_prob: 0.05

> `reconstruct` mode starts from noised real CrossDocked x/h/e states and tests whether the reverse chain plus predicted-edge decoder can run end to end.

## Metrics

| metric | value |
|---|---:|
| elapsed_seconds | 1.490306 |
| seconds_per_molecule | 0.372576 |
| mean_sampling_seconds | 0.174075 |
| p95_sampling_seconds | 0.417173 |
| atom_accuracy_mean | 0.141313 |
| position_rmse_mean | 0.022363 |
| exact_halfedge_match_rate | 0.000000 |
| bond_pair_set_match_rate | 0.000000 |
| bond_type_map_match_rate | 0.000000 |
| mean_halfedge_accuracy | 0.883976 |
| mean_bond_pair_precision | 0.540865 |
| mean_bond_pair_recall | 0.373512 |
| plain_sanitize_success_rate | 0.000000 |
| semantic_sanitize_success_rate | 0.000000 |
| mean_bad_bond_length_fraction | 0.459135 |
| total_bad_bond_length_count | 37.000000 |

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
