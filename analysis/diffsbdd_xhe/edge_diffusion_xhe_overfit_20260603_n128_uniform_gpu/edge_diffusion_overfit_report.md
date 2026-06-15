# Edge Diffusion x/h/e Overfit Report

- edge_npz: `analysis/edge_data_chain_20260602_repair_n1000/train_edge.npz`
- semantic_npz: `analysis/edge_semantic_bundle_20260602_n1000/train_edge_semantic.npz`
- device: `cuda`
- max_molecules: 128
- split_mode: `overfit`
- train_molecules: 128
- eval_molecules: 128
- epochs: 80
- batch_size: 8
- elapsed_seconds: 78.82
- seconds_per_epoch: 0.9852
- eval_timestep: 500
- pos_noise_std: 0.05
- pos_noise_mode: `t_scaled`
- atom_corruption_prob: 0.05
- edge_prior: `uniform`
- edge_class_weight_mode: `inverse_sqrt`
- reconstruction_validation: `True`

## Edge Class Counts And Weights

| bond type | train count | eval count | weight |
|---|---:|---:|---:|
| no_bond | 34810 | 34810 | 0.056746 |
| single | 1691 | 1691 | 0.257391 |
| double | 273 | 273 | 0.639615 |
| triple | 7 | 7 | 3.743252 |
| aromatic | 1220 | 1220 | 0.302996 |

## Final Metrics

| metric | value |
|---|---:|
| train_total | 0.248334 |
| train_pos | 0.000919 |
| train_atom | 0.104577 |
| train_edge | 0.142426 |
| train_bond_length | 0.000826 |
| eval_total | 0.247274 |
| eval_pos | 0.000590 |
| eval_atom | 0.077994 |
| eval_edge | 0.168426 |
| eval_bond_length | 0.000529 |
| eval_edge_accuracy | 0.985316 |
| eval_edge_balanced_accuracy | 0.716135 |
| eval_edge_real_bond_precision | 0.825651 |
| eval_edge_real_bond_recall | 0.825133 |
| eval_edge_real_bond_f1 | 0.825392 |
| eval_atom_accuracy | 0.983535 |
| eval_pos_rmse | 0.024327 |
| eval_bond_length_mae | 0.025928 |
| eval_edge_recall_single | 0.820225 |
| eval_edge_recall_double | 0.769231 |
| eval_edge_recall_triple | 0.142857 |
| eval_edge_recall_aromatic | 0.848361 |

## Final Reconstruction Metrics

| metric | value |
|---|---:|
| recon_molecules | 32.000000 |
| recon_exact_halfedge_match_rate | 0.125000 |
| recon_bond_pair_set_match_rate | 1.000000 |
| recon_bond_type_map_match_rate | 0.125000 |
| recon_plain_sanitize_success_rate | 0.562500 |
| recon_semantic_sanitize_success_rate | 0.562500 |
| recon_plain_smiles_no_stereo_match_rate | 0.156250 |
| recon_semantic_smiles_no_stereo_match_rate | 0.187500 |
| recon_mean_bad_bond_length_fraction | 0.000000 |

## Best Real-Bond F1 Epoch

- epoch: 77
- eval_edge_real_bond_f1: 0.841116
- eval_edge_real_bond_recall: 0.841116

## Files

- [edge_diffusion_overfit_metrics.csv](edge_diffusion_overfit_metrics.csv)
- [edge_diffusion_overfit_summary.json](edge_diffusion_overfit_summary.json)
- [edge_diffusion_overfit.pt](edge_diffusion_overfit.pt)
- [edge_diffusion_reconstruction_rows.csv](edge_diffusion_reconstruction_rows.csv)

![curves](edge_diffusion_overfit_curves.png)
