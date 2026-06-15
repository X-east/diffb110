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
- elapsed_seconds: 72.23
- seconds_per_epoch: 0.9029
- eval_timestep: 500
- pos_noise_std: 0.05
- pos_noise_mode: `t_scaled`
- atom_corruption_prob: 0.05
- edge_prior: `absorb`
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
| train_total | 0.198696 |
| train_pos | 0.000837 |
| train_atom | 0.060050 |
| train_edge | 0.137425 |
| train_bond_length | 0.000767 |
| eval_total | 0.266862 |
| eval_pos | 0.000594 |
| eval_atom | 0.064900 |
| eval_edge | 0.201103 |
| eval_bond_length | 0.000528 |
| eval_edge_accuracy | 0.980711 |
| eval_edge_balanced_accuracy | 0.831311 |
| eval_edge_real_bond_precision | 0.770291 |
| eval_edge_real_bond_recall | 0.770291 |
| eval_edge_real_bond_f1 | 0.770291 |
| eval_atom_accuracy | 0.985215 |
| eval_pos_rmse | 0.024389 |
| eval_bond_length_mae | 0.026195 |
| eval_edge_recall_single | 0.634536 |
| eval_edge_recall_double | 0.688645 |
| eval_edge_recall_triple | 0.857143 |
| eval_edge_recall_aromatic | 0.976229 |

## Final Reconstruction Metrics

| metric | value |
|---|---:|
| recon_molecules | 32.000000 |
| recon_exact_halfedge_match_rate | 0.031250 |
| recon_bond_pair_set_match_rate | 1.000000 |
| recon_bond_type_map_match_rate | 0.031250 |
| recon_plain_sanitize_success_rate | 0.093750 |
| recon_semantic_sanitize_success_rate | 0.093750 |
| recon_plain_smiles_no_stereo_match_rate | 0.031250 |
| recon_semantic_smiles_no_stereo_match_rate | 0.031250 |
| recon_mean_bad_bond_length_fraction | 0.000000 |

## Best Real-Bond F1 Epoch

- epoch: 72
- eval_edge_real_bond_f1: 0.848323
- eval_edge_real_bond_recall: 0.848323

## Files

- [edge_diffusion_overfit_metrics.csv](edge_diffusion_overfit_metrics.csv)
- [edge_diffusion_overfit_summary.json](edge_diffusion_overfit_summary.json)
- [edge_diffusion_overfit.pt](edge_diffusion_overfit.pt)
- [edge_diffusion_reconstruction_rows.csv](edge_diffusion_reconstruction_rows.csv)

![curves](edge_diffusion_overfit_curves.png)
