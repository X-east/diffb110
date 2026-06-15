# Edge Denoiser Train/Val Report

- edge_npz: `analysis/edge_data_chain_20260602_n1000/train_edge.npz`
- semantic_npz: `analysis/edge_semantic_bundle_20260602_n1000/train_edge_semantic.npz`
- device: `cuda`
- max_molecules: 1000
- split_mode: `train_val`
- train_molecules: 800
- eval_molecules: 200
- epochs: 80
- batch_size: 16
- model_kind: `node_edge`
- focal_gamma: 1.0
- class_weight_mode: `diffgui`
- no_bond_weight: 0.1
- pos_noise_std: 0.05
- reconstruction_validation: `True`
- reconstruction_timestep: 500
- reconstruction_limit: 100

## Class Counts And Weights

| bond type | count | weight |
|---|---:|---:|
| no_bond | 222488 | 0.100000 |
| single | 10392 | 1.000000 |
| double | 1657 | 1.000000 |
| triple | 49 | 1.000000 |
| aromatic | 8104 | 1.000000 |

## Eval Class Counts

| bond type | count |
|---|---:|
| no_bond | 53929 |
| single | 2574 |
| double | 439 |
| triple | 12 |
| aromatic | 1934 |

## Final Metrics

| metric | value |
|---|---:|
| train_loss | 0.005647 |
| eval_loss | 0.014114 |
| accuracy | 0.995092 |
| balanced_accuracy | 0.901591 |
| macro_precision | 0.936007 |
| macro_f1 | 0.917340 |
| real_bond_precision | 0.941722 |
| real_bond_recall | 0.941722 |
| real_bond_f1 | 0.941722 |
| no_bond_recall | 1.000000 |
| precision_single | 0.956182 |
| recall_single | 0.949495 |
| f1_single | 0.952826 |
| precision_double | 0.879679 |
| recall_double | 0.749431 |
| f1_double | 0.809348 |
| precision_triple | 0.909091 |
| recall_triple | 0.833333 |
| f1_triple | 0.869565 |
| precision_aromatic | 0.935084 |
| recall_aromatic | 0.975698 |
| f1_aromatic | 0.954959 |

## Final Reconstruction Metrics

| metric | value |
|---|---:|
| recon_molecules | 100.000000 |
| recon_exact_halfedge_match_rate | 0.440000 |
| recon_bond_pair_set_match_rate | 1.000000 |
| recon_bond_type_map_match_rate | 0.440000 |
| recon_plain_sanitize_success_rate | 0.920000 |
| recon_semantic_sanitize_success_rate | 0.910000 |
| recon_plain_smiles_no_stereo_match_rate | 0.430000 |
| recon_semantic_smiles_no_stereo_match_rate | 0.520000 |
| recon_mean_bad_bond_length_fraction | 0.000000 |

## Best Balanced Accuracy Epoch

- epoch: 77
- balanced_accuracy: 0.945689
- real_bond_recall: 0.943335

## Files

- [edge_denoiser_overfit_metrics.csv](edge_denoiser_overfit_metrics.csv)
- [edge_denoiser_overfit_summary.json](edge_denoiser_overfit_summary.json)
- [edge_denoiser_overfit.pt](edge_denoiser_overfit.pt)

![curves](edge_denoiser_overfit_curves.png)
