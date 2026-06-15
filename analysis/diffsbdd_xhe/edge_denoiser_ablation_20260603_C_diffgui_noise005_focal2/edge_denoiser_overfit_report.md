# Edge Denoiser Train/Val Report

- edge_npz: `analysis/edge_data_chain_20260602_n1000/train_edge.npz`
- semantic_npz: `analysis/edge_semantic_bundle_20260602_n1000/train_edge_semantic.npz`
- device: `cpu`
- max_molecules: 1000
- split_mode: `train_val`
- train_molecules: 800
- eval_molecules: 200
- epochs: 80
- batch_size: 16
- model_kind: `node_edge`
- focal_gamma: 2.0
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
| train_loss | 0.003497 |
| eval_loss | 0.016839 |
| accuracy | 0.994719 |
| balanced_accuracy | 0.901069 |
| macro_precision | 0.948093 |
| macro_f1 | 0.922254 |
| real_bond_precision | 0.937311 |
| real_bond_recall | 0.937689 |
| real_bond_f1 | 0.937500 |
| no_bond_recall | 0.999963 |
| precision_single | 0.957506 |
| recall_single | 0.936674 |
| f1_single | 0.946976 |
| precision_double | 0.855297 |
| recall_double | 0.753986 |
| f1_double | 0.801453 |
| precision_triple | 1.000000 |
| recall_triple | 0.833333 |
| f1_triple | 0.909091 |
| precision_aromatic | 0.927664 |
| recall_aromatic | 0.981386 |
| f1_aromatic | 0.953769 |

## Final Reconstruction Metrics

| metric | value |
|---|---:|
| recon_molecules | 100.000000 |
| recon_exact_halfedge_match_rate | 0.370000 |
| recon_bond_pair_set_match_rate | 0.990000 |
| recon_bond_type_map_match_rate | 0.370000 |
| recon_plain_sanitize_success_rate | 0.870000 |
| recon_semantic_sanitize_success_rate | 0.860000 |
| recon_plain_smiles_no_stereo_match_rate | 0.370000 |
| recon_semantic_smiles_no_stereo_match_rate | 0.470000 |
| recon_mean_bad_bond_length_fraction | 0.000204 |

## Best Balanced Accuracy Epoch

- epoch: 51
- balanced_accuracy: 0.951017
- real_bond_recall: 0.949385

## Files

- [edge_denoiser_overfit_metrics.csv](edge_denoiser_overfit_metrics.csv)
- [edge_denoiser_overfit_summary.json](edge_denoiser_overfit_summary.json)
- [edge_denoiser_overfit.pt](edge_denoiser_overfit.pt)

![curves](edge_denoiser_overfit_curves.png)
