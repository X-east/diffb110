# Edge Denoiser Train/Val Report

- edge_npz: `analysis/edge_data_chain_20260602_n1000/train_edge.npz`
- semantic_npz: `analysis/edge_semantic_bundle_20260602_n1000/train_edge_semantic.npz`
- device: `cuda`
- max_molecules: 128
- split_mode: `train_val`
- train_molecules: 96
- eval_molecules: 32
- epochs: 6
- batch_size: 16
- model_kind: `node_edge`
- focal_gamma: 1.0
- class_weight_mode: `diffgui`
- no_bond_weight: 0.1
- pos_noise_std: 0.05
- reconstruction_validation: `True`
- reconstruction_timestep: 500
- reconstruction_limit: 16

## Class Counts And Weights

| bond type | count | weight |
|---|---:|---:|
| no_bond | 26337 | 0.100000 |
| single | 1254 | 1.000000 |
| double | 208 | 1.000000 |
| triple | 7 | 1.000000 |
| aromatic | 960 | 1.000000 |

## Eval Class Counts

| bond type | count |
|---|---:|
| no_bond | 8473 |
| single | 437 |
| double | 65 |
| triple | 0 |
| aromatic | 260 |

## Final Metrics

| metric | value |
|---|---:|
| train_loss | 0.028014 |
| eval_loss | 0.041988 |
| accuracy | 0.984840 |
| balanced_accuracy | 0.647263 |
| macro_precision | 0.653327 |
| macro_f1 | 0.648956 |
| real_bond_precision | 0.817346 |
| real_bond_recall | 0.816273 |
| real_bond_f1 | 0.816809 |
| no_bond_recall | 1.000000 |
| precision_single | 0.827660 |
| recall_single | 0.890160 |
| f1_single | 0.857773 |
| precision_double | 0.565217 |
| recall_double | 0.600000 |
| f1_double | 0.582090 |
| precision_triple | 0.000000 |
| recall_triple | 0.000000 |
| f1_triple | 0.000000 |
| precision_aromatic | 0.873874 |
| recall_aromatic | 0.746154 |
| f1_aromatic | 0.804979 |

## Final Reconstruction Metrics

| metric | value |
|---|---:|
| recon_molecules | 16.000000 |
| recon_exact_halfedge_match_rate | 0.000000 |
| recon_bond_pair_set_match_rate | 1.000000 |
| recon_bond_type_map_match_rate | 0.000000 |
| recon_plain_sanitize_success_rate | 0.562500 |
| recon_semantic_sanitize_success_rate | 0.562500 |
| recon_plain_smiles_no_stereo_match_rate | 0.000000 |
| recon_semantic_smiles_no_stereo_match_rate | 0.000000 |
| recon_mean_bad_bond_length_fraction | 0.000000 |

## Best Balanced Accuracy Epoch

- epoch: 6
- balanced_accuracy: 0.647263
- real_bond_recall: 0.816273

## Files

- [edge_denoiser_overfit_metrics.csv](edge_denoiser_overfit_metrics.csv)
- [edge_denoiser_overfit_summary.json](edge_denoiser_overfit_summary.json)
- [edge_denoiser_overfit.pt](edge_denoiser_overfit.pt)

![curves](edge_denoiser_overfit_curves.png)
