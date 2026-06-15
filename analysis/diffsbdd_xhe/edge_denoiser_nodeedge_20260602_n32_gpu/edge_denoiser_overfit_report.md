# Edge Denoiser Overfit Report

- edge_npz: `analysis/edge_data_chain_20260602_n1000/train_edge.npz`
- semantic_npz: `analysis/edge_semantic_bundle_20260602_n1000/train_edge_semantic.npz`
- device: `cuda`
- max_molecules: 32
- epochs: 80
- batch_size: 8
- model_kind: `node_edge`
- focal_gamma: 1.0

## Class Counts And Weights

| bond type | count | weight |
|---|---:|---:|
| no_bond | 9717 | 0.069043 |
| single | 478 | 0.310986 |
| double | 66 | 0.831516 |
| triple | 3 | 3.403127 |
| aromatic | 311 | 0.385328 |

## Final Metrics

| metric | value |
|---|---:|
| train_loss | 0.004457 |
| eval_loss | 0.025489 |
| accuracy | 0.992624 |
| balanced_accuracy | 0.922260 |
| real_bond_recall | 0.909091 |
| no_bond_recall | 1.000000 |
| recall_single | 0.945607 |
| recall_double | 0.787879 |
| recall_triple | 1.000000 |
| recall_aromatic | 0.877814 |

## Best Balanced Accuracy Epoch

- epoch: 71
- balanced_accuracy: 0.955569
- real_bond_recall: 0.934732

## Files

- [edge_denoiser_overfit_metrics.csv](edge_denoiser_overfit_metrics.csv)
- [edge_denoiser_overfit_summary.json](edge_denoiser_overfit_summary.json)
- [edge_denoiser_overfit.pt](edge_denoiser_overfit.pt)

![curves](edge_denoiser_overfit_curves.png)
