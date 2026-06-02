# Edge Denoiser Overfit Report

- edge_npz: `analysis/edge_data_chain_20260602_n1000/train_edge.npz`
- semantic_npz: `analysis/edge_semantic_bundle_20260602_n1000/train_edge_semantic.npz`
- device: `cuda`
- max_molecules: 128
- epochs: 120
- batch_size: 16
- focal_gamma: 1.0

## Class Counts And Weights

| bond type | count | weight |
|---|---:|---:|
| no_bond | 34810 | 0.056746 |
| single | 1691 | 0.257391 |
| double | 273 | 0.639615 |
| triple | 7 | 3.743252 |
| aromatic | 1220 | 0.302996 |

## Final Metrics

| metric | value |
|---|---:|
| train_loss | 0.008982 |
| eval_loss | 0.048960 |
| accuracy | 0.979395 |
| balanced_accuracy | 0.837083 |
| real_bond_recall | 0.758383 |
| no_bond_recall | 0.999655 |
| recall_single | 0.662330 |
| recall_double | 0.780220 |
| recall_triple | 0.857143 |
| recall_aromatic | 0.886066 |

## Best Balanced Accuracy Epoch

- epoch: 102
- balanced_accuracy: 0.843396
- real_bond_recall: 0.768098

## Files

- [edge_denoiser_overfit_metrics.csv](edge_denoiser_overfit_metrics.csv)
- [edge_denoiser_overfit_summary.json](edge_denoiser_overfit_summary.json)
- [edge_denoiser_overfit.pt](edge_denoiser_overfit.pt)

![curves](edge_denoiser_overfit_curves.png)
