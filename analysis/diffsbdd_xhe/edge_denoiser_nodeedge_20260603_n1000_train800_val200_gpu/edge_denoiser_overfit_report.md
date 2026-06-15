# Edge Denoiser Overfit Report

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

## Class Counts And Weights

| bond type | count | weight |
|---|---:|---:|
| no_bond | 222488 | 0.056081 |
| single | 10392 | 0.259477 |
| double | 1657 | 0.649646 |
| triple | 49 | 3.740969 |
| aromatic | 8104 | 0.293828 |

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
| train_loss | 0.001931 |
| eval_loss | 0.014252 |
| accuracy | 0.994855 |
| balanced_accuracy | 0.956287 |
| real_bond_recall | 0.938899 |
| no_bond_recall | 1.000000 |
| recall_single | 0.931624 |
| recall_double | 0.890661 |
| recall_triple | 1.000000 |
| recall_aromatic | 0.959152 |

## Best Balanced Accuracy Epoch

- epoch: 62
- balanced_accuracy: 0.960656
- real_bond_recall: 0.951805

## Files

- [edge_denoiser_overfit_metrics.csv](edge_denoiser_overfit_metrics.csv)
- [edge_denoiser_overfit_summary.json](edge_denoiser_overfit_summary.json)
- [edge_denoiser_overfit.pt](edge_denoiser_overfit.pt)

![curves](edge_denoiser_overfit_curves.png)
