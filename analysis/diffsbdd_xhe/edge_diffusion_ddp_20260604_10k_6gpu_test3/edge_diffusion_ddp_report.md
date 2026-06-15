# Edge Diffusion DDP Training Report

## Scope

Standalone x/h/e edge diffusion supervised training. This path is intentionally decoupled from DDPO/RL and does not import `ddpo`.

## Configuration

- edge_npz: `analysis/edge_data_chain_20260604_repair_n10000/train_edge.npz`
- semantic_npz: `analysis/edge_semantic_bundle_20260604_n10000/train_edge_semantic.npz`
- device: `cuda:0`
- distributed: `True`
- world_size: 6
- max_molecules: 10000
- train_molecules: 9000
- eval_molecules: 1000
- epochs: 3
- batch_size_per_rank: 8
- global_batch_size: 48
- elapsed_seconds: 39.87
- seconds_per_epoch: 13.2906

## Final Metrics

| metric | value |
|---|---:|
| train_total | 0.180536 |
| train_pos | 0.000781 |
| train_atom | 0.051167 |
| train_edge | 0.128240 |
| train_bond_length | 0.000696 |
| eval_total | 0.193061 |
| eval_pos | 0.000588 |
| eval_atom | 0.044174 |
| eval_edge | 0.148032 |
| eval_bond_length | 0.000533 |
| eval_edge_accuracy | 0.987720 |
| eval_edge_balanced_accuracy | 0.853295 |
| eval_edge_real_bond_precision | 0.851561 |
| eval_edge_real_bond_recall | 0.851694 |
| eval_edge_real_bond_f1 | 0.851627 |
| eval_atom_accuracy | 0.989652 |
| eval_pos_rmse | 0.024246 |
| eval_bond_length_mae | 0.026026 |

## Best Real-Bond F1 Epoch

- epoch: 3
- eval_edge_real_bond_f1: 0.851627
- eval_edge_real_bond_recall: 0.851694

## Files

- [edge_diffusion_ddp_metrics.csv](edge_diffusion_ddp_metrics.csv)
- [edge_diffusion_ddp_summary.json](edge_diffusion_ddp_summary.json)
- [edge_diffusion_ddp_latest.pt](edge_diffusion_ddp_latest.pt)
- [edge_diffusion_ddp_curves.png](edge_diffusion_ddp_curves.png)

![curves](edge_diffusion_ddp_curves.png)
