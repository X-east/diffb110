# Edge Denoiser Evaluation Report

- checkpoint: `analysis/edge_denoiser_nodeedge_20260603_n1000_train800_val200_gpu/edge_denoiser_overfit.pt`
- device: `cuda`
- eval_split: `val`
- eval_molecules: 4
- timesteps: [0, 500]
- reconstruct_timestep: 500

## Classification Summary

| metric | value |
|---|---:|
| best_timestep | 0 |
| best_balanced_accuracy | 0.800000011920929 |
| best_real_bond_recall | 1.0 |
| worst_timestep | 500 |
| worst_balanced_accuracy | 0.7886792421340942 |
| worst_real_bond_recall | 0.95652174949646 |

![metrics_by_t](metrics_by_t.png)

![confusion_matrix_by_t](confusion_matrix_by_t.png)

## Reconstruction Summary

| metric | value |
|---|---:|
| molecules | 4 |
| exact_halfedge_match_rate | 0.5 |
| bond_pair_set_match_rate | 1.0 |
| bond_type_map_match_rate | 0.5 |
| mean_halfedge_accuracy | 0.9854826480150223 |
| mean_bond_pair_precision | 1.0 |
| mean_bond_pair_recall | 1.0 |
| plain_sanitize_success_rate | 0.5 |
| semantic_sanitize_success_rate | 0.5 |
| plain_smiles_no_stereo_match_rate | 0.5 |
| semantic_smiles_no_stereo_match_rate | 0.5 |
| mean_bad_bond_length_fraction | 0.0 |
| total_bad_bond_length_count | 0 |

## Files

- [metrics_by_t.csv](metrics_by_t.csv)
- [per_class_metrics_by_t.csv](per_class_metrics_by_t.csv)
- [confusion_matrix_by_t.csv](confusion_matrix_by_t.csv)
- [reconstruction_rows.csv](reconstruction_rows.csv)
- [edge_denoiser_eval_summary.json](edge_denoiser_eval_summary.json)
- [predicted_bond_plain_t500.sdf](predicted_bond_plain_t500.sdf)
- [predicted_bond_semantic_t500.sdf](predicted_bond_semantic_t500.sdf)
