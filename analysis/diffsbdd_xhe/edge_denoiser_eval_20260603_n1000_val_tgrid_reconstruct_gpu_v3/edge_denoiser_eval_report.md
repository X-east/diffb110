# Edge Denoiser Evaluation Report

- checkpoint: `analysis/edge_denoiser_nodeedge_20260603_n1000_train800_val200_gpu/edge_denoiser_overfit.pt`
- device: `cuda`
- eval_split: `val`
- eval_molecules: 200
- timesteps: [0, 50, 100, 250, 500, 750, 950, 999]
- reconstruct_timestep: 500

## Classification Summary

| metric | value |
|---|---:|
| best_timestep | 0 |
| best_balanced_accuracy | 0.9968003034591675 |
| best_real_bond_recall | 0.9975801706314087 |
| worst_timestep | 999 |
| worst_balanced_accuracy | 0.9147507548332214 |
| worst_real_bond_recall | 0.8836458921432495 |

![metrics_by_t](metrics_by_t.png)

![confusion_matrix_by_t](confusion_matrix_by_t.png)

## Reconstruction Summary

| metric | value |
|---|---:|
| molecules | 200 |
| exact_halfedge_match_rate | 0.45 |
| bond_pair_set_match_rate | 0.99 |
| bond_type_map_match_rate | 0.45 |
| mean_halfedge_accuracy | 0.9924224808812141 |
| mean_bond_pair_precision | 1.0 |
| mean_bond_pair_recall | 0.9988076923076923 |
| plain_sanitize_success_rate | 0.81 |
| semantic_sanitize_success_rate | 0.8 |
| plain_smiles_no_stereo_match_rate | 0.475 |
| semantic_smiles_no_stereo_match_rate | 0.535 |
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
