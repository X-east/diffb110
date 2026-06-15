# Edge Denoiser Evaluation Report

- checkpoint: `analysis/edge_denoiser_ablation_20260603_D_diffgui_noise010/edge_denoiser_overfit.pt`
- device: `cuda`
- eval_split: `val`
- eval_molecules: 200
- timesteps: [0, 50, 100, 250, 500, 750, 950, 999]
- reconstruct_timestep: 500

## Classification Summary

| metric | value |
|---|---:|
| best_timestep | 0 |
| best_balanced_accuracy | 0.9568319320678711 |
| best_real_bond_recall | 0.9879007935523987 |
| worst_timestep | 999 |
| worst_balanced_accuracy | 0.8377879858016968 |
| worst_real_bond_recall | 0.8868723511695862 |

![metrics_by_t](metrics_by_t.png)

![confusion_matrix_by_t](confusion_matrix_by_t.png)

## Reconstruction Summary

| metric | value |
|---|---:|
| molecules | 200 |
| exact_halfedge_match_rate | 0.36 |
| bond_pair_set_match_rate | 1.0 |
| bond_type_map_match_rate | 0.36 |
| mean_halfedge_accuracy | 0.9918768912553787 |
| mean_bond_pair_precision | 1.0 |
| mean_bond_pair_recall | 1.0 |
| plain_sanitize_success_rate | 0.975 |
| semantic_sanitize_success_rate | 0.97 |
| plain_smiles_no_stereo_match_rate | 0.375 |
| semantic_smiles_no_stereo_match_rate | 0.45 |
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
