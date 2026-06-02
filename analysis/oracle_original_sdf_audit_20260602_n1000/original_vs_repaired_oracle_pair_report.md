# Original vs Repaired Oracle Paired SDF

## Purpose

This export writes original ligand SDF records next to repaired oracle records built from processed coordinates and true halfedge labels. It is meant for visual inspection of reconstruction/repair differences.

## Inputs

- npz_path: `data/crossdocked_raw/processed_edge_noH_full/train.npz`
- raw_ligand_root: `data/crossdocked_raw/crossdocked_pocket10`
- dataset: `crossdock_full`
- samples: 1000

## Outputs

- [all paired SDF](original_vs_repaired_oracle_pairs.sdf): 2000 records
- [SMILES mismatch paired SDF](original_vs_repaired_oracle_smiles_mismatches.sdf): 260 records from 130 pairs
- [manifest CSV](original_vs_repaired_oracle_pair_manifest.csv)
- [summary JSON](original_vs_repaired_oracle_pair_summary.json)

## Record Layout

Each pair is written as two consecutive records: first `original`, then `repaired_oracle`. The SDF molecule title uses `sample_XXXXXX|variant|ligand_stem`.

Important properties on every record include:

- `pair_id`, `sample_idx`, `variant`, `counterpart_variant`
- `processed_name`, `original_sdf`, `repair_strategy`
- `this_smiles_*`, `counterpart_smiles_*`
- `comparison_smiles_no_stereo_match`, `comparison_bond_type_map_match`
- `semantic_warning`, `visual_priority`

## Repair Strategy Counts

| repair_strategy | count |
|---|---:|
| fix_aromatic | 3 |
| fix_valence | 127 |
| none | 870 |

## Interpretation Warning

The `repaired_oracle` records are visual comparison objects, not guaranteed faithful target molecules. Previous audit showed connectivity/atom counts are aligned, but repaired molecules can change aromatic/bond/charge semantics after RDKit sanitization. Prioritize the mismatch SDF for manual inspection.
