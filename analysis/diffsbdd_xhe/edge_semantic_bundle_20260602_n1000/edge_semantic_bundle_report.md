# Edge Semantic Bundle Report

- semantic_npz: `train_edge_semantic.npz`
- source_npz: `data/crossdocked_raw/processed_edge_noH_full/train.npz`
- raw_ligand_root: `data/crossdocked_raw/crossdocked_pocket10`
- dataset: `crossdock_full`
- samples: 1000

## Original SDF Load Status

| status | count |
|---|---:|
| success | 1000 |

## Semantic Match Rates

| metric | count | denominator | rate |
|---|---:|---:|---:|
| atom_count_match | 1000 | 1000 | 1.00000000 |
| atom_symbol_match | 1000 | 1000 | 1.00000000 |
| bond_count_match | 1000 | 1000 | 1.00000000 |
| bond_pair_set_match | 1000 | 1000 | 1.00000000 |
| bond_type_map_match | 1000 | 1000 | 1.00000000 |

## Total Formal Charge Counts

| total charge | count |
|---|---:|
| 0 | 1000 |

## Bond Type Counts

| bond type | processed | original SDF |
|---|---:|---:|
| aromatic | 10038 | 10038 |
| double | 2096 | 2096 |
| single | 12966 | 12966 |
| triple | 61 | 61 |

## Files

- [train_edge_semantic.npz](train_edge_semantic.npz)
- [edge_semantic_molecule_summary.csv](edge_semantic_molecule_summary.csv)
- [edge_semantic_manifest.json](edge_semantic_manifest.json)

![semantic match rates](edge_semantic_match_rates.png)

![total charge distribution](edge_semantic_total_charge_distribution.png)
