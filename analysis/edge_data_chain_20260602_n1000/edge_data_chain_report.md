# Edge Data Chain Report

## Outputs

- edge bundle: [train_edge.npz](train_edge.npz)
- oracle SDF: [oracle_reconstruct.sdf](oracle_reconstruct.sdf)
- oracle report: [edge_data_oracle_report.md](edge_data_oracle_report.md)
- transition smoke: [edge_transition_smoke_report.md](edge_transition_smoke_report.md)
- manifest: [edge_data_manifest.json](edge_data_manifest.json)

## Summary

- source_npz: `data/crossdocked_raw/processed_edge_noH_full/train.npz`
- dataset: `crossdock_full`
- molecules: 1000
- total_atoms: 23456
- total_halfedges: 301578
- total_real_bonds: 25161
- no_bond_fraction: 0.91656885

## Bond Type Counts

| bond type | count |
|---|---:|
| no_bond | 276417 |
| single | 12966 |
| double | 2096 |
| triple | 61 |
| aromatic | 10038 |

![bond type distribution](bond_type_distribution.png)

![bond length distribution](bond_length_distribution.png)
