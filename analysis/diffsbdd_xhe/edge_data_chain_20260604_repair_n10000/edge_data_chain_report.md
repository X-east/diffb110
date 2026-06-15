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
- molecules: 10000
- total_atoms: 236294
- total_halfedges: 3056103
- total_real_bonds: 253589
- no_bond_fraction: 0.91702210

## Bond Type Counts

| bond type | count |
|---|---:|
| no_bond | 2802514 |
| single | 132947 |
| double | 21075 |
| triple | 519 |
| aromatic | 99048 |

![bond type distribution](bond_type_distribution.png)

![bond length distribution](bond_length_distribution.png)
