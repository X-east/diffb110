# Edge Transition Smoke Report

- edge_npz: `analysis/edge_data_chain_20260604_repair_n10000/train_edge.npz`
- analyzed_molecules: 512
- total_halfedges: 156850
- total_real_bonds: 13008
- no_bond_fraction: 0.91706729
- one_hot_shape: `[156850, 5]`
- nll_shape: `[156850]`
- uniform_logits_nll_mean: 1.609438

| timestep | noise probability | changed fraction | label range |
|---:|---:|---:|---|
| 0 | 0.000100 | 0.000134 | 0..4 |
| 500 | 0.922203 | 0.739802 | 0..4 |
| 999 | 0.999960 | 0.799515 | 0..4 |
