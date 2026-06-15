# Edge Transition Smoke Report

- edge_npz: `analysis/edge_data_chain_20260602_repair_n1000/train_edge.npz`
- analyzed_molecules: 128
- total_halfedges: 38001
- total_real_bonds: 3191
- no_bond_fraction: 0.91602850
- one_hot_shape: `[38001, 5]`
- nll_shape: `[38001]`
- uniform_logits_nll_mean: 1.609438

| timestep | noise probability | changed fraction | label range |
|---:|---:|---:|---|
| 0 | 0.000100 | 0.000079 | 0..4 |
| 500 | 0.922203 | 0.739481 | 0..4 |
| 999 | 0.999960 | 0.794611 | 0..4 |
