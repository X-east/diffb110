# Edge Transition Smoke Report

- edge_npz: `analysis/edge_data_chain_20260602_n1000/train_edge.npz`
- analyzed_molecules: 256
- total_halfedges: 80322
- total_real_bonds: 6577
- no_bond_fraction: 0.91811711
- one_hot_shape: `[80322, 5]`
- nll_shape: `[80322]`
- uniform_logits_nll_mean: 1.609438

| timestep | noise probability | changed fraction | label range |
|---:|---:|---:|---|
| 0 | 0.001000 | 0.000722 | 0..4 |
| 500 | 0.501000 | 0.399567 | 0..4 |
| 999 | 1.000000 | 0.798137 | 0..4 |
