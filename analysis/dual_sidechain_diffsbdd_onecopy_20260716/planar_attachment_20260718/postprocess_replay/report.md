# DiffSBDD Dual-Sidechain Generation (generate-one-copy-other)

- Input inpaint SDF: `/data/disk6/DiffB110/analysis/dual_sidechain_diffsbdd_onecopy_20260716/inpaint_raw.sdf`
- Anchor1 (R1) core idx: `0`
- Anchor2 (R2) core idx: `11`
- add_n_nodes: `8`  n_samples: `20`
- clash_fraction: `0.7`

## Summary

- input molecules: 12
- symmetric success: 2
- clash (rejected): 0
- other (no extractable sidechain): 10

## Per-molecule

| idx | source | status | R1 atoms | R2 atoms | clash | min_dist | bond A | angles deg | oop deg | SMILES |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0 |  | r1_missing | 0 | 0 | 0 | inf | nan | nan/nan | nan |  |
| 1 |  | r1_missing | 0 | 0 | 0 | inf | nan | nan/nan | nan |  |
| 2 | R2 | disconnected | 1 | 1 | 0 | inf | nan | nan/nan | nan | C.C.C=CCc1ccc(F)c(-c2ccc(F)c(CC3=C=C3C)c2)c1O.O.O |
| 3 |  | r1_missing | 0 | 0 | 0 | inf | nan | nan/nan | nan |  |
| 4 | R1 | disconnected | 1 | 1 | 0 | inf | nan | nan/nan | nan | C=CCc1cc(F)c(Cl)c(-c2ccc(Cl)c(CC=C(O)O)c2)c1.N.O |
| 5 | R2 | disconnected | 1 | 1 | 0 | inf | nan | nan/nan | nan | C=CCc1cc(-c2cc([C@]3(C=N)C4=C[C@H]43)ccc2N)ccc1N.C[C@H](N)O |
| 6 |  | r1_missing | 0 | 0 | 0 | inf | nan | nan/nan | nan |  |
| 7 | R2 | disconnected | 1 | 1 | 0 | inf | nan | nan/nan | nan | F.F.F.NCC=CCc1ccc(F)c(-c2cc(CC=CF)c(F)cc2F)c1 |
| 8 |  | r1_missing | 0 | 0 | 0 | inf | nan | nan/nan | nan |  |
| 9 | R1 | success | 2 | 2 | 0 | 2.9655 | 1.554853 | 119.30283/119.30283 | 0.0 | C=CCc1c(CC)cc(CN)c(-c2cc(O)c(CN)c(CC=C(O)O)c2)c1O |
| 10 | R2 | invalid_indices | 0 | 0 | 0 | inf | nan | nan/nan | nan |  |
| 11 | R2 | success | 1 | 1 | 0 | 2.9085 | 1.3733 | 119.545763/119.545763 | 0.0 | C=C=C(CO)Cc1cc(-c2c(O)cc(N)c([C@H](O)C=C)c2CC)ccc1O |
