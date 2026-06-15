# Generated Reconstruction Benchmark Report

- checkpoint: `DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt`
- pdbfile: `data/8GJL.pdb`
- ref_ligand: `C:1101`
- resi_list: `None`
- samples: `1000`
- timesteps: `100`
- length_tolerance: `0.25`
- methods: `openbabel edm`
- local_bond_checkpoint: `analysis/local_bond_predictor_v0_20260602_small/best_model.pt`
- local_bond_device: `cpu`
- edge_denoiser_b_checkpoint: `analysis/edge_denoiser_ablation_20260603_B_diffgui_noise005/edge_denoiser_overfit.pt`
- edge_denoiser_c_checkpoint: `analysis/edge_denoiser_ablation_20260603_C_diffgui_noise005_focal2_gpu/edge_denoiser_overfit.pt`
- edge_denoiser_device: `cpu`
- edge_denoiser_timestep: `999`
- edge_denoiser_init_mode: `uniform_random`
- edge_denoiser_seed: `20260603`
- atom_bias: `{}`
- atom_temperature: `1.0`
- atom_local_mask: `{'enabled': False, 'candidate_cutoff': 2.2, 'terminal_degree_max': 1, 'terminal_only_elements': ('F', 'Cl', 'Br', 'I'), 'disable_elements': ('P', 'Br', 'I', 'B')}`
- geometry_refine: `{'enabled': False, 'steps': 10, 'learning_rate': 0.05, 'max_step': 0.05, 'max_total_shift': 0.3, 'use_pocket_clash': True}`

| method | sanitize | bad_bond_length | fragments | largest_frag | aromatic | benzene6 | bad_arom | QED | SA | TPSA | sec/mol |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| openbabel | 0.658 | 0.046 | 1.104 | 0.980 | 0.559 | 0.379 | 0.272 | 0.407 | 0.419 | 41.089 | 0.0018 |
| edm | 0.482 | 0.004 | 2.937 | 0.787 | 0.013 | 0.003 | 0.006 | 0.229 | 0.276 | 35.245 | 0.0017 |

![sanitize success](sanitize_success_by_method.png)

![bad bond length](bad_bond_length_by_method.png)

![fragment count](fragment_count_by_method.png)

![largest fragment](largest_fragment_by_method.png)

![aromatic rings](aromatic_ring_by_method.png)

![benzene-like rings](benzene_like_ring_by_method.png)

![abnormal aromatic rings](abnormal_aromatic_ring_by_method.png)

![element fractions](element_fraction_by_method.png)

![QED](qed_by_method.png)

![SA](sa_by_method.png)

![TPSA](tpsa_by_method.png)

![time](time_by_method.png)
