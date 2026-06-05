# Requirements Document

## Introduction

本项目旨在将 DiffSBDD（基于口袋条件的 E(n)-等变扩散模型）与 RLPF 框架（verl_diffusion）集成，使用 DDPO（Denoising Diffusion Policy Optimization）算法优化针对 RE-CmeB 蛋白（PDB: 8GJJ）的分子生成。

核心目标是通过强化学习优化 DiffSBDD 生成的分子，使其具有更高的对接亲和力（docking affinity）和更好的药物样性质（drug-likeness）。

## Glossary

- **DiffSBDD**: Structure-based Drug Design with Equivariant Diffusion Models，一个基于 E(n)-等变扩散模型的结构药物设计框架
- **RLPF**: Reinforcement Learning from Preference Feedback，用于扩散模型的强化学习框架（verl_diffusion）
- **DDPO**: Denoising Diffusion Policy Optimization，一种将扩散模型的去噪过程建模为 MDP 的策略优化算法
- **RE-CmeB**: 目标蛋白（PDB ID: 8GJJ），用于分子对接评估的蛋白质靶点
- **Pocket**: 蛋白质结合口袋，分子生成的条件输入，包含坐标和残基类型信息
- **Ligand**: 配体分子，扩散模型生成的目标3D分子结构
- **DataProto**: RLPF 框架中的标准数据传输协议，包含 TensorDict batch 和 non_tensor_batch
- **BaseModel**: RLPF 框架中模型的抽象基类接口，定义 sample()、sample_p_zs_given_zt()、compute_log_p_zs_given_zt() 方法
- **BaseReward**: RLPF 框架中奖励函数的抽象基类接口，定义 calculate_rewards() 方法
- **Vina Score**: AutoDock Vina 计算的对接分数，单位为 kcal/mol，负值越小表示结合亲和力越强
- **QED**: Quantitative Estimate of Drug-likeness，药物样性定量估计，范围 0-1，越高越好
- **SA Score**: Synthetic Accessibility Score，合成可及性分数，范围 1-10，越低越容易合成
- **SE(3) Equivariance**: 特殊欧几里得群等变性，保证模型对旋转和平移的等变性
- **batch_mask**: DiffSBDD 使用的节点到批次索引映射格式，形状为 [total_nodes]，值为批次索引
- **node_mask**: RLPF 使用的节点掩码格式，形状为 [B, N, 1]，值为 0 或 1
- **log_prob**: 对数概率，DDPO 算法计算重要性采样比率所需的关键量
- **EnVariationalDiffusion**: DiffSBDD 的核心扩散模型类，实现 E(n)-等变扩散过程
- **ConditionalDDPM**: DiffSBDD 的条件扩散模型，支持口袋条件生成
- **DDPOTrainer**: RLPF 框架的 DDPO 训练器，协调 rollout、reward、actor 组件
- **gamma**: 噪声调度函数，控制扩散过程中的信噪比
- **alpha_t**: 扩散过程中的信号保留系数
- **sigma_t**: 扩散过程中的噪声标准差

## Requirements

### Requirement 1: DiffSBDD 模型适配器

**User Story:** As a researcher, I want to use DiffSBDD model within the RLPF framework, so that I can apply DDPO training to optimize molecular generation.

#### Acceptance Criteria

1. WHEN the DiffSBDDModel adapter is initialized with a pretrained checkpoint path and pocket PDB file path THEN the DiffSBDDModel adapter SHALL load the EnVariationalDiffusion model weights and internally load and store the pocket coordinates, one_hot encoding, and mask as fixed constant conditions that are invisible to the RL framework
2. WHEN the sample() method is called with n_samples parameter THEN the DiffSBDDModel adapter SHALL internally replicate the fixed pocket n_samples times, generate molecules by iterating through T timesteps conditioned on the pocket, and return a DataProto containing latents list, log_probs list, timesteps list, means list, sigmas list, final coordinates x, and atom types h (pocket data is not exposed in the return value)
3. WHEN the sample_p_zs_given_zt() method is called with timestep s, timestep t, current latent zt, and masks THEN the DiffSBDDModel adapter SHALL internally combine zt with the fixed pocket data, call the underlying EnVariationalDiffusion.sample_p_zs_given_zt() method, and return only the ligand-related log probability, mean mu, and standard deviation sigma
4. WHEN the compute_log_p_zs_given_zt() method is called with sample zs, mean mu, and sigma THEN the DiffSBDDModel adapter SHALL compute the Gaussian log probability using the formula: log_prob = -0.5 * sum((zs - mu)^2 / sigma^2) - sum(log(sigma)) - 0.5 * D * log(2π) where D is the dimensionality
5. WHEN generating molecules THEN the DiffSBDDModel adapter SHALL call DiffSBDD's remove_mean_batch() method for coordinate centering to preserve SE(3) equivariance
6. WHEN the sample() method completes THEN the DiffSBDDModel adapter SHALL return coordinates and atom types that can be converted to valid RDKit molecules using DiffSBDD's build_molecule() function
7. WHEN the RL framework (DDPOTrainer, EDMRollout, EDMActor) interacts with DiffSBDDModel THEN the interface SHALL be identical to EDMModel, with pocket handling completely encapsulated inside DiffSBDDModel

### Requirement 2: Mask 格式转换工具

**User Story:** As a developer, I want to convert between DiffSBDD and RLPF mask formats, so that data can flow correctly between the two frameworks.

#### Acceptance Criteria

1. WHEN batch_mask_to_node_mask() is called with a DiffSBDD batch_mask tensor of shape [total_nodes] THEN the mask converter SHALL return an RLPF node_mask tensor of shape [B, N, 1] where B is the number of unique batch indices and N is the maximum node count across batches
2. WHEN node_mask_to_batch_mask() is called with an RLPF node_mask tensor of shape [B, N, 1] THEN the mask converter SHALL return a DiffSBDD batch_mask tensor of shape [total_valid_nodes] containing batch indices for each valid node
3. WHEN converting from batch_mask to node_mask and back to batch_mask THEN the mask converter SHALL produce an output that matches the original batch_mask for all valid nodes
4. IF the input batch_mask contains negative values or values exceeding the batch size THEN the mask converter SHALL raise a ValueError with a message containing the invalid value and expected range
5. IF the input node_mask contains values other than 0 or 1 THEN the mask converter SHALL raise a ValueError with a message indicating invalid mask values

### Requirement 3: 口袋数据加载工具

**User Story:** As a developer, I want to load and preprocess the RE-CmeB protein pocket data, so that it can be used as a fixed condition inside DiffSBDDModel.

#### Acceptance Criteria

1. WHEN PocketLoader is initialized with a PDB file path THEN the PocketLoader SHALL verify the file exists and store the path for subsequent loading
2. WHEN load_pocket() is called THEN the PocketLoader SHALL parse the PDB file using BioPython PDBParser and extract CA atom coordinates and residue types for each residue
3. WHEN the pocket is loaded THEN the PocketLoader SHALL center the coordinates by subtracting the centroid and encode residue types using DiffSBDD's aa_encoder dictionary
4. WHEN the pocket is loaded THEN the PocketLoader SHALL return a dictionary containing 'x' (coordinates tensor), 'one_hot' (residue type encoding), 'mask' (batch_mask format), and 'size' (number of residues)
5. IF the PDB file path does not exist THEN the PocketLoader SHALL raise a FileNotFoundError with a message containing the attempted file path
6. IF the PDB file cannot be parsed by BioPython THEN the PocketLoader SHALL raise a ValueError with a message containing the parsing error details
7. WHEN PocketLoader is used THEN it SHALL be called only by DiffSBDDModel during initialization, not by the RL framework components (DDPOTrainer, Rollout, Actor)

### Requirement 4: Docking 奖励函数

**User Story:** As a researcher, I want to evaluate generated molecules using AutoDock Vina docking scores, so that I can optimize for binding affinity to RE-CmeB.

#### Acceptance Criteria

1. WHEN DockingReward is initialized with receptor_path, center coordinates, and box_size THEN the DockingReward SHALL internally load the receptor PDBQT file and store the docking box parameters (pocket data is encapsulated inside DockingReward, invisible to RL framework)
2. WHEN calculate_rewards() is called with a DataProto containing 'x' coordinates and 'h' atom types THEN the DockingReward SHALL convert each molecule to PDBQT format using OpenBabel and compute Vina docking scores against the internally stored receptor
3. WHEN computing docking scores for a batch of N molecules with num_workers Ray workers THEN the DockingReward SHALL complete in time less than (N / num_workers + 1) * single_molecule_time seconds
4. IF a molecule fails PDBQT conversion THEN the DockingReward SHALL assign a penalty score of 0.0 and log a warning message containing the molecule index
5. IF Vina docking calculation fails for a molecule THEN the DockingReward SHALL assign a penalty score of 0.0 and continue processing remaining molecules
6. WHEN returning rewards THEN the DockingReward SHALL return a DataProto containing 'docking_scores' tensor with values normalized by dividing by -10.0 to produce positive rewards for good binders
7. WHEN DockingReward is used by DDPOTrainer THEN the interface SHALL be identical to ForceReward, with receptor/pocket handling completely encapsulated inside DockingReward

### Requirement 5: Drug-likeness 奖励函数

**User Story:** As a researcher, I want to evaluate generated molecules for drug-like properties, so that I can ensure generated molecules are pharmaceutically relevant.

#### Acceptance Criteria

1. WHEN DruglikenessReward is initialized THEN the DruglikenessReward SHALL import RDKit QED and SA score calculation modules
2. WHEN calculate_rewards() is called with a DataProto containing molecular coordinates and atom types THEN the DruglikenessReward SHALL convert molecules to RDKit Mol objects and compute QED scores in range [0, 1] and SA scores in range [1, 10]
3. IF a molecule cannot be converted to a valid RDKit Mol object THEN the DruglikenessReward SHALL assign QED score of 0.0 and SA score of 10.0 for that molecule
4. WHEN returning rewards THEN the DruglikenessReward SHALL return a DataProto containing 'qed_scores' tensor and 'sa_scores' tensor, with SA scores normalized as (10 - sa) / 9 to produce values in [0, 1]

### Requirement 6: 组合奖励函数

**User Story:** As a researcher, I want to combine multiple reward signals with configurable weights, so that I can balance different optimization objectives.

#### Acceptance Criteria

1. WHEN CompositeReward is initialized with a dictionary mapping reward names to (reward_function, weight) tuples THEN the CompositeReward SHALL store all reward functions and weights
2. WHEN calculate_rewards() is called THEN the CompositeReward SHALL call each component reward function and compute a weighted sum: total_reward = sum(weight_i * reward_i)
3. WHEN update_weights() is called with a new weights dictionary THEN the CompositeReward SHALL update the stored weights and apply the new weights to subsequent calculate_rewards() calls
4. IF any component reward function raises an exception THEN the CompositeReward SHALL log the error with the reward name and exception message, use a fallback value of 0.0 for that component, and continue computing other rewards

### Requirement 7: 训练配置管理

**User Story:** As a researcher, I want to configure all training parameters through a YAML file, so that I can easily reproduce and modify experiments.

#### Acceptance Criteria

1. WHEN the training script is launched with a config_path argument THEN the configuration loader SHALL load and parse the YAML file at that path
2. WHEN configuration is loaded THEN the configuration loader SHALL validate that required parameters (model.checkpoint, pocket.pdb_path, train.batch_size, train.learning_rate) are present
3. IF a required configuration parameter is missing THEN the configuration loader SHALL raise a ValueError with a message listing all missing required parameters
4. WHEN training starts THEN the training script SHALL log the complete configuration dictionary to wandb using wandb.config.update()

### Requirement 8: 端到端训练流程

**User Story:** As a researcher, I want to run DDPO training on DiffSBDD with docking-based rewards, so that I can optimize molecular generation for RE-CmeB binding.

#### Acceptance Criteria

1. WHEN training is started THEN the training script SHALL initialize DiffSBDDModel with the checkpoint path, PocketLoader with the PDB path, CompositeReward with configured weights, and DDPOTrainer with training parameters
2. WHEN each training iteration is processed THEN the DDPOTrainer SHALL generate a batch of molecules, compute rewards using CompositeReward, calculate advantages using GAE, and update the policy using PPO loss
3. WHEN training progresses THEN the training script SHALL log metrics to wandb including: mean_reward, mean_docking_score, mean_qed, mean_sa, validity_ratio, policy_loss, value_loss at each logging interval
4. WHEN the training step count reaches a checkpoint_interval multiple THEN the training script SHALL save model weights and optimizer state to a checkpoint file with the step number in the filename
5. WHEN training completes or receives a SIGINT signal THEN the training script SHALL save a final checkpoint, call ray.shutdown(), and log a completion message

### Requirement 9: 分子有效性过滤

**User Story:** As a researcher, I want to filter out invalid molecules during training, so that only valid molecules contribute to the reward signal.

#### Acceptance Criteria

1. WHEN molecules are generated THEN the validity filter SHALL attempt to build RDKit Mol objects using DiffSBDD's build_molecule() function and check validity using Chem.SanitizeMol()
2. WHEN a molecule fails RDKit sanitization THEN the validity filter SHALL mark that molecule as invalid and assign a penalty reward of -1.0
3. WHEN filtering is applied to a batch THEN the validity filter SHALL compute validity_ratio = num_valid / total and log this metric
4. IF all molecules in a batch are invalid THEN the validity filter SHALL skip the policy update for that batch and log a warning message containing the batch index

### Requirement 10: SE(3) 等变性保持

**User Story:** As a researcher, I want to ensure the RL training preserves DiffSBDD's SE(3) equivariance, so that the model maintains its geometric properties.

#### Acceptance Criteria

1. WHEN coordinates are processed during sampling THEN the DiffSBDDModel adapter SHALL call EnVariationalDiffusion.remove_mean_batch() to center coordinates after each denoising step
2. WHEN policy gradients are computed THEN the DDPOTrainer SHALL update only the parameters of the dynamics network without modifying the equivariant layer structure
3. WHEN rewards are computed for a molecule with coordinates x THEN the reward function SHALL produce the same reward value for any rotated version R @ x where R is a valid rotation matrix

### Requirement 11: Log Probability 计算与轨迹记录

**User Story:** As a developer, I want to compute and record log probabilities throughout the diffusion trajectory, so that DDPO can calculate importance sampling ratios correctly.

#### Acceptance Criteria

1. WHEN a denoising step from timestep t to s is performed THEN the DiffSBDDModel adapter SHALL compute mu using the formula: mu = zt / alpha_t_given_s - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_t
2. WHEN a denoising step is performed THEN the DiffSBDDModel adapter SHALL compute sigma using the formula: sigma = sigma_t_given_s * sigma_s / sigma_t
3. WHEN the full sampling trajectory is recorded THEN the DiffSBDDModel adapter SHALL store for each timestep: the latent zt, the computed mu, the computed sigma, and the log_prob of the transition
4. WHEN sample() returns THEN the returned DataProto SHALL contain 'log_probs' key with a tensor of shape [B, T] containing the sum of log probabilities for each sample across all timesteps

### Requirement 12: 分子格式转换

**User Story:** As a developer, I want to convert generated molecular representations to standard chemical formats, so that molecules can be evaluated by external tools.

#### Acceptance Criteria

1. WHEN coordinates x and atom types h are provided THEN the molecule converter SHALL use DiffSBDD's build_molecule() function to create an RDKit Mol object
2. WHEN an RDKit Mol object is created THEN the molecule converter SHALL call process_molecule() to add hydrogens and optimize geometry
3. WHEN PDBQT format is required for docking THEN the molecule converter SHALL use OpenBabel to convert from RDKit Mol to PDBQT string format
4. IF build_molecule() returns None THEN the molecule converter SHALL return None and not attempt further processing

### Requirement 13: 组件复用策略

**User Story:** As a developer, I want to maximize reuse of RLPF components while replacing only necessary parts, so that I can minimize integration complexity.

#### Acceptance Criteria

1. WHEN EDMActor is used for policy updates THEN the EDMActor code SHALL remain identical to the original RLPF implementation without any modifications (直接复用)
2. WHEN DDPOTrainer coordinates training THEN the DDPOTrainer code SHALL be modified only as needed to accommodate DiffSBDDModel and custom rewards (按需修改)
3. WHEN main_edm.py is used as entry point THEN the script SHALL be modified to initialize DiffSBDDModel instead of EDMModel and custom rewards instead of ForceReward (按需修改)
4. WHEN EDMRollout generates trajectories THEN the EDMRollout code MAY be modified as needed to work with DiffSBDDModel (按需修改)
5. WHEN DiffSBDDModel is called by EDMActor THEN DiffSBDDModel SHALL implement get_mask() and sample_p_zs_given_zt() methods with signatures matching EDMModel (替换 EDMModel)
6. WHEN ForceReward is replaced THEN the replacement reward (DockingReward or custom reward) SHALL implement the same BaseReward interface (calculate_rewards method) (不使用 ForceReward，替换为自定义奖励)
7. WHEN Filter is replaced THEN the replacement DiffSBDDFilter SHALL implement the same filter interface (替换 Filter)
8. WHEN DiffSBDD model is referenced THEN the DiffSBDD code SHALL remain read-only without any modifications (只读引用)

### Requirement 14: 数据格式兼容性

**User Story:** As a developer, I want DiffSBDDModel to return data in RLPF-compatible formats, so that EDMRollout and Filter can process the data without modification.

#### Acceptance Criteria

1. WHEN DiffSBDDModel.sample() returns atom types h THEN h SHALL be a dictionary with keys 'integer' and 'categorical', where h['categorical'] contains the one-hot encoded atom types tensor of shape [B, N, num_atom_types], matching the format expected by EDMRollout (which accesses h["categorical"])
2. WHEN molecules need to be converted to RDKit Mol objects for filtering or reward computation THEN the DiffSBDDFilter SHALL directly import and use DiffSBDD's `analysis.molecule_builder.build_molecule` function (which uses OpenBabel), because DiffSBDD generates drug-like molecules with 10 atom types (C, N, O, S, B, Br, Cl, P, I, F) that are not supported by RLPF's QM9-based build_molecule (which only supports H, C, N, O, F)
3. WHEN dataset_info is required for molecule building or validation THEN the system SHALL use DiffSBDD's crossdock dataset_params format containing atom_encoder, atom_decoder, aa_encoder, aa_decoder, and bond information compatible with drug-like molecules

