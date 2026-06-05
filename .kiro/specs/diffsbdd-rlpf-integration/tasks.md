# Implementation Plan

## DiffSBDD-RLPF Integration Tasks

- [x] 1. 项目结构和基础设施
  - [x] 1.1 创建 ddpo/ 目录结构
    - 创建 `ddpo/integration/`、`ddpo/rewards/`、`ddpo/utils/`、`tests/` 目录
    - 创建各目录的 `__init__.py` 文件
    - _Requirements: 7.1_

  - [x] 1.2 配置测试框架
    - 创建 `pyproject.toml` 或 `pytest.ini` 配置 pytest
    - 添加 hypothesis 依赖配置
    - 设置 hypothesis 最小运行次数为 100
    - _Requirements: Design - Testing Strategy_

  - [x] 1.3 创建 BaseReward 接口
    - 创建 `ddpo/rewards/base.py`
    - 定义 `BaseReward` 抽象基类
    - 定义 `calculate_rewards(data: DataProto) -> DataProto` 抽象方法
    - _Requirements: 4.7, 5.4, 6.1_

  - [x] 1.4 创建训练配置文件
    - 创建 `ddpo/configs/diffsbdd_ddpo.yaml`
    - 包含 model、pocket、reward、training 配置节
    - _Requirements: 7.1, 7.2_

- [x] 2. Mask 格式转换工具
  - [x] 2.1 实现 mask_utils.py
    - 实现 `batch_mask_to_node_mask()` 函数：[total_nodes] → [B, N, 1]
    - 实现 `node_mask_to_batch_mask()` 函数：[B, N, 1] → [total_nodes]
    - 添加输入验证和错误处理
    - _Requirements: 2.1, 2.2, 2.4, 2.5_

  - [x] 2.2 编写 mask 转换的属性测试
    - **Property 1: Mask Conversion Round-Trip**
    - **Validates: Requirements 2.3**

- [x] 3. 口袋数据加载 (复用 DiffSBDD 实现)
  - [x] 3.1 确认复用 DiffSBDD 的 `LigandPocketDDPM.prepare_pocket()` 方法
    - **不需要单独实现 PocketLoader**，DiffSBDD 已有完整实现
    - `LigandPocketDDPM.prepare_pocket()` 已实现：PDB 解析、CA 原子提取、坐标居中、one-hot 编码
    - 在 DiffSBDDModel 中直接调用 `self.model.prepare_pocket(residues)`
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_ ✅ 已由 DiffSBDD 满足

- [x] 4. DiffSBDD 模型封装
  - **依赖**: Task 2 (mask_utils)
  - **Git 分支**: `feature/diffsbdd-model`
  
  - [x] 4.1 实现 DiffSBDDModel 基础结构和口袋加载



    - 创建 `ddpo/integration/diffsbdd_model.py`
    - 继承 `BaseModel` 抽象基类
    - 实现 `__init__(checkpoint_path, pocket_pdb_path, ref_ligand_path, device)`:
      - 使用 `LigandPocketDDPM.load_from_checkpoint()` 加载预训练模型
      - 使用 BioPython 解析 PDB，调用 `utils.get_pocket_from_ligand()` 获取残基
      - 调用 `self.model.prepare_pocket(residues)` 加载口袋数据（返回 dict 包含 x, one_hot, size, mask）
      - 存储 `self.ddpm = self.model.ddpm` (ConditionalDDPM 实例)
      - **注意**: `prepare_pocket(residues, repeats=1)` 的 repeats 参数在 `_replicate_pocket` 中动态设置
    - 实现 `load(model_path)` 加载权重
    - 实现 `to(device)` 移动到指定设备
    - 存储关键属性: `self.T`, `self.n_dims`, `self.atom_nf`, `self.dataset_info`
    - **口袋数据结构** (来自 `prepare_pocket` 返回):
      ```python
      pocket = {
          'x': torch.Tensor,        # [N_residues, 3] 坐标
          'one_hot': torch.Tensor,  # [N_residues, 23] 残基类型 one-hot
          'size': torch.Tensor,     # [repeats] 每个 batch 的残基数
          'mask': torch.Tensor      # [N_residues * repeats] batch_mask 格式
      }
      ```
    - _Requirements: 1.1, 1.7, 3.1-3.6_
    - **Git**: `git commit -m "feat(model): implement DiffSBDDModel basic structure with pocket loading"`

  - [x] 4.2 实现 get_mask 和格式转换方法







    - 实现 `get_mask(nodesxsample, batch_size, max_n_nodes)`:
      - 生成 RLPF 格式的 `node_mask [B, N, 1]` 和 `edge_mask`
      - **注意**: DiffSBDD 不使用 edge_mask，可返回 None 或全零 tensor
      - 参考 RLPF EDMModel.get_mask() 实现
    - 实现 `_rlpf_to_diffsbdd_format(x, node_mask)`: [B,N,D] → [total_nodes, D]
      - 使用 `node_mask_to_batch_mask()` 获取 batch_mask
      - 提取有效节点: `x[node_mask.squeeze(-1).bool()]`
    - 实现 `_diffsbdd_to_rlpf_format(x, batch_mask, max_nodes)`: [total_nodes, D] → [B,N,D]
      - 创建 [B, N, D] 零张量
      - 使用 batch_mask 填充有效位置
    - 实现 `_replicate_pocket(batch_size)`: 复制口袋数据到指定 batch size
      - 调用 `self.model.prepare_pocket(self.residues, repeats=batch_size)`
      - 或手动复制: `pocket['x'].repeat(batch_size, 1)` 等
    - 使用 `ddpo/integration/mask_utils.py` 中的转换函数
    - _Requirements: 2.1, 2.2_
    - **Git**: `git commit -m "feat(model): implement get_mask and format conversion methods"`

  - [x] 4.3 实现 compute_log_p_zs_given_zt 方法



    - 实现高斯对数概率计算公式（参考 RLPF EDMModel）:
      ```python
      # 确保 sigma > epsilon 避免数值问题
      log_p = -0.5 * ((zs - mu) ** 2) / (sigma ** 2) * node_mask
      # 对维度求和并归一化
      p_zs_zt = log_p.sum(dim=tuple(range(1, log_p.ndim))) / (node_mask.sum(...) * D)
      ```
    - 支持 RLPF 格式输入 [B, N, D]
    - _Requirements: 1.4, 11.1, 11.2_
    - **Git**: `git commit -m "feat(model): implement compute_log_p_zs_given_zt"`

  - [x] 4.4 实现 sample_p_zs_given_zt 方法 🔴 关键



    - **问题**: DiffSBDD 的 `sample_p_zs_given_zt` 只返回 `(zs_lig, xh_pocket)`
    - **解决**: 手动计算 mu, sigma, log_prob（参考 DiffSBDD ConditionalDDPM.sample_p_zs_given_zt 源码）
    - 实现步骤:
      1. 转换输入格式: RLPF [B,N,D] → DiffSBDD [total_nodes, D]
      2. 复制口袋数据到 batch size: `pocket = self._replicate_pocket(batch_size)`
      3. 拼接口袋: `xh_pocket = torch.cat([pocket['x'], pocket['one_hot']], dim=1)`
      4. 计算 gamma_s, gamma_t:
         ```python
         gamma_s = self.ddpm.gamma(s)
         gamma_t = self.ddpm.gamma(t)
         ```
      5. 计算 sigma_and_alpha_t_given_s:
         ```python
         sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = \
             self.ddpm.sigma_and_alpha_t_given_s(gamma_t, gamma_s, zt_lig)
         sigma_s = self.ddpm.sigma(gamma_s, target_tensor=zt_lig)
         sigma_t = self.ddpm.sigma(gamma_t, target_tensor=zt_lig)
         ```
      6. 调用 `self.ddpm.dynamics()` 获取 eps_t:
         ```python
         eps_t_lig, _ = self.ddpm.dynamics(zt_lig, xh_pocket, t, lig_mask, pocket['mask'])
         ```
      7. 计算 mu (参考 DiffSBDD 源码):
         ```python
         mu_lig = zt_lig / alpha_t_given_s[lig_mask] - \
                  (sigma2_t_given_s / alpha_t_given_s / sigma_t)[lig_mask] * eps_t_lig
         ```
      8. 计算 sigma: `sigma = sigma_t_given_s * sigma_s / sigma_t`
      9. 采样 zs 并调用 `remove_mean_batch` 居中:
         ```python
         zs_lig, xh_pocket = self.ddpm.sample_normal_zero_com(
             mu_lig, xh_pocket, sigma, lig_mask, pocket['mask'])
         ```
      10. 转换输出格式: DiffSBDD → RLPF
      11. 计算 log_prob: `log_prob = self.compute_log_p_zs_given_zt(zs_rlpf, mu_rlpf, sigma, node_mask)`
    - 返回 `(zs, log_prob, mu, sigma)` 匹配 RLPF 接口
    - _Requirements: 1.3_
    - **Git**: `git commit -m "feat(model): implement sample_p_zs_given_zt with log_prob computation"`

  - [x] 4.5 编写 log_prob 计算的属性测试








    - **Property 10: Log Probability Gaussian Formula**
    - **Validates: Requirements 1.4, 11.1, 11.2**
    - 测试: 对于已知的 zs, mu, sigma，验证 log_prob 与解析公式一致
    - **Git**: `git commit -m "test(model): add property test for log_prob computation"`

  - [x] 4.6 实现 sample_p_xh_given_z0 方法



    - 封装 DiffSBDD 的 `sample_p_xh_given_z0` 方法
    - **DiffSBDD 返回**: `(x_lig, h_lig, x_pocket, h_pocket)` - 4 个值
    - **RLPF 期望返回**: `(x, h, mu, sigma, log_prob, zeros, xh)` - 7 个值
    - 实现步骤:
      1. 转换输入格式: RLPF → DiffSBDD
      2. 计算 gamma_0, sigma_x, mu_x（参考 DiffSBDD 源码）:
         ```python
         t_zeros = torch.zeros(size=(batch_size, 1), device=z0_lig.device)
         gamma_0 = self.ddpm.gamma(t_zeros)
         sigma_x = self.ddpm.SNR(-0.5 * gamma_0)
         net_out_lig, _ = self.ddpm.dynamics(z0_lig, xh_pocket, t_zeros, lig_mask, pocket['mask'])
         mu_x_lig = self.ddpm.compute_x_pred(net_out_lig, z0_lig, gamma_0, lig_mask)
         ```
      3. 采样并计算 log_prob
      4. **h 格式**: 直接返回 tensor（不需要字典包装）
         ```python
         # DiffSBDD 返回: h_lig = F.one_hot(torch.argmax(h_lig, dim=1), self.atom_nf)
         # 转换为 RLPF 格式 [B, N, num_atom_types]
         h_lig_rlpf = self._diffsbdd_to_rlpf_format(h_lig, lig_mask, max_nodes)
         # 直接返回 tensor，Rollout 会存到 batch["categorical"]
         ```
      5. 转换输出格式: DiffSBDD → RLPF
    - 返回 `(x, h, mu, sigma, log_prob, zeros, xh)` 其中 h 是 tensor
    - _Requirements: 1.6, 14.1_
    - **Git**: `git commit -m "feat(model): implement sample_p_xh_given_z0"`

  - [x] 4.7 实现 sample 方法（完整采样循环）
    - 参考 RLPF EDMModel.sample() 结构
    - **方法签名**: `sample(n_samples, n_nodes, node_mask, edge_mask, context=None, fix_noise=False, timestep=1000)`
    - 实现步骤:
      1. 初始化噪声 z（在口袋中心采样）:
         ```python
         # 参考 DiffSBDD sample_given_pocket
         pocket = self._replicate_pocket(n_samples)
         xh_pocket = torch.cat([pocket['x'], pocket['one_hot']], dim=1)
         
         # 在口袋中心采样初始噪声
         mu_lig_x = scatter_mean(pocket['x'], pocket['mask'], dim=0)
         mu_lig_h = torch.zeros((n_samples, self.atom_nf), device=device)
         mu_lig = torch.cat((mu_lig_x, mu_lig_h), dim=1)[lig_mask]
         sigma = torch.ones_like(pocket['size']).unsqueeze(1)
         z_lig, xh_pocket = self.ddpm.sample_normal_zero_com(
             mu_lig, xh_pocket, sigma, lig_mask, pocket['mask'])
         ```
      2. 转换初始 z 为 RLPF 格式并记录
      3. 循环 T 步: 调用 `sample_p_zs_given_zt`
         ```python
         for s in reversed(range(0, self.T)):
             s_array = torch.full((n_samples, 1), fill_value=s, device=device)
             t_array = s_array + 1
             s_array = s_array / self.T
             t_array = t_array / self.T
             z, logp, mu, sigma = self.sample_p_zs_given_zt(s_array, t_array, z, node_mask, edge_mask)
             latents.append(z)
             logps.append(logp)
             # ...
         ```
      4. 最后调用 `sample_p_xh_given_z0`
      5. 记录轨迹: latents, log_probs, timesteps, mus, sigmas
    - 调用 `remove_mean_batch` 保持 SE(3) 等变性
    - **h 格式**: 直接返回 tensor `h_lig [B, N, num_atom_types]`（不需要字典包装）
    - 返回 `(x, h, latents, logps, timesteps, mus, sigmas)` 匹配 RLPF 接口
    - _Requirements: 1.2, 1.5, 1.6, 10.1, 11.3, 11.4, 14.1_
    - **Git**: `git commit -m "feat(model): implement sample method with trajectory recording"`

  - [x]* 4.8 编写 sample 方法的属性测试



    - **Property 11: Trajectory Log Probs Shape** - log_probs 形状为 [B, T]
    - **Property 12: Sample Output Structure** - 返回包含所有必需键
    - **Property 14: h Format Tensor Structure** - h 是 tensor [B, N, num_atom_types]
    - **Validates: Requirements 1.2, 11.4, 14.1**
    - **Git**: `git commit -m "test(model): add property tests for sample method"`

- [x] 5. Checkpoint - 确保模型封装测试通过





  - 确保所有测试通过，如有问题请询问用户
  - **Git**: 合并分支并打标签
    ```bash
    git checkout develop
    git merge feature/diffsbdd-model -m "merge: DiffSBDDModel wrapper complete"
    git tag -a task-4-complete -m "Task 4: DiffSBDDModel 封装完成"
    ```

- [x] 6. 分子格式转换工具
  - [x] 6.1 实现 coords_to_smiles 转换函数
    - 创建 `ddpo/utils/molecule_converter.py`
    - 实现 `coords_to_rdkit_mol(x, h, dataset_info)`: 3D坐标 → RDKit Mol
    - 实现 `rdkit_mol_to_smiles(mol)`: RDKit Mol → Canonical SMILES
    - 实现 `coords_to_smiles(x, h, dataset_info)`: 3D坐标 → SMILES（便捷函数）
    - **关键决策**: 使用SMILES作为判别器输入格式
      - 避免构象差异（生成分子是结合构象，数据库药物是能量最低态）
      - Uni-Mol内部自动从SMILES生成一致的3D表示
      - 判别器关注化学结构，而不是特定构象
    - _Requirements: 12.1, 12.2, UNIMOL_INTEGRATION.md_
    - **Git**: `git commit -m "feat(utils): implement coords_to_smiles conversion"`

  - [x] 6.2 实现其他格式转换函数（用于对接）
    - 实现 `rdkit_mol_to_pdbqt(mol)`: RDKit Mol → PDBQT（用于Vina对接）
    - 实现 `process_molecule(mol)`: 添加氢原子和几何优化
    - 复用 DiffSBDD 的 `build_molecule()` 和 `sdf_to_pdbqt()`
    - _Requirements: 12.3, 12.4_
    - **Git**: `git commit -m "feat(utils): add PDBQT conversion for docking"`

  - [x]* 6.3 编写分子转换的属性测试
    - **Property 13: SMILES Canonicalization** - SMILES规范化一致性
    - **Property 16: Coords to SMILES Round-Trip** - 坐标→SMILES→Mol保持化学结构
    - **Validates: Requirements 12.1, 12.2**
    - **Git**: `git commit -m "test(utils): add property tests for molecule conversion"`

- [x] 7. Receptor PDBQT 准备 (复用 DiffSBDD 实现)
  - [x] 7.1 确认复用 DiffSBDD 现有实现
    - **不需要预先转换 PDBQT**，DiffSBDD 的 `smina_score()` 直接接受 PDB 文件
    - `calculate_qvina2_score()` 会自动处理 PDB → PDBQT 转换
    - 直接使用 `data/8GJJ-C-Cip.pdb` 作为 receptor 文件
    - _Requirements: 4.1_ ✅ 已由 DiffSBDD 满足

- [x] 8. Docking 奖励函数
  - **Git 分支**: `feature/docking-reward`
  - [x] 8.1 实现 DockingReward 类


    - 继承 `BaseReward`
    - 实现 `__init__()` 加载 receptor 文件路径
    - 直接调用 DiffSBDD 的 `build_molecule()` 和 `smina_score()`
    - _Requirements: 4.1, 4.2, 4.3_
    - **Git**: `git commit -m "feat(reward): implement DockingReward with Vina"`

  - [x] 8.2 实现 calculate_rewards 方法



    - 从 DataProto 提取 x, h
    - 调用 `build_molecule()` 构建 RDKit Mol
    - 调用 `smina_score()` 计算对接分数
    - 处理失败情况（返回 0.0）
    - 归一化奖励（除以 -10.0）
    - _Requirements: 4.4, 4.5, 4.6, 4.7_
    - **Git**: `git commit -m "feat(reward): implement calculate_rewards"`

  - [ ]* 8.3 编写 Docking 奖励的属性测试
    - **Property 4: Docking Reward Normalization**
    - **Property 9: Reward Rotation Invariance**
    - **Validates: Requirements 4.6, 10.3**
    - **Git**: `git commit -m "test(reward): add property tests for DockingReward"`

- [x] 9. Drug-likeness 奖励函数
  - **Git 分支**: `feature/druglikeness-reward`
  - [x] 9.1 实现 DruglikenessReward 类
    - 继承 `BaseReward`
    - 直接调用 DiffSBDD 的 `MoleculeProperties.calculate_qed()` 和 `calculate_sa()`
    - **注意**: DiffSBDD 的 `calculate_sa()` 已经归一化为 `(10-sa)/9`
    - _Requirements: 5.1, 5.2, 5.3, 5.4_
    - **Git**: `git commit -m "feat(reward): implement DruglikenessReward"`

  - [x]* 9.2 编写 Drug-likeness 的属性测试
    - **Property 5: QED Score Range**
    - **Property 6: SA Score Normalization**
    - **Validates: Requirements 5.2, 5.4**
    - **Git**: `git commit -m "test(reward): add property tests for DruglikenessReward"`

- [ ] 10. 流形判别器奖励函数（第二道关卡）⭐⭐⭐⭐⭐
  - **依赖**: Task 6 (molecule_converter)
  - **Git 分支**: `feature/manifold-discriminator`
  
  - [ ] 10.1 安装和配置 Uni-Mol
    - 安装 unimol-tools: `pip install unimol_tools --upgrade`
    - 安装 huggingface_hub: `pip install huggingface_hub`
    - 验证安装: `python -c "from unimol_tools import UniMolRepr; print('Uni-Mol OK')"`
    - 配置模型自动下载（通过huggingface_hub）
    - _Requirements: UNIMOL_INTEGRATION.md, tech.md_
    - **Git**: `git commit -m "chore(deps): add Uni-Mol dependencies"`

  - [ ] 10.2 实现 ManifoldDiscriminatorReward 基础结构
    - 创建 `ddpo/rewards/manifold_discriminator.py`
    - 继承 `BaseReward`
    - 实现 `__init__()`:
      - 初始化 Uni-Mol 编码器: `UniMolRepr(data_type='molecule', model_name='unimolv1')`
      - 加载判别器模型（Binary Classifier）
      - 设置惩罚权重（默认 -5.0）
    - _Requirements: UNIMOL_INTEGRATION.md Section 2_
    - **Git**: `git commit -m "feat(reward): implement ManifoldDiscriminatorReward structure"`

  - [ ] 10.3 实现化学结构特征提取
    - 实现 `extract_features(smiles)`:
      - **ECFP4指纹 (2048维)**: 从SMILES生成药效团指纹
      - **Uni-Mol表示 (512维)**: 调用 `self.unimol_encoder.get_repr([smiles])`
      - 拼接为 2560维特征向量
    - **注意**: 不包含PoseBusters统计
      - 数据库药物没有对接构象，无法计算PoseBusters
      - PoseBusters应在Stage 1（PhysicsReward）中单独检查生成分子
    - _Requirements: UNIMOL_INTEGRATION.md Section 2.2_
    - **Git**: `git commit -m "feat(reward): implement chemical structure feature extraction"`

  - [ ] 10.4 实现判别器奖励计算
    - 实现 `calculate_rewards(data: DataProto)`:
      1. 从 DataProto 提取 x, h
      2. 调用 `coords_to_smiles()` 转换为SMILES
      3. 调用 `extract_features(smiles)` 提取2560维特征
      4. 通过判别器获取 prob(real)
      5. 如果 prob(real) > 0.5: reward = 0（通过，继续Stage 3）
      6. 如果 prob(real) ≤ 0.5: reward = penalty_weight（不通过，跳过Stage 3）
    - 返回 DataProto 包含:
      - `discriminator_scores`: [B] 奖励/惩罚
      - `discriminator_probs`: [B] prob(real)
      - `discriminator_pass`: [B] boolean
    - _Requirements: UNIMOL_INTEGRATION.md Section 2.2_
    - **Git**: `git commit -m "feat(reward): implement discriminator reward calculation"`

  - [ ] 10.5 准备判别器训练数据
    - 创建 `scripts/prepare_discriminator_data.py`
    - 实现 `load_chembl_drugs()`: 从ChEMBL/ZINC下载真实药物SMILES
    - 实现 `prepare_real_features()`: 提取真实药物的2560维特征（ECFP4 + Uni-Mol）
    - 实现 `prepare_generated_features()`: 从早期训练提取生成分子特征
    - 保存训练数据: `discriminator_train_data.pkl`
    - _Requirements: UNIMOL_INTEGRATION.md Section 3_
    - **Git**: `git commit -m "feat(data): add discriminator training data preparation"`

  - [ ] 10.6 训练判别器模型
    - 创建 `scripts/train_discriminator.py`
    - 实现二分类器: `nn.Sequential(nn.Linear(2560, 512), nn.ReLU(), nn.Linear(512, 1), nn.Sigmoid())`
    - 训练目标: 区分真实药物（label=1）vs 生成分子（label=0）
    - 保存模型: `checkpoints/discriminator.pt`
    - _Requirements: UNIMOL_INTEGRATION.md Section 3_
    - **Git**: `git commit -m "feat(model): add discriminator training script"`

  - [ ]* 10.7 编写判别器奖励的属性测试
    - **Property 17: Discriminator Feature Dimension** - 特征维度为2560
    - **Property 18: Discriminator Probability Range** - prob(real) ∈ [0, 1]
    - **Property 19: SMILES Consistency** - 相同SMILES产生相同特征
    - **Validates: UNIMOL_INTEGRATION.md**
    - **Git**: `git commit -m "test(reward): add property tests for ManifoldDiscriminatorReward"`

- [ ] 11. 组合奖励函数（三阶段系统）⭐⭐⭐⭐⭐
  - **依赖**: Task 8 (DockingReward), Task 9 (DruglikenessReward)
  - **Git 分支**: `feature/composite-reward`
  - [x] 10.1 实现 CompositeReward 类
    - 实现 `__init__()` 存储奖励函数和权重
    - 实现 `calculate_rewards()` 加权求和
    - 实现 `update_weights()` 动态更新权重
    - 实现异常处理和 fallback
    - _Requirements: 6.1, 6.2, 6.3, 6.4_
    - **Git**: `git commit -m "feat(reward): implement CompositeReward with weighted sum"`

  - [x]* 10.2 编写组合奖励的属性测试
    - **Property 7: Composite Reward Weighted Sum**
    - **Validates: Requirements 6.2**
    - **Git**: `git commit -m "test(reward): add property test for CompositeReward"`

- [x] 11. 分子有效性过滤 (复用 DiffSBDD 实现)
  - **Git 分支**: `feature/diffsbdd-filter`
  - [x] 11.1 创建 DiffSBDDFilter 类
    - 创建 `ddpo/utils/diffsbdd_filter.py`
    - 继承或参考 RLPF `Filter` 类结构
    - 直接调用 DiffSBDD 的 `BasicMolecularMetrics.compute_validity()`
    - 直接调用 DiffSBDD 的 `build_molecule()` 和 `rdmol_to_smiles()`
    - **不需要重新实现**，DiffSBDD 已有完整实现
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 14.2_
    - **Git**: `git commit -m "feat(filter): create DiffSBDDFilter reusing DiffSBDD"`

  - [x]* 11.2 编写有效性过滤的属性测试
    - **Property 8: Validity Ratio Computation**
    - **Property 15: DiffSBDD build_molecule Compatibility**
    - **Validates: Requirements 9.3, 14.2**
    - **Git**: `git commit -m "test(filter): add property tests for DiffSBDDFilter"`

- [x] 12. Checkpoint - 确保奖励函数测试通过
  - 确保所有测试通过，如有问题请询问用户
  - **测试结果**: 74 个测试全部通过 ✅
  - **Git**: 合并所有奖励相关分支并打标签
    ```bash
    git checkout develop
    git merge feature/molecule-converter feature/receptor-pdbqt feature/docking-reward feature/druglikeness-reward feature/composite-reward feature/diffsbdd-filter -m "merge: reward system complete"
    git tag -a v0.2.0 -m "Reward system complete (docking + druglikeness + filter)"
    ```

- [x] 13. 训练入口脚本



  - **依赖**: Task 4 (DiffSBDDModel), Task 10 (CompositeReward), Task 11 (DiffSBDDFilter)
  - **Git 分支**: `feature/training-pipeline`


  - [x] 13.1 创建 main_diffsbdd.py
    - 复制 `main_edm.py` 结构
    - 替换 `EDMModel` → `DiffSBDDModel`
    - 替换 `ForceReward` → `CompositeReward`
    - 替换 `Filter` → `DiffSBDDFilter`
    - 添加口袋路径配置
    - 新增 `DiffSBDDDataLoader`、`DiffSBDDRollout`、`StabilityAwareReward` 组件
    - _Requirements: 8.1, 13.1, 13.2, 13.3, 13.4, 13.5, 13.6_
    - **Git**: `git commit -m "feat(train): create main_diffsbdd.py entry point"`



  - [x] 13.2 实现配置加载和验证
    - 实现 `load_config()` 加载 YAML 配置
    - 验证必需参数 (`model.checkpoint`, `pocket.pdb_path`, `train.batch_size`, `train.learning_rate`)
    - 实现 `wandb_stub` 本地日志替代方案


    - _Requirements: 7.1, 7.2, 7.3, 7.4_
    - **Git**: `git commit -m "feat(train): implement config loading and validation"`

  - [x] 13.3 实现训练循环
    - 初始化所有组件 (model, dataloader, rollout, reward, filter, actor, trainer)
    - 调用 `DDPOTrainer.fit()`
    - 实现 `signal_handler` 优雅退出
    - 实现 `trainer.clean_up()` 清理
    - _Requirements: 8.2, 8.3, 8.4, 8.5_
    - **Git**: `git commit -m "feat(train): implement training loop with DDPOTrainer"`

- [x] 14. 端到端集成测试
  - [x]* 14.1 编写集成测试




    - 测试完整训练流程（少量步数）
    - 验证 checkpoint 保存/加载
    - 验证 wandb 日志记录
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_
    - **Git**: `git commit -m "test(integration): add end-to-end training test"`




- [x] 15. Final Checkpoint - 确保所有测试通过
  - 确保所有测试通过，如有问题请询问用户
  - **测试结果**: 105 个测试全部通过 ✅
  - **验证结果**: 129 项检查全部通过 ✅
  - **Git**: 最终合并并打版本标签
    ```bash
    git checkout develop
    git merge feature/training-pipeline -m "merge: training pipeline complete"
    git tag -a v1.0.0 -m "DiffSBDD-RLPF integration complete"
    git checkout main
    git merge develop -m "release: v1.0.0 DiffSBDD-RLPF integration"
    ```

---

## Phase 2: 优化与增强

- [ ] 16. 日志系统增强
  - **Git 分支**: `feature/logging-enhancement`
  
  - [ ] 16.1 完善训练指标记录
    - 添加 `ppo/clipfrac` - PPO clip 比例（关键监控指标）
    - 添加 `filter/ratio` - 有效样本过滤比例
    - 添加 `metrics/stability` - 分子稳定性
    - 添加 `reward/std` - 奖励标准差
    - 添加 `train/learning_rate` - 当前学习率
    - 使用分层命名：`reward/*`, `ppo/*`, `train/*`, `metrics/*`
    - _Requirements: 日志记录对比分析文档_
    - **Git**: `git commit -m "feat(logging): add comprehensive training metrics"`

  - [ ] 16.2 添加奖励分量记录
    - 在 CompositeReward 中分离记录各奖励分量
    - 添加 `reward/docking`, `reward/qed`, `reward/sa` 单独记录
    - 便于分析各奖励对训练的贡献
    - **Git**: `git commit -m "feat(logging): add reward component logging"`

  - [ ] 16.3 完善 Checkpoint 保存
    - 添加 `clipfracs` - 所有 epoch 的 clipfrac 记录
    - 添加 `validities` - 所有 epoch 的 validity 记录
    - 添加 `best_reward` - 最佳奖励记录
    - 添加 `config` - 训练配置保存
    - **Git**: `git commit -m "feat(checkpoint): enhance checkpoint with full metrics"`

  - [ ]* 16.4 添加分子质量指标记录（可选）
    - 添加 `metrics/connectivity` - 连通性
    - 添加 `metrics/uniqueness` - 唯一性
    - 添加 `metrics/novelty` - 新颖性
    - 添加 `metrics/diversity` - 多样性
    - **Git**: `git commit -m "feat(logging): add molecule quality metrics"`

- [ ] 17. 奖励函数优化（待定）
  - **Git 分支**: `feature/reward-optimization`
  - **状态**: 待用户确定具体优化方向
  
  - [ ] 17.1 奖励函数改进（待定）
    - 可能方向：
      - 添加新的奖励分量（如 LogP、Lipinski 规则）
      - 调整奖励权重策略（动态权重、课程学习）
      - 改进奖励归一化方法
      - 添加奖励 shaping 或 clipping
    - _Requirements: 待用户确定_

  - [ ] 17.2 奖励函数测试更新（待定）
    - 根据 17.1 的改进添加相应测试
    - _Requirements: 待用户确定_

- [ ] 18. Checkpoint - Phase 2 完成
  - 确保所有新增功能测试通过
  - **Git**: 合并并打标签
    ```bash
    git checkout develop
    git merge feature/logging-enhancement feature/reward-optimization -m "merge: phase 2 enhancements"
    git tag -a v1.1.0 -m "Phase 2: Logging and reward enhancements"
    ```

---

## 任务依赖图

```
Phase 1: 核心集成 (v1.0.0) ✅
==========================================
Task 1 (基础设施) ✅
    │
    ├── Task 2 (mask_utils) ✅
    │       │
    │       └──────────────────┐
    │                          │
    └── Task 3 (口袋加载) ✅    │
                               │
                               ▼
                        Task 4 (DiffSBDDModel) ✅
                               │
                               ▼
                        Task 5 (Checkpoint) ✅
                               │
                               ▼
                        Task 6 (molecule_converter) ✅
                               │
                               ├── Task 7 (receptor PDBQT) ✅
                               │         │
                               │         ▼
                               ├── Task 8 (DockingReward) ✅
                               │         │
                               │         │
                               │   Task 9 (DruglikenessReward) ✅
                               │         │
                               │         ▼
                               ├── Task 10 (CompositeReward) ✅
                               │
                               └── Task 11 (DiffSBDDFilter) ✅
                                         │
                                         ▼
                        Task 12 (Checkpoint) ✅ ◄─────────────────────────┘
                               │
                               ▼
                        Task 13 (main_diffsbdd.py) ✅
                               │
                               ▼
                        Task 14 (集成测试) ✅
                               │
                               ▼
                        Task 15 (Final Checkpoint) ✅

Phase 2: 优化与增强 (v1.1.0)
==========================================
                        Task 15 ✅
                               │
                               ▼
                        Task 16 (日志系统增强)
                               │
                               ▼
                        Task 17 (奖励函数优化 - 待定)
                               │
                               ▼
                        Task 18 (Phase 2 Checkpoint)
```

## 变更记录

- **2026-01-16 (v9)**: 添加 Phase 2 任务
  - Task 16: 日志系统增强 - 添加缺失的训练指标（clipfrac, filter_ratio, stability 等）
  - Task 17: 奖励函数优化 - 待用户确定具体方向
  - Task 18: Phase 2 Checkpoint
  - 更新任务依赖图，区分 Phase 1 和 Phase 2

- **2026-01-09 (v8)**: 简化奖励函数任务
  - Task 6: 删除 molecule_converter，直接复用 DiffSBDD 实现
  - Task 7: 删除 receptor PDBQT 准备，`smina_score()` 直接接受 PDB 文件
  - Task 8: 简化 DockingReward，直接调用 DiffSBDD 的 `build_molecule()` 和 `smina_score()`
  - Task 9: 简化 DruglikenessReward，直接调用 DiffSBDD 的 `MoleculeProperties.calculate_qed/sa()`
  - Task 11: 简化 DiffSBDDFilter，直接调用 DiffSBDD 的 `BasicMolecularMetrics.compute_validity()`

- **2024-12-20 (v7)**: 确认组件复用策略文档同步
  - 确认 `.kiro/steering/structure.md` 和 `design.md` 中的组件复用策略表格一致
  - EDMActor 直接复用，不修改
  - DDPOTrainer 和 main_edm.py 按需修改
  - ForceReward 不使用，替换为自定义 DockingReward
  - EDMModel 不使用，替换为封装后的 DiffSBDDModel
  - DiffSBDD 只读引用，不修改

- **2024-12-20 (v6)**: 明确组件复用策略
  - 确认 EDMActor 直接复用，不修改
  - DDPOTrainer 和 main_edm.py 按需修改
  - EDMRollout 按需修改
  - ForceReward 不使用，替换为自定义 DockingReward
  - EDMModel 不使用，替换为 DiffSBDDModel
  - 由于复用 EDMActor，DiffSBDDModel 必须实现 RLPF 格式接口（get_mask, sample_p_zs_given_zt）
  - 更新设计文档和 steering 文档

- **2024-12-20 (v5)**: 简化坐标居中处理
  - 确认 DiffSBDD 的 `sample_normal_zero_com()` 内部已自动调用 `remove_mean_batch()`
  - 不需要额外处理坐标居中，DiffSBDD 内部已完成
  - 更新设计文档：坐标居中从 🟡 MEDIUM 改为 🟢 RESOLVED

- **2024-12-20 (v4)**: 简化 build_molecule 适配方案
  - 确认 DiffSBDD 已有完整的 `build_molecule` 实现（使用 OpenBabel）
  - Task 11 简化为：复制 RLPF Filter 代码，只修改 import 语句
  - 移除 Task 11 对 Task 6 的依赖（不需要 molecule_converter）
  - 更新设计文档：build_molecule 从 🔴 HIGH 改为 🟢 RESOLVED

- **2024-12-19 (v3)**: 更新 h 格式处理方案
  - 经分析确认：RLPF Rollout 只存储 `h["categorical"]` 到 `batch["categorical"]`
  - 下游模块 (Filter, Reward, Actor) 直接读取 tensor，不需要字典格式
  - 更新 Task 4.6, 4.7, 4.8：h 直接返回 tensor，不需要包装成字典
  - 更新 Property 14：从 "Dict Structure" 改为 "Tensor Structure"

- **2024-12-19 (v2)**: 重构 Task 4 (DiffSBDD 模型封装)
  - 调整子任务顺序：先实现 `compute_log_p_zs_given_zt`，再实现 `sample_p_zs_given_zt`
  - 新增 Task 4.6 `sample_p_xh_given_z0` 方法（原合并在 4.7 中）
  - 添加更详细的实现步骤和代码示例
  - 明确 DiffSBDD `sample_p_zs_given_zt` 不返回 log_prob 的问题及解决方案
  - 添加 `get_mask` 方法实现要求（RLPF EDMRollout 需要）

- **2024-12-19**: 删除 Task 4 (dataset_info 配置)
  - 决定直接使用 `from DiffSBDD.constants import dataset_params` 获取配置
  - 不再创建单独的 `ddpo/configs/dataset_info.py` 文件
  - 重新编号所有任务 (原 Task 5-16 → Task 4-15)
  - 更新任务依赖关系

- **2024-12-18**: 重构任务列表
  - 合并原 DiffSBDDFilter 和 ValidityFilter 为统一任务
  - 简化描述，明确复用 DiffSBDD 的 `build_molecule`
  - 添加任务依赖注释和依赖图
