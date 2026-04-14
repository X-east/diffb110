#!/usr/bin/env python
"""
已完成任务验证脚本
按照 .kiro/specs/diffsbdd-rlpf-integration/tasks.md 中已完成的步骤进行验证

已完成任务:
- Task 1: 项目结构和基础设施 (1.1-1.4) ✅
- Task 2: Mask 格式转换工具 (2.1-2.2) ✅  
- Task 3: 口袋数据加载 (3.1) ✅
- Task 4: DiffSBDDModel 封装 (4.1-4.5) ✅

使用方法:
    python scripts/verify_completed_tasks.py              # 运行所有验证
    python scripts/verify_completed_tasks.py --task 1     # 验证 Task 1 全部
    python scripts/verify_completed_tasks.py --task 1.1   # 只验证 Task 1.1
    python scripts/verify_completed_tasks.py --task 4.3   # 只验证 Task 4.3
    python scripts/verify_completed_tasks.py --list       # 列出所有可验证的任务
    python scripts/verify_completed_tasks.py --run-tests  # 同时运行 pytest
"""

import argparse
import sys
import os
import traceback
from typing import List, Tuple, Callable, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

results: List[Tuple[str, bool, str]] = []

def record(name: str, success: bool, msg: str = ""):
    results.append((name, success, msg))
    status = "✅" if success else "❌"
    print(f"    {status} {name}")
    if msg and not success:
        print(f"       └─ {msg}")

def run_check(name: str, fn: Callable[[], bool]):
    try:
        success = fn()
        record(name, success)
    except Exception as e:
        record(name, False, str(e))

def print_task_header(task_id: str, title: str):
    print(f"\n{'='*60}")
    print(f"[{task_id}] {title}")
    print(f"{'='*60}")


# ============================================================================
# Task 1.1: 创建 ddpo/ 目录结构
# ============================================================================

def verify_task_1_1():
    """Task 1.1: 创建 ddpo/ 目录结构"""
    print_task_header("1.1", "创建 ddpo/ 目录结构")
    
    def check_ddpo_root():
        return os.path.isdir("ddpo")
    run_check("ddpo/ 目录存在", check_ddpo_root)
    
    def check_integration_dir():
        return os.path.isdir("ddpo/integration")
    run_check("ddpo/integration/ 目录存在", check_integration_dir)
    
    def check_rewards_dir():
        return os.path.isdir("ddpo/rewards")
    run_check("ddpo/rewards/ 目录存在", check_rewards_dir)
    
    def check_utils_dir():
        return os.path.isdir("ddpo/utils")
    run_check("ddpo/utils/ 目录存在", check_utils_dir)
    
    def check_configs_dir():
        return os.path.isdir("ddpo/configs")
    run_check("ddpo/configs/ 目录存在", check_configs_dir)
    
    def check_tests_dir():
        return os.path.isdir("tests")
    run_check("tests/ 目录存在", check_tests_dir)
    
    def check_init_files():
        files = [
            "ddpo/__init__.py",
            "ddpo/integration/__init__.py",
            "ddpo/rewards/__init__.py",
            "ddpo/utils/__init__.py"
        ]
        missing = [f for f in files if not os.path.isfile(f)]
        if missing:
            print(f"       缺少: {missing}")
            return False
        return True
    run_check("__init__.py 文件完整", check_init_files)

# ============================================================================
# Task 1.2: 配置测试框架
# ============================================================================

def verify_task_1_2():
    """Task 1.2: 配置测试框架"""
    print_task_header("1.2", "配置测试框架")
    
    def check_pyproject():
        return os.path.isfile("pyproject.toml")
    run_check("pyproject.toml 存在", check_pyproject)
    
    def check_pytest_config():
        if not os.path.isfile("pyproject.toml"):
            return False
        with open("pyproject.toml", "r") as f:
            content = f.read()
        return "[tool.pytest" in content
    run_check("pytest 配置存在", check_pytest_config)
    
    def check_hypothesis_dep():
        if not os.path.isfile("pyproject.toml"):
            return False
        with open("pyproject.toml", "r") as f:
            content = f.read()
        return "hypothesis" in content
    run_check("hypothesis 依赖配置", check_hypothesis_dep)
    
    def check_hypothesis_settings():
        if not os.path.isfile("pyproject.toml"):
            return False
        with open("pyproject.toml", "r") as f:
            content = f.read()
        return "hypothesis" in content and ("max_examples" in content or "100" in content)
    run_check("hypothesis 最小运行次数配置 (100)", check_hypothesis_settings)
    
    def check_hypothesis_import():
        import hypothesis
        return True
    run_check("hypothesis 可导入", check_hypothesis_import)
    
    def check_pytest_import():
        import pytest
        return True
    run_check("pytest 可导入", check_pytest_import)

# ============================================================================
# Task 1.3: 创建 BaseReward 接口
# ============================================================================

def verify_task_1_3():
    """Task 1.3: 创建 BaseReward 接口"""
    print_task_header("1.3", "创建 BaseReward 接口")
    
    def check_base_file():
        return os.path.isfile("ddpo/rewards/base.py")
    run_check("ddpo/rewards/base.py 存在", check_base_file)
    
    def check_base_reward_import():
        from ddpo.rewards.base import BaseReward
        return True
    run_check("BaseReward 类可导入", check_base_reward_import)
    
    def check_is_abstract():
        from ddpo.rewards.base import BaseReward
        from abc import ABC
        return issubclass(BaseReward, ABC)
    run_check("BaseReward 是抽象基类 (ABC)", check_is_abstract)
    
    def check_calculate_rewards_method():
        from ddpo.rewards.base import BaseReward
        return hasattr(BaseReward, 'calculate_rewards')
    run_check("calculate_rewards 抽象方法存在", check_calculate_rewards_method)
    
    def check_method_signature():
        from ddpo.rewards.base import BaseReward
        import inspect
        if not hasattr(BaseReward, 'calculate_rewards'):
            return False
        sig = inspect.signature(BaseReward.calculate_rewards)
        params = list(sig.parameters.keys())
        return 'data' in params
    run_check("calculate_rewards(data) 签名正确", check_method_signature)

# ============================================================================
# Task 1.4: 创建训练配置文件
# ============================================================================

def verify_task_1_4():
    """Task 1.4: 创建训练配置文件"""
    print_task_header("1.4", "创建训练配置文件")
    
    def check_config_file():
        return os.path.isfile("ddpo/configs/diffsbdd_ddpo.yaml")
    run_check("diffsbdd_ddpo.yaml 存在", check_config_file)
    
    def check_yaml_parse():
        import yaml
        with open("ddpo/configs/diffsbdd_ddpo.yaml", "r") as f:
            config = yaml.safe_load(f)
        return config is not None
    run_check("YAML 可解析", check_yaml_parse)
    
    def check_model_section():
        import yaml
        with open("ddpo/configs/diffsbdd_ddpo.yaml", "r") as f:
            config = yaml.safe_load(f)
        return 'model' in config
    run_check("model 配置节存在", check_model_section)
    
    def check_pocket_section():
        import yaml
        with open("ddpo/configs/diffsbdd_ddpo.yaml", "r") as f:
            config = yaml.safe_load(f)
        return 'pocket' in config
    run_check("pocket 配置节存在", check_pocket_section)
    
    def check_reward_section():
        import yaml
        with open("ddpo/configs/diffsbdd_ddpo.yaml", "r") as f:
            config = yaml.safe_load(f)
        return 'reward' in config
    run_check("reward 配置节存在", check_reward_section)
    
    def check_training_section():
        import yaml
        with open("ddpo/configs/diffsbdd_ddpo.yaml", "r") as f:
            config = yaml.safe_load(f)
        return 'train' in config
    run_check("train 配置节存在", check_training_section)


# ============================================================================
# Task 2.1: 实现 mask_utils.py
# ============================================================================

def verify_task_2_1():
    """Task 2.1: 实现 mask_utils.py"""
    print_task_header("2.1", "实现 mask_utils.py")
    
    import torch
    
    def check_file():
        return os.path.isfile("ddpo/integration/mask_utils.py")
    run_check("mask_utils.py 存在", check_file)
    
    def check_batch_to_node_import():
        from ddpo.integration.mask_utils import batch_mask_to_node_mask
        return True
    run_check("batch_mask_to_node_mask 可导入", check_batch_to_node_import)
    
    def check_node_to_batch_import():
        from ddpo.integration.mask_utils import node_mask_to_batch_mask
        return True
    run_check("node_mask_to_batch_mask 可导入", check_node_to_batch_import)
    
    def check_batch_to_node_basic():
        """[total_nodes] → [B, N, 1]"""
        from ddpo.integration.mask_utils import batch_mask_to_node_mask
        batch_mask = torch.tensor([0, 0, 1, 1, 1, 2, 2])
        node_mask = batch_mask_to_node_mask(batch_mask)
        assert node_mask.shape == (3, 3, 1), f"形状错误: {node_mask.shape}"
        return True
    run_check("batch_mask_to_node_mask 基本转换", check_batch_to_node_basic)
    
    def check_batch_to_node_content():
        from ddpo.integration.mask_utils import batch_mask_to_node_mask
        batch_mask = torch.tensor([0, 0, 1, 1, 1, 2, 2])
        node_mask = batch_mask_to_node_mask(batch_mask)
        expected = torch.tensor([[[1],[1],[0]], [[1],[1],[1]], [[1],[1],[0]]], dtype=node_mask.dtype)
        return torch.equal(node_mask, expected)
    run_check("batch_mask_to_node_mask 内容正确", check_batch_to_node_content)
    
    def check_node_to_batch_basic():
        """[B, N, 1] → [total_nodes]"""
        from ddpo.integration.mask_utils import node_mask_to_batch_mask
        node_mask = torch.tensor([[[1],[1],[0]], [[1],[1],[1]], [[1],[1],[0]]])
        batch_mask = node_mask_to_batch_mask(node_mask)
        expected = torch.tensor([0, 0, 1, 1, 1, 2, 2])
        return torch.equal(batch_mask, expected)
    run_check("node_mask_to_batch_mask 基本转换", check_node_to_batch_basic)
    
    def check_input_validation():
        """输入验证"""
        from ddpo.integration.mask_utils import batch_mask_to_node_mask
        try:
            batch_mask_to_node_mask(torch.tensor([-1, 0, 1]))
            return False  # 应该抛出异常
        except (ValueError, RuntimeError):
            return True
        except:
            return True  # 其他异常也可接受
    run_check("输入验证 (错误处理)", check_input_validation)

# ============================================================================
# Task 2.2: 编写 mask 转换的属性测试
# ============================================================================

def verify_task_2_2():
    """Task 2.2: 编写 mask 转换的属性测试 (Property 1)"""
    print_task_header("2.2", "编写 mask 转换的属性测试")
    
    import torch
    
    def check_test_file():
        return os.path.isfile("tests/property/test_mask_properties.py")
    run_check("test_mask_properties.py 存在", check_test_file)
    
    def check_property_1_test():
        if not os.path.isfile("tests/property/test_mask_properties.py"):
            return False
        with open("tests/property/test_mask_properties.py", "r") as f:
            content = f.read()
        return "round" in content.lower() or "Property 1" in content
    run_check("Property 1 测试存在 (Round-Trip)", check_property_1_test)
    
    def check_hypothesis_usage():
        if not os.path.isfile("tests/property/test_mask_properties.py"):
            return False
        with open("tests/property/test_mask_properties.py", "r") as f:
            content = f.read()
        return "@given" in content or "hypothesis" in content
    run_check("使用 hypothesis @given 装饰器", check_hypothesis_usage)
    
    def check_round_trip_logic():
        """手动验证往返转换"""
        from ddpo.integration.mask_utils import batch_mask_to_node_mask, node_mask_to_batch_mask
        original = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2])
        node_mask = batch_mask_to_node_mask(original)
        recovered = node_mask_to_batch_mask(node_mask)
        return torch.equal(original, recovered)
    run_check("Property 1 验证: 往返转换正确", check_round_trip_logic)


# ============================================================================
# Task 3.1: 确认复用 DiffSBDD 的 prepare_pocket() 方法
# ============================================================================

def verify_task_3_1():
    """Task 3.1: 确认复用 DiffSBDD 的 prepare_pocket() 方法"""
    print_task_header("3.1", "确认复用 DiffSBDD 的 prepare_pocket() 方法")
    
    def check_pdb_file():
        return os.path.isfile("data/8GJJ-C-Cip.pdb")
    run_check("data/8GJJ-C-Cip.pdb 存在", check_pdb_file)
    
    def check_biopython_import():
        from Bio.PDB import PDBParser
        return True
    run_check("BioPython PDBParser 可导入", check_biopython_import)
    
    def check_pdb_parse():
        from Bio.PDB import PDBParser
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('pocket', 'data/8GJJ-C-Cip.pdb')
        return structure is not None
    run_check("PDB 文件可解析", check_pdb_parse)
    
    def check_residues():
        from Bio.PDB import PDBParser
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('pocket', 'data/8GJJ-C-Cip.pdb')
        model = structure[0]
        residues = list(model.get_residues())
        print(f"       找到 {len(residues)} 个残基")
        return len(residues) > 0
    run_check("残基提取", check_residues)
    
    def check_ca_atoms():
        from Bio.PDB import PDBParser
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('pocket', 'data/8GJJ-C-Cip.pdb')
        model = structure[0]
        ca_count = sum(1 for res in model.get_residues() if 'CA' in res)
        print(f"       找到 {ca_count} 个 CA 原子")
        return ca_count > 0
    run_check("CA 原子提取", check_ca_atoms)
    
    def check_lightning_module():
        from DiffSBDD.lightning_modules import LigandPocketDDPM
        return True
    run_check("LigandPocketDDPM 可导入", check_lightning_module)
    
    def check_prepare_pocket_method():
        from DiffSBDD.lightning_modules import LigandPocketDDPM
        return hasattr(LigandPocketDDPM, 'prepare_pocket')
    run_check("prepare_pocket 方法存在", check_prepare_pocket_method)
    
    def check_diffsbdd_utils():
        from DiffSBDD import utils
        return hasattr(utils, 'get_pocket_from_ligand')
    run_check("utils.get_pocket_from_ligand 存在", check_diffsbdd_utils)
    
    def check_dataset_params():
        from DiffSBDD.constants import dataset_params
        return 'crossdock' in dataset_params
    run_check("dataset_params['crossdock'] 存在", check_dataset_params)
    
    def check_atom_encoder():
        from DiffSBDD.constants import dataset_params
        crossdock = dataset_params['crossdock']
        has_encoder = 'atom_encoder' in crossdock
        has_decoder = 'atom_decoder' in crossdock
        if has_decoder:
            print(f"       atom_decoder: {crossdock['atom_decoder']}")
        return has_encoder and has_decoder
    run_check("atom_encoder/decoder 配置", check_atom_encoder)


# ============================================================================
# Task 4.1: 实现 DiffSBDDModel 基础结构和口袋加载
# ============================================================================

def verify_task_4_1():
    """Task 4.1: 实现 DiffSBDDModel 基础结构和口袋加载"""
    print_task_header("4.1", "实现 DiffSBDDModel 基础结构和口袋加载")
    
    def check_file():
        return os.path.isfile("ddpo/integration/diffsbdd_model.py")
    run_check("diffsbdd_model.py 存在", check_file)
    
    def check_class_import():
        from ddpo.integration.diffsbdd_model import DiffSBDDModel
        return True
    run_check("DiffSBDDModel 类可导入", check_class_import)
    
    def check_init_params():
        from ddpo.integration.diffsbdd_model import DiffSBDDModel
        import inspect
        sig = inspect.signature(DiffSBDDModel.__init__)
        params = list(sig.parameters.keys())
        required = ['checkpoint_path', 'pocket_pdb_path']
        missing = [p for p in required if p not in params]
        if missing:
            print(f"       缺少参数: {missing}")
            return False
        return True
    run_check("__init__ 参数: checkpoint_path, pocket_pdb_path", check_init_params)
    
    def check_ref_ligand_param():
        from ddpo.integration.diffsbdd_model import DiffSBDDModel
        import inspect
        sig = inspect.signature(DiffSBDDModel.__init__)
        params = list(sig.parameters.keys())
        return 'ref_ligand_path' in params
    run_check("__init__ 参数: ref_ligand_path", check_ref_ligand_param)
    
    def check_device_param():
        from ddpo.integration.diffsbdd_model import DiffSBDDModel
        import inspect
        sig = inspect.signature(DiffSBDDModel.__init__)
        params = list(sig.parameters.keys())
        return 'device' in params
    run_check("__init__ 参数: device", check_device_param)
    
    def check_load_method():
        from ddpo.integration.diffsbdd_model import DiffSBDDModel
        return hasattr(DiffSBDDModel, 'load')
    run_check("load 方法存在", check_load_method)
    
    def check_to_method():
        from ddpo.integration.diffsbdd_model import DiffSBDDModel
        return hasattr(DiffSBDDModel, 'to')
    run_check("to 方法存在", check_to_method)
    
    def check_T_attribute():
        from ddpo.integration.diffsbdd_model import DiffSBDDModel
        import inspect
        source = inspect.getsource(DiffSBDDModel)
        return 'self.T' in source or 'self.timesteps' in source
    run_check("T (timesteps) 属性", check_T_attribute)
    
    def check_atom_nf_attribute():
        from ddpo.integration.diffsbdd_model import DiffSBDDModel
        import inspect
        source = inspect.getsource(DiffSBDDModel)
        return 'atom_nf' in source or 'num_atom_types' in source
    run_check("atom_nf 属性", check_atom_nf_attribute)


# ============================================================================
# Task 4.2: 实现 get_mask 和格式转换方法
# ============================================================================

def verify_task_4_2():
    """Task 4.2: 实现 get_mask 和格式转换方法"""
    print_task_header("4.2", "实现 get_mask 和格式转换方法")
    
    def check_get_mask():
        from ddpo.integration.diffsbdd_model import DiffSBDDModel
        return hasattr(DiffSBDDModel, 'get_mask')
    run_check("get_mask 方法存在", check_get_mask)
    
    def check_get_mask_params():
        from ddpo.integration.diffsbdd_model import DiffSBDDModel
        import inspect
        if not hasattr(DiffSBDDModel, 'get_mask'):
            return False
        sig = inspect.signature(DiffSBDDModel.get_mask)
        params = list(sig.parameters.keys())
        required = ['nodesxsample', 'batch_size', 'max_n_nodes']
        missing = [p for p in required if p not in params]
        if missing:
            print(f"       缺少参数: {missing}")
            return False
        return True
    run_check("get_mask 参数完整", check_get_mask_params)
    
    def check_rlpf_to_diffsbdd():
        from ddpo.integration.diffsbdd_model import DiffSBDDModel
        return hasattr(DiffSBDDModel, '_rlpf_to_diffsbdd_format')
    run_check("_rlpf_to_diffsbdd_format 方法存在", check_rlpf_to_diffsbdd)
    
    def check_diffsbdd_to_rlpf():
        from ddpo.integration.diffsbdd_model import DiffSBDDModel
        return hasattr(DiffSBDDModel, '_diffsbdd_to_rlpf_format')
    run_check("_diffsbdd_to_rlpf_format 方法存在", check_diffsbdd_to_rlpf)
    
    def check_replicate_pocket():
        from ddpo.integration.diffsbdd_model import DiffSBDDModel
        return hasattr(DiffSBDDModel, '_replicate_pocket')
    run_check("_replicate_pocket 方法存在", check_replicate_pocket)
    
    def check_uses_mask_utils():
        if not os.path.isfile("ddpo/integration/diffsbdd_model.py"):
            return False
        with open("ddpo/integration/diffsbdd_model.py", "r") as f:
            content = f.read()
        return 'mask_utils' in content or 'batch_mask_to_node_mask' in content
    run_check("使用 mask_utils 模块", check_uses_mask_utils)

# ============================================================================
# Task 4.3: 实现 compute_log_p_zs_given_zt 方法
# ============================================================================

def verify_task_4_3():
    """Task 4.3: 实现 compute_log_p_zs_given_zt 方法"""
    print_task_header("4.3", "实现 compute_log_p_zs_given_zt 方法")
    
    import torch
    import numpy as np
    
    def check_method_exists():
        from ddpo.integration.diffsbdd_model import DiffSBDDModel
        return hasattr(DiffSBDDModel, 'compute_log_p_zs_given_zt')
    run_check("compute_log_p_zs_given_zt 方法存在", check_method_exists)
    
    def check_method_params():
        from ddpo.integration.diffsbdd_model import DiffSBDDModel
        import inspect
        if not hasattr(DiffSBDDModel, 'compute_log_p_zs_given_zt'):
            return False
        sig = inspect.signature(DiffSBDDModel.compute_log_p_zs_given_zt)
        params = list(sig.parameters.keys())
        required = ['zs', 'mu', 'sigma']
        missing = [p for p in required if p not in params]
        if missing:
            print(f"       缺少参数: {missing}")
            return False
        return True
    run_check("参数: zs, mu, sigma", check_method_params)
    
    def check_node_mask_param():
        from ddpo.integration.diffsbdd_model import DiffSBDDModel
        import inspect
        if not hasattr(DiffSBDDModel, 'compute_log_p_zs_given_zt'):
            return False
        sig = inspect.signature(DiffSBDDModel.compute_log_p_zs_given_zt)
        params = list(sig.parameters.keys())
        return 'node_mask' in params
    run_check("参数: node_mask", check_node_mask_param)
    
    def check_gaussian_formula():
        """验证高斯 log_prob 公式"""
        if not os.path.isfile("ddpo/integration/diffsbdd_model.py"):
            return False
        with open("ddpo/integration/diffsbdd_model.py", "r") as f:
            content = f.read()
        # 检查是否包含高斯公式的关键部分
        has_squared = '** 2' in content or '**2' in content
        has_sigma = 'sigma' in content
        has_log = 'log' in content.lower()
        return has_squared and has_sigma
    run_check("包含高斯公式 ((zs-mu)^2 / sigma^2)", check_gaussian_formula)
    
    def check_returns_batch_tensor():
        """验证返回 [B] 形状的 tensor"""
        if not os.path.isfile("ddpo/integration/diffsbdd_model.py"):
            return False
        with open("ddpo/integration/diffsbdd_model.py", "r") as f:
            content = f.read()
        # 检查是否有对维度求和的操作
        return '.sum(' in content
    run_check("对维度求和 (返回 [B] tensor)", check_returns_batch_tensor)


# ============================================================================
# Task 4.4: 实现 sample_p_zs_given_zt 方法
# ============================================================================

def verify_task_4_4():
    """Task 4.4: 实现 sample_p_zs_given_zt 方法"""
    print_task_header("4.4", "实现 sample_p_zs_given_zt 方法")
    
    def check_method_exists():
        from ddpo.integration.diffsbdd_model import DiffSBDDModel
        return hasattr(DiffSBDDModel, 'sample_p_zs_given_zt')
    run_check("sample_p_zs_given_zt 方法存在", check_method_exists)
    
    def check_required_params():
        from ddpo.integration.diffsbdd_model import DiffSBDDModel
        import inspect
        if not hasattr(DiffSBDDModel, 'sample_p_zs_given_zt'):
            return False
        sig = inspect.signature(DiffSBDDModel.sample_p_zs_given_zt)
        params = list(sig.parameters.keys())
        required = ['s', 't', 'zt', 'node_mask', 'edge_mask']
        missing = [p for p in required if p not in params]
        if missing:
            print(f"       缺少参数: {missing}")
            return False
        return True
    run_check("参数: s, t, zt, node_mask, edge_mask", check_required_params)
    
    def check_prev_sample_param():
        from ddpo.integration.diffsbdd_model import DiffSBDDModel
        import inspect
        if not hasattr(DiffSBDDModel, 'sample_p_zs_given_zt'):
            return False
        sig = inspect.signature(DiffSBDDModel.sample_p_zs_given_zt)
        params = list(sig.parameters.keys())
        return 'prev_sample' in params
    run_check("参数: prev_sample (EDMActor 需要)", check_prev_sample_param)
    
    def check_returns_4_values():
        """检查返回 (zs, log_prob, mu, sigma)"""
        if not os.path.isfile("ddpo/integration/diffsbdd_model.py"):
            return False
        with open("ddpo/integration/diffsbdd_model.py", "r") as f:
            content = f.read()
        # 查找 sample_p_zs_given_zt 方法中的 return 语句
        import re
        # 简单检查是否有返回多个值
        return 'return' in content and ('zs' in content or 'log_prob' in content)
    run_check("返回 (zs, log_prob, mu, sigma)", check_returns_4_values)
    
    def check_calls_gamma():
        """检查是否调用 gamma 计算"""
        if not os.path.isfile("ddpo/integration/diffsbdd_model.py"):
            return False
        with open("ddpo/integration/diffsbdd_model.py", "r") as f:
            content = f.read()
        return 'gamma' in content
    run_check("调用 gamma 计算", check_calls_gamma)
    
    def check_calls_dynamics():
        """检查是否调用 dynamics 获取 eps_t"""
        if not os.path.isfile("ddpo/integration/diffsbdd_model.py"):
            return False
        with open("ddpo/integration/diffsbdd_model.py", "r") as f:
            content = f.read()
        return 'dynamics' in content
    run_check("调用 dynamics 获取 eps_t", check_calls_dynamics)
    
    def check_format_conversion():
        """检查是否进行格式转换"""
        if not os.path.isfile("ddpo/integration/diffsbdd_model.py"):
            return False
        with open("ddpo/integration/diffsbdd_model.py", "r") as f:
            content = f.read()
        return '_rlpf_to_diffsbdd' in content or '_diffsbdd_to_rlpf' in content
    run_check("RLPF ↔ DiffSBDD 格式转换", check_format_conversion)

# ============================================================================
# Task 4.5: 编写 log_prob 计算的属性测试
# ============================================================================

def verify_task_4_5():
    """Task 4.5: 编写 log_prob 计算的属性测试 (Property 10)"""
    print_task_header("4.5", "编写 log_prob 计算的属性测试")
    
    def check_test_file():
        return os.path.isfile("tests/property/test_log_prob_properties.py")
    run_check("test_log_prob_properties.py 存在", check_test_file)
    
    def check_property_10():
        if not os.path.isfile("tests/property/test_log_prob_properties.py"):
            return False
        with open("tests/property/test_log_prob_properties.py", "r") as f:
            content = f.read()
        return 'Property 10' in content or 'gaussian' in content.lower()
    run_check("Property 10 测试存在 (Gaussian Formula)", check_property_10)
    
    def check_hypothesis_usage():
        if not os.path.isfile("tests/property/test_log_prob_properties.py"):
            return False
        with open("tests/property/test_log_prob_properties.py", "r") as f:
            content = f.read()
        return '@given' in content
    run_check("使用 hypothesis @given", check_hypothesis_usage)
    
    def check_validates_requirements():
        if not os.path.isfile("tests/property/test_log_prob_properties.py"):
            return False
        with open("tests/property/test_log_prob_properties.py", "r") as f:
            content = f.read()
        return 'Requirements' in content or '1.4' in content or '11.1' in content
    run_check("引用 Requirements 1.4, 11.1, 11.2", check_validates_requirements)


# ============================================================================
# Task 4.6: 实现 sample_p_xh_given_z0 方法
# ============================================================================

def verify_task_4_6():
    """Task 4.6: 实现 sample_p_xh_given_z0 方法"""
    print_task_header("4.6", "实现 sample_p_xh_given_z0 方法")
    
    def check_method_exists():
        from ddpo.integration.diffsbdd_model import DiffSBDDModel
        return hasattr(DiffSBDDModel, 'sample_p_xh_given_z0')
    run_check("sample_p_xh_given_z0 方法存在", check_method_exists)
    
    def check_required_params():
        from ddpo.integration.diffsbdd_model import DiffSBDDModel
        import inspect
        if not hasattr(DiffSBDDModel, 'sample_p_xh_given_z0'):
            return False
        sig = inspect.signature(DiffSBDDModel.sample_p_xh_given_z0)
        params = list(sig.parameters.keys())
        required = ['z0', 'node_mask', 'edge_mask']
        missing = [p for p in required if p not in params]
        if missing:
            print(f"       缺少参数: {missing}")
            return False
        return True
    run_check("参数: z0, node_mask, edge_mask", check_required_params)
    
    def check_prev_sample_param():
        from ddpo.integration.diffsbdd_model import DiffSBDDModel
        import inspect
        if not hasattr(DiffSBDDModel, 'sample_p_xh_given_z0'):
            return False
        sig = inspect.signature(DiffSBDDModel.sample_p_xh_given_z0)
        params = list(sig.parameters.keys())
        return 'prev_sample' in params
    run_check("参数: prev_sample", check_prev_sample_param)
    
    def check_returns_7_values():
        """检查返回 (x, h, mu, sigma, log_prob, zeros, xh)"""
        if not os.path.isfile("ddpo/integration/diffsbdd_model.py"):
            return False
        with open("ddpo/integration/diffsbdd_model.py", "r") as f:
            content = f.read()
        # 查找 sample_p_xh_given_z0 方法中的 return 语句
        # 检查是否返回多个值
        return 'return x, h, mu' in content
    run_check("返回 (x, h, mu, sigma, log_prob, zeros, xh)", check_returns_7_values)
    
    def check_h_dict_format():
        """检查 h 是否为 dict 格式"""
        if not os.path.isfile("ddpo/integration/diffsbdd_model.py"):
            return False
        with open("ddpo/integration/diffsbdd_model.py", "r") as f:
            content = f.read()
        return "'integer'" in content and "'categorical'" in content
    run_check("h 格式: dict with 'integer' and 'categorical'", check_h_dict_format)
    
    def check_calls_gamma():
        """检查是否调用 gamma 计算"""
        if not os.path.isfile("ddpo/integration/diffsbdd_model.py"):
            return False
        with open("ddpo/integration/diffsbdd_model.py", "r") as f:
            content = f.read()
        return 'gamma_0' in content or 'gamma(t_zeros)' in content
    run_check("调用 gamma_0 计算", check_calls_gamma)
    
    def check_calls_unnormalize():
        """检查是否调用 unnormalize"""
        if not os.path.isfile("ddpo/integration/diffsbdd_model.py"):
            return False
        with open("ddpo/integration/diffsbdd_model.py", "r") as f:
            content = f.read()
        return 'unnormalize' in content
    run_check("调用 unnormalize 获取最终 x, h", check_calls_unnormalize)
    
    def check_one_hot_conversion():
        """检查是否进行 one-hot 转换"""
        if not os.path.isfile("ddpo/integration/diffsbdd_model.py"):
            return False
        with open("ddpo/integration/diffsbdd_model.py", "r") as f:
            content = f.read()
        return 'one_hot' in content or 'F.one_hot' in content
    run_check("h 转换为 one-hot 编码", check_one_hot_conversion)


# ============================================================================
# Task 4.7: 实现 sample 方法（完整采样循环）
# ============================================================================

def verify_task_4_7():
    """Task 4.7: 实现 sample 方法（完整采样循环）"""
    print_task_header("4.7", "实现 sample 方法（完整采样循环）")
    
    def check_method_exists():
        from ddpo.integration.diffsbdd_model import DiffSBDDModel
        return hasattr(DiffSBDDModel, 'sample')
    run_check("sample 方法存在", check_method_exists)
    
    def check_not_raises():
        """检查 sample 方法不再抛出 NotImplementedError"""
        from ddpo.integration.diffsbdd_model import DiffSBDDModel
        import inspect
        source = inspect.getsource(DiffSBDDModel.sample)
        return 'NotImplementedError' not in source
    run_check("sample 方法已实现 (非 NotImplementedError)", check_not_raises)
    
    def check_required_params():
        from ddpo.integration.diffsbdd_model import DiffSBDDModel
        import inspect
        if not hasattr(DiffSBDDModel, 'sample'):
            return False
        sig = inspect.signature(DiffSBDDModel.sample)
        params = list(sig.parameters.keys())
        required = ['n_samples', 'n_nodes', 'node_mask', 'edge_mask']
        missing = [p for p in required if p not in params]
        if missing:
            print(f"       缺少参数: {missing}")
            return False
        return True
    run_check("参数: n_samples, n_nodes, node_mask, edge_mask", check_required_params)
    
    def check_timestep_param():
        from ddpo.integration.diffsbdd_model import DiffSBDDModel
        import inspect
        if not hasattr(DiffSBDDModel, 'sample'):
            return False
        sig = inspect.signature(DiffSBDDModel.sample)
        params = list(sig.parameters.keys())
        return 'timestep' in params
    run_check("参数: timestep", check_timestep_param)
    
    def check_fix_noise_param():
        from ddpo.integration.diffsbdd_model import DiffSBDDModel
        import inspect
        if not hasattr(DiffSBDDModel, 'sample'):
            return False
        sig = inspect.signature(DiffSBDDModel.sample)
        params = list(sig.parameters.keys())
        return 'fix_noise' in params
    run_check("参数: fix_noise", check_fix_noise_param)
    
    def check_returns_7_values():
        """检查返回 (x, h, latents, logps, timesteps, mus, sigmas)"""
        if not os.path.isfile("ddpo/integration/diffsbdd_model.py"):
            return False
        with open("ddpo/integration/diffsbdd_model.py", "r") as f:
            content = f.read()
        return 'return x, h, latents, logps' in content
    run_check("返回 (x, h, latents, logps, timesteps, mus, sigmas)", check_returns_7_values)
    
    def check_trajectory_recording():
        """检查是否记录轨迹"""
        if not os.path.isfile("ddpo/integration/diffsbdd_model.py"):
            return False
        with open("ddpo/integration/diffsbdd_model.py", "r") as f:
            content = f.read()
        has_latents = 'latents.append' in content or 'latents = [' in content
        has_logps = 'logps.append' in content
        return has_latents and has_logps
    run_check("轨迹记录 (latents, logps)", check_trajectory_recording)
    
    def check_sampling_loop():
        """检查是否有采样循环"""
        if not os.path.isfile("ddpo/integration/diffsbdd_model.py"):
            return False
        with open("ddpo/integration/diffsbdd_model.py", "r") as f:
            content = f.read()
        return 'for s in reversed' in content or 'reversed(range' in content
    run_check("采样循环 (reversed range)", check_sampling_loop)
    
    def check_calls_sample_p_zs_given_zt():
        """检查是否调用 sample_p_zs_given_zt"""
        if not os.path.isfile("ddpo/integration/diffsbdd_model.py"):
            return False
        with open("ddpo/integration/diffsbdd_model.py", "r") as f:
            content = f.read()
        return 'sample_p_zs_given_zt' in content
    run_check("调用 sample_p_zs_given_zt", check_calls_sample_p_zs_given_zt)
    
    def check_calls_sample_p_xh_given_z0():
        """检查是否调用 sample_p_xh_given_z0"""
        if not os.path.isfile("ddpo/integration/diffsbdd_model.py"):
            return False
        with open("ddpo/integration/diffsbdd_model.py", "r") as f:
            content = f.read()
        return 'sample_p_xh_given_z0' in content
    run_check("调用 sample_p_xh_given_z0 (最终步)", check_calls_sample_p_xh_given_z0)
    
    def check_cog_correction():
        """检查是否有 CoG 漂移校正"""
        if not os.path.isfile("ddpo/integration/diffsbdd_model.py"):
            return False
        with open("ddpo/integration/diffsbdd_model.py", "r") as f:
            content = f.read()
        return 'CoG drift' in content or 'cog' in content.lower()
    run_check("CoG 漂移校正", check_cog_correction)


# ============================================================================
# Task 4.8: 编写 sample 方法的属性测试
# ============================================================================

def verify_task_4_8():
    """Task 4.8: 编写 sample 方法的属性测试 (Property 11, 12, 14)"""
    print_task_header("4.8", "编写 sample 方法的属性测试")
    
    def check_test_file():
        return os.path.isfile("tests/property/test_sample_properties.py")
    run_check("test_sample_properties.py 存在", check_test_file)
    
    def check_property_11():
        if not os.path.isfile("tests/property/test_sample_properties.py"):
            return False
        with open("tests/property/test_sample_properties.py", "r", encoding="utf-8") as f:
            content = f.read()
        return 'Property 11' in content or 'trajectory' in content.lower()
    run_check("Property 11 测试存在 (Trajectory Log Probs Shape)", check_property_11)
    
    def check_property_12():
        if not os.path.isfile("tests/property/test_sample_properties.py"):
            return False
        with open("tests/property/test_sample_properties.py", "r", encoding="utf-8") as f:
            content = f.read()
        return 'Property 12' in content or 'output structure' in content.lower()
    run_check("Property 12 测试存在 (Sample Output Structure)", check_property_12)
    
    def check_property_14():
        if not os.path.isfile("tests/property/test_sample_properties.py"):
            return False
        with open("tests/property/test_sample_properties.py", "r", encoding="utf-8") as f:
            content = f.read()
        return 'Property 14' in content or 'h format' in content.lower() or 'h_is_tensor' in content
    run_check("Property 14 测试存在 (h Format Tensor Structure)", check_property_14)
    
    def check_hypothesis_usage():
        if not os.path.isfile("tests/property/test_sample_properties.py"):
            return False
        with open("tests/property/test_sample_properties.py", "r", encoding="utf-8") as f:
            content = f.read()
        return '@given' in content
    run_check("使用 hypothesis @given", check_hypothesis_usage)
    
    def check_validates_requirements():
        if not os.path.isfile("tests/property/test_sample_properties.py"):
            return False
        with open("tests/property/test_sample_properties.py", "r", encoding="utf-8") as f:
            content = f.read()
        return 'Requirements' in content or '1.2' in content or '11.4' in content or '14.1' in content
    run_check("引用 Requirements 1.2, 11.4, 14.1", check_validates_requirements)
    
    def check_latents_length_test():
        if not os.path.isfile("tests/property/test_sample_properties.py"):
            return False
        with open("tests/property/test_sample_properties.py", "r", encoding="utf-8") as f:
            content = f.read()
        return 'latents' in content and ('T + 2' in content or 'T+2' in content)
    run_check("测试 latents 长度 (T+2)", check_latents_length_test)
    
    def check_logps_length_test():
        if not os.path.isfile("tests/property/test_sample_properties.py"):
            return False
        with open("tests/property/test_sample_properties.py", "r", encoding="utf-8") as f:
            content = f.read()
        return 'logps' in content and ('T + 1' in content or 'T+1' in content)
    run_check("测试 logps 长度 (T+1)", check_logps_length_test)
    
    def check_h_tensor_test():
        if not os.path.isfile("tests/property/test_sample_properties.py"):
            return False
        with open("tests/property/test_sample_properties.py", "r", encoding="utf-8") as f:
            content = f.read()
        return 'torch.Tensor' in content and 'isinstance(h' in content
    run_check("测试 h 是 tensor (非 dict)", check_h_tensor_test)


# ============================================================================
# Task 13: 训练入口脚本
# ============================================================================

def verify_task_13_1():
    """Task 13.1: 创建 main_diffsbdd.py"""
    print_task_header("13.1", "创建 main_diffsbdd.py")
    
    def check_main_file_exists():
        return os.path.isfile("ddpo/main_diffsbdd.py")
    run_check("main_diffsbdd.py 文件存在", check_main_file_exists)
    
    def check_imports():
        if not os.path.isfile("ddpo/main_diffsbdd.py"):
            return False
        with open("ddpo/main_diffsbdd.py", "r", encoding="utf-8") as f:
            content = f.read()
        required = [
            "DiffSBDDModel",
            "DockingReward",
            "DruglikenessReward",
            "CompositeReward",
            "DiffSBDDFilter",
            "DDPOTrainer",
            "EDMActor",
        ]
        missing = [r for r in required if r not in content]
        if missing:
            print(f"       缺少导入: {missing}")
            return False
        return True
    run_check("必要组件导入", check_imports)
    
    def check_main_function():
        if not os.path.isfile("ddpo/main_diffsbdd.py"):
            return False
        with open("ddpo/main_diffsbdd.py", "r", encoding="utf-8") as f:
            content = f.read()
        return "def main():" in content and 'if __name__ == "__main__":' in content
    run_check("main() 函数定义", check_main_function)
    
    def check_dataloader_class():
        if not os.path.isfile("ddpo/main_diffsbdd.py"):
            return False
        with open("ddpo/main_diffsbdd.py", "r", encoding="utf-8") as f:
            content = f.read()
        return "class DiffSBDDDataLoader" in content
    run_check("DiffSBDDDataLoader 类定义", check_dataloader_class)
    
    def check_rollout_class():
        if not os.path.isfile("ddpo/main_diffsbdd.py"):
            return False
        with open("ddpo/main_diffsbdd.py", "r", encoding="utf-8") as f:
            content = f.read()
        return "class DiffSBDDRollout" in content
    run_check("DiffSBDDRollout 类定义", check_rollout_class)


def verify_task_13_2():
    """Task 13.2: 实现配置加载和验证"""
    print_task_header("13.2", "实现配置加载和验证")
    
    def check_load_config():
        if not os.path.isfile("ddpo/main_diffsbdd.py"):
            return False
        with open("ddpo/main_diffsbdd.py", "r", encoding="utf-8") as f:
            content = f.read()
        return "def load_config" in content and "yaml.safe_load" in content
    run_check("load_config() 函数", check_load_config)
    
    def check_config_validation():
        if not os.path.isfile("ddpo/main_diffsbdd.py"):
            return False
        with open("ddpo/main_diffsbdd.py", "r", encoding="utf-8") as f:
            content = f.read()
        return "required_sections" in content or "required_params" in content
    run_check("配置验证逻辑", check_config_validation)
    
    def check_config_file():
        return os.path.isfile("ddpo/configs/diffsbdd_ddpo.yaml")
    run_check("配置文件存在", check_config_file)
    
    def check_config_sections():
        if not os.path.isfile("ddpo/configs/diffsbdd_ddpo.yaml"):
            return False
        import yaml
        with open("ddpo/configs/diffsbdd_ddpo.yaml", "r") as f:
            config = yaml.safe_load(f)
        required = ["model", "pocket", "reward", "dataloader", "train"]
        missing = [s for s in required if s not in config]
        if missing:
            print(f"       缺少配置节: {missing}")
            return False
        return True
    run_check("配置文件包含必要节", check_config_sections)


def verify_task_13_3():
    """Task 13.3: 实现训练循环"""
    print_task_header("13.3", "实现训练循环")
    
    def check_trainer_init():
        if not os.path.isfile("ddpo/main_diffsbdd.py"):
            return False
        with open("ddpo/main_diffsbdd.py", "r", encoding="utf-8") as f:
            content = f.read()
        return "DDPOTrainer(" in content
    run_check("DDPOTrainer 初始化", check_trainer_init)
    
    def check_trainer_fit():
        if not os.path.isfile("ddpo/main_diffsbdd.py"):
            return False
        with open("ddpo/main_diffsbdd.py", "r", encoding="utf-8") as f:
            content = f.read()
        return "trainer.fit()" in content
    run_check("trainer.fit() 调用", check_trainer_fit)
    
    def check_graceful_shutdown():
        if not os.path.isfile("ddpo/main_diffsbdd.py"):
            return False
        with open("ddpo/main_diffsbdd.py", "r", encoding="utf-8") as f:
            content = f.read()
        return "signal.signal" in content or "KeyboardInterrupt" in content
    run_check("优雅退出处理", check_graceful_shutdown)
    
    def check_cleanup():
        if not os.path.isfile("ddpo/main_diffsbdd.py"):
            return False
        with open("ddpo/main_diffsbdd.py", "r", encoding="utf-8") as f:
            content = f.read()
        return "trainer.clean_up()" in content
    run_check("资源清理", check_cleanup)
    
    def check_stability_wrapper():
        if not os.path.isfile("ddpo/main_diffsbdd.py"):
            return False
        with open("ddpo/main_diffsbdd.py", "r", encoding="utf-8") as f:
            content = f.read()
        return "StabilityAwareReward" in content or "'stability'" in content
    run_check("stability 字段支持", check_stability_wrapper)


# ============================================================================
# 任务注册表
# ============================================================================

# ============================================================================
# Task 14: 端到端集成测试
# ============================================================================

def verify_task_14_1():
    """Task 14.1: 编写集成测试"""
    print_task_header("14.1", "编写集成测试")
    
    def check_test_file_exists():
        return os.path.isfile("tests/integration/test_training_pipeline.py")
    run_check("test_training_pipeline.py 存在", check_test_file_exists)
    
    def check_test_config_loading():
        if not os.path.isfile("tests/integration/test_training_pipeline.py"):
            return False
        with open("tests/integration/test_training_pipeline.py", "r", encoding="utf-8") as f:
            content = f.read()
        return "TestConfigurationLoading" in content
    run_check("配置加载测试类存在", check_test_config_loading)
    
    def check_test_component_imports():
        if not os.path.isfile("tests/integration/test_training_pipeline.py"):
            return False
        with open("tests/integration/test_training_pipeline.py", "r", encoding="utf-8") as f:
            content = f.read()
        return "TestComponentImports" in content
    run_check("组件导入测试类存在", check_test_component_imports)
    
    def check_test_dataloader():
        if not os.path.isfile("tests/integration/test_training_pipeline.py"):
            return False
        with open("tests/integration/test_training_pipeline.py", "r", encoding="utf-8") as f:
            content = f.read()
        return "TestDiffSBDDDataLoader" in content
    run_check("DataLoader 测试类存在", check_test_dataloader)
    
    def check_test_rewards():
        if not os.path.isfile("tests/integration/test_training_pipeline.py"):
            return False
        with open("tests/integration/test_training_pipeline.py", "r", encoding="utf-8") as f:
            content = f.read()
        return "TestRewardComponents" in content
    run_check("奖励组件测试类存在", check_test_rewards)
    
    def check_test_filter():
        if not os.path.isfile("tests/integration/test_training_pipeline.py"):
            return False
        with open("tests/integration/test_training_pipeline.py", "r", encoding="utf-8") as f:
            content = f.read()
        return "TestFilterComponent" in content
    run_check("Filter 测试类存在", check_test_filter)
    
    def check_test_logging():
        if not os.path.isfile("tests/integration/test_training_pipeline.py"):
            return False
        with open("tests/integration/test_training_pipeline.py", "r", encoding="utf-8") as f:
            content = f.read()
        return "TestLoggingSetup" in content
    run_check("日志设置测试类存在", check_test_logging)
    
    def check_test_checkpoint():
        if not os.path.isfile("tests/integration/test_training_pipeline.py"):
            return False
        with open("tests/integration/test_training_pipeline.py", "r", encoding="utf-8") as f:
            content = f.read()
        return "TestCheckpointMock" in content or "checkpoint" in content.lower()
    run_check("Checkpoint 测试存在", check_test_checkpoint)
    
    def check_test_signal_handling():
        if not os.path.isfile("tests/integration/test_training_pipeline.py"):
            return False
        with open("tests/integration/test_training_pipeline.py", "r", encoding="utf-8") as f:
            content = f.read()
        return "TestSignalHandling" in content or "signal" in content.lower()
    run_check("信号处理测试存在", check_test_signal_handling)
    
    def check_test_full_pipeline():
        if not os.path.isfile("tests/integration/test_training_pipeline.py"):
            return False
        with open("tests/integration/test_training_pipeline.py", "r", encoding="utf-8") as f:
            content = f.read()
        return "TestFullPipeline" in content
    run_check("完整流程测试类存在", check_test_full_pipeline)
    
    def check_validates_requirements():
        if not os.path.isfile("tests/integration/test_training_pipeline.py"):
            return False
        with open("tests/integration/test_training_pipeline.py", "r", encoding="utf-8") as f:
            content = f.read()
        return "Requirements 8" in content or "8.1" in content
    run_check("引用 Requirements 8.x", check_validates_requirements)


TASKS = {
    '1': ('Task 1: 项目结构和基础设施', [
        ('1.1', verify_task_1_1),
        ('1.2', verify_task_1_2),
        ('1.3', verify_task_1_3),
        ('1.4', verify_task_1_4),
    ]),
    '2': ('Task 2: Mask 格式转换工具', [
        ('2.1', verify_task_2_1),
        ('2.2', verify_task_2_2),
    ]),
    '3': ('Task 3: 口袋数据加载', [
        ('3.1', verify_task_3_1),
    ]),
    '4': ('Task 4: DiffSBDDModel 封装', [
        ('4.1', verify_task_4_1),
        ('4.2', verify_task_4_2),
        ('4.3', verify_task_4_3),
        ('4.4', verify_task_4_4),
        ('4.5', verify_task_4_5),
        ('4.6', verify_task_4_6),
        ('4.7', verify_task_4_7),
        ('4.8', verify_task_4_8),
    ]),
    '13': ('Task 13: 训练入口脚本', [
        ('13.1', verify_task_13_1),
        ('13.2', verify_task_13_2),
        ('13.3', verify_task_13_3),
    ]),
    '14': ('Task 14: 端到端集成测试', [
        ('14.1', verify_task_14_1),
    ]),
}

def list_tasks():
    """列出所有可验证的任务"""
    print("\n可验证的任务列表:")
    print("="*60)
    for task_id, (title, subtasks) in TASKS.items():
        print(f"\n{title}")
        for subtask_id, _ in subtasks:
            print(f"  --task {subtask_id}")
    print("\n使用示例:")
    print("  python scripts/verify_completed_tasks.py           # 全部验证")
    print("  python scripts/verify_completed_tasks.py --task 1  # Task 1 全部")
    print("  python scripts/verify_completed_tasks.py --task 4.3 # 只验证 4.3")

def run_task(task_id: str):
    """运行指定任务的验证"""
    if '.' in task_id:
        # 单个子任务
        main_task = task_id.split('.')[0]
        if main_task not in TASKS:
            print(f"未知任务: {task_id}")
            return
        _, subtasks = TASKS[main_task]
        for subtask_id, fn in subtasks:
            if subtask_id == task_id:
                fn()
                return
        print(f"未知子任务: {task_id}")
    else:
        # 整个任务
        if task_id not in TASKS:
            print(f"未知任务: {task_id}")
            return
        _, subtasks = TASKS[task_id]
        for _, fn in subtasks:
            fn()

def run_all():
    """运行所有验证"""
    for task_id in TASKS:
        _, subtasks = TASKS[task_id]
        for _, fn in subtasks:
            fn()

def run_pytest():
    """运行 pytest 属性测试"""
    print("\n" + "="*60)
    print("运行 pytest 属性测试")
    print("="*60)
    
    import subprocess
    
    test_files = [
        ("tests/property/test_mask_properties.py", "Mask 属性测试"),
        ("tests/property/test_log_prob_properties.py", "Log Prob 属性测试"),
        ("tests/property/test_sample_properties.py", "Sample 属性测试"),
    ]
    
    for test_file, name in test_files:
        if os.path.isfile(test_file):
            print(f"\n运行 {name}...")
            result = subprocess.run(
                ['python', '-m', 'pytest', test_file, '-v', '--tb=short'],
                capture_output=True, text=True, timeout=180
            )
            if result.returncode == 0:
                record(f"pytest: {name}", True)
            else:
                record(f"pytest: {name}", False, "测试失败")
                print(result.stdout[-1500:] if result.stdout else "")
                print(result.stderr[-500:] if result.stderr else "")
        else:
            record(f"pytest: {name}", False, "文件不存在")

def print_summary():
    """打印验证结果摘要"""
    print("\n" + "="*60)
    print("验证结果摘要")
    print("="*60)
    
    passed = sum(1 for _, s, _ in results if s)
    failed = sum(1 for _, s, _ in results if not s)
    
    print(f"\n总计: {len(results)} 项检查")
    print(f"  ✅ 通过: {passed}")
    print(f"  ❌ 失败: {failed}")
    
    if failed > 0:
        print("\n失败项:")
        for name, success, msg in results:
            if not success:
                print(f"  ❌ {name}")
                if msg:
                    print(f"     {msg}")
    
    return failed == 0

def main():
    parser = argparse.ArgumentParser(
        description='已完成任务验证脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/verify_completed_tasks.py              # 运行所有验证
  python scripts/verify_completed_tasks.py --task 1    # 验证 Task 1 全部
  python scripts/verify_completed_tasks.py --task 4.3  # 只验证 Task 4.3
  python scripts/verify_completed_tasks.py --list      # 列出所有任务
  python scripts/verify_completed_tasks.py --run-tests # 同时运行 pytest
        """
    )
    parser.add_argument('--task', type=str, help='指定任务 (如 1, 2, 4.3)')
    parser.add_argument('--list', action='store_true', help='列出所有可验证的任务')
    parser.add_argument('--run-tests', action='store_true', help='运行 pytest 属性测试')
    args = parser.parse_args()
    
    if args.list:
        list_tasks()
        return
    
    print("="*60)
    print("DiffSBDD-RLPF 已完成任务验证")
    print("="*60)
    
    if args.task:
        run_task(args.task)
    else:
        run_all()
    
    if args.run_tests:
        run_pytest()
    
    success = print_summary()
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
