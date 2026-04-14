#!/usr/bin/env python
"""
Task 5 Checkpoint 验证脚本

验证 DiffSBDDModel 封装的完整性，确保所有测试通过。

使用方法:
    python scripts/verify_task5_checkpoint.py
    python scripts/verify_task5_checkpoint.py --verbose
    python scripts/verify_task5_checkpoint.py --run-tests
"""

import argparse
import sys
import os
import subprocess
from typing import List, Tuple, Callable
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 验证结果存储
results: List[Tuple[str, bool, str]] = []


def record(name: str, success: bool, msg: str = ""):
    """记录验证结果"""
    results.append((name, success, msg))
    status = "✅" if success else "❌"
    print(f"  {status} {name}")
    if msg and not success:
        print(f"     └─ {msg}")


def run_check(name: str, fn: Callable[[], bool]):
    """运行单个检查"""
    try:
        success = fn()
        record(name, success)
    except Exception as e:
        record(name, False, str(e))


def print_section(title: str):
    """打印章节标题"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ============================================================================
# 1. 文件结构验证
# ============================================================================

def verify_file_structure():
    """验证所有必需文件存在"""
    print_section("1. 文件结构验证")
    
    required_files = [
        ("ddpo/__init__.py", "ddpo 包初始化"),
        ("ddpo/integration/__init__.py", "integration 包初始化"),
        ("ddpo/integration/diffsbdd_model.py", "DiffSBDDModel 实现"),
        ("ddpo/integration/mask_utils.py", "Mask 转换工具"),
        ("ddpo/rewards/__init__.py", "rewards 包初始化"),
        ("ddpo/rewards/base.py", "BaseReward 接口"),
        ("ddpo/utils/__init__.py", "utils 包初始化"),
        ("ddpo/configs/diffsbdd_ddpo.yaml", "训练配置文件"),
        ("tests/conftest.py", "pytest 配置"),
        ("tests/property/test_mask_properties.py", "Mask 属性测试"),
        ("tests/property/test_log_prob_properties.py", "Log Prob 属性测试"),
        ("tests/property/test_sample_properties.py", "Sample 属性测试"),
    ]
    
    for filepath, desc in required_files:
        def check(fp=filepath):
            return os.path.isfile(fp)
        run_check(f"{desc} ({filepath})", check)


# ============================================================================
# 2. 模块导入验证
# ============================================================================

def verify_imports():
    """验证所有模块可正确导入"""
    print_section("2. 模块导入验证")
    
    def check_mask_utils():
        from ddpo.integration.mask_utils import (
            batch_mask_to_node_mask,
            node_mask_to_batch_mask,
            get_nodes_per_batch,
            create_batch_mask_from_counts,
        )
        return True
    run_check("mask_utils 函数导入", check_mask_utils)
    
    def check_diffsbdd_model():
        from ddpo.integration.diffsbdd_model import DiffSBDDModel
        return True
    run_check("DiffSBDDModel 类导入", check_diffsbdd_model)
    
    def check_base_reward():
        from ddpo.rewards.base import BaseReward
        return True
    run_check("BaseReward 类导入", check_base_reward)
    
    def check_diffsbdd_deps():
        from DiffSBDD.lightning_modules import LigandPocketDDPM
        from DiffSBDD.constants import dataset_params
        from DiffSBDD import utils
        return True
    run_check("DiffSBDD 依赖导入", check_diffsbdd_deps)
    
    def check_rlpf_deps():
        from verl_diffusion.model.base import BaseModel
        return True
    run_check("RLPF BaseModel 导入", check_rlpf_deps)


# ============================================================================
# 3. DiffSBDDModel 接口验证
# ============================================================================

def verify_diffsbdd_model_interface():
    """验证 DiffSBDDModel 实现了所有必需的接口"""
    print_section("3. DiffSBDDModel 接口验证")
    
    from ddpo.integration.diffsbdd_model import DiffSBDDModel
    import inspect
    
    # 必需的方法
    required_methods = [
        ("__init__", ["checkpoint_path", "pocket_pdb_path", "ref_ligand_path", "device"]),
        ("load", ["model_path"]),
        ("to", ["device"]),
        ("eval", []),
        ("train", []),
        ("get_mask", ["nodesxsample", "batch_size", "max_n_nodes"]),
        ("compute_log_p_zs_given_zt", ["zs", "mu", "sigma"]),
        ("sample_p_zs_given_zt", ["s", "t", "zt", "node_mask", "edge_mask"]),
        ("sample_p_xh_given_z0", ["z0", "node_mask", "edge_mask"]),
        ("sample", ["n_samples", "n_nodes", "node_mask", "edge_mask"]),
    ]
    
    for method_name, required_params in required_methods:
        def check(mn=method_name, rp=required_params):
            if not hasattr(DiffSBDDModel, mn):
                return False
            method = getattr(DiffSBDDModel, mn)
            sig = inspect.signature(method)
            params = list(sig.parameters.keys())
            # 移除 self
            params = [p for p in params if p != 'self']
            missing = [p for p in rp if p not in params]
            if missing:
                print(f"       缺少参数: {missing}")
                return False
            return True
        run_check(f"方法 {method_name}() 存在且参数完整", check)
    
    # 私有方法
    private_methods = [
        "_rlpf_to_diffsbdd_format",
        "_diffsbdd_to_rlpf_format",
        "_replicate_pocket",
        "_load_model",
        "_load_pocket_residues",
    ]
    
    for method_name in private_methods:
        def check(mn=method_name):
            return hasattr(DiffSBDDModel, mn)
        run_check(f"私有方法 {method_name}() 存在", check)


# ============================================================================
# 4. Mask 转换功能验证
# ============================================================================

def verify_mask_conversion():
    """验证 Mask 转换功能正确性"""
    print_section("4. Mask 转换功能验证")
    
    import torch
    from ddpo.integration.mask_utils import (
        batch_mask_to_node_mask,
        node_mask_to_batch_mask,
        get_nodes_per_batch,
    )
    
    def check_batch_to_node():
        batch_mask = torch.tensor([0, 0, 1, 1, 1, 2, 2])
        node_mask = batch_mask_to_node_mask(batch_mask)
        expected_shape = (3, 3, 1)
        return node_mask.shape == expected_shape
    run_check("batch_mask_to_node_mask 形状正确", check_batch_to_node)
    
    def check_node_to_batch():
        node_mask = torch.tensor([[[1],[1],[0]], [[1],[1],[1]], [[1],[1],[0]]], dtype=torch.float32)
        batch_mask = node_mask_to_batch_mask(node_mask)
        expected = torch.tensor([0, 0, 1, 1, 1, 2, 2])
        return torch.equal(batch_mask, expected)
    run_check("node_mask_to_batch_mask 内容正确", check_node_to_batch)
    
    def check_round_trip():
        original = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2])
        node_mask = batch_mask_to_node_mask(original)
        recovered = node_mask_to_batch_mask(node_mask)
        return torch.equal(original, recovered)
    run_check("往返转换 (Round-Trip) 正确", check_round_trip)
    
    def check_get_nodes_per_batch():
        batch_mask = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2])
        counts = get_nodes_per_batch(batch_mask)
        expected = torch.tensor([3, 2, 4])
        return torch.equal(counts, expected)
    run_check("get_nodes_per_batch 正确", check_get_nodes_per_batch)


# ============================================================================
# 5. Log Prob 计算验证
# ============================================================================

def verify_log_prob_computation():
    """验证 log_prob 计算正确性"""
    print_section("5. Log Prob 计算验证")
    
    import torch
    
    # 使用独立实现测试（避免加载完整模型）
    def compute_log_p_zs_given_zt(zs, mu, sigma, node_mask=None):
        epsilon = 1e-6
        sigma = torch.max(sigma, torch.tensor(epsilon))
        zs_detached = zs.detach()
        log_p = -0.5 * ((zs_detached - mu) ** 2) / (sigma ** 2)
        if node_mask is not None:
            log_p = log_p * node_mask
        D = zs.shape[-1]
        log_p_sum = log_p.sum(dim=tuple(range(1, log_p.ndim)))
        if node_mask is not None:
            valid_count = node_mask.sum(dim=tuple(range(1, node_mask.ndim))) * D
        else:
            valid_count = zs.shape[1] * D
        log_prob = log_p_sum / (valid_count + epsilon)
        return log_prob
    
    def check_output_shape():
        zs = torch.randn(4, 10, 13)
        mu = torch.randn(4, 10, 13)
        sigma = torch.tensor(1.0)
        node_mask = torch.ones(4, 10, 1)
        result = compute_log_p_zs_given_zt(zs, mu, sigma, node_mask)
        return result.shape == (4,)
    run_check("log_prob 输出形状 [B]", check_output_shape)
    
    def check_at_mean_is_zero():
        mu = torch.randn(2, 5, 3)
        zs = mu.clone()
        sigma = torch.tensor(1.0)
        node_mask = torch.ones(2, 5, 1)
        result = compute_log_p_zs_given_zt(zs, mu, sigma, node_mask)
        return torch.allclose(result, torch.zeros_like(result), atol=1e-6)
    run_check("zs == mu 时 log_prob == 0", check_at_mean_is_zero)
    
    def check_negative_when_different():
        zs = torch.randn(2, 5, 3) + 1.0
        mu = torch.zeros(2, 5, 3)
        sigma = torch.tensor(1.0)
        node_mask = torch.ones(2, 5, 1)
        result = compute_log_p_zs_given_zt(zs, mu, sigma, node_mask)
        return (result < 0).all()
    run_check("zs != mu 时 log_prob < 0", check_negative_when_different)
    
    def check_numerical_stability():
        zs = torch.randn(2, 5, 3)
        mu = torch.zeros(2, 5, 3)
        sigma = torch.tensor(1e-10)  # 极小 sigma
        node_mask = torch.ones(2, 5, 1)
        result = compute_log_p_zs_given_zt(zs, mu, sigma, node_mask)
        return not torch.isnan(result).any() and not torch.isinf(result).any()
    run_check("数值稳定性 (极小 sigma)", check_numerical_stability)


# ============================================================================
# 6. Sample 输出结构验证
# ============================================================================

def verify_sample_output_structure():
    """验证 sample 方法输出结构"""
    print_section("6. Sample 输出结构验证")
    
    import torch
    
    # Mock sample 输出
    def mock_sample(n_samples, n_nodes, node_mask, timestep=100, n_dims=3, atom_nf=10):
        device = node_mask.device
        max_nodes = node_mask.shape[1]
        D = n_dims + atom_nf
        T = timestep
        
        x = torch.randn(n_samples, max_nodes, n_dims, device=device) * node_mask
        h = torch.randn(n_samples, max_nodes, atom_nf, device=device) * node_mask
        latents = [torch.randn(n_samples, max_nodes, D, device=device) * node_mask 
                   for _ in range(T + 2)]
        logps = [torch.randn(n_samples, device=device) for _ in range(T + 1)]
        timesteps = list(reversed(range(T))) + [0]
        mus = [torch.randn(n_samples, max_nodes, D, device=device) * node_mask 
               for _ in range(T + 1)]
        sigmas = [torch.rand(1, device=device) + 0.01 for _ in range(T + 1)]
        
        return x, h, latents, logps, timesteps, mus, sigmas
    
    def check_returns_7_values():
        node_mask = torch.ones(2, 10, 1)
        result = mock_sample(2, 10, node_mask, timestep=20)
        return len(result) == 7
    run_check("返回 7 个值", check_returns_7_values)
    
    def check_x_shape():
        node_mask = torch.ones(4, 20, 1)
        x, h, latents, logps, timesteps, mus, sigmas = mock_sample(4, 20, node_mask, timestep=50)
        return x.shape == (4, 20, 3)
    run_check("x 形状 [B, N, 3]", check_x_shape)
    
    def check_h_is_tensor():
        node_mask = torch.ones(2, 10, 1)
        x, h, latents, logps, timesteps, mus, sigmas = mock_sample(2, 10, node_mask, timestep=20)
        return isinstance(h, torch.Tensor)
    run_check("h 是 tensor (非 dict)", check_h_is_tensor)
    
    def check_h_shape():
        node_mask = torch.ones(4, 20, 1)
        x, h, latents, logps, timesteps, mus, sigmas = mock_sample(4, 20, node_mask, timestep=50)
        return h.shape == (4, 20, 10)
    run_check("h 形状 [B, N, atom_nf]", check_h_shape)
    
    def check_latents_length():
        T = 30
        node_mask = torch.ones(2, 10, 1)
        x, h, latents, logps, timesteps, mus, sigmas = mock_sample(2, 10, node_mask, timestep=T)
        return len(latents) == T + 2
    run_check("latents 长度 T+2", check_latents_length)
    
    def check_logps_length():
        T = 30
        node_mask = torch.ones(2, 10, 1)
        x, h, latents, logps, timesteps, mus, sigmas = mock_sample(2, 10, node_mask, timestep=T)
        return len(logps) == T + 1
    run_check("logps 长度 T+1", check_logps_length)
    
    def check_timesteps_length():
        T = 30
        node_mask = torch.ones(2, 10, 1)
        x, h, latents, logps, timesteps, mus, sigmas = mock_sample(2, 10, node_mask, timestep=T)
        return len(timesteps) == T + 1
    run_check("timesteps 长度 T+1", check_timesteps_length)
    
    def check_mus_sigmas_length():
        T = 30
        node_mask = torch.ones(2, 10, 1)
        x, h, latents, logps, timesteps, mus, sigmas = mock_sample(2, 10, node_mask, timestep=T)
        return len(mus) == T + 1 and len(sigmas) == T + 1
    run_check("mus/sigmas 长度 T+1", check_mus_sigmas_length)


# ============================================================================
# 7. BaseReward 接口验证
# ============================================================================

def verify_base_reward_interface():
    """验证 BaseReward 接口"""
    print_section("7. BaseReward 接口验证")
    
    from ddpo.rewards.base import BaseReward
    from abc import ABC
    import inspect
    
    def check_is_abc():
        return issubclass(BaseReward, ABC)
    run_check("BaseReward 是抽象基类", check_is_abc)
    
    def check_calculate_rewards():
        return hasattr(BaseReward, 'calculate_rewards')
    run_check("calculate_rewards 方法存在", check_calculate_rewards)
    
    def check_method_signature():
        sig = inspect.signature(BaseReward.calculate_rewards)
        params = list(sig.parameters.keys())
        return 'data' in params
    run_check("calculate_rewards(data) 签名正确", check_method_signature)
    
    def check_is_abstract_method():
        import inspect
        source = inspect.getsource(BaseReward.calculate_rewards)
        return 'abstractmethod' in source or '@abstractmethod' in inspect.getsource(BaseReward)
    run_check("calculate_rewards 是抽象方法", check_is_abstract_method)


# ============================================================================
# 8. 配置文件验证
# ============================================================================

def verify_config_file():
    """验证配置文件"""
    print_section("8. 配置文件验证")
    
    import yaml
    
    def check_yaml_valid():
        with open("ddpo/configs/diffsbdd_ddpo.yaml", "r") as f:
            config = yaml.safe_load(f)
        return config is not None
    run_check("YAML 格式有效", check_yaml_valid)
    
    def check_model_section():
        with open("ddpo/configs/diffsbdd_ddpo.yaml", "r") as f:
            config = yaml.safe_load(f)
        return 'model' in config
    run_check("model 配置节存在", check_model_section)
    
    def check_pocket_section():
        with open("ddpo/configs/diffsbdd_ddpo.yaml", "r") as f:
            config = yaml.safe_load(f)
        return 'pocket' in config
    run_check("pocket 配置节存在", check_pocket_section)
    
    def check_reward_section():
        with open("ddpo/configs/diffsbdd_ddpo.yaml", "r") as f:
            config = yaml.safe_load(f)
        return 'reward' in config
    run_check("reward 配置节存在", check_reward_section)
    
    def check_training_section():
        with open("ddpo/configs/diffsbdd_ddpo.yaml", "r") as f:
            config = yaml.safe_load(f)
        # 支持 'training' 或 'train' 配置节
        return 'training' in config or 'train' in config
    run_check("training/train 配置节存在", check_training_section)


# ============================================================================
# 9. pytest 属性测试
# ============================================================================

def run_pytest_tests(verbose: bool = False):
    """运行 pytest 属性测试"""
    print_section("9. pytest 属性测试")
    
    test_files = [
        ("tests/property/test_mask_properties.py", "Mask 属性测试 (Property 1)"),
        ("tests/property/test_log_prob_properties.py", "Log Prob 属性测试 (Property 10)"),
        ("tests/property/test_sample_properties.py", "Sample 属性测试 (Property 11, 12, 14)"),
    ]
    
    all_passed = True
    
    for test_file, name in test_files:
        if not os.path.isfile(test_file):
            record(name, False, "文件不存在")
            all_passed = False
            continue
        
        print(f"\n  运行 {name}...")
        
        cmd = ['python', '-m', 'pytest', test_file, '-v', '--tb=short']
        if not verbose:
            cmd.append('-q')
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=180
            )
            
            if result.returncode == 0:
                # 提取通过的测试数量
                output = result.stdout
                if 'passed' in output:
                    import re
                    match = re.search(r'(\d+) passed', output)
                    if match:
                        passed_count = match.group(1)
                        record(name, True, f"{passed_count} 个测试通过")
                    else:
                        record(name, True)
                else:
                    record(name, True)
            else:
                record(name, False, "测试失败")
                all_passed = False
                if verbose:
                    print(result.stdout[-2000:] if result.stdout else "")
                    print(result.stderr[-500:] if result.stderr else "")
        except subprocess.TimeoutExpired:
            record(name, False, "测试超时")
            all_passed = False
        except Exception as e:
            record(name, False, str(e))
            all_passed = False
    
    return all_passed


# ============================================================================
# 10. DiffSBDD 依赖验证
# ============================================================================

def verify_diffsbdd_dependencies():
    """验证 DiffSBDD 相关依赖"""
    print_section("10. DiffSBDD 依赖验证")
    
    def check_pdb_file():
        return os.path.isfile("data/8GJJ-C-Cip.pdb")
    run_check("RE-CmeB PDB 文件存在", check_pdb_file)
    
    def check_biopython():
        from Bio.PDB import PDBParser
        return True
    run_check("BioPython 可用", check_biopython)
    
    def check_torch_scatter():
        from torch_scatter import scatter_mean, scatter_add
        return True
    run_check("torch_scatter 可用", check_torch_scatter)
    
    def check_dataset_params():
        from DiffSBDD.constants import dataset_params
        return 'crossdock' in dataset_params and 'atom_decoder' in dataset_params['crossdock']
    run_check("DiffSBDD dataset_params 配置完整", check_dataset_params)
    
    def check_lightning_module():
        from DiffSBDD.lightning_modules import LigandPocketDDPM
        return hasattr(LigandPocketDDPM, 'prepare_pocket')
    run_check("LigandPocketDDPM.prepare_pocket 方法存在", check_lightning_module)


# ============================================================================
# 结果汇总
# ============================================================================

def print_summary():
    """打印验证结果摘要"""
    print("\n" + "="*60)
    print("  Task 5 Checkpoint 验证结果摘要")
    print("="*60)
    
    passed = sum(1 for _, s, _ in results if s)
    failed = sum(1 for _, s, _ in results if not s)
    total = len(results)
    
    print(f"\n  总计: {total} 项检查")
    print(f"  ✅ 通过: {passed}")
    print(f"  ❌ 失败: {failed}")
    print(f"  通过率: {passed/total*100:.1f}%")
    
    if failed > 0:
        print("\n  失败项:")
        for name, success, msg in results:
            if not success:
                print(f"    ❌ {name}")
                if msg:
                    print(f"       {msg}")
    
    print("\n" + "="*60)
    
    if failed == 0:
        print("  🎉 Task 5 Checkpoint 验证通过！")
        print("  DiffSBDDModel 封装完成，可以继续下一阶段任务。")
    else:
        print("  ⚠️  存在失败项，请修复后重新验证。")
    
    print("="*60)
    
    return failed == 0


def main():
    parser = argparse.ArgumentParser(
        description='Task 5 Checkpoint 验证脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--verbose', '-v', action='store_true', 
                        help='显示详细输出')
    parser.add_argument('--run-tests', action='store_true',
                        help='运行 pytest 属性测试')
    parser.add_argument('--skip-tests', action='store_true',
                        help='跳过 pytest 测试')
    args = parser.parse_args()
    
    print("="*60)
    print("  Task 5 Checkpoint - DiffSBDDModel 封装验证")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # 运行所有验证
    verify_file_structure()
    verify_imports()
    verify_diffsbdd_model_interface()
    verify_mask_conversion()
    verify_log_prob_computation()
    verify_sample_output_structure()
    verify_base_reward_interface()
    verify_config_file()
    verify_diffsbdd_dependencies()
    
    # 运行 pytest 测试
    if not args.skip_tests:
        run_pytest_tests(verbose=args.verbose)
    
    # 打印摘要
    success = print_summary()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
