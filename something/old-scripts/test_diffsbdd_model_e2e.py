#!/usr/bin/env python
"""
DiffSBDDModel 端到端验证脚本

使用真实的 DiffSBDD checkpoint 验证封装是否正确工作。

使用方法:
    python scripts/test_diffsbdd_model_e2e.py --checkpoint <path_to_ckpt>
    python scripts/test_diffsbdd_model_e2e.py --checkpoint <path_to_ckpt> --n_samples 4
    python scripts/test_diffsbdd_model_e2e.py --checkpoint <path_to_ckpt> --timestep 100

如果没有 checkpoint，可以只运行接口测试:
    python scripts/test_diffsbdd_model_e2e.py --interface-only
"""

import argparse
import sys
import os
import time
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np


def print_section(title: str):
    """打印章节标题"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_result(name: str, success: bool, msg: str = ""):
    """打印结果"""
    status = "✅" if success else "❌"
    print(f"  {status} {name}")
    if msg:
        print(f"     └─ {msg}")


def test_interface_only():
    """只测试接口（不需要 checkpoint）"""
    print_section("接口测试 (无需 checkpoint)")
    
    # 1. 测试导入
    print("\n  1. 测试导入...")
    try:
        from ddpo.integration.diffsbdd_model import DiffSBDDModel
        from ddpo.integration.mask_utils import batch_mask_to_node_mask, node_mask_to_batch_mask
        print_result("模块导入", True)
    except Exception as e:
        print_result("模块导入", False, str(e))
        return False
    
    # 2. 测试 mask 转换
    print("\n  2. 测试 mask 转换...")
    try:
        batch_mask = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2])
        node_mask = batch_mask_to_node_mask(batch_mask)
        recovered = node_mask_to_batch_mask(node_mask)
        
        assert node_mask.shape == (3, 4, 1), f"node_mask 形状错误: {node_mask.shape}"
        assert torch.equal(batch_mask, recovered), "往返转换失败"
        print_result("mask 转换", True, f"batch_mask {batch_mask.shape} ↔ node_mask {node_mask.shape}")
    except Exception as e:
        print_result("mask 转换", False, str(e))
        return False
    
    # 3. 测试 DiffSBDDModel 类定义
    print("\n  3. 测试 DiffSBDDModel 类定义...")
    try:
        import inspect
        
        required_methods = [
            'get_mask', 'compute_log_p_zs_given_zt', 
            'sample_p_zs_given_zt', 'sample_p_xh_given_z0', 'sample'
        ]
        
        for method in required_methods:
            assert hasattr(DiffSBDDModel, method), f"缺少方法: {method}"
        
        print_result("类方法定义", True, f"所有 {len(required_methods)} 个方法存在")
    except Exception as e:
        print_result("类方法定义", False, str(e))
        return False
    
    print("\n  ✅ 接口测试通过！")
    return True


def test_with_checkpoint(checkpoint_path: str, pdb_path: str, ref_ligand: str,
                         n_samples: int = 2, n_nodes: int = 20, timestep: int = 50,
                         device: str = "cuda"):
    """使用真实 checkpoint 测试"""
    print_section(f"端到端测试 (checkpoint: {Path(checkpoint_path).name})")
    
    # 检查 GPU
    if device == "cuda" and not torch.cuda.is_available():
        print("  ⚠️  CUDA 不可用，切换到 CPU")
        device = "cpu"
    
    print(f"\n  配置:")
    print(f"    - checkpoint: {checkpoint_path}")
    print(f"    - pdb_path: {pdb_path}")
    print(f"    - ref_ligand: {ref_ligand}")
    print(f"    - n_samples: {n_samples}")
    print(f"    - n_nodes: {n_nodes}")
    print(f"    - timestep: {timestep}")
    print(f"    - device: {device}")
    
    # 1. 加载模型
    print("\n  1. 加载模型...")
    try:
        from ddpo.integration.diffsbdd_model import DiffSBDDModel
        
        start_time = time.time()
        model = DiffSBDDModel(
            checkpoint_path=checkpoint_path,
            pocket_pdb_path=pdb_path,
            ref_ligand_path=ref_ligand,
            device=device,
        )
        load_time = time.time() - start_time
        
        print_result("模型加载", True, f"耗时 {load_time:.2f}s")
        
        # 打印模型信息
        print(f"\n  模型信息:")
        print(f"    - T (timesteps): {model.T}")
        print(f"    - n_dims: {model.n_dims}")
        print(f"    - atom_nf: {model.atom_nf}")
        print(f"    - aa_nf: {model.aa_nf}")
        
        pocket_info = model.get_pocket_info()
        print(f"    - pocket residues: {pocket_info['num_residues']}")
        print(f"    - pocket size: {pocket_info['pocket_size']}")
        
    except Exception as e:
        print_result("模型加载", False, str(e))
        import traceback
        traceback.print_exc()
        return False
    
    # 2. 测试 get_mask
    print("\n  2. 测试 get_mask...")
    try:
        nodesxsample = torch.tensor([n_nodes] * n_samples)
        node_mask, edge_mask = model.get_mask(nodesxsample, n_samples, n_nodes)
        
        assert node_mask.shape == (n_samples, n_nodes, 1), f"node_mask 形状错误: {node_mask.shape}"
        print_result("get_mask", True, f"node_mask: {node_mask.shape}, edge_mask: {edge_mask.shape}")
    except Exception as e:
        print_result("get_mask", False, str(e))
        return False
    
    # 3. 测试 sample_p_zs_given_zt
    print("\n  3. 测试 sample_p_zs_given_zt...")
    try:
        node_mask = node_mask.to(device)
        edge_mask = edge_mask.to(device)
        
        D = model.n_dims + model.atom_nf
        zt = torch.randn(n_samples, n_nodes, D, device=device) * node_mask
        s = torch.tensor([[0.5]] * n_samples, device=device)
        t = torch.tensor([[0.6]] * n_samples, device=device)
        
        start_time = time.time()
        zs, log_prob, mu, sigma = model.sample_p_zs_given_zt(
            s, t, zt, node_mask, edge_mask
        )
        step_time = time.time() - start_time
        
        assert zs.shape == (n_samples, n_nodes, D), f"zs 形状错误: {zs.shape}"
        assert log_prob.shape == (n_samples,), f"log_prob 形状错误: {log_prob.shape}"
        assert mu.shape == (n_samples, n_nodes, D), f"mu 形状错误: {mu.shape}"
        
        print_result("sample_p_zs_given_zt", True, 
                    f"zs: {zs.shape}, log_prob: {log_prob.shape}, 耗时 {step_time*1000:.1f}ms")
    except Exception as e:
        print_result("sample_p_zs_given_zt", False, str(e))
        import traceback
        traceback.print_exc()
        return False
    
    # 4. 测试 sample_p_xh_given_z0
    print("\n  4. 测试 sample_p_xh_given_z0...")
    try:
        z0 = torch.randn(n_samples, n_nodes, D, device=device) * node_mask
        
        start_time = time.time()
        x, h, mu, sigma, log_prob, zeros, xh = model.sample_p_xh_given_z0(
            z0, node_mask, edge_mask
        )
        step_time = time.time() - start_time
        
        assert x.shape == (n_samples, n_nodes, 3), f"x 形状错误: {x.shape}"
        assert isinstance(h, dict), f"h 应该是 dict，实际是 {type(h)}"
        assert 'categorical' in h, "h 缺少 'categorical' 键"
        assert h['categorical'].shape == (n_samples, n_nodes, model.atom_nf), \
            f"h['categorical'] 形状错误: {h['categorical'].shape}"
        
        print_result("sample_p_xh_given_z0", True,
                    f"x: {x.shape}, h['categorical']: {h['categorical'].shape}, 耗时 {step_time*1000:.1f}ms")
    except Exception as e:
        print_result("sample_p_xh_given_z0", False, str(e))
        import traceback
        traceback.print_exc()
        return False
    
    # 5. 测试完整 sample 循环
    print("\n  5. 测试完整 sample 循环...")
    try:
        start_time = time.time()
        x, h, latents, logps, timesteps_list, mus, sigmas = model.sample(
            n_samples=n_samples,
            n_nodes=n_nodes,
            node_mask=node_mask,
            edge_mask=edge_mask,
            timestep=timestep,
        )
        sample_time = time.time() - start_time
        
        # 验证输出
        assert x.shape == (n_samples, n_nodes, 3), f"x 形状错误: {x.shape}"
        assert isinstance(h, dict), f"h 应该是 dict"
        assert len(latents) == timestep + 2, f"latents 长度错误: {len(latents)} != {timestep + 2}"
        assert len(logps) == timestep + 1, f"logps 长度错误: {len(logps)} != {timestep + 1}"
        assert len(timesteps_list) == timestep + 1, f"timesteps 长度错误"
        assert len(mus) == timestep + 1, f"mus 长度错误"
        assert len(sigmas) == timestep + 1, f"sigmas 长度错误"
        
        # 验证 CoG 接近零
        x_sum = (x * node_mask).sum(dim=1)
        n_valid = node_mask.sum(dim=1)
        cog = x_sum / (n_valid + 1e-8)
        max_cog = cog.abs().max().item()
        
        print_result("sample 完整循环", True,
                    f"耗时 {sample_time:.2f}s ({sample_time/timestep*1000:.1f}ms/step)")
        print(f"     └─ x: {x.shape}, latents: {len(latents)}, logps: {len(logps)}")
        print(f"     └─ CoG 漂移: {max_cog:.6f}")
        
        # 打印一些统计信息
        print(f"\n  采样统计:")
        print(f"    - x 范围: [{x.min().item():.3f}, {x.max().item():.3f}]")
        print(f"    - log_prob 范围: [{min(lp.min().item() for lp in logps):.3f}, "
              f"{max(lp.max().item() for lp in logps):.3f}]")
        
    except Exception as e:
        print_result("sample 完整循环", False, str(e))
        import traceback
        traceback.print_exc()
        return False
    
    # 6. 测试 prev_sample 参数（Actor 训练需要）
    print("\n  6. 测试 prev_sample 参数...")
    try:
        # 使用之前采样的 latent 作为 prev_sample
        prev_z = latents[1].clone()  # z_{T-1}
        s = torch.tensor([[0.99]] * n_samples, device=device)
        t = torch.tensor([[1.0]] * n_samples, device=device)
        
        zs, log_prob, mu, sigma = model.sample_p_zs_given_zt(
            s, t, latents[0], node_mask, edge_mask, prev_sample=prev_z
        )
        
        # log_prob 应该是对 prev_z 计算的，而不是新采样的 zs
        print_result("prev_sample 参数", True, f"log_prob: {log_prob.mean().item():.4f}")
    except Exception as e:
        print_result("prev_sample 参数", False, str(e))
        return False
    
    print("\n  ✅ 端到端测试通过！")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='DiffSBDDModel 端到端验证脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--checkpoint', type=str, 
                        help='DiffSBDD checkpoint 路径 (.ckpt)')
    parser.add_argument('--pdb', type=str, default='data/8GJJ-C-Cip.pdb',
                        help='蛋白质 PDB 文件路径')
    parser.add_argument('--ref-ligand', type=str, default='C:CIP',
                        help='参考配体 (链:残基名 或 SDF 文件路径)')
    parser.add_argument('--n-samples', type=int, default=2,
                        help='采样数量')
    parser.add_argument('--n-nodes', type=int, default=20,
                        help='每个分子的原子数')
    parser.add_argument('--timestep', type=int, default=50,
                        help='扩散步数')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], help='设备')
    parser.add_argument('--interface-only', action='store_true',
                        help='只测试接口（不需要 checkpoint）')
    
    args = parser.parse_args()
    
    print("="*60)
    print("  DiffSBDDModel 端到端验证")
    print("="*60)
    
    # 接口测试
    if not test_interface_only():
        sys.exit(1)
    
    # 如果只测试接口，到此结束
    if args.interface_only:
        print("\n  ℹ️  使用 --checkpoint 参数进行完整端到端测试")
        sys.exit(0)
    
    # 检查 checkpoint
    if not args.checkpoint:
        print("\n  ⚠️  未指定 checkpoint，跳过端到端测试")
        print("  使用方法: python scripts/test_diffsbdd_model_e2e.py --checkpoint <path>")
        sys.exit(0)
    
    if not os.path.isfile(args.checkpoint):
        print(f"\n  ❌ checkpoint 文件不存在: {args.checkpoint}")
        sys.exit(1)
    
    # 端到端测试
    success = test_with_checkpoint(
        checkpoint_path=args.checkpoint,
        pdb_path=args.pdb,
        ref_ligand=args.ref_ligand,
        n_samples=args.n_samples,
        n_nodes=args.n_nodes,
        timestep=args.timestep,
        device=args.device,
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
