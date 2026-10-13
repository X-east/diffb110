#!/usr/bin/env python3
"""
DiffSBDD检查点综合分析工具
功能：
1. 基本检查点信息
2. 模型架构分析
3. EGNN层详细分析
4. 参数完整性验证
5. 生成分子验证
6. 多个检查点比较
"""

import torch
import json
import sys
import os
import re
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, Any, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "DiffSBDD"))

try:
    from ddpo.integration import DiffSBDDModel
    from ddpo.utils import annotate_batch_molecule_semantics
    from constants import dataset_params
    from ddpo.runtime import DiffSBDDRollout
    DDPSDD_AVAILABLE = True
except ImportError:
    DDPSDD_AVAILABLE = False
    print("⚠️  DiffSBDD模块不可用，部分功能将受限")

# =========================================
# 配置部分
# =========================================
DEFAULT_CHECKPOINT_PATH = "/home/md666/data/disk6/DiffB110/DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt"
#/home/md666/data/disk6/DiffB110/DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt
#/home/md666/data/disk6/DiffB110/runs/diffsbdd-ddpo-qed-sa-1gpu_20260401_201155/checkpoints/diffsbdd-ddpo-epoch=0001.pt
# =========================================
# 辅助函数
# =========================================

def format_size(size_bytes: int) -> str:
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def get_tensor_stats(tensor: torch.Tensor) -> Dict[str, Any]:
    """获取张量统计信息"""
    if tensor.numel() == 0:
        return {"shape": tuple(tensor.shape), "numel": 0}
    
    stats = {
        "shape": tuple(tensor.shape),
        "numel": tensor.numel(),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device) if hasattr(tensor, 'device') else "CPU",
    }
    
    if tensor.dtype in [torch.float16, torch.float32, torch.float64, torch.bfloat16]:
        try:
            stats.update({
                "mean": float(tensor.float().mean().item()),
                "std": float(tensor.float().std().item()),
                "min": float(tensor.float().min().item()),
                "max": float(tensor.float().max().item()),
                "nan_count": int(torch.isnan(tensor).sum().item()),
                "inf_count": int(torch.isinf(tensor).sum().item()),
            })
        except:
            pass
    
    return stats

# =========================================
# DiffSBDD检查点分析器
# =========================================

class DiffSBDDCheckpointAnalyzer:
    """DiffSBDD检查点综合分析器"""
    
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint = None
        self.state_dict = None
        self.analysis_results = {}
        
    def load_checkpoint(self) -> bool:
        """加载检查点"""
        if not self.checkpoint_path.exists():
            print(f"❌ 文件不存在: {self.checkpoint_path}")
            return False
        
        try:
            self.checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
            
            if "model_state_dict" in self.checkpoint:
                self.state_dict = self.checkpoint["model_state_dict"]
            
            return True
        except Exception as e:
            print(f"❌ 加载失败: {e}")
            return False
    
    def analyze_basic_info(self) -> Dict[str, Any]:
        """分析基本检查点信息"""
        print("\n" + "="*80)
        print("📁 基本检查点信息")
        print("="*80)
        
        info = {
            "file_path": str(self.checkpoint_path),
            "file_size_bytes": self.checkpoint_path.stat().st_size,
            "file_size_formatted": format_size(self.checkpoint_path.stat().st_size),
            "file_format": self.checkpoint_path.suffix,
            "keys": list(self.checkpoint.keys()) if self.checkpoint else [],
        }
        
        print(f"📁 检查点路径: {info['file_path']}")
        print(f"📊 文件大小: {info['file_size_formatted']}")
        print(f"📄 文件格式: {info['file_format']}")
        
        if self.checkpoint:
            print(f"\n🔑 检查点类型: training_checkpoint")
            print(f"🔑 包含的键 ({len(info['keys'])}个):")
            
            for i, key in enumerate(info['keys']):
                value = self.checkpoint[key]
                value_type = type(value).__name__
                
                if isinstance(value, dict):
                    print(f"  {i+1:2d}. {key:<25} (dict, {len(value)} items)")
                elif isinstance(value, torch.Tensor):
                    shape_str = str(tuple(value.shape)) if hasattr(value, 'shape') else "N/A"
                    print(f"  {i+1:2d}. {key:<25} (Tensor, shape={shape_str}, dtype={value.dtype})")
                elif isinstance(value, (list, tuple)):
                    print(f"  {i+1:2d}. {key:<25} ({value_type}, length={len(value)})")
                else:
                    print(f"  {i+1:2d}. {key:<25} ({value_type})")
        
        return info
    
    def analyze_model_architecture(self) -> Dict[str, Any]:
        """分析模型架构"""
        if not self.state_dict:
            print("❌ 没有模型状态字典")
            return {}
        
        print("\n" + "="*80)
        print("🔬 模型架构分析")
        print("="*80)
        
        # 基本统计
        total_params = sum(p.numel() for p in self.state_dict.values() 
                          if isinstance(p, torch.Tensor))
        num_layers = len(self.state_dict)
        
        print(f"📊 模型状态字典:")
        print(f"  参数层数: {num_layers}")
        print(f"  总参数量: {total_params:,}")
        print(f"  内存占用: {total_params * 4 / (1024 * 1024):.2f} MB")
        
        # 按模块分组
        modules = defaultdict(list)
        module_param_counts = defaultdict(int)
        
        for key, param in self.state_dict.items():
            # 解析模块名
            parts = key.split('.')
            
            if 'ddpm' in parts:
                idx = parts.index('ddpm')
                if idx + 1 < len(parts):
                    module_key = '.'.join(parts[:idx+2])
                else:
                    module_key = 'ddpm'
            elif 'dynamics' in parts:
                idx = parts.index('dynamics')
                module_key = 'dynamics.' + parts[idx+1] if idx+1 < len(parts) else 'dynamics'
            else:
                module_key = parts[0] if parts else 'other'
            
            modules[module_key].append(key)
            if isinstance(param, torch.Tensor):
                module_param_counts[module_key] += param.numel()
        
        # 显示模块结构
        print(f"\n🏗️  模块结构:")
        sorted_modules = sorted(module_param_counts.items(), key=lambda x: x[1], reverse=True)
        
        for module_name, param_count in sorted_modules[:10]:
            param_keys = modules[module_name]
            print(f"\n  📦 {module_name}:")
            print(f"    参数数量: {len(param_keys)}")
            print(f"    总参数量: {param_count:,} ({param_count/total_params*100:.1f}%)")
            
            # 显示前几个参数
            for i, key in enumerate(param_keys[:2]):
                param = self.state_dict[key]
                if isinstance(param, torch.Tensor):
                    print(f"      - {key}: {tuple(param.shape)}")
        
        # 分析EGNN层
        egnn_keys = [k for k in self.state_dict.keys() if 'egnn' in k.lower()]
        egnn_blocks = defaultdict(list)
        
        for key in egnn_keys:
            match = re.search(r'e_block_(\d+)', key)
            if match:
                block_num = int(match.group(1))
                egnn_blocks[block_num].append(key)
        
        if egnn_blocks:
            print(f"\n🔍 EGNN层分析:")
            print(f"  发现 {len(egnn_blocks)} 个EGNN块")
            
            egnn_total_params = 0
            for block_num in sorted(egnn_blocks.keys()):
                block_keys = egnn_blocks[block_num]
                block_params = sum(self.state_dict[k].numel() for k in block_keys 
                                  if isinstance(self.state_dict[k], torch.Tensor))
                egnn_total_params += block_params
                
                print(f"  🧱 EGNN Block {block_num}: {block_params:,} 参数")
            
            print(f"  📊 EGNN总参数: {egnn_total_params:,} ({egnn_total_params/total_params*100:.1f}%)")
        
        # 参数类型分布
        param_by_type = Counter()
        param_by_layer = Counter()
        
        for key, param in self.state_dict.items():
            if isinstance(param, torch.Tensor):
                param_by_type[str(param.dtype)] += param.numel()
                
                if 'weight' in key:
                    param_by_layer['weight'] += param.numel()
                elif 'bias' in key:
                    param_by_layer['bias'] += param.numel()
                elif 'norm' in key:
                    param_by_layer['norm'] += param.numel()
        
        print(f"\n🎨 参数分布:")
        print(f"  数据类型:")
        for dtype, count in param_by_type.most_common():
            print(f"    {dtype}: {count:,} ({count/total_params*100:.1f}%)")
        
        print(f"\n  参数类别:")
        for layer_type, count in param_by_layer.most_common():
            print(f"    {layer_type}: {count:,} ({count/total_params*100:.1f}%)")
        
        # 查找最大的参数
        param_sizes = []
        for key, param in self.state_dict.items():
            if isinstance(param, torch.Tensor):
                param_sizes.append((key, param.numel()))
        
        param_sizes.sort(key=lambda x: x[1], reverse=True)
        
        if param_sizes:
            print(f"\n🏆 最大的参数:")
            for key, size in param_sizes[:5]:
                param = self.state_dict[key]
                if isinstance(param, torch.Tensor):
                    print(f"  {key:<60} {size:,} 参数 (shape={tuple(param.shape)})")
        
        return {
            "total_params": total_params,
            "num_layers": num_layers,
            "modules": dict(module_param_counts),
            "egnn_blocks": len(egnn_blocks),
            "param_types": dict(param_by_type),
            "param_categories": dict(param_by_layer),
        }
    
    def analyze_training_info(self) -> Dict[str, Any]:
        """分析训练信息"""
        if not self.checkpoint:
            return {}
        
        print("\n" + "="*80)
        print("📅 训练信息分析")
        print("="*80)
        
        info = {}
        
        # 训练元数据
        if "epoch" in self.checkpoint:
            epoch = self.checkpoint["epoch"]
            info["epoch"] = epoch
            print(f"📅 训练进度:")
            print(f"  Epoch: {epoch}")
        
        if "seed" in self.checkpoint:
            seed = self.checkpoint["seed"]
            info["seed"] = seed
            print(f"  Seed: {seed}")
        
        # 配置信息
        if "config" in self.checkpoint:
            config = self.checkpoint["config"]
            info["config_keys"] = list(config.keys())
            
            print(f"\n⚙️  配置信息:")
            important_keys = ["model", "train", "dataset", "pocket"]
            for key in important_keys:
                if key in config:
                    print(f"  {key}:")
                    if isinstance(config[key], dict):
                        for k, v in list(config[key].items())[:3]:
                            print(f"    {k}: {v}")
                        if len(config[key]) > 3:
                            print(f"    ... 还有{len(config[key])-3}个")
                    else:
                        print(f"    {config[key]}")
        
        # 训练指标
        if "metrics" in self.checkpoint:
            metrics = self.checkpoint["metrics"]
            info["metrics"] = metrics
            
            print(f"\n📈 训练指标:")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.6f}")
                elif isinstance(value, torch.Tensor) and value.numel() == 1:
                    print(f"  {key}: {value.item():.6f}")
        
        # 优化器状态
        optimizer_keys = [k for k in self.checkpoint.keys() 
                         if 'optimizer' in k.lower() and 'state_dict' in k.lower()]
        
        for opt_key in optimizer_keys:
            if opt_key in self.checkpoint:
                opt_state = self.checkpoint[opt_key]
                print(f"\n⚡ 优化器状态 ({opt_key}):")
                
                if isinstance(opt_state, dict) and "state" in opt_state:
                    state = opt_state["state"]
                    print(f"  参数组数: {len(state)}")
                    
                    # 统计优化器参数
                    total_optim_params = 0
                    for param_id, param_state in list(state.items())[:3]:
                        if isinstance(param_state, dict):
                            state_keys = list(param_state.keys())
                            print(f"  参数 {param_id}: {state_keys}")
                            
                            for state_key, state_value in param_state.items():
                                if isinstance(state_value, torch.Tensor):
                                    total_optim_params += state_value.numel()
                    
                    print(f"  优化器状态总参数量: {total_optim_params:,}")
                    info["optimizer_params"] = total_optim_params
        
        return info
    
    def check_integrity(self) -> Dict[str, Any]:
        """检查完整性"""
        print("\n" + "="*80)
        print("🔧 完整性检查")
        print("="*80)
        
        checks = []
        
        # 必需字段
        required_keys = ["epoch", "config", "model_state_dict"]
        missing_keys = [k for k in required_keys if k not in self.checkpoint]
        
        if missing_keys:
            checks.append(f"❌ 缺少关键字段: {missing_keys}")
        else:
            checks.append("✅ 基本字段完整")
            
            # 检查模型参数
            if "model_state_dict" in self.checkpoint:
                state_dict = self.checkpoint["model_state_dict"]
                total_params = sum(p.numel() for p in state_dict.values() 
                                  if isinstance(p, torch.Tensor))
                
                if total_params > 1000000:  # 超过100万参数
                    checks.append(f"✅ 模型参数完整: {total_params:,} 参数")
                else:
                    checks.append(f"⚠️  模型参数可能不完整: 只有 {total_params:,} 参数")
            
            # 检查EGNN层
            egnn_keys = [k for k in self.state_dict.keys() if 'egnn' in k.lower()]
            egnn_blocks = set()
            
            for key in egnn_keys:
                match = re.search(r'e_block_(\d+)', key)
                if match:
                    egnn_blocks.add(int(match.group(1)))
            
            if egnn_blocks:
                max_block = max(egnn_blocks)
                if max_block >= 4:  # 至少有5个块（0-4）
                    checks.append(f"✅ EGNN结构完整: {max_block + 1} 个块")
                else:
                    checks.append(f"⚠️  EGNN块数可能不足: 只有 {max_block + 1} 个块")
            
            # 检查优化器
            if "optimizer_state_dict" in self.checkpoint:
                checks.append("✅ 包含优化器状态，可恢复训练")
            else:
                checks.append("⚠️  缺少优化器状态")
        
        # 数值检查
        nan_inf_issues = []
        for key, param in self.state_dict.items():
            if isinstance(param, torch.Tensor) and param.dtype in [torch.float16, torch.float32, torch.float64]:
                nan_count = torch.isnan(param).sum().item()
                inf_count = torch.isinf(param).sum().item()
                
                if nan_count > 0 or inf_count > 0:
                    nan_inf_issues.append(f"{key}: NaN={nan_count}, Inf={inf_count}")
        
        if nan_inf_issues:
            checks.append(f"❌ 发现数值问题: {len(nan_inf_issues)} 个参数有NaN/Inf")
            for issue in nan_inf_issues[:3]:
                checks.append(f"  - {issue}")
        else:
            checks.append("✅ 数值检查通过: 无NaN/Inf值")
        
        # 打印检查结果
        for i, check in enumerate(checks, 1):
            print(f"  {i}. {check}")
        
        return {"checks": checks, "missing_keys": missing_keys, "nan_inf_issues": len(nan_inf_issues)}
    
    def generate_molecules(self, num_samples: int = 5) -> Optional[Dict[str, Any]]:
        """生成分子验证模型"""
        if not DDPSDD_AVAILABLE:
            print("❌ DiffSBDD模块不可用，跳过分子生成")
            return None
        
        print("\n" + "="*80)
        print("🧪 分子生成验证")
        print("="*80)
        
        try:
            # 加载模型
            config = self.checkpoint.get("config", {})
            dataset_name = config.get("dataset", "crossdock")
            
            model = DiffSBDDModel(
                checkpoint_path=config.get("model", {}).get("checkpoint", ""),
                pocket_pdb_path=config.get("pocket", {}).get("pdb_path", ""),
                ref_ligand_path=config.get("pocket", {}).get("ref_ligand", ""),
                device="cpu",
                dataset_name=dataset_name,
            )
            
            # 加载权重
            model.model.load_state_dict(self.checkpoint["model_state_dict"])
            model.model.eval()
            
            # 生成分子
            rollout = DiffSBDDRollout(model=model, config=config)
            
            class MockPrompt:
                def __init__(self, pid=0):
                    self.pocket_id = pid
            
            prompts = [MockPrompt(i) for i in range(num_samples)]
            
            with torch.no_grad():
                samples = rollout.generate_minibatch(prompts)
                
                if hasattr(samples, 'batch'):
                    # 注释分子信息
                    dataset_info = dataset_params.get(dataset_name, {})
                    samples = annotate_batch_molecule_semantics(samples, dataset_info=dataset_info)
                    
                    # 提取结果
                    smiles_list = samples.batch.get("smiles", [])
                    validity_list = samples.batch.get("rdkit_valid", [])
                    qed_list = samples.batch.get("qed", [])
                    sa_list = samples.batch.get("sa", [])
                    
                    # 统计
                    valid_count = sum(int(v) for v in validity_list) if validity_list else 0
                    valid_qed = [q for q, v in zip(qed_list, validity_list) if v] if qed_list and validity_list else []
                    valid_sa = [s for s, v in zip(sa_list, validity_list) if v] if sa_list and validity_list else []
                    
                    print(f"🎯 生成 {num_samples} 个分子:")
                    print(f"  有效分子: {valid_count}/{num_samples} ({valid_count/num_samples*100:.1f}%)")
                    
                    if valid_qed:
                        print(f"  平均QED: {np.mean(valid_qed):.3f}")
                    
                    if valid_sa:
                        print(f"  平均SA: {np.mean(valid_sa):.3f}")
                    
                    # 显示生成的分子
                    if smiles_list:
                        print(f"\n🧬 生成的分子示例:")
                        for i, (smiles, valid) in enumerate(zip(smiles_list[:3], validity_list[:3])):
                            if i < len(smiles_list) and i < len(validity_list):
                                status = "✅" if valid else "❌"
                                print(f"  {status} {smiles[:50]}...")
                    
                    return {
                        "num_samples": num_samples,
                        "valid_count": valid_count,
                        "valid_ratio": valid_count / num_samples if num_samples > 0 else 0,
                        "avg_qed": np.mean(valid_qed) if valid_qed else 0,
                        "avg_sa": np.mean(valid_sa) if valid_sa else 0,
                    }
            
        except Exception as e:
            print(f"❌ 分子生成失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def compare_with_other(self, other_checkpoint_path: str) -> Dict[str, Any]:
        """与另一个检查点比较"""
        print("\n" + "="*80)
        print("🔀 检查点比较")
        print("="*80)
        
        try:
            other_checkpoint = torch.load(other_checkpoint_path, map_location='cpu')
            other_state_dict = other_checkpoint.get("model_state_dict", {})
            
            # 比较基本信息
            print(f"📊 比较 {self.checkpoint_path.name} 和 {Path(other_checkpoint_path).name}")
            
            # 参数数量比较
            self_params = sum(p.numel() for p in self.state_dict.values() 
                             if isinstance(p, torch.Tensor)) if self.state_dict else 0
            other_params = sum(p.numel() for p in other_state_dict.values() 
                              if isinstance(p, torch.Tensor)) if other_state_dict else 0
            
            print(f"\n📈 参数比较:")
            print(f"  {self.checkpoint_path.name}: {self_params:,} 参数")
            print(f"  {Path(other_checkpoint_path).name}: {other_params:,} 参数")
            print(f"  差异: {abs(self_params - other_params):,} 参数 ({abs(self_params - other_params)/max(self_params, 1)*100:.1f}%)")
            
            # 参数差异
            if self.state_dict and other_state_dict:
                common_keys = set(self.state_dict.keys()) & set(other_state_dict.keys())
                unique_self = set(self.state_dict.keys()) - set(other_state_dict.keys())
                unique_other = set(other_state_dict.keys()) - set(self.state_dict.keys())
                
                print(f"\n🔑 参数键比较:")
                print(f"  共同键: {len(common_keys)}")
                print(f"  {self.checkpoint_path.name} 独有: {len(unique_self)}")
                print(f"  {Path(other_checkpoint_path).name} 独有: {len(unique_other)}")
                
                # 计算参数差异
                if common_keys:
                    total_diff = 0
                    for key in list(common_keys)[:10]:  # 只比较前10个
                        if (isinstance(self.state_dict[key], torch.Tensor) and 
                            isinstance(other_state_dict[key], torch.Tensor)):
                            diff = torch.norm(self.state_dict[key].float() - other_state_dict[key].float())
                            total_diff += diff.item()
                    
                    avg_diff = total_diff / len(common_keys) if common_keys else 0
                    print(f"\n📊 平均参数差异: {avg_diff:.6f}")
            
            # 训练进度比较
            self_epoch = self.checkpoint.get("epoch", 0)
            other_epoch = other_checkpoint.get("epoch", 0)
            
            print(f"\n📅 训练进度比较:")
            print(f"  {self.checkpoint_path.name}: Epoch {self_epoch}")
            print(f"  {Path(other_checkpoint_path).name}: Epoch {other_epoch}")
            
            return {
                "self_params": self_params,
                "other_params": other_params,
                "common_keys": len(common_keys) if 'common_keys' in locals() else 0,
                "avg_diff": avg_diff if 'avg_diff' in locals() else 0,
            }
            
        except Exception as e:
            print(f"❌ 比较失败: {e}")
            return {}
    
    def full_analysis(self, 
                     generate_molecules: bool = False,
                     num_samples: int = 5,
                     compare_with: Optional[str] = None) -> Dict[str, Any]:
        """执行完整分析"""
        print("🚀 DiffSBDD检查点综合分析")
        print("="*80)
        
        # 加载检查点
        if not self.load_checkpoint():
            return {}
        
        # 执行各项分析
        results = {
            "basic_info": self.analyze_basic_info(),
            "architecture": self.analyze_model_architecture(),
            "training_info": self.analyze_training_info(),
            "integrity": self.check_integrity(),
        }
        
        # 分子生成
        if generate_molecules and DDPSDD_AVAILABLE:
            results["generation"] = self.generate_molecules(num_samples=num_samples)
        
        # 比较
        if compare_with:
            results["comparison"] = self.compare_with_other(compare_with)
        
        # 保存结果
        output_file = f"{self.checkpoint_path.stem}_full_analysis.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n💾 完整分析结果已保存到: {output_file}")
        
        return results

# =========================================
# 批量分析工具
# =========================================

def analyze_checkpoint_directory(
    directory_path: str,
    generate_molecules: bool = False,
    num_samples: int = 3
) -> None:
    """分析目录中的所有检查点"""
    dir_path = Path(directory_path)
    if not dir_path.exists():
        print(f"❌ 目录不存在: {directory_path}")
        return
    
    checkpoint_files = sorted(list(dir_path.glob("*.pt")) + list(dir_path.glob("*.ckpt")))
    
    if not checkpoint_files:
        print(f"❌ 没有找到检查点文件")
        return
    
    print(f"📁 找到 {len(checkpoint_files)} 个检查点文件")
    
    summary_results = []
    
    for i, checkpoint_file in enumerate(checkpoint_files):
        print(f"\n{'='*80}")
        print(f"📄 分析文件 {i+1}/{len(checkpoint_files)}: {checkpoint_file.name}")
        print(f"{'='*80}")
        
        analyzer = DiffSBDDCheckpointAnalyzer(str(checkpoint_file))
        
        if analyzer.load_checkpoint():
            # 基本分析
            basic_info = analyzer.analyze_basic_info()
            architecture = analyzer.analyze_model_architecture()
            training_info = analyzer.analyze_training_info()
            
            # 添加到汇总
            summary = {
                "file": checkpoint_file.name,
                "epoch": training_info.get("epoch", 0),
                "total_params": architecture.get("total_params", 0),
                "egnn_blocks": architecture.get("egnn_blocks", 0),
            }
            
            # 分子生成
            if generate_molecules and DDPSDD_AVAILABLE:
                generation = analyzer.generate_molecules(num_samples=num_samples)
                if generation:
                    summary.update({
                        "valid_ratio": generation.get("valid_ratio", 0),
                        "avg_qed": generation.get("avg_qed", 0),
                        "avg_sa": generation.get("avg_sa", 0),
                    })
            
            summary_results.append(summary)
    
    # 生成汇总报告
    if summary_results:
        print("\n" + "="*80)
        print("📊 汇总报告")
        print("="*80)
        
        # 按epoch排序
        summary_results.sort(key=lambda x: x.get("epoch", 0))
        
        print(f"{'Epoch':<6} {'Params':<12} {'EGNN':<6} {'Valid%':<8} {'QED':<8} {'SA':<8} {'File'}")
        print("-" * 80)
        
        for result in summary_results:
            print(f"{result.get('epoch', 0):<6} "
                  f"{result.get('total_params', 0):<12,} "
                  f"{result.get('egnn_blocks', 0):<6} "
                  f"{result.get('valid_ratio', 0)*100:<7.1f}% "
                  f"{result.get('avg_qed', 0):<8.3f} "
                  f"{result.get('avg_sa', 0):<8.3f} "
                  f"{result.get('file', '')}")
        
        # 保存汇总
        summary_file = dir_path / "checkpoint_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 汇总报告已保存: {summary_file}")

# =========================================
# 主程序
# =========================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DiffSBDD检查点综合分析工具")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT_PATH, 
                       help="检查点路径")
    parser.add_argument("--dir", type=str, help="检查点目录（批量分析）")
    parser.add_argument("--generate", action="store_true", help="生成分子验证")
    parser.add_argument("--num-samples", type=int, default=5, help="生成分子数量")
    parser.add_argument("--compare", type=str, help="与另一个检查点比较")
    parser.add_argument("--output", type=str, help="输出目录")
    
    args = parser.parse_args()
    
    if args.dir:
        # 批量分析模式
        analyze_checkpoint_directory(
            directory_path=args.dir,
            generate_molecules=args.generate,
            num_samples=args.num_samples
        )
    else:
        # 单文件分析模式
        analyzer = DiffSBDDCheckpointAnalyzer(args.checkpoint)
        
        results = analyzer.full_analysis(
            generate_molecules=args.generate,
            num_samples=args.num_samples,
            compare_with=args.compare
        )
    
    print("\n" + "="*80)
    print("✅ 分析完成")
    print("="*80)