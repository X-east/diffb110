#!/usr/bin/env python3
"""
PyTorch检查点分析工具
支持.pt和.ckpt格式
"""

import torch
import json
import sys
import os
from pathlib import Path
import numpy as np
from collections import defaultdict, Counter
import pprint
from typing import Dict, Any, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# =========================================
# 配置部分 - 修改这里的路径
# =========================================
CHECKPOINT_PATH = "/home/md666/data/disk6/DiffB110/OUT/diffsbdd-ddpo-qed-sa-1gpu_20260401_201155/checkpoints/diffsbdd-ddpo-epoch=0001.pt"

# 可选：比较两个检查点
COMPARE_PATH = None  # 设置为另一个检查点路径进行比较

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

def analyze_state_dict(state_dict: Dict[str, torch.Tensor], name: str = "模型") -> Dict[str, Any]:
    """分析状态字典"""
    print(f"\n🔍 分析 {name} 状态字典:")
    print("="*80)
    
    if not state_dict:
        print("  ❌ 状态字典为空")
        return {}
    
    # 基本统计
    total_params = 0
    total_size_bytes = 0
    param_shapes = Counter()
    param_types = Counter()
    
    param_details = {}
    
    for i, (key, param) in enumerate(list(state_dict.items())[:20]):  # 只分析前20个
        if isinstance(param, torch.Tensor):
            # 统计信息
            stats = get_tensor_stats(param)
            total_params += stats["numel"]
            total_size_bytes += param.element_size() * param.numel()
            
            # 记录形状
            shape_str = str(stats["shape"])
            param_shapes[shape_str] += 1
            
            # 记录类型
            dtype_str = stats["dtype"]
            param_types[dtype_str] += 1
            
            # 保存详细信息
            param_details[key] = stats
            
            # 打印前几个参数的详细信息
            if i < 5:
                print(f"  {i+1:2d}. {key}")
                print(f"      形状: {stats['shape']:<20} 参数数: {stats['numel']:,}")
                if 'mean' in stats:
                    print(f"      统计: μ={stats['mean']:.6f} ±{stats['std']:.6f}")
    
    # 汇总信息
    result = {
        "total_params": total_params,
        "total_size_mb": total_size_bytes / (1024 * 1024),
        "num_layers": len(state_dict),
        "common_shapes": dict(param_shapes.most_common(10)),
        "dtype_distribution": dict(param_types),
    }
    
    print(f"\n📊 汇总信息:")
    print(f"  参数层数: {len(state_dict)}")
    print(f"  总参数量: {total_params:,}")
    print(f"  内存占用: {result['total_size_mb']:.2f} MB")
    
    if param_shapes:
        print(f"\n  📏 常见参数形状:")
        for shape, count in param_shapes.most_common(5):
            print(f"    {shape:<30} x {count}")
    
    if param_types:
        print(f"\n  🎨 数据类型分布:")
        for dtype, count in param_types.most_common():
            print(f"    {dtype:<20} x {count}")
    
    return result

def find_largest_parameters(state_dict: Dict[str, torch.Tensor], top_n: int = 10) -> List[Tuple[str, int]]:
    """查找最大的参数"""
    sizes = []
    for key, param in state_dict.items():
        if isinstance(param, torch.Tensor):
            sizes.append((key, param.numel()))
    
    return sorted(sizes, key=lambda x: x[1], reverse=True)[:top_n]

def check_nan_inf(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, int]]:
    """检查NaN和Inf值"""
    issues = {}
    for key, param in state_dict.items():
        if isinstance(param, torch.Tensor) and param.dtype in [torch.float16, torch.float32, torch.float64]:
            nan_count = torch.isnan(param).sum().item()
            inf_count = torch.isinf(param).sum().item()
            
            if nan_count > 0 or inf_count > 0:
                issues[key] = {
                    "nan": int(nan_count),
                    "inf": int(inf_count),
                    "shape": tuple(param.shape)
                }
    
    return issues

# =========================================
# 主分析函数
# =========================================

def analyze_checkpoint(checkpoint_path: str, compare_path: str = None) -> Dict[str, Any]:
    """分析PyTorch检查点文件"""
    
    print("="*80)
    print("🧬 PyTorch检查点分析工具")
    print("="*80)
    print(f"📁 检查点路径: {checkpoint_path}")
    
    # 检查文件是否存在
    if not Path(checkpoint_path).exists():
        print(f"❌ 文件不存在: {checkpoint_path}")
        return {}
    
    # 获取文件信息
    file_size = Path(checkpoint_path).stat().st_size
    print(f"📊 文件大小: {format_size(file_size)}")
    print(f"📄 文件格式: {Path(checkpoint_path).suffix}")
    
    try:
        # 加载检查点
        print(f"\n⏳ 加载检查点...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print("✅ 检查点加载成功")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return {}
    
    result = {
        "file_path": checkpoint_path,
        "file_size_bytes": file_size,
        "keys": list(checkpoint.keys()),
        "checkpoint_type": "unknown"
    }
    
    # 识别检查点类型
    if "epoch" in checkpoint and "model_state_dict" in checkpoint:
        result["checkpoint_type"] = "training_checkpoint"
    elif "state_dict" in checkpoint and "optimizer" in checkpoint:
        result["checkpoint_type"] = "lightning_checkpoint"
    elif isinstance(checkpoint, dict) and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        result["checkpoint_type"] = "state_dict_only"
    
    print(f"\n🔑 检查点类型: {result['checkpoint_type']}")
    print(f"🔑 包含的键 ({len(result['keys'])}个):")
    for i, key in enumerate(result["keys"]):
        value = checkpoint[key]
        value_type = type(value).__name__
        
        if isinstance(value, dict):
            print(f"  {i+1:2d}. {key:<25} (dict, {len(value)} items)")
        elif isinstance(value, torch.Tensor):
            print(f"  {i+1:2d}. {key:<25} (Tensor, shape={tuple(value.shape)}, dtype={value.dtype})")
        elif isinstance(value, (list, tuple)):
            print(f"  {i+1:2d}. {key:<25} ({value_type}, length={len(value)})")
        else:
            print(f"  {i+1:2d}. {key:<25} ({value_type})")
    
    # 详细分析每个部分
    print("\n" + "="*80)
    print("🔍 详细分析")
    print("="*80)
    
    # 1. 训练元数据
    if "epoch" in checkpoint:
        print(f"\n📅 训练元数据:")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Step: {checkpoint.get('global_step', 'N/A')}")
        
        if "timestamp" in checkpoint:
            print(f"  时间戳: {checkpoint.get('timestamp', 'N/A')}")
    
    # 2. 配置信息
    if "config" in checkpoint:
        config = checkpoint["config"]
        print(f"\n⚙️  配置信息 ({len(config)}个键):")
        
        # 显示重要配置
        important_keys = ["model", "train", "dataset", "pocket", "optimizer", "learning_rate"]
        for key in important_keys:
            if key in config:
                print(f"  {key}:")
                if isinstance(config[key], dict):
                    for k, v in list(config[key].items())[:3]:  # 只显示前3个
                        print(f"    {k}: {v}")
                    if len(config[key]) > 3:
                        print(f"    ... 还有{len(config[key])-3}个")
                else:
                    print(f"    {config[key]}")
    
    # 3. 模型状态字典
    model_state_keys = []
    for key in ["model_state_dict", "state_dict", "model"]:
        if key in checkpoint:
            model_state_keys.append(key)
    
    for model_key in model_state_keys:
        if model_key in checkpoint:
            analyze_state_dict(checkpoint[model_key], f"{model_key}")
            
            # 查找最大的参数
            largest = find_largest_parameters(checkpoint[model_key], top_n=5)
            if largest:
                print(f"\n  🏆 最大的参数:")
                for name, size in largest:
                    print(f"    {name:<50} {size:,} 参数")
            
            # 检查NaN/Inf
            issues = check_nan_inf(checkpoint[model_key])
            if issues:
                print(f"\n  ⚠️  发现数值问题:")
                for key, issue in list(issues.items())[:5]:
                    print(f"    {key}: NaN={issue['nan']}, Inf={issue['inf']}, shape={issue['shape']}")
    
    # 4. 优化器状态
    optimizer_keys = []
    for key in ["optimizer_state_dict", "optimizer", "actor_optimizer", "optimizer_state"]:
        if key in checkpoint:
            optimizer_keys.append(key)
    
    for opt_key in optimizer_keys:
        if opt_key in checkpoint:
            opt_state = checkpoint[opt_key]
            print(f"\n⚡ 优化器状态 ({opt_key}):")
            
            if isinstance(opt_state, dict):
                print(f"  包含 {len(opt_state)} 个键")
                
                # 检查是否有'state'键（这是PyTorch优化器的标准格式）
                if "state" in opt_state and isinstance(opt_state["state"], dict):
                    print(f"  参数组数: {len(opt_state.get('state', {}))}")
                    
                    # 分析优化器状态
                    total_optim_params = 0
                    for param_id, param_state in list(opt_state["state"].items())[:3]:  # 只显示前3个
                        if isinstance(param_state, dict):
                            state_keys = list(param_state.keys())
                            print(f"  参数 {param_id}: {state_keys}")
                            
                            # 计算状态大小
                            for state_key, state_value in param_state.items():
                                if isinstance(state_value, torch.Tensor):
                                    total_optim_params += state_value.numel()
                    
                    print(f"  优化器状态总参数量: {total_optim_params:,}")
    
    # 5. 训练指标
    if "metrics" in checkpoint:
        metrics = checkpoint["metrics"]
        print(f"\n📈 训练指标 ({len(metrics)}个):")
        
        # 按类型分组显示
        scalar_metrics = {}
        tensor_metrics = {}
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                scalar_metrics[key] = value
            elif isinstance(value, torch.Tensor) and value.numel() == 1:
                scalar_metrics[key] = value.item()
            else:
                tensor_metrics[key] = value
        
        if scalar_metrics:
            print("  标量指标:")
            for key, value in list(scalar_metrics.items())[:10]:
                print(f"    {key:<25}: {value:.6f}")
        
        if tensor_metrics:
            print("  张量指标:")
            for key, value in list(tensor_metrics.items())[:5]:
                if isinstance(value, torch.Tensor):
                    print(f"    {key:<25}: shape={tuple(value.shape)}, mean={value.float().mean():.6f}")
    
    # 6. 其他重要信息
    other_keys = ["seed", "loss", "reward_mean", "qed", "sa", "valid_ratio", "batch_stats"]
    for key in other_keys:
        if key in checkpoint:
            value = checkpoint[key]
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                print(f"\n  📊 {key}: {value.item():.6f}")
            else:
                print(f"\n  📊 {key}: {value}")
    
    # 7. 生成建议
    print("\n" + "="*80)
    print("💡 分析与建议")
    print("="*80)
    
    suggestions = []
    
    # 检查模型权重
    for model_key in model_state_keys:
        if model_key in checkpoint:
            state_dict = checkpoint[model_key]
            
            # 检查是否有梯度
            requires_grad_params = 0
            for param in state_dict.values():
                if isinstance(param, torch.Tensor) and hasattr(param, 'requires_grad'):
                    if param.requires_grad:
                        requires_grad_params += 1
            
            if requires_grad_params > 0:
                suggestions.append(f"模型有 {requires_grad_params} 个参数需要梯度（训练中）")
            else:
                suggestions.append("模型参数不需要梯度（可能是评估模式）")
            
            break
    
    # 检查优化器状态
    for opt_key in optimizer_keys:
        if opt_key in checkpoint:
            suggestions.append("包含优化器状态，可用于恢复训练")
            break
    
    # 检查检查点完整性
    required_keys = ["epoch", "model_state_dict"]
    missing_keys = [k for k in required_keys if k not in checkpoint]
    if missing_keys:
        suggestions.append(f"⚠️  缺少关键字段: {missing_keys}")
    else:
        suggestions.append("✅ 检查点包含关键字段，完整性良好")
    
    # 打印建议
    for i, suggestion in enumerate(suggestions, 1):
        print(f"  {i}. {suggestion}")
    
    # 8. 比较两个检查点（如果提供了比较路径）
    if compare_path and Path(compare_path).exists():
        print("\n" + "="*80)
        print("🔀 比较检查点")
        print("="*80)
        
        try:
            checkpoint2 = torch.load(compare_path, map_location='cpu', weights_only=False)
            
            # 比较结构
            keys1 = set(checkpoint.keys())
            keys2 = set(checkpoint2.keys())
            
            print(f"  检查点1的键: {len(keys1)}")
            print(f"  检查点2的键: {len(keys2)}")
            print(f"  共同的键: {len(keys1 & keys2)}")
            print(f"  检查点1独有的键: {keys1 - keys2}")
            print(f"  检查点2独有的键: {keys2 - keys1}")
            
            # 比较模型权重
            for model_key in model_state_keys:
                if model_key in checkpoint and model_key in checkpoint2:
                    print(f"\n  🔄 比较 {model_key}:")
                    
                    params1 = checkpoint[model_key]
                    params2 = checkpoint2[model_key]
                    
                    # 检查参数差异
                    diff_norm = 0
                    total_params = 0
                    
                    for key in set(params1.keys()) & set(params2.keys()):
                        if isinstance(params1[key], torch.Tensor) and isinstance(params2[key], torch.Tensor):
                            diff = torch.norm(params1[key].float() - params2[key].float())
                            diff_norm += diff.item() ** 2
                            total_params += params1[key].numel()
                    
                    if total_params > 0:
                        avg_diff = np.sqrt(diff_norm) / len(params1)
                        print(f"    平均参数差异: {avg_diff:.6f}")
                    
                    break
            
        except Exception as e:
            print(f"❌ 比较失败: {e}")
    
    return result

# =========================================
# 批量分析函数
# =========================================

def analyze_checkpoint_directory(directory_path: str, pattern: str = "*.pt"):
    """分析目录中的所有检查点"""
    dir_path = Path(directory_path)
    if not dir_path.exists():
        print(f"❌ 目录不存在: {directory_path}")
        return
    
    checkpoint_files = list(dir_path.glob(pattern)) + list(dir_path.glob("*.ckpt"))
    
    if not checkpoint_files:
        print(f"❌ 没有找到{pattern}文件")
        return
    
    print(f"📁 找到 {len(checkpoint_files)} 个检查点文件")
    
    # 按修改时间排序
    checkpoint_files.sort(key=lambda x: x.stat().st_mtime)
    
    results = []
    
    for i, checkpoint_file in enumerate(checkpoint_files):
        print(f"\n{'='*80}")
        print(f"📄 分析文件 {i+1}/{len(checkpoint_files)}: {checkpoint_file.name}")
        print(f"{'='*80}")
        
        try:
            result = analyze_checkpoint(str(checkpoint_file))
            result["filename"] = checkpoint_file.name
            result["filetime"] = checkpoint_file.stat().st_mtime
            results.append(result)
        except Exception as e:
            print(f"❌ 分析失败: {e}")
    
    # 生成汇总报告
    if results:
        print("\n" + "="*80)
        print("📊 汇总报告")
        print("="*80)
        
        print(f"📁 分析目录: {directory_path}")
        print(f"📄 分析文件数: {len(results)}")
        print(f"📅 时间范围: {Path(results[0]['file_path']).name} 到 {Path(results[-1]['file_path']).name}")
        
        # 按类型分组
        type_counter = Counter([r.get("checkpoint_type", "unknown") for r in results])
        print(f"\n📊 检查点类型分布:")
        for ctype, count in type_counter.items():
            print(f"  {ctype:<20}: {count} 个")
        
        # 文件大小统计
        sizes = [r["file_size_bytes"] for r in results]
        if sizes:
            print(f"\n💾 文件大小统计:")
            print(f"  平均大小: {format_size(np.mean(sizes))}")
            print(f"  最小大小: {format_size(np.min(sizes))}")
            print(f"  最大大小: {format_size(np.max(sizes))}")
            print(f"  总大小: {format_size(np.sum(sizes))}")

# =========================================
# 主程序
# =========================================

if __name__ == "__main__":
    print("🚀 PyTorch检查点分析工具 v1.0")
    print("="*80)
    
    # 如果CHECKPOINT_PATH是目录，则分析目录
    if Path(CHECKPOINT_PATH).is_dir():
        print(f"📁 检测到目录，将分析目录中的所有.pt和.ckpt文件")
        analyze_checkpoint_directory(CHECKPOINT_PATH)
    else:
        # 分析单个文件
        result = analyze_checkpoint(CHECKPOINT_PATH, COMPARE_PATH)
        
        # 保存分析结果
        output_file = f"{Path(CHECKPOINT_PATH).stem}_analysis.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n💾 分析结果已保存到: {output_file}")
    
    print("\n" + "="*80)
    print("✅ 分析完成")
    print("="*80)