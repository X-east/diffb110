import argparse
import os
import sys
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def print_progress_bar(current, total, prefix='', suffix='', length=50):
    """显示单行进度条"""
    if total == 0: return
    filled = int(length * current // total)
    bar = '█' * filled + '-' * (length - filled)
    percent = f"{100 * current / total:.1f}"
    progress_str = f'{prefix} |{bar}| {percent}% {suffix}'
    sys.stdout.write(f'\r\033[K{progress_str}')
    sys.stdout.flush()

def filter_molecules(input_path, output_path, sanitize=True, keep_largest=True):
    """
    筛选有效分子
    
    Args:
        input_path: 输入 SDF 路径
        output_path: 输出 SDF 路径
        sanitize: 是否执行 RDKit Sanitization
        keep_largest: 是否仅保留最大片段
    """
    if not os.path.exists(input_path):
        logger.error(f"找不到输入文件: {input_path}")
        return

    # 使用 sanitize=False 加载，手动控制校验过程
    suppl = Chem.SDMolSupplier(str(input_path), sanitize=False, removeHs=False)
    total_count = len(suppl)
    
    logger.info(f"开始处理文件: {input_path} (总计 {total_count} 个分子)")
    
    # 准备输出目录
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(output_path))
    
    valid_count = 0
    failed_rdkit = 0
    failed_sanitize = 0
    processed_count = 0
    
    print() # 为进度条留行

    for i, mol in enumerate(suppl):
        processed_count += 1
        if i % 10 == 0 or i == total_count - 1:
            print_progress_bar(i + 1, total_count, prefix='处理进度', suffix=f'已发现有效: {valid_count}')

        # 1. RDKit 解析检查
        if mol is None:
            failed_rdkit += 1
            continue
        
        # 2. Sanitization 检查 (化学有效性)
        if sanitize:
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                failed_sanitize += 1
                continue
        
        # 3. 片段处理 (可选)
        if keep_largest:
            frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
            if len(frags) > 1:
                # 按照原子数排序，取最大的
                mol = max(frags, key=lambda m: m.GetNumAtoms())
        
        # 写入有效分子
        writer.write(mol)
        valid_count += 1

    writer.close()
    print() # 换行
    
    logger.info("-" * 40)
    logger.info(f"处理完成！")
    logger.info(f"总分子数: {total_count}")
    logger.info(f"有效分子: {valid_count} ({valid_count/total_count*100:.1f}%)")
    logger.info(f"RDKit 解析失败: {failed_rdkit}")
    logger.info(f"化学校验(Sanitize)失败: {failed_sanitize}")
    logger.info(f"结果已保存至: {output_path}")
    logger.info("-" * 40)

def main():
    parser = argparse.ArgumentParser(description="对生成的分子进行有效性筛选 (SDF 格式)")
    parser.add_argument("input", help="输入的 SDF 文件路径")
    parser.add_argument("--output", "-o", help="输出的 SDF 文件路径 (默认在原文件名后加 _filtered)")
    parser.add_argument("--no_sanitize", action="store_false", dest="sanitize", help="关闭 RDKit Sanitization 校验")
    parser.add_argument("--all_frags", action="store_false", dest="keep_largest", help="保留所有片段 (不执行最大片段提取)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if args.output:
        output_path = Path(args.output)
    else:
        # 默认输出路径: 路径/文件名_filtered.sdf
        output_path = input_path.parent / f"{input_path.stem}_filtered.sdf"
    
    filter_molecules(
        input_path, 
        output_path, 
        sanitize=args.sanitize, 
        keep_largest=args.keep_largest
    )

if __name__ == "__main__":
    main()
