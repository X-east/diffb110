#!/usr/bin/env python3
"""
诊断 UniDock 返回 0 分的原因
"""

import sys
import os
import tempfile
import subprocess
from pathlib import Path

# Add DiffSBDD to path
sys.path.insert(0, str(Path(__file__).parent.parent / "DiffSBDD"))

import torch
from analysis.molecule_builder import build_molecule, process_molecule
from constants import dataset_params
import utils


def test_molecule_generation():
    """测试分子生成"""
    print("=" * 70)
    print("1. 测试分子生成")
    print("=" * 70)
    
    # 创建一个简单的测试分子（甲烷 CH4）
    # Use the correct dataset key
    dataset_info = {
        'atom_decoder': ['C', 'N', 'O', 'S', 'B', 'Br', 'Cl', 'P', 'I', 'F'],
        'atom_encoder': {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'B': 4, 'Br': 5, 'Cl': 6, 'P': 7, 'I': 8, 'F': 9}
    }
    
    # C atom at origin
    positions = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    atom_types = torch.tensor([0], dtype=torch.long)  # C is index 0
    
    mol = build_molecule(
        positions=positions,
        atom_types=atom_types,
        dataset_info=dataset_info,
        add_coords=True,
        use_openbabel=True
    )
    
    if mol is None:
        print("❌ 分子生成失败")
        return None
    else:
        print(f"✓ 分子生成成功: {mol.GetNumAtoms()} atoms")
        return mol


def test_sdf_conversion(mol):
    """测试 SDF 文件写入"""
    print("\n" + "=" * 70)
    print("2. 测试 SDF 文件写入")
    print("=" * 70)
    
    with tempfile.NamedTemporaryFile(suffix='.sdf', delete=False) as tmp:
        sdf_file = tmp.name
    
    try:
        utils.write_sdf_file(sdf_file, [mol])
        
        if os.path.exists(sdf_file) and os.path.getsize(sdf_file) > 0:
            print(f"✓ SDF 文件写入成功: {sdf_file}")
            print(f"  文件大小: {os.path.getsize(sdf_file)} bytes")
            
            # 显示文件内容
            with open(sdf_file, 'r') as f:
                content = f.read()
                print(f"  内容预览:\n{content[:200]}...")
            
            return sdf_file
        else:
            print("❌ SDF 文件写入失败")
            return None
    except Exception as e:
        print(f"❌ SDF 写入异常: {e}")
        return None


def test_pdbqt_conversion(sdf_file):
    """测试 PDBQT 转换"""
    print("\n" + "=" * 70)
    print("3. 测试 SDF → PDBQT 转换")
    print("=" * 70)
    
    with tempfile.NamedTemporaryFile(suffix='.pdbqt', delete=False) as tmp:
        pdbqt_file = tmp.name
    
    try:
        # 检查 obabel 是否可用
        result = subprocess.run(
            ['obabel', '--version'],
            capture_output=True,
            text=True
        )
        print(f"✓ OpenBabel 版本: {result.stdout.strip()}")
        
        # 转换
        result = subprocess.run(
            ['obabel', '-isdf', sdf_file, '-opdbqt', '-O', pdbqt_file, '-h'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        print(f"  返回码: {result.returncode}")
        if result.stdout:
            print(f"  标准输出:\n{result.stdout}")
        if result.stderr:
            print(f"  标准错误:\n{result.stderr}")
        
        if os.path.exists(pdbqt_file) and os.path.getsize(pdbqt_file) > 0:
            print(f"✓ PDBQT 转换成功: {pdbqt_file}")
            print(f"  文件大小: {os.path.getsize(pdbqt_file)} bytes")
            
            # 显示文件内容
            with open(pdbqt_file, 'r') as f:
                content = f.read()
                print(f"  内容预览:\n{content[:300]}...")
            
            return pdbqt_file
        else:
            print("❌ PDBQT 转换失败")
            return None
            
    except Exception as e:
        print(f"❌ PDBQT 转换异常: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_receptor_file():
    """测试受体文件"""
    print("\n" + "=" * 70)
    print("4. 测试受体文件")
    print("=" * 70)
    
    receptor_pdb = "data/8GJJ-C-Cip.pdb"
    receptor_pdbqt = "data/8GJJ-C-Cip.pdbqt"
    
    # 检查 PDB 文件
    if os.path.exists(receptor_pdb):
        print(f"✓ PDB 文件存在: {receptor_pdb}")
        print(f"  文件大小: {os.path.getsize(receptor_pdb)} bytes")
    else:
        print(f"❌ PDB 文件不存在: {receptor_pdb}")
    
    # 检查 PDBQT 文件
    if os.path.exists(receptor_pdbqt):
        print(f"✓ PDBQT 文件存在: {receptor_pdbqt}")
        print(f"  文件大小: {os.path.getsize(receptor_pdbqt)} bytes")
        return receptor_pdbqt
    else:
        print(f"❌ PDBQT 文件不存在: {receptor_pdbqt}")
        print(f"  需要运行: bash scripts/prepare_receptor.sh")
        return None


def test_unidock_docking(ligand_pdbqt, receptor_pdbqt):
    """测试 UniDock docking"""
    print("\n" + "=" * 70)
    print("5. 测试 UniDock Docking")
    print("=" * 70)
    
    if ligand_pdbqt is None or receptor_pdbqt is None:
        print("❌ 缺少必要文件，跳过 docking 测试")
        return
    
    # 检查 unidock 是否可用
    try:
        result = subprocess.run(
            ['unidock', '--version'],
            capture_output=True,
            text=True
        )
        print(f"✓ UniDock 版本: {result.stdout.strip()}")
    except FileNotFoundError:
        print("❌ UniDock 未安装")
        return
    
    # 创建临时输出目录
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"  输出目录: {tmpdir}")
        
        # 构建 unidock 命令
        cmd = [
            'unidock',
            '--receptor', receptor_pdbqt,
            '--gpu_batch', ligand_pdbqt,
            '--search_mode', 'fast',
            '--num_modes', '1',
            '--center_x', '183.0',
            '--center_y', '180.0',
            '--center_z', '230.0',
            '--size_x', '25.0',
            '--size_y', '25.0',
            '--size_z', '25.0',
            '--dir', tmpdir
        ]
        
        print(f"  命令: {' '.join(cmd)}")
        
        try:
            # 设置 GPU 环境变量
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = '0'
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                env=env
            )
            
            print(f"  返回码: {result.returncode}")
            
            if result.stdout:
                print(f"  标准输出:\n{result.stdout}")
            
            if result.stderr:
                print(f"  标准错误:\n{result.stderr}")
            
            # 检查输出文件
            ligand_basename = Path(ligand_pdbqt).stem
            output_file = os.path.join(tmpdir, f"{ligand_basename}_out.pdbqt")
            
            if os.path.exists(output_file):
                print(f"✓ 输出文件生成: {output_file}")
                print(f"  文件大小: {os.path.getsize(output_file)} bytes")
                
                # 解析分数
                with open(output_file, 'r') as f:
                    content = f.read()
                    print(f"  内容预览:\n{content[:500]}...")
                    
                    import re
                    match = re.search(r'REMARK VINA RESULT:\s+([-+]?[0-9]*\.?[0-9]+)', content)
                    if match:
                        score = float(match.group(1))
                        print(f"\n✓ Docking 分数: {score} kcal/mol")
                    else:
                        print("\n❌ 未找到 VINA RESULT")
            else:
                print(f"❌ 输出文件未生成: {output_file}")
                
                # 列出目录内容
                print(f"  目录内容:")
                for item in os.listdir(tmpdir):
                    print(f"    - {item}")
                    
        except subprocess.TimeoutExpired:
            print("❌ Docking 超时")
        except Exception as e:
            print(f"❌ Docking 异常: {e}")
            import traceback
            traceback.print_exc()


def main():
    print("\n" + "=" * 70)
    print("UniDock 零分诊断工具")
    print("=" * 70)
    
    # 1. 测试分子生成
    mol = test_molecule_generation()
    if mol is None:
        print("\n❌ 分子生成失败，无法继续")
        return
    
    # 2. 测试 SDF 转换
    sdf_file = test_sdf_conversion(mol)
    if sdf_file is None:
        print("\n❌ SDF 转换失败，无法继续")
        return
    
    # 3. 测试 PDBQT 转换
    pdbqt_file = test_pdbqt_conversion(sdf_file)
    
    # 4. 测试受体文件
    receptor_pdbqt = test_receptor_file()
    
    # 5. 测试 UniDock docking
    test_unidock_docking(pdbqt_file, receptor_pdbqt)
    
    print("\n" + "=" * 70)
    print("诊断完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
