#!/usr/bin/env python3
"""
generate_from_ddpo_ckpt.py

从 DDPO 训练的 checkpoint 中提取模型权重，加载到原始 LigandPocketDDPM，
然后用 DiffSBDD 的标准生成流程产出分子。

用法示例：
    conda run -n dif python3 scripts/generate_from_ddpo_ckpt.py \\
        --ddpo_ckpt OUT/4gpu-druglikeness-128mol-100batch_20260315_002/checkpoints/epoch=0005.ckpt \\
        --ref_ckpt  DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt \\
        --pdbfile   data/8GJJ-C-Cip.pdb \\
        --ref_ligand C:1101 \\
        --n_samples 100 \\
        --outfile   OUT/generated_epoch5.sdf
"""

import argparse
import sys
from pathlib import Path

import torch

# ── project paths ────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "DiffSBDD"))

try:
    from openbabel import openbabel
    openbabel.obErrorLog.StopLogging()
except Exception:
    pass

import utils
from lightning_modules import LigandPocketDDPM


def extract_ddpo_state_dict(ddpo_ckpt_path: str) -> dict:
    """
    从 DDPO checkpoint 的 state_dict 中提取模型权重。

    DDPO checkpoint 的 state_dict key 格式：
        model.model.ddpm.*   (LightningDDPOModule.model.model.ddpm)
    原始 LigandPocketDDPM 的 state_dict key 格式：
        ddpm.*

    转换：去掉 'model.model.' 前缀。
    """
    print(f"Loading DDPO checkpoint: {ddpo_ckpt_path}")
    ckpt = torch.load(ddpo_ckpt_path, map_location='cpu')

    sd = ckpt.get('state_dict', {})
    epoch = ckpt.get('epoch', '?')
    print(f"  Checkpoint epoch: {epoch}")
    print(f"  state_dict tensors: {len(sd)}")

    # 提取 model.model.* 前缀的权重，去掉 'model.model.' 前缀
    prefix = 'model.model.'
    converted = {}
    for k, v in sd.items():
        if k.startswith(prefix):
            new_key = k[len(prefix):]
            converted[new_key] = v

    print(f"  Extracted tensors (model.model.* -> *): {len(converted)}")
    if not converted:
        raise ValueError(
            f"No keys with prefix '{prefix}' found in state_dict. "
            f"Available prefixes: {set('.'.join(k.split('.')[:2]) for k in sd.keys())}"
        )
    return converted


def main():
    parser = argparse.ArgumentParser(
        description="Generate molecules from DDPO-trained checkpoint"
    )
    parser.add_argument('--ddpo_ckpt', type=str, required=True,
                        help='Path to DDPO checkpoint (.ckpt)')
    parser.add_argument('--ref_ckpt', type=str,
                        default='DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt',
                        help='Reference DiffSBDD checkpoint for model architecture')
    parser.add_argument('--pdbfile', type=str, required=True,
                        help='Pocket PDB file')
    parser.add_argument('--ref_ligand', type=str, default=None,
                        help='Reference ligand (e.g. C:1101)')
    parser.add_argument('--resi_list', type=str, nargs='+', default=None,
                        help='Residue list for pocket (alternative to ref_ligand)')
    parser.add_argument('--outfile', type=str, default='generated.sdf',
                        help='Output SDF file path')
    parser.add_argument('--n_samples', type=int, default=100,
                        help='Number of molecules to generate')
    parser.add_argument('--batch_size', type=int, default=20,
                        help='Batch size for generation')
    parser.add_argument('--sanitize', action='store_true', default=True,
                        help='Sanitize generated molecules (default: True)')
    parser.add_argument('--relax', action='store_true',
                        help='Relax generated molecules')
    parser.add_argument('--timesteps', type=int, default=None,
                        help='Number of diffusion timesteps (default: model default)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # ── Step 1: 从 DDPO checkpoint 提取权重 ──────────────────────────────────
    ddpo_sd = extract_ddpo_state_dict(args.ddpo_ckpt)

    # ── Step 2: 加载原始 LigandPocketDDPM 架构（用 ref_ckpt 初始化结构）───────
    print(f"\nLoading model architecture from: {args.ref_ckpt}")
    model = LigandPocketDDPM.load_from_checkpoint(
        args.ref_ckpt, map_location=device
    )
    print("  Architecture loaded.")

    # ── Step 3: 替换权重为 DDPO 训练后的权重 ────────────────────────────────
    # ddpo_sd 的 key 格式是 'ddpm.*'（已去掉 'model.model.' 前缀）
    # LigandPocketDDPM 的 state_dict key 格式也是 'ddpm.*'
    missing, unexpected = model.load_state_dict(ddpo_sd, strict=False)
    if missing:
        print(f"  WARNING: Missing keys ({len(missing)}): {missing[:5]}")
    if unexpected:
        print(f"  WARNING: Unexpected keys ({len(unexpected)}): {unexpected[:5]}")
    if not missing and not unexpected:
        print("  Weights loaded successfully (strict match).")
    else:
        print(f"  Weights loaded with {len(missing)} missing, {len(unexpected)} unexpected keys.")

    model = model.to(device)
    model.eval()
    print("  Model set to eval mode.")

    # ── Step 4: 生成分子 ─────────────────────────────────────────────────────
    assert args.n_samples % args.batch_size == 0, \
        f"n_samples ({args.n_samples}) must be divisible by batch_size ({args.batch_size})"

    outfile = Path(args.outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating {args.n_samples} molecules")
    print(f"  PDB: {args.pdbfile}")
    print(f"  Ref ligand: {args.ref_ligand}")
    print(f"  Output: {outfile}")
    print(f"  Batch size: {args.batch_size}")
    print()

    molecules = []
    n_batches = args.n_samples // args.batch_size
    for i in range(n_batches):
        print(f"  Batch {i+1}/{n_batches} ...", end='', flush=True)
        batch = model.generate_ligands(
            args.pdbfile,
            args.batch_size,
            args.resi_list,
            args.ref_ligand,
            num_nodes_lig=None,
            sanitize=args.sanitize,
            largest_frag=True,
            relax_iter=(200 if args.relax else 0),
            resamplings=10,
            jump_length=1,
            timesteps=args.timesteps,
        )
        molecules.extend(batch)
        valid = sum(1 for m in batch if m is not None)
        print(f" {valid}/{args.batch_size} valid")

    # ── Step 5: 写出 SDF ──────────────────────────────────────────────────────
    utils.write_sdf_file(outfile, molecules)
    valid_total = sum(1 for m in molecules if m is not None)
    print(f"\nDone! {valid_total}/{args.n_samples} valid molecules written to: {outfile}")


if __name__ == '__main__':
    main()
