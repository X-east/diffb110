#!/bin/bash
# 一键生成1000个分子（基于DDPO checkpoint）

cd /home/md666/data/disk6/DiffB110

echo "生成1000个分子（4GPU，每批20个）..."
python generate_from_ddpo_batch.py \
    --ddpo_ckpt OUT/4gpu-druglikeness-128mol-100batch_20260315_002/checkpoints/epoch=0019.ckpt \
    --pdbfile data/8GJJ-C-Cip.pdb \
    --ref_ligand C:1101 \
    --total 1000 \
    --batch_size 20 \
    --num_gpus 4

echo ""
echo "生成完成！查看结果："
echo "  ls -lh generated_ligands/*/all_ligands_*.sdf"
