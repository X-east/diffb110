#!/bin/bash

# ============================================================================
# 多GPU分批生成配体脚本
# 用法:
#   bash generate_ligands_batch.sh [--num_gpus N] [--batch_size B] [--total N]
# ============================================================================

set -e  # 遇到错误立即退出

# ── 默认参数 ────────────────────────────────────────────────────────────
CHECKPOINT="/home/md666/data/disk6/DiffB110/OUT/4gpu-druglikeness-128mol-100batch_20260315_002/checkpoints/epoch=0019.ckpt"
PDBFILE="/home/md666/data/disk6/DiffB110/data/8GJJ-C-Cip.pdb"
REF_LIGAND="C:1101"
TOTAL_SAMPLES=1000
BATCH_SIZE=20          # 每个GPU每批生成的分子数（避免OOM）
NUM_GPUS=4             # 使用的GPU数量
OUTPUT_DIR="./generated_ligands"

# ── 解析命令行参数 ──────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --pdbfile)
            PDBFILE="$2"
            shift 2
            ;;
        --ref_ligand)
            REF_LIGAND="$2"
            shift 2
            ;;
        --total)
            TOTAL_SAMPLES="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# ── 计算分配策略 ────────────────────────────────────────────────────────
# 确保每个GPU分配的分子数能被batch_size整除
# 否则 generate_ligands.py 会因为 assert n_samples % batch_size == 0 而失败
SAMPLES_PER_GPU=$((TOTAL_SAMPLES / NUM_GPUS))
REMAINDER=$((TOTAL_SAMPLES % NUM_GPUS))

# 调整batch_size使其能整除每GPU分子数
# 如果不能整除，自动减小batch_size
if [ $((SAMPLES_PER_GPU % BATCH_SIZE)) -ne 0 ]; then
    # 找到最大的能整除SAMPLES_PER_GPU的batch_size
    for bs in $(seq $BATCH_SIZE -1 1); do
        if [ $((SAMPLES_PER_GPU % bs)) -eq 0 ]; then
            BATCH_SIZE=$bs
            break
        fi
    done
fi

# 创建输出目录（带时间戳）
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${OUTPUT_DIR}/${TIMESTAMP}_${TOTAL_SAMPLES}mols_${NUM_GPUS}gpus"
mkdir -p "$RUN_DIR"

echo "============================================================================"
echo "多GPU分批生成配体"
echo "============================================================================"
echo "Checkpoint:     $CHECKPOINT"
echo "PDB文件:        $PDBFILE"
echo "参考配体:       $REF_LIGAND"
echo "总分子数:       $TOTAL_SAMPLES"
echo "每批大小:       $BATCH_SIZE"
echo "GPU数量:        $NUM_GPUS"
echo "每GPU分子数:    $SAMPLES_PER_GPU (余数: $REMAINDER)"
echo "输出目录:       $RUN_DIR"
echo "============================================================================"

# ── 启动多GPU并行生成 ───────────────────────────────────────────────────
PIDS=()
for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
    # 计算该GPU需要生成的分子数（最后一个GPU处理余数）
    if [ $gpu_id -eq $((NUM_GPUS - 1)) ]; then
        N_SAMPLES=$((SAMPLES_PER_GPU + REMAINDER))
    else
        N_SAMPLES=$SAMPLES_PER_GPU
    fi
    
    # 跳过空任务
    if [ $N_SAMPLES -eq 0 ]; then
        continue
    fi
    
    # 输出文件名
    OUTFILE="${RUN_DIR}/ligands_gpu${gpu_id}.sdf"
    LOGFILE="${RUN_DIR}/gpu${gpu_id}.log"
    
    echo "[GPU $gpu_id] 生成 $N_SAMPLES 个分子 → $OUTFILE"
    
    # 后台启动生成任务（使用不同的随机种子确保不重复）
    CUDA_VISIBLE_DEVICES=$gpu_id python DiffSBDD/generate_ligands.py \
        "$CHECKPOINT" \
        --pdbfile "$PDBFILE" \
        --ref_ligand "$REF_LIGAND" \
        --outfile "$OUTFILE" \
        --n_samples $N_SAMPLES \
        --batch_size $BATCH_SIZE \
        --sanitize \
        > "$LOGFILE" 2>&1 &
    
    PIDS+=($!)
done

# ── 等待所有GPU任务完成 ─────────────────────────────────────────────────
echo ""
echo "等待所有GPU任务完成..."
FAILED=0
for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}
    if wait $pid; then
        echo "[GPU $i] ✓ 完成 (PID: $pid)"
    else
        echo "[GPU $i] ✗ 失败 (PID: $pid, exit code: $?)"
        FAILED=$((FAILED + 1))
    fi
done

# ── 合并所有SDF文件 ─────────────────────────────────────────────────────
if [ $FAILED -eq 0 ]; then
    echo ""
    echo "合并所有SDF文件..."
    MERGED_FILE="${RUN_DIR}/all_ligands_${TOTAL_SAMPLES}.sdf"
    cat ${RUN_DIR}/ligands_gpu*.sdf > "$MERGED_FILE"
    
    # 统计生成的分子数
    ACTUAL_COUNT=$(grep -c '$$$$' "$MERGED_FILE" || echo 0)
    
    echo "============================================================================"
    echo "✓ 生成完成"
    echo "============================================================================"
    echo "预期分子数:     $TOTAL_SAMPLES"
    echo "实际生成数:     $ACTUAL_COUNT"
    echo "合并文件:       $MERGED_FILE"
    echo "各GPU日志:      ${RUN_DIR}/gpu*.log"
    echo "============================================================================"
    
    # 可选：删除单独的GPU文件以节省空间
    # rm ${RUN_DIR}/ligands_gpu*.sdf
else
    echo ""
    echo "============================================================================"
    echo "✗ 有 $FAILED 个GPU任务失败，请检查日志文件"
    echo "============================================================================"
    exit 1
fi
