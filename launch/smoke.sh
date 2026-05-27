#!/bin/bash
# 冒烟测试 — 最小化 1-GPU 1-epoch 快速验证全链路
# 用法: bash launch/smoke.sh [config_path]
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONDA_SH="/home/md666/miniforge3/etc/profile.d/conda.sh"
CONDA_ENV_NAME="dif"
PYTHON_BIN="/home/md666/miniforge3/envs/dif/bin/python"

CONFIG_PATH="${1:-$PROJECT_ROOT/launch/smoke.yaml}"
LOG_LEVEL="${LOG_LEVEL:-WARNING}"
SMOKE_MASTER_PORT="${SMOKE_MASTER_PORT:-29500}"

# ── 前置检查 ──
[ ! -f "$CONFIG_PATH" ] && echo "[SMOKE ERROR] Config not found: $CONFIG_PATH" && exit 1
[ ! -f "$CONDA_SH" ]    && echo "[SMOKE ERROR] Conda init not found" && exit 1

source "$CONDA_SH"
conda activate "$CONDA_ENV_NAME"
[ ! -x "$PYTHON_BIN" ]  && echo "[SMOKE ERROR] Python not found" && exit 1

# ── 记录启动前已存在的进程（避免清理其他任务）──
_PORT="${SMOKE_MASTER_PORT}"
_PRE_EXISTING_PIDS=$( (lsof -ti tcp:"${_PORT}" 2>/dev/null || true) | sort -n | tr '\n' ' ')
export _SMOKE_PRE_EXISTING_PIDS="$_PRE_EXISTING_PIDS"

# ── 安全清理函数：只杀本次冒烟测试产生的进程 ──
cleanup_smoke() {
    local _exit_code=$?
    local _my_pid=$$  # 自己的 PID，绝对不能杀

    echo ""
    echo "[SMOKE CLEANUP] === 清理本次冒烟测试资源 ==="

    # 辅助函数：验证 PID 是否合法（正整数、非当前 shell、非 PID 1）
    _is_safe_to_kill() {
        local _p="$1"
        case "$_p" in
            ''|*[!0-9]*) return 1 ;;  # 空或非数字 → 拒绝
        esac
        [ "$_p" -eq 1 ]   && return 1  # init → 拒绝
        [ "$_p" -eq "$_my_pid" ] && return 1  # 自己 → 拒绝
        return 0
    }

    # 1. 杀掉 torchrun 进程树（如果还没退出）
    if [ -n "${_SMOKE_TORCHRUN_PID:-}" ] && _is_safe_to_kill "${_SMOKE_TORCHRUN_PID}"; then
        if kill -0 "${_SMOKE_TORCHRUN_PID}" 2>/dev/null; then
            echo "[SMOKE CLEANUP] 终止 torchrun 进程组 (PGID=${_SMOKE_TORCHRUN_PID})"
            # TERM 给进程组，等 2 秒让子进程优雅退出
            kill -TERM -"${_SMOKE_TORCHRUN_PID}" 2>/dev/null || true
            sleep 2
            # 强制 KILL 残余
            kill -KILL -"${_SMOKE_TORCHRUN_PID}" 2>/dev/null || true
            sleep 0.5
        fi
    fi

    # 2. 清理端口上新出现的进程（不碰启动前就存在的进程）
    local _current_pids
    _current_pids=$( (lsof -ti tcp:"${_PORT}" 2>/dev/null || true) | sort -n)
    if [ -n "$_current_pids" ]; then
        for _pid in $_current_pids; do
            # 跳过启动前就存在的 PID
            if [ -n "$_SMOKE_PRE_EXISTING_PIDS" ]; then
                echo " $_SMOKE_PRE_EXISTING_PIDS " | grep -qw " $_pid " 2>/dev/null && continue
            fi
            # 安全检查后再杀
            if _is_safe_to_kill "$_pid"; then
                echo "[SMOKE CLEANUP] 杀死残留进程 PID=${_pid} (端口 ${_PORT})"
                kill -KILL "$_pid" 2>/dev/null || true
            else
                echo "[SMOKE CLEANUP] 跳过不安全 PID=${_pid} (shell/init/非数字)"
            fi
        done
    fi

    # 3. 二次确认：端口是否已清空
    local _leftover
    _leftover=$( (lsof -ti tcp:"${_PORT}" 2>/dev/null || true) | sort -n)
    if [ -n "$_leftover" ]; then
        # 再次过滤启动前 PID
        local _real_leftover=""
        for _pid in $_leftover; do
            if [ -n "$_SMOKE_PRE_EXISTING_PIDS" ]; then
                echo " $_SMOKE_PRE_EXISTING_PIDS " | grep -qw " $_pid " 2>/dev/null && continue
            fi
            _real_leftover="$_real_leftover $_pid"
        done
        if [ -n "$_real_leftover" ]; then
            echo "[SMOKE CLEANUP] ⚠ 警告：端口 ${_PORT} 仍有残留进程:${_real_leftover}"
        else
            echo "[SMOKE CLEANUP] ✓ 端口 ${_PORT} 已清空（仅剩启动前进程）"
        fi
    else
        echo "[SMOKE CLEANUP] ✓ 端口 ${_PORT} 已清空"
    fi

    # 4. 显示 GPU 状态（确认显存已释放）
    if command -v nvidia-smi &>/dev/null; then
        echo "[SMOKE CLEANUP] GPU 显存状态:"
        nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader 2>/dev/null || true
    fi

    echo "[SMOKE CLEANUP] === 清理完成 ==="
    exit $_exit_code
}
trap cleanup_smoke EXIT

# ── 从 YAML 提取运行参数 ──
eval "$("$PYTHON_BIN" "$PROJECT_ROOT/scripts/yaml_launch_env.py" "$CONFIG_PATH")"

export BABEL_LIBDIR="/home/md666/miniforge3/envs/dif/lib/openbabel/3.1.0"
export CUDA_VISIBLE_DEVICES="0"
export PYTHONUNBUFFERED=1

mkdir -p "$PROJECT_ROOT/OUT"
cd "$PROJECT_ROOT"

echo "============================================================"
echo "SMOKE TEST — 1 GPU / 1 Epoch / ${YAML_PER_RANK_BATCH_SIZE} mol/batch"
echo "Config : $CONFIG_PATH"
echo "Port   : ${_PORT}"
echo "Start  : $(date '+%F %T')"
echo "Pre-existing PIDs on port ${_PORT}: ${_SMOKE_PRE_EXISTING_PIDS:-none}"
echo "============================================================"

START_TS=$(date +%s)

# 在子 shell 中启动 torchrun，记录 PID 用于清理
torchrun \
  --nproc_per_node=1 \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr="127.0.0.1" \
  --master_port="${_PORT}" \
  ddpo/main_diffsbdd_ddp.py \
    --config_path "$CONFIG_PATH" \
    --log_level "$LOG_LEVEL" &
_SMOKE_TORCHRUN_PID=$!
export _SMOKE_TORCHRUN_PID

# 等待 torchrun 完成（正常退出或 crash）
set +e
wait "$_SMOKE_TORCHRUN_PID"
_SMOKE_EXIT_CODE=$?
set -e

ELAPSED=$(( $(date +%s) - START_TS ))
if [ $_SMOKE_EXIT_CODE -eq 0 ]; then
    echo "============================================================"
    echo "SMOKE TEST PASSED  —  ${ELAPSED}s"
    echo "============================================================"
else
    echo "============================================================"
    echo "SMOKE TEST FAILED (exit=${_SMOKE_EXIT_CODE})  —  ${ELAPSED}s"
    echo "============================================================"
    exit $_SMOKE_EXIT_CODE
fi
