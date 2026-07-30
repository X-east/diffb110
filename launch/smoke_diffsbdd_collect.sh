#!/bin/bash
# 冒烟测试 — DiffSBDD 冻结策略轨迹收集（多卡 / 每卡不同随机种子）
# 用法: CONFIG_PATH=launch/smoke_diffsbdd_collect.yaml bash launch/smoke_diffsbdd_collect.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONDA_SH="/home/md666/miniforge3/etc/profile.d/conda.sh"
CONDA_ENV_NAME="dif"
PYTHON_BIN="/home/md666/miniforge3/envs/dif/bin/python"

# 收集使用 2 卡，验证每卡 seed=local_rank+1 不同
SMOKE_GPUS="${SMOKE_GPUS:-0,1}"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_ROOT/launch/smoke_diffsbdd_collect.yaml}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
SMOKE_MASTER_PORT="${SMOKE_MASTER_PORT:-29501}"

[ ! -f "$CONFIG_PATH" ] && echo "[SMOKE ERROR] Config not found: $CONFIG_PATH" && exit 1
[ ! -f "$CONDA_SH" ]    && echo "[SMOKE ERROR] Conda init not found" && exit 1

source "$CONDA_SH"
conda activate "$CONDA_ENV_NAME"
[ ! -x "$PYTHON_BIN" ]  && echo "[SMOKE ERROR] Python not found" && exit 1

# ── 前置检查 ──
_PORT="${SMOKE_MASTER_PORT}"
_PRE_EXISTING_PIDS=$( (lsof -ti tcp:"${_PORT}" 2>/dev/null || true) | sort -n | tr '\n' ' ')
export _SMOKE_PRE_EXISTING_PIDS="$_PRE_EXISTING_PIDS"

# ── 安全清理函数 ──
cleanup_smoke() {
    local _exit_code=$?
    local _my_pid=$$
    echo ""
    echo "[SMOKE CLEANUP] === 清理本次冒烟测试资源 ==="
    _is_safe_to_kill() {
        local _p="$1"
        case "$_p" in
            ''|*[!0-9]*) return 1 ;;
        esac
        [ "$_p" -eq 1 ]   && return 1
        [ "$_p" -eq "$_my_pid" ] && return 1
        return 0
    }
    if [ -n "${_SMOKE_TORCHRUN_PID:-}" ] && _is_safe_to_kill "${_SMOKE_TORCHRUN_PID}"; then
        if kill -0 "${_SMOKE_TORCHRUN_PID}" 2>/dev/null; then
            echo "[SMOKE CLEANUP] 终止 torchrun 进程组 (PGID=${_SMOKE_TORCHRUN_PID})"
            kill -TERM -"${_SMOKE_TORCHRUN_PID}" 2>/dev/null || true
            sleep 2
            kill -KILL -"${_SMOKE_TORCHRUN_PID}" 2>/dev/null || true
            sleep 0.5
        fi
    fi
    local _current_pids
    _current_pids=$( (lsof -ti tcp:"${_PORT}" 2>/dev/null || true) | sort -n)
    if [ -n "$_current_pids" ]; then
        for _pid in $_current_pids; do
            if [ -n "$_SMOKE_PRE_EXISTING_PIDS" ]; then
                echo " $_SMOKE_PRE_EXISTING_PIDS " | grep -qw " $_pid " 2>/dev/null && continue
            fi
            if _is_safe_to_kill "$_pid"; then
                echo "[SMOKE CLEANUP] 杀死残留进程 PID=${_pid} (端口 ${_PORT})"
                kill -KILL "$_pid" 2>/dev/null || true
            fi
        done
    fi
    local _leftover
    _leftover=$( (lsof -ti tcp:"${_PORT}" 2>/dev/null || true) | sort -n)
    if [ -n "$_leftover" ]; then
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
    if command -v nvidia-smi &>/dev/null; then
        echo "[SMOKE CLEANUP] GPU 显存状态:"
        nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader 2>/dev/null || true
    fi
    echo "[SMOKE CLEANUP] === 清理完成 ==="
    exit $_exit_code
}
trap cleanup_smoke EXIT

export BABEL_LIBDIR="/home/md666/miniforge3/envs/dif/lib/openbabel/3.1.0"
export CUDA_VISIBLE_DEVICES="$SMOKE_GPUS"
export PYTHONUNBUFFERED=1

mkdir -p "$PROJECT_ROOT/OUT"
cd "$PROJECT_ROOT"

echo "============================================================"
echo "SMOKE TEST — DiffSBDD 轨迹收集 / ${SMOKE_GPUS} GPU / 1000 steps"
echo "Config : $CONFIG_PATH"
echo "Port   : ${_PORT}"
echo "Start  : $(date '+%F %T')"
echo "Pre-existing PIDs on port ${_PORT}: ${_SMOKE_PRE_EXISTING_PIDS:-none}"
echo "============================================================"

START_TS=$(date +%s)

torchrun \
  --nproc_per_node=2 \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr="127.0.0.1" \
  --master_port="${_PORT}" \
  ddpo/main_diffsbdd_collect_ddp.py \
    --config "$CONFIG_PATH" \
    --log-level "$LOG_LEVEL" &
_SMOKE_TORCHRUN_PID=$!
export _SMOKE_TORCHRUN_PID

set +e
wait "$_SMOKE_TORCHRUN_PID"
_SMOKE_EXIT_CODE=$?
set -e

ELAPSED=$(( $(date +%s) - START_TS ))
if [ $_SMOKE_EXIT_CODE -ne 0 ]; then
    echo "============================================================"
    echo "SMOKE TEST FAILED (exit=${_SMOKE_EXIT_CODE})  —  ${ELAPSED}s"
    echo "============================================================"
    exit $_SMOKE_EXIT_CODE
fi

echo "============================================================"
echo "SMOKE TEST PASSED  —  ${ELAPSED}s"
echo "============================================================"

# ── 轻量校验：shard 形状 + 每卡种子不同 ──
echo "[SMOKE VERIFY] 校验 shard 形状与每卡随机种子差异"
"$PYTHON_BIN" - <<'PY'
import torch, json, glob, os
from pathlib import Path

run_dirs = sorted(glob.glob("runs/diffsbdd_collect/smoke_diffsbdd_collect_*"))
if not run_dirs:
    raise SystemExit("未找到 runs/diffsbdd_collect/smoke_diffsbdd_collect_* 运行目录")
run_dir = Path(run_dirs[-1])
print("run_dir:", run_dir)

shards = sorted(glob.glob(str(run_dir / "shards" / "rank_*" / "shard_*.pt")))
if not shards:
    raise SystemExit("未找到任何 shard")
print("shard 数:", len(shards))

required = {"latents", "logps", "timesteps", "nodesxsample", "x", "categorical"}
all_ok = True
per_rank_smiles = {}
for sh in shards:
    p = torch.load(sh, map_location="cpu", weights_only=False)
    missing = required - set(p.keys())
    if missing:
        print("缺失字段", sh, missing); all_ok = False; continue
    latents = p["latents"]; logps = p["logps"]; timesteps = p["timesteps"]
    md = p["metadata"]
    t = md["num_timesteps"]
    ok = (latents.dim() == 4 and latents.shape[1] == t + 2
          and logps.dim() == 2 and logps.shape[1] == t + 1
          and timesteps.dim() == 2 and timesteps.shape[1] == t + 1
          and p["x"].shape[0] == latents.shape[0])
    print(f"{Path(sh).name}: latents={tuple(latents.shape)} logps={tuple(logps.shape)} "
          f"timesteps={tuple(timesteps.shape)} ok={ok} seed={md['seed']}")
    if not ok:
        all_ok = False
    rank = md["rank"]
    per_rank_smiles.setdefault(rank, set())
    for r in p.get("reconstruction_rows", []):
        if r.get("smiles"):
            per_rank_smiles[rank].add(r["smiles"])

# 每卡种子不同 ⇒ 产出的 SMILES 集合应不同（除非极小概率碰撞）
ranks = sorted(per_rank_smiles.keys())
print("各 rank 有效 SMILES 数:", {r: len(per_rank_smiles[r]) for r in ranks})
if len(ranks) >= 2:
    diff = per_rank_smiles[ranks[0]] != per_rank_smiles[ranks[1]]
    print("rank0 与 rank1 分子集合不同 (证明每卡不同种子):", diff)
    if not diff:
        all_ok = False
else:
    print("注意: 只检测到 1 个 rank 的 shard，无法验证跨卡种子差异")

if not all_ok:
    raise SystemExit("SMOKE VERIFY FAILED")
print("[SMOKE VERIFY] PASSED — 轨迹可存储且每卡随机种子不同")
PY

echo "============================================================"
echo "SMOKE + VERIFY DONE  —  ${ELAPSED}s"
echo "============================================================"
