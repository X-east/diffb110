#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-$ROOT/launch/diffgui_deppa_pbfast_cpu4_lr1e5_wd1e4_b8x7_t200_100ep_20260715.yaml}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-7}"
MASTER_PORT="${MASTER_PORT:-29876}"
CONDA_ENV="${CONDA_ENV:-diffgui}"
PB_ENV="${PB_ENV:-dif}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/runs/diffgui-deppa-pbfast-cpu4-lr1e5-wd1e4-b8x7-t200-100ep_${RUN_STAMP}}"
LOG_PATH="${LOG_PATH:-$OUTPUT_DIR/logs/training.log}"
PRECHECK_ONLY="${PRECHECK_ONLY:-0}"

export CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

mkdir -p "$OUTPUT_DIR/logs" "$OUTPUT_DIR/config" "$OUTPUT_DIR/launch_snapshot"
cp "$CONFIG_PATH" "$OUTPUT_DIR/config/launch_config.yaml"
cp "${BASH_SOURCE[0]}" "$OUTPUT_DIR/launch_snapshot/$(basename "${BASH_SOURCE[0]}")"
CONFIG_SNAPSHOT="$OUTPUT_DIR/config/launch_config.yaml"

exec >>"$LOG_PATH" 2>&1

echo "[DiffGui-PBFast] start=$(date '+%F %T')"
echo "[DiffGui-PBFast] root=$ROOT"
echo "[DiffGui-PBFast] config=$CONFIG_SNAPSHOT"
echo "[DiffGui-PBFast] output=$OUTPUT_DIR"
echo "[DiffGui-PBFast] env=$CONDA_ENV pb_env=$PB_ENV"
echo "[DiffGui-PBFast] cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
echo "[DiffGui-PBFast] nproc=$NPROC_PER_NODE batch_per_rank=8 global_batch=56 master_port=$MASTER_PORT"
echo "[DiffGui-PBFast] steps=200 epochs=100 ppo_epochs=1 checkpoint_every=1 seed_mode=gpu_id"
echo "[DiffGui-PBFast] pb_config=dock_fast molecule_timeout_seconds=120 logical_cpus_per_rank=4"

cd "$ROOT"
conda run -n "$PB_ENV" python -c "from posebusters import PoseBusters; pb=PoseBusters(config='dock_fast'); functions=[module.get('function') for module in pb.config['modules']]; assert 'energy_ratio' not in functions; print('PoseBusters dock_fast preflight OK', functions)"
conda run -n "$CONDA_ENV" python -c "import torch, yaml; cfg=yaml.safe_load(open('$CONFIG_SNAPSHOT')); model=cfg['diffgui_model']; train=cfg['diffgui_training']; pb=cfg['diffgui_posebusters_filter']; assert model['checkpoint']==model['base_checkpoint']=='reference project/DiffGui/ckpt/trained.pt'; assert int(train.get('epoch_offset', 0))==0; assert train['epochs']==100; assert train['batch_size']==8; assert train['checkpoint_every_epochs']==1; assert cfg['diffgui_docking_batch_sweep']['num_timesteps']==200; assert cfg['diffgui_docking_batch_sweep']['seed_mode']=='gpu_id'; assert pb['enabled'] is True; assert pb['config']=='dock_fast'; assert pb['molecule_timeout_seconds']==120; assert cfg['diffgui_runtime']['cpu_affinity']['logical_cpus_per_rank']==4; assert cfg['train']['ppo_epochs']==1; print('DiffGui epoch0 config preflight OK', torch.__version__, torch.cuda.device_count())"

if [[ "$PRECHECK_ONLY" == "1" ]]; then
  echo "[DiffGui-PBFast] PRECHECK_ONLY completed"
  exit 0
fi

conda run -n "$CONDA_ENV" torchrun \
  --nproc_per_node="$NPROC_PER_NODE" \
  --master_port="$MASTER_PORT" \
  ddpo/main_diffgui_docking_ddp.py \
  --config "$CONFIG_SNAPSHOT" \
  --output-dir "$OUTPUT_DIR" \
  --log-level "$LOG_LEVEL"

echo "[DiffGui-PBFast] completed=$(date '+%F %T')"
