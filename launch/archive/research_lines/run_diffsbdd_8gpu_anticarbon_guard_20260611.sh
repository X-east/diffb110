#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="$ROOT/launch/diffsbdd_8gpu_anticarbon_guard_20260611.yaml"
LOG_PATH="$ROOT/launch/diffsbdd_8gpu_anticarbon_guard_20260611.nohup.log"

cd "$ROOT"

echo "[DiffSBDD-8GPU] anti-carbon-drift formal test"
echo "[DiffSBDD-8GPU] config=$CONFIG_PATH"
echo "[DiffSBDD-8GPU] reward=docking+LE+ECFP/FCFP+logP_distribution+chemistry_guard"
echo "[DiffSBDD-8GPU] batch=8GPU x 64, rollout_accum=1, ppo_epochs=2"
echo "[DiffSBDD-8GPU] log=$LOG_PATH"

CONFIG_PATH="$CONFIG_PATH" LOG_LEVEL=INFO bash "$ROOT/launch/run.sh"
