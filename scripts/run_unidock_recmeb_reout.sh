#!/usr/bin/env bash
set -euo pipefail

# ===========================================
# RE-CmeB / UniDock 批量对接脚本
# ===========================================

# 输入文件
RECEPTOR_PDBQT="/home/md666/data/disk6/DiffB110/data/Recmeb_receptor.pdbqt"
LIGAND_SDF="/home/md666/data/disk6/DiffB110/bishe/RE-out/260407generated_10000_lipinski_filtered.sdf"

# 输出目录
OUT_DIR="/home/md666/data/disk6/DiffB110/bishe/RE-out/unidock_260407generated_10000_lipinski_filtered"

# 盒子参数
CENTER_X="207.568"
CENTER_Y="212.495"
CENTER_Z="172.884"
SIZE_X="20"
SIZE_Y="20"
SIZE_Z="20"

# UniDock 参数
SCORING="vina"
SEARCH_MODE="balance"      # fast / balance / detail
NUM_MODES="1"
SEED="42"
VERBOSITY="1"

UNIDOCK_BIN="/opt/unidock-1.1.3/bin/unidock"


echo "[Step 0] Checking required files and tools..."

[[ -x "${UNIDOCK_BIN}" ]] || { echo "ERROR: unidock not executable: ${UNIDOCK_BIN}"; exit 1; }
[[ -f "${RECEPTOR_PDBQT}" ]] || { echo "ERROR: receptor PDBQT not found: ${RECEPTOR_PDBQT}"; exit 1; }
[[ -f "${LIGAND_SDF}" ]] || { echo "ERROR: ligand SDF not found: ${LIGAND_SDF}"; exit 1; }

mkdir -p "${OUT_DIR}"


echo "unidock: ${UNIDOCK_BIN}"
echo "receptor: ${RECEPTOR_PDBQT}"
echo "ligands: ${LIGAND_SDF}"
echo "output: ${OUT_DIR}"

echo "[Step 1] Running UniDock batch docking..."
"${UNIDOCK_BIN}" \
  --receptor "${RECEPTOR_PDBQT}" \
  --gpu_batch "${LIGAND_SDF}" \
  --center_x "${CENTER_X}" \
  --center_y "${CENTER_Y}" \
  --center_z "${CENTER_Z}" \
  --size_x "${SIZE_X}" \
  --size_y "${SIZE_Y}" \
  --size_z "${SIZE_Z}" \
  --scoring "${SCORING}" \
  --search_mode "${SEARCH_MODE}" \
  --num_modes "${NUM_MODES}" \
  --seed "${SEED}" \
  --verbosity "${VERBOSITY}" \
  --dir "${OUT_DIR}"

echo "[Step 2] Listing output files..."
/bin/ls -lh "${OUT_DIR}"

echo "Done. Results are in: ${OUT_DIR}"
