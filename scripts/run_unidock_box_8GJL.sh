#!/usr/bin/env bash
set -euo pipefail

# ===========================================
# 配置区（按需修改）
# ===========================================

# 1) 已去除共晶配体后的受体 PDB
RECEPTOR_PDB="/home/md666/data/disk6/DiffB110/data/8GJL-minimized.pdb"

# 2) 受体 PDBQT 输出路径（脚本会自动生成）
RECEPTOR_PDBQT="/home/md666/data/disk6/DiffB110/data/8GJL-minimized.pdbqt"

# 3) 待对接配体（你给的是 cip.pdbqt）
LIGAND_PDBQT="/home/md666/data/disk6/DiffB110/data/cip.pdbqt"

# 4) Uni-Dock 输出目录
OUT_DIR="/home/md666/data/disk6/DiffB110/OUT/unidock_test_8GJL_box"

# 5) Box 参数（你提供的中心 + 20A）
CENTER_X="207.293"
CENTER_Y="212.430"
CENTER_Z="172.771"
SIZE_X="20"
SIZE_Y="20"
SIZE_Z="20"

# 6) Uni-Dock 其他参数
SCORING="vina"         # vina / vinardo / ad4
EXHAUSTIVENESS="16"
NUM_MODES="9"
SEED="20260407"

# 7) 工具路径（按你当前环境预置）
UNIDOCK_BIN="$(command -v unidock || true)"
PYTHONSH="/opt/mgltools_x86_64Linux2_1.5.7/bin/pythonsh"
PREPARE_RECEPTOR_PY="/opt/mgltools_x86_64Linux2_1.5.7/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py"


# ===========================================
# Step 0: 基础检查
# ===========================================
echo "[Step 0] Checking required files and tools..."

if [[ -z "${UNIDOCK_BIN}" ]]; then
  echo "ERROR: unidock not found in PATH."
  exit 1
fi

[[ -f "${RECEPTOR_PDB}" ]] || { echo "ERROR: receptor PDB not found: ${RECEPTOR_PDB}"; exit 1; }
[[ -f "${LIGAND_PDBQT}" ]] || { echo "ERROR: ligand PDBQT not found: ${LIGAND_PDBQT}"; exit 1; }
[[ -x "${PYTHONSH}" ]] || { echo "ERROR: pythonsh not executable: ${PYTHONSH}"; exit 1; }
[[ -f "${PREPARE_RECEPTOR_PY}" ]] || { echo "ERROR: prepare_receptor4.py not found: ${PREPARE_RECEPTOR_PY}"; exit 1; }

mkdir -p "${OUT_DIR}"

echo "unidock: ${UNIDOCK_BIN}"
echo "receptor PDB: ${RECEPTOR_PDB}"
echo "ligand PDBQT: ${LIGAND_PDBQT}"
echo "output dir: ${OUT_DIR}"


# ===========================================
# Step 1: 受体 PDB -> PDBQT
# ===========================================
echo "[Step 1] Converting receptor PDB to PDBQT..."
"${PYTHONSH}" "${PREPARE_RECEPTOR_PY}" \
  -r "${RECEPTOR_PDB}" \
  -o "${RECEPTOR_PDBQT}" \
  -A hydrogens

[[ -f "${RECEPTOR_PDBQT}" ]] || { echo "ERROR: receptor PDBQT was not generated: ${RECEPTOR_PDBQT}"; exit 1; }
echo "receptor PDBQT generated: ${RECEPTOR_PDBQT}"


# ===========================================
# Step 2: Uni-Dock box 对接
# ===========================================
echo "[Step 2] Running Uni-Dock (box mode)..."
"${UNIDOCK_BIN}" \
  --receptor "${RECEPTOR_PDBQT}" \
  --ligand "${LIGAND_PDBQT}" \
  --scoring "${SCORING}" \
  --center_x "${CENTER_X}" \
  --center_y "${CENTER_Y}" \
  --center_z "${CENTER_Z}" \
  --size_x "${SIZE_X}" \
  --size_y "${SIZE_Y}" \
  --size_z "${SIZE_Z}" \
  --exhaustiveness "${EXHAUSTIVENESS}" \
  --num_modes "${NUM_MODES}" \
  --seed "${SEED}" \
  --dir "${OUT_DIR}" \
  --verbosity 2


# ===========================================
# Step 3: 输出检查
# ===========================================
echo "[Step 3] Listing output files..."
ls -lh "${OUT_DIR}"

echo "Done. Check docking poses and scores in: ${OUT_DIR}"
