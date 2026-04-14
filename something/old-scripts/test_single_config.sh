#!/bin/bash
# Test a single configuration file
# Usage: bash scripts/test_single_config.sh <config_name>
# Example: bash scripts/test_single_config.sh ddp_8gpu_training.yaml
# Environment: dif (conda)
# Created: 2026-03-06

if [ $# -eq 0 ]; then
    echo "Usage: bash scripts/test_single_config.sh <config_name>"
    echo ""
    echo "Available configs:"
    echo "  - ddp_2gpu_debug.yaml"
    echo "  - ddp_4gpu_debug.yaml"
    echo "  - ddp_8gpu_training.yaml"
    echo "  - ddp_8gpu.yaml"
    echo "  - ddp_8gpu_production.yaml"
    echo "  - multi_gpu_cpu_test.yaml"
    echo "  - gpu_rl_test.yaml"
    exit 1
fi

CONFIG_NAME=$1
CONFIG_PATH="ddpo/configs/$CONFIG_NAME"

# Check if config exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "❌ Config not found: $CONFIG_PATH"
    exit 1
fi

# Activate conda environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate dif

# Set OpenBabel plugin path
export BABEL_LIBDIR=~/miniforge3/envs/dif/lib/openbabel/3.1.0

# Create log directory
mkdir -p log/config-tests

# Extract config name (without .yaml)
CONFIG_BASE="${CONFIG_NAME%.yaml}"
LOG_FILE="log/config-tests/${CONFIG_BASE}_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================="
echo "Testing: $CONFIG_NAME"
echo "Config path: $CONFIG_PATH"
echo "Log file: $LOG_FILE"
echo "Started at: $(date)"
echo "=========================================="
echo ""

# Run test
python ddpo/main_diffsbdd_lightning.py --config "$CONFIG_PATH" 2>&1 | tee "$LOG_FILE"

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Test passed: $CONFIG_NAME"
else
    echo "❌ Test failed: $CONFIG_NAME (exit code: $EXIT_CODE)"
fi
echo "Finished at: $(date)"
echo "Log saved to: $LOG_FILE"
echo "=========================================="

exit $EXIT_CODE
