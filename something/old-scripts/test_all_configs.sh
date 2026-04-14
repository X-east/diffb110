#!/bin/bash
# Test all YAML configurations in ddpo/configs
# Environment: dif (conda)
# Created: 2026-03-06

set -e  # Exit on error

# Activate conda environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate dif

# Set OpenBabel plugin path
export BABEL_LIBDIR=~/miniforge3/envs/dif/lib/openbabel/3.1.0

# Create log directory
mkdir -p log/config-tests

# Test configurations
CONFIGS=(
    "ddp_2gpu_debug.yaml"
    "ddp_4gpu_debug.yaml"
    "ddp_8gpu_training.yaml"
    "ddp_8gpu.yaml"
    "ddp_8gpu_production.yaml"
    "multi_gpu_cpu_test.yaml"
    "gpu_rl_test.yaml"
)

# Test each configuration
for config in "${CONFIGS[@]}"; do
    echo "=========================================="
    echo "Testing: $config"
    echo "=========================================="
    
    config_path="ddpo/configs/$config"
    
    # Check if config exists
    if [ ! -f "$config_path" ]; then
        echo "❌ Config not found: $config_path"
        continue
    fi
    
    # Extract config name (without .yaml)
    config_name="${config%.yaml}"
    log_file="log/config-tests/${config_name}_$(date +%Y%m%d_%H%M%S).log"
    
    echo "Log file: $log_file"
    echo "Starting test at $(date)"
    
    # Run test (2 epochs only for quick validation)
    if python ddpo/main_diffsbdd_lightning.py --config "$config_path" 2>&1 | tee "$log_file"; then
        echo "✅ Test passed: $config"
    else
        echo "❌ Test failed: $config"
        echo "Check log: $log_file"
    fi
    
    echo ""
done

echo "=========================================="
echo "All tests completed!"
echo "=========================================="
