#!/bin/bash
# Test 8 GPU training with CPU monitoring
# Usage: bash scripts/test_8gpu_with_cpu_monitoring.sh [config_name]
# Example: bash scripts/test_8gpu_with_cpu_monitoring.sh ddp_8gpu_training_balanced.yaml
# Environment: dif (conda)
# Created: 2026-03-06

CONFIG_NAME=${1:-"ddp_8gpu_training.yaml"}
CONFIG_PATH="ddpo/configs/$CONFIG_NAME"

# Check if config exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "❌ Config not found: $CONFIG_PATH"
    echo ""
    echo "Available configs:"
    ls -1 ddpo/configs/ddp_8gpu*.yaml
    exit 1
fi

# Activate conda environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate dif

# Set OpenBabel plugin path
export BABEL_LIBDIR=~/miniforge3/envs/dif/lib/openbabel/3.1.0

# Limit threads per process
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Create log directory
CONFIG_BASE="${CONFIG_NAME%.yaml}"
LOG_DIR="log/8gpu-cpu-tests"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/${CONFIG_BASE}_${TIMESTAMP}.log"
CPU_LOG="$LOG_DIR/${CONFIG_BASE}_${TIMESTAMP}_cpu.log"

echo "=========================================="
echo "8 GPU Training with CPU Monitoring"
echo "=========================================="
echo "Config: $CONFIG_NAME"
echo "Log file: $LOG_FILE"
echo "CPU log: $CPU_LOG"
echo "Started at: $(date)"
echo ""

# Extract Ray CPU count from config
RAY_CPUS=$(grep "num_cpus:" "$CONFIG_PATH" | head -1 | awk '{print $2}')
echo "Ray CPU workers: $RAY_CPUS"
echo ""

# Start CPU monitoring in background
(
    echo "Timestamp,CPU_Usage(%),Load_Avg_1m,Load_Avg_5m,Python_Processes,Ray_Processes"
    while true; do
        CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
        LOAD_AVG=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1","$2}' | tr -d ',')
        PYTHON_PROCS=$(ps aux | grep python | grep -v grep | wc -l)
        RAY_PROCS=$(ps aux | grep ray | grep -v grep | wc -l)
        echo "$(date +%H:%M:%S),$CPU_USAGE,$LOAD_AVG,$PYTHON_PROCS,$RAY_PROCS"
        sleep 10
    done
) > "$CPU_LOG" &

CPU_MONITOR_PID=$!

echo "CPU monitoring started (PID: $CPU_MONITOR_PID)"
echo ""

# Run training
python ddpo/main_diffsbdd_lightning.py --config "$CONFIG_PATH" 2>&1 | tee "$LOG_FILE"

EXIT_CODE=$?

# Stop CPU monitoring
kill $CPU_MONITOR_PID 2>/dev/null

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Test passed: $CONFIG_NAME"
else
    echo "❌ Test failed: $CONFIG_NAME (exit code: $EXIT_CODE)"
fi
echo "Finished at: $(date)"
echo ""
echo "Logs saved:"
echo "  Training: $LOG_FILE"
echo "  CPU usage: $CPU_LOG"
echo ""

# Show CPU usage summary
if [ -f "$CPU_LOG" ]; then
    echo "CPU Usage Summary:"
    echo "----------------------------------------"
    tail -n +2 "$CPU_LOG" | awk -F',' '
    BEGIN {
        sum_cpu=0; count=0; max_cpu=0;
        sum_load1=0; sum_load5=0;
        max_python=0; max_ray=0;
    }
    {
        cpu=$2; load1=$3; load5=$4; python=$5; ray=$6;
        sum_cpu+=cpu; count++;
        if(cpu>max_cpu) max_cpu=cpu;
        sum_load1+=load1; sum_load5+=load5;
        if(python>max_python) max_python=python;
        if(ray>max_ray) max_ray=ray;
    }
    END {
        if(count>0) {
            printf "  Avg CPU Usage: %.1f%%\n", sum_cpu/count;
            printf "  Max CPU Usage: %.1f%%\n", max_cpu;
            printf "  Avg Load (1m): %.2f\n", sum_load1/count;
            printf "  Avg Load (5m): %.2f\n", sum_load5/count;
            printf "  Max Python Processes: %d\n", max_python;
            printf "  Max Ray Processes: %d\n", max_ray;
        }
    }'
    echo "----------------------------------------"
fi

echo "=========================================="

exit $EXIT_CODE
