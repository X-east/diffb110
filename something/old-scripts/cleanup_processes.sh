#!/bin/bash
# Cleanup all training and Ray processes
# Created: 2026-03-06

echo "=========================================="
echo "Cleaning up all training processes..."
echo "=========================================="

# Count processes before cleanup
PYTHON_COUNT=$(ps aux | grep python | grep -v grep | wc -l)
echo "Python processes before cleanup: $PYTHON_COUNT"

# Kill main training processes
echo ""
echo "Stopping main training processes..."
pkill -f "main_diffsbdd_lightning.py"
pkill -f "main_diffsbdd.py"
sleep 2

# Kill Ray processes
echo "Stopping Ray processes..."
pkill -f "ray::calculate_single_docking_score"
pkill -f "ray/core/src/ray"
pkill -f "ray/dashboard"
pkill -f "ray/_private"
pkill -f "ray/autoscaler"
sleep 2

# Kill any remaining Python processes from dif environment
echo "Stopping remaining dif environment processes..."
pkill -f "miniforge3/envs/dif"
sleep 2

# Force kill if needed
echo "Force killing any stubborn processes..."
pkill -9 -f "main_diffsbdd"
pkill -9 -f "ray::"
sleep 1

# Clean up Ray temp files
echo ""
echo "Cleaning up Ray temporary files..."
rm -rf /tmp/ray/session_* 2>/dev/null
echo "Ray temp files cleaned"

# Count processes after cleanup
PYTHON_COUNT_AFTER=$(ps aux | grep python | grep -v grep | wc -l)
echo ""
echo "=========================================="
echo "Cleanup completed!"
echo "Python processes after cleanup: $PYTHON_COUNT_AFTER"
echo "Processes killed: $((PYTHON_COUNT - PYTHON_COUNT_AFTER))"
echo "=========================================="

# Show remaining processes if any
if [ $PYTHON_COUNT_AFTER -gt 5 ]; then
    echo ""
    echo "Warning: Some processes may still be running:"
    ps aux | grep -E "(python|ray)" | grep -v grep | grep -v "gpustat" | head -10
fi
