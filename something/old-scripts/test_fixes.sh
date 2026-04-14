#!/bin/bash
# Quick test script to verify bug fixes

echo "========================================="
echo "Testing Bug Fixes"
echo "========================================="
echo ""

# Test 1: Diagnostic script
echo "[1/3] Running diagnostic script..."
conda run -n dif python scripts/diagnose_issues.py 2>&1 | grep -E "(✓|✗|PASS|FAIL)"
echo ""

# Test 2: Import test
echo "[2/3] Testing imports..."
conda run -n dif python -c "
from ddpo.configs.dataset_info import get_crossdock_info
from ddpo.utils import DiffSBDDFilter
from ddpo.rewards import DruglikenessReward, CompositeReward
print('✓ All imports successful')
" 2>&1
echo ""

# Test 3: Check modified files
echo "[3/3] Checking modified files..."
if grep -q "return None" RLPF-verl_diffusion/verl_diffusion/trainer/ddpo_trainer.py; then
    echo "✓ DDPOTrainer fix applied"
else
    echo "✗ DDPOTrainer fix NOT found"
fi

if grep -q "if samples is None:" RLPF-verl_diffusion/verl_diffusion/trainer/ddpo_trainer.py; then
    echo "✓ Batch skip logic added"
else
    echo "✗ Batch skip logic NOT found"
fi

if grep -q "get_crossdock_info" ddpo/configs/dataset_info.py; then
    echo "✓ get_crossdock_info() function added"
else
    echo "✗ get_crossdock_info() NOT found"
fi

echo ""
echo "========================================="
echo "Test Complete"
echo "========================================="
