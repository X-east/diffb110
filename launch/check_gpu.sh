#!/bin/bash

# GPU Diagnostic Script
# Check GPU availability and usage

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║              GPU Diagnostic Report                             ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check CUDA availability
echo "1. CUDA Status:"
python3 << 'EOF'
import torch
print(f"   CUDA Available: {torch.cuda.is_available()}")
print(f"   CUDA Version: {torch.version.cuda}")
print(f"   cuDNN Version: {torch.backends.cudnn.version()}")
EOF

echo ""
echo "2. GPU Devices:"
python3 << 'EOF'
import torch
num_gpus = torch.cuda.device_count()
print(f"   Total GPUs: {num_gpus}")
for i in range(num_gpus):
    props = torch.cuda.get_device_properties(i)
    print(f"   GPU {i}: {props.name}")
    print(f"      Memory: {props.total_memory / 1e9:.2f} GB")
    print(f"      Compute Capability: {props.major}.{props.minor}")
EOF

echo ""
echo "3. Current GPU Memory Usage:"
python3 << 'EOF'
import torch
num_gpus = torch.cuda.device_count()
for i in range(num_gpus):
    allocated = torch.cuda.memory_allocated(i) / 1e9
    reserved = torch.cuda.memory_reserved(i) / 1e9
    total = torch.cuda.get_device_properties(i).total_memory / 1e9
    print(f"   GPU {i}:")
    print(f"      Allocated: {allocated:.2f} GB")
    print(f"      Reserved: {reserved:.2f} GB")
    print(f"      Total: {total:.2f} GB")
    print(f"      Free: {total - reserved:.2f} GB")
EOF

echo ""
echo "4. PyTorch Configuration:"
python3 << 'EOF'
import torch
print(f"   PyTorch Version: {torch.__version__}")
print(f"   Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
print(f"   Deterministic: {torch.are_deterministic_algorithms_enabled()}")
EOF

echo ""
echo "5. Distributed Training Support:"
python3 << 'EOF'
import torch
import torch.distributed as dist
print(f"   Distributed Available: {dist.is_available()}")
print(f"   Backend Available: {dist.is_backend_available('nccl')}")
EOF

echo ""
echo "6. PyTorch Lightning:"
python3 << 'EOF'
try:
    import pytorch_lightning as pl
    print(f"   Lightning Version: {pl.__version__}")
    print(f"   Lightning Available: True")
except ImportError:
    print(f"   Lightning Available: False")
EOF

echo ""
echo "7. Ray Status:"
python3 << 'EOF'
try:
    import ray
    print(f"   Ray Version: {ray.__version__}")
    print(f"   Ray Available: True")
    if ray.is_initialized():
        print(f"   Ray Initialized: True")
        resources = ray.available_resources()
        print(f"   Available Resources: {resources}")
    else:
        print(f"   Ray Initialized: False")
except ImportError:
    print(f"   Ray Available: False")
EOF

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║              Recommendations                                   ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "For multi-GPU training:"
echo "  • Use: ./run_training_ddp.sh (PyTorch Lightning with DDP)"
echo "  • This automatically distributes work across all available GPUs"
echo ""
echo "To monitor GPU usage during training:"
echo "  • In another terminal: watch -n 1 nvidia-smi"
echo ""
