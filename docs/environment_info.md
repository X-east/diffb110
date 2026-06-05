# Environment Info

**最后更新**: 2026-05-08

## 1. Host and OS
- Hostname: `md`
- User: `md666`
- OS: `Ubuntu 24.04.3 LTS (Noble Numbat)`
- Kernel: `Linux 6.17.0-14-generic`

## 2. CPU
- CPU model: `Intel(R) Xeon(R) Silver 4110 CPU @ 2.10GHz`
- Sockets: `2`
- Cores per socket: `8`
- Threads per core: `2`
- Logical CPUs: `32`
- NUMA nodes: `2`
- NUMA node0 CPUs: `0-7,16-23`
- NUMA node1 CPUs: `8-15,24-31`

## 3. Memory
- Total RAM: `440 GiB`
- Swap: `8.0 GiB`

## 4. GPU
- GPU count: `8`
- GPU model: `NVIDIA GeForce RTX 3080`
- Memory per GPU: `10240 MiB` (~9.65 GiB visible to PyTorch)
- Driver version: `590.48.01`
- `nvidia-smi` reported CUDA version: `13.1`

### GPU topology
- GPU0-3 are attached to NUMA node 0
- GPU4-7 are attached to NUMA node 1
- Near-pair topology:
  - `(GPU0, GPU1)`, `(GPU2, GPU3)`, `(GPU4, GPU5)`, `(GPU6, GPU7)`

## 5. CUDA / Deep Learning Stack
- NVIDIA kernel module present
- `/dev/nvidia0` ... `/dev/nvidia7` available
- `nvcc` path: `/usr/bin/nvcc`
- `nvcc` version: `CUDA 12.0 (V12.0.140)`
- `/usr/local/cuda` symlink: not present
- `CUDA_HOME`: not set
- `CUDA_VISIBLE_DEVICES`: not set (all GPUs visible by default)

### PyTorch
- Python env: `dif`
- Python version: `3.10.19`
- PyTorch version: `2.0.1+cu118`
- PyTorch CUDA runtime: `11.8`
- cuDNN enabled: `True`
- cuDNN version: `8700`
- `torch.cuda.is_available()`: `True`
- `torch.cuda.device_count()`: `8`

### CUDA layers note
1. Driver capability (`nvidia-smi`): `CUDA 13.1`
2. System compiler toolkit (`nvcc`): `CUDA 12.0`
3. PyTorch binary runtime: `cu118`

Not automatically a problem — driver is new enough, toolkit is 12.0, PyTorch was built against CUDA 11.8.

## 6. Conda
- Conda root: `/home/md666/miniforge3`
- Conda version: `25.11.0`
- Solver: `libmamba`

### Available conda environments
- `base` → `/home/md666/miniforge3`
- `dif` → `/home/md666/miniforge3/envs/dif`

### Active project environment
- `CONDA_DEFAULT_ENV=dif`
- `CONDA_PREFIX=/home/md666/miniforge3/envs/dif`

## 7. Python / Packages in `dif`
- Python executable: `/home/md666/miniforge3/envs/dif/bin/python`
- `pip` wrapper status: broken (`cannot execute: required file not found`) — use `python -m pip ...`

### Installed package snapshot
- `rdkit==2025.09.5`
- `numpy==1.26.4`
- `scipy==1.15.2`
- `pandas==2.3.3`
- `scikit-learn==1.7.2`
- `torch==2.0.1+cu118`
- `pytorch_lightning==1.9.4`
- `openbabel==3.1.0`
- `prolif==2.1.0`
- `MDAnalysis==2.9.0`
- `yaml==6.0.3`
- `biopython` — status uncertain (may have been installed since last snapshot)
- `unidock` — conda-installed UniDock docking package

### Not installed in `dif`
- `torch_geometric`
- `mdtraj`

## 8. Filesystem / Storage
- `/` mounted from `/dev/sdg2`
- Root size: `879G`

## 9. Current Project Location
- `PWD` from environment: `/home/md666/data/disk6/DiffB110`（Symbolic link）

## 10. Relevant Environment Variables

- `CONDA_DEFAULT_ENV=dif`
- `CONDA_PREFIX=/home/md666/miniforge3/envs/dif`
- `BABEL_LIBDIR=/home/md666/miniforge3/envs/dif/lib/openbabel/3.1.0`
- `SCHRODINGER=/opt/schrodinger2021-2`

### Included PATH entries
- `/home/md666/miniforge3/envs/dif/bin`
- `/home/md666/miniforge3/condabin`
- `/opt/schrodinger2021-2`

## 11. Known Issues / Notes

- `CUDA_VISIBLE_DEVICES` is not set; all GPUs visible by default
- `pip` launcher is broken; use `python -m pip ...` as workaround
- Some packages emit compatibility warnings (PyTorch >= 2.2 expected)
- Current primary docking backend is **UniDock** (conda-installed), not AutoDock-GPU
- AutoDock-GPU local binary directory has been removed from the project root (see `docs/autodock_gpu_binary_debug.md` for historical debug record)
