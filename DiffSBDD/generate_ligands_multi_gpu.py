"""
Multi-GPU target-conditioned ligand generation (DiffSBDD).

This script is based on `DiffSBDD/generate_ligands.py`, but parallelizes sampling
across multiple GPUs by spawning one worker process per GPU.

Design notes:
- Each GPU worker loads the checkpoint once and generates its own shard.
- Workers write shard SDFs to disk; the parent process merges shards into one SDF.
- All user-tunable configuration is grouped at the top of the file.
"""

from __future__ import annotations

# =========================
# Configuration (edit here)
# =========================

# Model / conditioning
CHECKPOINT = "/home/md666/data/disk6/DiffB110/DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt"
PDBFILE = "/home/md666/data/disk6/DiffB110/data/8GJL-hbond-maestro.pdb"

# Exactly one of the following pocket definitions must be used.
# - If you know pocket residues: set RESI_LIST and keep REF_LIGAND=None
# - If you have a reference ligand (in PDB as "A:401" or as an SDF path): set REF_LIGAND and keep RESI_LIST=None
RESI_LIST: list[str] | None = None  # e.g. ["A:45", "A:46", "A:47"]
REF_LIGAND: str | None = "C:1101"      # e.g. "A:401" or "/path/to/ref_ligand.sdf"

# Output
OUTFILE = "/home/md666/data/disk6/DiffB110/bishe/RE-out/260429lsj-RE-generated_500.sdf"
TMP_DIR = "/home/md666/data/disk6/DiffB110/bishe/RE-out/260429lsj-RE_generate_500"

# Sampling scale
N_SAMPLES_TOTAL = 500

# GPUs to use (0-indexed torch device ids)
GPUS = [1]  # change to e.g. [0,1,2,3,4,5] for 6 GPUs

# Per-GPU batching (controls VRAM usage). Must divide each GPU's shard size.
BATCH_SIZE_PER_GPU = 25

# Optional control knobs (kept consistent with `generate_ligands.py`)
NUM_NODES_LIG: int | None = None  # fixed ligand node count, or None to sample from size distribution
ALL_FRAGS = True                 # if True, keep all fragments; else keep largest fragment
SANITIZE = False                   # RDKit sanitize in molecule postprocess
RELAX = False                     # force-field relaxation (slow)
RESAMPLINGS = 10
JUMP_LENGTH = 1
TIMESTEPS: int | None = None      # None uses training value

# Reproducibility
BASE_SEED = 20260407

# =========================
# End configuration section
# =========================

import argparse
import os
import queue
import threading
import time
from dataclasses import dataclass
from multiprocessing import Queue
from pathlib import Path
from typing import Optional

import torch
from rdkit import Chem
from tqdm import tqdm  # 新增：用于显示进度条

from openbabel import openbabel

openbabel.obErrorLog.StopLogging()  # suppress OpenBabel messages

import utils
from lightning_modules import LigandPocketDDPM


@dataclass(frozen=True)
class JobConfig:
    checkpoint: Path
    pdbfile: str
    resi_list: list[str] | None
    ref_ligand: str | None
    outfile: Path
    tmp_dir: Path
    n_samples_total: int
    gpus: list[int]
    batch_size_per_gpu: int
    num_nodes_lig: int | None
    all_frags: bool
    sanitize: bool
    relax: bool
    resamplings: int
    jump_length: int
    timesteps: int | None
    base_seed: int


def _split_counts(total: int, n_parts: int) -> list[int]:
    base = total // n_parts
    rem = total % n_parts
    return [base + (1 if i < rem else 0) for i in range(n_parts)]


def _ensure_divisible_or_raise(per_gpu_counts: list[int], batch_size: int) -> None:
    bad = [c for c in per_gpu_counts if (c % batch_size) != 0]
    if bad:
        raise ValueError(
            f"Each GPU shard size must be divisible by batch_size_per_gpu={batch_size}. "
            f"Got shard sizes: {per_gpu_counts}. "
            f"Tip: change BATCH_SIZE_PER_GPU or N_SAMPLES_TOTAL, or set GPUS count differently."
        )


def _merge_sdf(shard_paths: list[Path], out_path: Path) -> None:
    """
    合并SDF分片文件，无过滤。直接以二进制形式追加所有内容。
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, 'wb') as out_f:  # 以二进制写入模式打开最终文件
        for p in shard_paths:
            if p.exists():
                with open(p, 'rb') as in_f:  # 以二进制读取模式打开每个分片
                    # 将整个分片文件内容追加到最终文件
                    out_f.write(in_f.read())


def _worker_generate_shard(
    *,
    gpu_id: int,
    shard_idx: int,
    n_samples: int,
    cfg: JobConfig,
    shard_outfile: Path,
    progress_queue: Queue,  # 新增：接收进度队列，用于上报进度
) -> None:
    # Make CUDA placement explicit per process.
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = f"cuda:{gpu_id}"
    else:
        device = "cpu"

    # Deterministic-ish per worker (diffusion sampling may still have nondeterminism).
    seed = int(cfg.base_seed) + int(shard_idx) * 10_000 + int(gpu_id)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Load model on this device.
    model = LigandPocketDDPM.load_from_checkpoint(cfg.checkpoint, map_location=device)
    model = model.to(device)

    if cfg.num_nodes_lig is not None:
        num_nodes_lig = torch.ones(n_samples, dtype=int) * int(cfg.num_nodes_lig)
    else:
        num_nodes_lig = None

    if n_samples == 0:
        utils.write_sdf_file(shard_outfile, [])
        progress_queue.put(('worker_done', gpu_id, 0))  # 新增：报告任务完成
        return

    if cfg.batch_size_per_gpu is None:
        batch_size = n_samples
    else:
        batch_size = int(cfg.batch_size_per_gpu)

    if n_samples % batch_size != 0:
        raise ValueError(
            f"Worker shard n_samples={n_samples} must be divisible by batch_size={batch_size}. "
            f"Adjust BATCH_SIZE_PER_GPU or choose GPUS such that shard sizes divide cleanly."
        )

    molecules: list[Chem.Mol | None] = []
    n_batches = n_samples // batch_size
    for batch_idx in range(n_batches):
        batch_mols = model.generate_ligands(
            cfg.pdbfile,
            batch_size,
            cfg.resi_list,
            cfg.ref_ligand,
            num_nodes_lig,
            cfg.sanitize,
            largest_frag=(not cfg.all_frags),
            relax_iter=(200 if cfg.relax else 0),
            resamplings=cfg.resamplings,
            jump_length=cfg.jump_length,
            timesteps=cfg.timesteps,
        )
        molecules.extend(batch_mols)

        # 新增：计算当前批次生成的有效分子数，并上报进度
        valid_in_batch = len([m for m in batch_mols if m is not None])
        progress_queue.put(('progress', gpu_id, valid_in_batch))

    # 新增：所有批次生成完毕，写入文件前，报告最终完成状态
    total_valid = len([m for m in molecules if m is not None])
    progress_queue.put(('worker_done', gpu_id, total_valid))

    utils.write_sdf_file(shard_outfile, molecules)


def _progress_monitor(
    per_gpu_counts: list[int],
    gpu_ids: list[int],
    progress_queue: Queue,
    total_samples: int
) -> None:
    """
    进度监控函数。运行在独立线程中，监听进度队列，更新进度条。
    """
    # 初始化进度条
    # 为每个GPU创建一个独立的进度条
    gpu_bars = {}
    for gpu_id, gpu_total in zip(gpu_ids, per_gpu_counts):
        gpu_bars[gpu_id] = tqdm(
            total=gpu_total,
            desc=f"GPU {gpu_id}",
            position=len(gpu_bars),  # 设置垂直位置，避免重叠
            leave=True,
            bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}',
            unit='mol',
        )

    # 创建一个总进度条
    total_bar = tqdm(
        total=total_samples,
        desc="总计",
        position=len(gpu_bars),
        leave=True,
        bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}',
        unit='mol',
    )

    # 用于跟踪每个GPU已生成的分子数和完成状态
    gpu_progress = {gpu_id: 0 for gpu_id in gpu_ids}
    gpu_finished = {gpu_id: False for gpu_id in gpu_ids}

    try:
        while not all(gpu_finished.values()):
            try:
                # 非阻塞地从队列获取进度更新，等待0.5秒
                msg_type, gpu_id, increment = progress_queue.get(timeout=0.5)
                if msg_type == 'progress':
                    # 更新单个GPU进度条
                    gpu_bars[gpu_id].update(increment)
                    gpu_progress[gpu_id] += increment
                    # 更新总进度条
                    total_bar.update(increment)
                elif msg_type == 'worker_done':
                    # 标记该GPU任务完成
                    gpu_finished[gpu_id] = True
                    gpu_bars[gpu_id].close()  # 关闭该GPU的进度条
            except queue.Empty:
                # 队列为空，继续循环等待
                continue
            except Exception as e:
                # 避免因进度显示异常导致主任务失败
                print(f"\n进度监控线程发生异常（可忽略）: {e}")
                break

    finally:
        # 确保所有进度条被正确关闭
        total_bar.close()
        for bar in gpu_bars.values():
            bar.close()

        # 打印最终摘要
        print("\n" + "="*50)
        print("生成任务完成摘要:")
        for gpu_id in gpu_ids:
            print(f"  GPU {gpu_id}: 生成了 {gpu_progress[gpu_id]}/{per_gpu_counts[gpu_ids.index(gpu_id)]} 个分子")
        print(f"  总计: 生成了 {sum(gpu_progress.values())}/{total_samples} 个分子")
        print("="*50)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Multi-GPU ligand generation based on DiffSBDD/generate_ligands.py. "
                    "All parameters have editable defaults at the top of this file."
    )
    p.add_argument("--checkpoint", type=Path, default=Path(CHECKPOINT))
    p.add_argument("--pdbfile", type=str, default=PDBFILE)
    p.add_argument("--resi_list", type=str, nargs="+", default=RESI_LIST)
    p.add_argument("--ref_ligand", type=str, default=REF_LIGAND)
    p.add_argument("--outfile", type=Path, default=Path(OUTFILE))
    p.add_argument("--tmp_dir", type=Path, default=Path(TMP_DIR))
    p.add_argument("--n_samples", type=int, default=N_SAMPLES_TOTAL)
    p.add_argument("--gpus", type=int, nargs="+", default=GPUS)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE_PER_GPU)
    p.add_argument("--num_nodes_lig", type=int, default=NUM_NODES_LIG)
    p.add_argument("--all_frags", action="store_true", default=ALL_FRAGS)
    p.add_argument("--sanitize", action="store_true", default=SANITIZE)
    p.add_argument("--relax", action="store_true", default=RELAX)
    p.add_argument("--resamplings", type=int, default=RESAMPLINGS)
    p.add_argument("--jump_length", type=int, default=JUMP_LENGTH)
    p.add_argument("--timesteps", type=int, default=TIMESTEPS)
    p.add_argument("--seed", type=int, default=BASE_SEED)
    return p


def main() -> None:
    args = build_argparser().parse_args()

    if (args.resi_list is None) == (args.ref_ligand is None):
        raise ValueError("Exactly one of --resi_list or --ref_ligand must be set.")

    if not args.gpus:
        raise ValueError("--gpus must be a non-empty list, e.g. --gpus 0 1 2 3")

    cfg = JobConfig(
        checkpoint=args.checkpoint,
        pdbfile=args.pdbfile,
        resi_list=args.resi_list,
        ref_ligand=args.ref_ligand,
        outfile=args.outfile,
        tmp_dir=args.tmp_dir,
        n_samples_total=int(args.n_samples),
        gpus=[int(x) for x in args.gpus],
        batch_size_per_gpu=int(args.batch_size),
        num_nodes_lig=args.num_nodes_lig,
        all_frags=bool(args.all_frags),
        sanitize=bool(args.sanitize),
        relax=bool(args.relax),
        resamplings=int(args.resamplings),
        jump_length=int(args.jump_length),
        timesteps=args.timesteps,
        base_seed=int(args.seed),
    )

    cfg.tmp_dir.mkdir(parents=True, exist_ok=True)

    per_gpu_counts = _split_counts(cfg.n_samples_total, len(cfg.gpus))
    _ensure_divisible_or_raise(per_gpu_counts, cfg.batch_size_per_gpu)

    # 新增：创建用于进程间通信的进度队列
    progress_queue = Queue()

    # 新增：启动进度监控线程
    monitor_thread = threading.Thread(
        target=_progress_monitor,
        args=(per_gpu_counts, cfg.gpus, progress_queue, cfg.n_samples_total),
        daemon=True  # 设置为守护线程，主线程退出时自动结束
    )
    monitor_thread.start()

    # 短暂等待，确保进度条初始化完成
    time.sleep(0.5)

    # Spawn one process per GPU.
    ctx = torch.multiprocessing.get_context("fork")
    procs: list[torch.multiprocessing.Process] = []
    shard_paths: list[Path] = []

    for shard_idx, (gpu_id, n_samples) in enumerate(zip(cfg.gpus, per_gpu_counts)):
        shard_out = cfg.tmp_dir / f"shard_gpu{gpu_id:02d}_idx{shard_idx:02d}_n{n_samples}.sdf"
        shard_paths.append(shard_out)
        proc = ctx.Process(
            target=_worker_generate_shard,
            kwargs=dict(
                gpu_id=int(gpu_id),
                shard_idx=int(shard_idx),
                n_samples=int(n_samples),
                cfg=cfg,
                shard_outfile=shard_out,
                progress_queue=progress_queue,  # 新增：传递进度队列给工作进程
            ),
        )
        proc.start()
        procs.append(proc)

    # Wait for all workers, fail fast if any worker fails.
    for proc in procs:
        proc.join()

    # 新增：等待进度监控线程结束（最多等待3秒）
    monitor_thread.join(timeout=3.0)

    failed = [p.exitcode for p in procs if p.exitcode != 0]
    if failed:
        raise RuntimeError(f"One or more GPU workers failed. Exit codes: {[p.exitcode for p in procs]}")

    # Merge shard SDFs into the final output.
    _merge_sdf(shard_paths, cfg.outfile)


if __name__ == "__main__":
    # Avoid oversubscribing CPU threads when using multiple GPU processes.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    main()