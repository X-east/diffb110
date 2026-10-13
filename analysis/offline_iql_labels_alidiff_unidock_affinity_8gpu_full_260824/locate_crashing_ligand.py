#!/usr/bin/env python
# 定位让 UniDock 1.1.3 model.cpp(1101) 崩溃的首个配体（二分法）。
# 输入：worker_00/batch_0000 已有的 ligand_*.sdf 与 ligands.txt（16965 个）。
# 方法：对 ligands.txt 的前 N 行做子集提交，二分找到最小 N 使 UniDock 崩溃。
import os, sys, subprocess, tempfile, json

BASE = os.path.dirname(os.path.abspath(__file__))
BATCH = os.path.join(BASE, "batches/worker_00/batch_0000")
INPUTS = os.path.join(BATCH, "inputs")
LIGANDS_TXT = os.path.join(INPUTS, "ligands.txt")
RECEPTOR = "/data/disk6/DiffB110/data/8gjl.pdbqt"
UNIDOCK = "/opt/unidock-1.1.3/bin/unidock"

with open(LIGANDS_TXT) as f:
    ALL = [l.strip() for l in f if l.strip()]
N = len(ALL)
print(f"[locate] total ligands={N}", flush=True)


def run_subset(count):
    """提交前 count 个配体，返回 (crashed_bool, rc, stderr_tail)。"""
    sub = ALL[:count]
    tmp = tempfile.mkdtemp(prefix="loc_")
    idx = os.path.join(tmp, "subset.txt")
    with open(idx, "w") as f:
        f.write("\n".join(sub) + "\n")
    outdir = os.path.join(tmp, "poses")
    os.makedirs(outdir, exist_ok=True)
    cmd = [
        UNIDOCK, "--receptor", RECEPTOR, "--ligand_index", idx,
        "--center_x", "207.293", "--center_y", "212.43", "--center_z", "172.771",
        "--size_x", "20.0", "--size_y", "20.0", "--size_z", "20.0",
        "--scoring", "vina", "--num_modes", "1", "--seed", "42",
        "--verbosity", "1", "--cpu", "0", "--max_gpu_memory", "0",
        "--dir", outdir, "--search_mode", "fast",
    ]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    except subprocess.TimeoutExpired:
        return (True, -1, "TIMEOUT")
    rc = p.returncode
    err = (p.stderr or "")[-1500:]
    crashed = rc != 0 or "An internal error occurred" in (p.stdout or "") or "model.cpp" in err
    return (crashed, rc, err)


# 1) 确认全量崩溃
crashed, rc, err = run_subset(N)
print(f"[locate] full-set crashed={crashed} rc={rc}", flush=True)
if not crashed:
    print("[locate] full set did NOT crash -> cannot reproduce here", flush=True)
    sys.exit(0)

# 2) 指数下界：找到最小 2^k 仍崩溃
lo = 1
hi = N
while lo < hi:
    mid = (lo + hi) // 2
    c, _, e = run_subset(mid)
    if c:
        hi = mid  # 前半段就崩 -> 坏配体在前半
    else:
        lo = mid + 1  # 前半段不崩 -> 坏配体在后半
    print(f"[locate] mid={mid} crashed={c} lo={lo} hi={hi}", flush=True)

first_crash = lo
print(f"[locate] FIRST_CRASHING_INDEX={first_crash}", flush=True)

# 3) 确认 first_crash-1 不崩，first_crash 崩（隔离单点）
c0, _, _ = run_subset(first_crash - 1)
c1, _, e1 = run_subset(first_crash)
print(f"[locate] idx {first_crash-1} crashed={c0}; idx {first_crash} crashed={c1}", flush=True)
print(f"[locate] crash stderr tail for idx {first_crash}:\n{e1}", flush=True)
print(f"[locate] CRASHING_LIGAND_FILE={ALL[first_crash-1]}", flush=True)
