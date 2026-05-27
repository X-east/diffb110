"""Simulate logp_distribution reward under realistic small-batch conditions.

Samples 500 molecules from 5829 baseline molecules, computes per-bin
proportion gaps vs ChEMBL reference, and reports reward distributions.
"""
import sys
sys.path.insert(0, ".")

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

# ── Constants ──
BINS = [-np.inf, 0.5, 1.5, 2.5, 3.0, 3.5, 4.5, 5.5, np.inf]
N_BINS = len(BINS) - 1
BIN_LABELS = [
    "(-∞,0.5)", "[0.5,1.5)", "[1.5,2.5)", "[2.5,3.0)",
    "[3.0,3.5)", "[3.5,4.5)", "[4.5,5.5)", "(5.5,+∞)",
]
REF_PROPORTIONS = np.array([0.044, 0.093, 0.206, 0.140, 0.143, 0.227, 0.108, 0.039])
SCALE = 12.0
MIN_R = -2.0
MAX_R = 2.0
BATCH_SIZE = 500
N_TRIALS = 500
LAPLACE_SMOOTH = True

# ── Load molecules ──
sdf = "OUT/260407_generate_ligands_10000/260407generated_10000_filtered.sdf"
suppl = Chem.SDMolSupplier(sdf)
logps = []
for mol in suppl:
    if mol is not None:
        try:
            logps.append(float(Descriptors.MolLogP(mol)))
        except Exception:
            pass
logps = np.array(logps)
print(f"Loaded {len(logps)} molecules, LogP mean={np.mean(logps):.2f} std={np.std(logps):.2f}")

# ── Bin the whole dataset ──
bin_indices = np.digitize(logps, BINS) - 1
bin_indices = np.clip(bin_indices, 0, N_BINS - 1)

# ── Simulate batch sampling ──
all_rewards = []
all_gaps_per_bin = {i: [] for i in range(N_BINS)}
all_props_per_bin = {i: [] for i in range(N_BINS)}

for trial in range(N_TRIALS):
    idx = np.random.choice(len(logps), size=BATCH_SIZE, replace=False)
    batch_bins = bin_indices[idx]
    bin_counts = np.array([np.sum(batch_bins == i) for i in range(N_BINS)], dtype=np.float64)

    if LAPLACE_SMOOTH:
        props = (bin_counts + 1.0) / (BATCH_SIZE + N_BINS)
    else:
        props = bin_counts / BATCH_SIZE
        props = np.maximum(props, 1e-8)
        props /= props.sum()

    gaps = REF_PROPORTIONS - props
    for i in range(N_BINS):
        all_gaps_per_bin[i].append(gaps[i])
        all_props_per_bin[i].append(props[i])

    # Per-molecule rewards
    for b in batch_bins:
        r = SCALE * gaps[b]
        r = np.clip(r, MIN_R, MAX_R)
        all_rewards.append(r)

all_rewards = np.array(all_rewards)

# ── Report ──
print(f"\n{'='*90}")
print(f"  Simulation: {N_TRIALS} trials × {BATCH_SIZE} mols = {N_TRIALS*BATCH_SIZE} total")
print(f"{'='*90}")

print(f"\n  Per-molecule reward distribution:")
print(f"    mean  = {np.mean(all_rewards):+.3f}")
print(f"    std   = {np.std(all_rewards):.3f}")
print(f"    min   = {np.min(all_rewards):+.3f}")
print(f"    max   = {np.max(all_rewards):+.3f}")
for pct in [5, 25, 50, 75, 95]:
    print(f"    P{pct:02d}  = {np.percentile(all_rewards, pct):+.3f}")

print(f"\n  Per-bin gap (REF - sample):")
print(f"  {'Bin':<14s} {'Ref%':>6s} {'Sample%':>9s} {'GapMean':>9s} {'GapStd':>8s} {'WorstGap':>10s}")
print(f"  {'-'*60}")
for i in range(N_BINS):
    gaps_i = np.array(all_gaps_per_bin[i])
    props_i = np.array(all_props_per_bin[i])
    worst = gaps_i[np.argmax(np.abs(gaps_i))]
    print(f"  {BIN_LABELS[i]:<14s} {REF_PROPORTIONS[i]*100:5.1f}% "
          f"{np.mean(props_i)*100:8.2f}% {np.mean(gaps_i):+9.4f} "
          f"{np.std(gaps_i):8.4f} {worst:+10.4f}")

# ── What fraction of molecules get penalty > 0.5 / > 1.0? ──
print(f"\n  Penalty severity (per-molecule):")
for thresh in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
    neg = np.sum(all_rewards < -thresh)
    pos = np.sum(all_rewards > thresh)
    print(f"    |reward| > {thresh:.1f}: {neg:>6d} neg + {pos:>6d} pos = {neg+pos:>6d} "
          f"({(neg+pos)/len(all_rewards)*100:5.1f}%)")

# ── What if we cheat and use all 5829 molecules at once (batch size effect) ──
print(f"\n  Batch size effect (all 5829 at once vs 500):")
for bs in [500, 1000, 2000, 5829]:
    idx = np.random.choice(len(logps), size=min(bs, len(logps)), replace=False)
    batch_bins = bin_indices[idx]
    bin_counts = np.array([np.sum(batch_bins == i) for i in range(N_BINS)], dtype=np.float64)
    if LAPLACE_SMOOTH:
        props = (bin_counts + 1.0) / (bs + N_BINS)
    else:
        props = bin_counts / bs
        props /= props.sum()
    gaps = REF_PROPORTIONS - props
    r_max = np.max(np.abs(SCALE * gaps))
    print(f"    bs={bs:4d}: max_gap={np.max(np.abs(gaps)):.4f}, "
          f"max_unclipped_reward={r_max:.2f}")
