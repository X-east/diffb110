"""Aggregate DiffSBDD online DDPO run metrics from runs/diffsbdd/*/training_summary.csv.

For each run dir, extract:
  - run name, num epochs
  - first vs last epoch: rdkit_valid_ratio, reward_mean
  - docking/qed/sa if present (single-reward vs main differ in column names)
Just prints a compact table for building the audit doc.
"""
import csv
import glob
import os
import sys

ROOT = "/home/md666/data/disk6/DiffB110/runs/diffsbdd"

KEYS_FLOAT = [
    "rdkit_valid_ratio",
    "reward_mean",
    "avg_docking",
    "avg_qed",
    "avg_sa",
    "avg_kl",
    "avg_ppo_loss",
]


def first_last(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    if not rows:
        return None
    return rows[0], rows[-1], len(rows)


def fget(row, key):
    v = row.get(key, "")
    if v in (None, ""):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def main():
    dirs = sorted(
        d for d in glob.glob(os.path.join(ROOT, "*")) if os.path.isdir(d)
    )
    print(f"# run_dir | epochs | valid_first | valid_last | reward_first | reward_last | dock_last | qed_last | sa_last | kl_last")
    for d in dirs:
        ts = os.path.join(d, "training_summary.csv")
        name = os.path.basename(d)
        if not os.path.exists(ts):
            print(f"{name} | NO_CSV")
            continue
        res = first_last(ts)
        if res is None:
            print(f"{name} | EMPTY")
            continue
        first, last, n = res
        vf = fget(first, "rdkit_valid_ratio")
        vl = fget(last, "rdkit_valid_ratio")
        rf = fget(first, "reward_mean")
        rl = fget(last, "reward_mean")
        dl = fget(last, "avg_docking")
        ql = fget(last, "avg_qed")
        sl = fget(last, "avg_sa")
        kl = fget(last, "avg_kl")
        def fmt(x):
            return f"{x:.4f}" if isinstance(x, float) else ("NA" if x is None else str(x))
        print(
            f"{name} | {n} | {fmt(vf)} | {fmt(vl)} | {fmt(rf)} | {fmt(rl)} | {fmt(dl)} | {fmt(ql)} | {fmt(sl)} | {fmt(kl)}"
        )


if __name__ == "__main__":
    main()
