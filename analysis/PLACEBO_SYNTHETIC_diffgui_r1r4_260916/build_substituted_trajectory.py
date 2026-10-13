#!/usr/bin/env python
"""SYNTHETIC COUNTERFACTUAL builder -- NOT A TRAINING RESULT.

Constructs a counterfactual per-epoch molecule set for the DiffGui RL run such
that the AGGREGATE R1-R4 violation rate is flat at a chosen target, while the
R4b / R4c sub-rule molecule COUNTS are preserved exactly.

How the two goals coexist
-------------------------
Every substitution is required to be R4b/R4c-neutral on both sides:

    adding a violation : clean real  ->  pool molecule that IS R1-R4 violating
                                         but IS R4b-clean and R4c-clean
    removing a violation: violating real (R4b-clean, R4c-clean)
                                         ->  clean pool molecule

Because both the incoming and outgoing molecule are R4b/R4c-clean in the first
case, and the outgoing one is R4b/R4c-clean in the second, the R4b and R4c
counts never change.  Only the R1-R4 union count moves.

Substitutes are matched on heavy-atom count (exact match preferred) so the
size distribution is preserved.

Outputs use the same column layout as the real per-epoch molecules.csv.
See README.md in this directory: this artifact must never be presented as a
real model output.

Usage
-----
    python build_substituted_trajectory.py --pilot 5     # first 5 epochs only
    python build_substituted_trajectory.py               # all epochs
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = Path("/home/md666/data/disk6/DiffB110")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from scripts.offline_rl.structure_pathology_rules import (  # noqa: E402
    compute_structure_pathology_from_smiles as PATHOLOGY,
)

REAL_RUN = ROOT / "runs/diffgui-deppa-full-lr1e5-wd1e4-b8x8-t200-100ep_20260626_165809"
TARGET_R14 = 0.684          # = real epoch-1 aggregate R1-R4 violation rate
SEED = 20260916
HAC_TOLERANCE = 2           # fall back to nearest HAC within this many atoms

POOL_PATTERNS = [
    # 1000-step DiffGui products
    "analysis/benchmark/8gjl_c1101_benchmark_20260630_all_1000steps/all_1000step_collection/sanitize_valid/diffgui.sanitize_valid.sdf",
    "analysis/benchmark/8gjl_c1101_benchmark_20260701_diffgui_rep1_1000steps_seed1_7/filtered_sanitize_valid/diffgui.sanitize_valid.sdf",
    "analysis/benchmark/8gjl_c1101_benchmark_20260701_all_rep2_1000steps_seed8_14/filtered_sanitize_valid/diffgui.sanitize_valid.sdf",
    "analysis/benchmark/8gjl_c1101_benchmark_20260701_all_rep3_1000steps_seed15_21/filtered_sanitize_valid/diffgui.sanitize_valid.sdf",
    "analysis/benchmark/benchmark5/five_model_repeats_1000steps_260827/rep1[012]_seed*/filtered_sanitize_valid/diffgui.sanitize_valid.sdf",
    # 200/500-step DiffGui products -- used only as a donor pool for
    # R4b/R4c-clean substitutes under an exact heavy-atom match.  The pool's own
    # R4b/R4c rate is irrelevant because substitutes are picked clean.
    "analysis/benchmark/diffgui_step_sweep_500_200_1000attempts_20260707_run2/[25]00steps_rep*/filtered_sanitize_valid/diffgui.sanitize_valid.sdf",
]

LOGGER = logging.getLogger("placebo-substituted")


def facts_of(smiles: str):
    """Return (hac, r14_flag, r4b_flag, r4c_flag) or None."""
    f = PATHOLOGY(smiles)
    if f is None:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return (mol.GetNumHeavyAtoms(),
            int(f.structural_violation_count) > 0,
            bool(f.r4b_three_ring_over_attached_violation),
            bool(f.r4c_three_ring_spiro_or_fused_violation))


def build_pool() -> tuple[list[dict], list[dict]]:
    """Load every pool SDF once, dedupe by resolved path, and profile molecules."""
    seen: set[str] = set()
    pool: list[dict] = []
    sources: list[dict] = []
    for pat in POOL_PATTERNS:
        for p in sorted(glob.glob(str(ROOT / pat))):
            rp = str(Path(p).resolve())
            if rp in seen:
                continue
            seen.add(rp)
            n = 0
            for m in Chem.SDMolSupplier(rp, sanitize=True, removeHs=False):
                if m is None:
                    continue
                smi = Chem.MolToSmiles(m)
                fa = facts_of(smi)
                if fa is None:
                    continue
                pool.append({"smiles": smi, "hac": fa[0], "r14": fa[1],
                             "r4b": fa[2], "r4c": fa[3],
                             "src": rp, "src_idx": n, "is_own": 0})
                n += 1
            sources.append({"source_path": rp, "n_loaded": n})
            LOGGER.info("pool source %s -> %d molecules", rp, n)
    return pool, sources


def build_own_pool() -> tuple[list[dict], dict]:
    """Donor molecules taken from the run's OWN per-epoch outputs.

    Added because the external 1000/200/500-step pools have almost no
    low-heavy-atom pathological molecules (pathology correlates with size), so
    the "add a violation" direction was starved at small HAC.  Entries carry
    ``src_epoch`` so a molecule is never used to substitute inside its own epoch.
    """
    pool: list[dict] = []
    for p in sorted(glob.glob(str(REAL_RUN / "epochs/epoch_*/molecules.csv"))):
        ep = int(re.search(r"epoch_(\d+)", p).group(1))
        with open(p, newline="") as fh:
            for i, r in enumerate(csv.DictReader(fh)):
                smi = (r.get("canonical_smiles") or r.get("smiles") or "").strip()
                fa = facts_of(smi) if smi else None
                if fa is None:
                    continue
                pool.append({"smiles": smi, "hac": fa[0], "r14": fa[1],
                             "r4b": fa[2], "r4c": fa[3],
                             "src": f"{p}#own", "src_idx": i,
                             "src_epoch": ep, "is_own": 1})
    return pool, {"source_path": "OWN RUN epochs (all)", "n_loaded": len(pool)}


def index_pool(pool: list[dict]) -> dict:
    """Index pool molecules by (hac, property-key)."""
    idx: dict[tuple[int, str], list[int]] = defaultdict(list)
    for i, m in enumerate(pool):
        key = f"r14={int(m['r14'])};r4b={int(m['r4b'])};r4c={int(m['r4c'])}"
        idx[(m["hac"], key)].append(i)
    return idx


def pick(idx: dict, pool: list[dict], hac: int, want_key: str,
         usage: Counter, rng: random.Random,
         exclude_epoch: int | None = None) -> tuple[int, bool] | None:
    """Pick a pool molecule with the requested HAC and property key.

    The objective is lexicographic, in this priority order:

        minimise ( |hac - target_hac| ,  reuse count ,  is_own )

    1. **heavy-atom fidelity** -- never trade the size distribution away;
    2. **reuse** -- spread molecules over the whole trajectory;
    3. **is_own** -- at equal size and equal reuse, prefer a donor from the
       external DiffGui pools so the substitution keeps its "drawn from the
       broad pool" meaning; the run's own molecules are only pulled in where
       the external pools have no candidate (small heavy-atom counts).

    ``usage`` is GLOBAL across all epochs.  ``exclude_epoch`` skips own-run
    donors belonging to the epoch being built, so a molecule never substitutes
    inside its own epoch.

    Returns:
        ``(pool_index, reused)`` or ``None`` when nothing is available.
    """
    best: tuple[tuple[int, int, int], int] | None = None
    for (h, key), cands in idx.items():
        if key != want_key:
            continue
        gap = abs(h - hac)
        for c in cands:
            if exclude_epoch is not None and pool[c].get("src_epoch") == exclude_epoch:
                continue
            score = (gap, usage.get(c, 0), pool[c].get("is_own", 0))
            if best is None or score < best[0]:
                best = (score, c)
            elif score == best[0] and rng.random() < 0.5:
                best = (score, c)
    if best is None:
        return None
    choice = best[1]
    usage[choice] = usage.get(choice, 0) + 1
    return choice, usage[choice] > 1


def process_epoch(epoch: int, pool: list[dict], idx: dict, manifest: list[dict],
                  rng: random.Random, target_rate: float,
                  usage: Counter) -> tuple[list[dict], dict]:
    """Build the substituted molecule list for one epoch, aiming at ``target_rate``.

    ``usage`` is threaded in from the caller and shared across epochs so pool
    molecules spread over the whole trajectory while the HAC match is kept tight.
    """
    src = REAL_RUN / f"epochs/epoch_{epoch:04d}/molecules.csv"
    with open(src, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"empty molecules.csv: {src}")

    # attach pathology facts to the real molecules (keep only parseable ones)
    keep: list[dict] = []
    for r in rows:
        smi = (r.get("canonical_smiles") or r.get("smiles") or "").strip()
        fa = facts_of(smi) if smi else None
        if fa is None:
            continue
        keep.append({**r, "_smi": smi, "_hac": fa[0], "_r14": fa[1],
                     "_r4b": fa[2], "_r4c": fa[3]})

    n = len(keep)
    k_cur = sum(1 for r in keep if r["_r14"])
    k_tgt = int(round(target_rate * n))
    diff = k_tgt - k_cur

    subs: list[tuple[int, int, str]] = []       # (real_idx, pool_idx, direction)
    failed = 0
    reused = 0

    if diff > 0:
        # need MORE violations: replace clean real -> violating-but-R4b/R4c-clean pool
        targets = [i for i, r in enumerate(keep) if not r["_r14"]]
        rng.shuffle(targets)
        for i in targets[:diff]:
            got = pick(idx, pool, keep[i]["_hac"], "r14=1;r4b=0;r4c=0", usage, rng,
                       exclude_epoch=epoch)
            if got is None:
                failed += 1
                continue
            j, was_reused = got
            reused += int(was_reused)
            subs.append((i, j, "add_violation"))
    elif diff < 0:
        # need FEWER violations: replace violating(R4b/R4c clean) real -> clean pool
        targets = [i for i, r in enumerate(keep)
                   if r["_r14"] and not r["_r4b"] and not r["_r4c"]]
        rng.shuffle(targets)
        for i in targets[: -diff]:
            got = pick(idx, pool, keep[i]["_hac"], "r14=0;r4b=0;r4c=0", usage, rng,
                       exclude_epoch=epoch)
            if got is None:
                failed += 1
                continue
            j, was_reused = got
            reused += int(was_reused)
            subs.append((i, j, "remove_violation"))

    sub_map = {i: j for i, j, _ in subs}
    out: list[dict] = []
    for i, r in enumerate(keep):
        base = {k: v for k, v in r.items() if not k.startswith("_")}
        if i in sub_map:
            p = pool[sub_map[i]]
            rec = {k: "" for k in base}
            rec.update({
                "epoch": epoch, "rank": "", "seed": "",
                "local_mol_index": "", "global_mol_id": "",
                "original_name": f"sub_e{epoch:04d}_{i:03d}",
                "rdkit_valid": 1, "sanitize_status": "ok",
                "num_atoms": p["hac"], "smiles": p["smiles"],
                "canonical_smiles": p["smiles"], "murcko_scaffold": "",
            })
            out.append(rec)
        else:
            out.append(base)

    k_new = k_cur + sum(1 for _, _, d in subs if d == "add_violation") \
        - sum(1 for _, _, d in subs if d == "remove_violation")
    stat = {
        "epoch": epoch, "n": n,
        "k_before": k_cur, "k_target": k_tgt, "k_after": k_new,
        "rate_before_pct": k_cur / n * 100, "rate_after_pct": k_new / n * 100,
        "n_substituted": len(subs), "substitute_pct": len(subs) / n * 100,
        "n_failed_match": failed, "n_reused": reused,
        "r4b_before": sum(1 for r in keep if r["_r4b"]),
        "r4c_before": sum(1 for r in keep if r["_r4c"]),
    }
    for i, j, d in subs:
        manifest.append({
            "epoch": epoch, "real_index": i, "direction": d,
            "real_smiles": keep[i]["_smi"], "real_hac": keep[i]["_hac"],
            "pool_index": j, "pool_smiles": pool[j]["smiles"], "pool_hac": pool[j]["hac"],
            "pool_source": pool[j]["src"], "pool_source_index": pool[j]["src_idx"],
            "hac_delta": pool[j]["hac"] - keep[i]["_hac"],
        })
    return out, stat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=int, default=0,
                    help="only process the first N epochs (0 = all)")
    ap.add_argument("--level", type=float, default=TARGET_R14 * 100,
                    help="target mean of the aggregate R1-R4 rate, in percent")
    ap.add_argument("--no-detrend", action="store_true",
                    help="do not remove the linear trend (keep the real slope)")
    args = ap.parse_args()
    args.detrend = not args.no_detrend

    (HERE / "logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.FileHandler(HERE / "logs/build.log"),
                  logging.StreamHandler(sys.stdout)],
    )
    LOGGER.warning("SYNTHETIC COUNTERFACTUAL BUILD -- output is NOT a model result")

    for d in ("pools", "epochs"):
        (HERE / d).mkdir(exist_ok=True)

    pool, sources = build_pool()
    own_pool, own_src = build_own_pool()
    LOGGER.info("own-run donors added: %d", len(own_pool))
    pool = pool + own_pool
    sources.append(own_src)
    n_viol = sum(1 for m in pool if m["r14"])
    LOGGER.info("pool total=%d  R1-R4 violating=%d (%.1f%%)  R4b=%.1f%%  R4c=%.1f%%",
                len(pool), n_viol, n_viol / len(pool) * 100,
                sum(1 for m in pool if m["r4b"]) / len(pool) * 100,
                sum(1 for m in pool if m["r4c"]) / len(pool) * 100)

    with open(HERE / "pools/pool_index.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["smiles", "hac", "r14", "r4b", "r4c",
                                           "src", "src_idx", "src_epoch"],
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(pool)
    idx = index_pool(pool)

    epochs = sorted(int(re.search(r"epoch_(\d+)", p).group(1))
                    for p in glob.glob(str(REAL_RUN / "epochs/epoch_*/molecules.csv")))
    if args.pilot:
        epochs = epochs[: args.pilot]

    # ---- pass 1: real per-epoch R1-R4 rate, then build per-epoch targets ----
    real_rate: dict[int, float] = {}
    for ep in epochs:
        with open(REAL_RUN / f"epochs/epoch_{ep:04d}/molecules.csv", newline="") as fh:
            rows = list(csv.DictReader(fh))
        num = den = 0
        for r in rows:
            smi = (r.get("canonical_smiles") or r.get("smiles") or "").strip()
            fa = facts_of(smi) if smi else None
            if fa is None:
                continue
            den += 1
            num += int(fa[1])
        real_rate[ep] = num / den if den else 0.0

    import numpy as _np
    E = _np.asarray(epochs, dtype=float)
    Rr = _np.asarray([real_rate[e] for e in epochs]) * 100.0
    if args.detrend and len(epochs) > 2:
        slope = float(_np.polyfit(E, Rr, 1)[0])
        det = Rr - slope * (E - E.mean())          # trend removed, noise kept
    else:
        slope, det = 0.0, Rr
    shift = args.level - det.mean()                # move the level to --level
    targets = {e: (det[i] + shift) / 100.0 for i, e in enumerate(epochs)}
    LOGGER.info("real mean=%.2f%% SD=%.2fpp slope=%+.4f pp/ep | target mean=%.2f%% SD=%.2fpp",
                Rr.mean(), Rr.std(), slope, _np.mean(list(targets.values())) * 100,
                _np.std([targets[e] * 100 for e in epochs]))

    manifest: list[dict] = []
    stats: list[dict] = []
    usage_global: Counter = Counter()      # shared across ALL epochs
    for ep in epochs:
        rng = random.Random(SEED + ep)
        out, st = process_epoch(ep, pool, idx, manifest, rng, targets[ep], usage_global)
        stats.append(st)
        dst = HERE / f"epochs/epoch_{ep:04d}_molecules.csv"
        with open(dst, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
            w.writeheader()
            w.writerows(out)
        print(f"[ep {ep:3d}] n={st['n']:3d}  R1-R4 {st['rate_before_pct']:5.1f}% -> "
              f"{st['rate_after_pct']:5.1f}%  (target {TARGET_R14*100:.1f}%)  "
              f"替换 {st['n_substituted']} 个 ({st['substitute_pct']:.1f}%)  "
              f"匹配失败 {st['n_failed_match']}", flush=True)

    cols = ["epoch", "n", "k_before", "k_target", "k_after", "rate_before_pct",
            "rate_after_pct", "n_substituted", "substitute_pct", "n_failed_match",
            "n_reused", "r4b_before", "r4c_before"]
    with open(HERE / "r1r5_trajectory.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(stats)

    (HERE / "manifest.json").write_text(json.dumps({
        "WARNING": "SYNTHETIC COUNTERFACTUAL — NOT A TRAINING RESULT. "
                   "Constructed by controlled substitution for methodological self-test only. "
                   "Must never be presented as a model output.",
        "real_run": str(REAL_RUN),
        "target_r1r4_rate": TARGET_R14,
        "seed": SEED,
        "hac_tolerance": HAC_TOLERANCE,
        "pool_sources": sources,
        "pool_size": len(pool),
        "epochs_processed": len(epochs),
        "n_substitutions_total": len(manifest),
        "substitutions": manifest,
    }, indent=2), encoding="utf-8")

    avg = sum(s["substitute_pct"] for s in stats) / len(stats)
    n_uniq = len({s["pool_index"] for s in manifest})
    print(f"\n{'='*66}\n平均替换比例 = {avg:.2f}%  (每个 epoch 约 "
          f"{sum(s['n_substituted'] for s in stats)/len(stats):.2f} 个分子)")
    print(f"替换条目 {len(manifest)} 条 | 唯一池分子 {n_uniq} 个 | "
          f"复用 {sum(s['n_reused'] for s in stats)} 次 | "
          f"匹配失败 {sum(s['n_failed_match'] for s in stats)} 次")
    print(f"[wrote] {HERE}/r1r5_trajectory.csv")
    print(f"[wrote] {HERE}/manifest.json")


if __name__ == "__main__":
    main()
