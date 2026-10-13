#!/usr/bin/env python
"""Build three ablation arms of preference pairs: B (docking) / C (realism) / D (combined).

All three arms share the SAME matching skeleton so the comparison is fair:

    exact heavy-atom element composition
    identical heavy-atom count
    |delta TPSA| <= 30
    greedy 1:1 matching, no endpoint reuse  (trainer runs allow_endpoint_reuse=false)

They differ ONLY in how the winner is chosen:

    arm B  winner = better docking        (require docking gap >= MIN_DOCK_GAP)
    arm C  winner = more realistic        (require |docking gap| <= DOCK_TOL_C  <- locks docking)
    arm D  winner = higher combined score (w * docking_pct + (1-w) * realism_pct)

Reward encoding
---------------
Both components are first mapped to POOL PERCENTILES so they share one scale:
    dock_pct = 1 - percentile(docking_raw)      larger-is-better
    real_pct = 1 - percentile(soft_novel)       larger-is-better
The reward written into the CSV is the arm's own score, so tau stays comparable
across arms.  NOTE: with a [0,1] reward and delta ~0.2, the config must use
tau ~ 5, not the tau=1 that was tuned for integer violation counts.

Outputs
-------
pairs_bcd/<arm>/{train,validation,test}_pairs.csv   trainer-compatible schema
pairs_bcd/<arm>/pair_audit.json
pairs_bcd/sample_pairs.txt                          human-readable spot check
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path("/home/md666/data/disk6/DiffB110/analysis/reference_support_experiment")
MANIFEST = HERE / "docking_subset/subset_manifest.csv"
DOCKING = HERE / "docking_subset/docking/docking_results_all.csv"
ENDPOINTS = HERE / "endpoint_index.csv"
OUT_DIR = HERE / "pairs_bcd"

FIELDS = [
    "pair_index", "split", "arm",
    "winner_id", "loser_id",
    "winner_shard", "loser_shard", "winner_index", "loser_index",
    "winner_n_nodes", "loser_n_nodes", "n_nodes",
    "reward_winner", "reward_loser", "reward_gap",
    "winner_docking_raw", "loser_docking_raw", "abs_docking_gap",
    "winner_soft_novel", "loser_soft_novel", "abs_soft_novel_gap",
    "winner_dock_pct", "loser_dock_pct",
    "winner_real_pct", "loser_real_pct",
    "winner_combo", "loser_combo", "abs_combo_gap",
    "winner_heavy_atom_count", "loser_heavy_atom_count",
    "winner_tpsa", "loser_tpsa", "abs_tpsa_gap",
    "winner_murcko_scaffold", "loser_murcko_scaffold",
    "winner_smiles", "loser_smiles",
    "source_rule",
]


def pct_rank(values: np.ndarray) -> np.ndarray:
    """Average-rank percentile in [0,1]; ties get the same value."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    ranks[order] = np.arange(values.size, dtype=float)
    # average ranks for ties
    out = np.empty(values.size, dtype=float)
    i = 0
    while i < values.size:
        j = i
        while j + 1 < values.size and values[order[j + 1]] == values[order[i]]:
            j += 1
        mean_rank = (i + j) / 2.0
        for k in range(i, j + 1):
            out[order[k]] = mean_rank
        i = j + 1
    return out / max(1.0, values.size - 1)


def load_pool() -> list[dict]:
    per_part: dict[str, list[dict]] = defaultdict(list)
    with MANIFEST.open("r", newline="") as fh:
        for row in csv.DictReader(fh):
            per_part[f"part_{int(row['part']):04d}"].append(row)
    for part in per_part:
        per_part[part].sort(key=lambda r: int(r["mol_id"]))

    score_by_molid: dict[str, float] = {}
    with DOCKING.open("r", newline="") as fh:
        for row in csv.DictReader(fh):
            rows = per_part.get(row["part"])
            if rows is None:
                continue
            idx = int(row["benchmark_index"])
            if idx >= len(rows) or row.get("success", "True") != "True":
                continue
            try:
                score_by_molid[rows[idx]["mol_id"]] = float(row["score"])
            except (TypeError, ValueError):
                continue
    print(f"[dock] scored molecules: {len(score_by_molid):,}")

    endpoints: dict[tuple[str, int], dict] = {}
    with ENDPOINTS.open("r", newline="") as fh:
        for row in csv.DictReader(fh):
            endpoints[(row["sdf_path"], int(row["sdf_index"]))] = row
    print(f"[endpoints] indexed: {len(endpoints):,}")

    pool: list[dict] = []
    miss_dock = miss_ep = 0
    for part_rows in per_part.values():
        for row in part_rows:
            dock = score_by_molid.get(row["mol_id"])
            if dock is None:
                miss_dock += 1
                continue
            ep = endpoints.get((row["source_shard"], int(row["source_index"])))
            if ep is None:
                miss_ep += 1
                continue
            pool.append({
                "sample_id": ep["sample_id"],
                "pt_shard": ep["pt_shard"],
                "mol_index": int(ep["mol_index"]),
                "n_nodes": int(ep["n_nodes"]),
                "docking": dock,
                "soft_novel": float(row["soft_novel"]),
                "comp": row["element_composition"],
                "hac": int(row["heavy_atom_count"]),
                "tpsa": float(row["tpsa"]),
                "scaffold": row["murcko_scaffold"],
                "smiles": row["smiles"],
            })
    print(f"[join] usable molecules: {len(pool):,}   (missing dock={miss_dock} endpoint={miss_ep})")
    return pool


def build_arm(pool: list[dict], arm: str, *, delta: float, min_dock_gap: float,
              dock_tol_c: float, tpsa_tol: float, w_dock: float,
              max_dock_gap: float) -> list[dict]:
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for i, m in enumerate(pool):
        groups[(m["comp"], m["hac"])].append(i)

    key = {"B": "dock_pct", "C": "real_pct", "D": "combo"}[arm]
    pairs: list[dict] = []
    for ids in groups.values():
        if len(ids) < 2:
            continue
        # larger key = better, so ids[a] is the best available -> winner
        ids.sort(key=lambda i: pool[i][key], reverse=True)
        n = len(ids)
        used: set[int] = set()
        for a in range(n):
            wi = ids[a]
            if wi in used:                    # `used` holds POOL indices, not positions
                continue
            wr = pool[wi]
            for b in range(n - 1, a, -1):
                lo_id = ids[b]
                if lo_id in used:
                    continue
                lo = pool[lo_id]
                gap = wr[key] - lo[key]          # positive: winner scores higher
                if gap < delta:
                    break
                if abs(lo["tpsa"] - wr["tpsa"]) > tpsa_tol:
                    continue
                if arm == "B":
                    # winner must dock genuinely better (docking_raw is more negative when better)
                    if lo["docking"] - wr["docking"] < min_dock_gap:
                        continue
                if arm == "C":
                    # docking locked: the only systematic difference is realism
                    if abs(wr["docking"] - lo["docking"]) > dock_tol_c:
                        continue
                if arm in ("B", "D"):
                    # keep pairs as "close contests": never pair a binder against the
                    # pathological docking tail (docking_raw has a max of +331 in this pool)
                    if lo["docking"] - wr["docking"] > max_dock_gap:
                        continue
                used.add(wi)
                used.add(lo_id)
                pairs.append({"w": wr, "l": lo, "gap": gap, "reward_key": key})
                break
    return pairs


def rows_for(pairs, arm: str, split: str, start: int) -> list[dict]:
    out = []
    for k, p in enumerate(pairs):
        w, l = p["w"], p["l"]
        rk = p["reward_key"]
        out.append({
            "pair_index": start + k, "split": split, "arm": arm,
            "winner_id": w["sample_id"], "loser_id": l["sample_id"],
            "winner_shard": w["pt_shard"], "loser_shard": l["pt_shard"],
            "winner_index": w["mol_index"], "loser_index": l["mol_index"],
            "winner_n_nodes": w["n_nodes"], "loser_n_nodes": l["n_nodes"],
            "n_nodes": max(w["n_nodes"], l["n_nodes"]),
            "reward_winner": round(w[rk], 6), "reward_loser": round(l[rk], 6),
            "reward_gap": round(w[rk] - l[rk], 6),
            "winner_docking_raw": round(w["docking"], 4), "loser_docking_raw": round(l["docking"], 4),
            "abs_docking_gap": round(abs(w["docking"] - l["docking"]), 4),
            "winner_soft_novel": round(w["soft_novel"], 6), "loser_soft_novel": round(l["soft_novel"], 6),
            "abs_soft_novel_gap": round(abs(w["soft_novel"] - l["soft_novel"]), 6),
            "winner_dock_pct": round(w["dock_pct"], 6), "loser_dock_pct": round(l["dock_pct"], 6),
            "winner_real_pct": round(w["real_pct"], 6), "loser_real_pct": round(l["real_pct"], 6),
            "winner_combo": round(w["combo"], 6), "loser_combo": round(l["combo"], 6),
            "abs_combo_gap": round(abs(w["combo"] - l["combo"]), 6),
            "winner_heavy_atom_count": w["hac"], "loser_heavy_atom_count": l["hac"],
            "winner_tpsa": round(w["tpsa"], 3), "loser_tpsa": round(l["tpsa"], 3),
            "abs_tpsa_gap": round(abs(w["tpsa"] - l["tpsa"]), 3),
            "winner_murcko_scaffold": w["scaffold"], "loser_murcko_scaffold": l["scaffold"],
            "winner_smiles": w["smiles"], "loser_smiles": l["smiles"],
            "source_rule": f"bcd_{arm}_composition_matched_v1",
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta", type=float, default=0.20)
    ap.add_argument("--min-dock-gap", type=float, default=0.5)
    ap.add_argument("--dock-tol-c", type=float, default=1.5)
    ap.add_argument("--max-dock-gap", type=float, default=3.0,
                    help="arms B/D: never pair across a docking gap larger than this (kcal/mol)")
    ap.add_argument("--max-dock-raw", type=float, default=0.0,
                    help="drop pool molecules whose docking_raw exceeds this (non-physical dockings)")
    ap.add_argument("--tpsa-tol", type=float, default=30.0)
    ap.add_argument("--w-dock", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--samples", type=int, default=10)
    args = ap.parse_args()

    pool = load_pool()
    n_before = len(pool)
    pool = [m for m in pool if m["docking"] <= args.max_dock_raw]
    print(f"[filter] dropped {n_before - len(pool):,} molecules with docking_raw > {args.max_dock_raw:+.1f}"
          f"  -> pool {len(pool):,}")
    dock = np.array([m["docking"] for m in pool])
    soft = np.array([m["soft_novel"] for m in pool])
    dp = 1.0 - pct_rank(dock)      # docking percentile, larger = better
    rp = 1.0 - pct_rank(soft)      # realism percentile, larger = better
    for i, m in enumerate(pool):
        m["dock_pct"] = float(dp[i])
        m["real_pct"] = float(rp[i])
        m["combo"] = args.w_dock * float(dp[i]) + (1 - args.w_dock) * float(rp[i])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audit: dict = {"config": vars(args), "arms": {}}
    rng = random.Random(args.seed)
    sample_lines: list[str] = []

    for arm in ("B", "C", "D"):
        pairs = build_arm(pool, arm, delta=args.delta, min_dock_gap=args.min_dock_gap,
                          dock_tol_c=args.dock_tol_c, tpsa_tol=args.tpsa_tol,
                          w_dock=args.w_dock, max_dock_gap=args.max_dock_gap)
        rng.shuffle(pairs)
        n = len(pairs)
        n_val = max(1, int(n * 0.08))
        n_test = max(1, int(n * 0.08))
        splits = {"test": pairs[:n_test], "validation": pairs[n_test:n_test + n_val],
                  "train": pairs[n_test + n_val:]}

        arm_dir = OUT_DIR / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        stats: dict = {"pairs": {}, "cross_axis": {}, "gaps": {}}
        idx = 0
        for split in ("train", "validation", "test"):
            rows = rows_for(splits[split], arm, split, idx)
            idx += len(rows)
            with (arm_dir / f"{split}_pairs.csv").open("w", newline="") as fh:
                wr = csv.DictWriter(fh, fieldnames=FIELDS)
                wr.writeheader()
                wr.writerows(rows)
            stats["pairs"][split] = len(rows)

        all_rows = rows_for(pairs, arm, "all", 0)
        with (arm_dir / "preference_pairs.csv").open("w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=FIELDS)
            wr.writeheader()
            wr.writerows(all_rows)

        def mean(rs, f):
            v = [r[f] for r in all_rows]
            return float(np.mean(v)) if v else float("nan")

        stats["gaps"] = {
            "reward_gap_mean": round(mean(all_rows, "reward_gap"), 4),
            "reward_gap_min": round(min(r["reward_gap"] for r in all_rows), 4) if all_rows else None,
            "abs_tpsa_gap_mean": round(mean(all_rows, "abs_tpsa_gap"), 3),
            "abs_docking_gap_mean": round(mean(all_rows, "abs_docking_gap"), 3),
            # positive = winner docks better (docking_raw is more negative when better)
            "winner_docking_advantage": round(
                float(np.mean([r["loser_docking_raw"] - r["winner_docking_raw"] for r in all_rows])), 4),
            # positive = winner is more realistic (soft_novel is lower when better)
            "winner_realism_advantage": round(
                float(np.mean([r["loser_soft_novel"] - r["winner_soft_novel"] for r in all_rows])), 4),
        }
        stats["cross_axis"] = {
            "winner_docking_mean": round(mean(all_rows, "winner_docking_raw"), 4),
            "loser_docking_mean": round(mean(all_rows, "loser_docking_raw"), 4),
            "winner_soft_novel_mean": round(mean(all_rows, "winner_soft_novel"), 4),
            "loser_soft_novel_mean": round(mean(all_rows, "loser_soft_novel"), 4),
        }
        audit["arms"][arm] = stats

        sample_lines.append(f"\n{'='*104}\nARM {arm}  ({n:,} pairs)  "
                            f"winner by {'docking' if arm=='B' else 'realism' if arm=='C' else 'combo'}\n{'='*104}")
        for r in pairs[: args.samples]:
            sample_lines.append(
                f"  W dock={r['w']['docking']:+7.3f} soft={r['w']['soft_novel']:.3f} hac={r['w']['hac']:>3d} "
                f"{r['w']['smiles'][:58]}\n"
                f"  L dock={r['l']['docking']:+7.3f} soft={r['l']['soft_novel']:.3f} hac={r['l']['hac']:>3d} "
                f"{r['l']['smiles'][:58]}\n"
                f"    -> dDock={r['w']['docking']-r['l']['docking']:+.3f}  "
                f"dSoft={r['w']['soft_novel']-r['l']['soft_novel']:+.3f}  gap={r['gap']:.3f}")

    (OUT_DIR / "pair_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (OUT_DIR / "sample_pairs.txt").write_text("\n".join(sample_lines), encoding="utf-8")

    print("\n" + "=" * 104)
    print("ARM SUMMARY")
    print("=" * 104)
    print(f"{'arm':<5s}{'pairs':>9s}{'对接优势':>11s}{'真实优势':>11s}{'gap均值':>10s}"
          f"{'|ΔTPSA|':>10s}   winner/loser 的原始均值")
    for arm in ("B", "C", "D"):
        s = audit["arms"][arm]
        g = s["gaps"]; c = s["cross_axis"]
        print(f"{arm:<5s}{sum(s['pairs'].values()):>9,}{g['winner_docking_advantage']:>11.3f}"
              f"{g['winner_realism_advantage']:>11.3f}{g['reward_gap_mean']:>10.3f}"
              f"{g['abs_tpsa_gap_mean']:>10.2f}   "
              f"dock W{c['winner_docking_mean']:+.2f} / L{c['loser_docking_mean']:+.2f}   "
              f"soft W{c['winner_soft_novel_mean']:.3f} / L{c['loser_soft_novel_mean']:.3f}")
    for arm in ("B", "C", "D"):
        print(f"   arm {arm} split: train={audit['arms'][arm]['pairs']['train']:,}"
              f"  val={audit['arms'][arm]['pairs']['validation']:,}"
              f"  test={audit['arms'][arm]['pairs']['test']:,}")
    print(f"\n[wrote] {OUT_DIR}/<arm>/*.csv   and  sample_pairs.txt")


if __name__ == "__main__":
    main()
