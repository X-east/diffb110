#!/usr/bin/env python
"""Phase 2: build reference-realness preference pairs for E2PO.

Winner/loser is decided by ``soft_novel`` (lower = closer to real medicinal
chemistry), under the SAME four confounder constraints used by the existing
structural pair set, so the only systematic difference inside a pair is realism:

    exact heavy-atom element composition
    identical heavy-atom count
    |delta TPSA| <= 30
    |delta docking| <= 1.5

Rewards follow the E2PO soft-label convention:
    reward = 1 - soft_novel      (higher = more realistic)

The output CSV schema matches ``analysis/diffsbdd_structure_pairs_matched_260831``
so ``ddpo/global_dpo`` needs no code change - only ``data.pair_dir``.

Joins:
    endpoint_index.csv                 sample_id <-> (.pt shard, mol_index) <-> (sdf, index)
    candidate_200k_profiles.csv        realism + matching descriptors
    docking_subset/subset_manifest.csv molecule order inside each part
    docking_subset/docking/...         UniDock score
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

HERE = Path("/home/md666/data/disk6/DiffB110/analysis/reference_support_experiment")
ENDPOINT_INDEX = HERE / "endpoint_index.csv"
PROFILES = HERE / "candidate_200k_profiles.csv"
MANIFEST = HERE / "docking_subset/subset_manifest.csv"
DOCKING = HERE / "docking_subset/docking/docking_results_all.csv"
OUT_DIR = HERE / "novel_pairs"

FIELDS = [
    "pair_index", "split",
    "winner_id", "loser_id",
    "winner_docking_raw", "loser_docking_raw",
    "reward_winner", "reward_loser", "reward_gap",
    "winner_shard", "loser_shard",
    "winner_index", "loser_index",
    "winner_n_nodes", "loser_n_nodes", "n_nodes",
    "source_rule",
    "winner_soft_novel", "loser_soft_novel", "soft_novel_gap",
    "winner_null_pct", "loser_null_pct",
    "winner_formula", "loser_formula",
    "winner_heavy_atom_count", "loser_heavy_atom_count",
    "winner_tpsa", "loser_tpsa", "abs_tpsa_gap",
    "abs_docking_gap",
    "winner_murcko_scaffold", "loser_murcko_scaffold",
    "winner_smiles", "loser_smiles",
]


def load_molecules() -> dict[tuple[str, int], dict]:
    """(sdf_path, sdf_index) -> realism + matching descriptors."""
    mols: dict[tuple[str, int], dict] = {}
    with PROFILES.open("r", newline="") as fh:
        for row in csv.DictReader(fh):
            mols[(row["shard"], int(row["shard_index"]))] = {
                "smiles": row["smiles"],
                "soft_novel": float(row["soft_novel"]),
                "null_pct": float(row["null_percentile"]),
                "formula": row["element_composition"],
                "hac": int(row["heavy_atom_count"]),
                "tpsa": float(row["tpsa"]),
                "scaffold": row["murcko_scaffold"],
            }
    return mols


def load_docking_by_location() -> dict[tuple[str, int], float]:
    """(source_shard, source_index) -> docking score."""
    # manifest rows inside a part are written in order == benchmark_index
    per_part: dict[str, list[dict]] = defaultdict(list)
    with MANIFEST.open("r", newline="") as fh:
        for row in csv.DictReader(fh):
            per_part[row["part"]].append(row)
    for part in per_part:
        per_part[part].sort(key=lambda r: int(r["mol_id"]))
    lookup: dict[tuple[str, str, int], dict] = {}
    for part, rows in per_part.items():
        for i, row in enumerate(rows):
            lookup[(part, row["source_shard"], int(row["source_index"]))] = row

    scores: dict[tuple[str, int], float] = {}
    with DOCKING.open("r", newline="") as fh:
        for row in csv.DictReader(fh):
            part = row["part"]
            idx = int(row["benchmark_index"])
            rows = per_part.get(part)
            if rows is None or idx >= len(rows):
                continue
            m = rows[idx]
            try:
                score = float(row["score"])
            except (TypeError, ValueError):
                continue
            scores[(m["source_shard"], int(m["source_index"]))] = score
    return scores


def load_endpoints() -> dict[tuple[str, int], dict]:
    """(sdf_path, sdf_index) -> endpoint identity."""
    eps: dict[tuple[str, int], dict] = {}
    with ENDPOINT_INDEX.open("r", newline="") as fh:
        for row in csv.DictReader(fh):
            eps[(row["sdf_path"], int(row["sdf_index"]))] = {
                "sample_id": row["sample_id"],
                "pt_shard": row["pt_shard"],
                "mol_index": int(row["mol_index"]),
                "n_nodes": int(row["n_nodes"]),
            }
    return eps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta", type=float, default=0.25)
    ap.add_argument("--tpsa-tol", type=float, default=30.0)
    ap.add_argument("--dock-tol", type=float, default=1.5)
    ap.add_argument("--max-losers-per-winner", type=int, default=1,
                    help="1 = strict matching (no endpoint reuse, as the trainer requires)")
    ap.add_argument("--seed", type=int, default=20260914)
    args = ap.parse_args()

    mols = load_molecules()
    scores = load_docking_by_location()
    eps = load_endpoints()
    print(f"[load] profiles={len(mols):,}  docked={len(scores):,}  endpoints={len(eps):,}")

    # ---- join ------------------------------------------------------------- #
    records: list[dict] = []
    missing = {"profile": 0, "endpoint": 0, "dock": 0}
    for key, meta in mols.items():
        ep = eps.get(key)
        if ep is None:
            missing["endpoint"] += 1
            continue
        dock = scores.get(key)
        if dock is None:
            missing["dock"] += 1
            continue
        records.append({**meta, **ep, "docking": dock, "key": key})
    print(f"[join] usable molecules: {len(records):,}  missing={missing}")

    # ---- group by (formula, hac) ------------------------------------------ #
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in records:
        groups[(r["formula"], r["hac"])].append(r)
    print(f"[group] {len(groups):,} (formula, hac) groups")

    # ---- greedy matching, no endpoint reuse -------------------------------- #
    pairs: list[dict] = []
    for key, items in groups.items():
        if len(items) < 2:
            continue
        items.sort(key=lambda r: r["soft_novel"])
        used: set[int] = set()
        n = len(items)
        for i in range(n):
            if i in used:
                continue
            w = items[i]
            got = 0
            for j in range(n - 1, i, -1):
                if j in used:
                    continue
                l = items[j]
                if l["soft_novel"] - w["soft_novel"] < args.delta:
                    break
                if abs(l["tpsa"] - w["tpsa"]) > args.tpsa_tol:
                    continue
                if abs(l["docking"] - w["docking"]) > args.dock_tol:
                    continue
                used.add(i)
                used.add(j)
                pairs.append({"w": w, "l": l})
                got += 1
                if got >= args.max_losers_per_winner:
                    break
    print(f"[pairs] delta={args.delta}  matched pairs: {len(pairs):,}")

    if not pairs:
        raise SystemExit("no pairs produced - relax delta or tolerances")

    # ---- split ------------------------------------------------------------- #
    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    n = len(pairs)
    n_val = max(1, int(n * 0.08))
    n_test = max(1, int(n * 0.08))
    splits = {
        "test": pairs[:n_test],
        "validation": pairs[n_test:n_test + n_val],
        "train": pairs[n_test + n_val:],
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audit = {"delta": args.delta, "tpsa_tol": args.tpsa_tol, "dock_tol": args.dock_tol,
             "source_rule": f"novel_soft_frac_exact_formula_hac_tpsa_dock_delta{args.delta}",
             "usable_molecules": len(records), "groups": len(groups), "pairs": {}}

    all_pairs = []
    for split, items in splits.items():
        rows = []
        for k, p in enumerate(items):
            w, l = p["w"], p["l"]
            rows.append({
                "pair_index": len(all_pairs) + k,
                "split": split,
                "winner_id": w["sample_id"], "loser_id": l["sample_id"],
                "winner_docking_raw": round(w["docking"], 4),
                "loser_docking_raw": round(l["docking"], 4),
                "reward_winner": round(1.0 - w["soft_novel"], 6),
                "reward_loser": round(1.0 - l["soft_novel"], 6),
                "reward_gap": round(l["soft_novel"] - w["soft_novel"], 6),
                "winner_shard": w["pt_shard"], "loser_shard": l["pt_shard"],
                "winner_index": w["mol_index"], "loser_index": l["mol_index"],
                "winner_n_nodes": w["n_nodes"], "loser_n_nodes": l["n_nodes"],
                "n_nodes": max(w["n_nodes"], l["n_nodes"]),
                "source_rule": audit["source_rule"],
                "winner_soft_novel": round(w["soft_novel"], 6),
                "loser_soft_novel": round(l["soft_novel"], 6),
                "soft_novel_gap": round(l["soft_novel"] - w["soft_novel"], 6),
                "winner_null_pct": round(w["null_pct"], 5),
                "loser_null_pct": round(l["null_pct"], 5),
                "winner_formula": w["formula"], "loser_formula": l["formula"],
                "winner_heavy_atom_count": w["hac"], "loser_heavy_atom_count": l["hac"],
                "winner_tpsa": round(w["tpsa"], 3), "loser_tpsa": round(l["tpsa"], 3),
                "abs_tpsa_gap": round(abs(w["tpsa"] - l["tpsa"]), 3),
                "abs_docking_gap": round(abs(w["docking"] - l["docking"]), 3),
                "winner_murcko_scaffold": w["scaffold"], "loser_murcko_scaffold": l["scaffold"],
                "winner_smiles": w["smiles"], "loser_smiles": l["smiles"],
            })
        all_pairs.extend(rows)
        with (OUT_DIR / f"{split}_pairs.csv").open("w", newline="") as fh:
            wt = csv.DictWriter(fh, fieldnames=FIELDS)
            wt.writeheader()
            wt.writerows(rows)
        audit["pairs"][split] = len(rows)
        if rows:
            gaps = [r["soft_novel_gap"] for r in rows]
            dtpsa = [r["abs_tpsa_gap"] for r in rows]
            ddock = [r["abs_docking_gap"] for r in rows]
            audit[f"{split}_stats"] = {
                "mean_soft_novel_gap": round(sum(gaps) / len(gaps), 4),
                "mean_abs_tpsa_gap": round(sum(dtpsa) / len(dtpsa), 4),
                "mean_abs_docking_gap": round(sum(ddock) / len(ddock), 4),
                "winner_own_scaffold_rate": round(
                    sum(1 for r in rows if r["winner_murcko_scaffold"]
                        and r["winner_murcko_scaffold"] != r["loser_murcko_scaffold"]) / len(rows), 4),
            }

    with (OUT_DIR / "preference_pairs.csv").open("w", newline="") as fh:
        wt = csv.DictWriter(fh, fieldnames=FIELDS)
        wt.writeheader()
        wt.writerows(all_pairs)
    (OUT_DIR / "pair_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print()
    for split in ("train", "validation", "test"):
        s = audit.get(f"{split}_stats")
        if s:
            print(f"  {split:<11s} n={audit['pairs'][split]:>6,}  "
                  f"Δnovel={s['mean_soft_novel_gap']:.3f}  "
                  f"ΔTPSA={s['mean_abs_tpsa_gap']:.2f}  "
                  f"Δdock={s['mean_abs_docking_gap']:.3f}")
    print(f"\n[wrote] {OUT_DIR}")


if __name__ == "__main__":
    main()
