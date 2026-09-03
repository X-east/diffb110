#!/usr/bin/env python3
"""Build 50:50 docking/ChEMBL preference pairs from matched endpoints.

The existing composition-matched pairs provide endpoint compatibility. This
script re-orients and filters those pairs using train-frozen empirical ranks:
both objectives are mapped to [0, 1], with lower docking_raw treated as better.
Technical failures are excluded and never used as chemical negatives.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rank_map(values: np.ndarray, reference: np.ndarray, higher_is_better: bool) -> np.ndarray:
    ref = np.sort(reference if higher_is_better else -reference)
    query = values if higher_is_better else -values
    return np.searchsorted(ref, query, side="right") / max(len(ref), 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docking-sidecar", type=Path, required=True)
    ap.add_argument("--chembl-sidecar", type=Path, required=True)
    ap.add_argument("--base-pair-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--min-docking-gap", type=float, default=0.3)
    ap.add_argument("--max-chembl-drop", type=float, default=0.03)
    ap.add_argument("--min-score-gap", type=float, default=0.10)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dock = pd.read_csv(args.docking_sidecar, low_memory=False)
    chem = pd.read_csv(args.chembl_sidecar, low_memory=False)
    dock_cols = ["sample_id", "rdkit_valid", "docking_success", "docking_raw"]
    chem_cols = ["sample_id", "ecfp4_success", "ecfp4_top5_mean_similarity"]
    merged = dock[dock_cols].merge(chem[chem_cols], on="sample_id", validate="one_to_one")
    merged["valid_joint"] = (
        merged["rdkit_valid"].eq(1)
        & merged["docking_success"].eq(1)
        & merged["ecfp4_success"].eq(1)
        & np.isfinite(merged["docking_raw"])
        & np.isfinite(merged["ecfp4_top5_mean_similarity"])
    )
    # Train-only empirical CDFs are frozen and reused for all splits.
    # The ChEMBL sidecar carries the frozen split assignment. Use every
    # train-split endpoint for the empirical CDF, not only endpoints selected
    # into the structural base pairs.
    train_ids = set(chem.loc[chem["split"].astype(str).eq("train"), "sample_id"].astype(str)) if "split" in chem else set()
    if not train_ids:
        raise RuntimeError("chembl sidecar has no train split assignments")
    train_ref = merged[merged.sample_id.isin(train_ids) & merged.valid_joint]
    if train_ref.empty:
        raise RuntimeError("no valid train reference endpoints")
    merged["docking_rank"] = np.nan
    merged["chembl_rank"] = np.nan
    valid = merged.valid_joint
    merged.loc[valid, "docking_rank"] = rank_map(
        merged.loc[valid, "docking_raw"].to_numpy(float),
        train_ref.docking_raw.to_numpy(float), False,
    )
    merged.loc[valid, "chembl_rank"] = rank_map(
        merged.loc[valid, "ecfp4_top5_mean_similarity"].to_numpy(float),
        train_ref.ecfp4_top5_mean_similarity.to_numpy(float), True,
    )
    merged["combined_score"] = 0.5 * merged["docking_rank"] + 0.5 * merged["chembl_rank"]
    merged.to_csv(args.output_dir / "combined_reward_sidecar.csv", index=False)
    lookup = merged.set_index("sample_id")
    summaries = {"status": "completed", "weights": {"docking": 0.5, "chembl": 0.5}, "splits": {}}
    out_fields = [
        "pair_index", "split", "winner_id", "loser_id", "reward_winner", "reward_loser",
        "reward_gap", "winner_shard", "loser_shard", "winner_index", "loser_index",
        "winner_n_nodes", "loser_n_nodes", "n_nodes", "loser_category", "source_rule",
    ]
    for split in ("train", "validation", "test"):
        base = pd.read_csv(args.base_pair_dir / f"{split}_pairs.csv", low_memory=False)
        accepted = []
        rejected = {"invalid": 0, "docking_gap": 0, "chembl_drop": 0, "score_gap": 0, "tie": 0}
        for _, row in base.iterrows():
            a, b = str(row.winner_id), str(row.loser_id)
            if a not in lookup.index or b not in lookup.index:
                rejected["invalid"] += 1; continue
            ra, rb = lookup.loc[a], lookup.loc[b]
            if not bool(ra.valid_joint) or not bool(rb.valid_joint):
                rejected["invalid"] += 1; continue
            if float(ra.combined_score) == float(rb.combined_score):
                rejected["tie"] += 1; continue
            w, l = (ra, rb) if ra.combined_score > rb.combined_score else (rb, ra)
            if float(l.docking_raw) - float(w.docking_raw) < args.min_docking_gap:
                rejected["docking_gap"] += 1; continue
            if float(w.ecfp4_top5_mean_similarity) - float(l.ecfp4_top5_mean_similarity) < -args.max_chembl_drop:
                rejected["chembl_drop"] += 1; continue
            gap = float(w.combined_score) - float(l.combined_score)
            if gap < args.min_score_gap:
                rejected["score_gap"] += 1; continue
            accepted.append({
                "pair_index": len(accepted), "split": split, "winner_id": str(w.name), "loser_id": str(l.name),
                "reward_winner": float(w.combined_score), "reward_loser": float(l.combined_score),
                "reward_gap": gap, "winner_shard": str(row.winner_shard if str(w.name)==a else row.loser_shard),
                "loser_shard": str(row.loser_shard if str(l.name)==b else row.winner_shard),
                "winner_index": int(row.winner_index if str(w.name)==a else row.loser_index),
                "loser_index": int(row.loser_index if str(l.name)==b else row.winner_index),
                "winner_n_nodes": int(row.winner_n_nodes if str(w.name)==a else row.loser_n_nodes),
                "loser_n_nodes": int(row.loser_n_nodes if str(l.name)==b else row.winner_n_nodes),
                "n_nodes": int(row.n_nodes), "loser_category": "combined_lower_score",
                "source_rule": "composition_matched_docking_chembl_50_50_v1",
            })
        pd.DataFrame(accepted, columns=out_fields).to_csv(args.output_dir / f"{split}_pairs.csv", index=False)
        gaps = [x["reward_gap"] for x in accepted]
        summaries["splits"][split] = {"base_pairs": int(len(base)), "accepted_pairs": int(len(accepted)), "rejected": rejected,
            "reward_gap_min": float(min(gaps)) if gaps else None, "reward_gap_mean": float(np.mean(gaps)) if gaps else None,
            "manifest_sha256": sha256(args.output_dir / f"{split}_pairs.csv")}
    summaries["source"] = {"docking_sidecar": str(args.docking_sidecar.resolve()), "chembl_sidecar": str(args.chembl_sidecar.resolve()),
                            "base_pair_dir": str(args.base_pair_dir.resolve()), "train_reference_endpoints": len(train_ref)}
    summaries["thresholds"] = {"min_docking_gap": args.min_docking_gap, "max_chembl_drop": args.max_chembl_drop, "min_score_gap": args.min_score_gap}
    (args.output_dir / "pair_manifest_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
