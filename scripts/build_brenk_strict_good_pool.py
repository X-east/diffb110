#!/usr/bin/env python3
"""Build a strict-Brenk exploratory pool and a reproducible ~400 molecule subset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-count", type=int, default=400)
    return parser.parse_args()


def add_scaffold(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["murcko_scaffold"] = [
        MurckoScaffold.MurckoScaffoldSmilesFromSmiles(str(smiles)) or "__acyclic__"
        for smiles in frame["canonical_smiles"]
    ]
    return frame


def score(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    # Percentile ranks make the score scale-independent and fully deterministic.
    frame["quality_score"] = (
        frame["qed"].rank(method="first", ascending=False, pct=True)
        + frame["sa_raw"].rank(method="first", ascending=True, pct=True)
        + frame["umap_nearest_chembl_distance"].rank(method="first", ascending=True, pct=True)
        + frame["umap_chembl_k20_distance"].rank(method="first", ascending=True, pct=True)
        + frame["lipinski_violation_count"].rank(method="first", ascending=True, pct=True)
    )
    return frame.sort_values(
        ["quality_score", "trajectory_id"], ascending=[True, True], kind="mergesort"
    )


def select_diverse(frame: pd.DataFrame, target: int) -> pd.DataFrame:
    ranked = score(frame)
    chosen: list[int] = []
    seen_scaffolds: set[str] = set()
    # First pass: one best molecule per scaffold, giving broad structural coverage.
    for idx, row in ranked.iterrows():
        if row["murcko_scaffold"] not in seen_scaffolds:
            chosen.append(idx)
            seen_scaffolds.add(row["murcko_scaffold"])
        if len(chosen) >= target:
            break
    # If fewer scaffolds exist, fill deterministically by the composite score.
    if len(chosen) < target:
        for idx in ranked.index:
            if idx not in chosen:
                chosen.append(idx)
            if len(chosen) >= target:
                break
    selected = ranked.loc[chosen].copy()
    selected["selection_rank"] = range(1, len(selected) + 1)
    selected["candidate_good_label"] = True
    return selected


def write_sdf(facts: pd.DataFrame, path: Path, label: str) -> int:
    wanted = set(facts["trajectory_id"].astype(str))
    count = 0
    writer = Chem.SDWriter(str(path))
    writer.SetKekulize(False)
    try:
        for sdf_name in sorted(facts["sdf_path"].dropna().astype(str).unique()):
            for molecule in Chem.SDMolSupplier(sdf_name, removeHs=False, sanitize=False):
                if molecule is None or not molecule.HasProp("sample_id"):
                    continue
                sample_id = molecule.GetProp("sample_id")
                if sample_id in wanted:
                    molecule.SetProp("screen_label", label)
                    writer.write(molecule)
                    count += 1
    finally:
        writer.close()
    return count


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    facts = pd.read_csv(args.facts, low_memory=False)
    required = [
        "rdkit_valid", "qed_ge_0_4", "sa_le_6", "logp_in_neg1_to_5",
        "tpsa_in_20_to_140", "fused_rings_le_4", "complex_rings_le_1",
        "brenk_eq_0", "single_fragment", "allowed_organic_elements",
        "mol_wt_in_120_to_600", "heavy_atoms_in_10_to_35", "hba_le_10",
        "hbd_le_5", "formal_charge_abs_le_1", "lipinski_violations_le_1",
        "carbon_fraction_le_0_85", "three_member_rings_allowed_new",
    ]
    missing = [column for column in required if column not in facts.columns]
    if missing:
        raise ValueError(f"missing required filter columns: {missing}")
    strict_mask = facts[required].fillna(False).astype(bool).all(axis=1)
    strict = facts[strict_mask].copy()
    strict = add_scaffold(strict)
    selected = select_diverse(strict, min(args.target_count, len(strict)))
    strict.to_csv(args.output_dir / "brenk_strict_pool_facts.csv.gz", index=False, compression="gzip")
    selected.to_csv(args.output_dir / "candidate_good_400_facts.csv.gz", index=False, compression="gzip")
    pool_sdf = args.output_dir / "brenk_strict_pool.sdf"
    good_sdf = args.output_dir / "candidate_good_400.sdf"
    pool_count = write_sdf(strict, pool_sdf, "brenk_strict_pool")
    good_count = write_sdf(selected, good_sdf, "candidate_good")
    selected[["trajectory_id", "trajectory_shard", "sdf_path", "candidate_good_label"]].to_csv(
        args.output_dir / "candidate_good_400_trajectory_manifest.csv", index=False
    )
    fig, ax = plt.subplots(figsize=(9, 7), dpi=180)
    ax.scatter(facts["umap1"], facts["umap2"], s=0.35, alpha=0.08, c="#9aa0a6", label="All valid")
    ax.scatter(strict["umap1"], strict["umap2"], s=0.65, alpha=0.16, c="#2b6cb0", label=f"Brenk strict ({len(strict):,})")
    ax.scatter(selected["umap1"], selected["umap2"], s=7, alpha=0.8, c="#c53030", label=f"Candidate good ({len(selected):,})")
    ax.set(xlabel="UMAP 1", ylabel="UMAP 2", title="Strict Brenk pool and ~400 candidate-good subset")
    ax.legend(frameon=False, markerscale=2)
    fig.tight_layout()
    fig.savefig(args.output_dir / "brenk_strict_candidate_good_umap.png", dpi=220)
    plt.close(fig)
    summary = {
        "status": "complete",
        "input_rows": int(len(facts)),
        "strict_brenk_pool_count": int(len(strict)),
        "candidate_good_count": int(len(selected)),
        "pool_sdf_count": int(pool_count),
        "candidate_sdf_count": int(good_count),
        "condition": "All explicitly listed RDKit/property/Brenk gates; new three-member-ring policy; carbon_fraction<=0.85",
        "selection": "lowest sum of percentile ranks for QED (descending), SA, nearest/k20 ChEMBL UMAP distance, and Lipinski violations; one best molecule per Murcko scaffold first",
        "label_boundary": "candidate_good is a chemistry-screening label, not measured activity or docking-confirmed goodness",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "README.md").write_text(
        "# 严格 Brenk 候选池\n\n"
        "主候选池是按全部条件通过的全部分子，不强行限制为 400 个；400 个文件只是可选的 scaffold 多样子集。\n"
        "本目录逐项使用 RDKit/property/Brenk 严格门，并使用新的三元环规则（允许0–2个、全碳、不共享环原子、外部邻接不超过1个）。\n"
        "`candidate_good_400` 仅表示候选正样本，不代表活性或结合正确性；正式 IQL 前仍需构建有限 reward sidecar、冻结无泄漏 split、检查 scaffold 支持，并用 BC-all/filtered BC/AWR 作对照。\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
