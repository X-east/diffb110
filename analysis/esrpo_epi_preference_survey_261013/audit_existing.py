"""Audit existing EPI screens and cross-source ESRPO artifacts without mutation."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch


LOGGER = logging.getLogger(__name__)


def file_hash(path: Path) -> str:
    """Return a file SHA256 for comparison with a training manifest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Write current counts, provenance checks, and a readable historical plot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root, output = args.project_root, args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(output / "audit.log")])
    sys.path.insert(0, str(root))
    from ddpo.global_dpo.endpoint_data import EndpointStore, load_pair_manifest, validate_split_pairs

    screen_dir = root / "analysis/rnd_epi_chembl_similarity_20260618"
    groups = pd.read_csv(screen_dir / "group_summary.csv")
    groups.to_csv(output / "historical_53_seed_groups.csv", index=False)
    matches = pd.read_csv(screen_dir / "matched_candidates.csv.gz",
                          usecols=["chembl_id", "canonical_smiles", "ecfp4_max", "fcfp4_max", "matched_groups"])
    winners = pd.read_csv(root / "data/crosssource/chembl_endpoint_shards_260929/winner_index.csv")
    winner_ids = set(winners["sample_id"])
    pool_rows = []
    for name, mask in [("ECFP4>=0.70", matches.ecfp4_max.ge(0.70)),
                       ("ECFP4>=0.55", matches.ecfp4_max.ge(0.55)),
                       ("ECFP4>=0.40", matches.ecfp4_max.ge(0.40)),
                       ("FCFP4>=0.55", matches.fcfp4_max.ge(0.55))]:
        subset = matches[mask]
        pool_rows.append({"rule": name, "reference_count": 53,
                          "screen_rows": len(subset), "unique_smiles": subset.canonical_smiles.nunique(),
                          "ids_in_existing_winner_index": len(set(subset.chembl_id) & winner_ids),
                          "scope": "historical 53 seeds; not four-seed counts or validated positives"})
    pd.DataFrame(pool_rows).to_csv(output / "historical_pool_endpoint_overlap.csv", index=False)
    pair_dir = root / "analysis/crosssource_pairs_260929/pairs_size"
    manifest = json.loads((root / "OUT/diffsbdd_esrpo_crosssource_size_smoke/run_manifest.json").read_text())
    splits, pair_counts, hash_checks = {}, {}, {}
    for split in ("train", "validation", "test"):
        path = pair_dir / f"{split}_pairs.csv"
        splits[split] = load_pair_manifest(path)
        pair_counts[split] = len(splits[split])
        stored = manifest.get(f"{split}_pair_manifest_sha256")
        actual = file_hash(path)
        hash_checks[split] = {"stored": stored, "current": actual, "matches": stored == actual}
    validate_split_pairs(splits["train"], splits["validation"], splits["test"])
    frame = pd.concat([pd.read_csv(pair_dir / f"{s}_pairs.csv") for s in splits], ignore_index=True)
    store = EndpointStore()
    endpoint_samples = []
    for split, pairs in splits.items():
        pair = pairs[0]
        for side in ("winner", "loser"):
            sample = store.sample(getattr(pair, f"{side}_shard"), getattr(pair, f"{side}_index"),
                                  expected_sample_id=getattr(pair, f"{side}_id"))
            endpoint_samples.append({"split": split, "side": side, "id": sample.sample_id,
                                     "n_nodes": sample.n_nodes, "x_shape": list(sample.x.shape),
                                     "v_shape": list(sample.v.shape)})
    screen = json.loads((root / "analysis/epi_pharmacophore_screen_261013/summary.json").read_text())
    snapshot = (root / "OUT/diffsbdd_esrpo_crosssource_size_smoke/config.snapshot.yaml").read_text()
    checkpoint = torch.load(root / "OUT/diffsbdd_esrpo_crosssource_size_smoke/checkpoints/diffsbdd-global-dpo-epoch=0001.pt",
                            map_location="cpu", weights_only=False)
    optimizer_lrs = [group["lr"] for group in checkpoint["optimizer_state_dict"]["param_groups"]]
    result = {
        "audit_date": "2026-10-13", "environment": "dif", "project_root": str(root.resolve()),
        "seed_scope_confirmed": ["PABN", "NMP", "MBX2319", "D13-9001"],
        "historical_screen_scope": 53, "historical_matched_rows": len(matches),
        "winner_index_rows": len(winners), "winner_unique_ids": winners.sample_id.nunique(),
        "winner_shards": len(list((root / "data/crosssource/chembl_endpoint_shards_260929").glob("shard_*.pt"))),
        "historical_pool_endpoint_overlap": pool_rows,
        "current_crosssource_pair_counts": pair_counts,
        "current_pair_total": len(frame),
        "max_pair_ha_gap": int((frame.winner_n_nodes - frame.loser_n_nodes).abs().max()),
        "endpoint_split_leakage_check": "passed", "endpoint_reader_samples": endpoint_samples,
        "pair_hash_checks_against_completed_training": hash_checks,
        "training_snapshot": {k: manifest.get(k) for k in ["status", "started_at", "train_pairs", "validation_pairs", "test_pairs", "variant", "objective"]},
        "snapshot_learning_rate_lines": [line.strip() for line in snapshot.splitlines() if "learning_rate:" in line],
        "checkpoint_optimizer_learning_rates": optimizer_lrs,
        "pair_audit": json.loads((pair_dir / "pair_audit.json").read_text()),
        "existing_generated_pharmacophore_screen": {k: screen[k] for k in ["counts", "screened_top_records", "screened_unique_structures"]},
        "limitations": ["No new four-seed ChEMBL screen or training was run.",
                        "ChEMBL ID overlap is availability, not structure-version or pose validation.",
                        "Sample endpoint reads are not an exhaustive shard integrity audit.",
                        "ID-based split checks do not prove scaffold separation."]}
    (output / "current_state_audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    selected = groups[groups.group_id.isin(["G02", "G03", "G04", "G05", "G07", "G10"])].copy()
    labels = ["ECFP4>=.70 or ECFP6>=.65", "FCFP4>=.70", "ECFP4>=.55 or ECFP6>=.50",
              "FCFP4>=.55", "ECFP4/6 or FCFP4>=.40", "FP union>=.30 or MACCS>=.65"]
    plt.rcParams.update({"font.size": 14, "axes.titlesize": 18, "axes.labelsize": 16})
    fig, ax = plt.subplots(figsize=(13, 5.8))
    ax.barh(labels, selected.cumulative_hits, color=["#3579ae"] * 4 + ["#c68b37"] * 2)
    ax.set_xscale("log")
    ax.set_xlim(100, 1_500_000)
    for y, count in enumerate(selected.cumulative_hits):
        ax.text(count * 1.13, y, f"{count:,}", va="center")
    ax.invert_yaxis()
    ax.set_xlabel("Cumulative hits (log scale; groups overlap)")
    ax.set_title("Existing ChEMBL screen: all 53 EPI references")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.2)
    fig.text(0.02, 0.01, "Not four-reference counts. FCFP/MACCS are 2D fingerprints, not 3D pharmacophore matches.", fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for ext in ("png", "svg"):
        fig.savefig(output / f"historical_53_seed_pool_sizes.{ext}", dpi=180)
    plt.close(fig)
    LOGGER.info("audit complete: winners=%d current_pairs=%d pool_rules=%d", len(winners), len(frame), len(pool_rows))
    LOGGER.info("training manifest hash matches: %s", {k: v["matches"] for k, v in hash_checks.items()})


if __name__ == "__main__":
    main()
