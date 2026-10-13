"""Evaluate the frozen original DiffSBDD checkpoint on the smoke pairs."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ddpo.global_dpo.endpoint_data import EndpointStore, load_pair_manifest
from ddpo.global_dpo.global_score import compute_global_score
from ddpo.global_dpo.paired_noise import build_paired_noisy_batch
from ddpo.integration.diffsbdd_model import DiffSBDDModel


OUT = Path(__file__).resolve().parent
PAIR_DIR = ROOT / "analysis/diffsbdd_global_e2po_good_bad_alidiff_260825/pairs"
CHECKPOINT = ROOT / "DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt"
POCKET = ROOT / "data/8GJL.pdb"
SIDECAR = ROOT / "analysis/offline_iql_labels_alidiff_unidock_affinity_8gpu_full_260824/reward_sidecar.csv"
TIMESTEPS = (50, 125, 250, 375, 500)
VARIANTS = {
    "x_only": (1.0, 0.0),
    "v_only": (0.0, 1.0),
    "native_joint": (0.23076923076923078, 0.7692307692307693),
}


def endpoint_items(store: EndpointStore, pair) -> list[object]:
    """Load one pair with identity checks."""
    return [
        store.sample(pair.winner_shard, pair.winner_index, expected_sample_id=pair.winner_id),
        store.sample(pair.loser_shard, pair.loser_index, expected_sample_id=pair.loser_id),
    ]


@torch.no_grad()
def evaluate_split(model: DiffSBDDModel, pairs: list[object]) -> dict[str, list[float]]:
    """Evaluate all variants using one native model forward per pair/timestep."""
    store = EndpointStore()
    result = {variant: [] for variant in VARIANTS}
    batch_pairs = 2
    for bucket_index, timestep in enumerate(TIMESTEPS):
        correct = {variant: 0 for variant in VARIANTS}
        total = 0
        for batch_index, batch_start in enumerate(range(0, len(pairs), batch_pairs)):
            batch = pairs[batch_start : batch_start + batch_pairs]
            items = [item for pair in batch for item in endpoint_items(store, pair)]
            noisy = build_paired_noisy_batch(
                model,
                items,
                timestep,
                timestep_grid=500,
                noise_seed=1 + 1_000_003 + bucket_index * 100_003 + batch_index,
                noise_coupling="independent",
            )
            scored = compute_global_score(model, noisy, x_weight=1.0, v_weight=1.0)
            x_loss = scored.x_loss.reshape(-1, 2)
            v_loss = scored.v_loss.reshape(-1, 2)
            for variant, (x_weight, v_weight) in VARIANTS.items():
                score = -(x_weight * x_loss + v_weight * v_loss)
                correct[variant] += int((score[:, 0] > score[:, 1]).sum().item())
            total += len(batch)
        for variant in VARIANTS:
            result[variant].append(correct[variant] / total)
    return result


def main() -> None:
    """Run baseline evaluation and write CSV/JSON artifacts."""
    torch.manual_seed(1)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the original-checkpoint baseline")
    model = DiffSBDDModel(
        checkpoint_path=str(CHECKPOINT),
        pocket_pdb_path=str(POCKET),
        ref_ligand_path="C:1101",
        device="cuda:0",
        dataset_name="crossdock",
    )
    rows: list[dict[str, object]] = []
    for split in ("validation", "test"):
        pairs = load_pair_manifest(
            PAIR_DIR / f"{split}_pairs.csv",
            reward_sidecar=SIDECAR,
            reward_key="reward_total_raw",
            reward_sign=1.0,
            allow_endpoint_reuse=True,
        )[:4096]
        scores = evaluate_split(model, pairs)
        for variant, values in scores.items():
            for timestep, accuracy in zip(TIMESTEPS, values):
                rows.append(
                    {
                        "split": split,
                        "variant": variant,
                        "timestep": timestep,
                        "pair_accuracy": accuracy,
                    }
                )
    with (OUT / "original_baseline_timestep_accuracy.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "variant", "timestep", "pair_accuracy"])
        writer.writeheader()
        writer.writerows(rows)
    summary: dict[str, dict[str, float]] = {}
    for split in ("validation", "test"):
        for variant in VARIANTS:
            values = [
                float(row["pair_accuracy"])
                for row in rows
                if row["split"] == split and row["variant"] == variant
            ]
            summary[f"{split}:{variant}"] = {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
            }
    (OUT / "original_baseline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
