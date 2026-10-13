"""Run DiffSBDD-native Global-DPO gradient and short-pilot audits."""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_scatter import scatter_add

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ddpo.integration.diffsbdd_model import DiffSBDDModel


OUT = Path(__file__).resolve().parent
MANIFEST = ROOT / "analysis/offline_iql_extreme_good_bad_pool_260822/inputs/selected_extremes_manifest.csv.gz"
CHECKPOINT = ROOT / "DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt"
POCKET = ROOT / "data/8GJL.pdb"
REF_LIGAND = "C:1101"
T = 1000
SEED = 1
ATOM_DIM = 10


def parse_args() -> argparse.Namespace:
    """Parse the isolated experiment options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("gradient", "pilot"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pair-count", type=int, default=256)
    parser.add_argument("--batch-pairs", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument(
        "--timesteps",
        type=int,
        nargs="+",
        default=[1, 100, 250, 500, 750, 1000],
    )
    return parser.parse_args()


def write_pair_manifest(pairs: list[dict[str, object]], path: Path) -> None:
    """Persist the exact frozen winner/loser pairs used by one run."""
    rows = []
    for index, pair in enumerate(pairs):
        winner = pair["winner"]
        loser = pair["loser"]
        rows.append(
            {
                "pair_index": index,
                "n_nodes": int(pair["n_nodes"]),
                "hard_bad_category": str(pair["hard_bad_category"]),
                "winner_trajectory_id": winner["trajectory_id"],
                "loser_trajectory_id": loser["trajectory_id"],
                "winner_shard": winner["trajectory_shard"],
                "winner_mol_index": int(winner["mol_index"]),
                "loser_shard": loser["trajectory_shard"],
                "loser_mol_index": int(loser["mol_index"]),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def load_rows() -> pd.DataFrame:
    """Load the frozen endpoint manifest and enforce finite labels."""
    frame = pd.read_csv(MANIFEST, compression="gzip")
    frame = frame[frame["extreme_label"].isin(["good", "hard_bad"])].copy()
    frame["heavy_atom_count"] = pd.to_numeric(frame["heavy_atom_count"], errors="raise").astype(int)
    frame["split"] = frame["split"].fillna("unknown")
    return frame.reset_index(drop=True)


def make_pairs(frame: pd.DataFrame, split: str, count: int, seed: int) -> list[dict[str, object]]:
    """Build exact-heavy-atom-count winner/loser pairs within one split."""
    sub = frame[frame["split"] == split]
    rng = random.Random(seed)
    pairs: list[dict[str, object]] = []
    for n_nodes, group in sub.groupby("heavy_atom_count", sort=True):
        good = group[group["extreme_label"] == "good"].to_dict("records")
        bad = group[group["extreme_label"] == "hard_bad"].to_dict("records")
        rng.shuffle(good)
        rng.shuffle(bad)
        for winner, loser in zip(good, bad):
            pairs.append(
                {
                    "winner": winner,
                    "loser": loser,
                    "n_nodes": int(n_nodes),
                    "hard_bad_category": str(loser.get("hard_bad_category", "unknown")),
                }
            )
    rng.shuffle(pairs)
    return pairs[: min(count, len(pairs))]


class ShardCache:
    """Load trajectory shards once and expose terminal endpoint tensors."""

    def __init__(self) -> None:
        self._cache: dict[str, dict[str, object]] = {}

    def item(self, row: dict[str, object]) -> dict[str, object]:
        """Return one endpoint without padding nodes."""
        path = str(row["trajectory_shard"])
        if path not in self._cache:
            self._cache[path] = torch.load(path, map_location="cpu", weights_only=False)
        shard = self._cache[path]
        index = int(row["mol_index"])
        n_nodes = int(shard["nodesxsample"][index])
        return {
            "sample_id": str(row["trajectory_id"]),
            "label": str(row["extreme_label"]),
            "n_nodes": n_nodes,
            "x": shard["x"][index, :n_nodes].float(),
            "v": shard["categorical"][index, :n_nodes].float(),
        }


def flatten_pairs(pairs: Iterable[dict[str, object]], cache: ShardCache) -> list[dict[str, object]]:
    """Flatten pairs in winner-then-loser order."""
    items: list[dict[str, object]] = []
    for pair in pairs:
        items.extend((cache.item(pair["winner"]), cache.item(pair["loser"])))
    return items


def pack(items: list[dict[str, object]], device: torch.device) -> dict[str, torch.Tensor]:
    """Pack variable-size endpoints into DiffSBDD flat tensors."""
    sizes = torch.tensor([int(item["n_nodes"]) for item in items], device=device)
    mask = torch.repeat_interleave(torch.arange(len(items), device=device), sizes).long()
    return {
        "x": torch.cat([item["x"] for item in items], dim=0).to(device),
        "one_hot": torch.cat([item["v"] for item in items], dim=0).to(device),
        "size": sizes.long(),
        "mask": mask,
    }


def build_noisy_batch(
    model: DiffSBDDModel,
    items: list[dict[str, object]],
    timestep: int,
    noise_seed: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Create a common-noise DiffSBDD denoising batch for paired endpoints."""
    device = torch.device(model.device)
    ligand = pack(items, device)
    pocket = model._replicate_pocket(len(items))
    ddpm = model.ddpm
    ligand, pocket = ddpm.normalize(ligand, pocket)
    xh0_lig = torch.cat([ligand["x"], ligand["one_hot"]], dim=1)
    xh0_pocket = torch.cat([pocket["x"], pocket["one_hot"]], dim=1)
    xh0_lig[:, : ddpm.n_dims], xh0_pocket[:, : ddpm.n_dims] = ddpm.remove_mean_batch(
        xh0_lig[:, : ddpm.n_dims],
        xh0_pocket[:, : ddpm.n_dims],
        ligand["mask"],
        pocket["mask"],
    )
    t = torch.full(
        (len(items), 1),
        float(timestep) / float(T),
        device=device,
        dtype=xh0_lig.dtype,
    )
    gamma_t = ddpm.inflate_batch_array(ddpm.gamma(t), ligand["x"])
    alpha_t = ddpm.alpha(gamma_t, xh0_lig)
    sigma_t = ddpm.sigma(gamma_t, xh0_lig)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(noise_seed))
    eps = torch.randn(xh0_lig.shape, device=device, dtype=xh0_lig.dtype, generator=generator)
    # Paired rows have equal N by construction; share epsilon within each pair.
    offset = 0
    for pair_index in range(0, len(items), 2):
        n_nodes = int(items[pair_index]["n_nodes"])
        eps[offset + n_nodes : offset + 2 * n_nodes] = eps[offset : offset + n_nodes]
        offset += 2 * n_nodes
    z_t_lig = alpha_t[ligand["mask"]] * xh0_lig + sigma_t[ligand["mask"]] * eps
    z_t_lig[:, : ddpm.n_dims], xh0_pocket[:, : ddpm.n_dims] = ddpm.remove_mean_batch(
        z_t_lig[:, : ddpm.n_dims],
        xh0_pocket[:, : ddpm.n_dims],
        ligand["mask"],
        pocket["mask"],
    )
    return (ligand, pocket), xh0_pocket, z_t_lig, t


def component_losses(
    model: DiffSBDDModel,
    items: list[dict[str, object]],
    timestep: int,
    noise_seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-sample x/v epsilon MSE with native T=1000 semantics."""
    (ligand, pocket), xh0_pocket, z_t_lig, t = build_noisy_batch(
        model, items, timestep, noise_seed
    )
    prediction, _ = model.ddpm.dynamics(
        z_t_lig, xh0_pocket, t, ligand["mask"], pocket["mask"]
    )
    squared = (z_t_lig.new_zeros(z_t_lig.shape) + 0.0)
    # The model predicts epsilon; reconstruct epsilon from the same forward-noise
    # relation through the current z_t and x0 is unnecessary for the audit only
    # if the sampled epsilon is retained, so build it deterministically again.
    ddpm = model.ddpm
    gamma_t = ddpm.inflate_batch_array(ddpm.gamma(t), ligand["x"])
    alpha_t = ddpm.alpha(gamma_t, torch.cat([ligand["x"], ligand["one_hot"]], dim=1))
    sigma_t = ddpm.sigma(gamma_t, torch.cat([ligand["x"], ligand["one_hot"]], dim=1))
    # Recreate the exact common-noise tensor used above.
    generator = torch.Generator(device=z_t_lig.device)
    generator.manual_seed(int(noise_seed))
    eps = torch.randn(z_t_lig.shape, device=z_t_lig.device, dtype=z_t_lig.dtype, generator=generator)
    offset = 0
    for pair_index in range(0, len(items), 2):
        n_nodes = int(items[pair_index]["n_nodes"])
        eps[offset + n_nodes : offset + 2 * n_nodes] = eps[offset : offset + n_nodes]
        offset += 2 * n_nodes
    del squared, alpha_t, sigma_t
    squared_error = (eps - prediction).square()
    x_sum = scatter_add(squared_error[:, : ddpm.n_dims].sum(-1), ligand["mask"], dim=0)
    v_sum = scatter_add(squared_error[:, ddpm.n_dims :].sum(-1), ligand["mask"], dim=0)
    n_nodes = ligand["size"].to(dtype=squared_error.dtype)
    return (
        x_sum / (float(ddpm.n_dims) * n_nodes),
        v_sum / (float(ddpm.atom_nf) * n_nodes),
    )


def make_model(device: str) -> DiffSBDDModel:
    """Load the original DiffSBDD checkpoint and expose T=1000 explicitly."""
    if not torch.cuda.is_available() and device.startswith("cuda"):
        raise RuntimeError("CUDA is required for this experiment; use an escalated dif preflight.")
    model = DiffSBDDModel(
        checkpoint_path=str(CHECKPOINT),
        pocket_pdb_path=str(POCKET),
        ref_ligand_path=REF_LIGAND,
        device=device,
        dataset_name="crossdock",
    )
    model.T = T
    model.model.eval()
    model.ddpm.eval()
    return model


def parameter_gradient(model: DiffSBDDModel, loss: torch.Tensor) -> tuple[float, torch.Tensor]:
    """Backpropagate one scalar and return norm plus a flattened gradient."""
    model.model.zero_grad(set_to_none=True)
    loss.backward()
    values = [p.grad.detach().reshape(-1) for p in model.model.parameters() if p.grad is not None]
    if not values:
        raise RuntimeError("No model gradients were produced.")
    vector = torch.cat(values)
    return float(torch.linalg.vector_norm(vector).detach().cpu()), vector.detach().cpu()


def dpo_surrogate(x_loss: torch.Tensor, v_loss: torch.Tensor, weights: tuple[float, float], beta: float) -> torch.Tensor:
    """Compute a local pairwise DPO surrogate from denoising scores."""
    score = -(weights[0] * x_loss + weights[1] * v_loss)
    margin = score[0::2] - score[1::2]
    return F.softplus(-float(beta) * margin).mean()


def run_gradient_audit(args: argparse.Namespace) -> None:
    """Measure x/v preference-gradient scales across the T=1000 schedule."""
    OUT.mkdir(parents=True, exist_ok=True)
    frame = load_rows()
    pairs = make_pairs(frame, "train", args.pair_count, SEED)
    if not pairs:
        raise RuntimeError("No train pairs were available.")
    cache = ShardCache()
    model = make_model(args.device)
    modes = {
        "x_only": (1.0, 0.0),
        "v_only": (0.0, 1.0),
        "native_joint": (3.0 / 13.0, 10.0 / 13.0),
        "component_balanced": (0.5, 0.5),
    }
    rows: list[dict[str, object]] = []
    for timestep in args.timesteps:
        batch_gradients: dict[str, list[tuple[float, torch.Tensor]]] = {mode: [] for mode in modes}
        batch_losses: dict[str, list[float]] = {mode: [] for mode in modes}
        for batch_start in range(0, len(pairs), args.batch_pairs):
            batch_pairs = pairs[batch_start : batch_start + args.batch_pairs]
            items = flatten_pairs(batch_pairs, cache)
            for mode, weights in modes.items():
                model.model.zero_grad(set_to_none=True)
                x_loss, v_loss = component_losses(
                    model, items, int(timestep), SEED + int(timestep) + batch_start
                )
                loss = dpo_surrogate(x_loss, v_loss, weights, args.beta)
                norm, vector = parameter_gradient(model, loss)
                batch_gradients[mode].append((norm, vector))
                batch_losses[mode].append(float(loss.detach().cpu()))
        for mode, weights in modes.items():
            norms = [value[0] for value in batch_gradients[mode]]
            rows.append(
                {
                    "timestep": int(timestep),
                    "mode": mode,
                    "x_weight": weights[0],
                    "v_weight": weights[1],
                    "dpo_loss": float(np.mean(batch_losses[mode])),
                    "gradient_norm": float(np.median(norms)),
                    "gradient_norm_mean": float(np.mean(norms)),
                    "pair_count": len(pairs),
                    "batch_pairs": args.batch_pairs,
                }
            )
        x_batches = batch_gradients["x_only"]
        v_batches = batch_gradients["v_only"]
        ratios = [x_norm / max(v_norm, 1e-12) for (x_norm, _), (v_norm, _) in zip(x_batches, v_batches)]
        cosines = [
            float(torch.nn.functional.cosine_similarity(x_vector, v_vector, dim=0).item())
            for (_, x_vector), (_, v_vector) in zip(x_batches, v_batches)
        ]
        rows.append(
            {
                "timestep": int(timestep),
                "mode": "x_v_diagnostics",
                "x_weight": None,
                "v_weight": None,
                "dpo_loss": None,
                "gradient_norm": None,
                "gradient_norm_mean": None,
                "x_over_v_gradient_norm": float(np.median(ratios)),
                "cosine_x_v": float(np.median(cosines)),
                "equal_gradient_v_weight": float(np.median(ratios)),
                "pair_count": len(pairs),
                "batch_pairs": args.batch_pairs,
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "gradient_audit.csv", index=False)
    diagnostics = result[result["mode"] == "x_v_diagnostics"].copy()
    summary = {
        "requested_timesteps": args.timesteps,
        "T": T,
        "checkpoint": str(CHECKPOINT),
        "checkpoint_native_model_T_before_override": int(model.ddpm.T),
        "pair_count": len(pairs),
        "train_rows": int(len(frame[frame["split"] == "train"])),
        "hard_bad_categories": frame[frame["split"] == "train"]["hard_bad_category"].value_counts().to_dict(),
        "median_equal_gradient_v_weight": float(diagnostics["equal_gradient_v_weight"].median()),
        "max_equal_gradient_v_weight": float(diagnostics["equal_gradient_v_weight"].max()),
    }
    (OUT / "gradient_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].plot(diagnostics["timestep"], diagnostics["x_over_v_gradient_norm"], marker="o", linewidth=2.5)
    axes[0].set_xlabel("Timestep t / 1000")
    axes[0].set_ylabel("||g_x|| / ||g_v||")
    axes[0].set_title("Gradient scale ratio")
    axes[1].plot(diagnostics["timestep"], diagnostics["cosine_x_v"], marker="o", linewidth=2.5)
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_xlabel("Timestep t / 1000")
    axes[1].set_ylabel("cos(g_x, g_v)")
    axes[1].set_title("x/v gradient alignment")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "gradient_audit.png", dpi=180)
    fig.savefig(OUT / "gradient_audit.svg")
    plt.close(fig)


def evaluate_pairs(
    model: DiffSBDDModel,
    pairs: list[dict[str, object]],
    cache: ShardCache,
    weights: tuple[float, float],
    seed: int,
    batch_pairs: int,
) -> dict[str, float]:
    """Evaluate pair preference accuracy and margin on a fixed set."""
    model.model.eval()
    margins: list[float] = []
    with torch.no_grad():
        for batch_start in range(0, len(pairs), batch_pairs):
            current_pairs = pairs[batch_start : batch_start + batch_pairs]
            if not current_pairs:
                continue
            x_loss, v_loss = component_losses(
                model,
                flatten_pairs(current_pairs, cache),
                500,
                seed + batch_start,
            )
            score = -(weights[0] * x_loss + weights[1] * v_loss)
            margins.extend((score[0::2] - score[1::2]).cpu().tolist())
    values = np.asarray(margins, dtype=float)
    return {
        "pair_accuracy": float(np.mean(values > 0.0)) if len(values) else float("nan"),
        "mean_margin": float(np.mean(values)) if len(values) else float("nan"),
        "pairs": int(len(values)),
    }


def run_pilot(args: argparse.Namespace) -> None:
    """Train short native/component-balanced/gradient-balanced pilots."""
    OUT.mkdir(parents=True, exist_ok=True)
    frame = load_rows()
    train_pairs = make_pairs(frame, "train", args.pair_count, SEED)
    val_pairs = make_pairs(frame, "validation", max(64, args.pair_count // 4), SEED + 1)
    if not train_pairs or not val_pairs:
        raise RuntimeError("Pilot requires train and validation pairs.")
    cache = ShardCache()
    write_pair_manifest(train_pairs, OUT / "train_pairs.csv")
    write_pair_manifest(val_pairs, OUT / "validation_pairs.csv")
    device = args.device
    reference = make_model(device)
    reference.model.eval()
    for parameter in reference.model.parameters():
        parameter.requires_grad_(False)
    audit_path = OUT / "gradient_audit.csv"
    gradient_v_weight = 1.0
    if audit_path.exists():
        audit = pd.read_csv(audit_path)
        diagnostic = audit[audit["mode"] == "x_v_diagnostics"]["equal_gradient_v_weight"].dropna()
        if len(diagnostic):
            gradient_v_weight = float(np.clip(np.median(diagnostic), 1e-3, 1e4))
    variants = {
        "native_joint": (3.0 / 13.0, 10.0 / 13.0),
        "component_balanced": (0.5, 0.5),
        "gradient_balanced": (1.0, gradient_v_weight),
    }
    metrics: list[dict[str, object]] = []
    for variant, weights in variants.items():
        current = make_model(device)
        optimizer = torch.optim.Adam(current.model.parameters(), lr=1e-6)
        for epoch in range(args.epochs):
            current.model.train()
            rng = random.Random(SEED + epoch)
            shuffled = list(train_pairs)
            rng.shuffle(shuffled)
            epoch_losses: list[float] = []
            for batch_start in range(0, len(shuffled), args.batch_pairs):
                batch_pairs = shuffled[batch_start : batch_start + args.batch_pairs]
                items = flatten_pairs(batch_pairs, cache)
                timestep = rng.randint(1, T)
                x_cur, v_cur = component_losses(current, items, timestep, SEED + epoch + batch_start)
                with torch.no_grad():
                    x_ref, v_ref = component_losses(reference, items, timestep, SEED + epoch + batch_start)
                cur_score = -(weights[0] * x_cur + weights[1] * v_cur)
                ref_score = -(weights[0] * x_ref + weights[1] * v_ref)
                cur_margin = cur_score[0::2] - cur_score[1::2]
                ref_margin = ref_score[0::2] - ref_score[1::2]
                loss = -F.logsigmoid(float(args.beta) * (cur_margin - ref_margin)).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(current.model.parameters(), 1.0)
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()))
            validation = evaluate_pairs(
                current,
                val_pairs,
                cache,
                weights,
                SEED + epoch * 1000,
                args.batch_pairs,
            )
            metrics.append(
                {
                    "variant": variant,
                    "epoch": epoch,
                    "train_dpo_loss": float(np.mean(epoch_losses)),
                    "gradient_v_weight": gradient_v_weight,
                    **validation,
                }
            )
        torch.save(
            {
                "state_dict": current.model.state_dict(),
                "variant": variant,
                "weights": weights,
                "T": T,
                "checkpoint": str(CHECKPOINT),
            },
            OUT / f"pilot_{variant}.ckpt",
        )
        del optimizer, current
        gc.collect()
        torch.cuda.empty_cache()
    pd.DataFrame(metrics).to_csv(OUT / "pilot_metrics.csv", index=False)
    summary = {
        "T": T,
        "checkpoint": str(CHECKPOINT),
        "train_pair_count": len(train_pairs),
        "validation_pair_count": len(val_pairs),
        "epochs": args.epochs,
        "gradient_v_weight_used": gradient_v_weight,
        "variants": variants,
    }
    (OUT / "pilot_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    metric_frame = pd.DataFrame(metrics)
    for variant, group in metric_frame.groupby("variant"):
        ax.plot(group["epoch"], group["pair_accuracy"], marker="o", linewidth=2.5, label=variant)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation pair accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "pilot_pair_accuracy.png", dpi=180)
    fig.savefig(OUT / "pilot_pair_accuracy.svg")
    plt.close(fig)


def main() -> None:
    """Run the selected experiment."""
    args = parse_args()
    global OUT
    OUT = args.output_dir.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.mode == "gradient":
        run_gradient_audit(args)
    else:
        run_pilot(args)


if __name__ == "__main__":
    main()
