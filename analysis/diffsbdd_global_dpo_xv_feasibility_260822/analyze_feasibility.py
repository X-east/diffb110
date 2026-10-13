"""Quantify DiffSBDD x/v Global-DPO feasibility on the frozen extreme pool."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch_scatter import scatter_add

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ddpo.integration.diffsbdd_model import DiffSBDDModel


MANIFEST = ROOT / "analysis/offline_iql_extreme_good_bad_pool_260822/inputs/selected_extremes_manifest.csv.gz"
CHECKPOINT = ROOT / "DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt"
POCKET = ROOT / "data/8GJL.pdb"
REF_LIGAND = "C:1101"
TIMESTEPS = (1, 50, 125, 250, 375, 500)
SAMPLES_PER_CLASS = 128
BATCH_SIZE = 6
SEED = 1


def _load_endpoint_rows(rows: pd.DataFrame) -> list[dict[str, object]]:
    """Load selected terminal x/v tensors from native trajectory shards."""
    outputs: list[dict[str, object]] = []
    for shard_path, group in rows.groupby("trajectory_shard", sort=True):
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        for row in group.itertuples(index=False):
            index = int(row.mol_index)
            n_nodes = int(shard["nodesxsample"][index])
            outputs.append(
                {
                    "sample_id": row.trajectory_id,
                    "label": row.extreme_label,
                    "n_nodes": n_nodes,
                    "x": shard["x"][index, :n_nodes].float(),
                    "v": shard["categorical"][index, :n_nodes].float(),
                }
            )
    return outputs


def _pack_batch(items: list[dict[str, object]], device: torch.device) -> dict[str, torch.Tensor]:
    """Pack variable-sized terminal molecules into DiffSBDD flat format."""
    sizes = torch.tensor([int(item["n_nodes"]) for item in items], device=device)
    mask = torch.repeat_interleave(
        torch.arange(len(items), device=device), sizes
    ).long()
    return {
        "x": torch.cat([item["x"] for item in items], dim=0).to(device),
        "one_hot": torch.cat([item["v"] for item in items], dim=0).to(device),
        "size": sizes.long(),
        "mask": mask,
    }


@torch.no_grad()
def _component_losses(
    model: DiffSBDDModel,
    items: list[dict[str, object]],
    timestep: int,
    generator_seed: int,
) -> pd.DataFrame:
    """Measure original-model coordinate and atom-channel epsilon MSE."""
    device = torch.device(model.device)
    ligand = _pack_batch(items, device)
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
        float(timestep) / float(ddpm.T),
        device=device,
        dtype=xh0_lig.dtype,
    )
    gamma_t = ddpm.inflate_batch_array(ddpm.gamma(t), ligand["x"])
    alpha_t = ddpm.alpha(gamma_t, xh0_lig)
    sigma_t = ddpm.sigma(gamma_t, xh0_lig)
    torch.manual_seed(generator_seed)
    torch.cuda.manual_seed_all(generator_seed)
    eps = ddpm.sample_gaussian(xh0_lig.shape, device=device)
    z_t_lig = alpha_t[ligand["mask"]] * xh0_lig + sigma_t[ligand["mask"]] * eps
    z_t_lig[:, : ddpm.n_dims], xh0_pocket[:, : ddpm.n_dims] = ddpm.remove_mean_batch(
        z_t_lig[:, : ddpm.n_dims],
        xh0_pocket[:, : ddpm.n_dims],
        ligand["mask"],
        pocket["mask"],
    )
    prediction, _ = ddpm.dynamics(
        z_t_lig, xh0_pocket, t, ligand["mask"], pocket["mask"]
    )
    squared = (eps - prediction).square()
    x_sum = scatter_add(squared[:, : ddpm.n_dims].sum(-1), ligand["mask"], dim=0)
    v_sum = scatter_add(squared[:, ddpm.n_dims :].sum(-1), ligand["mask"], dim=0)
    n = ligand["size"].to(dtype=squared.dtype)
    x_mse = x_sum / (float(ddpm.n_dims) * n)
    v_mse = v_sum / (float(ddpm.atom_nf) * n)
    total = (x_sum + v_sum) / ((float(ddpm.n_dims + ddpm.atom_nf)) * n)
    x_fraction = x_sum / (x_sum + v_sum).clamp_min(1e-12)
    rows = []
    for index, item in enumerate(items):
        rows.append(
            {
                "sample_id": item["sample_id"],
                "label": item["label"],
                "n_nodes": item["n_nodes"],
                "timestep": timestep,
                "x_mse_per_dimension": float(x_mse[index].cpu()),
                "v_mse_per_dimension": float(v_mse[index].cpu()),
                "combined_mse_per_dimension": float(total[index].cpu()),
                "x_fraction_of_raw_squared_error": float(x_fraction[index].cpu()),
            }
        )
    return pd.DataFrame(rows)


def _descriptor_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute class separation for explicit molecular descriptor shortcuts."""
    columns = [
        "heavy_atom_count",
        "carbon_fraction",
        "hetero_atom_count",
        "mol_wt",
        "logp",
        "tpsa",
        "total_ring_count",
        "aromatic_ring_count",
        "three_member_ring_count",
        "fused_ring_count",
        "complex_ring_count",
        "brenk_count",
        "qed",
        "sa_raw",
        "umap_nearest_chembl_distance",
    ]
    target = (frame["extreme_label"] == "hard_bad").astype(int).to_numpy()
    rows = []
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        valid = values.notna().to_numpy()
        good = values[frame["extreme_label"] == "good"].dropna().to_numpy()
        bad = values[frame["extreme_label"] == "hard_bad"].dropna().to_numpy()
        pooled = math.sqrt((np.var(good, ddof=1) + np.var(bad, ddof=1)) / 2.0)
        standardized = (np.mean(bad) - np.mean(good)) / pooled if pooled > 0 else 0.0
        auc = roc_auc_score(target[valid], values.to_numpy()[valid])
        rows.append(
            {
                "descriptor": column,
                "good_mean": float(np.mean(good)),
                "hard_bad_mean": float(np.mean(bad)),
                "standardized_bad_minus_good": float(standardized),
                "univariate_auc": float(max(auc, 1.0 - auc)),
                "auc_direction": "higher_is_bad" if auc >= 0.5 else "lower_is_bad",
            }
        )
    return pd.DataFrame(rows).sort_values("univariate_auc", ascending=False)


def _plot_descriptor_shortcuts(frame: pd.DataFrame) -> None:
    """Plot strongest univariate class shortcuts."""
    shown = frame[frame["descriptor"] != "umap_nearest_chembl_distance"].head(10)
    shown = shown.sort_values("univariate_auc")
    fig, ax = plt.subplots(figsize=(12, 7.5))
    colors = ["#d95f02" if value >= 0.7 else "#1b9e77" for value in shown["univariate_auc"]]
    ax.barh(shown["descriptor"], shown["univariate_auc"], color=colors)
    ax.axvline(0.5, color="black", linewidth=1.5, linestyle="--", label="random")
    ax.axvline(0.7, color="#d95f02", linewidth=1.5, linestyle=":", label="AUC 0.70")
    ax.set_xlim(0.45, 1.0)
    ax.set_xlabel("Univariate AUC for good vs hard-bad", fontsize=16)
    ax.set_title("Learnable Endpoint Label Shortcuts (Direct Distance Excluded)", fontsize=19)
    ax.tick_params(axis="both", labelsize=13)
    ax.legend(fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "descriptor_shortcut_auc.png", dpi=180)
    fig.savefig(OUT / "descriptor_shortcut_auc.svg")
    plt.close(fig)


def _plot_loss_components(frame: pd.DataFrame) -> None:
    """Plot x/v denoising errors by timestep and class."""
    summary = (
        frame.groupby(["timestep", "label"])[["x_mse_per_dimension", "v_mse_per_dimension"]]
        .mean()
        .reset_index()
    )
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharex=True)
    for label, color in (("good", "#1b9e77"), ("hard_bad", "#d95f02")):
        sub = summary[summary["label"] == label]
        axes[0].plot(sub["timestep"], sub["x_mse_per_dimension"], marker="o", linewidth=2.5, color=color, label=label)
        axes[1].plot(sub["timestep"], sub["v_mse_per_dimension"], marker="o", linewidth=2.5, color=color, label=label)
    axes[0].set_title("Coordinate x channels", fontsize=18)
    axes[1].set_title("Atom-type v channels", fontsize=18)
    for ax in axes:
        ax.set_xlabel("Checkpoint-native timestep t / 500", fontsize=15)
        ax.set_ylabel("Original-model epsilon MSE / dimension", fontsize=15)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=13)
    fig.suptitle("Native DiffSBDD Denoising Error on Extreme Endpoints", fontsize=19)
    fig.tight_layout()
    fig.savefig(OUT / "native_xv_denoising_error_by_timestep.png", dpi=180)
    fig.savefig(OUT / "native_xv_denoising_error_by_timestep.svg")
    plt.close(fig)


def _loss_separability(frame: pd.DataFrame) -> pd.DataFrame:
    """Measure whether the frozen model's endpoint losses already rank labels."""
    rows = []
    for timestep, group in frame.groupby("timestep"):
        target = (group["label"] == "hard_bad").astype(int).to_numpy()
        for component in (
            "x_mse_per_dimension",
            "v_mse_per_dimension",
            "combined_mse_per_dimension",
        ):
            values = group[component].to_numpy()
            auc = roc_auc_score(target, values)
            rows.append(
                {
                    "timestep": int(timestep),
                    "component": component,
                    "signed_auc_higher_loss_is_bad": float(auc),
                    "direction_free_auc": float(max(auc, 1.0 - auc)),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    """Run the feasibility diagnostics and write durable artifacts."""
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(MANIFEST)
    descriptor = _descriptor_diagnostics(frame)
    descriptor.to_csv(OUT / "descriptor_shortcut_diagnostics.csv", index=False)
    _plot_descriptor_shortcuts(descriptor)

    sampled = (
        frame.groupby("extreme_label", group_keys=False)
        .sample(n=SAMPLES_PER_CLASS, random_state=SEED)
        .sort_values(["trajectory_shard", "mol_index"])
    )
    sampled[["trajectory_id", "trajectory_shard", "mol_index", "extreme_label"]].to_csv(
        OUT / "denoising_probe_samples.csv", index=False
    )
    items = _load_endpoint_rows(sampled)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DiffSBDDModel(
        str(CHECKPOINT), str(POCKET), REF_LIGAND, device=device
    )
    model.eval()
    all_losses = []
    for timestep in TIMESTEPS:
        for start in range(0, len(items), BATCH_SIZE):
            batch = items[start : start + BATCH_SIZE]
            all_losses.append(
                _component_losses(
                    model,
                    batch,
                    timestep,
                    generator_seed=SEED + timestep * 1000 + start,
                )
            )
    losses = pd.concat(all_losses, ignore_index=True)
    losses.to_csv(OUT / "native_xv_denoising_probe.csv", index=False)
    loss_summary = (
        losses.groupby(["timestep", "label"])
        .agg(
            samples=("sample_id", "size"),
            x_mse=("x_mse_per_dimension", "mean"),
            v_mse=("v_mse_per_dimension", "mean"),
            combined_mse=("combined_mse_per_dimension", "mean"),
            x_raw_error_fraction=("x_fraction_of_raw_squared_error", "mean"),
        )
        .reset_index()
    )
    loss_summary.to_csv(OUT / "native_xv_denoising_summary.csv", index=False)
    separability = _loss_separability(losses)
    separability.to_csv(OUT / "native_xv_loss_separability.csv", index=False)
    _plot_loss_components(losses)

    summary = {
        "checkpoint": str(CHECKPOINT),
        "checkpoint_native_timesteps": int(model.T),
        "x_dimensions": int(model.n_dims),
        "atom_type_dimensions": int(model.atom_nf),
        "normalization_factors": list(model.ddpm.norm_values),
        "device": device,
        "probe_samples_per_class": SAMPLES_PER_CLASS,
        "timesteps": list(TIMESTEPS),
        "strongest_nonlabel_shortcuts": descriptor[
            descriptor["descriptor"] != "umap_nearest_chembl_distance"
        ].head(5).to_dict(orient="records"),
        "loss_summary": loss_summary.to_dict(orient="records"),
        "loss_separability": separability.to_dict(orient="records"),
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
