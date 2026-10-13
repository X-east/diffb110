"""Summarize the 8GJL DePPA single-GPU smoke artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from rdkit import Chem


RUN_DIR = Path(__file__).resolve().parent
CSV_PATH = RUN_DIR / "output/8GJL/raw.csv"
SDF_PATH = RUN_DIR / "output/8GJL/raw.sdf"
BASE_CKPT = RUN_DIR.parents[1] / "DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt"
SMOKE_CKPT = RUN_DIR / "output/8GJL/model.ckpt"


def main() -> None:
    """Write JSON/CSV summaries and the per-molecule metric plot."""
    with CSV_PATH.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    mols = [mol for mol in Chem.SDMolSupplier(str(SDF_PATH), removeHs=False) if mol]
    smiles = [Chem.MolToSmiles(mol) for mol in mols]

    qed = np.asarray([float(row["qed"]) for row in rows])
    sa_reward = np.asarray([float(row["sa"]) for row in rows])
    qvina_reward = np.asarray([float(row["vina_score"]) for row in rows])
    diversity = np.asarray([float(row["distance"]) for row in rows])
    atom_count = np.asarray([int(row["num_atoms"]) for row in rows])

    base = torch.load(BASE_CKPT, map_location="cpu")
    smoke = torch.load(SMOKE_CKPT, map_location="cpu")
    changed = []
    for key, before in base["state_dict"].items():
        after = smoke["state_dict"][key]
        if torch.is_tensor(before) and before.shape == after.shape:
            delta = (after - before).abs()
            if delta.max().item() > 0:
                changed.append(delta.max().item())

    summary = {
        "attempts": len(rows),
        "written": len(mols),
        "sanitize_valid": len(mols),
        "connected": sum(float(row["connectivity"]) == 1.0 for row in rows),
        "unique_smiles": len(set(smiles)),
        "qed_mean": float(qed.mean()),
        "sa_reward_mean": float(sa_reward.mean()),
        "qvina_raw_mean": float((-qvina_reward).mean()),
        "qvina_raw_min": float((-qvina_reward).min()),
        "qvina_raw_max": float((-qvina_reward).max()),
        "diversity_mean": float(diversity.mean()),
        "atom_count_mean": float(atom_count.mean()),
        "changed_state_tensors": len(changed),
        "total_state_tensors": len(base["state_dict"]),
        "max_abs_weight_delta": float(max(changed) if changed else 0.0),
        "lightning_checkpoint_reloadable": False,
    }
    (RUN_DIR / "smoke_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (RUN_DIR / "smoke_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary.keys())
        writer.writeheader()
        writer.writerow(summary)

    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.2), constrained_layout=True)
    panels = [
        (qed, "QED", "#2a9d8f"),
        (sa_reward, "SA reward (higher is easier)", "#457b9d"),
        (-qvina_reward, "Raw QVina score (kcal/mol)", "#e76f51"),
        (diversity, "Fingerprint diversity", "#6d597a"),
    ]
    for ax, (values, title, color) in zip(axes.flat, panels):
        ax.bar(x, values, color=color)
        ax.set_xticks(x, [str(i + 1) for i in x])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("DePPA 8GJL single-GPU smoke: 4 generated molecules")
    fig.savefig(RUN_DIR / "smoke_molecule_metrics.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
