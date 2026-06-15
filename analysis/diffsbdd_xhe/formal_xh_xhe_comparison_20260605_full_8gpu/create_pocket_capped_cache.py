"""Create a pocket-size capped CrossDocked edge cache.

The full-atom CrossDocked pocket graph has rare very large pockets. Those
outliers can dominate EGNN edge activations even when the ligand itself is
small. This utility keeps the data format unchanged while removing samples with
too many pocket nodes, then reindexes molecule, pocket, and halfedge masks.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scipy.ndimage import gaussian_filter
except ImportError:  # pragma: no cover - scipy is expected in the project env.
    gaussian_filter = None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - plots are helpful but not required.
    plt = None


SPLITS = ("train", "val", "test")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/crossdocked_raw/processed_edge_noH_10class_full"),
        help="Input processed edge cache directory.",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=Path("data/crossdocked_raw/processed_edge_noH_10class_full_pocket1000"),
        help="Output capped processed edge cache directory.",
    )
    parser.add_argument(
        "--max-pocket-nodes",
        type=int,
        default=1000,
        help="Keep samples with pocket node count <= this value.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing split files in the target directory.",
    )
    return parser.parse_args()


def _counts(mask: np.ndarray, n_samples: int) -> np.ndarray:
    """Count nodes or halfedges per sample id."""
    sample_ids = np.asarray(mask, dtype=np.int64)
    return np.bincount(sample_ids, minlength=n_samples)


def _describe(values: np.ndarray) -> dict[str, float | int]:
    """Return compact distribution statistics."""
    if len(values) == 0:
        return {
            "n": 0,
            "min": 0,
            "p50": 0,
            "p90": 0,
            "p95": 0,
            "p99": 0,
            "p999": 0,
            "max": 0,
            "mean": 0.0,
        }
    return {
        "n": int(len(values)),
        "min": int(np.min(values)),
        "p50": int(np.percentile(values, 50)),
        "p90": int(np.percentile(values, 90)),
        "p95": int(np.percentile(values, 95)),
        "p99": int(np.percentile(values, 99)),
        "p999": int(np.percentile(values, 99.9)),
        "max": int(np.max(values)),
        "mean": float(np.mean(values)),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write a CSV file with a stable header."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _joint_size_distribution(
    lig_mask: np.ndarray,
    pocket_mask: np.ndarray,
    smooth_sigma: float = 1.0,
) -> np.ndarray:
    """Build the DiffSBDD ligand-pocket node-count histogram."""
    lig_ids, lig_counts = np.unique(lig_mask, return_counts=True)
    pocket_ids, pocket_counts = np.unique(pocket_mask, return_counts=True)
    if not np.array_equal(lig_ids, pocket_ids):
        raise ValueError("Ligand and pocket masks do not contain the same sample ids")

    histogram = np.zeros((int(np.max(lig_counts)) + 1, int(np.max(pocket_counts)) + 1))
    for n_lig, n_pocket in zip(lig_counts, pocket_counts):
        histogram[int(n_lig), int(n_pocket)] += 1

    if smooth_sigma is not None and gaussian_filter is not None:
        histogram = gaussian_filter(
            histogram,
            sigma=smooth_sigma,
            order=0,
            mode="constant",
            cval=0.0,
            truncate=4.0,
        )
    return histogram


def _filter_split(
    source_path: Path,
    target_path: Path,
    report_dir: Path,
    split: str,
    max_pocket_nodes: int,
    overwrite: bool,
) -> dict[str, Any]:
    """Filter one split and save the reindexed NPZ arrays."""
    if target_path.exists() and not overwrite:
        raise FileExistsError(f"{target_path} already exists; pass --overwrite")

    data = np.load(source_path, allow_pickle=True)
    names = data["names"]
    n_samples = len(names)

    lig_mask_old = np.asarray(data["lig_mask"], dtype=np.int64)
    pocket_mask_old = np.asarray(data["pocket_mask"], dtype=np.int64)
    bond_mask_old = np.asarray(data["lig_bond_mask"], dtype=np.int64)

    lig_counts = _counts(lig_mask_old, n_samples)
    pocket_counts = _counts(pocket_mask_old, n_samples)
    halfedge_counts = _counts(bond_mask_old, n_samples)
    keep_sample = pocket_counts <= max_pocket_nodes
    keep_indices = np.flatnonzero(keep_sample)
    removed_indices = np.flatnonzero(~keep_sample)

    old_to_new_sample = np.full(n_samples, -1, dtype=np.int64)
    old_to_new_sample[keep_indices] = np.arange(len(keep_indices), dtype=np.int64)

    lig_node_keep = keep_sample[lig_mask_old]
    pocket_node_keep = keep_sample[pocket_mask_old]
    halfedge_keep = keep_sample[bond_mask_old]

    old_to_new_lig_node = np.full(len(lig_mask_old), -1, dtype=np.int64)
    old_to_new_lig_node[np.flatnonzero(lig_node_keep)] = np.arange(
        int(np.sum(lig_node_keep)),
        dtype=np.int64,
    )

    new_lig_bond_index = old_to_new_lig_node[data["lig_bond_index"][:, halfedge_keep]]
    if np.any(new_lig_bond_index < 0):
        raise ValueError(f"{split} halfedge reindexing produced negative node ids")

    arrays = {
        "names": names[keep_sample],
        "lig_coords": data["lig_coords"][lig_node_keep],
        "lig_one_hot": data["lig_one_hot"][lig_node_keep],
        "lig_mask": old_to_new_sample[lig_mask_old[lig_node_keep]].astype(data["lig_mask"].dtype),
        "pocket_coords": data["pocket_coords"][pocket_node_keep],
        "pocket_one_hot": data["pocket_one_hot"][pocket_node_keep],
        "pocket_mask": old_to_new_sample[pocket_mask_old[pocket_node_keep]].astype(
            data["pocket_mask"].dtype
        ),
        "lig_bond_index": new_lig_bond_index,
        "lig_bond_type": data["lig_bond_type"][halfedge_keep],
        "lig_bond_mask": old_to_new_sample[bond_mask_old[halfedge_keep]].astype(
            data["lig_bond_mask"].dtype
        ),
        "num_lig_bonds": data["num_lig_bonds"][keep_sample],
    }
    np.savez(target_path, **arrays)

    removed_rows = [
        {
            "old_index": int(idx),
            "name": str(names[idx]),
            "ligand_atoms": int(lig_counts[idx]),
            "pocket_nodes": int(pocket_counts[idx]),
            "total_nodes": int(lig_counts[idx] + pocket_counts[idx]),
            "halfedges": int(halfedge_counts[idx]),
        }
        for idx in removed_indices
    ]
    removed_rows.sort(key=lambda row: row["pocket_nodes"], reverse=True)
    _write_csv(
        report_dir / f"removed_{split}.csv",
        removed_rows,
        ["old_index", "name", "ligand_atoms", "pocket_nodes", "total_nodes", "halfedges"],
    )

    stats = {
        "split": split,
        "source_samples": int(n_samples),
        "kept_samples": int(np.sum(keep_sample)),
        "removed_samples": int(np.sum(~keep_sample)),
        "removed_fraction": float(np.mean(~keep_sample)),
        "source_ligand_atoms": _describe(lig_counts),
        "source_pocket_nodes": _describe(pocket_counts),
        "source_total_nodes": _describe(lig_counts + pocket_counts),
        "kept_ligand_atoms": _describe(lig_counts[keep_sample]),
        "kept_pocket_nodes": _describe(pocket_counts[keep_sample]),
        "kept_total_nodes": _describe((lig_counts + pocket_counts)[keep_sample]),
        "removed_top": removed_rows[:20],
    }
    return stats


def _plot_train_distribution(
    report_dir: Path,
    train_stats: dict[str, Any],
    source_dir: Path,
    target_dir: Path,
) -> None:
    """Save simple before/after train pocket and total-node histograms."""
    if plt is None:
        return

    with np.load(source_dir / "train.npz", allow_pickle=True) as source:
        source_n = len(source["names"])
        source_lig = _counts(source["lig_mask"], source_n)
        source_pocket = _counts(source["pocket_mask"], source_n)
    with np.load(target_dir / "train.npz", allow_pickle=True) as target:
        target_n = len(target["names"])
        target_lig = _counts(target["lig_mask"], target_n)
        target_pocket = _counts(target["pocket_mask"], target_n)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(source_pocket, bins=80, alpha=0.55, label="source")
    axes[0].hist(target_pocket, bins=80, alpha=0.55, label="capped")
    axes[0].axvline(
        int(train_stats["kept_pocket_nodes"]["max"]),
        color="tab:red",
        linestyle="--",
        linewidth=1,
        label="kept max",
    )
    axes[0].set_xlabel("pocket nodes")
    axes[0].set_ylabel("train samples")
    axes[0].legend()

    axes[1].hist(source_lig + source_pocket, bins=80, alpha=0.55, label="source")
    axes[1].hist(target_lig + target_pocket, bins=80, alpha=0.55, label="capped")
    axes[1].set_xlabel("total nodes")
    axes[1].set_ylabel("train samples")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(report_dir / "train_node_distribution_before_after.png", dpi=180)
    plt.close(fig)


def main() -> None:
    """Create all capped splits and summary artifacts."""
    args = _parse_args()
    args.target_dir.mkdir(parents=True, exist_ok=True)
    report_dir = args.target_dir / "cap_report"
    report_dir.mkdir(parents=True, exist_ok=True)

    stats = []
    for split in SPLITS:
        split_stats = _filter_split(
            args.source_dir / f"{split}.npz",
            args.target_dir / f"{split}.npz",
            report_dir,
            split,
            args.max_pocket_nodes,
            args.overwrite,
        )
        stats.append(split_stats)
        print(
            f"{split}: kept {split_stats['kept_samples']}/"
            f"{split_stats['source_samples']} "
            f"(removed {split_stats['removed_samples']})"
        )

    with np.load(args.target_dir / "train.npz", allow_pickle=True) as train:
        size_distribution = _joint_size_distribution(train["lig_mask"], train["pocket_mask"])
    np.save(args.target_dir / "size_distribution.npy", size_distribution)

    smiles_path = args.source_dir / "train_smiles.npy"
    if smiles_path.exists():
        shutil.copy2(smiles_path, args.target_dir / "train_smiles.npy")

    summary_rows = []
    for item in stats:
        source_pocket = item["source_pocket_nodes"]
        kept_pocket = item["kept_pocket_nodes"]
        source_total = item["source_total_nodes"]
        kept_total = item["kept_total_nodes"]
        summary_rows.append(
            {
                "split": item["split"],
                "source_samples": item["source_samples"],
                "kept_samples": item["kept_samples"],
                "removed_samples": item["removed_samples"],
                "removed_fraction": item["removed_fraction"],
                "source_pocket_p99": source_pocket["p99"],
                "source_pocket_max": source_pocket["max"],
                "kept_pocket_p99": kept_pocket["p99"],
                "kept_pocket_max": kept_pocket["max"],
                "source_total_p99": source_total["p99"],
                "source_total_max": source_total["max"],
                "kept_total_p99": kept_total["p99"],
                "kept_total_max": kept_total["max"],
            }
        )
    _write_csv(
        report_dir / "split_summary.csv",
        summary_rows,
        [
            "split",
            "source_samples",
            "kept_samples",
            "removed_samples",
            "removed_fraction",
            "source_pocket_p99",
            "source_pocket_max",
            "kept_pocket_p99",
            "kept_pocket_max",
            "source_total_p99",
            "source_total_max",
            "kept_total_p99",
            "kept_total_max",
        ],
    )

    report = {
        "source_dir": str(args.source_dir),
        "target_dir": str(args.target_dir),
        "max_pocket_nodes": args.max_pocket_nodes,
        "splits": stats,
        "size_distribution_shape": list(size_distribution.shape),
        "train_smiles_policy": "copied unchanged; original train_smiles.npy is an evaluation reference list, not sample-aligned",
    }
    (report_dir / "cap_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.target_dir / "summary.txt").write_text(
        "\n".join(
            [
                f"# {args.target_dir.name}",
                "",
                f"Source: {args.source_dir}",
                f"Max pocket nodes: {args.max_pocket_nodes}",
                "Policy: keep samples with pocket_nodes <= max_pocket_nodes.",
                "Masks and ligand halfedge indices are reindexed after filtering.",
                "size_distribution.npy is recomputed from the capped train split.",
                "train_smiles.npy is copied unchanged because it is an evaluation reference list.",
                "Detailed report: cap_report/cap_report.json",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _plot_train_distribution(report_dir, stats[0], args.source_dir, args.target_dir)
    print(f"wrote capped cache: {args.target_dir}")


if __name__ == "__main__":
    main()
