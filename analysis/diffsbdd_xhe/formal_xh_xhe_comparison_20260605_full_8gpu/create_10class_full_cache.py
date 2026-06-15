"""Create a 10-class CrossDocked edge cache compatible with DiffSBDD checkpoints."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np


SOURCE_DIR = Path("data/crossdocked_raw/processed_edge_noH_full")
TARGET_DIR = Path("data/crossdocked_raw/processed_edge_noH_10class_full")
SPLITS = ("train", "val", "test")


def _convert_split(split: str) -> None:
    """Drop the all-zero ``others`` one-hot column for one split."""
    source_path = SOURCE_DIR / f"{split}.npz"
    target_path = TARGET_DIR / f"{split}.npz"
    data = np.load(source_path, allow_pickle=True)
    lig_one_hot = data["lig_one_hot"]
    pocket_one_hot = data["pocket_one_hot"]
    if lig_one_hot.shape[1] != 11 or pocket_one_hot.shape[1] != 11:
        raise ValueError(f"{split} is not 11-class data")
    if np.any(lig_one_hot[:, -1]) or np.any(pocket_one_hot[:, -1]):
        raise ValueError(f"{split} has non-zero values in the others column")

    arrays = {key: data[key] for key in data.files}
    arrays["lig_one_hot"] = lig_one_hot[:, :10].copy()
    arrays["pocket_one_hot"] = pocket_one_hot[:, :10].copy()
    np.savez(target_path, **arrays)
    print(
        f"converted {split}: "
        f"lig_one_hot {lig_one_hot.shape} -> {arrays['lig_one_hot'].shape}, "
        f"pocket_one_hot {pocket_one_hot.shape} -> {arrays['pocket_one_hot'].shape}"
    )


def main() -> None:
    """Build the target cache directory."""
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        _convert_split(split)

    for filename in ("size_distribution.npy", "train_smiles.npy"):
        shutil.copyfile(SOURCE_DIR / filename, TARGET_DIR / filename)

    summary = (
        "# processed_edge_noH_10class_full\n\n"
        "This cache is derived from processed_edge_noH_full by dropping the "
        "all-zero trailing 'others' one-hot column from ligand and pocket atom "
        "features. It is intended for warm-starting from the original "
        "10-class DiffSBDD crossdocked_fullatom_cond checkpoint.\n"
    )
    (TARGET_DIR / "summary.txt").write_text(summary, encoding="utf-8")
    print(f"wrote {TARGET_DIR}")


if __name__ == "__main__":
    main()
