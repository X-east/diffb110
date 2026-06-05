from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np


SRC_DIR = Path("data/crossdocked_raw/processed_edge_noH_full")
OUT_DIR = Path("analysis/diffsbdd_xhe_supervised_20260604_20k_6gpu/data")


def write_subset(src_path: Path, dst_path: Path, max_molecules: int | None) -> None:
    with np.load(src_path, allow_pickle=True) as data:
        n_total = len(data["names"])
        n = n_total if max_molecules is None else min(max_molecules, n_total)
        lig_sel = data["lig_mask"] < n
        pocket_sel = data["pocket_mask"] < n
        lig_one_hot = _drop_empty_others_column(data["lig_one_hot"][lig_sel])
        pocket_one_hot = _drop_empty_others_column(data["pocket_one_hot"][pocket_sel])
        payload = {
            "names": data["names"][:n],
            "lig_coords": data["lig_coords"][lig_sel],
            "lig_one_hot": lig_one_hot,
            "lig_mask": data["lig_mask"][lig_sel],
            "pocket_coords": data["pocket_coords"][pocket_sel],
            "pocket_one_hot": pocket_one_hot,
            "pocket_mask": data["pocket_mask"][pocket_sel],
        }
        if "receptors" in data.files:
            payload["receptors"] = data["receptors"][:n]
        if "lig_bond_index" in data.files and "lig_bond_mask" in data.files:
            bond_sel = data["lig_bond_mask"] < n
            payload["lig_bond_index"] = data["lig_bond_index"][:, bond_sel]
            payload["lig_bond_type"] = data["lig_bond_type"][bond_sel]
            payload["lig_bond_mask"] = data["lig_bond_mask"][bond_sel]
            payload["num_lig_bonds"] = data["num_lig_bonds"][:n]
        np.savez(dst_path, **payload)
        print(
            dst_path,
            "molecules=", n,
            "lig_atoms=", int(lig_sel.sum()),
            "pocket_nodes=", int(pocket_sel.sum()),
            "directed_bonds=", int(payload.get("lig_bond_type", np.array([])).shape[0]),
            "lig_one_hot_dim=", int(lig_one_hot.shape[1]),
            "pocket_one_hot_dim=", int(pocket_one_hot.shape[1]),
        )


def _drop_empty_others_column(one_hot: np.ndarray) -> np.ndarray:
    if one_hot.shape[1] == 11 and float(one_hot[:, 10].sum()) == 0.0:
        return one_hot[:, :10]
    return one_hot


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_subset(SRC_DIR / "train.npz", OUT_DIR / "train.npz", 20000)
    write_subset(SRC_DIR / "val.npz", OUT_DIR / "val.npz", None)
    shutil.copy2(SRC_DIR / "size_distribution.npy", OUT_DIR / "size_distribution.npy")
    print("wrote", OUT_DIR)


if __name__ == "__main__":
    main()
