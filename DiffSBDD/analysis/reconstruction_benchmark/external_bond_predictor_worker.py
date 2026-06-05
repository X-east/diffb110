"""Run external pretrained bond predictors and emit halfedge predictions."""

from __future__ import annotations

import argparse
import pickle
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


SYMBOL_TO_ATOMIC_NUMBER = {
    "H": 1,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Br": 35,
    "I": 53,
}

DIFFGUI_NONAROMATIC_NODE_TYPE = {
    "H": 0,
    "B": 1,
    "C": 2,
    "N": 4,
    "O": 6,
    "F": 8,
    "P": 9,
    "S": 11,
    "Cl": 13,
    "Br": 14,
    "I": 15,
}


class EasyDict(dict):
    """Small easydict-compatible fallback for checkpoint unpickling."""

    def __getattr__(self, key: str) -> Any:
        """Return dictionary items through attribute access."""
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        """Assign dictionary items through attribute access."""
        self[key] = value


def _install_easydict_fallback() -> None:
    """Install a minimal easydict module when the external env lacks it."""
    if "easydict" in sys.modules:
        return
    module = types.ModuleType("easydict")
    module.EasyDict = EasyDict
    sys.modules["easydict"] = module


def _prepare_external_imports(provider_root: Path) -> None:
    """Prioritize one external project's top-level models/utils packages."""
    for name in list(sys.modules):
        if name == "models" or name.startswith("models.") or name == "utils" or name.startswith("utils."):
            del sys.modules[name]
    root_str = str(provider_root)
    if root_str in sys.path:
        sys.path.remove(root_str)
    sys.path.insert(0, root_str)


def _load_checkpoint(checkpoint: Path, device: torch.device) -> dict[str, Any]:
    """Load a checkpoint that may contain easydict config objects."""
    _install_easydict_fallback()
    return torch.load(str(checkpoint), map_location=device)


def _all_halfedges(num_atoms: int, device: torch.device) -> torch.Tensor:
    """Return upper-triangular halfedge indices."""
    return torch.triu_indices(num_atoms, num_atoms, offset=1, device=device)


def _prediction_to_numpy(pred_halfedge: torch.Tensor, halfedge_index: torch.Tensor) -> dict[str, np.ndarray]:
    """Convert predictor logits to filtered bond arrays."""
    edge_type = pred_halfedge.softmax(dim=-1).argmax(dim=-1)
    is_bond = (edge_type > 0) & (edge_type <= 4)
    return {
        "bond_index": halfedge_index[:, is_bond].detach().cpu().numpy().astype(np.int64),
        "bond_type": edge_type[is_bond].detach().cpu().numpy().astype(np.int64),
    }


def _load_molsnapper_model(payload: dict[str, Any], device: torch.device) -> torch.nn.Module:
    """Load the MolSnapper ligand-only bond predictor."""
    provider_root = Path(payload["provider_root"])
    _prepare_external_imports(provider_root)
    from models.bond_predictor import BondPredictor  # type: ignore[import-not-found]

    checkpoint = _load_checkpoint(Path(payload["checkpoint"]), device)
    model = BondPredictor(
        checkpoint["config"].model,
        num_node_types=8,
        num_edge_types=5,
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model


def _predict_molsnapper(
    model: torch.nn.Module,
    sample: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Predict ligand halfedge types with MolSnapper."""
    supported_atomic_numbers = [6, 7, 8, 9, 15, 16, 17]
    atomic_to_node = {atomic_number: idx for idx, atomic_number in enumerate(supported_atomic_numbers)}
    atom_symbols = sample["atom_symbols"]
    node_indices = []
    unsupported_symbols = []
    for symbol in atom_symbols:
        atomic_number = SYMBOL_TO_ATOMIC_NUMBER.get(symbol)
        if atomic_number not in atomic_to_node:
            unsupported_symbols.append(symbol)
        else:
            node_indices.append(atomic_to_node[atomic_number])
    if unsupported_symbols:
        return {
            "sample_idx": sample["sample_idx"],
            "error": "UnsupportedAtoms:" + ",".join(sorted(set(unsupported_symbols))),
            "seconds": 0.0,
        }

    started_at = time.perf_counter()
    coords = torch.as_tensor(sample["coords"], dtype=torch.float32, device=device)
    coords = coords - coords.mean(dim=0, keepdim=True)
    node_type = torch.as_tensor(node_indices, dtype=torch.long, device=device)
    h_node = F.one_hot(node_type, num_classes=8).float()
    batch_node = torch.zeros(len(node_indices), dtype=torch.long, device=device)
    halfedge_index = _all_halfedges(len(node_indices), device)
    edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=1)
    batch_edge = torch.zeros(edge_index.shape[1], dtype=torch.long, device=device)
    timestep = torch.zeros(1, dtype=torch.long, device=device)
    with torch.no_grad():
        pred_halfedge = model(h_node, coords, batch_node, edge_index, batch_edge, timestep)
    result = _prediction_to_numpy(pred_halfedge, halfedge_index)
    return {
        "sample_idx": sample["sample_idx"],
        "error": "",
        "seconds": time.perf_counter() - started_at,
        **result,
    }


def _load_diffgui_model(payload: dict[str, Any], device: torch.device) -> torch.nn.Module:
    """Load the DiffGui protein-ligand bond predictor."""
    provider_root = Path(payload["provider_root"])
    _prepare_external_imports(provider_root)
    from models.bond_predictor import BondPredictor  # type: ignore[import-not-found]

    checkpoint = _load_checkpoint(Path(payload["checkpoint"]), device)
    model = BondPredictor(
        config=checkpoint["config"].model,
        protein_node_types=28,
        ligand_node_types=16,
        num_edge_types=5,
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model


def _predict_diffgui(
    model: torch.nn.Module,
    sample: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Predict ligand halfedge types with DiffGui."""
    atom_symbols = sample["atom_symbols"]
    node_indices = []
    unsupported_symbols = []
    for symbol in atom_symbols:
        if symbol not in DIFFGUI_NONAROMATIC_NODE_TYPE:
            unsupported_symbols.append(symbol)
        else:
            node_indices.append(DIFFGUI_NONAROMATIC_NODE_TYPE[symbol])
    if unsupported_symbols:
        return {
            "sample_idx": sample["sample_idx"],
            "error": "UnsupportedAtoms:" + ",".join(sorted(set(unsupported_symbols))),
            "seconds": 0.0,
        }

    started_at = time.perf_counter()
    ligand_pos = torch.as_tensor(sample["coords"], dtype=torch.float32, device=device)
    protein_pos = torch.as_tensor(sample.get("protein_pos", np.zeros((0, 3))), dtype=torch.float32, device=device)
    protein_node = torch.as_tensor(sample.get("protein_node", np.zeros((0, 28))), dtype=torch.float32, device=device)
    if len(protein_pos) > 0:
        offset = protein_pos.mean(dim=0, keepdim=True)
    else:
        offset = ligand_pos.mean(dim=0, keepdim=True)
    ligand_pos = ligand_pos - offset
    protein_pos = protein_pos - offset

    ligand_node = F.one_hot(
        torch.as_tensor(node_indices, dtype=torch.long, device=device),
        num_classes=16,
    ).float()
    protein_batch = torch.zeros(len(protein_pos), dtype=torch.long, device=device)
    ligand_batch = torch.zeros(len(ligand_pos), dtype=torch.long, device=device)
    halfedge_index = _all_halfedges(len(ligand_pos), device)
    ligand_edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=1)
    ligand_edge_batch = torch.zeros(ligand_edge_index.shape[1], dtype=torch.long, device=device)
    timestep = torch.zeros(1, dtype=torch.long, device=device)
    with torch.no_grad():
        pred_halfedge = model(
            protein_node,
            protein_pos,
            protein_batch,
            ligand_node,
            ligand_pos,
            ligand_batch,
            ligand_edge_index,
            ligand_edge_batch,
            timestep,
        )
    result = _prediction_to_numpy(pred_halfedge, halfedge_index)
    return {
        "sample_idx": sample["sample_idx"],
        "error": "",
        "seconds": time.perf_counter() - started_at,
        **result,
    }


def run_worker(provider: str, input_path: Path, output_path: Path) -> None:
    """Run one provider over all serialized samples."""
    with input_path.open("rb") as handle:
        payload = pickle.load(handle)
    requested_device = str(payload.get("device") or "cpu")
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)

    if provider == "molsnapper":
        model = _load_molsnapper_model(payload, device)
        predictor = _predict_molsnapper
    elif provider == "diffgui":
        model = _load_diffgui_model(payload, device)
        predictor = _predict_diffgui
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    results = []
    for sample in payload["samples"]:
        try:
            results.append(predictor(model, sample, device))
        except Exception as exc:  # noqa: BLE001 - worker reports per-sample failures.
            results.append({
                "sample_idx": sample["sample_idx"],
                "error": type(exc).__name__,
                "seconds": 0.0,
            })

    with output_path.open("wb") as handle:
        pickle.dump(results, handle)


def parse_args() -> argparse.Namespace:
    """Parse worker command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["diffgui", "molsnapper"], required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_worker(args.provider, args.input, args.output)
