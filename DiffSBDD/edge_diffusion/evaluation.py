"""Evaluation helpers for standalone edge denoising experiments."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem

from DiffSBDD.edge_diffusion.dataset import EdgeBatchCollator, EdgeBundleDataset
from DiffSBDD.edge_diffusion.oracle_reconstruction import OracleReconstructionResult, build_oracle_mol
from DiffSBDD.edge_diffusion.semantic_oracle_reconstruction import build_semantic_oracle_mol
from DiffSBDD.edge_diffusion.transitions import EdgeCategoricalTransition
from DiffSBDD.edge_utils import BOND_TYPE_NAMES


@dataclass(frozen=True)
class EdgeReconstructionEvalConfig:
    """Configuration for predicted-bond reconstruction validation."""

    timestep: int = 500
    limit: int | None = 100
    seed: int = 100_000
    min_bond_length: float = 0.7
    max_bond_length: float = 2.1
    write_sdf: bool = False
    sdf_prefix: str = "predicted_bond"


def batch_confusion(target: torch.Tensor, pred: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Return target-by-predicted confusion matrix for one batch."""
    flat = target.long() * num_classes + pred.long()
    counts = torch.bincount(flat, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes)


def classification_metrics(
    confusion: torch.Tensor,
    eval_loss: float,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    """Compute overall and per-class precision, recall, and F1."""
    confusion = confusion.float()
    true_positive = torch.diag(confusion)
    support = confusion.sum(dim=1)
    predicted = confusion.sum(dim=0)
    precision = _safe_divide(true_positive, predicted)
    recall = _safe_divide(true_positive, support)
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)
    accuracy = float(true_positive.sum().item() / max(1.0, confusion.sum().item()))
    real_support = support[1:].sum()
    real_predicted = predicted[1:].sum()
    real_true_positive = true_positive[1:].sum()
    metrics = {
        "eval_loss": float(eval_loss),
        "accuracy": accuracy,
        "balanced_accuracy": float(recall.mean().item()),
        "macro_precision": float(precision.mean().item()),
        "macro_f1": float(f1.mean().item()),
        "real_bond_precision": _safe_scalar_divide(real_true_positive, real_predicted),
        "real_bond_recall": _safe_scalar_divide(real_true_positive, real_support),
        "real_bond_f1": _safe_scalar_divide(2.0 * real_true_positive, real_support + real_predicted),
    }
    class_rows = []
    for idx in range(confusion.shape[0]):
        class_rows.append({
            "class_id": idx,
            "class_name": BOND_TYPE_NAMES[idx],
            "support": int(support[idx].item()),
            "predicted": int(predicted[idx].item()),
            "true_positive": int(true_positive[idx].item()),
            "precision": float(precision[idx].item()),
            "recall": float(recall[idx].item()),
            "f1": float(f1[idx].item()),
        })
    return metrics, class_rows


@torch.no_grad()
def evaluate_edge_reconstruction(
    model: torch.nn.Module,
    dataset: EdgeBundleDataset,
    eval_indices: list[int],
    edge_npz: Path,
    semantic_npz: Path | None,
    transition: EdgeCategoricalTransition,
    config: EdgeReconstructionEvalConfig,
    device: torch.device,
    outdir: Path | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Predict bonds for selected molecules and reconstruct RDKit molecules."""
    edge_data = np.load(edge_npz, allow_pickle=True)
    semantic_data = None if semantic_npz is None else np.load(semantic_npz, allow_pickle=True)
    selected_indices = eval_indices if config.limit is None else eval_indices[:config.limit]
    generator = torch.Generator(device=device)
    generator.manual_seed(int(config.seed) + int(config.timestep))
    plain_writer = None
    semantic_writer = None
    if config.write_sdf:
        if outdir is None:
            raise ValueError("outdir is required when write_sdf=True")
        plain_writer = Chem.SDWriter(str(outdir / f"{config.sdf_prefix}_plain_t{config.timestep}.sdf"))
        semantic_writer = Chem.SDWriter(str(outdir / f"{config.sdf_prefix}_semantic_t{config.timestep}.sdf"))

    rows: list[dict[str, object]] = []
    strategy_counts: Counter[str] = Counter()
    was_training = bool(model.training)
    model.eval()
    try:
        for global_idx in selected_indices:
            sample = dataset[global_idx]
            batch = EdgeBatchCollator()([sample]).to(device)
            t = torch.tensor([int(config.timestep)], dtype=torch.long, device=device)
            noised_edge_type = transition.q_sample(
                batch.target_edge_type,
                t[batch.edge_batch],
                generator=generator,
            )
            logits = model(batch, noised_edge_type, t)
            pred_edge_type = logits.argmax(dim=-1).detach().cpu()
            target_edge_type = batch.target_edge_type.detach().cpu()
            halfedge_index = batch.halfedge_index.detach().cpu()
            positions = batch.ligand_pos.detach().cpu()
            atom_symbols = [str(value) for value in edge_data["ligand_atom_symbol"][global_idx]]
            plain_result = build_oracle_mol(
                atom_symbols=atom_symbols,
                positions=positions,
                halfedge_index=halfedge_index,
                halfedge_type=pred_edge_type,
            )
            semantic_result = build_semantic_result(
                semantic_data=semantic_data,
                global_idx=global_idx,
                atom_symbols=atom_symbols,
                positions=positions,
                halfedge_index=halfedge_index,
                halfedge_type=pred_edge_type,
            )
            row = build_reconstruction_row(
                global_idx=global_idx,
                name=str(sample["name"]),
                atom_symbols=atom_symbols,
                positions=positions,
                halfedge_index=halfedge_index,
                target_edge_type=target_edge_type,
                pred_edge_type=pred_edge_type,
                plain_result=plain_result,
                semantic_result=semantic_result,
                original_smiles=(
                    "" if semantic_data is None
                    else str(semantic_data["original_smiles_no_stereo"][global_idx])
                ),
                min_bond_length=config.min_bond_length,
                max_bond_length=config.max_bond_length,
            )
            rows.append(row)
            strategy_counts[f"plain:{plain_result.repair_strategy}"] += 1
            strategy_counts[f"semantic:{semantic_result.repair_strategy}"] += 1
            if plain_writer is not None:
                write_result_mol(plain_writer, plain_result.mol, row, prefix="plain")
            if semantic_writer is not None:
                write_result_mol(semantic_writer, semantic_result.mol, row, prefix="semantic")
    finally:
        if plain_writer is not None:
            plain_writer.close()
        if semantic_writer is not None:
            semantic_writer.close()
        if was_training:
            model.train()
        edge_data.close()
        if semantic_data is not None:
            semantic_data.close()
    return rows, summarize_reconstruction(rows, strategy_counts)


def build_semantic_result(
    semantic_data: np.lib.npyio.NpzFile | None,
    global_idx: int,
    atom_symbols: list[str],
    positions: torch.Tensor,
    halfedge_index: torch.Tensor,
    halfedge_type: torch.Tensor,
) -> OracleReconstructionResult:
    """Build semantic-aware reconstruction when sidecar metadata is available."""
    if semantic_data is None:
        return build_oracle_mol(atom_symbols, positions, halfedge_index, halfedge_type)
    return build_semantic_oracle_mol(
        atom_symbols=atom_symbols,
        positions=positions,
        halfedge_index=halfedge_index,
        halfedge_type=halfedge_type,
        formal_charges=semantic_data["original_formal_charge"][global_idx],
        total_hs=semantic_data["original_total_h"][global_idx],
        explicit_hs=semantic_data["original_explicit_h"][global_idx],
        no_implicit_flags=semantic_data["original_no_implicit"][global_idx],
        aromatic_atom_flags=semantic_data["original_aromatic_atom"][global_idx],
        chiral_tags=semantic_data["original_chiral_tag"][global_idx],
        bond_pairs=semantic_data["original_bond_pair"][global_idx],
        bond_stereo=semantic_data["original_bond_stereo"][global_idx],
    )


def build_reconstruction_row(
    global_idx: int,
    name: str,
    atom_symbols: list[str],
    positions: torch.Tensor,
    halfedge_index: torch.Tensor,
    target_edge_type: torch.Tensor,
    pred_edge_type: torch.Tensor,
    plain_result: OracleReconstructionResult,
    semantic_result: OracleReconstructionResult,
    original_smiles: str,
    min_bond_length: float,
    max_bond_length: float,
) -> dict[str, object]:
    """Build one reconstruction metrics row."""
    bond_compare = compare_halfedge_bond_maps(halfedge_index, target_edge_type, pred_edge_type)
    length_stats = predicted_bond_length_stats(
        atom_symbols=atom_symbols,
        positions=positions,
        halfedge_index=halfedge_index,
        pred_edge_type=pred_edge_type,
        min_bond_length=min_bond_length,
        max_bond_length=max_bond_length,
    )
    row = {
        "sample_idx": global_idx,
        "name": name,
        "num_atoms": len(atom_symbols),
        "num_halfedges": int(target_edge_type.numel()),
        "target_bond_count": int((target_edge_type > 0).sum().item()),
        "pred_bond_count": int((pred_edge_type > 0).sum().item()),
        "halfedge_accuracy": float((target_edge_type == pred_edge_type).float().mean().item()),
        "exact_halfedge_match": int(bool(torch.equal(target_edge_type, pred_edge_type))),
        **bond_compare,
        **length_stats,
        "plain_sanitize_success": int(plain_result.sanitize_success),
        "plain_repair_strategy": plain_result.repair_strategy,
        "plain_sanitize_error": plain_result.sanitize_error or "",
        "plain_smiles_no_stereo": smiles_no_stereo(plain_result.mol),
        "semantic_sanitize_success": int(semantic_result.sanitize_success),
        "semantic_repair_strategy": semantic_result.repair_strategy,
        "semantic_sanitize_error": semantic_result.sanitize_error or "",
        "semantic_smiles_no_stereo": smiles_no_stereo(semantic_result.mol),
        "original_smiles_no_stereo": original_smiles,
    }
    row["plain_smiles_no_stereo_match"] = int(
        bool(original_smiles) and row["plain_smiles_no_stereo"] == original_smiles
    )
    row["semantic_smiles_no_stereo_match"] = int(
        bool(original_smiles) and row["semantic_smiles_no_stereo"] == original_smiles
    )
    return row


def compare_halfedge_bond_maps(
    halfedge_index: torch.Tensor,
    target_edge_type: torch.Tensor,
    pred_edge_type: torch.Tensor,
) -> dict[str, object]:
    """Compare target and predicted nonzero halfedge bond pair/type maps."""
    target_map = halfedge_bond_map(halfedge_index, target_edge_type)
    pred_map = halfedge_bond_map(halfedge_index, pred_edge_type)
    target_pairs = set(target_map)
    pred_pairs = set(pred_map)
    shared = target_pairs & pred_pairs
    type_mismatch = sum(int(target_map[pair] != pred_map[pair]) for pair in shared)
    true_positive_pairs = len(shared)
    precision = true_positive_pairs / max(1, len(pred_pairs))
    recall = true_positive_pairs / max(1, len(target_pairs))
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "bond_pair_set_match": int(target_pairs == pred_pairs),
        "bond_type_map_match": int(target_map == pred_map),
        "missing_bond_count": len(target_pairs - pred_pairs),
        "extra_bond_count": len(pred_pairs - target_pairs),
        "bond_type_mismatch_count": type_mismatch,
        "bond_pair_precision": precision,
        "bond_pair_recall": recall,
        "bond_pair_f1": f1,
    }


def halfedge_bond_map(
    halfedge_index: torch.Tensor,
    halfedge_type: torch.Tensor,
) -> dict[tuple[int, int], int]:
    """Return nonzero bond pair/type map from halfedge tensors."""
    mapping: dict[tuple[int, int], int] = {}
    for edge_idx, edge_type_tensor in enumerate(halfedge_type):
        edge_type = int(edge_type_tensor)
        if edge_type <= 0:
            continue
        atom_i = int(halfedge_index[0, edge_idx])
        atom_j = int(halfedge_index[1, edge_idx])
        mapping[tuple(sorted((atom_i, atom_j)))] = edge_type
    return mapping


def predicted_bond_length_stats(
    atom_symbols: list[str],
    positions: torch.Tensor,
    halfedge_index: torch.Tensor,
    pred_edge_type: torch.Tensor,
    min_bond_length: float,
    max_bond_length: float,
) -> dict[str, object]:
    """Return coarse predicted-bond length diagnostics."""
    del atom_symbols
    distances = []
    bad_count = 0
    for edge_idx, edge_type_tensor in enumerate(pred_edge_type):
        if int(edge_type_tensor) <= 0:
            continue
        atom_i = int(halfedge_index[0, edge_idx])
        atom_j = int(halfedge_index[1, edge_idx])
        distance = float(torch.linalg.norm(positions[atom_i] - positions[atom_j]).item())
        distances.append(distance)
        bad_count += int(distance < min_bond_length or distance > max_bond_length)
    if not distances:
        return {
            "pred_bond_length_mean": 0.0,
            "pred_bond_length_p95": 0.0,
            "bad_bond_length_count": 0,
            "bad_bond_length_fraction": 0.0,
        }
    return {
        "pred_bond_length_mean": float(np.mean(distances)),
        "pred_bond_length_p95": float(np.percentile(distances, 95)),
        "bad_bond_length_count": bad_count,
        "bad_bond_length_fraction": bad_count / len(distances),
    }


def summarize_reconstruction(
    rows: list[dict[str, object]],
    strategy_counts: Counter[str],
) -> dict[str, object]:
    """Summarize reconstruction rows."""
    return {
        "molecules": len(rows),
        "exact_halfedge_match_rate": _mean_int(rows, "exact_halfedge_match"),
        "bond_pair_set_match_rate": _mean_int(rows, "bond_pair_set_match"),
        "bond_type_map_match_rate": _mean_int(rows, "bond_type_map_match"),
        "mean_halfedge_accuracy": _mean_float(rows, "halfedge_accuracy"),
        "mean_bond_pair_precision": _mean_float(rows, "bond_pair_precision"),
        "mean_bond_pair_recall": _mean_float(rows, "bond_pair_recall"),
        "plain_sanitize_success_rate": _mean_int(rows, "plain_sanitize_success"),
        "semantic_sanitize_success_rate": _mean_int(rows, "semantic_sanitize_success"),
        "plain_smiles_no_stereo_match_rate": _mean_int(rows, "plain_smiles_no_stereo_match"),
        "semantic_smiles_no_stereo_match_rate": _mean_int(rows, "semantic_smiles_no_stereo_match"),
        "mean_bad_bond_length_fraction": _mean_float(rows, "bad_bond_length_fraction"),
        "total_bad_bond_length_count": sum(int(row["bad_bond_length_count"]) for row in rows),
        "repair_strategy_counts": dict(sorted(strategy_counts.items())),
    }


def reconstruction_metric_defaults() -> dict[str, float]:
    """Return placeholder reconstruction metrics for epochs where it is skipped."""
    return {
        "recon_molecules": 0.0,
        "recon_exact_halfedge_match_rate": float("nan"),
        "recon_bond_pair_set_match_rate": float("nan"),
        "recon_bond_type_map_match_rate": float("nan"),
        "recon_mean_halfedge_accuracy": float("nan"),
        "recon_plain_sanitize_success_rate": float("nan"),
        "recon_semantic_sanitize_success_rate": float("nan"),
        "recon_plain_smiles_no_stereo_match_rate": float("nan"),
        "recon_semantic_smiles_no_stereo_match_rate": float("nan"),
        "recon_mean_bad_bond_length_fraction": float("nan"),
        "recon_total_bad_bond_length_count": float("nan"),
    }


def reconstruction_summary_for_training(summary: dict[str, object]) -> dict[str, float]:
    """Prefix reconstruction summary fields for training metrics CSV rows."""
    return {
        "recon_molecules": float(summary["molecules"]),
        "recon_exact_halfedge_match_rate": float(summary["exact_halfedge_match_rate"]),
        "recon_bond_pair_set_match_rate": float(summary["bond_pair_set_match_rate"]),
        "recon_bond_type_map_match_rate": float(summary["bond_type_map_match_rate"]),
        "recon_mean_halfedge_accuracy": float(summary["mean_halfedge_accuracy"]),
        "recon_plain_sanitize_success_rate": float(summary["plain_sanitize_success_rate"]),
        "recon_semantic_sanitize_success_rate": float(summary["semantic_sanitize_success_rate"]),
        "recon_plain_smiles_no_stereo_match_rate": float(summary["plain_smiles_no_stereo_match_rate"]),
        "recon_semantic_smiles_no_stereo_match_rate": float(summary["semantic_smiles_no_stereo_match_rate"]),
        "recon_mean_bad_bond_length_fraction": float(summary["mean_bad_bond_length_fraction"]),
        "recon_total_bad_bond_length_count": float(summary["total_bad_bond_length_count"]),
    }


def reconstruction_fieldnames() -> list[str]:
    """Return fieldnames for empty reconstruction output."""
    return [
        "sample_idx", "name", "num_atoms", "num_halfedges", "target_bond_count",
        "pred_bond_count", "halfedge_accuracy", "exact_halfedge_match",
        "bond_pair_set_match", "bond_type_map_match", "missing_bond_count",
        "extra_bond_count", "bond_type_mismatch_count", "bond_pair_precision",
        "bond_pair_recall", "bond_pair_f1", "pred_bond_length_mean",
        "pred_bond_length_p95", "bad_bond_length_count", "bad_bond_length_fraction",
        "plain_sanitize_success", "plain_repair_strategy", "plain_sanitize_error",
        "plain_smiles_no_stereo", "semantic_sanitize_success",
        "semantic_repair_strategy", "semantic_sanitize_error",
        "semantic_smiles_no_stereo", "original_smiles_no_stereo",
        "plain_smiles_no_stereo_match", "semantic_smiles_no_stereo_match",
    ]


def write_result_mol(writer: Chem.SDWriter, mol: Chem.Mol | None, row: dict[str, object], prefix: str) -> None:
    """Write a reconstructed molecule with metrics as SDF properties."""
    if mol is None:
        return
    if not int(row[f"{prefix}_sanitize_success"]):
        return
    mol.SetProp("_Name", f"{row['sample_idx']}|{row['name']}|{prefix}")
    for key, value in row.items():
        if key == "name":
            continue
        mol.SetProp(str(key), str(value))
    try:
        writer.write(mol)
    except Exception:  # noqa: BLE001 - RDKit SDF writer can fail on rare aromatic states
        return


def smiles_no_stereo(mol: Chem.Mol | None) -> str:
    """Return canonical no-stereo SMILES for a sanitized molecule."""
    if mol is None:
        return ""
    try:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    except Exception:  # noqa: BLE001 - RDKit can fail for partially repaired mols
        return ""


def _safe_divide(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    """Return numerator / denominator with zero where denominator is zero."""
    return torch.where(
        denominator > 0,
        numerator / denominator.clamp_min(1e-12),
        torch.zeros_like(numerator),
    )


def _safe_scalar_divide(numerator: torch.Tensor, denominator: torch.Tensor) -> float:
    """Return scalar division with zero for empty denominators."""
    if float(denominator.item()) <= 0:
        return 0.0
    return float((numerator / denominator.clamp_min(1e-12)).item())


def _mean_int(rows: list[dict[str, object]], key: str) -> float:
    """Return mean of integer 0/1 fields."""
    if not rows:
        return 0.0
    return sum(int(row[key]) for row in rows) / len(rows)


def _mean_float(rows: list[dict[str, object]], key: str) -> float:
    """Return mean of numeric fields."""
    if not rows:
        return 0.0
    return float(sum(float(row[key]) for row in rows) / len(rows))
