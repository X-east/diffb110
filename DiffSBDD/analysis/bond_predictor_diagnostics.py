"""Diagnose bond predictors and DiffSBDD point-cloud geometry."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from rdkit import Chem, RDLogger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIFFSBDD_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_ROOT = Path(__file__).resolve().parent
for extra in (PROJECT_ROOT, DIFFSBDD_ROOT, ANALYSIS_ROOT):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import constants
from dataset import ProcessedLigandPocketDataset
from edge_data_oracle_report import _build_oracle_mol, _lookup_reference_length
from generated_reconstruction_ablation import _sample_point_clouds
from lightning_modules import LigandPocketDDPM
from reconstruction_benchmark import (
    ReconstructionAttempt,
    create_reconstruction_methods,
    evaluate_reconstruction_attempt,
)

RDLogger.DisableLog("rdApp.*")

BOND_RECOVERY_FIELDS = [
    "true_bond_count",
    "pred_bond_count",
    "pred_to_true_bond_ratio",
    "edge_precision",
    "edge_recall",
    "edge_f1",
    "typed_precision",
    "typed_recall",
    "typed_f1",
    "pair_type_accuracy",
    "no_bond_recall",
]

CLEAN_NOISE_NUMERIC_FIELDS = [
    "sanitize_success",
    "num_atoms",
    "num_bonds",
    "bad_bond_length_rate",
    "mean_bond_length_abs_error",
    "chemistry_problem_count",
    "valence_problem_count",
    "num_fragments",
    "largest_fragment_atom_frac",
    "ring_count",
    "aromatic_ring_count",
    "three_member_ring_count",
    "carbon_fraction",
    "hetero_atom_count",
    "halogen_count",
    "qed",
    "sa",
    "logp",
    "tpsa",
    "reconstruction_seconds",
    *BOND_RECOVERY_FIELDS,
]

GEOMETRY_NUMERIC_FIELDS = [
    "num_atoms",
    "carbon_fraction",
    "hetero_atom_fraction",
    "halogen_fraction",
    "nearest_neighbor_min",
    "nearest_neighbor_mean",
    "nearest_neighbor_q10",
    "nearest_neighbor_q50",
    "nearest_neighbor_q90",
    "all_pair_distance_q10",
    "all_pair_distance_q50",
    "all_pair_distance_q90",
    "clash_pair_count",
    "clash_pair_fraction",
    "candidate_bond_pair_count",
    "candidate_bond_pair_fraction",
    "candidate_bond_pairs_per_atom",
    "true_bond_count",
    "candidate_to_true_bond_ratio",
    "true_bond_candidate_recall",
    "candidate_false_positive_pairs",
]

GENERATED_RECONSTRUCTION_NUMERIC_FIELDS = [
    "sanitize_success",
    "num_atoms",
    "num_bonds",
    "bad_bond_length_rate",
    "mean_bond_length_abs_error",
    "chemistry_problem_count",
    "valence_problem_count",
    "num_fragments",
    "largest_fragment_atom_frac",
    "ring_count",
    "aromatic_ring_count",
    "three_member_ring_count",
    "carbon_fraction",
    "hetero_atom_count",
    "halogen_count",
    "qed",
    "sa",
    "logp",
    "tpsa",
    "reconstruction_seconds",
]

PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2", "#FF9DA6"]
BOND_TYPE_NAMES = {
    0: "no_bond",
    1: "single",
    2: "double",
    3: "triple",
    4: "aromatic",
}


@dataclass(frozen=True)
class LabeledPointCloud:
    """One ligand point cloud with optional true halfedge labels."""

    sample_idx: int
    name: str
    coords: torch.Tensor
    atom_types: torch.Tensor
    atom_symbols: list[str]
    context: dict[str, Any]
    halfedge_index: torch.Tensor | None = None
    halfedge_type: torch.Tensor | None = None


def _resolve_device(requested_device: str | None) -> str:
    """Return a usable torch device string, falling back when CUDA is hidden."""
    if not requested_device:
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested_device


def _format_value(value: object) -> object:
    """Format floats for stable CSV output."""
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    """Write CSV rows using a fixed field order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_value(row.get(key, "")) for key in fieldnames})


def _mean(values: Iterable[float]) -> float:
    """Return the arithmetic mean or 0 for empty inputs."""
    values_list = list(values)
    return sum(values_list) / len(values_list) if values_list else 0.0


def _quantile(values: list[float], q: float) -> float:
    """Return a deterministic linear quantile for a list of floats."""
    if not values:
        return 0.0
    sorted_values = sorted(values)
    position = (len(sorted_values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _summarize_rows(
    rows: list[dict[str, object]],
    group_fields: list[str],
    numeric_fields: list[str],
) -> list[dict[str, object]]:
    """Aggregate numeric row fields by group."""
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in group_fields)].append(row)

    summary_rows = []
    for key, group_rows in sorted(grouped.items(), key=lambda item: item[0]):
        summary: dict[str, object] = dict(zip(group_fields, key))
        summary["samples"] = len(group_rows)
        for field in numeric_fields:
            values = []
            for row in group_rows:
                value = row.get(field)
                if value == "" or value is None:
                    continue
                try:
                    values.append(float(value))
                except (TypeError, ValueError):
                    continue
            summary[f"mean_{field}"] = _mean(values)
        error_counts = Counter(str(row.get("error", "")) for row in group_rows)
        if "" in error_counts:
            del error_counts[""]
        summary["top_error"] = error_counts.most_common(1)[0][0] if error_counts else ""
        summary["top_error_count"] = error_counts.most_common(1)[0][1] if error_counts else 0
        summary_rows.append(summary)
    return summary_rows


def _dataset_atom_symbols(one_hot: torch.Tensor, dataset_info: dict[str, Any]) -> tuple[torch.Tensor, list[str]]:
    """Convert ligand one-hot atom features to integer ids and symbols."""
    atom_types = one_hot.argmax(dim=-1).long()
    atom_symbols = [dataset_info["atom_decoder"][int(atom_idx)] for atom_idx in atom_types]
    return atom_types, atom_symbols


def _select_dataset_indices(dataset_size: int, max_samples: int, seed: int, sample_mode: str) -> list[int]:
    """Select a deterministic subset of dataset indices."""
    limit = min(int(max_samples), int(dataset_size))
    if sample_mode == "head":
        return list(range(limit))
    generator = torch.Generator().manual_seed(int(seed))
    indices = torch.randperm(dataset_size, generator=generator)[:limit].tolist()
    return sorted(int(idx) for idx in indices)


def _load_crossdocked_samples(
    npz_path: Path,
    dataset_info: dict[str, Any],
    *,
    max_samples: int,
    seed: int,
    sample_mode: str,
) -> list[LabeledPointCloud]:
    """Load a fixed subset of CrossDocked ligand point clouds with true bonds."""
    dataset = ProcessedLigandPocketDataset(npz_path, center=False)
    selected_indices = _select_dataset_indices(len(dataset), max_samples, seed, sample_mode)
    samples = []
    for sample_idx in selected_indices:
        item = dataset[sample_idx]
        atom_types, atom_symbols = _dataset_atom_symbols(item["lig_one_hot"], dataset_info)
        samples.append(LabeledPointCloud(
            sample_idx=int(sample_idx),
            name=str(item["names"]),
            coords=item["lig_coords"].float(),
            atom_types=atom_types,
            atom_symbols=atom_symbols,
            context={},
            halfedge_index=item["ligand_halfedge_index"].long(),
            halfedge_type=item["ligand_halfedge_type"].long(),
        ))
    return samples


def _with_coordinate_noise(samples: list[LabeledPointCloud], noise_std: float, seed: int) -> list[LabeledPointCloud]:
    """Return copies of samples with deterministic isotropic coordinate noise."""
    noisy_samples = []
    for sample in samples:
        if noise_std == 0:
            coords = sample.coords.clone()
        else:
            generator = torch.Generator().manual_seed(int(seed) + sample.sample_idx * 1009 + int(round(noise_std * 10000)))
            noise = torch.randn(sample.coords.shape, generator=generator, dtype=sample.coords.dtype)
            coords = sample.coords + float(noise_std) * noise
        noisy_samples.append(LabeledPointCloud(
            sample_idx=sample.sample_idx,
            name=sample.name,
            coords=coords,
            atom_types=sample.atom_types,
            atom_symbols=sample.atom_symbols,
            context=sample.context,
            halfedge_index=sample.halfedge_index,
            halfedge_type=sample.halfedge_type,
        ))
    return noisy_samples


def _method_sample(sample: LabeledPointCloud) -> dict[str, Any]:
    """Convert a labeled sample to the reconstruction benchmark sample shape."""
    return {
        "sample_idx": sample.sample_idx,
        "coords": sample.coords,
        "atom_types": sample.atom_types,
        "context": sample.context,
    }


def _rdkit_edges(mol: Chem.Mol | None) -> dict[tuple[int, int], int]:
    """Return undirected RDKit edges as compact bond type ids."""
    if mol is None:
        return {}
    bond_type_to_id = {
        Chem.BondType.SINGLE: 1,
        Chem.BondType.DOUBLE: 2,
        Chem.BondType.TRIPLE: 3,
        Chem.BondType.AROMATIC: 4,
    }
    edges = {}
    for bond in mol.GetBonds():
        edge_type = bond_type_to_id.get(bond.GetBondType(), 0)
        if edge_type <= 0:
            continue
        atom_i = int(bond.GetBeginAtomIdx())
        atom_j = int(bond.GetEndAtomIdx())
        edges[tuple(sorted((atom_i, atom_j)))] = edge_type
    return edges


def _true_edge_map(sample: LabeledPointCloud) -> dict[tuple[int, int], int]:
    """Return true positive halfedges as an undirected bond map."""
    if sample.halfedge_index is None or sample.halfedge_type is None:
        return {}
    true_edges = {}
    for edge_idx, edge_type_tensor in enumerate(sample.halfedge_type):
        edge_type = int(edge_type_tensor)
        if edge_type <= 0:
            continue
        atom_i = int(sample.halfedge_index[0, edge_idx])
        atom_j = int(sample.halfedge_index[1, edge_idx])
        true_edges[tuple(sorted((atom_i, atom_j)))] = edge_type
    return true_edges


def _safe_divide(numerator: float, denominator: float) -> float:
    """Divide while returning 0 for a zero denominator."""
    return numerator / denominator if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    """Return F1 from precision and recall."""
    return _safe_divide(2.0 * precision * recall, precision + recall)


def _bond_recovery_metrics(sample: LabeledPointCloud, mol: Chem.Mol | None) -> dict[str, float | int]:
    """Compare predicted RDKit bonds against true CrossDocked halfedge labels."""
    if sample.halfedge_index is None or sample.halfedge_type is None:
        return {field: 0 for field in BOND_RECOVERY_FIELDS}

    true_edges = _true_edge_map(sample)
    pred_edges = _rdkit_edges(mol)
    true_edge_keys = set(true_edges)
    pred_edge_keys = set(pred_edges)
    edge_tp = len(true_edge_keys & pred_edge_keys)
    edge_fp = len(pred_edge_keys - true_edge_keys)
    edge_fn = len(true_edge_keys - pred_edge_keys)
    typed_tp = sum(
        int(pred_edges[pair] == true_edges[pair])
        for pair in true_edge_keys & pred_edge_keys
    )

    edge_precision = _safe_divide(edge_tp, edge_tp + edge_fp)
    edge_recall = _safe_divide(edge_tp, edge_tp + edge_fn)
    typed_precision = _safe_divide(typed_tp, len(pred_edge_keys))
    typed_recall = _safe_divide(typed_tp, len(true_edge_keys))

    correct_pairs = 0
    true_no_bonds = 0
    true_no_bond_correct = 0
    for edge_idx, true_type_tensor in enumerate(sample.halfedge_type):
        atom_i = int(sample.halfedge_index[0, edge_idx])
        atom_j = int(sample.halfedge_index[1, edge_idx])
        pair = tuple(sorted((atom_i, atom_j)))
        true_type = int(true_type_tensor)
        pred_type = pred_edges.get(pair, 0)
        correct_pairs += int(pred_type == true_type)
        if true_type == 0:
            true_no_bonds += 1
            true_no_bond_correct += int(pred_type == 0)

    return {
        "true_bond_count": len(true_edge_keys),
        "pred_bond_count": len(pred_edge_keys),
        "pred_to_true_bond_ratio": _safe_divide(len(pred_edge_keys), len(true_edge_keys)),
        "edge_precision": edge_precision,
        "edge_recall": edge_recall,
        "edge_f1": _f1(edge_precision, edge_recall),
        "typed_precision": typed_precision,
        "typed_recall": typed_recall,
        "typed_f1": _f1(typed_precision, typed_recall),
        "pair_type_accuracy": _safe_divide(correct_pairs, int(sample.halfedge_type.numel())),
        "no_bond_recall": _safe_divide(true_no_bond_correct, true_no_bonds),
    }


def _bond_type_confusion_counts(sample: LabeledPointCloud, mol: Chem.Mol | None) -> Counter[tuple[int, int]]:
    """Count true-vs-predicted halfedge type pairs for one molecule."""
    counts: Counter[tuple[int, int]] = Counter()
    if sample.halfedge_index is None or sample.halfedge_type is None:
        return counts
    pred_edges = _rdkit_edges(mol)
    for edge_idx, true_type_tensor in enumerate(sample.halfedge_type):
        atom_i = int(sample.halfedge_index[0, edge_idx])
        atom_j = int(sample.halfedge_index[1, edge_idx])
        pair = tuple(sorted((atom_i, atom_j)))
        true_type = int(true_type_tensor)
        pred_type = int(pred_edges.get(pair, 0))
        counts[(true_type, pred_type)] += 1
    return counts


def _confusion_rows(
    confusion: Counter[tuple[float, str, int, int]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Convert aggregate confusion counts to CSV-ready rows and per-type metrics."""
    rows = []
    metric_rows = []
    groups = sorted({(noise, method) for noise, method, _, _ in confusion})
    for noise_std, method in groups:
        for true_type in sorted(BOND_TYPE_NAMES):
            support = sum(confusion[(noise_std, method, true_type, pred_type)] for pred_type in BOND_TYPE_NAMES)
            predicted_as_type = sum(confusion[(noise_std, method, other_true, true_type)] for other_true in BOND_TYPE_NAMES)
            true_positive = confusion[(noise_std, method, true_type, true_type)]
            precision = _safe_divide(true_positive, predicted_as_type)
            recall = _safe_divide(true_positive, support)
            metric_rows.append({
                "noise_std": noise_std,
                "method": method,
                "bond_type_id": true_type,
                "bond_type": BOND_TYPE_NAMES[true_type],
                "support": support,
                "predicted_as_type": predicted_as_type,
                "precision": precision,
                "recall": recall,
                "f1": _f1(precision, recall),
            })
            for pred_type in sorted(BOND_TYPE_NAMES):
                count = confusion[(noise_std, method, true_type, pred_type)]
                rows.append({
                    "noise_std": noise_std,
                    "method": method,
                    "true_type_id": true_type,
                    "true_type": BOND_TYPE_NAMES[true_type],
                    "pred_type_id": pred_type,
                    "pred_type": BOND_TYPE_NAMES[pred_type],
                    "count": count,
                    "row_fraction": _safe_divide(count, support),
                })
    return rows, metric_rows


def _oracle_attempt(sample: LabeledPointCloud) -> ReconstructionAttempt:
    """Build one molecule using true CrossDocked halfedge labels."""
    started_at = time.perf_counter()
    if sample.halfedge_index is None or sample.halfedge_type is None:
        return ReconstructionAttempt("oracle_true_bond", None, "MissingTrueHalfedges", 0.0)
    mol, sanitize_error = _build_oracle_mol(
        sample.atom_symbols,
        sample.coords,
        sample.halfedge_index,
        sample.halfedge_type,
    )
    return ReconstructionAttempt(
        method="oracle_true_bond",
        raw_mol=mol,
        build_error="" if sanitize_error is None else sanitize_error,
        seconds=time.perf_counter() - started_at,
    )


def _evaluate_attempt(
    *,
    sample: LabeledPointCloud,
    attempt: ReconstructionAttempt,
    length_tolerance: float,
    extra_fields: dict[str, object],
) -> tuple[dict[str, object], Chem.Mol | None]:
    """Evaluate a reconstruction attempt and append diagnostic metadata."""
    row, sanitized_mol = evaluate_reconstruction_attempt(
        sample_idx=sample.sample_idx,
        attempt=attempt,
        coords=sample.coords,
        atom_symbols=sample.atom_symbols,
        length_tolerance=length_tolerance,
    )
    row.update({
        "source_sample_idx": sample.sample_idx,
        "name": sample.name,
        **extra_fields,
    })
    row.update(_bond_recovery_metrics(sample, attempt.raw_mol))
    return row, sanitized_mol


def _run_clean_noise_experiment(
    *,
    args: argparse.Namespace,
    base_samples: list[LabeledPointCloud],
    dataset_info: dict[str, Any],
    outdir: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Run clean and noisy CrossDocked bond predictor diagnostics."""
    method_names = list(args.methods)
    include_oracle = "oracle_true_bond" in method_names or "oracle" in method_names
    method_names = [
        method
        for method in method_names
        if method not in {"oracle_true_bond", "oracle"}
    ]
    method_objects = create_reconstruction_methods(
        method_names,
        dataset_info,
        project_root=PROJECT_ROOT,
        cache_dir=outdir / "external_bond_predictor_cache",
        external_device=args.external_bond_device,
        local_bond_checkpoint=args.local_bond_checkpoint,
        local_bond_device=args.local_bond_device,
        diffgui_env=args.diffgui_env,
        molsnapper_env=args.molsnapper_env,
    )

    per_molecule_rows: list[dict[str, object]] = []
    confusion_counter: Counter[tuple[float, str, int, int]] = Counter()
    for noise_std in args.noise_stds:
        noisy_samples = _with_coordinate_noise(base_samples, float(noise_std), int(args.seed))
        sample_by_idx = {sample.sample_idx: sample for sample in noisy_samples}
        extra_fields = {"experiment": "clean_noise", "noise_std": float(noise_std)}

        if include_oracle:
            for sample in noisy_samples:
                attempt = _oracle_attempt(sample)
                row, _ = _evaluate_attempt(
                    sample=sample,
                    attempt=attempt,
                    length_tolerance=args.length_tolerance,
                    extra_fields=extra_fields,
                )
                for (true_type, pred_type), count in _bond_type_confusion_counts(sample, attempt.raw_mol).items():
                    confusion_counter[(float(noise_std), "oracle_true_bond", true_type, pred_type)] += count
                per_molecule_rows.append(row)

        for method in method_objects:
            benchmark_samples = [_method_sample(sample) for sample in noisy_samples]
            reconstruct_many = getattr(method, "reconstruct_many", None)
            if callable(reconstruct_many):
                attempts = reconstruct_many(benchmark_samples, dataset_info)
            else:
                attempts = [
                    (
                        sample_dict,
                        method.reconstruct(
                            sample_dict["coords"],
                            sample_dict["atom_types"],
                            dataset_info,
                            context=sample_dict.get("context"),
                        ),
                    )
                    for sample_dict in benchmark_samples
                ]
            for sample_dict, attempt in attempts:
                sample = sample_by_idx[int(sample_dict["sample_idx"])]
                row, _ = _evaluate_attempt(
                    sample=sample,
                    attempt=attempt,
                    length_tolerance=args.length_tolerance,
                    extra_fields=extra_fields,
                )
                for (true_type, pred_type), count in _bond_type_confusion_counts(sample, attempt.raw_mol).items():
                    confusion_counter[(float(noise_std), attempt.method, true_type, pred_type)] += count
                per_molecule_rows.append(row)

    summary_rows = _summarize_rows(
        per_molecule_rows,
        group_fields=["noise_std", "method"],
        numeric_fields=CLEAN_NOISE_NUMERIC_FIELDS,
    )
    confusion_rows, type_metric_rows = _confusion_rows(confusion_counter)

    clean_noise_fields = [
        "experiment",
        "noise_std",
        "method",
        "source_sample_idx",
        "sample_idx",
        "name",
        "num_atoms",
        "sanitize_success",
        "error",
        "num_bonds",
        "checked_bond_lengths",
        "bad_bond_lengths",
        "bad_bond_length_rate",
        "mean_bond_length_abs_error",
        "chemistry_problem_count",
        "valence_problem_count",
        "num_fragments",
        "largest_fragment_atom_frac",
        "ring_count",
        "aromatic_ring_count",
        "three_member_ring_count",
        "fused_ring_pair_count",
        "carbon_fraction",
        "hetero_atom_count",
        "halogen_count",
        "formal_charge_abs_sum",
        "drug_metrics_available",
        "qed",
        "sa",
        "logp",
        "tpsa",
        "hba",
        "hbd",
        "rotatable_bonds",
        "mol_weight",
        "lipinski",
        "reconstruction_seconds",
        *BOND_RECOVERY_FIELDS,
        "smiles",
    ]
    summary_fields = ["noise_std", "method", "samples"] + [
        f"mean_{field}" for field in CLEAN_NOISE_NUMERIC_FIELDS
    ] + ["top_error", "top_error_count"]
    confusion_fields = [
        "noise_std",
        "method",
        "true_type_id",
        "true_type",
        "pred_type_id",
        "pred_type",
        "count",
        "row_fraction",
    ]
    type_metric_fields = [
        "noise_std",
        "method",
        "bond_type_id",
        "bond_type",
        "support",
        "predicted_as_type",
        "precision",
        "recall",
        "f1",
    ]
    _write_csv(outdir / "clean_noise_per_molecule.csv", per_molecule_rows, clean_noise_fields)
    _write_csv(outdir / "clean_noise_summary.csv", summary_rows, summary_fields)
    _write_csv(outdir / "clean_noise_bond_type_confusion.csv", confusion_rows, confusion_fields)
    _write_csv(outdir / "clean_noise_bond_type_metrics.csv", type_metric_rows, type_metric_fields)
    _plot_clean_noise(outdir, summary_rows)
    _plot_clean_confusion(outdir, confusion_rows)
    return per_molecule_rows, summary_rows


def _pair_candidate_status(
    atom_a: str,
    atom_b: str,
    distance: float,
    tolerance: float,
) -> bool:
    """Return whether a pair distance sits near any known covalent-bond reference."""
    for bond_type in (1, 2, 3, 4):
        reference = _lookup_reference_length(atom_a, atom_b, bond_type)
        if reference is not None and abs(distance - reference) <= tolerance:
            return True
    return False


def _geometry_metrics(
    sample: LabeledPointCloud,
    *,
    source: str,
    length_tolerance: float,
    clash_threshold: float,
) -> dict[str, object]:
    """Calculate point-cloud geometry and composition metrics."""
    coords = sample.coords.float()
    atom_symbols = sample.atom_symbols
    num_atoms = len(atom_symbols)
    halogens = {"F", "Cl", "Br", "I"}
    carbon_count = sum(int(symbol == "C") for symbol in atom_symbols)
    hetero_count = sum(int(symbol not in {"C", "H"}) for symbol in atom_symbols)
    halogen_count = sum(int(symbol in halogens) for symbol in atom_symbols)
    element_counts = Counter(atom_symbols)

    pair_distances: list[float] = []
    candidate_pairs: set[tuple[int, int]] = set()
    clash_count = 0
    if num_atoms >= 2:
        halfedge_index = torch.triu_indices(num_atoms, num_atoms, offset=1)
        for edge_idx in range(halfedge_index.shape[1]):
            atom_i = int(halfedge_index[0, edge_idx])
            atom_j = int(halfedge_index[1, edge_idx])
            distance = float(torch.linalg.norm(coords[atom_i] - coords[atom_j]).item())
            pair_distances.append(distance)
            clash_count += int(distance < clash_threshold)
            if _pair_candidate_status(atom_symbols[atom_i], atom_symbols[atom_j], distance, length_tolerance):
                candidate_pairs.add((atom_i, atom_j))

        distance_matrix = torch.cdist(coords, coords)
        distance_matrix.fill_diagonal_(float("inf"))
        nearest_distances = distance_matrix.min(dim=1).values.cpu().tolist()
    else:
        nearest_distances = []

    true_edges = _true_edge_map(sample)
    true_edge_keys = set(true_edges)
    candidate_true = candidate_pairs & true_edge_keys
    candidate_false = candidate_pairs - true_edge_keys if true_edge_keys else set()
    all_pair_count = num_atoms * (num_atoms - 1) // 2
    row: dict[str, object] = {
        "source": source,
        "sample_idx": sample.sample_idx,
        "name": sample.name,
        "num_atoms": num_atoms,
        "carbon_count": carbon_count,
        "nitrogen_count": element_counts.get("N", 0),
        "oxygen_count": element_counts.get("O", 0),
        "sulfur_count": element_counts.get("S", 0),
        "phosphorus_count": element_counts.get("P", 0),
        "halogen_count": halogen_count,
        "hetero_atom_count": hetero_count,
        "carbon_fraction": _safe_divide(carbon_count, num_atoms),
        "hetero_atom_fraction": _safe_divide(hetero_count, num_atoms),
        "halogen_fraction": _safe_divide(halogen_count, num_atoms),
        "all_pair_count": all_pair_count,
        "nearest_neighbor_min": min(nearest_distances) if nearest_distances else 0.0,
        "nearest_neighbor_mean": _mean(nearest_distances),
        "nearest_neighbor_q10": _quantile(nearest_distances, 0.10),
        "nearest_neighbor_q50": _quantile(nearest_distances, 0.50),
        "nearest_neighbor_q90": _quantile(nearest_distances, 0.90),
        "all_pair_distance_q10": _quantile(pair_distances, 0.10),
        "all_pair_distance_q50": _quantile(pair_distances, 0.50),
        "all_pair_distance_q90": _quantile(pair_distances, 0.90),
        "clash_pair_count": clash_count,
        "clash_pair_fraction": _safe_divide(clash_count, all_pair_count),
        "candidate_bond_pair_count": len(candidate_pairs),
        "candidate_bond_pair_fraction": _safe_divide(len(candidate_pairs), all_pair_count),
        "candidate_bond_pairs_per_atom": _safe_divide(len(candidate_pairs), num_atoms),
        "true_bond_count": len(true_edge_keys),
        "candidate_to_true_bond_ratio": _safe_divide(len(candidate_pairs), len(true_edge_keys)),
        "true_bond_candidate_recall": _safe_divide(len(candidate_true), len(true_edge_keys)),
        "candidate_false_positive_pairs": len(candidate_false),
    }
    return row


def _load_generated_point_clouds(args: argparse.Namespace, dataset_info: dict[str, Any]) -> list[LabeledPointCloud]:
    """Sample DiffSBDD point clouds and return them in the diagnostic sample shape."""
    del dataset_info
    device = torch.device(_resolve_device(args.device))
    if args.seed is not None:
        torch.manual_seed(int(args.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed))

    model = LigandPocketDDPM.load_from_checkpoint(args.checkpoint, map_location=device)
    model = model.to(device)
    generated = _sample_point_clouds(
        model,
        args.pdbfile,
        n_samples=args.generated_samples,
        batch_size=args.generated_batch_size,
        pocket_ids=args.resi_list,
        ref_ligand=args.ref_ligand,
        timesteps=args.generated_timesteps,
        num_nodes_lig=args.num_nodes_lig,
        n_nodes_bias=args.n_nodes_bias,
        n_nodes_min=args.n_nodes_min,
    )
    generated_samples = []
    for sample in generated:
        atom_types = sample["atom_types"].long()
        atom_symbols = [model.dataset_info["atom_decoder"][int(atom_idx)] for atom_idx in atom_types]
        generated_samples.append(LabeledPointCloud(
            sample_idx=int(sample["sample_idx"]),
            name=f"generated_{int(sample['sample_idx'])}",
            coords=sample["coords"].float(),
            atom_types=atom_types,
            atom_symbols=atom_symbols,
            context=sample.get("context", {}),
        ))
    return generated_samples


def _run_generated_geometry_experiment(
    *,
    args: argparse.Namespace,
    base_samples: list[LabeledPointCloud],
    dataset_info: dict[str, Any],
    outdir: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Run DiffSBDD generated point-cloud geometry and reconstruction diagnostics."""
    generated_samples = _load_generated_point_clouds(args, dataset_info)
    geometry_rows = [
        _geometry_metrics(
            sample,
            source="crossdocked_clean",
            length_tolerance=args.length_tolerance,
            clash_threshold=args.clash_threshold,
        )
        for sample in base_samples
    ]
    geometry_rows.extend(
        _geometry_metrics(
            sample,
            source="diffsbdd_generated",
            length_tolerance=args.length_tolerance,
            clash_threshold=args.clash_threshold,
        )
        for sample in generated_samples
    )
    geometry_summary = _summarize_rows(
        geometry_rows,
        group_fields=["source"],
        numeric_fields=GEOMETRY_NUMERIC_FIELDS,
    )

    reconstruction_rows: list[dict[str, object]] = []
    method_objects = create_reconstruction_methods(
        args.generated_reconstruction_methods,
        dataset_info,
        project_root=PROJECT_ROOT,
        cache_dir=outdir / "external_bond_predictor_cache_generated",
        external_device=args.external_bond_device,
        local_bond_checkpoint=args.local_bond_checkpoint,
        local_bond_device=args.local_bond_device,
        diffgui_env=args.diffgui_env,
        molsnapper_env=args.molsnapper_env,
    )
    sample_by_idx = {sample.sample_idx: sample for sample in generated_samples}
    for method in method_objects:
        benchmark_samples = [_method_sample(sample) for sample in generated_samples]
        reconstruct_many = getattr(method, "reconstruct_many", None)
        if callable(reconstruct_many):
            attempts = reconstruct_many(benchmark_samples, dataset_info)
        else:
            attempts = [
                (
                    sample_dict,
                    method.reconstruct(
                        sample_dict["coords"],
                        sample_dict["atom_types"],
                        dataset_info,
                        context=sample_dict.get("context"),
                    ),
                )
                for sample_dict in benchmark_samples
            ]
        for sample_dict, attempt in attempts:
            sample = sample_by_idx[int(sample_dict["sample_idx"])]
            row, _ = evaluate_reconstruction_attempt(
                sample_idx=sample.sample_idx,
                attempt=attempt,
                coords=sample.coords,
                atom_symbols=sample.atom_symbols,
                length_tolerance=args.length_tolerance,
            )
            row.update({
                "experiment": "generated_reconstruction",
                "source": "diffsbdd_generated",
                "source_sample_idx": sample.sample_idx,
                "name": sample.name,
            })
            reconstruction_rows.append(row)

    reconstruction_summary = _summarize_rows(
        reconstruction_rows,
        group_fields=["method"],
        numeric_fields=GENERATED_RECONSTRUCTION_NUMERIC_FIELDS,
    )

    geometry_fields = [
        "source",
        "sample_idx",
        "name",
        "num_atoms",
        "carbon_count",
        "nitrogen_count",
        "oxygen_count",
        "sulfur_count",
        "phosphorus_count",
        "halogen_count",
        "hetero_atom_count",
        "carbon_fraction",
        "hetero_atom_fraction",
        "halogen_fraction",
        "all_pair_count",
        "nearest_neighbor_min",
        "nearest_neighbor_mean",
        "nearest_neighbor_q10",
        "nearest_neighbor_q50",
        "nearest_neighbor_q90",
        "all_pair_distance_q10",
        "all_pair_distance_q50",
        "all_pair_distance_q90",
        "clash_pair_count",
        "clash_pair_fraction",
        "candidate_bond_pair_count",
        "candidate_bond_pair_fraction",
        "candidate_bond_pairs_per_atom",
        "true_bond_count",
        "candidate_to_true_bond_ratio",
        "true_bond_candidate_recall",
        "candidate_false_positive_pairs",
    ]
    geometry_summary_fields = ["source", "samples"] + [
        f"mean_{field}" for field in GEOMETRY_NUMERIC_FIELDS
    ] + ["top_error", "top_error_count"]
    reconstruction_fields = [
        "experiment",
        "source",
        "method",
        "source_sample_idx",
        "sample_idx",
        "name",
        "num_atoms",
        "sanitize_success",
        "error",
        "num_bonds",
        "checked_bond_lengths",
        "bad_bond_lengths",
        "bad_bond_length_rate",
        "mean_bond_length_abs_error",
        "chemistry_problem_count",
        "valence_problem_count",
        "num_fragments",
        "largest_fragment_atom_frac",
        "ring_count",
        "aromatic_ring_count",
        "three_member_ring_count",
        "fused_ring_pair_count",
        "carbon_fraction",
        "hetero_atom_count",
        "halogen_count",
        "formal_charge_abs_sum",
        "drug_metrics_available",
        "qed",
        "sa",
        "logp",
        "tpsa",
        "hba",
        "hbd",
        "rotatable_bonds",
        "mol_weight",
        "lipinski",
        "reconstruction_seconds",
        "smiles",
    ]
    reconstruction_summary_fields = ["method", "samples"] + [
        f"mean_{field}" for field in GENERATED_RECONSTRUCTION_NUMERIC_FIELDS
    ] + ["top_error", "top_error_count"]
    _write_csv(outdir / "generated_geometry_per_molecule.csv", geometry_rows, geometry_fields)
    _write_csv(outdir / "generated_geometry_summary.csv", geometry_summary, geometry_summary_fields)
    _write_csv(outdir / "generated_reconstruction_per_molecule.csv", reconstruction_rows, reconstruction_fields)
    _write_csv(outdir / "generated_reconstruction_summary.csv", reconstruction_summary, reconstruction_summary_fields)
    _plot_generated_geometry(outdir, geometry_rows, geometry_summary, reconstruction_summary)
    return geometry_rows, geometry_summary, reconstruction_rows, reconstruction_summary


def _plot_clean_noise(outdir: Path, summary_rows: list[dict[str, object]]) -> None:
    """Plot clean/noise metrics by method."""
    specs = [
        ("mean_sanitize_success", "sanitize success", "sanitize_by_noise_method.png"),
        ("mean_edge_f1", "edge F1", "edge_f1_by_noise_method.png"),
        ("mean_typed_f1", "typed bond F1", "typed_f1_by_noise_method.png"),
        ("mean_bad_bond_length_rate", "bad bond length rate", "bad_bond_length_by_noise_method.png"),
        ("mean_pred_to_true_bond_ratio", "predicted / true bonds", "predicted_bond_ratio_by_noise_method.png"),
        ("mean_reconstruction_seconds", "seconds / molecule", "time_by_noise_method.png"),
    ]
    methods = sorted({str(row["method"]) for row in summary_rows})
    for field, ylabel, filename in specs:
        plt.figure(figsize=(7, 4))
        for color_idx, method in enumerate(methods):
            rows = sorted(
                [row for row in summary_rows if row["method"] == method],
                key=lambda item: float(item["noise_std"]),
            )
            xs = [float(row["noise_std"]) for row in rows]
            ys = [float(row.get(field, 0.0)) for row in rows]
            plt.plot(xs, ys, marker="o", label=method, color=PALETTE[color_idx % len(PALETTE)])
        plt.xlabel("coordinate noise std (Angstrom)")
        plt.ylabel(ylabel)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(outdir / filename, dpi=180)
        plt.close()


def _plot_clean_confusion(outdir: Path, confusion_rows: list[dict[str, object]]) -> None:
    """Plot row-normalized clean bond-type confusion matrices by method."""
    clean_rows = [row for row in confusion_rows if float(row["noise_std"]) == 0.0]
    methods = sorted({str(row["method"]) for row in clean_rows})
    if not methods:
        return

    cols = 3
    rows = (len(methods) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.1, rows * 3.8), squeeze=False)
    labels = [BOND_TYPE_NAMES[idx] for idx in sorted(BOND_TYPE_NAMES)]
    for method_idx, method in enumerate(methods):
        ax = axes[method_idx // cols][method_idx % cols]
        matrix = torch.zeros((len(labels), len(labels)), dtype=torch.float32)
        for row in clean_rows:
            if row["method"] != method:
                continue
            true_type = int(row["true_type_id"])
            pred_type = int(row["pred_type_id"])
            matrix[true_type, pred_type] = float(row["row_fraction"])
        im = ax.imshow(matrix.numpy(), vmin=0.0, vmax=1.0, cmap="Blues")
        ax.set_title(method, fontsize=10)
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(labels)), labels, fontsize=8)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        for true_type in range(len(labels)):
            for pred_type in range(len(labels)):
                value = float(matrix[true_type, pred_type])
                if value >= 0.01:
                    ax.text(pred_type, true_type, f"{value:.2f}", ha="center", va="center", fontsize=7)
    for empty_idx in range(len(methods), rows * cols):
        axes[empty_idx // cols][empty_idx % cols].axis("off")
    fig.subplots_adjust(right=0.88, hspace=0.65, wspace=0.45)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75, label="row fraction")
    plt.savefig(outdir / "clean_bond_type_confusion_by_method.png", dpi=180)
    plt.close()


def _plot_generated_geometry(
    outdir: Path,
    geometry_rows: list[dict[str, object]],
    geometry_summary: list[dict[str, object]],
    reconstruction_summary: list[dict[str, object]],
) -> None:
    """Plot generated point-cloud geometry diagnostics."""
    for field, xlabel, filename in [
        ("nearest_neighbor_mean", "nearest-neighbor mean distance (Angstrom)", "nearest_neighbor_clean_vs_generated.png"),
        ("candidate_bond_pairs_per_atom", "candidate bond pairs per atom", "candidate_bond_pairs_clean_vs_generated.png"),
        ("clash_pair_fraction", "clash pair fraction", "clash_fraction_clean_vs_generated.png"),
    ]:
        plt.figure(figsize=(7, 4))
        for color_idx, source in enumerate(sorted({str(row["source"]) for row in geometry_rows})):
            values = [float(row[field]) for row in geometry_rows if row["source"] == source]
            plt.hist(values, bins=35, alpha=0.55, label=source, color=PALETTE[color_idx % len(PALETTE)])
        plt.xlabel(xlabel)
        plt.ylabel("molecules")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(outdir / filename, dpi=180)
        plt.close()

    sources = [str(row["source"]) for row in geometry_summary]
    composition_fields = [
        ("mean_carbon_fraction", "carbon"),
        ("mean_hetero_atom_fraction", "hetero"),
        ("mean_halogen_fraction", "halogen"),
    ]
    x_positions = list(range(len(composition_fields)))
    width = 0.36
    plt.figure(figsize=(7, 4))
    for source_idx, source in enumerate(sources):
        row = next(item for item in geometry_summary if item["source"] == source)
        values = [float(row[field]) for field, _ in composition_fields]
        offsets = [x + (source_idx - (len(sources) - 1) / 2) * width for x in x_positions]
        plt.bar(offsets, values, width=width, label=source, color=PALETTE[source_idx % len(PALETTE)])
    plt.xticks(x_positions, [label for _, label in composition_fields])
    plt.ylabel("fraction")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / "atom_composition_clean_vs_generated.png", dpi=180)
    plt.close()

    if reconstruction_summary:
        methods = [str(row["method"]) for row in reconstruction_summary]
        values = [float(row["mean_bad_bond_length_rate"]) for row in reconstruction_summary]
        plt.figure(figsize=(7, 4))
        plt.bar(methods, values, color=PALETTE[:len(methods)])
        plt.ylabel("bad bond length rate")
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        plt.savefig(outdir / "generated_bad_bond_length_by_method.png", dpi=180)
        plt.close()


def _markdown_table(rows: list[dict[str, object]], columns: list[tuple[str, str]], limit: int | None = None) -> str:
    """Return a compact markdown table."""
    selected_rows = rows[:limit] if limit is not None else rows
    lines = [
        "| " + " | ".join(title for title, _ in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in selected_rows:
        cells = []
        for _, key in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                cells.append(f"{value:.4f}")
            else:
                try:
                    cells.append(f"{float(value):.4f}" if key.startswith("mean_") else str(value))
                except (TypeError, ValueError):
                    cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _write_report(
    *,
    outdir: Path,
    args: argparse.Namespace,
    metadata: dict[str, object],
    clean_noise_summary: list[dict[str, object]],
    geometry_summary: list[dict[str, object]],
    generated_reconstruction_summary: list[dict[str, object]],
) -> Path:
    """Write the top-level diagnostic Markdown report."""
    report_path = outdir / "bond_predictor_diagnostics_report.md"
    clean_rows = [row for row in clean_noise_summary if float(row["noise_std"]) == 0.0]
    with report_path.open("w") as handle:
        handle.write("# Bond Predictor Diagnostics Report\n\n")
        handle.write("## Metadata\n\n")
        for key, value in metadata.items():
            handle.write(f"- {key}: `{value}`\n")

        if clean_rows:
            handle.write("\n## Clean CrossDocked Predictor Test\n\n")
            handle.write(_markdown_table(
                clean_rows,
                [
                    ("method", "method"),
                    ("sanitize", "mean_sanitize_success"),
                    ("edge_f1", "mean_edge_f1"),
                    ("typed_f1", "mean_typed_f1"),
                    ("pred/true", "mean_pred_to_true_bond_ratio"),
                    ("bad_len", "mean_bad_bond_length_rate"),
                    ("sec/mol", "mean_reconstruction_seconds"),
                ],
            ))
            handle.write("\n\n")

        if clean_noise_summary:
            handle.write("## Noise Robustness Sweep\n\n")
            handle.write("![edge F1](edge_f1_by_noise_method.png)\n\n")
            handle.write("![typed F1](typed_f1_by_noise_method.png)\n\n")
            handle.write("![clean bond type confusion](clean_bond_type_confusion_by_method.png)\n\n")
            handle.write("![bad bond length](bad_bond_length_by_noise_method.png)\n\n")
            handle.write("![predicted bond ratio](predicted_bond_ratio_by_noise_method.png)\n\n")
            handle.write("![time](time_by_noise_method.png)\n\n")

        if geometry_summary:
            handle.write("## Generated Point Cloud Geometry\n\n")
            handle.write(_markdown_table(
                geometry_summary,
                [
                    ("source", "source"),
                    ("samples", "samples"),
                    ("atoms", "mean_num_atoms"),
                    ("C frac", "mean_carbon_fraction"),
                    ("hetero frac", "mean_hetero_atom_fraction"),
                    ("NN mean", "mean_nearest_neighbor_mean"),
                    ("clash frac", "mean_clash_pair_fraction"),
                    ("cand/atom", "mean_candidate_bond_pairs_per_atom"),
                    ("true recall", "mean_true_bond_candidate_recall"),
                ],
            ))
            handle.write("\n\n")
            handle.write("![nearest neighbor](nearest_neighbor_clean_vs_generated.png)\n\n")
            handle.write("![candidate bonds](candidate_bond_pairs_clean_vs_generated.png)\n\n")
            handle.write("![clash fraction](clash_fraction_clean_vs_generated.png)\n\n")
            handle.write("![atom composition](atom_composition_clean_vs_generated.png)\n\n")

        if generated_reconstruction_summary:
            handle.write("## Generated Reconstruction\n\n")
            handle.write(_markdown_table(
                generated_reconstruction_summary,
                [
                    ("method", "method"),
                    ("samples", "samples"),
                    ("sanitize", "mean_sanitize_success"),
                    ("bad_len", "mean_bad_bond_length_rate"),
                    ("fragments", "mean_num_fragments"),
                    ("QED", "mean_qed"),
                    ("sec/mol", "mean_reconstruction_seconds"),
                ],
            ))
            handle.write("\n\n")
            handle.write("![generated bad length](generated_bad_bond_length_by_method.png)\n\n")

        handle.write("## Files\n\n")
        handle.write("- [clean_noise_per_molecule.csv](clean_noise_per_molecule.csv)\n")
        handle.write("- [clean_noise_summary.csv](clean_noise_summary.csv)\n")
        handle.write("- [clean_noise_bond_type_confusion.csv](clean_noise_bond_type_confusion.csv)\n")
        handle.write("- [clean_noise_bond_type_metrics.csv](clean_noise_bond_type_metrics.csv)\n")
        handle.write("- [generated_geometry_per_molecule.csv](generated_geometry_per_molecule.csv)\n")
        handle.write("- [generated_geometry_summary.csv](generated_geometry_summary.csv)\n")
        handle.write("- [generated_reconstruction_per_molecule.csv](generated_reconstruction_per_molecule.csv)\n")
        handle.write("- [generated_reconstruction_summary.csv](generated_reconstruction_summary.csv)\n")

    return report_path


def run_diagnostics(args: argparse.Namespace) -> Path:
    """Run all requested diagnostics and write report files."""
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    dataset_info = constants.dataset_params[args.dataset]
    npz_path = args.processed_dir / f"{args.split}.npz"
    base_samples = _load_crossdocked_samples(
        npz_path,
        dataset_info,
        max_samples=args.max_samples,
        seed=args.seed,
        sample_mode=args.sample_mode,
    )

    clean_noise_summary: list[dict[str, object]] = []
    geometry_summary: list[dict[str, object]] = []
    generated_reconstruction_summary: list[dict[str, object]] = []
    if not args.skip_clean_noise:
        _, clean_noise_summary = _run_clean_noise_experiment(
            args=args,
            base_samples=base_samples,
            dataset_info=dataset_info,
            outdir=outdir,
        )
    if not args.skip_generated:
        _, geometry_summary, _, generated_reconstruction_summary = _run_generated_geometry_experiment(
            args=args,
            base_samples=base_samples,
            dataset_info=dataset_info,
            outdir=outdir,
        )

    metadata = {
        "processed_npz": npz_path,
        "split": args.split,
        "crossdocked_samples": len(base_samples),
        "sample_mode": args.sample_mode,
        "seed": args.seed,
        "noise_stds": " ".join(str(item) for item in args.noise_stds),
        "methods": " ".join(args.methods),
        "generated_samples": 0 if args.skip_generated else args.generated_samples,
        "generated_timesteps": args.generated_timesteps,
        "generated_reconstruction_methods": " ".join(args.generated_reconstruction_methods),
        "length_tolerance": args.length_tolerance,
        "clash_threshold": args.clash_threshold,
        "requested_device": args.device,
        "effective_device": _resolve_device(args.device),
        "external_bond_device": args.external_bond_device,
        "local_bond_device": args.local_bond_device,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    return _write_report(
        outdir=outdir,
        args=args,
        metadata=metadata,
        clean_noise_summary=clean_noise_summary,
        geometry_summary=geometry_summary,
        generated_reconstruction_summary=generated_reconstruction_summary,
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=Path("data/crossdocked_raw/processed_edge_noH_full"))
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--dataset", type=str, default="crossdock")
    parser.add_argument("--max-samples", type=int, default=300)
    parser.add_argument("--sample-mode", type=str, choices=["random", "head"], default="random")
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--noise-stds", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.2, 0.3, 0.5])
    parser.add_argument("--methods", type=str, nargs="+", default=[
        "oracle_true_bond",
        "openbabel",
        "edm",
        "diffgui_bond",
        "molsnapper_bond",
        "local_bond_v0",
    ])
    parser.add_argument("--generated-reconstruction-methods", type=str, nargs="+", default=["openbabel", "edm"])
    parser.add_argument("--length-tolerance", type=float, default=0.25)
    parser.add_argument("--clash-threshold", type=float, default=0.70)
    parser.add_argument("--outdir", type=Path, default=Path("analysis") / f"bond_predictor_diagnostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--external-bond-device", type=str, default="cuda:0")
    parser.add_argument("--local-bond-device", type=str, default="cuda:0")
    parser.add_argument("--local-bond-checkpoint", type=Path, default=Path("analysis/local_bond_predictor_v0_20260602_small/best_model.pt"))
    parser.add_argument("--diffgui-env", type=str, default="diffgui")
    parser.add_argument("--molsnapper-env", type=str, default="molsnapper_modern_ref")
    parser.add_argument("--checkpoint", type=Path, default=Path("DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt"))
    parser.add_argument("--pdbfile", type=Path, default=Path("data/8GJL.pdb"))
    parser.add_argument("--resi-list", type=str, nargs="+", default=None)
    parser.add_argument("--ref-ligand", type=str, default="C:1101")
    parser.add_argument("--generated-samples", type=int, default=300)
    parser.add_argument("--generated-batch-size", type=int, default=25)
    parser.add_argument("--generated-timesteps", type=int, default=10)
    parser.add_argument("--num-nodes-lig", type=int, default=None)
    parser.add_argument("--n-nodes-bias", type=int, default=0)
    parser.add_argument("--n-nodes-min", type=int, default=0)
    parser.add_argument("--skip-clean-noise", action="store_true")
    parser.add_argument("--skip-generated", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    report = run_diagnostics(parse_args())
    print(f"Wrote bond predictor diagnostics report to {report}")
