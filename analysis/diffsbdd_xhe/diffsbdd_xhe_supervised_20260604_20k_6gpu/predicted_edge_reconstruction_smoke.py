"""Evaluate DiffSBDD-xhe predicted edges by RDKit reconstruction on val ligands."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from argparse import Namespace
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from rdkit import Chem, RDLogger
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
DIFFSBDD_ROOT = REPO_ROOT / "DiffSBDD"
for import_path in (REPO_ROOT, DIFFSBDD_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from DiffSBDD.edge_diffusion.evaluation import (  # noqa: E402
    compare_halfedge_bond_maps,
    predicted_bond_length_stats,
    smiles_no_stereo,
)
from DiffSBDD.edge_diffusion.oracle_reconstruction import build_oracle_mol  # noqa: E402
from DiffSBDD.edge_utils import BOND_TYPE_NAMES  # noqa: E402
from lightning_modules import LigandPocketDDPM  # noqa: E402

RDLogger.DisableLog("rdApp.*")


EDGE_LABELS = [BOND_TYPE_NAMES[idx] for idx in sorted(BOND_TYPE_NAMES)]


def run_smoke(args: argparse.Namespace) -> Path:
    """Run fixed-timestep predicted-edge reconstruction smoke evaluation."""
    _set_seed(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    device = _choose_device(args.force_cpu)

    config = _load_config(args.config)
    model = _build_model(config)
    load_status = _load_state(model, args.checkpoint, strict=args.strict)
    model.setup("fit")
    model.to(device)
    model.eval()

    dataset = _select_dataset(model, args.split)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=dataset.collate_fn,
        pin_memory=(device.type == "cuda"),
    )
    timesteps = _parse_timesteps(args.timesteps, model.ddpm.T)

    rows: list[dict[str, object]] = []
    confusion_by_t = {
        int(timestep): torch.zeros((len(EDGE_LABELS), len(EDGE_LABELS)), dtype=torch.long)
        for timestep in timesteps
    }
    sdf_writers = _make_sdf_writers(args.outdir, timesteps)
    molecules_seen = 0
    try:
        with torch.no_grad():
            for batch in loader:
                batch_molecules = int(batch["num_lig_atoms"].numel())
                if args.max_molecules is not None:
                    remaining = int(args.max_molecules) - molecules_seen
                    if remaining <= 0:
                        break
                    if batch_molecules > remaining:
                        batch = _trim_batch(dataset, batch, remaining)
                        batch_molecules = remaining

                ligand, pocket = model.get_ligand_and_pocket(batch)
                for timestep in timesteps:
                    generator = torch.Generator(device=device)
                    generator.manual_seed(int(args.seed) + int(timestep) * 1009 + molecules_seen)
                    edge_output = _predict_edges_at_t(
                        model=model,
                        ligand=ligand,
                        pocket=pocket,
                        timestep=int(timestep),
                        generator=generator,
                    )
                    target = edge_output.target_edge_type.detach().cpu().long()
                    pred = edge_output.pred_edge_logits.argmax(dim=-1).detach().cpu().long()
                    noised = edge_output.noised_edge_type.detach().cpu().long()
                    confusion_by_t[int(timestep)] += _confusion(target, pred, len(EDGE_LABELS))
                    batch_rows = _build_batch_rows(
                        batch=batch,
                        timestep=int(timestep),
                        target_edge_type=target,
                        pred_edge_type=pred,
                        noised_edge_type=noised,
                        dataset_info=model.dataset_info,
                        global_start_index=molecules_seen,
                        min_bond_length=args.min_bond_length,
                        max_bond_length=args.max_bond_length,
                    )
                    rows.extend(batch_rows)
                    _write_sdf_rows(
                        writers=sdf_writers[int(timestep)],
                        rows=batch_rows,
                    )
                molecules_seen += batch_molecules
    finally:
        for writer_pair in sdf_writers.values():
            for writer in writer_pair.values():
                writer.close()

    elapsed_seconds = time.time() - started_at
    summary_rows = _summarize_by_t(rows)
    summary = {
        "task": "diffsbdd_xhe_predicted_edge_reconstruction_smoke",
        "scope": "teacher_forced_real_val_point_clouds",
        "decoupled_from_rl": True,
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "device": str(device),
        "seed": int(args.seed),
        "batch_size": int(args.batch_size),
        "molecules": int(molecules_seen),
        "timesteps": timesteps,
        "elapsed_seconds": elapsed_seconds,
        "seconds_per_molecule_per_timestep": elapsed_seconds / max(1, molecules_seen * len(timesteps)),
        "load_status": load_status,
        "summary_by_timestep": summary_rows,
        "confusion_by_timestep": {
            str(timestep): confusion.tolist()
            for timestep, confusion in confusion_by_t.items()
        },
        "files": {
            "rows_csv": "predicted_edge_reconstruction_rows.csv",
            "summary_csv": "predicted_edge_reconstruction_summary.csv",
            "summary_json": "predicted_edge_reconstruction_summary.json",
            "metrics_plot": "predicted_edge_reconstruction_metrics.png",
            "confusion_csv": "predicted_edge_reconstruction_confusion.csv",
        },
    }
    rows_path = args.outdir / "predicted_edge_reconstruction_rows.csv"
    summary_csv_path = args.outdir / "predicted_edge_reconstruction_summary.csv"
    confusion_csv_path = args.outdir / "predicted_edge_reconstruction_confusion.csv"
    summary_json_path = args.outdir / "predicted_edge_reconstruction_summary.json"
    plot_path = args.outdir / "predicted_edge_reconstruction_metrics.png"
    _write_csv(rows_path, rows)
    _write_csv(summary_csv_path, summary_rows)
    _write_confusion_csv(confusion_csv_path, confusion_by_t)
    summary_json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    _plot_summary(plot_path, summary_rows)
    return _write_report(
        outdir=args.outdir,
        summary=summary,
        rows_path=rows_path,
        summary_csv_path=summary_csv_path,
        summary_json_path=summary_json_path,
        confusion_csv_path=confusion_csv_path,
        plot_path=plot_path,
    )


def _namespaceify(value: Any) -> Any:
    """Convert nested dictionaries to argparse namespaces."""
    if isinstance(value, dict):
        return Namespace(**{key: _namespaceify(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_namespaceify(item) for item in value]
    return value


def _load_config(path: Path) -> Namespace:
    """Load a YAML config as a namespace tree."""
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return _namespaceify(config)


def _build_model(args: Namespace) -> LigandPocketDDPM:
    """Instantiate the DiffSBDD Lightning module like train.py."""
    histogram = np.load(Path(args.datadir, "size_distribution.npy")).tolist()
    return LigandPocketDDPM(
        outdir=Path(args.logdir, args.run_name),
        dataset=args.dataset,
        datadir=args.datadir,
        batch_size=args.batch_size,
        lr=args.lr,
        egnn_params=args.egnn_params,
        diffusion_params=args.diffusion_params,
        num_workers=args.num_workers,
        augment_noise=args.augment_noise,
        augment_rotation=args.augment_rotation,
        clip_grad=args.clip_grad,
        eval_epochs=args.eval_epochs,
        eval_params=args.eval_params,
        visualize_sample_epoch=args.visualize_sample_epoch,
        visualize_chain_epoch=args.visualize_chain_epoch,
        auxiliary_loss=args.auxiliary_loss,
        loss_params=args.loss_params,
        mode=args.mode,
        node_histogram=histogram,
        pocket_representation=args.pocket_representation,
        virtual_nodes=args.virtual_nodes,
        diffusion_mode=getattr(args, "diffusion_mode", "xh"),
        bond_diffusion=getattr(args, "bond_diffusion", None),
    )


def _load_state(model: LigandPocketDDPM, checkpoint_path: Path, *, strict: bool) -> dict[str, int]:
    """Load a Lightning checkpoint and return compatibility counts."""
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    state_dict = checkpoint.get("state_dict", checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=strict)
    return {
        "missing": len(incompatible.missing_keys),
        "unexpected": len(incompatible.unexpected_keys),
    }


def _select_dataset(model: LigandPocketDDPM, split: str):
    """Return the requested model dataset split."""
    if split == "train":
        return model.train_dataset
    if split == "val":
        return model.val_dataset
    raise ValueError(f"unsupported split: {split}")


def _trim_batch(dataset, batch: dict[str, Any], keep: int) -> dict[str, Any]:
    """Rebuild a partial batch from the first ``keep`` molecules."""
    names = batch["names"][:keep]
    indices = []
    name_to_index = {str(dataset.data["names"][idx]): idx for idx in range(len(dataset))}
    for name in names:
        indices.append(name_to_index[str(name)])
    return dataset.collate_fn([dataset[idx] for idx in indices])


@torch.no_grad()
def _predict_edges_at_t(
    model: LigandPocketDDPM,
    ligand: dict[str, torch.Tensor],
    pocket: dict[str, torch.Tensor],
    timestep: int,
    generator: torch.Generator,
):
    """Predict clean halfedge logits at one fixed categorical diffusion timestep."""
    ddpm = model.ddpm
    ligand_norm = _clone_tensor_dict(ligand)
    pocket_norm = _clone_tensor_dict(pocket)
    ligand_norm, pocket_norm = ddpm.normalize(ligand_norm, pocket_norm)
    batch_size = int(ligand_norm["size"].shape[0])
    t_int = torch.full(
        (batch_size, 1),
        int(timestep),
        dtype=torch.float32,
        device=ligand_norm["x"].device,
    )
    t = t_int / ddpm.T
    gamma_t = ddpm.inflate_batch_array(ddpm.gamma(t), ligand_norm["x"])

    xh0_lig = torch.cat([ligand_norm["x"], ligand_norm["one_hot"]], dim=1)
    xh0_pocket = torch.cat([pocket_norm["x"], pocket_norm["one_hot"]], dim=1)
    xh0_lig[:, : ddpm.n_dims], xh0_pocket[:, : ddpm.n_dims] = ddpm.remove_mean_batch(
        xh0_lig[:, : ddpm.n_dims],
        xh0_pocket[:, : ddpm.n_dims],
        ligand_norm["mask"],
        pocket_norm["mask"],
    )
    z_t_lig, xh_pocket, _ = ddpm.noised_representation(
        xh0_lig,
        xh0_pocket,
        ligand_norm["mask"],
        pocket_norm["mask"],
        gamma_t,
    )
    target_edge_type = ligand_norm["halfedge_type"].long()
    edge_batch = ligand_norm["edge_batch"].long()
    log_v0 = ddpm.edge_transition.index_to_log_onehot(target_edge_type)
    noised_edge_type, _ = ddpm.edge_transition.q_vt_sample(
        log_v0,
        t_int.reshape(-1).long(),
        edge_batch,
        generator=generator,
    )
    _, _, pred_edge_logits = ddpm.dynamics(
        z_t_lig,
        xh_pocket,
        t,
        ligand_norm["mask"],
        pocket_norm["mask"],
        halfedge_index=ligand_norm["halfedge_index"],
        noised_edge_type=noised_edge_type,
        edge_batch=edge_batch,
        return_edge_logits=True,
    )
    return ddpm._edge_output_from_logits(
        pred_edge_logits=pred_edge_logits,
        noised_edge_type=noised_edge_type,
        target_edge_type=target_edge_type,
        edge_batch=edge_batch,
        batch_size=batch_size,
    )


def _clone_tensor_dict(values: dict[str, Any]) -> dict[str, Any]:
    """Clone tensor values and shallow-copy everything else."""
    return {key: value.clone() if torch.is_tensor(value) else value for key, value in values.items()}


def _build_batch_rows(
    batch: dict[str, Any],
    timestep: int,
    target_edge_type: torch.Tensor,
    pred_edge_type: torch.Tensor,
    noised_edge_type: torch.Tensor,
    dataset_info: dict[str, Any],
    global_start_index: int,
    min_bond_length: float,
    max_bond_length: float,
) -> list[dict[str, object]]:
    """Build per-molecule reconstruction rows from a concatenated DiffSBDD batch."""
    rows: list[dict[str, object]] = []
    atom_decoder = dataset_info["atom_decoder"]
    atom_offset = 0
    edge_offset = 0
    for local_idx, (num_atoms_raw, num_edges_raw) in enumerate(
        zip(batch["num_lig_atoms"].tolist(), batch["num_ligand_halfedges"].tolist())
    ):
        num_atoms = int(num_atoms_raw)
        num_edges = int(num_edges_raw)
        atom_slice = slice(atom_offset, atom_offset + num_atoms)
        edge_slice = slice(edge_offset, edge_offset + num_edges)
        positions = batch["lig_coords"][atom_slice].float().cpu()
        atom_type = batch["lig_one_hot"][atom_slice].argmax(dim=-1).long().cpu()
        atom_symbols = [_decode_atom_symbol(atom_decoder[int(atom_idx)]) for atom_idx in atom_type.tolist()]
        halfedge_index = batch["ligand_halfedge_index"][:, edge_slice].long().cpu() - atom_offset
        target = target_edge_type[edge_slice].long().cpu()
        predicted = pred_edge_type[edge_slice].long().cpu()
        noised = noised_edge_type[edge_slice].long().cpu()

        oracle_result = build_oracle_mol(
            atom_symbols=atom_symbols,
            positions=positions,
            halfedge_index=halfedge_index,
            halfedge_type=target,
        )
        predicted_result = build_oracle_mol(
            atom_symbols=atom_symbols,
            positions=positions,
            halfedge_index=halfedge_index,
            halfedge_type=predicted,
        )
        oracle_smiles = smiles_no_stereo(oracle_result.mol)
        predicted_smiles = smiles_no_stereo(predicted_result.mol)
        bond_compare = compare_halfedge_bond_maps(halfedge_index, target, predicted)
        length_stats = predicted_bond_length_stats(
            atom_symbols=atom_symbols,
            positions=positions,
            halfedge_index=halfedge_index,
            pred_edge_type=predicted,
            min_bond_length=min_bond_length,
            max_bond_length=max_bond_length,
        )
        row = {
            "sample_idx": global_start_index + local_idx,
            "local_batch_idx": local_idx,
            "name": str(batch["names"][local_idx]),
            "timestep": int(timestep),
            "num_atoms": num_atoms,
            "num_halfedges": num_edges,
            "target_bond_count": int((target > 0).sum().item()),
            "pred_bond_count": int((predicted > 0).sum().item()),
            "noised_bond_count": int((noised > 0).sum().item()),
            "halfedge_accuracy": float((target == predicted).float().mean().item()),
            "exact_halfedge_match": int(bool(torch.equal(target, predicted))),
            "noised_change_fraction": float((target != noised).float().mean().item()),
            **bond_compare,
            **length_stats,
            "oracle_sanitize_success": int(oracle_result.sanitize_success),
            "oracle_repair_strategy": oracle_result.repair_strategy,
            "oracle_sanitize_error": oracle_result.sanitize_error or "",
            "oracle_smiles_no_stereo": oracle_smiles,
            "pred_sanitize_success": int(predicted_result.sanitize_success),
            "pred_repair_strategy": predicted_result.repair_strategy,
            "pred_sanitize_error": predicted_result.sanitize_error or "",
            "pred_smiles_no_stereo": predicted_smiles,
            "pred_vs_oracle_smiles_match": int(
                bool(oracle_smiles) and predicted_smiles == oracle_smiles
            ),
            "_oracle_mol": oracle_result.mol,
            "_pred_mol": predicted_result.mol,
            **_rdkit_descriptor_row(predicted_result.mol),
        }
        rows.append(row)
        atom_offset += num_atoms
        edge_offset += num_edges
    return rows


def _rdkit_descriptor_row(mol: Chem.Mol | None) -> dict[str, object]:
    """Return compact RDKit structural diagnostics for a reconstructed molecule."""
    if mol is None:
        return _empty_descriptor_row()
    try:
        fragments = Chem.GetMolFrags(mol)
        ring_info = mol.GetRingInfo()
        atom_rings = ring_info.AtomRings()
        aromatic_rings = sum(
            int(all(mol.GetAtomWithIdx(atom_idx).GetIsAromatic() for atom_idx in ring))
            for ring in atom_rings
        )
        three_member_rings = sum(int(len(ring) == 3) for ring in atom_rings)
        formal_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
        return {
            "pred_num_fragments": len(fragments),
            "pred_num_rings": len(atom_rings),
            "pred_num_aromatic_rings": aromatic_rings,
            "pred_num_three_member_rings": three_member_rings,
            "pred_formal_charge": formal_charge,
            "pred_heavy_atoms": mol.GetNumHeavyAtoms(),
        }
    except Exception:
        return _empty_descriptor_row()


def _empty_descriptor_row() -> dict[str, object]:
    """Return zero-valued descriptor diagnostics."""
    return {
        "pred_num_fragments": 0,
        "pred_num_rings": 0,
        "pred_num_aromatic_rings": 0,
        "pred_num_three_member_rings": 0,
        "pred_formal_charge": 0,
        "pred_heavy_atoms": 0,
    }


def _decode_atom_symbol(symbol: Any) -> str:
    """Normalize atom decoder symbols for RDKit."""
    text = str(symbol)
    if "." in text:
        text = text.split(".", maxsplit=1)[0]
    if "_" in text:
        text = text.split("_", maxsplit=1)[0]
    return text


def _confusion(target: torch.Tensor, pred: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Return target-by-predicted confusion matrix."""
    flat = target.long() * num_classes + pred.long()
    return torch.bincount(flat, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def _make_sdf_writers(outdir: Path, timesteps: list[int]) -> dict[int, dict[str, Chem.SDWriter]]:
    """Create per-timestep oracle and predicted SDF writers."""
    writers = {}
    for timestep in timesteps:
        writers[int(timestep)] = {
            "oracle": Chem.SDWriter(str(outdir / f"oracle_edges_t{int(timestep)}.sdf")),
            "predicted": Chem.SDWriter(str(outdir / f"predicted_edges_t{int(timestep)}.sdf")),
        }
    return writers


def _write_sdf_rows(writers: dict[str, Chem.SDWriter], rows: list[dict[str, object]]) -> None:
    """Write sanitized oracle and predicted molecules from row metadata."""
    for row in rows:
        _write_mol(writers["oracle"], row.get("_oracle_mol"), row, "oracle")
        _write_mol(writers["predicted"], row.get("_pred_mol"), row, "pred")


def _write_mol(
    writer: Chem.SDWriter,
    mol_value: object,
    row: dict[str, object],
    prefix: str,
) -> None:
    """Write a reconstructed coordinate-bearing molecule with row properties."""
    if not isinstance(mol_value, Chem.Mol):
        return
    smiles = str(row.get(f"{prefix}_smiles_no_stereo", ""))
    if not smiles:
        return
    mol = Chem.Mol(mol_value)
    mol.SetProp("_Name", f"{row['sample_idx']}|t{row['timestep']}|{prefix}|{row['name']}")
    for key, value in row.items():
        if key.startswith("_") or key.endswith("smiles_no_stereo") or key == "name":
            continue
        mol.SetProp(str(key), str(value))
    try:
        writer.write(mol)
    except Exception:
        return


def _summarize_by_t(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Aggregate per-molecule rows into one summary row per timestep."""
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["timestep"])].append(row)
    summary_rows = []
    for timestep in sorted(grouped):
        group = grouped[timestep]
        summary_rows.append({
            "timestep": timestep,
            "molecules": len(group),
            "oracle_sanitize_success_rate": _mean_int(group, "oracle_sanitize_success"),
            "pred_sanitize_success_rate": _mean_int(group, "pred_sanitize_success"),
            "pred_vs_oracle_smiles_match_rate": _mean_int(group, "pred_vs_oracle_smiles_match"),
            "exact_halfedge_match_rate": _mean_int(group, "exact_halfedge_match"),
            "mean_halfedge_accuracy": _mean_float(group, "halfedge_accuracy"),
            "mean_bond_pair_precision": _mean_float(group, "bond_pair_precision"),
            "mean_bond_pair_recall": _mean_float(group, "bond_pair_recall"),
            "mean_bond_pair_f1": _mean_float(group, "bond_pair_f1"),
            "mean_bad_bond_length_fraction": _mean_float(group, "bad_bond_length_fraction"),
            "mean_noised_change_fraction": _mean_float(group, "noised_change_fraction"),
            "mean_target_bond_count": _mean_float(group, "target_bond_count"),
            "mean_pred_bond_count": _mean_float(group, "pred_bond_count"),
            "mean_missing_bond_count": _mean_float(group, "missing_bond_count"),
            "mean_extra_bond_count": _mean_float(group, "extra_bond_count"),
            "mean_bond_type_mismatch_count": _mean_float(group, "bond_type_mismatch_count"),
            "mean_pred_fragments": _mean_float(group, "pred_num_fragments"),
            "mean_pred_rings": _mean_float(group, "pred_num_rings"),
            "mean_pred_aromatic_rings": _mean_float(group, "pred_num_aromatic_rings"),
            "mean_pred_three_member_rings": _mean_float(group, "pred_num_three_member_rings"),
        })
    return summary_rows


def _mean_int(rows: list[dict[str, object]], key: str) -> float:
    """Return mean of integer indicator fields."""
    if not rows:
        return 0.0
    return sum(int(row[key]) for row in rows) / len(rows)


def _mean_float(rows: list[dict[str, object]], key: str) -> float:
    """Return mean of numeric fields."""
    if not rows:
        return 0.0
    return float(np.mean([float(row[key]) for row in rows]))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write rows to CSV with stable field discovery."""
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key.startswith("_"):
                continue
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: value for key, value in row.items() if not key.startswith("_")})


def _write_confusion_csv(path: Path, confusion_by_t: dict[int, torch.Tensor]) -> None:
    """Write target-by-predicted confusion counts."""
    rows = []
    for timestep, confusion in sorted(confusion_by_t.items()):
        for target_idx, target_name in enumerate(EDGE_LABELS):
            for pred_idx, pred_name in enumerate(EDGE_LABELS):
                rows.append({
                    "timestep": timestep,
                    "target": target_name,
                    "predicted": pred_name,
                    "count": int(confusion[target_idx, pred_idx].item()),
                })
    _write_csv(path, rows)


def _plot_summary(path: Path, summary_rows: list[dict[str, object]]) -> None:
    """Plot reconstruction metrics across timesteps."""
    timesteps = [int(row["timestep"]) for row in summary_rows]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    metric_sets = [
        ("sanitize/smiles", ["oracle_sanitize_success_rate", "pred_sanitize_success_rate", "pred_vs_oracle_smiles_match_rate"]),
        ("bond recovery", ["mean_bond_pair_precision", "mean_bond_pair_recall", "mean_bond_pair_f1"]),
        ("edge accuracy", ["mean_halfedge_accuracy", "exact_halfedge_match_rate", "mean_noised_change_fraction"]),
        ("structure issues", ["mean_bad_bond_length_fraction", "mean_pred_three_member_rings", "mean_pred_fragments"]),
    ]
    for axis, (title, keys) in zip(axes.flat, metric_sets):
        for key in keys:
            axis.plot(timesteps, [float(row[key]) for row in summary_rows], marker="o", label=key)
        axis.set_title(title)
        axis.set_xlabel("timestep")
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_report(
    outdir: Path,
    summary: dict[str, object],
    rows_path: Path,
    summary_csv_path: Path,
    summary_json_path: Path,
    confusion_csv_path: Path,
    plot_path: Path,
) -> Path:
    """Write the Markdown report for the smoke evaluation."""
    report_path = outdir / "predicted_edge_reconstruction_report.md"
    summary_rows = summary["summary_by_timestep"]
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("# DiffSBDD-xhe Predicted Edge Reconstruction Smoke\n\n")
        handle.write("## 目的\n\n")
        handle.write(
            "本测试验证训练后的 DiffSBDD-xhe edge head 是否能在真实 CrossDocked "
            "val 点云与真实 atom type 条件下，把 noised halfedge type 预测回可重建分子。"
            "这是 teacher-forced reconstruction smoke，不是完整自由采样，也不接入 RL。\n\n"
        )
        handle.write("## 配置\n\n")
        handle.write(f"- checkpoint: `{summary['checkpoint']}`\n")
        handle.write(f"- config: `{summary['config']}`\n")
        handle.write(f"- split: `{summary['split']}`\n")
        handle.write(f"- device: `{summary['device']}`\n")
        handle.write(f"- molecules: {summary['molecules']}\n")
        handle.write(f"- timesteps: `{summary['timesteps']}`\n")
        handle.write(f"- elapsed_seconds: {_format_float(summary['elapsed_seconds'])}\n")
        handle.write(
            "- seconds_per_molecule_per_timestep: "
            f"{_format_float(summary['seconds_per_molecule_per_timestep'])}\n\n"
        )
        handle.write("## 汇总\n\n")
        fields = [
            "timestep",
            "pred_sanitize_success_rate",
            "pred_vs_oracle_smiles_match_rate",
            "mean_bond_pair_f1",
            "mean_halfedge_accuracy",
            "mean_bad_bond_length_fraction",
            "mean_pred_bond_count",
            "mean_target_bond_count",
            "mean_pred_three_member_rings",
        ]
        handle.write("| " + " | ".join(fields) + " |\n")
        handle.write("|" + "|".join(["---:" for _ in fields]) + "|\n")
        for row in summary_rows:
            values = [
                str(row[field]) if field == "timestep" else _format_float(row[field])
                for field in fields
            ]
            handle.write("| " + " | ".join(values) + " |\n")
        handle.write("\n## 解释边界\n\n")
        handle.write(
            "- 如果这里 sanitize/键恢复差，说明 edge head 或 x/h/e 训练还不够；"
            "不能直接进入完整 sampler/RL。\n"
        )
        handle.write(
            "- 如果这里表现好，也只证明真实点云上的 edge denoising/reconstruction 可行；"
            "还需要完整 e sampler 与生成点云测试，才能判断真实生成质量提升。\n\n"
        )
        handle.write("## 文件\n\n")
        for path in [rows_path, summary_csv_path, summary_json_path, confusion_csv_path, plot_path]:
            handle.write(f"- [{path.name}]({path.name})\n")
        for timestep in summary["timesteps"]:
            handle.write(f"- [oracle_edges_t{timestep}.sdf](oracle_edges_t{timestep}.sdf)\n")
            handle.write(f"- [predicted_edges_t{timestep}.sdf](predicted_edges_t{timestep}.sdf)\n")
        handle.write(f"\n![metrics]({plot_path.name})\n")
    return report_path


def _parse_timesteps(raw: str, max_timestep: int) -> list[int]:
    """Parse comma-separated timesteps and validate range."""
    timesteps = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not timesteps:
        raise ValueError("timesteps must not be empty")
    for timestep in timesteps:
        if timestep < 0 or timestep > max_timestep:
            raise ValueError(f"timestep must be within [0, {max_timestep}], got {timestep}")
    return timesteps


def _choose_device(force_cpu: bool) -> torch.device:
    """Choose CUDA when available unless CPU is forced."""
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _set_seed(seed: int) -> None:
    """Seed Python, NumPy, and torch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _format_float(value: object) -> str:
    """Format a scalar for Markdown."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "nan"
    return f"{number:.6f}"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("analysis/diffsbdd_xhe_supervised_20260604_20k_6gpu/diffsbdd_xhe_20k_6gpu_b4_3ep.yaml"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "analysis/diffsbdd_xhe_supervised_20260604_20k_6gpu/"
            "lightning/diffsbdd-xhe-20k-6gpu-260604-b4-3ep/checkpoints/last.ckpt"
        ),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("analysis/diffsbdd_xhe_predicted_reconstruction_20260605_val"),
    )
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-molecules", type=int, default=None)
    parser.add_argument("--timesteps", default="0,10,50,100,250,500")
    parser.add_argument("--min-bond-length", type=float, default=0.7)
    parser.add_argument("--max-bond-length", type=float, default=2.1)
    parser.add_argument("--seed", type=int, default=260605)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--strict", action="store_true", default=True)
    parser.add_argument("--no-strict", dest="strict", action="store_false")
    return parser.parse_args()


if __name__ == "__main__":
    report = run_smoke(parse_args())
    print(f"Wrote predicted-edge reconstruction smoke report to {report}")
