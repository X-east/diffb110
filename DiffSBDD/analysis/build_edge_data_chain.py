"""Build a standalone ligand halfedge bundle for edge diffusion prototypes."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIFFSBDD_ROOT = Path(__file__).resolve().parents[1]
for extra in (PROJECT_ROOT, DIFFSBDD_ROOT):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import constants
from analysis.edge_data_oracle_report import generate_report
from analysis.edge_transition_smoke import run_smoke
from dataset import ProcessedLigandPocketDataset
from edge_utils import BOND_TYPE_NAMES


def _object_array(values: list[np.ndarray | str]) -> np.ndarray:
    """Create a stable object array for variable-sized per-molecule fields."""
    array = np.empty(len(values), dtype=object)
    for idx, value in enumerate(values):
        array[idx] = value
    return array


def _choose_limit(dataset: ProcessedLigandPocketDataset, max_samples: int | None) -> int:
    """Return the number of molecules to materialize."""
    if max_samples is None:
        return len(dataset)
    return min(max_samples, len(dataset))


def _write_edge_bundle(
    dataset: ProcessedLigandPocketDataset,
    dataset_name: str,
    source_npz: Path,
    out_path: Path,
    max_samples: int | None,
) -> dict[str, object]:
    """Write variable-sized ligand/pocket edge arrays into one npz bundle."""
    dataset_info = constants.dataset_params[dataset_name]
    atom_decoder = dataset_info["atom_decoder"]
    limit = _choose_limit(dataset, max_samples)

    names: list[str] = []
    ligand_pos: list[np.ndarray] = []
    ligand_atom_type: list[np.ndarray] = []
    ligand_atom_symbol: list[np.ndarray] = []
    ligand_halfedge_index: list[np.ndarray] = []
    ligand_halfedge_type: list[np.ndarray] = []
    pocket_pos: list[np.ndarray] = []
    pocket_one_hot: list[np.ndarray] = []
    num_atoms: list[int] = []
    num_halfedges: list[int] = []
    num_real_bonds: list[int] = []
    bond_type_counts = {name: 0 for name in BOND_TYPE_NAMES.values()}

    for idx in range(limit):
        sample = dataset[idx]
        atom_type = sample["lig_one_hot"].argmax(dim=-1).cpu().numpy().astype(np.int64)
        atom_symbols = np.asarray([atom_decoder[int(atom_idx)] for atom_idx in atom_type], dtype=object)
        halfedge_type = sample["ligand_halfedge_type"].cpu().numpy().astype(np.int64)

        names.append(str(sample["names"]))
        ligand_pos.append(sample["lig_coords"].cpu().numpy().astype(np.float32))
        ligand_atom_type.append(atom_type)
        ligand_atom_symbol.append(atom_symbols)
        ligand_halfedge_index.append(sample["ligand_halfedge_index"].cpu().numpy().astype(np.int64))
        ligand_halfedge_type.append(halfedge_type)
        pocket_pos.append(sample["pocket_coords"].cpu().numpy().astype(np.float32))
        pocket_one_hot.append(sample["pocket_one_hot"].cpu().numpy().astype(np.float32))
        num_atoms.append(int(sample["num_lig_atoms"]))
        num_halfedges.append(int(sample["num_ligand_halfedges"]))
        num_real_bonds.append(int((sample["ligand_halfedge_type"] > 0).sum().item()))
        for bond_id, bond_name in BOND_TYPE_NAMES.items():
            bond_type_counts[bond_name] += int((sample["ligand_halfedge_type"] == bond_id).sum().item())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        schema_version=np.array("edge_bundle_v1"),
        created_at=np.array(datetime.now().isoformat(timespec="seconds")),
        source_npz=np.array(str(source_npz)),
        dataset=np.array(dataset_name),
        names=_object_array(names),
        ligand_pos=_object_array(ligand_pos),
        ligand_atom_type=_object_array(ligand_atom_type),
        ligand_atom_symbol=_object_array(ligand_atom_symbol),
        ligand_halfedge_index=_object_array(ligand_halfedge_index),
        ligand_halfedge_type=_object_array(ligand_halfedge_type),
        pocket_pos=_object_array(pocket_pos),
        pocket_one_hot=_object_array(pocket_one_hot),
        num_ligand_atoms=np.asarray(num_atoms, dtype=np.int64),
        num_ligand_halfedges=np.asarray(num_halfedges, dtype=np.int64),
        num_ligand_bonds=np.asarray(num_real_bonds, dtype=np.int64),
    )

    return {
        "edge_npz": str(out_path),
        "source_npz": str(source_npz),
        "dataset": dataset_name,
        "molecules": limit,
        "total_atoms": int(sum(num_atoms)),
        "total_halfedges": int(sum(num_halfedges)),
        "total_real_bonds": int(sum(num_real_bonds)),
        "no_bond_fraction": (
            bond_type_counts["no_bond"] / max(1, sum(bond_type_counts.values()))
        ),
        "bond_type_counts": bond_type_counts,
    }


def _write_manifest(path: Path, summary: dict[str, object]) -> None:
    """Write a JSON manifest for the edge bundle."""
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))


def _write_bond_counts(path: Path, summary: dict[str, object]) -> None:
    """Write bond class counts for quick inspection."""
    counts = summary["bond_type_counts"]
    total = max(1, sum(int(value) for value in counts.values()))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["bond_type", "count", "fraction"])
        writer.writeheader()
        for bond_name, count in counts.items():
            writer.writerow({
                "bond_type": bond_name,
                "count": int(count),
                "fraction": f"{int(count) / total:.8f}",
            })


def build_edge_data_chain(args: argparse.Namespace) -> Path:
    """Build the edge bundle, oracle report, oracle SDF, and transition smoke."""
    args.outdir.mkdir(parents=True, exist_ok=True)
    dataset = ProcessedLigandPocketDataset(args.npz_path, center=False)
    edge_npz = args.outdir / args.edge_npz_name

    summary = _write_edge_bundle(
        dataset=dataset,
        dataset_name=args.dataset,
        source_npz=args.npz_path,
        out_path=edge_npz,
        max_samples=args.max_samples,
    )
    _write_manifest(args.outdir / "edge_data_manifest.json", summary)
    _write_bond_counts(args.outdir / "train_edge_bond_type_counts.csv", summary)

    report_args = argparse.Namespace(
        npz_path=args.npz_path,
        dataset=args.dataset,
        outdir=args.outdir,
        max_samples=args.max_samples,
        length_tolerance=args.length_tolerance,
        sdf_path=args.outdir / "oracle_reconstruct.sdf",
    )
    oracle_report = generate_report(report_args)

    smoke_args = argparse.Namespace(
        edge_npz=edge_npz,
        outdir=args.outdir,
        max_molecules=min(args.transition_max_molecules, int(summary["molecules"])),
        num_classes=5,
        num_timesteps=args.num_timesteps,
        seed=args.seed,
    )
    transition_report = run_smoke(smoke_args)

    chain_report = args.outdir / "edge_data_chain_report.md"
    with chain_report.open("w") as handle:
        handle.write("# Edge Data Chain Report\n\n")
        handle.write("## Outputs\n\n")
        handle.write(f"- edge bundle: [{edge_npz.name}]({edge_npz.name})\n")
        handle.write("- oracle SDF: [oracle_reconstruct.sdf](oracle_reconstruct.sdf)\n")
        handle.write(f"- oracle report: [{oracle_report.name}]({oracle_report.name})\n")
        handle.write(f"- transition smoke: [{transition_report.name}]({transition_report.name})\n")
        handle.write("- manifest: [edge_data_manifest.json](edge_data_manifest.json)\n\n")
        handle.write("## Summary\n\n")
        handle.write(f"- source_npz: `{args.npz_path}`\n")
        handle.write(f"- dataset: `{args.dataset}`\n")
        handle.write(f"- molecules: {summary['molecules']}\n")
        handle.write(f"- total_atoms: {summary['total_atoms']}\n")
        handle.write(f"- total_halfedges: {summary['total_halfedges']}\n")
        handle.write(f"- total_real_bonds: {summary['total_real_bonds']}\n")
        handle.write(f"- no_bond_fraction: {summary['no_bond_fraction']:.8f}\n\n")
        handle.write("## Bond Type Counts\n\n")
        handle.write("| bond type | count |\n")
        handle.write("|---|---:|\n")
        for bond_name, count in summary["bond_type_counts"].items():
            handle.write(f"| {bond_name} | {count} |\n")
        handle.write("\n![bond type distribution](bond_type_distribution.png)\n\n")
        handle.write("![bond length distribution](bond_length_distribution.png)\n")

    return chain_report


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--npz-path",
        type=Path,
        default=Path("data/crossdocked_raw/processed_edge_noH_full/train.npz"),
    )
    parser.add_argument("--dataset", choices=sorted(constants.dataset_params.keys()), default="crossdock_full")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("analysis") / f"edge_data_chain_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument("--edge-npz-name", default="train_edge.npz")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--transition-max-molecules", type=int, default=256)
    parser.add_argument("--num-timesteps", type=int, default=1000)
    parser.add_argument("--length-tolerance", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    report = build_edge_data_chain(parse_args())
    print(f"Wrote edge data chain report to {report}")
