"""Compare reconstruction backends on the same generated DiffSBDD atom clouds."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from Bio.PDB import PDBParser
from rdkit import Chem, RDLogger
from torch_scatter import scatter_mean

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIFFSBDD_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_ROOT = Path(__file__).resolve().parent
for extra in (PROJECT_ROOT, DIFFSBDD_ROOT, ANALYSIS_ROOT):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import constants
import utils
from edge_data_oracle_report import _lookup_reference_length
from equivariant_diffusion.conditional_model import ConditionalDDPM
from equivariant_diffusion.en_diffusion import EnVariationalDiffusion
from lightning_modules import LigandPocketDDPM
from reconstruction import DEFAULT_RECONSTRUCTION_METHOD, reconstruct_molecule

RDLogger.DisableLog("rdApp.*")

RDKIT_BOND_TYPE_TO_ID = {
    Chem.BondType.SINGLE: 1,
    Chem.BondType.DOUBLE: 2,
    Chem.BondType.TRIPLE: 3,
    Chem.BondType.AROMATIC: 4,
}


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    """Write rows to CSV with stable columns."""
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _format_float(value: object) -> object:
    """Format floats for stable CSV output."""
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def _safe_sanitize(mol: Chem.Mol | None) -> tuple[Chem.Mol | None, str]:
    """Sanitize a copy of an RDKit molecule."""
    if mol is None:
        return None, "no_mol"
    copied = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(copied)
    except Exception as exc:  # noqa: BLE001 - RDKit exposes multiple exception types.
        return copied, type(exc).__name__
    return copied, ""


def _mol_edges(mol: Chem.Mol | None) -> dict[tuple[int, int], int]:
    """Return undirected molecule bonds as compact bond type ids."""
    if mol is None:
        return {}
    edges: dict[tuple[int, int], int] = {}
    for bond in mol.GetBonds():
        edge_type = RDKIT_BOND_TYPE_TO_ID.get(bond.GetBondType(), 0)
        if edge_type <= 0:
            continue
        atom_i = int(bond.GetBeginAtomIdx())
        atom_j = int(bond.GetEndAtomIdx())
        edges[tuple(sorted((atom_i, atom_j)))] = edge_type
    return edges


def _bond_length_metrics(
    mol: Chem.Mol | None,
    atom_symbols: list[str],
    coords: torch.Tensor,
    tolerance: float,
) -> dict[str, float | int]:
    """Compare inferred bond lengths against element/bond-order references."""
    checked = 0
    bad = 0
    abs_errors: list[float] = []
    for (atom_i, atom_j), edge_type in _mol_edges(mol).items():
        ref_length = _lookup_reference_length(atom_symbols[atom_i], atom_symbols[atom_j], edge_type)
        if ref_length is None:
            continue
        length = float(torch.linalg.norm(coords[atom_i] - coords[atom_j]).item())
        abs_error = abs(length - ref_length)
        checked += 1
        bad += int(abs_error > tolerance)
        abs_errors.append(abs_error)
    return {
        "checked_bond_lengths": checked,
        "bad_bond_lengths": bad,
        "bad_bond_length_rate": bad / max(1, checked),
        "mean_bond_length_abs_error": sum(abs_errors) / len(abs_errors) if abs_errors else 0.0,
    }


def _molecule_metrics(mol: Chem.Mol | None, atom_symbols: list[str]) -> dict[str, float | int | str]:
    """Calculate compact chemistry and topology metrics for one reconstructed molecule."""
    if mol is None or mol.GetNumAtoms() == 0:
        return {
            "num_bonds": 0,
            "num_fragments": 0,
            "largest_fragment_atom_frac": 0.0,
            "ring_count": 0,
            "aromatic_ring_count": 0,
            "three_member_ring_count": 0,
            "carbon_fraction": 0.0,
            "hetero_atom_count": 0,
            "halogen_count": 0,
            "smiles": "",
        }

    fragments = Chem.GetMolFrags(mol, sanitizeFrags=False)
    largest_fragment_atoms = max((len(fragment) for fragment in fragments), default=0)
    ring_info = mol.GetRingInfo()
    atom_rings = list(ring_info.AtomRings())
    aromatic_rings = 0
    for ring in atom_rings:
        if all(mol.GetAtomWithIdx(atom_idx).GetIsAromatic() for atom_idx in ring):
            aromatic_rings += 1
    halogens = {"F", "Cl", "Br", "I"}
    carbon_count = sum(int(symbol == "C") for symbol in atom_symbols)
    hetero_count = sum(int(symbol not in {"C", "H"}) for symbol in atom_symbols)
    halogen_count = sum(int(symbol in halogens) for symbol in atom_symbols)
    try:
        smiles = Chem.MolToSmiles(mol)
    except Exception:  # noqa: BLE001 - SMILES conversion can fail on malformed molecules.
        smiles = ""
    return {
        "num_bonds": int(mol.GetNumBonds()),
        "num_fragments": len(fragments),
        "largest_fragment_atom_frac": largest_fragment_atoms / max(1, mol.GetNumAtoms()),
        "ring_count": len(atom_rings),
        "aromatic_ring_count": aromatic_rings,
        "three_member_ring_count": sum(int(len(ring) == 3) for ring in atom_rings),
        "carbon_fraction": carbon_count / max(1, len(atom_symbols)),
        "hetero_atom_count": hetero_count,
        "halogen_count": halogen_count,
        "smiles": smiles,
    }


def _load_pocket(model: LigandPocketDDPM, pdb_file: Path, pocket_ids: list[str] | None, ref_ligand: str | None, repeats: int) -> dict[str, torch.Tensor]:
    """Load and encode a pocket for repeated ligand generation."""
    pdb_struct = PDBParser(QUIET=True).get_structure("", str(pdb_file))[0]
    if pocket_ids is not None:
        residues = [
            pdb_struct[item.split(":")[0]][(" ", int(item.split(":")[1]), " ")]
            for item in pocket_ids
        ]
    elif ref_ligand is not None:
        residues = utils.get_pocket_from_ligand(pdb_struct, ref_ligand)
    else:
        raise ValueError("Exactly one of pocket_ids or ref_ligand must be provided.")
    return model.prepare_pocket(residues, repeats=repeats)


@torch.no_grad()
def _sample_point_clouds(
    model: LigandPocketDDPM,
    pdb_file: Path,
    *,
    n_samples: int,
    batch_size: int,
    pocket_ids: list[str] | None,
    ref_ligand: str | None,
    timesteps: int | None,
    num_nodes_lig: int | None,
    n_nodes_bias: int,
    n_nodes_min: int,
) -> list[dict[str, Any]]:
    """Sample ligand atom clouds once without reconstructing bonds."""
    model.ddpm.eval()
    outputs: list[dict[str, Any]] = []

    for batch_start in range(0, n_samples, batch_size):
        current_batch = min(batch_size, n_samples - batch_start)
        pocket = _load_pocket(model, pdb_file, pocket_ids, ref_ligand, repeats=current_batch)
        pocket_com_before = scatter_mean(pocket["x"], pocket["mask"], dim=0)

        if num_nodes_lig is None:
            node_counts = model.ddpm.size_distribution.sample_conditional(n1=None, n2=pocket["size"])
        else:
            node_counts = torch.full(
                (current_batch,),
                int(num_nodes_lig),
                device=model.device,
                dtype=constants.INT_TYPE,
            )
        node_counts = torch.clamp(node_counts + int(n_nodes_bias), min=int(n_nodes_min))

        if type(model.ddpm) == EnVariationalDiffusion:
            lig_mask = utils.num_nodes_to_batch_mask(current_batch, node_counts, model.device)
            ligand = {
                "x": torch.zeros((len(lig_mask), model.x_dims), device=model.device, dtype=constants.FLOAT_TYPE),
                "one_hot": torch.zeros((len(lig_mask), model.atom_nf), device=model.device, dtype=constants.FLOAT_TYPE),
                "size": node_counts,
                "mask": lig_mask,
            }
            lig_mask_fixed = torch.zeros(len(lig_mask), device=model.device)
            pocket_mask_fixed = torch.ones(len(pocket["mask"]), device=model.device)
            xh_lig, xh_pocket, lig_mask, pocket_mask = model.ddpm.inpaint(
                ligand,
                pocket,
                lig_mask_fixed,
                pocket_mask_fixed,
                timesteps=timesteps,
            )
        elif type(model.ddpm) == ConditionalDDPM:
            xh_lig, xh_pocket, lig_mask, pocket_mask = model.ddpm.sample_given_pocket(
                pocket,
                node_counts,
                timesteps=timesteps,
            )
        else:
            raise NotImplementedError(f"Unsupported ddpm type: {type(model.ddpm).__name__}")

        pocket_com_after = scatter_mean(xh_pocket[:, :model.x_dims], pocket_mask, dim=0)
        xh_pocket[:, :model.x_dims] += (pocket_com_before - pocket_com_after)[pocket_mask]
        xh_lig[:, :model.x_dims] += (pocket_com_before - pocket_com_after)[lig_mask]

        coords = xh_lig[:, :model.x_dims].detach().cpu()
        atom_types = xh_lig[:, model.x_dims:].argmax(1).detach().cpu()
        lig_mask_cpu = lig_mask.detach().cpu()
        coord_list = utils.batch_to_list(coords, lig_mask_cpu)
        atom_type_list = utils.batch_to_list(atom_types, lig_mask_cpu)
        for local_idx, (sample_coords, sample_atom_types) in enumerate(zip(coord_list, atom_type_list)):
            outputs.append({
                "sample_idx": batch_start + local_idx,
                "coords": sample_coords,
                "atom_types": sample_atom_types.long(),
            })
    return outputs


def _summarize(rows: list[dict[str, object]], methods: list[str]) -> list[dict[str, object]]:
    """Aggregate per-molecule rows by reconstruction method."""
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)

    numeric_fields = [
        "sanitize_success",
        "num_atoms",
        "num_bonds",
        "checked_bond_lengths",
        "bad_bond_lengths",
        "bad_bond_length_rate",
        "mean_bond_length_abs_error",
        "num_fragments",
        "largest_fragment_atom_frac",
        "ring_count",
        "aromatic_ring_count",
        "three_member_ring_count",
        "carbon_fraction",
        "hetero_atom_count",
        "halogen_count",
        "reconstruction_seconds",
    ]
    summary_rows = []
    for method in methods:
        method_rows = grouped[method]
        row: dict[str, object] = {"method": method, "samples": len(method_rows)}
        for field in numeric_fields:
            values = [float(item[field]) for item in method_rows if item[field] != ""]
            row[f"mean_{field}"] = sum(values) / len(values) if values else 0.0
        summary_rows.append(row)
    return summary_rows


def _plot_summary(outdir: Path, summary_rows: list[dict[str, object]]) -> None:
    """Write summary bar plots."""
    methods = [str(row["method"]) for row in summary_rows]
    plot_specs = [
        ("mean_sanitize_success", "sanitize success", "sanitize_success_by_method.png"),
        ("mean_bad_bond_length_rate", "bad bond length rate", "bad_bond_length_by_method.png"),
        ("mean_num_fragments", "fragments", "fragment_count_by_method.png"),
        ("mean_reconstruction_seconds", "seconds / molecule", "time_by_method.png"),
    ]
    for field, ylabel, filename in plot_specs:
        values = [float(row[field]) for row in summary_rows]
        plt.figure(figsize=(7, 4))
        plt.bar(methods, values, color=["#4C78A8", "#F58518", "#54A24B", "#B279A2"][:len(methods)])
        plt.ylabel(ylabel)
        plt.xticks(rotation=15, ha="right")
        plt.ylim(0, max(1.0, max(values) * 1.15 if values else 1.0))
        plt.tight_layout()
        plt.savefig(outdir / filename, dpi=180)
        plt.close()


def _write_sdf_outputs(outdir: Path, molecules_by_method: dict[str, list[Chem.Mol]]) -> None:
    """Write one SDF containing sanitized molecules per reconstruction method."""
    for method, mols in molecules_by_method.items():
        writer = Chem.SDWriter(str(outdir / f"{method}_sanitized.sdf"))
        try:
            for mol in mols:
                writer.write(mol)
        finally:
            writer.close()


def generate_report(args: argparse.Namespace) -> Path:
    """Generate atom clouds once, reconstruct with each backend, and report metrics."""
    if (args.resi_list is None) == (args.ref_ligand is None):
        raise ValueError("Exactly one of --resi-list or --ref-ligand must be set.")

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    if args.seed is not None:
        torch.manual_seed(int(args.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed))

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = LigandPocketDDPM.load_from_checkpoint(args.checkpoint, map_location=device)
    model = model.to(device)
    dataset_info = model.dataset_info

    methods = list(dict.fromkeys(args.methods))
    point_clouds = _sample_point_clouds(
        model,
        args.pdbfile,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        pocket_ids=args.resi_list,
        ref_ligand=args.ref_ligand,
        timesteps=args.timesteps,
        num_nodes_lig=args.num_nodes_lig,
        n_nodes_bias=args.n_nodes_bias,
        n_nodes_min=args.n_nodes_min,
    )

    rows: list[dict[str, object]] = []
    molecules_by_method: dict[str, list[Chem.Mol]] = defaultdict(list)
    for sample in point_clouds:
        coords = sample["coords"]
        atom_types = sample["atom_types"]
        atom_symbols = [dataset_info["atom_decoder"][int(atom_idx)] for atom_idx in atom_types]
        for method in methods:
            start = time.perf_counter()
            build_error = ""
            try:
                raw_mol = reconstruct_molecule(
                    coords,
                    atom_types,
                    dataset_info,
                    method=method,
                    add_coords=True,
                )
            except Exception as exc:  # noqa: BLE001 - reconstruction failures are recorded per method.
                raw_mol = None
                build_error = type(exc).__name__
            sanitized_mol, sanitize_error = _safe_sanitize(raw_mol)
            elapsed = time.perf_counter() - start
            final_error = build_error or sanitize_error
            mol_for_metrics = sanitized_mol if sanitize_error == "" else raw_mol
            if sanitized_mol is not None and final_error == "":
                molecules_by_method[method].append(sanitized_mol)

            row: dict[str, object] = {
                "sample_idx": int(sample["sample_idx"]),
                "method": method,
                "num_atoms": len(atom_symbols),
                "sanitize_success": int(final_error == ""),
                "error": final_error,
                "reconstruction_seconds": elapsed,
            }
            row.update(_bond_length_metrics(mol_for_metrics, atom_symbols, coords, args.length_tolerance))
            row.update(_molecule_metrics(mol_for_metrics, atom_symbols))
            rows.append({key: _format_float(value) for key, value in row.items()})

    per_molecule_fields = [
        "sample_idx",
        "method",
        "num_atoms",
        "sanitize_success",
        "error",
        "num_bonds",
        "checked_bond_lengths",
        "bad_bond_lengths",
        "bad_bond_length_rate",
        "mean_bond_length_abs_error",
        "num_fragments",
        "largest_fragment_atom_frac",
        "ring_count",
        "aromatic_ring_count",
        "three_member_ring_count",
        "carbon_fraction",
        "hetero_atom_count",
        "halogen_count",
        "reconstruction_seconds",
        "smiles",
    ]
    _write_csv(outdir / "generated_reconstruction_per_molecule.csv", rows, per_molecule_fields)

    summary_rows = _summarize(rows, methods)
    summary_fields = ["method", "samples"] + [
        key
        for key in summary_rows[0]
        if key not in {"method", "samples"}
    ]
    _write_csv(
        outdir / "generated_reconstruction_summary.csv",
        [{key: _format_float(value) for key, value in row.items()} for row in summary_rows],
        summary_fields,
    )
    _plot_summary(outdir, summary_rows)
    _write_sdf_outputs(outdir, molecules_by_method)

    report_path = outdir / "generated_reconstruction_ablation_report.md"
    with report_path.open("w") as handle:
        handle.write("# Generated Reconstruction Ablation Report\n\n")
        handle.write(f"- checkpoint: `{args.checkpoint}`\n")
        handle.write(f"- pdbfile: `{args.pdbfile}`\n")
        handle.write(f"- ref_ligand: `{args.ref_ligand}`\n")
        handle.write(f"- resi_list: `{args.resi_list}`\n")
        handle.write(f"- samples: {len(point_clouds)}\n")
        handle.write(f"- timesteps: {args.timesteps}\n")
        handle.write(f"- length_tolerance: {args.length_tolerance}\n\n")
        handle.write("| method | sanitize | bad_bond_length | fragments | aromatic_rings | seconds/mol |\n")
        handle.write("|---|---:|---:|---:|---:|---:|\n")
        for row in summary_rows:
            handle.write(
                f"| {row['method']} "
                f"| {float(row['mean_sanitize_success']):.3f} "
                f"| {float(row['mean_bad_bond_length_rate']):.3f} "
                f"| {float(row['mean_num_fragments']):.3f} "
                f"| {float(row['mean_aromatic_ring_count']):.3f} "
                f"| {float(row['mean_reconstruction_seconds']):.4f} |\n"
            )
        handle.write("\n![sanitize success](sanitize_success_by_method.png)\n\n")
        handle.write("![bad bond length](bad_bond_length_by_method.png)\n\n")
        handle.write("![fragment count](fragment_count_by_method.png)\n\n")
        handle.write("![time](time_by_method.png)\n")

    return report_path


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt"))
    parser.add_argument("--pdbfile", type=Path, default=Path("data/8GJL.pdb"))
    parser.add_argument("--resi-list", type=str, nargs="+", default=None)
    parser.add_argument("--ref-ligand", type=str, default="C:1101")
    parser.add_argument("--outdir", type=Path, default=Path("analysis") / f"generated_reconstruction_ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--timesteps", type=int, default=10)
    parser.add_argument("--num-nodes-lig", type=int, default=None)
    parser.add_argument("--n-nodes-bias", type=int, default=0)
    parser.add_argument("--n-nodes-min", type=int, default=0)
    parser.add_argument("--methods", type=str, nargs="+", default=[DEFAULT_RECONSTRUCTION_METHOD, "edm"])
    parser.add_argument("--length-tolerance", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    report = generate_report(parse_args())
    print(f"Wrote generated reconstruction ablation report to {report}")
