"""Compare reconstruction backends on the same generated DiffSBDD atom clouds."""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

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
from equivariant_diffusion.conditional_model import ConditionalDDPM
from equivariant_diffusion.en_diffusion import EnVariationalDiffusion
from lightning_modules import LigandPocketDDPM
from reconstruction import DEFAULT_RECONSTRUCTION_METHOD
from reconstruction_benchmark import (
    create_reconstruction_methods,
    evaluate_reconstruction_attempt,
    probe_external_bond_resources,
    write_benchmark_outputs,
)

RDLogger.DisableLog("rdApp.*")


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
    method_objects = create_reconstruction_methods(args.methods, dataset_info)
    method_names = [method.name for method in method_objects]
    external_resources = [] if args.skip_external_probe else probe_external_bond_resources(PROJECT_ROOT)
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
        for method in method_objects:
            attempt = method.reconstruct(coords, atom_types, dataset_info)
            row, sanitized_mol = evaluate_reconstruction_attempt(
                sample_idx=int(sample["sample_idx"]),
                attempt=attempt,
                coords=coords,
                atom_symbols=atom_symbols,
                length_tolerance=args.length_tolerance,
            )
            if sanitized_mol is not None:
                molecules_by_method[attempt.method].append(sanitized_mol)
            rows.append(row)

    metadata = {
        "checkpoint": args.checkpoint,
        "pdbfile": args.pdbfile,
        "ref_ligand": args.ref_ligand,
        "resi_list": args.resi_list,
        "samples": len(point_clouds),
        "timesteps": args.timesteps,
        "length_tolerance": args.length_tolerance,
        "methods": " ".join(method_names),
    }
    return write_benchmark_outputs(
        outdir=outdir,
        rows=rows,
        method_names=method_names,
        molecules_by_method=molecules_by_method,
        metadata=metadata,
        external_resources=external_resources,
    )


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
    parser.add_argument("--skip-external-probe", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    report = generate_report(parse_args())
    print(f"Wrote generated reconstruction benchmark report to {report}")
