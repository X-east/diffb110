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
from generation_refinement import (
    AtomBiasConfig,
    AtomLocalMaskConfig,
    AtomScoreDiagnosticsCollector,
    GeometryRefineConfig,
    atom_bias_tensor,
    build_log_ratio_atom_bias,
    dataset_atom_distribution,
    parse_element_float_map,
    parse_element_tuple,
    refine_ligand_batch_positions,
    write_atom_score_diagnostics,
)
from lightning_modules import LigandPocketDDPM
from reconstruction import DEFAULT_RECONSTRUCTION_METHOD
from reconstruction_benchmark import (
    create_reconstruction_methods,
    evaluate_reconstruction_attempt,
    probe_external_bond_resources,
    write_benchmark_outputs,
)

RDLogger.DisableLog("rdApp.*")

DIFFGUI_PROTEIN_ATOMIC_NUMBERS = ["H", "C", "N", "O", "S", "Se"]
DIFFGUI_AA_NAME_NUMBER = {
    "ALA": 0,
    "CYS": 1,
    "ASP": 2,
    "GLU": 3,
    "PHE": 4,
    "GLY": 5,
    "HIS": 6,
    "ILE": 7,
    "LYS": 8,
    "LEU": 9,
    "MET": 10,
    "ASN": 11,
    "PRO": 12,
    "GLN": 13,
    "ARG": 14,
    "SER": 15,
    "THR": 16,
    "VAL": 17,
    "TRP": 18,
    "TYR": 19,
}
DIFFGUI_BACKBONE_NAMES = {"CA", "C", "N", "O"}


def _diffgui_protein_features(model: LigandPocketDDPM, residues: list[Any]) -> torch.Tensor | None:
    """Build DiffGui-style 28D protein atom features for one full-atom pocket."""
    if model.pocket_representation != "full-atom":
        return None
    features = []
    for residue in residues:
        residue_name = residue.get_resname().upper()
        aa_idx = DIFFGUI_AA_NAME_NUMBER.get(residue_name, len(DIFFGUI_AA_NAME_NUMBER))
        for atom in residue.get_atoms():
            element = atom.element.capitalize()
            if element not in model.pocket_type_encoder and element == "H":
                continue
            element_feature = [float(element == item) for item in DIFFGUI_PROTEIN_ATOMIC_NUMBERS]
            aa_feature = [0.0] * (len(DIFFGUI_AA_NAME_NUMBER) + 1)
            aa_feature[aa_idx] = 1.0
            backbone_feature = [float(atom.get_name() in DIFFGUI_BACKBONE_NAMES)]
            features.append(element_feature + aa_feature + backbone_feature)
    if not features:
        return torch.zeros((0, 28), device=model.device, dtype=constants.FLOAT_TYPE)
    return torch.tensor(features, device=model.device, dtype=constants.FLOAT_TYPE)


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
    pocket = model.prepare_pocket(residues, repeats=repeats)
    diffgui_features = _diffgui_protein_features(model, residues)
    if diffgui_features is not None:
        pocket["diffgui_protein_node"] = diffgui_features.repeat(repeats, 1)
    return pocket


def _json_safe(value: Any) -> Any:
    """Convert paths and tuples in nested metadata into JSON-safe values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


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
    atom_bias_config: AtomBiasConfig | None,
    atom_local_mask_config: AtomLocalMaskConfig | None,
    geometry_refine_config: GeometryRefineConfig | None,
    atom_score_diagnostics: AtomScoreDiagnosticsCollector | None,
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
            if atom_bias_config is not None:
                raise ValueError(
                    "atom_bias_config is only implemented for conditional "
                    "generation, where atom scores are available before argmax."
                )
            if atom_local_mask_config is not None and atom_local_mask_config.enabled:
                raise ValueError(
                    "atom_local_mask_config is only implemented for conditional "
                    "generation, where atom scores are available before argmax."
                )
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
            atom_score_bias = atom_bias_tensor(
                model.lig_type_decoder,
                atom_bias_config,
                device=model.device,
                dtype=constants.FLOAT_TYPE,
            )
            atom_temperature = (
                atom_bias_config.temperature if atom_bias_config is not None else 1.0
            )
            xh_lig, xh_pocket, lig_mask, pocket_mask = model.ddpm.sample_given_pocket(
                pocket,
                node_counts,
                timesteps=timesteps,
                atom_score_bias=atom_score_bias,
                atom_temperature=atom_temperature,
                atom_local_mask_config=atom_local_mask_config,
                atom_decoder=model.lig_type_decoder,
                atom_score_diagnostics=atom_score_diagnostics,
            )
        else:
            raise NotImplementedError(f"Unsupported ddpm type: {type(model.ddpm).__name__}")

        pocket_com_after = scatter_mean(xh_pocket[:, :model.x_dims], pocket_mask, dim=0)
        xh_pocket[:, :model.x_dims] += (pocket_com_before - pocket_com_after)[pocket_mask]
        xh_lig[:, :model.x_dims] += (pocket_com_before - pocket_com_after)[lig_mask]

        atom_type_device = xh_lig[:, model.x_dims:].argmax(1).detach()
        refined_coords, _ = refine_ligand_batch_positions(
            xh_lig[:, :model.x_dims],
            atom_type_device,
            lig_mask,
            model.dataset_info,
            geometry_refine_config,
            pocket_positions=xh_pocket[:, :model.x_dims],
            pocket_mask=pocket_mask,
        )

        coords = refined_coords.detach().cpu()
        atom_types = atom_type_device.cpu()
        lig_mask_cpu = lig_mask.detach().cpu()
        pocket_mask_cpu = pocket_mask.detach().cpu()
        coord_list = utils.batch_to_list(coords, lig_mask_cpu)
        atom_type_list = utils.batch_to_list(atom_types, lig_mask_cpu)
        protein_pos_list = utils.batch_to_list(
            xh_pocket[:, :model.x_dims].detach().cpu(),
            pocket_mask_cpu,
        )
        protein_node = pocket.get("diffgui_protein_node")
        protein_node_list = (
            utils.batch_to_list(protein_node.detach().cpu(), pocket_mask_cpu)
            if protein_node is not None
            else [None] * len(coord_list)
        )
        for local_idx, (sample_coords, sample_atom_types) in enumerate(zip(coord_list, atom_type_list)):
            outputs.append({
                "sample_idx": batch_start + local_idx,
                "coords": sample_coords,
                "atom_types": sample_atom_types.long(),
                "context": {
                    "protein_pos": protein_pos_list[local_idx],
                    "protein_node": protein_node_list[local_idx],
                },
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

    atom_bias = parse_element_float_map(args.atom_bias)
    baseline_fraction = parse_element_float_map(args.atom_prior_baseline)
    if baseline_fraction:
        target_fraction = dataset_atom_distribution(dataset_info)
        prior_bias = build_log_ratio_atom_bias(
            atom_decoder=dataset_info["atom_decoder"],
            target_fraction=target_fraction,
            baseline_fraction=baseline_fraction,
            strength=args.atom_prior_strength,
            clamp=args.atom_bias_clamp,
        )
        for element, value in prior_bias.items():
            atom_bias[element] = atom_bias.get(element, 0.0) + value
    atom_bias_config = None
    if atom_bias or args.atom_temperature != 1.0:
        atom_bias_config = AtomBiasConfig(
            bias_by_element=atom_bias,
            temperature=args.atom_temperature,
        )
    atom_local_mask_config = AtomLocalMaskConfig(
        enabled=args.atom_local_mask,
        candidate_cutoff=args.atom_local_mask_candidate_cutoff,
        terminal_degree_max=args.atom_local_mask_terminal_degree_max,
        terminal_only_elements=parse_element_tuple(args.atom_local_mask_terminal_only_elements),
        disable_elements=parse_element_tuple(args.atom_local_mask_disable_elements),
    )
    geometry_refine_config = GeometryRefineConfig(
        enabled=args.geometry_refine,
        steps=args.geometry_refine_steps,
        learning_rate=args.geometry_refine_lr,
        max_step=args.geometry_refine_max_step,
        max_total_shift=args.geometry_refine_max_total_shift,
        use_pocket_clash=not args.geometry_refine_no_pocket_clash,
    )
    atom_score_diagnostics = None
    if args.atom_score_diagnostics_dir is not None:
        atom_score_diagnostics = AtomScoreDiagnosticsCollector(dataset_info["atom_decoder"])

    method_objects = create_reconstruction_methods(
        args.methods,
        dataset_info,
        project_root=PROJECT_ROOT,
        cache_dir=outdir / "external_bond_predictor_cache",
        external_device=args.external_bond_device,
        local_bond_checkpoint=args.local_bond_checkpoint,
        local_bond_device=args.local_bond_device,
        edge_denoiser_b_checkpoint=args.edge_denoiser_b_checkpoint,
        edge_denoiser_c_checkpoint=args.edge_denoiser_c_checkpoint,
        edge_denoiser_device=args.edge_denoiser_device,
        edge_denoiser_timestep=args.edge_denoiser_timestep,
        edge_denoiser_init_mode=args.edge_denoiser_init_mode,
        edge_denoiser_seed=args.edge_denoiser_seed,
        diffgui_env=args.diffgui_env,
        molsnapper_env=args.molsnapper_env,
    )
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
        atom_bias_config=atom_bias_config,
        atom_local_mask_config=atom_local_mask_config,
        geometry_refine_config=geometry_refine_config,
        atom_score_diagnostics=atom_score_diagnostics,
    )

    rows: list[dict[str, object]] = []
    molecules_by_method: dict[str, list[Chem.Mol]] = defaultdict(list)
    for method in method_objects:
        reconstruct_many = getattr(method, "reconstruct_many", None)
        if callable(reconstruct_many):
            method_attempts = reconstruct_many(point_clouds, dataset_info)
        else:
            method_attempts = [
                (
                    sample,
                    method.reconstruct(
                        sample["coords"],
                        sample["atom_types"],
                        dataset_info,
                        context=sample.get("context"),
                    ),
                )
                for sample in point_clouds
            ]
        for sample, attempt in method_attempts:
            atom_types = sample["atom_types"]
            atom_symbols = [dataset_info["atom_decoder"][int(atom_idx)] for atom_idx in atom_types]
            row, sanitized_mol = evaluate_reconstruction_attempt(
                sample_idx=int(sample["sample_idx"]),
                attempt=attempt,
                coords=sample["coords"],
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
        "local_bond_checkpoint": args.local_bond_checkpoint,
        "local_bond_device": args.local_bond_device,
        "edge_denoiser_b_checkpoint": args.edge_denoiser_b_checkpoint,
        "edge_denoiser_c_checkpoint": args.edge_denoiser_c_checkpoint,
        "edge_denoiser_device": args.edge_denoiser_device,
        "edge_denoiser_timestep": args.edge_denoiser_timestep,
        "edge_denoiser_init_mode": args.edge_denoiser_init_mode,
        "edge_denoiser_seed": args.edge_denoiser_seed,
        "atom_bias": dict(atom_bias_config.bias_by_element) if atom_bias_config is not None else {},
        "atom_temperature": atom_bias_config.temperature if atom_bias_config is not None else 1.0,
        "atom_local_mask": {
            "enabled": atom_local_mask_config.enabled,
            "candidate_cutoff": atom_local_mask_config.candidate_cutoff,
            "terminal_degree_max": atom_local_mask_config.terminal_degree_max,
            "terminal_only_elements": atom_local_mask_config.terminal_only_elements,
            "disable_elements": atom_local_mask_config.disable_elements,
        },
        "geometry_refine": {
            "enabled": geometry_refine_config.enabled,
            "steps": geometry_refine_config.steps,
            "learning_rate": geometry_refine_config.learning_rate,
            "max_step": geometry_refine_config.max_step,
            "max_total_shift": geometry_refine_config.max_total_shift,
            "use_pocket_clash": geometry_refine_config.use_pocket_clash,
        },
    }
    report_path = write_benchmark_outputs(
        outdir=outdir,
        rows=rows,
        method_names=method_names,
        molecules_by_method=molecules_by_method,
        metadata=metadata,
        external_resources=external_resources,
    )
    if atom_score_diagnostics is not None:
        write_atom_score_diagnostics(
            args.atom_score_diagnostics_dir,
            atom_score_diagnostics,
            extra_summary=_json_safe(metadata),
        )
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
    parser.add_argument("--external-bond-device", type=str, default="cpu")
    parser.add_argument(
        "--local-bond-checkpoint",
        type=Path,
        default=Path("analysis/local_bond_predictor_v0_20260602_small/best_model.pt"),
    )
    parser.add_argument("--local-bond-device", type=str, default="cpu")
    parser.add_argument(
        "--edge-denoiser-b-checkpoint",
        type=Path,
        default=Path("analysis/edge_denoiser_ablation_20260603_B_diffgui_noise005/edge_denoiser_overfit.pt"),
    )
    parser.add_argument(
        "--edge-denoiser-c-checkpoint",
        type=Path,
        default=Path("analysis/edge_denoiser_ablation_20260603_C_diffgui_noise005_focal2_gpu/edge_denoiser_overfit.pt"),
    )
    parser.add_argument("--edge-denoiser-device", type=str, default="cpu")
    parser.add_argument("--edge-denoiser-timestep", type=int, default=999)
    parser.add_argument(
        "--edge-denoiser-init-mode",
        choices=["uniform_random", "all_no_bond"],
        default="uniform_random",
    )
    parser.add_argument("--edge-denoiser-seed", type=int, default=20260603)
    parser.add_argument("--diffgui-env", type=str, default="diffgui")
    parser.add_argument("--molsnapper-env", type=str, default="molsnapper_modern_ref")
    parser.add_argument("--atom-bias", type=str, default=None)
    parser.add_argument("--atom-prior-baseline", type=str, default=None)
    parser.add_argument("--atom-prior-strength", type=float, default=0.5)
    parser.add_argument("--atom-bias-clamp", type=float, default=0.5)
    parser.add_argument("--atom-temperature", type=float, default=1.0)
    parser.add_argument("--atom-local-mask", action="store_true")
    parser.add_argument("--atom-local-mask-candidate-cutoff", type=float, default=2.20)
    parser.add_argument("--atom-local-mask-terminal-degree-max", type=int, default=1)
    parser.add_argument("--atom-local-mask-terminal-only-elements", type=str, default="F,Cl,Br,I")
    parser.add_argument("--atom-local-mask-disable-elements", type=str, default="P,Br,I,B")
    parser.add_argument("--atom-score-diagnostics-dir", type=Path, default=None)
    parser.add_argument("--geometry-refine", action="store_true")
    parser.add_argument("--geometry-refine-steps", type=int, default=10)
    parser.add_argument("--geometry-refine-lr", type=float, default=0.05)
    parser.add_argument("--geometry-refine-max-step", type=float, default=0.05)
    parser.add_argument("--geometry-refine-max-total-shift", type=float, default=0.30)
    parser.add_argument("--geometry-refine-no-pocket-clash", action="store_true")
    parser.add_argument("--skip-external-probe", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    report = generate_report(parse_args())
    print(f"Wrote generated reconstruction benchmark report to {report}")
