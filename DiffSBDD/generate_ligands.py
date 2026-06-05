import argparse
from pathlib import Path

import torch
from openbabel import openbabel
openbabel.obErrorLog.StopLogging()  # suppress OpenBabel messages

import utils
from generation_refinement import (
    AtomBiasConfig,
    AtomLocalMaskConfig,
    AtomScoreDiagnosticsCollector,
    GeometryRefineConfig,
    build_log_ratio_atom_bias,
    dataset_atom_distribution,
    parse_element_float_map,
    parse_element_tuple,
    write_atom_score_diagnostics,
)
from lightning_modules import LigandPocketDDPM
from reconstruction import DEFAULT_RECONSTRUCTION_METHOD


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--pdbfile', type=str)
    parser.add_argument('--resi_list', type=str, nargs='+', default=None)
    parser.add_argument('--ref_ligand', type=str, default=None)
    parser.add_argument('--outfile', type=Path)
    parser.add_argument('--n_samples', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--num_nodes_lig', type=int, default=None)
    parser.add_argument('--all_frags', action='store_true')
    parser.add_argument('--sanitize', action='store_true')
    parser.add_argument('--relax', action='store_true')
    parser.add_argument('--resamplings', type=int, default=10)
    parser.add_argument('--jump_length', type=int, default=1)
    parser.add_argument('--timesteps', type=int, default=None)
    parser.add_argument(
        '--atom_bias',
        type=str,
        default=None,
        help='Comma-separated atom score bias, e.g. C:-0.10,N:0.05,O:0.05.',
    )
    parser.add_argument(
        '--atom_prior_baseline',
        type=str,
        default=None,
        help='Comma-separated generated baseline fractions used for log-ratio '
             'prior correction, e.g. C:0.792,N:0.08,O:0.10.',
    )
    parser.add_argument('--atom_prior_strength', type=float, default=0.5)
    parser.add_argument('--atom_bias_clamp', type=float, default=0.5)
    parser.add_argument('--atom_temperature', type=float, default=1.0)
    parser.add_argument(
        '--atom_local_mask',
        action='store_true',
        help='Mask locally implausible atom types before final atom argmax.',
    )
    parser.add_argument('--atom_local_mask_candidate_cutoff', type=float, default=2.20)
    parser.add_argument('--atom_local_mask_terminal_degree_max', type=int, default=1)
    parser.add_argument(
        '--atom_local_mask_terminal_only_elements',
        type=str,
        default='F,Cl,Br,I',
        help='Comma-separated elements only allowed at terminal candidate degree.',
    )
    parser.add_argument(
        '--atom_local_mask_disable_elements',
        type=str,
        default='P,Br,I,B',
        help='Comma-separated elements masked globally when local mask is enabled.',
    )
    parser.add_argument(
        '--atom_score_diagnostics_dir',
        type=Path,
        default=None,
        help='Optional directory for raw/corrected atom score diagnostics.',
    )
    parser.add_argument('--geometry_refine', action='store_true')
    parser.add_argument('--geometry_refine_steps', type=int, default=10)
    parser.add_argument('--geometry_refine_lr', type=float, default=0.05)
    parser.add_argument('--geometry_refine_max_step', type=float, default=0.05)
    parser.add_argument('--geometry_refine_max_total_shift', type=float, default=0.30)
    parser.add_argument('--geometry_refine_no_pocket_clash', action='store_true')
    parser.add_argument(
        '--reconstruction_method',
        type=str,
        default=DEFAULT_RECONSTRUCTION_METHOD,
        choices=('openbabel', 'obabel', 'edm', 'distance', 'distance_rule'),
        help='Bond inference backend used when rebuilding RDKit molecules.',
    )
    args = parser.parse_args()

    pdb_id = Path(args.pdbfile).stem

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.batch_size is None:
        args.batch_size = args.n_samples
    assert args.n_samples % args.batch_size == 0

    # Load model
    model = LigandPocketDDPM.load_from_checkpoint(
        args.checkpoint, map_location=device)
    model = model.to(device)

    atom_bias = parse_element_float_map(args.atom_bias)
    baseline_fraction = parse_element_float_map(args.atom_prior_baseline)
    if baseline_fraction:
        target_fraction = dataset_atom_distribution(model.dataset_info)
        prior_bias = build_log_ratio_atom_bias(
            atom_decoder=model.dataset_info['atom_decoder'],
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
        terminal_only_elements=parse_element_tuple(
            args.atom_local_mask_terminal_only_elements),
        disable_elements=parse_element_tuple(
            args.atom_local_mask_disable_elements),
    )

    geometry_refine_config = GeometryRefineConfig(
        enabled=args.geometry_refine,
        steps=args.geometry_refine_steps,
        learning_rate=args.geometry_refine_lr,
        max_step=args.geometry_refine_max_step,
        max_total_shift=args.geometry_refine_max_total_shift,
        use_pocket_clash=not args.geometry_refine_no_pocket_clash,
    )

    if atom_bias_config is not None:
        print(f"Atom bias enabled: {dict(atom_bias_config.bias_by_element)}, "
              f"temperature={atom_bias_config.temperature}")
    if atom_local_mask_config.enabled:
        print(
            "Atom local mask enabled: "
            f"candidate_cutoff={atom_local_mask_config.candidate_cutoff}, "
            f"terminal_degree_max={atom_local_mask_config.terminal_degree_max}, "
            f"terminal_only={atom_local_mask_config.terminal_only_elements}, "
            f"disabled={atom_local_mask_config.disable_elements}"
        )
    if geometry_refine_config.enabled:
        print(f"Geometry refine enabled: steps={geometry_refine_config.steps}, "
              f"lr={geometry_refine_config.learning_rate}, "
              f"max_step={geometry_refine_config.max_step}, "
              f"max_total_shift={geometry_refine_config.max_total_shift}")
    atom_score_diagnostics = None
    if args.atom_score_diagnostics_dir is not None:
        atom_score_diagnostics = AtomScoreDiagnosticsCollector(
            model.dataset_info['atom_decoder'])
        print(f"Atom score diagnostics enabled: {args.atom_score_diagnostics_dir}")

    if args.num_nodes_lig is not None:
        num_nodes_lig = torch.ones(args.n_samples, dtype=int) * \
                        args.num_nodes_lig
    else:
        num_nodes_lig = None

    molecules = []
    for i in range(args.n_samples // args.batch_size):
        molecules_batch = model.generate_ligands(
            args.pdbfile, args.batch_size, args.resi_list, args.ref_ligand,
            num_nodes_lig, args.sanitize, largest_frag=not args.all_frags,
            relax_iter=(200 if args.relax else 0),
            resamplings=args.resamplings, jump_length=args.jump_length,
            timesteps=args.timesteps,
            reconstruction_method=args.reconstruction_method,
            atom_bias_config=atom_bias_config,
            atom_local_mask_config=atom_local_mask_config,
            geometry_refine_config=geometry_refine_config,
            atom_score_diagnostics=atom_score_diagnostics)
        molecules.extend(molecules_batch)

    # Make SDF files
    utils.write_sdf_file(args.outfile, molecules)
    if atom_score_diagnostics is not None:
        extra_summary = {
            "checkpoint": str(args.checkpoint),
            "pdbfile": args.pdbfile,
            "ref_ligand": args.ref_ligand,
            "n_samples": args.n_samples,
            "batch_size": args.batch_size,
            "timesteps": args.timesteps,
            "atom_bias": dict(atom_bias_config.bias_by_element)
            if atom_bias_config is not None else {},
            "atom_temperature": atom_bias_config.temperature
            if atom_bias_config is not None else 1.0,
            "atom_local_mask": {
                "enabled": atom_local_mask_config.enabled,
                "candidate_cutoff": atom_local_mask_config.candidate_cutoff,
                "terminal_degree_max": atom_local_mask_config.terminal_degree_max,
                "terminal_only_elements": atom_local_mask_config.terminal_only_elements,
                "disable_elements": atom_local_mask_config.disable_elements,
            },
            "outfile": str(args.outfile),
        }
        write_atom_score_diagnostics(
            args.atom_score_diagnostics_dir,
            atom_score_diagnostics,
            extra_summary=extra_summary,
        )
