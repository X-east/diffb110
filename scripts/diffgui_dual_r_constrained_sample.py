"""Run inference-only DiffGui dual-R constrained fragment sampling."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIFFGUI_ROOT = PROJECT_ROOT / "reference project" / "DiffGui"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(DIFFGUI_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFGUI_ROOT))

_OPENBABEL_LIBDIR = Path(sys.prefix) / "lib" / "openbabel" / "3.1.0"
if _OPENBABEL_LIBDIR.is_dir():
    os.environ.setdefault("BABEL_LIBDIR", str(_OPENBABEL_LIBDIR))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from rdkit import Chem  # noqa: E402
from torch_geometric.data import Batch  # noqa: E402
from torch_geometric.transforms import Compose  # noqa: E402

from ddpo.constraints.dual_sidechain import DualSubstitutionSpec, audit_dual_sidechain  # noqa: E402
from ddpo.diffgui.constraints import (  # noqa: E402
    DiffGuiDualRSlotLayout,
    apply_dual_r_constraints_to_log_probs,
    apply_dual_r_constraints_to_types,
    batch_diffgui_dual_r_constraint_specs,
    force_onehot_from_types,
)
from ddpo.diffgui.runtime.timestep_config import force_diff_config_num_timesteps, load_diffgui_network_state  # noqa: E402
from models.diffusion import index_to_log_onehot, log_sample_categorical  # noqa: E402
from models.model import DiffGui  # noqa: E402
from utils.data import PDBProtein, parse_lig_file  # noqa: E402
from utils.dataset import ProteinLigandData, to_torch_dict  # noqa: E402
from utils.misc import load_config as load_diffgui_config  # noqa: E402
from utils.misc import seed_all  # noqa: E402
from utils.reconstruct import MolReconsError, reconstruct_from_generated_with_edges  # noqa: E402
from utils.sample_utils import seperate_outputs  # noqa: E402
from utils.transforms import FeatureComplexWithFrag, make_data_placeholder  # noqa: E402


LOGGER = logging.getLogger("diffgui_dual_r_constrained_sample")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p0-dir", required=True, type=Path, help="P0 input directory.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory.")
    parser.add_argument("--sample-config", type=Path, default=DIFFGUI_ROOT / "configs/sample/sample.yml")
    parser.add_argument("--checkpoint", type=Path, default=DIFFGUI_ROOT / "ckpt/trained.pt")
    parser.add_argument("--device", default="cuda:0", help="Torch device, e.g. cuda:2.")
    parser.add_argument("--num-graphs", type=int, default=2, help="Number of molecules to sample.")
    parser.add_argument("--chunk-size", type=int, default=0, help="Optional sequential sampling chunk size to reduce GPU memory.")
    parser.add_argument("--num-timesteps", type=int, default=20, help="Diffusion steps for smoke inference.")
    parser.add_argument("--force-r-chain", action="store_true", help="Force consecutive R slots on each side to single bonds.")
    parser.add_argument("--r-chain-bond-type", type=int, default=1, help="Bond type id used by --force-r-chain.")
    parser.add_argument(
        "--drop-disconnected-r-slots",
        action="store_true",
        help="Post-process generated molecules by deleting R slots disconnected from the fixed core before audit.",
    )
    parser.add_argument("--seed", type=int, default=260618)
    parser.add_argument("--gui-strength", type=float, default=0.0)
    parser.add_argument("--name-prefix", default="diffgui_dual_r")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> int:
    """Run constrained fragment inference and write SDF/CSV/debug artifacts."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(args.output_dir / "run.log", str(args.log_level))
    LOGGER.info("starting DiffGui dual-R constrained sample args=%s", vars(args))

    if args.num_graphs <= 0 or args.num_timesteps <= 0:
        raise ValueError("--num-graphs and --num-timesteps must be positive.")
    seed_all(int(args.seed))
    device = torch.device(str(args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false in this process.")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    p0 = _load_p0(args.p0_dir)
    model, featurizer, sample_cfg = _load_model(
        sample_config=args.sample_config,
        checkpoint=args.checkpoint,
        num_timesteps=int(args.num_timesteps),
        device=device,
    )
    rows, written = _run_sampling_chunks(
        args=args,
        p0=p0,
        model=model,
        featurizer=featurizer,
        sample_cfg=sample_cfg,
        device=device,
    )
    summary = _write_summary(args.output_dir, args=args, rows=rows, written=written, p0=p0)
    LOGGER.info("completed constrained sample written=%d summary=%s", written, summary)
    return 0


def _run_sampling_chunks(
    *,
    args: argparse.Namespace,
    p0: dict[str, Any],
    model: torch.nn.Module,
    featurizer: Any,
    sample_cfg: Any,
    device: torch.device,
) -> tuple[list[dict[str, Any]], int]:
    """Run one or more sequential sampling chunks and merge output artifacts."""
    total_graphs = int(args.num_graphs)
    chunk_size = int(args.chunk_size or total_graphs)
    if chunk_size <= 0:
        raise ValueError("--chunk-size must be positive when provided.")
    all_rows: list[dict[str, Any]] = []
    all_mols: list[Chem.Mol] = []
    all_node_debug: list[dict[str, Any]] = []
    all_halfedge_debug: list[dict[str, Any]] = []
    offset = 0
    chunk_index = 0
    while offset < total_graphs:
        current = min(chunk_size, total_graphs - offset)
        LOGGER.info("sampling chunk index=%d offset=%d size=%d total=%d", chunk_index, offset, current, total_graphs)
        batch, holder, batch_lab = _build_batch(
            p0=p0,
            featurizer=featurizer,
            sample_cfg=sample_cfg,
            num_graphs=current,
            device=device,
        )
        specs = batch_diffgui_dual_r_constraint_specs(
            layout=p0["layout"],
            halfedge_index=holder["halfedge_index"],
            num_graphs=current,
            num_node_types=featurizer.atom_feat_dim,
            num_edge_types=featurizer.bond_feat_dim,
            core_bond_types=p0["core_bond_types"],
            attachment_bond_type=int(p0["attachment_bond_type"]),
            allowed_node_types=tuple(p0["allowed_node_types"]),
            require_r_chain_edges=bool(args.force_r_chain),
            r_chain_bond_type=int(args.r_chain_bond_type),
        )
        outputs = constrained_sample_frag(
            model=model,
            n_graphs=current,
            protein_node=batch.protein_atom_feat.float(),
            protein_pos=batch.protein_pos,
            protein_batch=batch.protein_element_batch,
            frag_node=batch.frag_atom_feat_full,
            frag_pos=batch.frag_pos,
            frag_batch=batch.frag_element_batch,
            ligand_batch=holder["batch_node"],
            halfedge_index=holder["halfedge_index"],
            halfedge_batch=holder["batch_halfedge"],
            specs=specs,
            core_size=int(p0["layout"].core_size),
            batch_lab=batch_lab,
            gui_strength=float(args.gui_strength),
        )
        chunk_rows, chunk_mols = _decode_reconstruct_and_audit(
            outputs=outputs,
            featurizer=featurizer,
            batch_node=holder["batch_node"].detach().cpu().numpy(),
            halfedge_index=holder["halfedge_index"].detach().cpu().numpy(),
            batch_halfedge=holder["batch_halfedge"].detach().cpu().numpy(),
            n_graphs=current,
            name_prefix=str(args.name_prefix),
            audit_spec=p0["audit_spec"],
            layout=p0["layout"],
            mol_index_offset=offset,
            drop_disconnected_r_slots=bool(args.drop_disconnected_r_slots),
        )
        all_rows.extend(chunk_rows)
        all_mols.extend(chunk_mols)
        all_node_debug.extend(_debug_type_rows(outputs["final_types"]["node"], "node", chunk_index, offset))
        all_halfedge_debug.extend(_debug_type_rows(outputs["final_types"]["halfedge"], "edge", chunk_index, offset))
        offset += current
        chunk_index += 1
        if device.type == "cuda":
            torch.cuda.empty_cache()
    written = _write_molecules(args.output_dir / "generated.sdf", all_mols)
    _write_csv(args.output_dir / "reconstruction_audit.csv", all_rows)
    _write_csv(args.output_dir / "final_node_types.csv", all_node_debug)
    _write_csv(args.output_dir / "final_halfedge_types.csv", all_halfedge_debug)
    return all_rows, written


@torch.no_grad()
def constrained_sample_frag(
    *,
    model: torch.nn.Module,
    n_graphs: int,
    protein_node: torch.Tensor,
    protein_pos: torch.Tensor,
    protein_batch: torch.Tensor,
    frag_node: torch.Tensor,
    frag_pos: torch.Tensor,
    frag_batch: torch.Tensor,
    ligand_batch: torch.Tensor,
    halfedge_index: torch.Tensor,
    halfedge_batch: torch.Tensor,
    specs,
    core_size: int,
    batch_lab: torch.Tensor | None,
    gui_strength: float,
) -> dict[str, Any]:
    """Sample DiffGui states while enforcing fixed-core and paired-R constraints."""
    core_model = getattr(model, "module", model)
    if getattr(core_model, "categorical_space", None) != "discrete":
        raise ValueError("constrained_sample_frag currently requires discrete DiffGui categorical space.")
    device = ligand_batch.device
    n_nodes = int(ligand_batch.shape[0])
    n_halfedges = int(halfedge_batch.shape[0])
    _, ligand_node_h, node_log_state = core_model.node_transition.sample_init(n_nodes)
    _, ligand_halfedge_h, halfedge_log_state = core_model.edge_transition.sample_init(n_halfedges)

    pocket_center_pos = _scatter_mean(protein_pos, protein_batch)
    ligand_center_pos = pocket_center_pos[ligand_batch]
    ligand_pos_init = ligand_center_pos + torch.randn_like(ligand_center_pos)
    protein_pos_centered, ligand_pos, offset = _center_pos(
        protein_pos,
        ligand_pos_init,
        protein_batch,
        ligand_batch,
        mode=core_model.center_pos_mode,
    )
    frag_pos_centered = frag_pos - offset[frag_batch]

    frag_node_mask = _core_node_mask(ligand_batch, core_size=core_size)
    frag_node_type = frag_node.long()
    frag_node_onehot = core_model.node_transition.onehot_encode(frag_node_type)
    frag_node_log = index_to_log_onehot(frag_node_type, core_model.ligand_node_types)

    num_timesteps = int(core_model.num_timesteps)
    ligand_node_traj = torch.zeros(num_timesteps + 1, n_nodes, ligand_node_h.shape[-1], device=device, dtype=ligand_node_h.dtype)
    ligand_pos_traj = torch.zeros(num_timesteps + 1, n_nodes, 3, device=device, dtype=ligand_pos.dtype)
    ligand_halfedge_traj = torch.zeros(num_timesteps + 1, n_halfedges, ligand_halfedge_h.shape[-1], device=device, dtype=ligand_halfedge_h.dtype)
    ligand_edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=1)
    ligand_edge_batch = torch.cat([halfedge_batch, halfedge_batch], dim=0)

    ligand_pos[frag_node_mask] = frag_pos_centered
    ligand_node_h[frag_node_mask] = frag_node_onehot
    node_log_state[frag_node_mask] = frag_node_log
    node_type = ligand_node_h.argmax(dim=-1)
    halfedge_type = ligand_halfedge_h.argmax(dim=-1)
    node_type, halfedge_type = _apply_specs_to_types(node_type, halfedge_type, specs)
    ligand_node_h = force_onehot_from_types(node_type, core_model.ligand_node_types).to(device=device, dtype=ligand_node_h.dtype)
    ligand_halfedge_h = force_onehot_from_types(halfedge_type, core_model.num_edge_types).to(device=device, dtype=ligand_halfedge_h.dtype)
    ligand_node_traj[0] = ligand_node_h
    ligand_pos_traj[0] = ligand_pos + offset[ligand_batch]
    ligand_halfedge_traj[0] = ligand_halfedge_h

    pred_pos = pred_node = pred_halfedge = None
    for traj_idx, step in enumerate(range(num_timesteps)[::-1]):
        time_step = torch.full((n_graphs,), int(step), dtype=torch.long, device=device)
        ligand_pos[frag_node_mask] = frag_pos_centered
        ligand_node_h[frag_node_mask] = frag_node_onehot
        node_log_state[frag_node_mask] = frag_node_log

        ligand_edge_h = torch.cat([ligand_halfedge_h, ligand_halfedge_h], dim=0)
        pred_pos, pred_node, pred_halfedge = _predict_step(
            model,
            protein_node=protein_node,
            protein_pos=protein_pos_centered,
            protein_batch=protein_batch,
            ligand_node_pert=ligand_node_h,
            ligand_pos_pert=ligand_pos,
            ligand_batch=ligand_batch,
            ligand_edge_pert=ligand_edge_h,
            ligand_edge_index=ligand_edge_index,
            ligand_edge_batch=ligand_edge_batch,
            time_step=time_step,
            batch_lab=batch_lab,
            gui_strength=gui_strength,
        )
        ligand_pos_prev = core_model.pos_transition.get_prev_from_recon(
            x_t=ligand_pos,
            x_recon=pred_pos,
            t=time_step,
            batch=ligand_batch,
        )
        node_log_recon = F.log_softmax(pred_node, dim=-1)
        node_log_state = core_model.node_transition.q_v_posterior(
            node_log_recon,
            node_log_state,
            time_step,
            ligand_batch,
            v0_prob=True,
        )
        halfedge_log_recon = F.log_softmax(pred_halfedge, dim=-1)
        halfedge_log_state = core_model.edge_transition.q_v_posterior(
            halfedge_log_recon,
            halfedge_log_state,
            time_step,
            halfedge_batch,
            v0_prob=True,
        )
        node_log_state, halfedge_log_state = _apply_specs_to_log_probs(node_log_state, halfedge_log_state, specs)
        node_log_state[frag_node_mask] = frag_node_log

        node_type = log_sample_categorical(node_log_state)
        halfedge_type = log_sample_categorical(halfedge_log_state)
        node_type, halfedge_type = _apply_specs_to_types(node_type, halfedge_type, specs)
        node_type[frag_node_mask] = frag_node_type

        ligand_node_h = core_model.node_transition.onehot_encode(node_type)
        ligand_halfedge_h = core_model.edge_transition.onehot_encode(halfedge_type)
        ligand_pos = ligand_pos_prev
        ligand_pos[frag_node_mask] = frag_pos_centered

        write_idx = traj_idx + 1
        ligand_node_traj[write_idx] = ligand_node_h
        ligand_pos_traj[write_idx] = ligand_pos + offset[ligand_batch]
        ligand_halfedge_traj[write_idx] = ligand_halfedge_h

    final_pos = ligand_pos + offset[ligand_batch]
    return {
        "pred": [ligand_node_h, final_pos, ligand_halfedge_h],
        "traj": [ligand_node_traj, ligand_pos_traj, ligand_halfedge_traj],
        "final_types": {
            "node": ligand_node_h.argmax(dim=-1),
            "halfedge": ligand_halfedge_h.argmax(dim=-1),
        },
    }


def _predict_step(
    model: torch.nn.Module,
    *,
    protein_node: torch.Tensor,
    protein_pos: torch.Tensor,
    protein_batch: torch.Tensor,
    ligand_node_pert: torch.Tensor,
    ligand_pos_pert: torch.Tensor,
    ligand_batch: torch.Tensor,
    ligand_edge_pert: torch.Tensor,
    ligand_edge_index: torch.Tensor,
    ligand_edge_batch: torch.Tensor,
    time_step: torch.Tensor,
    batch_lab: torch.Tensor | None,
    gui_strength: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the same prediction branch as DiffGui sample/sample_frag."""
    core_model = getattr(model, "module", model)
    train_mode = str(core_model.config.train_mode)
    if train_mode in ("ori", "no_bond"):
        return model.classifier_free(
            protein_node,
            protein_pos,
            protein_batch,
            ligand_node_pert,
            ligand_pos_pert,
            ligand_batch,
            ligand_edge_pert,
            ligand_edge_index,
            ligand_edge_batch,
            float(gui_strength),
            time_step,
            batch_lab,
        )
    if train_mode in ("no_lab", "no_both"):
        preds = model(
            protein_node,
            protein_pos,
            protein_batch,
            ligand_node_pert,
            ligand_pos_pert,
            ligand_batch,
            ligand_edge_pert,
            ligand_edge_index,
            ligand_edge_batch,
            time_step,
            batch_lab,
        )
        return preds["pred_ligand_pos"], preds["pred_ligand_node"], preds["pred_ligand_halfedge"]
    raise ValueError(f"Unsupported DiffGui train_mode={train_mode!r}.")


def _load_p0(p0_dir: Path) -> dict[str, Any]:
    """Load P0 JSON/CSV/SDF artifacts."""
    layout_payload = _read_json(p0_dir / "slot_layout.json")
    constraint_payload = _read_json(p0_dir / "constraint_spec.json")
    site_payload = _read_json(p0_dir / "site_spec.json")
    layout = DiffGuiDualRSlotLayout(
        core_size=int(layout_payload["core_size"]),
        left_anchor_idx=int(layout_payload["left_anchor_idx"]),
        right_anchor_idx=int(layout_payload["right_anchor_idx"]),
        r_slots_per_side=int(layout_payload["r_slots_per_side"]),
        left_r_indices=tuple(int(value) for value in layout_payload["left_r_indices"]),
        right_r_indices=tuple(int(value) for value in layout_payload["right_r_indices"]),
    )
    core_bond_types: dict[tuple[int, int], int] = {}
    attachment_bond_type = 1
    with (p0_dir / "halfedge_role_table.csv").open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            role = str(row["role"])
            if role == "required_core_bond":
                core_bond_types[(int(row["node_i"]), int(row["node_j"]))] = int(row["forced_type"])
            elif role.startswith("required_attachment"):
                attachment_bond_type = int(row["forced_type"])
    audit_spec_payload = site_payload["adjusted_spec_after_leaving_group_removal"]
    return {
        "p0_dir": p0_dir,
        "layout": layout,
        "constraint_payload": constraint_payload,
        "site_payload": site_payload,
        "core_bond_types": core_bond_types,
        "attachment_bond_type": attachment_bond_type,
        "allowed_node_types": tuple(int(value) for value in constraint_payload.get("allowed_node_types", [])),
        "audit_spec": DualSubstitutionSpec(
            left_core_idx=int(audit_spec_payload["left_core_idx"]),
            right_core_idx=int(audit_spec_payload["right_core_idx"]),
            left_leaving_atom_idx=None,
            right_leaving_atom_idx=None,
            max_branch_atoms=audit_spec_payload.get("max_branch_atoms"),
        ),
    }


def _load_model(
    *,
    sample_config: Path,
    checkpoint: Path,
    num_timesteps: int,
    device: torch.device,
) -> tuple[torch.nn.Module, Any, Any]:
    """Load reference DiffGui checkpoint."""
    sample_cfg = load_diffgui_config(sample_config)
    ckpt = torch.load(str(checkpoint), map_location=device)
    train_cfg = ckpt["config"]
    force_diff_config_num_timesteps(train_cfg.model.diff, int(num_timesteps))
    featurizer = FeatureComplexWithFrag(train_cfg.data.transform.ligand_atom_mode, sample=True)
    model = DiffGui(
        config=train_cfg.model,
        protein_node_types=featurizer.protein_feat_dim,
        ligand_node_types=featurizer.atom_feat_dim,
        num_edge_types=featurizer.bond_feat_dim,
    ).to(device)
    load_report = load_diffgui_network_state(model, ckpt["model"])
    model.eval()
    LOGGER.info("loaded DiffGui checkpoint=%s iteration=%s load_report=%s", checkpoint, ckpt.get("iteration"), load_report)
    return model, featurizer, sample_cfg


def _build_batch(
    *,
    p0: dict[str, Any],
    featurizer: Any,
    sample_cfg: Any,
    num_graphs: int,
    device: torch.device,
) -> tuple[Any, dict[str, torch.Tensor], torch.Tensor]:
    """Build protein/fragment batch and ligand placeholder."""
    site = p0["site_payload"]
    protein_dict = PDBProtein(site["protein_pdb"]).to_dict_atom()
    empty_ligand = {
        "element": torch.empty([0], dtype=torch.long),
        "hybridization": torch.empty([0], dtype=torch.long),
        "pos": torch.empty([0, 3], dtype=torch.float32),
        "bond_index": torch.empty([2, 0], dtype=torch.long),
        "bond_type": torch.empty([0], dtype=torch.long),
        "atom_feature": torch.empty([0, 8], dtype=torch.float32),
    }
    frag_dict = parse_lig_file(str(p0["p0_dir"] / "core_fragment.sdf"))
    data = ProteinLigandData.protein_ligand_dicts(
        protein_dict=to_torch_dict(protein_dict),
        ligand_dict=empty_ligand,
        frag_dict=to_torch_dict(frag_dict),
    )
    data = Compose([featurizer])(data)
    batch = Batch.from_data_list([data.clone() for _ in range(num_graphs)], follow_batch=featurizer.follow_batch).to(device)
    layout = p0["layout"]
    holder = make_data_placeholder([layout.total_nodes] * num_graphs, device=device)
    batch_lab = _make_batch_lab(sample_cfg, num_graphs, device)
    return batch, holder, batch_lab


def _decode_reconstruct_and_audit(
    *,
    outputs: dict[str, Any],
    featurizer: Any,
    batch_node: np.ndarray,
    halfedge_index: np.ndarray,
    batch_halfedge: np.ndarray,
    n_graphs: int,
    name_prefix: str,
    audit_spec: DualSubstitutionSpec,
    layout: DiffGuiDualRSlotLayout,
    mol_index_offset: int = 0,
    drop_disconnected_r_slots: bool = False,
) -> tuple[list[dict[str, Any]], list[Chem.Mol]]:
    """Decode final states, reconstruct molecules, and audit dual-R facts."""
    outputs_np = {
        "pred": [value.detach().cpu().numpy() for value in outputs["pred"]],
        "traj": [value.detach().cpu().numpy() for value in outputs["traj"]],
    }
    split_outputs = seperate_outputs(outputs_np, n_graphs, batch_node, halfedge_index, batch_halfedge)
    rows: list[dict[str, Any]] = []
    mols: list[Chem.Mol] = []
    for local_index, output_mol in enumerate(split_outputs):
        mol_index = int(mol_index_offset) + local_index
        row: dict[str, Any] = {
            "mol_index": mol_index,
            "reconstruction_status": "ok",
            "smiles": "",
            "audit_failure_type": "",
            "dual_success": 0,
        }
        try:
            mol_info = featurizer.decode_output(
                pred_node=output_mol["pred"][0],
                pred_pos=output_mol["pred"][1],
                pred_halfedge=output_mol["pred"][2],
                halfedge_index=output_mol["halfedge_index"],
            )
            rdmol = reconstruct_from_generated_with_edges(mol_info, add_edge=None)
            raw_slot_audit = _strict_slot_audit(rdmol, layout)
            audit_layout = layout
            if drop_disconnected_r_slots:
                rdmol, dropped_slots, audit_layout = _drop_disconnected_r_slots(rdmol, layout)
                row["postprocess_drop_disconnected_r_slots"] = 1
                row["postprocess_dropped_r_slots"] = " ".join(str(value) for value in dropped_slots)
                row["postprocess_dropped_r_slot_count"] = len(dropped_slots)
                row["postprocess_effective_r_slot_count"] = len(audit_layout.r_indices())
                row["raw_disconnected_r_slot_count"] = raw_slot_audit["disconnected_r_slot_count"]
                row["raw_mol_fragment_count"] = raw_slot_audit["mol_fragment_count"]
            else:
                row["postprocess_drop_disconnected_r_slots"] = 0
                row["postprocess_dropped_r_slots"] = ""
                row["postprocess_dropped_r_slot_count"] = 0
                row["postprocess_effective_r_slot_count"] = len(layout.r_indices())
                row["raw_disconnected_r_slot_count"] = raw_slot_audit["disconnected_r_slot_count"]
                row["raw_mol_fragment_count"] = raw_slot_audit["mol_fragment_count"]
            audit = audit_dual_sidechain(rdmol, audit_spec, molecule_index=mol_index)
            row.update(audit.to_dict())
            row.update(_strict_slot_audit(rdmol, audit_layout))
            row["audit_failure_type"] = audit.failure_type
            row["dual_success"] = int(audit.dual_success)
            row["strict_dual_success"] = int(
                audit.dual_success
                and int(row["disconnected_r_slot_count"]) == 0
                and int(row["non_target_core_bond_count"]) == 0
                and int(row["r_slots_connected_to_core_count"]) == int(row["r_slot_count"])
            )
            row["nontrivial_sidechain_pair"] = int(
                int(row.get("left_atom_count", 0)) > 1 and int(row.get("right_atom_count", 0)) > 1
            )
            row["strict_failure_type"] = _classify_strict_failure(row)
            row["quality_label"] = _classify_quality(row)
            row["smiles"] = Chem.MolToSmiles(rdmol)
            rdmol.SetProp("_Name", f"{name_prefix}_{mol_index:04d}")
            rdmol.SetProp("mol_index", str(mol_index))
            rdmol.SetProp("dual_success", str(int(audit.dual_success)))
            rdmol.SetProp("strict_dual_success", str(int(row["strict_dual_success"])))
            rdmol.SetProp("strict_failure_type", str(row["strict_failure_type"]))
            rdmol.SetProp("quality_label", str(row["quality_label"]))
            rdmol.SetProp("disconnected_r_slot_count", str(int(row["disconnected_r_slot_count"])))
            rdmol.SetProp("postprocess_dropped_r_slot_count", str(int(row["postprocess_dropped_r_slot_count"])))
            rdmol.SetProp("audit_failure_type", audit.failure_type)
            mols.append(rdmol)
        except MolReconsError as exc:
            row["reconstruction_status"] = f"reconstruct_error:{type(exc).__name__}"
            row["strict_failure_type"] = "reconstruction_failed"
            row["quality_label"] = "reconstruction_failed"
        except Exception as exc:  # noqa: BLE001 - durable failure rows are required.
            row["reconstruction_status"] = f"error:{type(exc).__name__}:{exc}"
            row["strict_failure_type"] = "reconstruction_failed"
            row["quality_label"] = "reconstruction_failed"
        rows.append(row)
    return rows, mols


def _debug_type_rows(types: torch.Tensor, kind: str, chunk_index: int, node_or_edge_offset: int) -> list[dict[str, Any]]:
    """Return final categorical type ids for debugging across chunks."""
    return [
        {
            "chunk_index": int(chunk_index),
            "batch_offset": int(node_or_edge_offset),
            f"{kind}_idx": int(idx),
            "type": int(value),
        }
        for idx, value in enumerate(types.detach().cpu().tolist())
    ]


def _write_molecules(path: Path, mols: list[Chem.Mol]) -> int:
    """Write molecules to SDF and return the molecule count."""
    writer = Chem.SDWriter(str(path))
    writer.SetKekulize(False)
    try:
        for mol in mols:
            writer.write(mol)
    finally:
        writer.close()
    return len(mols)


def _drop_disconnected_r_slots(
    mol: Chem.Mol,
    layout: DiffGuiDualRSlotLayout,
) -> tuple[Chem.Mol, list[int], DiffGuiDualRSlotLayout]:
    """Delete R slots outside the core component and return the remapped layout."""
    num_atoms = int(mol.GetNumAtoms())
    core_slots = [idx for idx in layout.core_indices() if idx < num_atoms]
    if not core_slots:
        return mol, [], layout
    atom_to_frag: dict[int, int] = {}
    for frag_idx, atom_indices in enumerate(Chem.GetMolFrags(mol, sanitizeFrags=False)):
        for atom_idx in atom_indices:
            atom_to_frag[int(atom_idx)] = int(frag_idx)
    core_frag_ids = {atom_to_frag[idx] for idx in core_slots if idx in atom_to_frag}
    disconnected = [
        idx
        for idx in layout.r_indices()
        if idx < num_atoms and atom_to_frag.get(idx) not in core_frag_ids
    ]
    if not disconnected:
        return mol, [], layout
    remapped_layout = _layout_after_atom_deletion(layout, num_atoms=num_atoms, deleted_indices=disconnected)
    editable = Chem.RWMol(mol)
    for atom_idx in sorted(disconnected, reverse=True):
        editable.RemoveAtom(int(atom_idx))
    trimmed = editable.GetMol()
    try:
        Chem.SanitizeMol(trimmed)
    except Exception:
        pass
    return trimmed, disconnected, remapped_layout


def _layout_after_atom_deletion(
    layout: DiffGuiDualRSlotLayout,
    *,
    num_atoms: int,
    deleted_indices: list[int],
) -> DiffGuiDualRSlotLayout:
    """Return a layout remapped after deleting generated R slots."""
    deleted = set(int(value) for value in deleted_indices)

    def map_index(index: int) -> int | None:
        if index >= num_atoms or index in deleted:
            return None
        return int(index - sum(1 for deleted_idx in deleted if deleted_idx < index))

    left_values: list[int] = []
    for idx in layout.left_r_indices:
        mapped = map_index(idx)
        if mapped is not None:
            left_values.append(mapped)
    right_values: list[int] = []
    for idx in layout.right_r_indices:
        mapped = map_index(idx)
        if mapped is not None:
            right_values.append(mapped)
    left_r = tuple(left_values)
    right_r = tuple(right_values)
    if len(left_r) != len(right_r) or not left_r:
        return layout
    left_anchor = map_index(layout.left_anchor_idx)
    right_anchor = map_index(layout.right_anchor_idx)
    if left_anchor is None or right_anchor is None:
        return layout
    return DiffGuiDualRSlotLayout(
        core_size=layout.core_size,
        left_anchor_idx=left_anchor,
        right_anchor_idx=right_anchor,
        r_slots_per_side=len(left_r),
        left_r_indices=left_r,
        right_r_indices=right_r,
    )


def _strict_slot_audit(mol: Chem.Mol, layout: DiffGuiDualRSlotLayout) -> dict[str, Any]:
    """Audit slot-level connectivity that branch isomorphism alone can miss."""
    num_atoms = int(mol.GetNumAtoms())
    r_slots = [idx for idx in layout.r_indices() if idx < num_atoms]
    core_slots = [idx for idx in layout.core_indices() if idx < num_atoms]
    atom_to_frag: dict[int, int] = {}
    for frag_idx, atom_indices in enumerate(Chem.GetMolFrags(mol, sanitizeFrags=False)):
        for atom_idx in atom_indices:
            atom_to_frag[int(atom_idx)] = int(frag_idx)
    core_frag_ids = {atom_to_frag[idx] for idx in core_slots if idx in atom_to_frag}
    connected_r_slots = [idx for idx in r_slots if atom_to_frag.get(idx) in core_frag_ids]
    disconnected_r_slots = [idx for idx in r_slots if idx not in connected_r_slots]
    allowed_core_r = {
        frozenset((layout.left_anchor_idx, layout.left_root_idx)),
        frozenset((layout.right_anchor_idx, layout.right_root_idx)),
    }
    core_set = set(core_slots)
    r_set = set(r_slots)
    non_target_core_bonds = []
    for bond in mol.GetBonds():
        begin = int(bond.GetBeginAtomIdx())
        end = int(bond.GetEndAtomIdx())
        if ((begin in core_set and end in r_set) or (end in core_set and begin in r_set)) and (
            frozenset((begin, end)) not in allowed_core_r
        ):
            non_target_core_bonds.append((begin, end))
    return {
        "r_slot_count": len(r_slots),
        "r_slots_connected_to_core_count": len(connected_r_slots),
        "disconnected_r_slot_count": len(disconnected_r_slots),
        "disconnected_r_slots": " ".join(str(idx) for idx in disconnected_r_slots),
        "mol_fragment_count": len(set(atom_to_frag.values())),
        "non_target_core_bond_count": len(non_target_core_bonds),
        "non_target_core_bonds": " ".join(f"{a}-{b}" for a, b in non_target_core_bonds),
    }


def _classify_strict_failure(row: dict[str, Any]) -> str:
    """Return the first strict failure mode visible in an audit row."""
    if str(row.get("reconstruction_status")) != "ok":
        return "reconstruction_failed"
    if not int(row.get("dual_success", 0)):
        failure = str(row.get("audit_failure_type") or "formal_dual_failure")
        return f"formal_{failure}"
    if int(row.get("non_target_core_bond_count", 0)) > 0:
        return "non_target_core_bond"
    if int(row.get("disconnected_r_slot_count", 0)) > 0:
        return "disconnected_r_slots"
    if int(row.get("r_slots_connected_to_core_count", 0)) < int(row.get("r_slot_count", 0)):
        return "incomplete_r_slot_connectivity"
    if int(row.get("mol_fragment_count", 1)) > 1:
        return "extra_molecular_fragments"
    if not int(row.get("strict_dual_success", 0)):
        return "strict_unknown"
    return "ok"


def _classify_quality(row: dict[str, Any]) -> str:
    """Classify strict successes as nontrivial or trivial sidechain pairs."""
    if str(row.get("strict_failure_type")) != "ok":
        return str(row.get("strict_failure_type"))
    if int(row.get("nontrivial_sidechain_pair", 0)):
        return "nontrivial_sidechain_pair"
    return "trivial_sidechain_pair"


def _write_summary(output_dir: Path, *, args: argparse.Namespace, rows: list[dict[str, Any]], written: int, p0: dict[str, Any]) -> dict[str, Any]:
    """Write run summary and Markdown report."""
    total = len(rows)
    success = sum(int(row.get("dual_success", 0)) for row in rows)
    strict_success = sum(int(row.get("strict_dual_success", 0)) for row in rows)
    nontrivial = sum(int(row.get("nontrivial_sidechain_pair", 0)) for row in rows)
    recon_ok = sum(1 for row in rows if str(row.get("reconstruction_status")) == "ok")
    failures: dict[str, int] = {}
    strict_failures: dict[str, int] = {}
    quality_counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get("audit_failure_type") or row.get("reconstruction_status") or "unknown")
        failures[key] = failures.get(key, 0) + 1
        strict_key = str(row.get("strict_failure_type") or "unknown")
        strict_failures[strict_key] = strict_failures.get(strict_key, 0) + 1
        quality_key = str(row.get("quality_label") or "unknown")
        quality_counts[quality_key] = quality_counts.get(quality_key, 0) + 1
    plot_files = _write_summary_plots(
        output_dir=output_dir,
        total=total,
        written=written,
        recon_ok=recon_ok,
        success=success,
        strict_success=strict_success,
        nontrivial=nontrivial,
        strict_failures=strict_failures,
        quality_counts=quality_counts,
    )
    summary = {
        "p0_dir": str(args.p0_dir),
        "num_graphs": int(args.num_graphs),
        "num_timesteps": int(args.num_timesteps),
        "seed": int(args.seed),
        "device": str(args.device),
        "force_r_chain": bool(args.force_r_chain),
        "r_chain_bond_type": int(args.r_chain_bond_type),
        "drop_disconnected_r_slots": bool(args.drop_disconnected_r_slots),
        "written_molecules": int(written),
        "reconstruction_ok": int(recon_ok),
        "dual_success": int(success),
        "strict_dual_success": int(strict_success),
        "nontrivial_sidechain_pair": int(nontrivial),
        "failure_counts": failures,
        "strict_failure_counts": strict_failures,
        "quality_counts": quality_counts,
        "plots": [path.name for path in plot_files],
        "layout": p0["layout"].to_dict(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# DiffGui Dual-R Constrained Sample",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| num graphs | {total} |",
        f"| written molecules | {written} |",
        f"| reconstruction ok | {recon_ok} |",
        f"| dual success | {success} |",
        f"| strict dual success | {strict_success} |",
        f"| nontrivial sidechain pair | {nontrivial} |",
        "",
    ]
    for plot_path in plot_files:
        lines.extend([f"![{plot_path.stem}]({plot_path.name})", ""])
    lines.extend(
        [
            "## Strict Failure Counts",
            "",
            "| failure | count |",
            "|---|---:|",
        ]
    )
    for key, value in sorted(strict_failures.items()):
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "## Quality Counts",
            "",
            "| quality | count |",
            "|---|---:|",
        ]
    )
    for key, value in sorted(quality_counts.items()):
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "## Example Molecules",
            "",
            "| mol_index | strict_failure_type | quality_label | SMILES |",
            "|---:|---|---|---|",
        ]
    )
    for row in rows[:10]:
        lines.append(
            "| {mol_index} | {strict_failure_type} | {quality_label} | `{smiles}` |".format(
                mol_index=row.get("mol_index", ""),
                strict_failure_type=row.get("strict_failure_type", ""),
                quality_label=row.get("quality_label", ""),
                smiles=row.get("smiles", ""),
            )
        )
    lines.extend(
        [
            "",
        "## Failure Counts",
        "",
        "| failure | count |",
        "|---|---:|",
        ]
    )
    for key, value in sorted(failures.items()):
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- [generated.sdf](generated.sdf)",
            "- [reconstruction_audit.csv](reconstruction_audit.csv)",
            "- [final_node_types.csv](final_node_types.csv)",
            "- [final_halfedge_types.csv](final_halfedge_types.csv)",
            "- [summary.json](summary.json)",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def _write_summary_plots(
    *,
    output_dir: Path,
    total: int,
    written: int,
    recon_ok: int,
    success: int,
    strict_success: int,
    nontrivial: int,
    strict_failures: dict[str, int],
    quality_counts: dict[str, int],
) -> list[Path]:
    """Write compact PNG plots for the run report."""
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError:
        LOGGER.warning("matplotlib is unavailable; skipping summary plots")
        return []

    plot_files: list[Path] = []
    metrics_path = output_dir / "summary_metrics.png"
    labels = ["total", "written", "recon_ok", "dual", "strict", "nontrivial"]
    values = [total, written, recon_ok, success, strict_success, nontrivial]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    bars = ax.bar(labels, values, color=["#4C78A8", "#72B7B2", "#54A24B", "#F58518", "#E45756", "#B279A2"])
    ax.set_ylabel("count")
    ax.set_title("DiffGui dual-R run summary")
    ax.set_ylim(0, max(max(values), 1) * 1.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(value), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(metrics_path, dpi=180)
    plt.close(fig)
    plot_files.append(metrics_path)

    counts_path = output_dir / "failure_quality_counts.png"
    merged = {f"strict:{key}": value for key, value in strict_failures.items()}
    merged.update({f"quality:{key}": value for key, value in quality_counts.items()})
    items = sorted(merged.items(), key=lambda item: (-item[1], item[0]))
    if items:
        fig, ax = plt.subplots(figsize=(8.2, max(3.6, 0.34 * len(items))))
        labels = [key for key, _value in items]
        values = [value for _key, value in items]
        y_pos = np.arange(len(items))
        ax.barh(y_pos, values, color="#6F9FD8")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel("count")
        ax.set_title("Strict failure and quality labels")
        ax.set_xlim(0, max(values) * 1.25 if values else 1)
        for y_idx, value in enumerate(values):
            ax.text(value, y_idx, f" {value}", va="center", fontsize=9)
        fig.tight_layout()
        fig.savefig(counts_path, dpi=180)
        plt.close(fig)
        plot_files.append(counts_path)
    return plot_files


def _apply_specs_to_log_probs(
    node_log_probs: torch.Tensor,
    halfedge_log_probs: torch.Tensor,
    specs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply all graph-specific constraint specs to batched log probabilities."""
    out_nodes = node_log_probs
    out_edges = halfedge_log_probs
    for spec in specs:
        out_nodes, out_edges = apply_dual_r_constraints_to_log_probs(out_nodes, out_edges, spec)
    return out_nodes, out_edges


def _apply_specs_to_types(
    node_types: torch.Tensor,
    halfedge_types: torch.Tensor,
    specs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply all graph-specific constraint specs to batched sampled type ids."""
    out_nodes = node_types
    out_edges = halfedge_types
    for spec in specs:
        out_nodes, out_edges = apply_dual_r_constraints_to_types(out_nodes, out_edges, spec)
    return out_nodes, out_edges


def _core_node_mask(ligand_batch: torch.Tensor, *, core_size: int) -> torch.Tensor:
    """Return mask for the first core_size ligand slots of every graph."""
    mask = torch.zeros_like(ligand_batch, dtype=torch.bool)
    for graph_idx in torch.unique(ligand_batch).tolist():
        indices = torch.where(ligand_batch == int(graph_idx))[0]
        mask[indices[: int(core_size)]] = True
    return mask


def _center_pos(
    protein_pos: torch.Tensor,
    ligand_pos: torch.Tensor,
    protein_batch: torch.Tensor,
    ligand_batch: torch.Tensor,
    *,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Center protein and ligand coordinates like DiffGui."""
    if mode == "protein":
        offset = _scatter_mean(protein_pos, protein_batch)
        return protein_pos - offset[protein_batch], ligand_pos - offset[ligand_batch], offset
    if mode == "none":
        max_graph = int(torch.max(torch.cat([protein_batch, ligand_batch])).item()) + 1
        offset = torch.zeros(max_graph, 3, dtype=protein_pos.dtype, device=protein_pos.device)
        return protein_pos, ligand_pos, offset
    raise ValueError(f"Unsupported center_pos_mode={mode!r}.")


def _scatter_mean(values: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Small scatter_mean helper to avoid importing another project wrapper."""
    num_graphs = int(batch.max().item()) + 1
    sums = torch.zeros(num_graphs, values.shape[-1], dtype=values.dtype, device=values.device)
    counts = torch.zeros(num_graphs, 1, dtype=values.dtype, device=values.device)
    sums.index_add_(0, batch.long(), values)
    counts.index_add_(0, batch.long(), torch.ones(values.shape[0], 1, dtype=values.dtype, device=values.device))
    return sums / torch.clamp(counts, min=1.0)


def _make_batch_lab(config: Any, batch_size: int, device: torch.device) -> torch.Tensor:
    """Create DiffGui property-conditioning labels."""
    values = [
        float(config.model.logp),
        float(config.model.tpsa),
        float(config.model.sa),
        float(config.model.qed),
        float(config.model.aff),
    ]
    return torch.tensor([values for _ in range(batch_size)], dtype=torch.float32, device=device)


def _read_json(path: Path) -> dict[str, Any]:
    """Read JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write CSV rows with stable field ordering."""
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _setup_logging(log_path: Path, log_level: str) -> None:
    """Configure console and durable file logging."""
    LOGGER.setLevel(getattr(logging, log_level))
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(stream)
    LOGGER.addHandler(file_handler)


if __name__ == "__main__":
    raise SystemExit(main())
