"""Run a decoupled DiffGui 100-step rollout and PPO replay smoke test."""

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
from rdkit import Chem  # noqa: E402
from torch_geometric.data import Batch  # noqa: E402
from torch_geometric.transforms import Compose  # noqa: E402

from ddpo.diffgui.runtime.actor import DiffGuiActor  # noqa: E402
from ddpo.diffgui.runtime.rollout import DiffGuiPromptBatch, DiffGuiRollout  # noqa: E402
from models.model import DiffGui  # noqa: E402
from utils.data import PDBProtein, parse_lig_file  # noqa: E402
from utils.dataset import ProteinLigandData, to_torch_dict  # noqa: E402
from utils.misc import load_config, seed_all  # noqa: E402
from utils.reconstruct import MolReconsError, reconstruct_from_generated_with_edges  # noqa: E402
from utils.sample_utils import seperate_outputs  # noqa: E402
from utils.transforms import FeatureComplex, make_data_placeholder  # noqa: E402


logger = logging.getLogger("diffgui_100step_smoke")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the smoke run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-config", type=Path, default=DIFFGUI_ROOT / "configs/sample/sample.yml")
    parser.add_argument("--checkpoint", type=Path, default=DIFFGUI_ROOT / "ckpt/trained.pt")
    parser.add_argument("--pocket-pdb", type=Path, default=DIFFGUI_ROOT / "sample/3ztx_pocket.pdb")
    parser.add_argument("--ligand-sdf", type=str, default="None")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "analysis/diffgui_100step_rollout_smoke_20260609")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-timesteps", type=int, default=100)
    parser.add_argument("--num-graphs", type=int, default=2)
    parser.add_argument("--ligand-atoms", type=int, default=24)
    parser.add_argument("--seed", type=int, default=260609)
    parser.add_argument("--gui-strength", type=float, default=0.0)
    parser.add_argument("--actor-lr", type=float, default=1e-6)
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--graph-micro-batch-size", type=int, default=1)
    parser.add_argument("--freeze-mode", type=str, default="none", choices=["none", "heads_only", "last_n_blocks"])
    parser.add_argument("--train-last-n-blocks", type=int, default=1)
    parser.add_argument("--add-edge", type=str, default="None")
    return parser.parse_args()


def main() -> None:
    """Run the smoke and write durable local artifacts."""
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(output_dir / "rollout.log")

    logger.info("starting DiffGui 100-step rollout smoke")
    logger.info("args=%s", vars(args))
    if args.num_timesteps <= 0:
        raise ValueError("--num-timesteps must be positive")
    if args.num_graphs <= 0 or args.ligand_atoms <= 0:
        raise ValueError("--num-graphs and --ligand-atoms must be positive")

    seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false in this process.")

    config = load_config(args.sample_config)
    ckpt = torch.load(str(args.checkpoint), map_location=device)
    train_config = ckpt["config"]
    _force_num_timesteps(train_config.model.diff, args.num_timesteps)
    logger.info("loaded checkpoint iteration=%s", ckpt.get("iteration"))
    logger.info("forced model.diff.num_timesteps=%d", int(train_config.model.diff.num_timesteps))

    featurizer = FeatureComplex(ckpt["config"].data.transform.ligand_atom_mode, sample=True)
    transform = Compose([featurizer])
    model = DiffGui(
        config=train_config.model,
        protein_node_types=featurizer.protein_feat_dim,
        ligand_node_types=featurizer.atom_feat_dim,
        num_edge_types=featurizer.bond_feat_dim,
    ).to(device)
    load_report = _load_matching_network_weights(model, ckpt["model"])
    model.eval()
    logger.info("model.num_timesteps=%d", int(model.num_timesteps))
    logger.info("load_report=%s", load_report)

    data = _build_pocket_data(args.pocket_pdb, args.ligand_sdf)
    data = transform(data)
    batch = Batch.from_data_list(
        [data.clone() for _ in range(args.num_graphs)],
        follow_batch=featurizer.follow_batch,
    ).to(device)
    holder = make_data_placeholder(
        n_nodes_list=[args.ligand_atoms] * args.num_graphs,
        device=device,
    )
    prompt = DiffGuiPromptBatch(
        protein_node=batch.protein_atom_feat.float(),
        protein_pos=batch.protein_pos,
        protein_batch=batch.protein_element_batch,
        ligand_batch=holder["batch_node"],
        halfedge_index=holder["halfedge_index"],
        halfedge_batch=holder["batch_halfedge"],
        batch_lab=_make_batch_lab(config, args.num_graphs, device),
    )

    rollout = DiffGuiRollout(
        model,
        {
            "diffgui_rollout": {
                "num_timesteps": args.num_timesteps,
                "gui_strength": args.gui_strength,
                "cpu_offload": True,
                "synthetic_advantage_mode": "linear",
                "synthetic_advantage_scale": 1.0,
            }
        },
    )
    policy_batch = rollout.generate_policy_batch(prompt)
    policy_batch.validate()
    logger.info(
        "policy_batch old_logprob_shape=%s timesteps_shape=%s",
        tuple(policy_batch.old_logprob.shape),
        tuple(policy_batch.timesteps.shape),
    )

    actor = DiffGuiActor(
        model,
        {
            "train": {
                "learning_rate": args.actor_lr,
                "clip_range": 0.2,
                "max_grad_norm": 1.0,
                "ppo_epochs": 1,
                "diffgui_logprob_reduction": "mean",
                "precision": args.precision,
                "graph_micro_batch_size": args.graph_micro_batch_size,
                "freeze": {
                    "mode": args.freeze_mode,
                    "train_last_n_blocks": args.train_last_n_blocks,
                },
            }
        },
    )
    actor_metrics = actor.update_policy(policy_batch)
    logger.info("actor_metrics=%s", actor_metrics)

    rows, written_mols = _write_reconstructed_outputs(
        policy_batch=policy_batch,
        featurizer=featurizer,
        n_graphs=args.num_graphs,
        batch_node=holder["batch_node"].detach().cpu().numpy(),
        halfedge_index=holder["halfedge_index"].detach().cpu().numpy(),
        batch_halfedge=holder["batch_halfedge"].detach().cpu().numpy(),
        output_dir=output_dir,
        add_edge=None if args.add_edge == "None" else args.add_edge,
    )

    summary = {
        "stage": "diffgui_100step_rollout_smoke",
        "checkpoint": str(args.checkpoint),
        "pocket_pdb": str(args.pocket_pdb),
        "device": str(device),
        "num_timesteps": int(args.num_timesteps),
        "model_num_timesteps": int(model.num_timesteps),
        "num_graphs": int(args.num_graphs),
        "ligand_atoms": int(args.ligand_atoms),
        "gui_strength": float(args.gui_strength),
        "precision": str(args.precision),
        "graph_micro_batch_size": int(args.graph_micro_batch_size),
        "freeze_mode": str(args.freeze_mode),
        "train_last_n_blocks": int(args.train_last_n_blocks),
        "old_logprob_shape": list(policy_batch.old_logprob.shape),
        "timesteps_shape": list(policy_batch.timesteps.shape),
        "actor_metrics": actor_metrics,
        "load_report": load_report,
        "reconstruction": {
            "attempted": len(rows),
            "written_mols": int(written_mols),
            "success": int(sum(1 for row in rows if row["status"] == "ok")),
        },
        "outputs": {
            "log": str(output_dir / "rollout.log"),
            "sdf": str(output_dir / "generated_100step_smoke.sdf"),
            "csv": str(output_dir / "molecules.csv"),
            "summary": str(output_dir / "summary.json"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("summary=%s", json.dumps(summary, ensure_ascii=False))


def _setup_logging(log_path: Path) -> None:
    """Configure readable terminal and file logging."""
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)


def _force_num_timesteps(diff_config: Any, num_timesteps: int) -> None:
    """Force DiffGui diffusion config to a shorter self-consistent schedule."""
    diff_config.num_timesteps = int(num_timesteps)
    diff_bond = diff_config.diff_bond
    if str(diff_bond.beta_schedule) == "segment":
        original = [int(value) for value in diff_bond.time_segment]
        diff_bond.time_segment = _scale_segments(original, int(num_timesteps))


def _scale_segments(segments: list[int], total: int) -> list[int]:
    """Scale segment lengths while preserving the original proportions."""
    raw = np.asarray(segments, dtype=np.float64)
    scaled = raw / raw.sum() * float(total)
    floored = np.floor(scaled).astype(np.int64)
    floored = np.maximum(floored, 1)
    while int(floored.sum()) < total:
        floored[int(np.argmax(scaled - floored))] += 1
    while int(floored.sum()) > total:
        candidates = np.where(floored > 1)[0]
        floored[int(candidates[np.argmax(floored[candidates] - scaled[candidates])])] -= 1
    return [int(value) for value in floored.tolist()]


def _load_matching_network_weights(model: torch.nn.Module, checkpoint_state: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Load checkpoint weights while keeping regenerated 100-step transition buffers."""
    model_state = model.state_dict()
    filtered: dict[str, torch.Tensor] = {}
    skipped: list[dict[str, Any]] = []
    regenerated_prefixes = (
        "pos_transition.",
        "node_transition.",
        "edge_transition.",
        "time_emb.",
    )
    for key, value in checkpoint_state.items():
        if key.startswith(regenerated_prefixes):
            skipped.append(
                {
                    "key": key,
                    "checkpoint_shape": list(value.shape),
                    "model_shape": list(model_state[key].shape) if key in model_state else None,
                    "reason": "regenerated_from_current_num_timesteps",
                }
            )
            continue
        model_value = model_state.get(key)
        if model_value is not None and tuple(model_value.shape) == tuple(value.shape):
            filtered[key] = value
            continue
        skipped.append(
            {
                "key": key,
                "checkpoint_shape": None if not hasattr(value, "shape") else list(value.shape),
                "model_shape": None if model_value is None else list(model_value.shape),
            }
        )
    non_transition_skipped = [
        item["key"]
        for item in skipped
        if not str(item["key"]).startswith(regenerated_prefixes)
    ]
    if non_transition_skipped:
        raise RuntimeError(f"Unexpected non-transition checkpoint shape mismatches: {non_transition_skipped[:10]}")
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    return {
        "loaded_keys": len(filtered),
        "skipped_regenerated_keys": len(skipped),
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
    }


def _build_pocket_data(pocket_pdb_path: Path, ligand_sdf_path: str) -> Any:
    """Build a DiffGui ProteinLigandData object for denovo pocket generation."""
    pocket_dict = PDBProtein(str(pocket_pdb_path)).to_dict_atom()
    if ligand_sdf_path != "None":
        ligand_dict = parse_lig_file(ligand_sdf_path)
    else:
        ligand_dict = {
            "element": torch.empty([0], dtype=torch.long),
            "hybridization": torch.empty([0], dtype=torch.long),
            "pos": torch.empty([0, 3], dtype=torch.float),
            "bond_index": torch.empty([2, 0], dtype=torch.long),
            "bond_type": torch.empty([0], dtype=torch.long),
            "atom_feature": torch.empty([0, 8], dtype=torch.float),
        }
    return ProteinLigandData.protein_ligand_dicts(
        protein_dict=to_torch_dict(pocket_dict),
        ligand_dict=to_torch_dict(ligand_dict),
    )


def _make_batch_lab(config: Any, num_graphs: int, device: torch.device) -> torch.Tensor:
    """Create DiffGui label conditioning tensor for the sample batch."""
    values = [
        float(config.model.logp),
        float(config.model.tpsa),
        float(config.model.sa),
        float(config.model.qed),
        float(config.model.aff),
    ]
    return torch.tensor([values for _ in range(num_graphs)], dtype=torch.float32, device=device)


def _write_reconstructed_outputs(
    *,
    policy_batch: Any,
    featurizer: Any,
    n_graphs: int,
    batch_node: np.ndarray,
    halfedge_index: np.ndarray,
    batch_halfedge: np.ndarray,
    output_dir: Path,
    add_edge: str | None,
) -> tuple[list[dict[str, Any]], int]:
    """Decode final DiffGui states and write molecule CSV/SDF artifacts."""
    node_traj = policy_batch.node_traj.detach().cpu().numpy()
    pos_traj = policy_batch.pos_centered_traj.detach().cpu().numpy()
    halfedge_traj = policy_batch.halfedge_traj.detach().cpu().numpy()
    outputs_np = {
        "pred": [node_traj[-1], pos_traj[-1], halfedge_traj[-1]],
        "traj": [node_traj, pos_traj, halfedge_traj],
    }
    split_outputs = seperate_outputs(outputs_np, n_graphs, batch_node, halfedge_index, batch_halfedge)
    rows: list[dict[str, Any]] = []
    sdf_path = output_dir / "generated_100step_smoke.sdf"
    writer = Chem.SDWriter(str(sdf_path))
    written = 0
    for mol_index, output_mol in enumerate(split_outputs):
        row = {
            "mol_index": mol_index,
            "status": "ok",
            "smiles": "",
            "num_atoms": int((batch_node == mol_index).sum()),
            "old_logprob_mean": float(policy_batch.old_logprob[:, mol_index].float().mean().item()),
        }
        try:
            mol_info = featurizer.decode_output(
                pred_node=output_mol["pred"][0],
                pred_pos=output_mol["pred"][1],
                pred_halfedge=output_mol["pred"][2],
                halfedge_index=output_mol["halfedge_index"],
            )
            rdmol = reconstruct_from_generated_with_edges(mol_info, add_edge=add_edge)
            smiles = Chem.MolToSmiles(rdmol)
            row["smiles"] = smiles
            rdmol.SetProp("_Name", f"diffgui100_{mol_index:03d}_oldlogp_{row['old_logprob_mean']:.4f}")
            rdmol.SetProp("diffgui_num_timesteps", str(int(policy_batch.old_logprob.shape[0])))
            rdmol.SetProp("old_logprob_mean", f"{row['old_logprob_mean']:.8f}")
            writer.write(rdmol)
            written += 1
        except MolReconsError as exc:
            row["status"] = f"reconstruct_error:{type(exc).__name__}"
        except Exception as exc:
            row["status"] = f"error:{type(exc).__name__}"
        rows.append(row)
    writer.close()

    csv_path = output_dir / "molecules.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["mol_index", "status", "smiles", "num_atoms", "old_logprob_mean"]
        writer_csv = csv.DictWriter(handle, fieldnames=fieldnames)
        writer_csv.writeheader()
        writer_csv.writerows(rows)
    return rows, written


if __name__ == "__main__":
    main()
