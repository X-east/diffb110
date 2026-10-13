"""Run DiffGui rollout + docking reward + PPO replay batch-size sweeps.

This is a DiffGui-only benchmark entrypoint.  It intentionally does not import
or call the DiffSBDD DDPO training entrypoint; the only shared code is the
model-agnostic reward stack.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIFFGUI_ROOT = PROJECT_ROOT / "reference project" / "代码" / "DiffGui"
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
from rdkit.Chem.Scaffolds import MurckoScaffold  # noqa: E402
from torch_geometric.data import Batch  # noqa: E402
from torch_geometric.transforms import Compose  # noqa: E402

from ddpo.diffgui.runtime.actor import DiffGuiActor, DiffGuiPolicyBatch  # noqa: E402
from ddpo.diffgui.runtime.rollout import DiffGuiPromptBatch, DiffGuiRollout  # noqa: E402
from ddpo.diffgui.runtime.timestep_config import force_diff_config_num_timesteps, load_diffgui_network_state  # noqa: E402
from models.model import DiffGui  # noqa: E402
from utils.data import PDBProtein, parse_lig_file  # noqa: E402
from utils.dataset import ProteinLigandData, to_torch_dict  # noqa: E402
from utils.misc import load_config as load_diffgui_config  # noqa: E402
from utils.misc import seed_all  # noqa: E402
from utils.evaluation.atom_num import get_space_size, sample_atom_num  # noqa: E402
from utils.reconstruct import MolReconsError, reconstruct_from_generated_with_edges  # noqa: E402
from utils.sample_utils import seperate_outputs  # noqa: E402
from utils.transforms import FeatureComplex, make_data_placeholder  # noqa: E402


LOGGER = logging.getLogger("DiffGuiDockingSweep")


@dataclass
class GpuMemoryMonitor:
    """Background memory monitor for one physical GPU."""

    gpu_id: str
    stop_event: threading.Event
    thread: threading.Thread
    samples: list[dict[str, Any]]
    errors: list[str]
    sample_path: Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="YAML config path.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output directory.")
    parser.add_argument("--batch-sizes", type=str, default=None, help="Comma-separated graph batch sizes.")
    parser.add_argument("--device", type=str, default=None, help="Override DiffGui torch device, e.g. cuda:7.")
    parser.add_argument("--docking-gpu-id", type=int, default=None, help="Override physical GPU id used by UniDock.")
    parser.add_argument("--reward-batch-size", type=int, default=None, help="Override docking reward chunk size.")
    parser.add_argument("--num-timesteps", type=int, default=None, help="Override DiffGui diffusion steps.")
    parser.add_argument("--gui-strength", type=float, default=None, help="Override DiffGui classifier-free guidance strength.")
    parser.add_argument("--precision", type=str, default=None, choices=["fp32", "fp16", "bf16"], help="Override PPO replay precision.")
    parser.add_argument("--graph-micro-batch-size", type=int, default=None, help="Override PPO graph microbatch size.")
    parser.add_argument("--freeze-mode", type=str, default=None, choices=["none", "heads_only", "last_n_blocks"])
    parser.add_argument("--train-last-n-blocks", type=int, default=None)
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    """Run the configured batch-size sweep."""
    args = parse_args()
    config = _load_yaml(args.config)
    sweep_cfg = config.setdefault("diffgui_docking_batch_sweep", {})
    train_cfg = config.setdefault("train", {})
    reward_cfg = config.setdefault("reward", {})
    docking_cfg = reward_cfg.setdefault("docking", {})
    if args.output_dir is not None:
        sweep_cfg["output_dir"] = str(args.output_dir)
    if args.batch_sizes is not None:
        sweep_cfg["batch_sizes"] = [int(value) for value in args.batch_sizes.split(",") if value.strip()]
    if args.device is not None:
        sweep_cfg["device"] = str(args.device)
        if args.docking_gpu_id is None:
            device_index = _cuda_index_from_device(args.device)
            if device_index is not None:
                docking_cfg["gpu_id"] = int(device_index)
    if args.docking_gpu_id is not None:
        docking_cfg["gpu_id"] = int(args.docking_gpu_id)
    if args.reward_batch_size is not None:
        sweep_cfg["reward_batch_size"] = int(args.reward_batch_size)
    if args.num_timesteps is not None:
        sweep_cfg["num_timesteps"] = int(args.num_timesteps)
    if args.gui_strength is not None:
        sweep_cfg["gui_strength"] = float(args.gui_strength)
    if args.precision is not None:
        sweep_cfg["precision"] = str(args.precision)
        train_cfg["precision"] = str(args.precision)
    if args.graph_micro_batch_size is not None:
        sweep_cfg["graph_micro_batch_size"] = int(args.graph_micro_batch_size)
        train_cfg["graph_micro_batch_size"] = int(args.graph_micro_batch_size)
    if args.freeze_mode is not None:
        train_cfg.setdefault("freeze", {})["mode"] = str(args.freeze_mode)
    if args.train_last_n_blocks is not None:
        train_cfg.setdefault("freeze", {})["train_last_n_blocks"] = int(args.train_last_n_blocks)

    output_dir = _resolve_path(sweep_cfg.get("output_dir", "analysis/diffgui_docking_batch_sweep_20260609"))
    output_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(output_dir / "run.log", args.log_level)

    LOGGER.info("starting DiffGui docking batch sweep")
    LOGGER.info("config_path=%s", args.config.resolve())
    LOGGER.info("output_dir=%s", output_dir)
    LOGGER.info("git=%s", _git_snapshot())
    LOGGER.info("conda_prefix=%s python=%s", os.environ.get("CONDA_PREFIX", ""), sys.executable)
    LOGGER.info("cuda_visible_devices=%s", os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    LOGGER.info("resolved_sweep_cfg=%s", json.dumps(sweep_cfg, ensure_ascii=False))

    batch_sizes = _read_batch_sizes(sweep_cfg)
    if not batch_sizes:
        raise ValueError("diffgui_docking_batch_sweep.batch_sizes must not be empty.")

    config_snapshot = output_dir / "resolved_config.yaml"
    config_snapshot.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")

    rows: list[dict[str, Any]] = []
    failed = False
    for batch_size in batch_sizes:
        batch_dir = output_dir / f"batch_{batch_size:03d}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        try:
            row = run_one_batch(
                config=config,
                batch_size=batch_size,
                batch_dir=batch_dir,
            )
            LOGGER.info(
                "batch=%d done total=%.2fs recon=%d/%d docking_success=%d reward_mean=%.4f ppo_loss=%.4f",
                batch_size,
                row.get("total_seconds", math.nan),
                row.get("reconstruction_success", 0),
                row.get("reconstruction_attempted", 0),
                row.get("docking_success", 0),
                row.get("reward_total_mean", math.nan),
                row.get("PPO_Loss", math.nan),
            )
        except Exception as exc:
            failed = True
            LOGGER.exception("batch=%d failed", batch_size)
            row = {
                "batch_size": batch_size,
                "status": f"failed:{type(exc).__name__}",
                "error": str(exc),
            }
            (batch_dir / "error.json").write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
        rows.append(row)
        _write_summary_artifacts(output_dir, rows, config)

    if failed:
        raise SystemExit(1)


def _cuda_index_from_device(value: str) -> int | None:
    """Parse a physical CUDA index from a device string such as ``cuda:7``."""
    device = str(value).strip().lower()
    if device == "cuda":
        return 0
    if not device.startswith("cuda:"):
        return None
    try:
        return int(device.split(":", 1)[1])
    except ValueError:
        return None


def _seed_diffgui(seed: int) -> None:
    """Seed the DiffGui reference runtime, including NumPy-based size sampling."""
    seed_all(int(seed))


def run_one_batch(
    *,
    config: dict[str, Any],
    batch_size: int,
    batch_dir: Path,
) -> dict[str, Any]:
    """Run one rollout/reward/PPO benchmark batch."""
    sweep_cfg = config.get("diffgui_docking_batch_sweep", {})
    model_cfg = config.get("diffgui_model", {})
    train_cfg = config.get("train", {})
    device = torch.device(str(sweep_cfg.get("device", "cuda:0")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false in this process.")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    reward_batch_size = int(sweep_cfg.get("reward_batch_size") or batch_size)
    if reward_batch_size <= 0:
        raise ValueError("reward_batch_size must be positive.")

    seed = int(sweep_cfg.get("seed_base", 260609)) + batch_size
    seed_all(seed)
    monitor = _start_gpu_memory_monitor(_physical_gpu_id_from_config(config, device), batch_dir, float(sweep_cfg.get("memory_poll_interval", 1.0)))
    started = time.perf_counter()
    timing: dict[str, float] = {}
    model: torch.nn.Module | None = None
    try:
        phase_start = time.perf_counter()
        sample_config = _resolve_path(model_cfg.get("sample_config", DIFFGUI_ROOT / "configs/sample/sample.yml"))
        checkpoint = _resolve_path(model_cfg.get("checkpoint", DIFFGUI_ROOT / "ckpt/trained.pt"))
        pocket_pdb = _resolve_path(model_cfg.get("pocket_pdb", DIFFGUI_ROOT / "sample/3ztx_pocket.pdb"))
        ligand_sdf = str(model_cfg.get("ligand_sdf", "None"))
        num_timesteps = int(sweep_cfg.get("num_timesteps", 100))
        ligand_atoms = int(sweep_cfg.get("ligand_atoms", 24))

        model, featurizer, sample_cfg = _load_diffgui_model(
            sample_config=sample_config,
            checkpoint=checkpoint,
            num_timesteps=num_timesteps,
            device=device,
        )
        timing["model_load_seconds"] = time.perf_counter() - phase_start

        phase_start = time.perf_counter()
        prompt, batch_node, halfedge_index, batch_halfedge = _build_prompt(
            sample_cfg=sample_cfg,
            featurizer=featurizer,
            pocket_pdb=pocket_pdb,
            ligand_sdf=ligand_sdf,
            batch_size=batch_size,
            ligand_atoms=ligand_atoms,
            sweep_cfg=sweep_cfg,
            device=device,
        )
        rollout = DiffGuiRollout(
            model,
            {
                "diffgui_rollout": {
                    "num_timesteps": num_timesteps,
                    "auto_configure_timesteps": bool(sweep_cfg.get("auto_configure_timesteps", True)),
                    "gui_strength": float(sweep_cfg.get("gui_strength", 0.0)),
                    "cpu_offload": True,
                    "synthetic_advantage_mode": "zeros",
                    "synthetic_advantage_scale": 1.0,
                }
            },
        )
        policy_batch = rollout.generate_policy_batch(prompt)
        policy_batch.validate()
        timing["rollout_seconds"] = time.perf_counter() - phase_start

        phase_start = time.perf_counter()
        reconstruction_rows, written_mols = _write_reconstructed_outputs(
            policy_batch=policy_batch,
            featurizer=featurizer,
            n_graphs=batch_size,
            batch_node=batch_node,
            halfedge_index=halfedge_index,
            batch_halfedge=batch_halfedge,
            output_dir=batch_dir,
            name_prefix=f"diffgui_dock_bs{batch_size:03d}",
        )
        timing["reconstruction_seconds"] = time.perf_counter() - phase_start

        phase_start = time.perf_counter()
        reward_rows = _score_reconstructed_sdf(
            config=config,
            input_sdf=batch_dir / "generated.sdf",
            output_dir=batch_dir,
            batch_size=reward_batch_size,
        )
        timing["reward_seconds"] = time.perf_counter() - phase_start

        rewards_by_graph = _rewards_by_graph_index(
            reconstruction_rows=reconstruction_rows,
            reward_rows=reward_rows,
            failure_penalty=float(sweep_cfg.get("failure_advantage", -1.0)),
        )
        advantages = _make_advantages(
            rewards_by_graph,
            normalize=bool(sweep_cfg.get("normalize_advantages", True)),
            epsilon=float(sweep_cfg.get("advantage_epsilon", 1.0e-6)),
            clamp_min=sweep_cfg.get("advantage_clip_min", None),
            clamp_max=sweep_cfg.get("advantage_clip_max", None),
        )
        policy_batch = _replace_policy_advantages(policy_batch, advantages)

        phase_start = time.perf_counter()
        actor = DiffGuiActor(
            model,
            {
                "train": {
                    "learning_rate": float(train_cfg.get("learning_rate", 1.0e-6)),
                    "adam_beta1": float(train_cfg.get("adam_beta1", 0.9)),
                    "adam_beta2": float(train_cfg.get("adam_beta2", 0.999)),
                    "adam_weight_decay": float(train_cfg.get("adam_weight_decay", 0.0)),
                    "adam_epsilon": float(train_cfg.get("adam_epsilon", 1.0e-8)),
                    "clip_range": float(train_cfg.get("clip_range", 0.2)),
                    "max_grad_norm": float(train_cfg.get("max_grad_norm", 1.0)),
                    "ppo_epochs": int(train_cfg.get("ppo_epochs", 1)),
                    "diffgui_logprob_reduction": str(train_cfg.get("diffgui_logprob_reduction", "mean")),
                    "precision": str(train_cfg.get("precision", sweep_cfg.get("precision", "bf16"))),
                    "graph_micro_batch_size": int(train_cfg.get("graph_micro_batch_size", sweep_cfg.get("graph_micro_batch_size", 2))),
                    "freeze": dict(train_cfg.get("freeze", {})),
                }
            },
        )
        actor_metrics = actor.update_policy(policy_batch)
        timing["ppo_seconds"] = time.perf_counter() - phase_start

        total_seconds = time.perf_counter() - started
        memory = _stop_gpu_memory_monitor(monitor)
        torch_peak_mb = _torch_peak_memory_mb(device)
        row = _build_batch_summary_row(
            config=config,
            batch_size=batch_size,
            reward_batch_size=reward_batch_size,
            seed=seed,
            timing=timing,
            total_seconds=total_seconds,
            reconstruction_rows=reconstruction_rows,
            written_mols=written_mols,
            reward_rows=reward_rows,
            rewards_by_graph=rewards_by_graph,
            advantages=advantages,
            policy_batch=policy_batch,
            actor_metrics=actor_metrics,
            torch_peak_mb=torch_peak_mb,
            gpu_memory=memory,
            output_dir=batch_dir,
        )
        (batch_dir / "summary.json").write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
        write_molecule_records_csv(
            batch_dir / "molecules_with_rewards.csv",
            reconstruction_rows=reconstruction_rows,
            reward_rows=reward_rows,
            rewards_by_graph=rewards_by_graph,
            advantages=advantages,
            seed=seed,
        )
        return row
    finally:
        if monitor is not None and not monitor.stop_event.is_set():
            _stop_gpu_memory_monitor(monitor)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file."""
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _read_batch_sizes(sweep_cfg: dict[str, Any]) -> list[int]:
    """Read positive batch sizes from config."""
    values = sweep_cfg.get("batch_sizes", [4, 8, 16])
    if isinstance(values, str):
        raw = [int(value) for value in values.split(",") if value.strip()]
    else:
        raw = [int(value) for value in values]
    return [value for value in raw if value > 0]


def _resolve_path(value: Any) -> Path:
    """Resolve paths relative to project root."""
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _torch_load_checkpoint(checkpoint: Path, device: torch.device) -> dict[str, Any]:
    """Load a torch checkpoint across PyTorch versions."""
    try:
        return torch.load(str(checkpoint), map_location=device, weights_only=False)
    except TypeError:
        return torch.load(str(checkpoint), map_location=device)


def _setup_logging(log_path: Path, log_level: str) -> None:
    """Configure concise terminal and durable file logs."""
    LOGGER.setLevel(getattr(logging, log_level))
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(stream)
    LOGGER.addHandler(file_handler)
    logging.getLogger("DiffSBDD-DDPO").setLevel(logging.INFO)
    logging.getLogger("DiffSBDD-DDPO").addHandler(file_handler)


def _git_snapshot() -> dict[str, str]:
    """Return a small git snapshot for the run log."""
    snapshot: dict[str, str] = {}
    for key, cmd in {
        "commit": ["git", "rev-parse", "--short", "HEAD"],
        "status_short": ["git", "status", "--short"],
    }.items():
        try:
            result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10)
            snapshot[key] = result.stdout.strip()
        except Exception as exc:
            snapshot[key] = f"unavailable:{type(exc).__name__}"
    return snapshot


def _load_diffgui_model(
    *,
    sample_config: Path,
    checkpoint: Path,
    num_timesteps: int,
    device: torch.device,
) -> tuple[torch.nn.Module, Any, Any]:
    """Load a reference or PPO-trained DiffGui checkpoint with corrected timestep buffers."""
    sample_cfg = load_diffgui_config(sample_config)
    train_cfg, model_state, checkpoint_meta = _load_diffgui_checkpoint_state(
        checkpoint=checkpoint,
        device=device,
        visited=set(),
    )
    force_diff_config_num_timesteps(train_cfg.model.diff, num_timesteps)
    featurizer = FeatureComplex(train_cfg.data.transform.ligand_atom_mode, sample=True)
    model = DiffGui(
        config=train_cfg.model,
        protein_node_types=featurizer.protein_feat_dim,
        ligand_node_types=featurizer.atom_feat_dim,
        num_edge_types=featurizer.bond_feat_dim,
    ).to(device)
    load_report = load_diffgui_network_state(model, model_state)
    model.eval()
    LOGGER.info(
        "loaded DiffGui checkpoint=%s checkpoint_meta=%s load_report=%s",
        checkpoint,
        checkpoint_meta,
        load_report,
    )
    LOGGER.info("DiffGui model.num_timesteps=%d", int(model.num_timesteps))
    return model, featurizer, sample_cfg


def _load_diffgui_checkpoint_state(
    *,
    checkpoint: Path,
    device: torch.device,
    visited: set[Path],
) -> tuple[Any, dict[str, torch.Tensor], dict[str, Any]]:
    """Return the base DiffGui config plus model weights from supported checkpoints."""
    resolved = checkpoint.resolve()
    if resolved in visited:
        raise RuntimeError(f"DiffGui checkpoint base chain contains a cycle at {resolved}")
    visited.add(resolved)

    ckpt = _torch_load_checkpoint(resolved, device)
    if "model" in ckpt and "config" in ckpt:
        return (
            ckpt["config"],
            ckpt["model"],
            {
                "format": "reference_diffgui",
                "checkpoint": str(resolved),
                "iteration": ckpt.get("iteration", ""),
            },
        )

    if "model_state_dict" in ckpt:
        saved_config = ckpt.get("config")
        if not isinstance(saved_config, dict):
            raise KeyError(f"DiffGui training checkpoint has no dict config: {resolved}")
        model_cfg = saved_config.get("diffgui_model") or {}
        base_checkpoint_value = model_cfg.get("base_checkpoint") or model_cfg.get("checkpoint")
        if not base_checkpoint_value:
            raise KeyError(f"DiffGui training checkpoint has no base checkpoint in config: {resolved}")
        base_checkpoint = _resolve_path(base_checkpoint_value)
        train_cfg, _, base_meta = _load_diffgui_checkpoint_state(
            checkpoint=base_checkpoint,
            device=device,
            visited=visited,
        )
        return (
            train_cfg,
            ckpt["model_state_dict"],
            {
                "format": "diffgui_ddp_training",
                "checkpoint": str(resolved),
                "epoch": ckpt.get("epoch", ""),
                "stage": ckpt.get("stage", ""),
                "base_checkpoint": str(base_checkpoint),
                "base_meta": base_meta,
            },
        )

    raise KeyError(f"Unsupported DiffGui checkpoint format: {resolved}")


def _build_prompt(
    *,
    sample_cfg: Any,
    featurizer: Any,
    pocket_pdb: Path,
    ligand_sdf: str,
    batch_size: int,
    ligand_atoms: int,
    sweep_cfg: dict[str, Any] | None,
    device: torch.device,
) -> tuple[DiffGuiPromptBatch, np.ndarray, np.ndarray, np.ndarray]:
    """Build the DiffGui prompt tensors for one denovo batch."""
    transform = Compose([featurizer])
    data = _build_pocket_data(pocket_pdb, ligand_sdf)
    data = transform(data)
    ligand_atom_counts = _sample_ligand_atom_counts(
        data=data,
        batch_size=batch_size,
        ligand_atoms=ligand_atoms,
        sweep_cfg=sweep_cfg or {},
    )
    pocket_batch = Batch.from_data_list(
        [data.clone() for _ in range(batch_size)],
        follow_batch=featurizer.follow_batch,
    ).to(device)
    holder = make_data_placeholder(
        n_nodes_list=ligand_atom_counts,
        device=device,
    )
    prompt = DiffGuiPromptBatch(
        protein_node=pocket_batch.protein_atom_feat.float(),
        protein_pos=pocket_batch.protein_pos,
        protein_batch=pocket_batch.protein_element_batch,
        ligand_batch=holder["batch_node"],
        halfedge_index=holder["halfedge_index"],
        halfedge_batch=holder["batch_halfedge"],
        batch_lab=_make_batch_lab(sample_cfg, batch_size, device),
    )
    return (
        prompt,
        holder["batch_node"].detach().cpu().numpy(),
        holder["halfedge_index"].detach().cpu().numpy(),
        holder["batch_halfedge"].detach().cpu().numpy(),
    )


def _sample_ligand_atom_counts(
    *,
    data: Any,
    batch_size: int,
    ligand_atoms: int,
    sweep_cfg: dict[str, Any],
) -> list[int]:
    """Sample per-graph ligand atom counts following DiffGui reference modes."""
    mode = str(sweep_cfg.get("ligand_size_mode", "fixed")).lower()
    if mode == "fixed":
        return [int(ligand_atoms)] * int(batch_size)
    if mode == "priori":
        pocket_pos = data.protein_pos.detach().cpu().numpy()
        pocket_size = get_space_size(pocket_pos)
        return [int(sample_atom_num(pocket_size)) for _ in range(int(batch_size))]
    if mode == "range":
        mean = float(sweep_cfg.get("ligand_atoms_range_mean", 24.923464980477522))
        std = float(sweep_cfg.get("ligand_atoms_range_std", 5.516291901819105))
        counts = np.random.normal(mean, std, size=int(batch_size)).astype("int64")
        min_atoms = int(sweep_cfg.get("ligand_atoms_min", 1))
        max_atoms = sweep_cfg.get("ligand_atoms_max")
        counts = np.maximum(counts, min_atoms)
        if max_atoms is not None:
            counts = np.minimum(counts, int(max_atoms))
        return [int(value) for value in counts]
    if mode == "ref":
        if not hasattr(data, "ligand_element") or len(data.ligand_element) == 0:
            raise ValueError("ligand_size_mode='ref' requires a non-empty reference ligand_sdf.")
        return [int(len(data.ligand_element))] * int(batch_size)
    raise ValueError("diffgui_docking_batch_sweep.ligand_size_mode must be fixed, priori, range, or ref.")


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


def _make_batch_lab(config: Any, batch_size: int, device: torch.device) -> torch.Tensor:
    """Create DiffGui label conditioning tensor for the sample batch."""
    values = [
        float(config.model.logp),
        float(config.model.tpsa),
        float(config.model.sa),
        float(config.model.qed),
        float(config.model.aff),
    ]
    return torch.tensor([values for _ in range(batch_size)], dtype=torch.float32, device=device)


def _write_reconstructed_outputs(
    *,
    policy_batch: DiffGuiPolicyBatch,
    featurizer: Any,
    n_graphs: int,
    batch_node: np.ndarray,
    halfedge_index: np.ndarray,
    batch_halfedge: np.ndarray,
    output_dir: Path,
    name_prefix: str,
) -> tuple[list[dict[str, Any]], int]:
    """Decode final DiffGui states and write generated SDF/CSV artifacts."""
    if policy_batch.pred_node is not None and policy_batch.pred_pos is not None and policy_batch.pred_halfedge is not None:
        pred_node = policy_batch.pred_node.detach().cpu().numpy()
        pred_pos = policy_batch.pred_pos.detach().cpu().numpy()
        pred_halfedge = policy_batch.pred_halfedge.detach().cpu().numpy()
    else:
        pred_node = policy_batch.node_traj[-1].detach().cpu().numpy()
        pred_pos = policy_batch.pos_centered_traj[-1].detach().cpu().numpy()
        pred_halfedge = policy_batch.halfedge_traj[-1].detach().cpu().numpy()
    outputs_np = {
        "pred": [pred_node, pred_pos, pred_halfedge],
        "traj": [
            policy_batch.node_traj.detach().cpu().numpy(),
            policy_batch.pos_centered_traj.detach().cpu().numpy(),
            policy_batch.halfedge_traj.detach().cpu().numpy(),
        ],
    }
    split_outputs = seperate_outputs(outputs_np, n_graphs, batch_node, halfedge_index, batch_halfedge)
    rows: list[dict[str, Any]] = []
    sdf_path = output_dir / "generated.sdf"
    writer = Chem.SDWriter(str(sdf_path))
    writer.SetKekulize(False)
    written = 0
    try:
        for mol_index, output_mol in enumerate(split_outputs):
            row = {
                "mol_index": mol_index,
                "reconstruction_status": "ok",
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
                rdmol = reconstruct_from_generated_with_edges(mol_info, add_edge=None)
                row["smiles"] = Chem.MolToSmiles(rdmol)
                rdmol.SetProp("_Name", f"{name_prefix}_{mol_index:04d}")
                rdmol.SetProp("mol_index", str(mol_index))
                rdmol.SetProp("old_logprob_mean", f"{row['old_logprob_mean']:.8f}")
                rdmol.SetProp("diffgui_num_timesteps", str(int(policy_batch.old_logprob.shape[0])))
                writer.write(rdmol)
                written += 1
            except MolReconsError as exc:
                row["reconstruction_status"] = f"reconstruct_error:{type(exc).__name__}"
            except Exception as exc:
                row["reconstruction_status"] = f"error:{type(exc).__name__}"
            rows.append(row)
    finally:
        writer.close()

    with (output_dir / "reconstruction.csv").open("w", newline="", encoding="utf-8") as handle:
        writer_csv = csv.DictWriter(
            handle,
            fieldnames=["mol_index", "reconstruction_status", "smiles", "num_atoms", "old_logprob_mean"],
        )
        writer_csv.writeheader()
        writer_csv.writerows(rows)
    return rows, written


def _override_nested_key(cfg: dict, keys: tuple, value) -> None:
    """Set a nested dict key by path, creating intermediate dicts when missing."""
    d = cfg
    for k in keys[:-1]:
        if k not in d or not isinstance(d[k], dict):
            d[k] = {}
        d = d[k]
    d[keys[-1]] = value


def _score_reconstructed_sdf(
    *,
    config: dict[str, Any],
    input_sdf: Path,
    output_dir: Path,
    batch_size: int,
) -> list[dict[str, Any]]:
    """Score reconstructed molecules through the separate DiffSBDD reward env."""
    sweep_cfg = config.get("diffgui_docking_batch_sweep", {})
    reward_env = str(sweep_cfg.get("reward_env", "dif"))
    reward_cfg = dict(config)
    # Subprocess has CUDA_VISIBLE_DEVICES=<single_gpu> -> only cuda:0 is valid.
    _override_nested_key(reward_cfg, ("reward", "chembl_classifier", "device"), "cuda:0")
    reward_cfg["diffgui_reward_only"] = {
        "input_sdf": str(input_sdf),
        "output_dir": str(output_dir),
        "mol_id_prefix": "diffgui_dock",
        "batch_size": int(batch_size),
        "device": str(sweep_cfg.get("reward_device", "cpu")),
        "remove_hs": False,
        "strict_connectivity": True,
        "scored_sdf": "docking_scored.sdf",
        "rewards_csv": "docking_rewards.csv",
        "summary_json": "docking_reward_summary.json",
        "manifest_json": "docking_reward_manifest.json",
    }
    reward_config_path = output_dir / "reward_only_config.yaml"
    reward_config_path.write_text(
        yaml.safe_dump(reward_cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    log_path = output_dir / "reward_only_subprocess.log"
    cmd = [
        _conda_executable(),
        "run",
        "-n",
        reward_env,
        "python",
        "ddpo/main_diffgui_reward_only.py",
        "--config",
        str(reward_config_path),
        "--log-level",
        "INFO",
    ]
    LOGGER.info("launch reward-only subprocess env=%s cmd=%s", reward_env, " ".join(cmd))
    reward_cuda = str(sweep_cfg.get("reward_cuda_visible", ""))
    env = os.environ.copy()
    if reward_cuda:
        env["CUDA_VISIBLE_DEVICES"] = reward_cuda
    with log_path.open("w", encoding="utf-8") as log_file:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=int(sweep_cfg.get("reward_timeout_seconds", 1800)),
        )
    if result.returncode != 0:
        raise RuntimeError(f"reward-only subprocess failed rc={result.returncode}; see {log_path}")
    rows = _read_csv_rows(output_dir / "docking_rewards.csv")
    LOGGER.info("reward-only rows=%d log=%s", len(rows), log_path)
    return rows


def _conda_executable() -> str:
    """Resolve a real conda executable for nested reward-only subprocesses."""
    candidates = [
        os.environ.get("CONDA_EXE"),
        shutil.which("conda"),
        str(Path.home() / "miniforge3" / "bin" / "conda"),
        str(Path.home() / "miniconda3" / "bin" / "conda"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    raise FileNotFoundError(f"Could not locate conda executable; checked={candidates}")


def _rewards_by_graph_index(
    *,
    reconstruction_rows: list[dict[str, Any]],
    reward_rows: list[dict[str, Any]],
    failure_penalty: float,
) -> list[float]:
    """Map SDF reward rows back to the original graph indices."""
    rewards = [float(failure_penalty)] * len(reconstruction_rows)
    ok_indices = [int(row["mol_index"]) for row in reconstruction_rows if row.get("reconstruction_status") == "ok"]
    for reward_row in reward_rows:
        graph_index = _reward_graph_index(reward_row, ok_indices)
        if graph_index is None:
            continue
        rewards[graph_index] = _reward_value(reward_row, failure_penalty)
    return rewards


def _reward_value(row: dict[str, Any], failure_penalty: float) -> float:
    """Read the scalar reward used as PPO advantage source."""
    for key in ("reward_total", "reward_total_raw", "docking_reward"):
        value = row.get(key)
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            return numeric
    return float(failure_penalty)


def _make_advantages(
    rewards: list[float],
    *,
    normalize: bool,
    epsilon: float,
    clamp_min: Any,
    clamp_max: Any,
) -> torch.Tensor:
    """Convert scalar rewards to graph-level advantages."""
    values = torch.tensor(rewards, dtype=torch.float32)
    if normalize and values.numel() > 1:
        std = values.std(unbiased=False)
        if float(std.item()) > epsilon:
            values = (values - values.mean()) / (std + epsilon)
        else:
            values = values - values.mean()
    if clamp_min is not None or clamp_max is not None:
        min_value = -float("inf") if clamp_min is None else float(clamp_min)
        max_value = float("inf") if clamp_max is None else float(clamp_max)
        values = values.clamp(min=min_value, max=max_value)
    return values


def _replace_policy_advantages(policy_batch: DiffGuiPolicyBatch, advantages: torch.Tensor) -> DiffGuiPolicyBatch:
    """Replace policy advantages without touching trajectory tensors."""
    policy_batch.advantages = advantages.to(device=policy_batch.advantages.device, dtype=torch.float32)
    policy_batch.validate()
    return policy_batch


def write_molecule_records_csv(
    path: Path,
    *,
    reconstruction_rows: list[dict[str, Any]],
    reward_rows: list[dict[str, Any]],
    rewards_by_graph: list[float],
    advantages: torch.Tensor,
    epoch: int | None = None,
    rank: int | None = None,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Write one clear molecule row per original graph, including failures."""
    path.parent.mkdir(parents=True, exist_ok=True)
    reward_by_graph = _reward_rows_by_graph_index(reconstruction_rows, reward_rows)
    rows: list[dict[str, Any]] = []
    for graph_index, reconstruction_row in enumerate(reconstruction_rows):
        reward_row = reward_by_graph.get(graph_index, {})
        row = _build_molecule_record(
            graph_index=graph_index,
            reconstruction_row=reconstruction_row,
            reward_row=reward_row,
            reward_for_advantage=rewards_by_graph[graph_index] if graph_index < len(rewards_by_graph) else 0.0,
            advantage=float(advantages[graph_index].item()) if graph_index < int(advantages.numel()) else 0.0,
            epoch=epoch,
            rank=rank,
            seed=seed,
        )
        rows.append(row)

    fieldnames = _molecule_record_fieldnames(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fieldnames})
    return rows


def _reward_rows_by_graph_index(
    reconstruction_rows: list[dict[str, Any]],
    reward_rows: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """Map reward-only SDF source_index rows back to original graph indices."""
    ok_indices = [int(row["mol_index"]) for row in reconstruction_rows if row.get("reconstruction_status") == "ok"]
    mapped: dict[int, dict[str, Any]] = {}
    for reward_row in reward_rows:
        graph_index = _reward_graph_index(reward_row, ok_indices)
        if graph_index is None:
            continue
        mapped[graph_index] = reward_row
    return mapped


def _reward_graph_index(reward_row: dict[str, Any], ok_indices: list[int]) -> int | None:
    """Map filtered reward rows back to their original policy graph index."""
    try:
        direct_index = int(float(reward_row.get("source_graph_index", "")))
    except (TypeError, ValueError):
        direct_index = -1
    if direct_index in set(ok_indices):
        return direct_index
    try:
        source_index = int(float(reward_row.get("source_index", -1)))
    except (TypeError, ValueError):
        return None
    if 0 <= source_index < len(ok_indices):
        return ok_indices[source_index]
    return None


def _build_molecule_record(
    *,
    graph_index: int,
    reconstruction_row: dict[str, Any],
    reward_row: dict[str, Any],
    reward_for_advantage: float,
    advantage: float,
    epoch: int | None,
    rank: int | None,
    seed: int | None,
) -> dict[str, Any]:
    """Build one molecule-level audit row."""
    smiles = str(reward_row.get("smiles") or reconstruction_row.get("smiles") or "")
    canonical_smiles, inchikey, scaffold_smiles = _molecule_identity(smiles)
    reconstruction_status = str(reconstruction_row.get("reconstruction_status", "missing_reconstruction_row"))
    rdkit_valid = int(float(reward_row.get("rdkit_valid", 0) or 0))
    docking_raw = _safe_float(reward_row.get("docking_raw"), 0.0)
    docking_success = int(bool(reward_row) and docking_raw != 0.0)
    row: dict[str, Any] = {
        "epoch": "" if epoch is None else int(epoch),
        "rank": "" if rank is None else int(rank),
        "seed": "" if seed is None else int(seed),
        "local_mol_index": int(graph_index),
        "global_mol_id": _global_mol_id(epoch=epoch, rank=rank, graph_index=graph_index),
        "reconstruction_status": reconstruction_status,
        "reconstruction_success": int(reconstruction_status == "ok"),
        "reward_found": int(bool(reward_row)),
        "reward_source_index": reward_row.get("source_index", ""),
        "original_name": reward_row.get("original_name", ""),
        "rdkit_valid": rdkit_valid,
        "sanitize_status": reward_row.get("sanitize_status") or _sanitize_fallback(reconstruction_status),
        "docking_success": docking_success,
        "num_atoms": reconstruction_row.get("num_atoms", reward_row.get("nodesxsample", "")),
        "nodesxsample": reward_row.get("nodesxsample", reconstruction_row.get("num_atoms", "")),
        "smiles": smiles,
        "canonical_smiles": canonical_smiles,
        "inchikey": inchikey,
        "murcko_scaffold": scaffold_smiles,
        "old_logprob_mean": reconstruction_row.get("old_logprob_mean", ""),
        "reward_for_advantage": float(reward_for_advantage),
        "advantage": float(advantage),
    }
    for key, value in reward_row.items():
        if key not in row:
            row[key] = value
    return row


def _sanitize_fallback(reconstruction_status: str) -> str:
    """Return a meaningful sanitize_status when no reward row is available."""
    if reconstruction_status in ("ok", "generated"):
        return "ok"
    if reconstruction_status.startswith("fragment"):
        return f"reconstruct:{reconstruction_status}"
    return f"reconstruct:{reconstruction_status}"


def _molecule_identity(smiles: str) -> tuple[str, str, str]:
    """Return canonical SMILES, InChIKey, and Murcko scaffold when possible."""
    if not smiles:
        return "", "", ""
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        mol = None
    if mol is None:
        return "", "", ""
    canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
    try:
        inchikey = Chem.MolToInchiKey(mol)
    except Exception:
        inchikey = ""
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        scaffold = ""
    return canonical, inchikey, scaffold


def _global_mol_id(*, epoch: int | None, rank: int | None, graph_index: int) -> str:
    """Return a stable molecule id for merged DiffGui records."""
    epoch_text = "e????" if epoch is None else f"e{int(epoch):04d}"
    rank_text = "r??" if rank is None else f"r{int(rank):02d}"
    return f"{epoch_text}_{rank_text}_m{int(graph_index):04d}"


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Parse finite floats from CSV values."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    return numeric if math.isfinite(numeric) else float(default)


def _csv_value(value: Any) -> Any:
    """Convert values into CSV-friendly scalars."""
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _molecule_record_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    """Return readable molecule CSV fields with dynamic reward columns appended."""
    preferred = [
        "epoch",
        "rank",
        "seed",
        "local_mol_index",
        "global_mol_id",
        "reconstruction_status",
        "reconstruction_success",
        "reward_found",
        "reward_source_index",
        "original_name",
        "rdkit_valid",
        "sanitize_status",
        "docking_success",
        "num_atoms",
        "nodesxsample",
        "smiles",
        "canonical_smiles",
        "inchikey",
        "murcko_scaffold",
        "old_logprob_mean",
        "reward_for_advantage",
        "advantage",
        "reward_total",
        "reward_total_raw",
        "stability",
        "docking_raw",
        "docking_reward",
        "ligand_efficiency_raw",
    ]
    fields = [field for field in preferred if any(field in row for row in rows)]
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields


def _build_batch_summary_row(
    *,
    config: dict[str, Any],
    batch_size: int,
    reward_batch_size: int,
    seed: int,
    timing: dict[str, float],
    total_seconds: float,
    reconstruction_rows: list[dict[str, Any]],
    written_mols: int,
    reward_rows: list[dict[str, Any]],
    rewards_by_graph: list[float],
    advantages: torch.Tensor,
    policy_batch: DiffGuiPolicyBatch,
    actor_metrics: dict[str, float],
    torch_peak_mb: float,
    gpu_memory: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Build one flat summary row for CSV/JSON outputs."""
    reward_values = np.asarray(rewards_by_graph, dtype=np.float64)
    docking_success = sum(1 for row in reward_rows if float(row.get("docking_raw", 0.0) or 0.0) != 0.0)
    valid_count = sum(1 for row in reward_rows if int(row.get("rdkit_valid", 0) or 0) == 1)
    docking_success_ratio_scored = float(docking_success / len(reward_rows)) if reward_rows else 0.0
    docking_success_ratio_total = float(docking_success / len(reconstruction_rows)) if reconstruction_rows else 0.0
    ligand_counts = _policy_ligand_atom_counts(policy_batch, batch_size)
    row: dict[str, Any] = {
        "status": "ok",
        "batch_size": int(batch_size),
        "reward_batch_size": int(reward_batch_size),
        "seed": int(seed),
        "num_timesteps": int(config.get("diffgui_docking_batch_sweep", {}).get("num_timesteps", 100)),
        "ligand_size_mode": str(config.get("diffgui_docking_batch_sweep", {}).get("ligand_size_mode", "fixed")),
        "ligand_atoms": int(round(float(np.mean(ligand_counts)))) if ligand_counts.size else 0,
        "ligand_atoms_config": int(config.get("diffgui_docking_batch_sweep", {}).get("ligand_atoms", 24)),
        "ligand_atoms_min": int(np.min(ligand_counts)) if ligand_counts.size else 0,
        "ligand_atoms_mean": float(np.mean(ligand_counts)) if ligand_counts.size else 0.0,
        "ligand_atoms_max": int(np.max(ligand_counts)) if ligand_counts.size else 0,
        "total_seconds": float(total_seconds),
        **{key: float(value) for key, value in timing.items()},
        "seconds_per_graph": float(total_seconds / batch_size),
        "reconstruction_attempted": int(len(reconstruction_rows)),
        "reconstruction_success": int(written_mols),
        "reconstruction_success_ratio": float(written_mols / len(reconstruction_rows)) if reconstruction_rows else 0.0,
        "reward_records": int(len(reward_rows)),
        "reward_rdkit_valid": int(valid_count),
        "docking_success": int(docking_success),
        "docking_success_ratio": docking_success_ratio_total,
        "docking_success_ratio_total": docking_success_ratio_total,
        "docking_success_ratio_scored": docking_success_ratio_scored,
        "reward_total_mean": float(np.mean(reward_values)) if reward_values.size else 0.0,
        "reward_total_min": float(np.min(reward_values)) if reward_values.size else 0.0,
        "reward_total_max": float(np.max(reward_values)) if reward_values.size else 0.0,
        "advantage_mean": float(advantages.mean().item()) if advantages.numel() else 0.0,
        "advantage_std": float(advantages.std(unbiased=False).item()) if advantages.numel() else 0.0,
        "old_logprob_shape": list(policy_batch.old_logprob.shape),
        "torch_peak_allocated_mb": float(torch_peak_mb),
        "gpu_monitor_peak_used_mb": float(gpu_memory.get("peak_used_mb", 0.0)),
        "outputs_generated_sdf": str(output_dir / "generated.sdf"),
        "outputs_scored_sdf": str(output_dir / "docking_scored.sdf"),
        "outputs_reward_csv": str(output_dir / "docking_rewards.csv"),
        "outputs_molecule_csv": str(output_dir / "molecules_with_rewards.csv"),
    }
    row.update({key: float(value) for key, value in actor_metrics.items() if _is_number(value)})
    return row


def _policy_ligand_atom_counts(policy_batch: DiffGuiPolicyBatch, batch_size: int) -> np.ndarray:
    """Return per-graph ligand atom counts from the policy batch graph ids."""
    graph_ids = policy_batch.ligand_batch.detach().cpu().long()
    counts = torch.bincount(graph_ids, minlength=int(batch_size))
    return counts.numpy().astype(np.int64)


def _is_number(value: Any) -> bool:
    """Return True for finite numeric values."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric)


def _write_all_molecule_csv(
    path: Path,
    reconstruction_rows: list[dict[str, Any]],
    rewards_by_graph: list[float],
    advantages: torch.Tensor,
) -> None:
    """Write per-graph reconstruction and reward values."""
    fieldnames = [
        "mol_index",
        "reconstruction_status",
        "smiles",
        "num_atoms",
        "old_logprob_mean",
        "reward_for_advantage",
        "advantage",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(reconstruction_rows):
            out = dict(row)
            out["reward_for_advantage"] = rewards_by_graph[index]
            out["advantage"] = float(advantages[index].item())
            writer.writerow({key: out.get(key, "") for key in fieldnames})


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    """Read CSV rows into dictionaries."""
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_summary_artifacts(output_dir: Path, rows: list[dict[str, Any]], config: dict[str, Any]) -> None:
    """Write sweep CSV/JSON and a compact plot."""
    csv_path = output_dir / "batch_sweep_summary.csv"
    fieldnames = _union_fieldnames(rows)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    summary = {
        "stage": "diffgui_docking_batch_sweep",
        "config_name": config.get("name", ""),
        "rows": rows,
        "summary_csv": str(csv_path),
    }
    (output_dir / "batch_sweep_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_summary_plot(output_dir / "batch_sweep_summary.png", rows)


def _union_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    """Return stable CSV fieldnames."""
    preferred = [
        "status",
        "batch_size",
        "reward_batch_size",
        "total_seconds",
        "seconds_per_graph",
        "rollout_seconds",
        "reward_seconds",
        "ppo_seconds",
        "reconstruction_success",
        "reconstruction_success_ratio",
        "docking_success",
        "docking_success_ratio",
        "docking_success_ratio_total",
        "docking_success_ratio_scored",
        "reward_total_mean",
        "PPO_Loss",
        "PPO_ApproxKL",
        "torch_peak_allocated_mb",
        "gpu_monitor_peak_used_mb",
    ]
    keys = [key for key in preferred if any(key in row for row in rows)]
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    return keys


def _write_summary_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a quick PNG plot for the batch sweep."""
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if not ok_rows:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        LOGGER.warning("matplotlib unavailable, skip plot: %s", exc)
        return

    batch = [int(row["batch_size"]) for row in ok_rows]
    total = [float(row.get("total_seconds", 0.0)) for row in ok_rows]
    peak = [float(row.get("gpu_monitor_peak_used_mb") or row.get("torch_peak_allocated_mb") or 0.0) for row in ok_rows]
    recon = [float(row.get("reconstruction_success_ratio", 0.0)) for row in ok_rows]
    dock = [float(row.get("docking_success_ratio", 0.0)) for row in ok_rows]

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    axes[0].plot(batch, total, marker="o", color="#2E6F9E")
    axes[0].set_xlabel("batch size")
    axes[0].set_ylabel("seconds")
    axes[0].set_title("Total Time")
    axes[1].plot(batch, peak, marker="o", color="#7A5C99")
    axes[1].set_xlabel("batch size")
    axes[1].set_ylabel("MB")
    axes[1].set_title("Peak GPU Memory")
    axes[2].plot(batch, recon, marker="o", label="reconstruction", color="#4B8B3B")
    axes[2].plot(batch, dock, marker="s", label="docking", color="#C46C3A")
    axes[2].set_xlabel("batch size")
    axes[2].set_ylim(-0.02, 1.02)
    axes[2].set_title("Success Ratio")
    axes[2].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _start_gpu_memory_monitor(gpu_id: str, output_dir: Path, poll_interval: float) -> GpuMemoryMonitor | None:
    """Start background nvidia-smi polling for one physical GPU."""
    if poll_interval <= 0:
        poll_interval = 1.0
    sample_path = output_dir / "gpu_memory_samples.csv"
    samples: list[dict[str, Any]] = []
    errors: list[str] = []
    stop_event = threading.Event()
    fieldnames = ["elapsed_seconds", "timestamp_unix", "gpu", "memory_used_mb", "memory_free_mb"]
    with sample_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()

    start_time = time.perf_counter()

    def _poll() -> None:
        while not stop_event.is_set():
            elapsed = time.perf_counter() - start_time
            now = time.time()
            try:
                row = _query_gpu_memory(gpu_id)
                if row:
                    sample = {
                        "elapsed_seconds": float(elapsed),
                        "timestamp_unix": float(now),
                        **row,
                    }
                    samples.append(sample)
                    with sample_path.open("a", newline="", encoding="utf-8") as file_obj:
                        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
                        writer.writerow(sample)
            except (OSError, subprocess.CalledProcessError, ValueError) as exc:
                errors.append(str(exc))
            stop_event.wait(poll_interval)

    thread = threading.Thread(target=_poll, name="diffgui-docking-gpu-memory", daemon=True)
    thread.start()
    return GpuMemoryMonitor(
        gpu_id=gpu_id,
        stop_event=stop_event,
        thread=thread,
        samples=samples,
        errors=errors,
        sample_path=sample_path,
    )


def _stop_gpu_memory_monitor(monitor: GpuMemoryMonitor | None) -> dict[str, Any]:
    """Stop the monitor and return a compact memory summary."""
    if monitor is None:
        return {}
    monitor.stop_event.set()
    monitor.thread.join(timeout=3.0)
    peak = max((int(row["memory_used_mb"]) for row in monitor.samples), default=0)
    return {
        "gpu_id": monitor.gpu_id,
        "sample_path": str(monitor.sample_path),
        "sample_count": len(monitor.samples),
        "peak_used_mb": peak,
        "errors": monitor.errors[:10],
    }


def _query_gpu_memory(gpu_id: str) -> dict[str, Any] | None:
    """Query current used/free memory for one physical GPU id."""
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    for line in output.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        if parts[0] == str(gpu_id):
            return {
                "gpu": parts[0],
                "memory_used_mb": int(parts[1]),
                "memory_free_mb": int(parts[2]),
            }
    return None


def _physical_gpu_id_from_config(config: dict[str, Any], device: torch.device) -> str:
    """Resolve the physical GPU id used by torch and UniDock."""
    docking_cfg = config.get("reward", {}).get("docking", {})
    if "gpu_id" in docking_cfg:
        return str(int(docking_cfg["gpu_id"]))
    if device.type == "cuda" and device.index is not None:
        return str(device.index)
    return "0"


def _torch_peak_memory_mb(device: torch.device) -> float:
    """Return torch peak allocated memory in MB."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))


if __name__ == "__main__":
    main()
