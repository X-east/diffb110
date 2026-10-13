"""Export a DiffGui DDP training checkpoint into the reference sample.py format.

Reference DiffGui ``scripts/sample.py`` expects a checkpoint dict with:
``{"model": state_dict, "config": train_config, "iteration": int}``.
Project DDP checkpoints instead store ``model_state_dict`` plus a project-level
training config, so the historical sampler cannot load them directly. This
script rebuilds a reference-format snapshot by pairing the DDP weights with the
train config of the reference base checkpoint, exactly like
``scripts/diffgui_docking_batch_sweep.py::_load_diffgui_checkpoint_state`` does
at runtime.

Timestep-derived buffers (``pos_transition.``, ``node_transition.``,
``edge_transition.``, ``time_emb.``) are intentionally left untouched; the
sampler regenerates them from the sampling step-count override.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIFFGUI_ROOT = PROJECT_ROOT / "reference project" / "代码" / "DiffGui"

TIMESTEP_DERIVED_STATE_PREFIXES = (
    "pos_transition.",
    "node_transition.",
    "edge_transition.",
    "time_emb.",
)

LOGGER = logging.getLogger("export_diffgui_reference_ckpt")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the checkpoint export."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ddp-checkpoint", required=True, type=Path, help="Project DDP training checkpoint.")
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=DIFFGUI_ROOT / "ckpt" / "trained.pt",
        help="Reference DiffGui checkpoint providing the train config.",
    )
    parser.add_argument("--output", required=True, type=Path, help="Output reference-format checkpoint path.")
    parser.add_argument("--report-json", type=Path, help="Optional JSON report path.")
    parser.add_argument("--expect-epoch", type=int, help="Optional expected epoch value check.")
    return parser.parse_args()


def _torch_load(path: Path) -> dict[str, Any]:
    """Load a torch checkpoint with CPU-only mapping."""
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def _compare_state_dicts(base_state: dict[str, Any], ddp_state: dict[str, Any]) -> dict[str, Any]:
    """Compare reference and DDP weights, tolerating only timestep-derived buffers."""
    base_keys = set(base_state)
    ddp_keys = set(ddp_state)
    only_base = sorted(base_keys - ddp_keys)
    only_ddp = sorted(ddp_keys - base_keys)
    if only_base or only_ddp:
        raise RuntimeError(
            "state dict key sets differ: only_base=%s only_ddp=%s" % (only_base[:10], only_ddp[:10])
        )

    mismatched: list[dict[str, Any]] = []
    for key in sorted(base_keys):
        base_shape = tuple(getattr(base_state[key], "shape", ()))
        ddp_shape = tuple(getattr(ddp_state[key], "shape", ()))
        if base_shape != ddp_shape:
            mismatched.append(
                {
                    "key": key,
                    "base_shape": list(base_shape),
                    "ddp_shape": list(ddp_shape),
                    "timestep_derived": key.startswith(TIMESTEP_DERIVED_STATE_PREFIXES),
                }
            )
    unexpected = [item for item in mismatched if not item["timestep_derived"]]
    if unexpected:
        raise RuntimeError(f"unexpected non-timestep shape mismatches: {unexpected[:10]}")
    return {
        "tensors": len(ddp_keys),
        "shape_mismatches": len(mismatched),
        "timestep_derived_mismatches": len(mismatched),
        "non_timestep_mismatches": 0,
    }


def main() -> None:
    """Build the reference-format snapshot and write a durable report."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    ddp_path = args.ddp_checkpoint.resolve()
    base_path = args.base_checkpoint.resolve()
    output_path = args.output.resolve()
    if not ddp_path.is_file():
        raise FileNotFoundError(f"DDP checkpoint not found: {ddp_path}")
    if not base_path.is_file():
        raise FileNotFoundError(f"base checkpoint not found: {base_path}")

    ddp_ckpt = _torch_load(ddp_path)
    if "model_state_dict" not in ddp_ckpt:
        raise KeyError(f"DDP checkpoint has no model_state_dict: {ddp_path}")
    base_ckpt = _torch_load(base_path)
    if "model" not in base_ckpt or "config" not in base_ckpt:
        raise KeyError(f"base checkpoint is not reference format: {base_path}")

    ddp_state = dict(ddp_ckpt["model_state_dict"])
    if any(key.startswith("module.") for key in ddp_state):
        raise RuntimeError("DDP state dict unexpectedly keeps a 'module.' prefix")
    comparison = _compare_state_dicts(dict(base_ckpt["model"]), ddp_state)

    ddp_config = ddp_ckpt.get("config")
    if not isinstance(ddp_config, dict):
        raise KeyError(f"DDP checkpoint has no dict config: {ddp_path}")
    model_cfg = ddp_config.get("diffgui_model") or {}
    training_timesteps = model_cfg.get("num_timesteps") or model_cfg.get("sampling_num_timesteps")
    epoch = ddp_ckpt.get("epoch")
    if args.expect_epoch is not None and epoch != args.expect_epoch:
        raise RuntimeError(f"checkpoint epoch={epoch}, expected {args.expect_epoch}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot: dict[str, Any] = {
        "config": base_ckpt["config"],
        "model": ddp_state,
        "iteration": base_ckpt.get("iteration", ""),
        "epoch": epoch,
        "source_checkpoint": str(ddp_path),
        "source_stage": str(ddp_ckpt.get("stage", "")),
        "source_training_timesteps": training_timesteps,
    }
    torch.save(snapshot, str(output_path))

    report = {
        "output_checkpoint": str(output_path),
        "source_ddp_checkpoint": str(ddp_path),
        "base_checkpoint": str(base_path),
        "epoch": epoch,
        "source_stage": snapshot["source_stage"],
        "source_training_timesteps": training_timesteps,
        "base_config_num_timesteps": int(dict(dict(base_ckpt["config"])["model"])["diff"].num_timesteps),
        "state_dict": comparison,
        "sampling_note": (
            "sampling step count must be set via sample.num_timesteps in the sampler YAML; "
            "timestep-derived buffers are regenerated by scripts/sample.py"
        ),
    }
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    LOGGER.info("export report=%s", json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
