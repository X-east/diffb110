#!/usr/bin/env python3
"""Export a trained offline actor into a native DiffSBDD Lightning checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def _sha256_file(path: Path) -> str:
    """Return the SHA256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    """Parse immutable source and output checkpoint paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--offline-checkpoint", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--metadata-json", type=Path, required=True)
    return parser.parse_args()


def _actor_state(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Extract and normalize the DiffSBDD policy state from one trainer checkpoint."""
    try:
        raw_state = payload["trainer"]["actor_updater"]["policy"]
    except KeyError as exc:
        raise KeyError("offline checkpoint lacks trainer.actor_updater.policy") from exc
    if not isinstance(raw_state, dict):
        raise TypeError("offline actor policy state must be a mapping")
    normalized: dict[str, torch.Tensor] = {}
    for name, tensor in raw_state.items():
        if not isinstance(name, str) or not name.startswith("model."):
            raise ValueError(f"unexpected actor state key: {name!r}")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"actor state {name!r} is not a tensor")
        normalized[name.removeprefix("model.")] = tensor
    return normalized


def main() -> int:
    """Write a native checkpoint with exact trained actor weights."""
    args = _parse_args()
    for path in (args.base_checkpoint, args.offline_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_checkpoint.exists() or args.metadata_json.exists():
        raise FileExistsError("export outputs already exist")

    base = torch.load(args.base_checkpoint, map_location="cpu")
    offline = torch.load(args.offline_checkpoint, map_location="cpu")
    if not isinstance(base, dict) or not isinstance(base.get("state_dict"), dict):
        raise TypeError("base checkpoint is not a Lightning state_dict checkpoint")
    if not isinstance(offline, dict):
        raise TypeError("offline checkpoint payload must be a mapping")
    actor = _actor_state(offline)
    base_state = base["state_dict"]
    if set(actor) != set(base_state):
        missing = sorted(set(base_state) - set(actor))[:5]
        extra = sorted(set(actor) - set(base_state))[:5]
        raise ValueError(f"actor/base state keys differ: missing={missing}, extra={extra}")
    for name, tensor in actor.items():
        if tuple(tensor.shape) != tuple(base_state[name].shape):
            raise ValueError(f"shape mismatch for {name}: {tensor.shape} != {base_state[name].shape}")
        base_state[name] = tensor.detach().cpu().clone()

    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(base, args.output_checkpoint)
    exported = torch.load(args.output_checkpoint, map_location="cpu")
    if exported["state_dict"].keys() != actor.keys():
        raise RuntimeError("exported checkpoint state keys do not match actor state")
    if any(not torch.equal(exported["state_dict"][name], tensor.cpu()) for name, tensor in actor.items()):
        raise RuntimeError("exported checkpoint tensors do not exactly match actor state")

    metadata = {
        "format": "diffsbdd_offline_actor_export_v1",
        "base_checkpoint": str(args.base_checkpoint.resolve()),
        "base_checkpoint_sha256": _sha256_file(args.base_checkpoint),
        "offline_checkpoint": str(args.offline_checkpoint.resolve()),
        "offline_checkpoint_sha256": _sha256_file(args.offline_checkpoint),
        "output_checkpoint": str(args.output_checkpoint.resolve()),
        "output_checkpoint_sha256": _sha256_file(args.output_checkpoint),
        "state_tensor_count": len(actor),
        "offline_epoch": offline.get("epoch"),
        "offline_global_step": offline.get("trainer", {}).get("global_step"),
    }
    args.metadata_json.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
