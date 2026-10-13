"""Export a Global DPO checkpoint as a DiffSBDD Lightning checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    """Parse source, base, and destination checkpoint paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dpo-checkpoint", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def export_checkpoint(dpo_checkpoint: Path, base_checkpoint: Path, output: Path) -> dict[str, object]:
    """Validate matching model keys and write a generation-compatible checkpoint."""
    dpo_payload = torch.load(dpo_checkpoint, map_location="cpu", weights_only=False)
    base_payload = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
    dpo_state = dpo_payload.get("model_state_dict")
    base_state = base_payload.get("state_dict")
    if not isinstance(dpo_state, dict) or not isinstance(base_state, dict):
        raise TypeError("expected DPO model_state_dict and base Lightning state_dict")
    if set(dpo_state) != set(base_state):
        missing = sorted(set(base_state).difference(dpo_state))
        unexpected = sorted(set(dpo_state).difference(base_state))
        raise ValueError(f"state_dict keys differ: missing={missing[:5]} unexpected={unexpected[:5]}")
    for key in base_state:
        if tuple(dpo_state[key].shape) != tuple(base_state[key].shape):
            raise ValueError(f"state_dict shape differs for {key}: {dpo_state[key].shape} vs {base_state[key].shape}")
    exported = dict(base_payload)
    exported["state_dict"] = dpo_state
    exported["global_dpo_source"] = str(dpo_checkpoint)
    exported["global_dpo_epoch"] = dpo_payload.get("epoch")
    exported["global_dpo_variant"] = dpo_payload.get("resolved_config", {}).get("variant")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(exported, output)
    return {"output": str(output), "tensors": len(dpo_state), "epoch": exported["global_dpo_epoch"]}


def main() -> None:
    """Validate matching model keys and write a generation-compatible checkpoint."""
    args = parse_args()
    result = export_checkpoint(args.dpo_checkpoint, args.base_checkpoint, args.output)
    print(f"exported={result['output']} tensors={result['tensors']} epoch={result['epoch']}")


if __name__ == "__main__":
    main()
