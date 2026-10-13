"""Evaluate saved Global-DPO checkpoints on the frozen validation pairs."""

from __future__ import annotations

import gc
import importlib.util
import json
from pathlib import Path

import pandas as pd
import torch


ROOT = Path("/data/disk6/DiffB110")
RUN_DIR = ROOT / "analysis/diffsbdd_global_dpo_t1000_full_pilot_260823"
PILOT_PATH = ROOT / "analysis/diffsbdd_global_dpo_t1000_pilot_260822/run_pilot.py"


def load_pilot_module():
    """Load the existing pilot helpers without changing their source."""
    spec = importlib.util.spec_from_file_location("global_dpo_pilot", PILOT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import pilot module from {PILOT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate() -> None:
    """Evaluate reference and every checkpoint currently present."""
    pilot = load_pilot_module()
    frame = pilot.load_rows()
    pairs = pilot.make_pairs(frame, "validation", 4322, pilot.SEED + 1)
    cache = pilot.ShardCache()
    variants = {
        "native_joint": (3.0 / 13.0, 10.0 / 13.0),
        "component_balanced": (0.5, 0.5),
    }
    rows: list[dict[str, object]] = []
    for variant, weights in variants.items():
        reference = pilot.make_model("cuda:0")
        reference_result = pilot.evaluate_pairs(
            reference, pairs, cache, weights, pilot.SEED, batch_pairs=4
        )
        rows.append(
            {
                "model": "reference",
                "variant": variant,
                "checkpoint": str(pilot.CHECKPOINT),
                "weights_x": weights[0],
                "weights_v": weights[1],
                **reference_result,
            }
        )
        del reference
        gc.collect()
        torch.cuda.empty_cache()

        checkpoint_path = RUN_DIR / f"pilot_{variant}.ckpt"
        if not checkpoint_path.exists():
            continue
        current = pilot.make_model("cuda:0")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        current.model.load_state_dict(payload["state_dict"], strict=True)
        current_result = pilot.evaluate_pairs(
            current, pairs, cache, weights, pilot.SEED, batch_pairs=4
        )
        rows.append(
            {
                "model": "trained",
                "variant": variant,
                "checkpoint": str(checkpoint_path),
                "weights_x": weights[0],
                "weights_v": weights[1],
                **current_result,
            }
        )
        del current
        gc.collect()
        torch.cuda.empty_cache()

    result = pd.DataFrame(rows)
    result.to_csv(RUN_DIR / "saved_checkpoint_validation.csv", index=False)
    summary = {
        "validation_pairs": len(pairs),
        "timestep": 500,
        "batch_pairs": 4,
        "rows": rows,
        "note": "The saved pilot itself used batch_pairs=1; this audit uses batch_pairs=4 for throughput.",
    }
    (RUN_DIR / "saved_checkpoint_validation.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(result.to_string(index=False))


if __name__ == "__main__":
    evaluate()
