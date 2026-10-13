"""Continue the completed one-epoch Global-DPO variants through epoch 10."""

from __future__ import annotations

import gc
import importlib.util
import json
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path("/data/disk6/DiffB110")
SOURCE_RUN = ROOT / "analysis/diffsbdd_global_dpo_t1000_full_pilot_260823"
OUTPUT_DIR = SOURCE_RUN / "continue_to10"
PILOT_PATH = ROOT / "analysis/diffsbdd_global_dpo_t1000_pilot_260822/run_pilot.py"
TOTAL_EPOCHS = 10
PAIR_COUNT = 17_289
BATCH_PAIRS = 1
LEARNING_RATE = 1e-6
DEVICE = "cuda:0"
VARIANTS = {
    "native_joint": (3.0 / 13.0, 10.0 / 13.0),
    "component_balanced": (0.5, 0.5),
    "gradient_balanced": (1.0, 1.0),
}


def load_pilot_module():
    """Load the existing pilot implementation without modifying it."""
    spec = importlib.util.spec_from_file_location("global_dpo_pilot", PILOT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import pilot module from {PILOT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_logging() -> logging.Logger:
    """Create the durable continuation log."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("global_dpo_continue_to10")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(OUTPUT_DIR / "continue_to10.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def save_metrics(rows: list[dict[str, object]]) -> None:
    """Write accumulated scalar metrics after every completed epoch."""
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "metrics.csv", index=False)


def save_status(payload: dict[str, object]) -> None:
    """Write a machine-readable status snapshot."""
    (OUTPUT_DIR / "status.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def continue_variant(
    pilot,
    variant: str,
    weights: tuple[float, float],
    train_pairs,
    val_pairs,
    cache,
    rows: list[dict[str, object]],
    logger: logging.Logger,
) -> None:
    """Continue one variant from epoch 1 through epoch 10."""
    source_checkpoint = SOURCE_RUN / f"pilot_{variant}.ckpt"
    if not source_checkpoint.exists():
        raise FileNotFoundError(source_checkpoint)

    current = pilot.make_model(DEVICE)
    payload = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    current.model.load_state_dict(payload["state_dict"], strict=True)
    optimizer = torch.optim.Adam(current.model.parameters(), lr=LEARNING_RATE)
    logger.info(
        "variant=%s start_epoch=1 source=%s optimizer_state=reset",
        variant,
        source_checkpoint,
    )

    for epoch in range(1, TOTAL_EPOCHS):
        current.model.train()
        rng = random.Random(pilot.SEED + epoch)
        shuffled = list(train_pairs)
        rng.shuffle(shuffled)
        epoch_losses: list[float] = []
        for batch_start in range(0, len(shuffled), BATCH_PAIRS):
            batch_pairs = shuffled[batch_start : batch_start + BATCH_PAIRS]
            items = pilot.flatten_pairs(batch_pairs, cache)
            timestep = rng.randint(1, pilot.T)
            x_cur, v_cur = pilot.component_losses(
                current, items, timestep, pilot.SEED + epoch + batch_start
            )
            with torch.no_grad():
                x_ref, v_ref = pilot.component_losses(
                    reference, items, timestep, pilot.SEED + epoch + batch_start
                )
            cur_score = -(weights[0] * x_cur + weights[1] * v_cur)
            ref_score = -(weights[0] * x_ref + weights[1] * v_ref)
            cur_margin = cur_score[0::2] - cur_score[1::2]
            ref_margin = ref_score[0::2] - ref_score[1::2]
            loss = -torch.nn.functional.logsigmoid(
                (cur_margin - ref_margin)
            ).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(current.model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

        validation = pilot.evaluate_pairs(
            current,
            val_pairs,
            cache,
            weights,
            pilot.SEED + epoch * 1000,
            BATCH_PAIRS,
        )
        row = {
            "variant": variant,
            "epoch": epoch,
            "train_dpo_loss": float(np.mean(epoch_losses)),
            "weights_x": weights[0],
            "weights_v": weights[1],
            "gradient_v_weight": 1.0,
            **validation,
        }
        rows.append(row)
        save_metrics(rows)
        checkpoint_path = OUTPUT_DIR / f"pilot_{variant}_epoch_{epoch:04d}.ckpt"
        torch.save(
            {
                "state_dict": current.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "variant": variant,
                "weights": weights,
                "T": pilot.T,
                "epoch": epoch,
                "source_checkpoint": str(source_checkpoint),
            },
            checkpoint_path,
        )
        save_status(
            {
                "status": "running",
                "variant": variant,
                "completed_epoch": epoch,
                "total_epochs": TOTAL_EPOCHS,
                "checkpoint": str(checkpoint_path),
            }
        )
        logger.info(
            "variant=%s epoch=%d/%d loss=%.6f val_accuracy=%.6f val_margin=%.6f checkpoint=%s",
            variant,
            epoch,
            TOTAL_EPOCHS - 1,
            row["train_dpo_loss"],
            row["pair_accuracy"],
            row["mean_margin"],
            checkpoint_path,
        )

    del optimizer, current
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    """Run all three continuation jobs sequentially."""
    logger = configure_logging()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this continuation run")
    pilot = load_pilot_module()
    frame = pilot.load_rows()
    train_pairs = pilot.make_pairs(frame, "train", PAIR_COUNT, pilot.SEED)
    val_pairs = pilot.make_pairs(frame, "validation", PAIR_COUNT // 4, pilot.SEED + 1)
    if len(train_pairs) != PAIR_COUNT or len(val_pairs) != 3548:
        raise RuntimeError(f"unexpected pair counts: train={len(train_pairs)} val={len(val_pairs)}")

    global reference
    reference = pilot.make_model(DEVICE)
    reference.model.eval()
    for parameter in reference.model.parameters():
        parameter.requires_grad_(False)
    cache = pilot.ShardCache()
    rows: list[dict[str, object]] = []
    save_status(
        {
            "status": "running",
            "total_epochs": TOTAL_EPOCHS,
            "train_pairs": len(train_pairs),
            "validation_pairs": len(val_pairs),
            "batch_pairs": BATCH_PAIRS,
            "learning_rate": LEARNING_RATE,
            "device": DEVICE,
        }
    )
    logger.info(
        "started continuation train_pairs=%d val_pairs=%d variants=%s",
        len(train_pairs),
        len(val_pairs),
        ",".join(VARIANTS),
    )
    for variant, weights in VARIANTS.items():
        continue_variant(
            pilot,
            variant,
            weights,
            train_pairs,
            val_pairs,
            cache,
            rows,
            logger,
        )
    save_status({"status": "completed", "total_epochs": TOTAL_EPOCHS, "rows": len(rows)})
    logger.info("completed continuation through epoch %d", TOTAL_EPOCHS)


if __name__ == "__main__":
    main()
