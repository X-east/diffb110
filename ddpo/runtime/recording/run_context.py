from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class RunContext:
    run_id: str
    run_name: str
    started_at: str
    output_dir: str
    logs_dir: str
    epochs_dir: str
    checkpoints_dir: str
    lightning_logs_dir: str
    config_snapshot_path: str
    training_summary_csv_path: str
    run_manifest_path: str
    checkpoint_manifest_path: str


def _substitute_output_template(output_dir_template: str, run_name: str, now: datetime) -> str:
    date_str = now.strftime("%Y%m%d")
    time_str = now.strftime("%H%M%S")
    output_dir = output_dir_template
    output_dir = output_dir.replace("<run_name>", run_name)
    output_dir = output_dir.replace("<YYYYMMDD>", date_str)
    output_dir = output_dir.replace("<seq>", time_str)
    return output_dir


def create_run_context(config: dict, save_path: Optional[str] = None) -> RunContext:
    env_output_dir = os.environ.get("DIFFSBDD_RUN_OUTPUT_DIR")
    output_dir_template = (
        env_output_dir
        or save_path
        or config.get("output", {}).get("dir", "./exp/diffsbdd-ddpo-lightning")
    )

    now = datetime.now()
    run_name = config.get("name", "diffsbdd-ddpo").replace(" ", "-").lower()
    output_dir = _substitute_output_template(output_dir_template, run_name, now)

    output_path = Path(output_dir)
    logs_dir = str(output_path / "logs")
    epochs_dir = str(output_path / "epochs")
    checkpoints_dir = str(output_path / "checkpoints")
    lightning_logs_dir = str(output_path / "lightning_logs")

    for path in [
        output_path,
        Path(logs_dir),
        Path(epochs_dir),
        Path(checkpoints_dir),
        Path(lightning_logs_dir),
    ]:
        path.mkdir(parents=True, exist_ok=True)

    run_id = now.isoformat(timespec="seconds")
    started_at = now.isoformat(timespec="seconds")

    return RunContext(
        run_id=run_id,
        run_name=run_name,
        started_at=started_at,
        output_dir=str(output_path),
        logs_dir=logs_dir,
        epochs_dir=epochs_dir,
        checkpoints_dir=checkpoints_dir,
        lightning_logs_dir=lightning_logs_dir,
        config_snapshot_path=str(output_path / "config.yaml"),
        training_summary_csv_path=str(output_path / "training_summary.csv"),
        run_manifest_path=str(output_path / "run_manifest.json"),
        checkpoint_manifest_path=str(output_path / "checkpoint_manifest.csv"),
    )


def write_run_manifest(run_context: RunContext, payload: dict) -> None:
    with open(run_context.run_manifest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
