"""Summarize the DiffSBDD-xhe DiffGui-style scratch long test."""

from __future__ import annotations

import csv
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

_cache_root = Path(tempfile.gettempdir()) / "diffsbdd_xhe_longtest_cache"
_cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_root))
os.environ.setdefault("MPLCONFIGDIR", str(_cache_root / "matplotlib"))

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RUN_NAME = "diffsbdd-xhe-diffgui-scratch-pocket1000-8gpu-260606-b2acc4-8ep"
LIGHTNING_DIR = ROOT / "lightning" / RUN_NAME
METRICS_PATH = LIGHTNING_DIR / "metrics" / "metrics.csv"
LOG_PATH = ROOT / "train_xhe_diffgui_scratch_pocket1000_8gpu_b2acc4_8ep.log"
CHECKPOINT_DIR = LIGHTNING_DIR / "checkpoints"


@dataclass
class ProgressLine:
    """One parsed rank-0 train progress line."""

    epoch: int
    total_epochs: int
    batch_idx: int
    global_step: int
    elapsed_min: float
    loss: float | None
    edge_acc: float | None
    real_bond_acc: float | None
    edge_loss_weighted: float | None
    bond_length_loss_weighted: float | None


def _to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _to_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(float(value))


def _format_number(value: float | int | None) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    if abs(value) >= 100:
        return f"{value:.3g}"
    if abs(value) >= 1:
        return f"{value:.4g}"
    return f"{value:.4f}"


def read_metrics() -> list[dict[str, Any]]:
    """Read Lightning metrics rows, keeping numeric values where available."""

    if not METRICS_PATH.exists():
        return []
    rows: list[dict[str, Any]] = []
    with METRICS_PATH.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            row: dict[str, Any] = {}
            for key, value in raw_row.items():
                if key is None:
                    continue
                if value == "":
                    row[key] = None
                    continue
                try:
                    row[key] = float(value)
                except ValueError:
                    row[key] = value
            rows.append(row)
    return rows


def extract_epoch_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract validation epoch rows from the mixed train/validation CSV."""

    epoch_rows: list[dict[str, Any]] = []
    for row in rows:
        if row.get("loss/val") is None:
            continue
        epoch_rows.append(
            {
                "epoch": _to_int(row.get("epoch")),
                "step": _to_int(row.get("step")),
                "loss_val": _to_float(row.get("loss/val")),
                "edge_acc_val": _to_float(row.get("edge_smoke_acc/val")),
                "real_bond_acc_val": _to_float(row.get("edge_smoke_real_bond_acc/val")),
                "edge_loss_val": _to_float(row.get("edge_loss_weighted/val")),
                "bond_length_loss_val": _to_float(row.get("bond_length_loss_weighted/val")),
                "error_t_lig_val": _to_float(row.get("error_t_lig/val")),
            }
        )
    return epoch_rows


def parse_progress_lines() -> list[ProgressLine]:
    """Parse concise progress markers from the rank-0 log."""

    if not LOG_PATH.exists():
        return []

    pattern = re.compile(
        r"\[train-progress\] epoch=(?P<epoch>\d+)/(?P<total>\d+) "
        r"batch_idx=(?P<batch>\d+) global_step=(?P<step>\d+) "
        r"elapsed_min=(?P<elapsed>[-+0-9.eE]+) "
        r"loss=(?P<loss>[-+0-9.eE]+).*?"
        r"edge_loss_weighted=(?P<edge_loss>[-+0-9.eE]+).*?"
        r"bond_length_loss_weighted=(?P<bond_loss>[-+0-9.eE]+).*?"
        r"edge_smoke_acc=(?P<edge_acc>[-+0-9.eE]+).*?"
        r"edge_smoke_real_bond_acc=(?P<real_acc>[-+0-9.eE]+)"
    )

    progress: list[ProgressLine] = []
    with LOG_PATH.open(errors="replace") as handle:
        for line in handle:
            match = pattern.search(line)
            if match is None:
                continue
            progress.append(
                ProgressLine(
                    epoch=int(match.group("epoch")),
                    total_epochs=int(match.group("total")),
                    batch_idx=int(match.group("batch")),
                    global_step=int(match.group("step")),
                    elapsed_min=float(match.group("elapsed")),
                    loss=float(match.group("loss")),
                    edge_acc=float(match.group("edge_acc")),
                    real_bond_acc=float(match.group("real_acc")),
                    edge_loss_weighted=float(match.group("edge_loss")),
                    bond_length_loss_weighted=float(match.group("bond_loss")),
                )
            )
    return progress


def write_progress_csv(progress: list[ProgressLine]) -> Path:
    """Write parsed progress lines to a compact CSV."""

    output = ROOT / "metrics_so_far.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "total_epochs",
                "batch_idx",
                "global_step",
                "elapsed_min",
                "loss",
                "edge_acc",
                "real_bond_acc",
                "edge_loss_weighted",
                "bond_length_loss_weighted",
            ],
        )
        writer.writeheader()
        for item in progress:
            writer.writerow(item.__dict__)
    return output


def list_checkpoints() -> list[dict[str, Any]]:
    """Return checkpoint names, sizes, and mtimes."""

    if not CHECKPOINT_DIR.exists():
        return []
    checkpoints: list[dict[str, Any]] = []
    for path in sorted(CHECKPOINT_DIR.glob("*.ckpt")):
        stat = path.stat()
        checkpoints.append(
            {
                "name": path.name,
                "size_mb": stat.st_size / (1024 * 1024),
                "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
            }
        )
    return checkpoints


def make_plot(progress: list[ProgressLine], epoch_rows: list[dict[str, Any]]) -> Path | None:
    """Plot the key xhe long-test metrics observed so far."""

    if not progress and not epoch_rows:
        return None

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    if progress:
        steps = [item.global_step for item in progress]
        axes[0].plot(steps, [item.loss for item in progress], label="train progress loss", lw=1.6)
        axes[0].set_ylabel("loss")
        axes[0].legend(loc="best")

        axes[1].plot(steps, [item.edge_acc for item in progress], label="edge acc", lw=1.4)
        axes[1].plot(steps, [item.real_bond_acc for item in progress], label="real bond acc", lw=1.4)
        axes[1].set_ylabel("accuracy")
        axes[1].set_ylim(-0.05, 1.05)
        axes[1].legend(loc="best")

        axes[2].plot(
            steps,
            [item.edge_loss_weighted for item in progress],
            label="edge loss weighted",
            lw=1.2,
        )
        axes[2].plot(
            steps,
            [item.bond_length_loss_weighted for item in progress],
            label="bond length loss weighted",
            lw=1.2,
        )

    if epoch_rows:
        val_steps = [int(row["step"]) for row in epoch_rows if row["step"] is not None]
        real_acc = [row["real_bond_acc_val"] for row in epoch_rows if row["step"] is not None]
        edge_acc = [row["edge_acc_val"] for row in epoch_rows if row["step"] is not None]
        axes[1].scatter(val_steps, edge_acc, marker="o", label="val edge acc", zorder=4)
        axes[1].scatter(val_steps, real_acc, marker="o", label="val real bond acc", zorder=4)
        axes[1].legend(loc="best")

    axes[2].set_ylabel("component loss")
    axes[2].set_xlabel("global step")
    axes[2].legend(loc="best")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()

    output = ROOT / "early_metrics_so_far.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def write_report(
    progress: list[ProgressLine],
    epoch_rows: list[dict[str, Any]],
    checkpoints: list[dict[str, Any]],
    plot_path: Path | None,
) -> Path:
    """Write a Markdown status report for the long test."""

    latest = progress[-1] if progress else None
    latest_epoch = epoch_rows[-1] if epoch_rows else None
    has_error = False
    if LOG_PATH.exists():
        text_tail = LOG_PATH.read_text(errors="replace")[-20000:]
        has_error = any(marker in text_tail for marker in ["Traceback", "out of memory", "RuntimeError", "CUDA out"])

    summary = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "latest_progress": latest.__dict__ if latest else None,
        "latest_epoch_validation": latest_epoch,
        "checkpoints": checkpoints,
        "has_recent_error_marker": has_error,
    }
    (ROOT / "longtest_status.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    lines = [
        "# DiffSBDD-xhe DiffGui-style scratch 长测状态",
        "",
        f"- 更新时间: `{summary['updated_at']}`",
        f"- 配置: [`diffsbdd_xhe_diffgui_scratch_pocket1000_8gpu_b2acc4_8ep.yaml`](diffsbdd_xhe_diffgui_scratch_pocket1000_8gpu_b2acc4_8ep.yaml)",
        f"- 启动脚本: [`run_xhe_diffgui_scratch_pocket1000_8gpu_b2acc4_8ep.sh`](run_xhe_diffgui_scratch_pocket1000_8gpu_b2acc4_8ep.sh)",
        f"- 训练日志: [`train_xhe_diffgui_scratch_pocket1000_8gpu_b2acc4_8ep.log`](train_xhe_diffgui_scratch_pocket1000_8gpu_b2acc4_8ep.log)",
        f"- 原始 metrics: [`metrics.csv`](lightning/{RUN_NAME}/metrics/metrics.csv)",
        f"- 解析进度 CSV: [`metrics_so_far.csv`](metrics_so_far.csv)",
        f"- 状态 JSON: [`longtest_status.json`](longtest_status.json)",
    ]
    if plot_path is not None:
        lines.extend(["", f"![xhe early metrics]({plot_path.name})"])

    lines.extend(["", "## 当前进度", ""])
    if latest is None:
        lines.append("- 尚未解析到 `train-progress`。")
    else:
        lines.append(
            "- 最新 train-progress: "
            f"epoch `{latest.epoch}/{latest.total_epochs}`, batch `{latest.batch_idx}`, "
            f"step `{latest.global_step}`, elapsed `{latest.elapsed_min:.1f}` min, "
            f"loss `{_format_number(latest.loss)}`, edge_acc `{_format_number(latest.edge_acc)}`, "
            f"real_bond_acc `{_format_number(latest.real_bond_acc)}`。"
        )

    if latest_epoch is not None:
        lines.append(
            "- 最新 epoch validation: "
            f"epoch `{latest_epoch['epoch']}`, step `{latest_epoch['step']}`, "
            f"loss/val `{_format_number(latest_epoch['loss_val'])}`, "
            f"edge_acc/val `{_format_number(latest_epoch['edge_acc_val'])}`, "
            f"real_bond_acc/val `{_format_number(latest_epoch['real_bond_acc_val'])}`, "
            f"bond_length_loss/val `{_format_number(latest_epoch['bond_length_loss_val'])}`, "
            f"error_t_lig/val `{_format_number(latest_epoch['error_t_lig_val'])}`。"
        )
    lines.append(f"- 最近日志错误标记: `{'yes' if has_error else 'no'}`。")

    lines.extend(["", "## Epoch 验证汇总", ""])
    if epoch_rows:
        lines.append("| epoch | step | loss/val | edge_acc/val | real_bond_acc/val | bond_length_loss/val | error_t_lig/val |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|")
        for row in epoch_rows:
            lines.append(
                "| "
                f"{row['epoch']} | {row['step']} | {_format_number(row['loss_val'])} | "
                f"{_format_number(row['edge_acc_val'])} | {_format_number(row['real_bond_acc_val'])} | "
                f"{_format_number(row['bond_length_loss_val'])} | {_format_number(row['error_t_lig_val'])} |"
            )
    else:
        lines.append("尚无 epoch-end validation。")

    lines.extend(["", "## Checkpoint", ""])
    if checkpoints:
        lines.append("| 文件 | 大小 MB | mtime |")
        lines.append("|---|---:|---|")
        for item in checkpoints:
            lines.append(f"| `{item['name']}` | {item['size_mb']:.1f} | `{item['mtime']}` |")
    else:
        lines.append("尚无 checkpoint。")

    lines.extend(
        [
            "",
            "## 初步判断",
            "",
            "- 该长测用于验证 `xhe` scratch + absorb prior + segment schedule + diffusion KL edge loss 是否能在真实 DiffSBDD 训练链里学习边。",
            "- 仅看前期 `train-progress`，整体 edge accuracy 很快升高，但 real-bond accuracy 起步低且波动大；是否真正改善生成重建，必须等训练完成后做 generation/reconstruction 验证。",
            "- 当前 validation x/h loss 仍偏高，说明坐标/原子类型分支还没稳定，不能只凭 edge accuracy 判断成功。",
        ]
    )

    output = ROOT / "longtest_status.md"
    output.write_text("\n".join(lines) + "\n")
    return output


def main() -> None:
    """Generate all long-test summary artifacts."""

    rows = read_metrics()
    epoch_rows = extract_epoch_rows(rows)
    progress = parse_progress_lines()
    write_progress_csv(progress)
    checkpoints = list_checkpoints()
    plot_path = make_plot(progress, epoch_rows)
    report_path = write_report(progress, epoch_rows, checkpoints, plot_path)
    print(report_path)


if __name__ == "__main__":
    main()
