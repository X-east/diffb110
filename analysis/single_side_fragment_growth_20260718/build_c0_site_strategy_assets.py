"""Build compact evidence tables and a comparison plot for C0-only growth."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties


OUT_DIR = Path(__file__).resolve().parent
CHINESE_FONT = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")

ROWS = [
    {
        "model": "DiffGui",
        "fixed_core": 2,
        "c0_start_condition": 1,
        "root_type_can_be_generated": 2,
        "variable_size": 2,
        "implementation_level": "采样器硬约束",
        "recommendation": "首选",
    },
    {
        "model": "DiffSBDD",
        "fixed_core": 2,
        "c0_start_condition": 1,
        "root_type_can_be_generated": 1,
        "variable_size": 2,
        "implementation_level": "固定根seed或坐标mask",
        "recommendation": "可做对照",
    },
    {
        "model": "PMDM",
        "fixed_core": 2,
        "c0_start_condition": 1,
        "root_type_can_be_generated": 1,
        "variable_size": 1,
        "implementation_level": "固定根seed或坐标mask",
        "recommendation": "次选",
    },
]

METRICS = [
    ("fixed_core", "固定核心"),
    ("c0_start_condition", "指定 C0 生长起点"),
    ("root_type_can_be_generated", "根原子类型可生成"),
    ("variable_size", "不指定精确新增数"),
]


def write_csv() -> None:
    """Write the model capability matrix."""
    path = OUT_DIR / "c0_unique_site_strategy.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROWS[0].keys())
        writer.writeheader()
        writer.writerows(ROWS)


def plot_matrix() -> None:
    """Plot native and achievable C0-site constraint capability."""
    matrix = np.asarray([[row[key] for key, _ in METRICS] for row in ROWS])
    fig, ax = plt.subplots(figsize=(9.2, 3.6), constrained_layout=True)
    image = ax.imshow(matrix, vmin=0, vmax=2, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(METRICS)), [label for _, label in METRICS], fontproperties=CHINESE_FONT)
    ax.set_yticks(range(len(ROWS)), [row["model"] for row in ROWS])
    ax.tick_params(axis="x", rotation=18)
    labels = {0: "原生无", 1: "需小改", 2: "原生有"}
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = int(matrix[row_idx, col_idx])
            ax.text(
                col_idx,
                row_idx,
                labels[value],
                ha="center",
                va="center",
                fontsize=9,
                fontproperties=CHINESE_FONT,
            )
    ax.set_title("三模型指定 C0 为片段生长起点的能力对比", fontproperties=CHINESE_FONT)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02, ticks=[0, 1, 2])
    colorbar.ax.set_yticklabels(["原生无", "需小改", "原生有"], fontproperties=CHINESE_FONT)
    fig.savefig(OUT_DIR / "c0_unique_site_strategy.png", dpi=220)
    fig.savefig(OUT_DIR / "c0_unique_site_strategy.svg")
    plt.close(fig)


if __name__ == "__main__":
    write_csv()
    plot_matrix()
