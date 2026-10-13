"""Plot the evidence readiness map for the midterm defense."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "evidence_inventory.csv"
OUTPUT_PNG = ROOT / "midterm_evidence_readiness.png"
OUTPUT_SVG = ROOT / "midterm_evidence_readiness.svg"
FONT = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")


def main() -> None:
    """Create a compact presentation-readiness plot."""
    data = pd.read_csv(INPUT)
    data = data.sort_values(["readiness_score", "section"], ascending=[True, True])

    colors = {1: "#C84A45", 2: "#D69B2D", 3: "#27866D"}
    labels = {1: "暂不使用", 2: "需限制表述", 3: "直接可用"}

    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 14})
    fig, ax = plt.subplots(figsize=(12.5, 7.2), constrained_layout=True)
    ax.barh(
        data["section"],
        data["readiness_score"],
        color=[colors[value] for value in data["readiness_score"]],
        height=0.64,
    )
    for y, value in enumerate(data["readiness_score"]):
        ax.text(value + 0.05, y, labels[value], va="center", fontsize=13, fontproperties=FONT)

    ax.set_xlim(0, 3.75)
    ax.set_xticks([1, 2, 3], ["1\n未形成结果", "2\n阶段性证据", "3\n可作为核心结果"], fontproperties=FONT)
    ax.set_yticks(range(len(data)), data["section"], fontproperties=FONT)
    ax.set_xlabel("中期答辩证据成熟度", fontproperties=FONT, fontsize=15)
    ax.set_title(
        "RE-CmeB 中期答辩：现有数据可用性分层",
        fontproperties=FONT,
        fontweight="bold",
        fontsize=18,
    )
    ax.grid(axis="x", color="#D7D7D7", linewidth=0.8, alpha=0.7)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.savefig(OUTPUT_PNG, dpi=240, facecolor="white")
    fig.savefig(OUTPUT_SVG, facecolor="white")


if __name__ == "__main__":
    main()
