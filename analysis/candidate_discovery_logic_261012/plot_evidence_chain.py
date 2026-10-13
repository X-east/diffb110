"""Draw the proposed evidence chain for RE-CmeB candidate discovery."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


LOGGER = logging.getLogger(__name__)


def draw_evidence_chain(output_dir: Path) -> None:
    """Export a conceptual diagram; no empirical result is depicted."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "Noto Sans CJK JP",
            "font.size": 17,
            "axes.unicode_minus": False,
            "svg.fonttype": "path",
        }
    )
    fig, ax = plt.subplots(figsize=(17, 9))
    fig.patch.set_facecolor("#F5F7FA")
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.text(0.45, 8.45, "RE-CmeB：生成优化如何连接到候选发现", fontsize=29,
            fontweight="bold", color="#17324D", va="center")
    ax.text(0.45, 7.75, "研究终点：目标功能、结构合理性与可获得性共同支撑的候选分子",
            fontsize=20, color="#40556D")
    columns = [
        (0.45, "生成策略", "分别比较各自底座与优化模型\n\nDiffGui 原始 / RL 优化\nDiffSBDD 原始 / ESRPO 优化",
         "生成层证据", "性质与结构分布的变化\n完整报告有效 / 无效尝试", "#E6EDF8"),
        (6.15, "统一搜索与筛选", "冻结标准与候选选择流程\n\n结构检查、去重、性质约束\n可获得性审查、独立再评价",
         "搜索层证据", "固定预算下合格候选与骨架数\n达到候选目标所需成本", "#E2F1EC"),
        (11.85, "候选组合与验证", "保留具有不同骨架的候选\n\n逐分子审查结合与结构证据\n检验目标功能及实验可行性",
         "分子层证据", "筛选后 top-k 的独立质量\n外排抑制 / 抗生素增敏等功能", "#FFF0DB"),
    ]
    for x, title, body, tier, evidence, color in columns:
        ax.add_patch(FancyBboxPatch((x, 3.45), 4.7, 3.6,
                                   boxstyle="round,pad=0.12,rounding_size=0.16",
                                   facecolor=color, edgecolor="none"))
        ax.text(x + 0.22, 6.55, title, fontsize=24, fontweight="bold", color="#17324D")
        ax.text(x + 0.22, 5.78, body, fontsize=18, color="#263A4C",
                linespacing=1.65, va="top")
        ax.add_patch(FancyBboxPatch((x, 1.35), 4.7, 1.7,
                                   boxstyle="round,pad=0.12,rounding_size=0.16",
                                   facecolor="white", edgecolor="#DCE3EB"))
        ax.text(x + 0.22, 2.52, tier, fontsize=20, fontweight="bold", color="#17324D")
        ax.text(x + 0.22, 2.10, evidence, fontsize=17, color="#40556D", va="top", linespacing=1.5)
    for x in (5.3, 11.0):
        ax.add_patch(FancyArrowPatch((x, 5.2), (x + 0.62, 5.2),
                                    arrowstyle="-|>", mutation_scale=27,
                                    linewidth=2.5, color="#76899B"))
    ax.text(8.5, 0.55, "方法贡献通过搜索收益验证；候选价值通过独立评价与功能实验验证",
            fontsize=19, fontweight="bold", color="#17324D", ha="center")
    ax.text(16.5, 0.07, "研究设计示意，无实验数据", fontsize=12, color="#6F7F8F", ha="right")
    for extension in ("png", "svg"):
        output_path = output_dir / f"candidate_discovery_evidence_chain.{extension}"
        fig.savefig(output_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
        LOGGER.info("Saved %s", output_path)
    plt.close(fig)


def main() -> None:
    """Parse the export location and generate the diagram."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    draw_evidence_chain(args.output_dir)


if __name__ == "__main__":
    main()
