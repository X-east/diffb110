"""Draw the documented matching workflow and reproduce three reported example metrics."""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


LOGGER = logging.getLogger(__name__)
EXAMPLES = [("1 → 4", 0.227, 0.151), ("2 → 5", 0.582, 0.085), ("3 → 6", 0.716, 0.135)]


def save(fig: plt.Figure, name: str, output: Path, record: Path) -> None:
    """Keep plot data and self-contained report images together."""
    fig.savefig(output / f"{name}.png", dpi=160, bbox_inches="tight", facecolor="white")
    fig.savefig(output / f"{name}.svg", bbox_inches="tight", facecolor="white")
    shutil.copy2(output / f"{name}.png", record / f"{name}.png")
    plt.close(fig)


def workflow(output: Path, record: Path) -> None:
    """Separate the paper's named tools from Pharmit's documented search mechanism."""
    fig, ax = plt.subplots(figsize=(12, 7.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    rows = [
        (0.68, "论文明确报告", [("LeDock 对接", "5 个姿态\nRMSD 设置 1 Å"),
                            ("Pharmit 匹配选例", "已知抑制剂\n与生成分子的药效团"),
                            ("PyMOL 作图", "结构重叠\n与口袋作用模式")], "#e8eef8"),
        (0.25, "Pharmit 工具机制", [("定义 3D 查询", "特征类型、坐标\n容差半径、启用状态"),
                               ("检索候选构象", "对应特征点空间匹配\n并重新配准姿态"),
                               ("按匹配质量筛选", "药效团匹配 RMSD\n及其他可选条件")], "#e3f0e9"),
    ]
    for y, label, blocks, color in rows:
        ax.text(0.02, y + 0.27, label, fontsize=20, weight="bold", color="#17324d")
        for i, (title, description) in enumerate(blocks):
            x = 0.03 + i * 0.33
            ax.add_patch(FancyBboxPatch((x, y), 0.28, 0.22, boxstyle="round,pad=0.012,rounding_size=0.02",
                                       facecolor=color, edgecolor="#6284a3", linewidth=1.7))
            ax.text(x + 0.14, y + 0.158, title, ha="center", fontsize=19, weight="bold", color="#17324d")
            ax.text(x + 0.14, y + 0.072, description, ha="center", va="center", fontsize=17, color="#315b68")
            if i < 2:
                ax.annotate("", xy=(x + 0.32, y + 0.11), xytext=(x + 0.294, y + 0.11),
                            arrowprops={"arrowstyle": "->", "lw": 2.2, "color": "#6284a3"})
    ax.text(0.5, 0.08, "作者的查询特征点、半径与候选库构象配置未定位到公开记录。", ha="center", fontsize=17, color="#815e2e")
    ax.text(0.5, 0.015, "上行为论文使用工具的概括，不表示已恢复全部操作顺序；下行为工具文档解释。", ha="center", fontsize=13, color="#647180")
    save(fig, "pharmit_matching_workflow", output, record)


def example_metrics(output: Path, record: Path) -> None:
    """Plot reported example values without treating them as population statistics."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.9))
    for ax, index, title, ylabel, color in [(axes[0], 1, "药效团相关原子的重叠", "报告 RMSD / Å", "#577eaa"),
                                          (axes[1], 2, "图中报告的结构相似度", "Similarity", "#5a947c")]:
        values = [x[index] for x in EXAMPLES]
        ax.bar(range(3), values, width=0.55, color=color)
        ax.set_xticks(range(3), [x[0] for x in EXAMPLES], fontsize=17)
        ax.set_title(title, fontsize=21, pad=15)
        ax.set_ylabel(ylabel, fontsize=18)
        ax.set_xlabel("参照抑制剂 → 生成分子", fontsize=17)
        ax.tick_params(axis="y", labelsize=15)
        ax.set_ylim(0, max(values) * 1.28)
        for i, value in enumerate(values):
            ax.text(i, value + max(values) * 0.055, f"{value:.3f}", ha="center", fontsize=18)
        ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, 0.015, "来源：LS-MolGen Fig. 3 的三个选例；不代表 5,000 个分子的总体表现。", ha="center", fontsize=13, color="#647180")
    fig.tight_layout(rect=[0, 0.045, 1, 1])
    save(fig, "reported_matching_examples", output, record)


def main() -> None:
    """Generate source-grounded figures, recording exact values in CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for font in fm.findSystemFonts():
        if "NotoSansCJK-Regular" in font:
            fm.fontManager.addfont(font)
            plt.rcParams["font.family"] = fm.FontProperties(fname=font).get_name()
            break
    plt.rcParams.update({"font.size": 16, "axes.unicode_minus": False, "svg.fonttype": "none"})
    args.record_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "reported_examples.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["reference_to_generated", "reported_pharmacophore_rmsd_angstrom", "reported_structure_similarity"])
        writer.writerows(EXAMPLES)
    workflow(args.output_dir, args.record_dir)
    example_metrics(args.output_dir, args.record_dir)
    shutil.copy2(args.output_dir / "figure_check" / "ls_molgen_page7.png", args.record_dir / "ls_molgen_original_page7.png")
    LOGGER.info("Saved two figures, three exact example rows, and the original figure page")


if __name__ == "__main__":
    main()
