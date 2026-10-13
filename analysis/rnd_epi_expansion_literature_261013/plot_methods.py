"""Draw the literature-backed EPI expansion method comparison."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


LOGGER = logging.getLogger(__name__)


def draw_comparison(output_dir: Path) -> list[Path]:
    """Render a conceptual comparison, without asserting measured enrichment."""
    font_path = font_manager.findfont("Noto Sans CJK JP", fallback_to_default=True)
    font = font_manager.FontProperties(fname=font_path)
    fig, ax = plt.subplots(figsize=(14, 9))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.text(0.3, 8.55, "已知抑制剂如何扩展为候选集合", fontproperties=font,
            fontsize=25, weight="bold", color="#18334D")
    ax.text(0.3, 8.08, "文献方法比较：检索命中与活性阳性需要分别记录", fontproperties=font,
            fontsize=17, color="#465769")
    for x, title in [(0.3, "方法与参考研究"), (3.25, "保留的参考信息"), (8.25, "扩展规则与结果")]:
        ax.text(x, 7.35, title, fontproperties=font, fontsize=19, weight="bold")
    rows = [
        ("二维相似性\nMexY · 2023", "已知弱抑制剂小檗碱", "PubChem相似性 >0.8\n约1万候选后对接、聚类"),
        ("药效团\nAcrB · 2024", "32个抑制剂/底物\n共同化学特征的空间排列", "筛6566个独特分子\n药效团命中207个，再对接"),
        ("骨架替换 + 药效团\nNorA · 2020", "已知侧链及相对位置\n两个模型 ModB / ModC", "双模型 ≥1.7 或单模型 >2.0\n4种骨架、8个合成衍生物"),
        ("三维形状 / MIF\nZipA 2005 / NorA 2011", "生物活性构象的空间轮廓\n或 GRID 分子相互作用场", "ROCS / FLAP 搜索新骨架\n阈值须按方法、靶点验证"),
        ("生物电子等排替换\nNorA · 2021", "母体与局部替换位置", "42种替换结构后对接筛选\n合成并测试 5-NPPP"),
    ]
    for index, (label, reference, rule) in enumerate(rows):
        y = 6.62 - index * 1.23
        ax.text(0.3, y, label, va="center", fontproperties=font, fontsize=17,
                color="#18334D")
        for x, width, content, color in [(3.15, 4.25, reference, "#EDF3FA"),
                                         (8.15, 5.25, rule, "#EAF5F0")]:
            ax.add_patch(FancyBboxPatch((x, y - 0.44), width, 0.88,
                                       boxstyle="round,pad=0.12", facecolor=color,
                                       edgecolor="#CED9E3", linewidth=1.0))
            ax.text(x + 0.15, y, content, va="center", fontproperties=font,
                    fontsize=17, color="#18334D")
        ax.add_patch(FancyArrowPatch((7.52, y), (8.02, y), arrowstyle="->",
                                     mutation_scale=17, color="#6A8193", linewidth=1.5))
    ax.text(0.3, 0.48, "NorA属于MFS；ZipA不是外排泵。其方法先例不能直接赋予RE-CmeB活性标签。",
            fontproperties=font, fontsize=15, color="#465769")
    paths = [output_dir / "epi_expansion_method_comparison.png",
             output_dir / "epi_expansion_method_comparison.svg"]
    for path in paths:
        fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return paths


def main() -> None:
    """Export the comparison figure and the traceable evidence CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--record-dir", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rows = json.loads((args.output_dir / "evidence_table.json").read_text(encoding="utf-8"))
    sources = {row["source_id"]: row for row in json.loads(
        (args.output_dir / "sources" / "source_manifest.json").read_text(encoding="utf-8"))}
    for row in rows:
        source = sources[row["source_id"]]
        row.update({"doi": source.get("doi", ""),
                    "title": source.get("title", ""),
                    "authors": source.get("authorString", ""),
                    "year": source.get("pubYear", ""),
                    "verification": source.get("verification", "")})
    with (args.output_dir / "evidence_table.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    paths = draw_comparison(args.output_dir)
    if args.record_dir:
        args.record_dir.mkdir(parents=True, exist_ok=True)
        for path in paths:
            shutil.copy2(path, args.record_dir / path.name)
    LOGGER.info("Exported %d evidence entries and %d figures", len(rows), len(paths))


if __name__ == "__main__":
    main()
