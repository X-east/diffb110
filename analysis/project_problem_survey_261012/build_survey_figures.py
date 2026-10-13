"""Build source-linked comparison tables and figures for the project survey."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


LOGGER = logging.getLogger(__name__)
PAPER_TITLES = [
    "Structure-based drug design with equivariant diffusion models",
    "Target-aware 3D molecular generation based on guided equivariant diffusion",
    "Aligning Target-Aware Molecule Diffusion Models with Exact Energy Optimization",
    "Decomposed Direct Preference Optimization for Structure-Based Drug Design",
    "PoseBusters: AI-based docking methods fail to generate physically valid poses or generalise to novel sequences",
    "From Theory to Therapy: Reframing SBDD Model Evaluation via Practical Metrics",
    "Structure-Based Generation of 3D Small-Molecule Drugs: Are We There Yet?",
    "LS-MolGen: Ligand-and-Structure Dual-Driven Deep Reinforcement Learning for Target-Specific Molecular Generation Improves Binding Affinity and Novelty",
    "Sample Efficiency Matters: A Benchmark for Practical Molecular Optimization",
    "Sample efficient reinforcement learning with active learning for molecular design",
    "Identification of nanomolar adenosine A2A receptor ligands using reinforcement learning and structure-based drug design",
    "Synthetically Feasible De Novo Molecular Design of Leads Based on a Reinforcement Learning Model: AI-Assisted Discovery of an Anti-IBD Lead Targeting CXCR4",
    "Generative AI for designing and validating easily synthesizable and structurally novel antibiotics",
    "Virtual screening and biological activity evaluation of novel efflux pump inhibitors targeting RE-CmeB in Campylobacter spp.",
    "Artificial Intelligence-Assisted Optimization of Antipigmentation Tyrosinase Inhibitors: De Novo Molecular Generation Based on a Low Activity Lead Compound",
]


def save_figure(fig: plt.Figure, output_dir: Path, name: str) -> None:
    """Export a figure in raster and vector formats."""
    for extension in ("png", "svg"):
        fig.savefig(output_dir / f"{name}.{extension}", dpi=170, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    plt.close(fig)


def build_comparison(record: Path, output_dir: Path) -> None:
    """Export the document's evidence comparisons without changing their wording."""
    text = record.read_text()
    matches = list(re.finditer(r"^### P(\d{2}) (.+)$", text, re.MULTILINE))
    rows = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else text.index("\n## 五、")
        section = text[match.end():end].strip()
        fields = {}
        for label in ("问题与做法", "证据结构", "结论边界与借鉴"):
            found = re.search(rf"\*\*{label}：\*\*(.+?)(?:\n\n|$)", section, re.DOTALL)
            fields[label] = found.group(1).strip() if found else ""
        source = section.split("来源：", 1)[1].strip()
        rows.append({"id": f"P{match.group(1)}", "short_title": match.group(2),
                     "english_title": PAPER_TITLES[index], **fields, "source_and_reading_scope": source})
    if len(rows) != len(PAPER_TITLES):
        raise ValueError(f"Expected {len(PAPER_TITLES)} paper comparisons, found {len(rows)}")
    with (output_dir / "literature_argument_comparison.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "literature_argument_comparison.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2))


def plot_observations(output_dir: Path, audit_csv: Path) -> None:
    """Plot historical training observations with their source and sample limitations."""
    with audit_csv.open(encoding="utf-8-sig") as stream:
        rows = {int(row["epoch"]): row for row in csv.DictReader(stream)}
    observations = [
        {"epoch": epoch, "records": int(rows[epoch]["records"]),
         "docking_n": int(rows[epoch]["canonical_and_docking_n"]),
         "docking_mean": float(rows[epoch]["docking_mean_selected"]),
         "three_member_percent": ring, "spiro_percent": spiro}
        for epoch, ring, spiro in ((1, 12.3, 12.3), (79, 37.9, 39.7))
    ]
    data = {"kind": "historical_training_observations_not_independent_generation",
            "docking_source": str(audit_csv),
            "ring_source": "docs/overview/DiffGui-RL训练汇总.md: ep1/ep79 table; reported percentages, not recalculated",
            "observations": observations}
    (output_dir / "survey_figure_data.json").write_text(json.dumps(data, ensure_ascii=False, indent=2))
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.3), gridspec_kw={"width_ratios": [1, 1.35]})
    colors = ["#6F91BE", "#D88A46"]
    x = np.arange(2)
    means = [row["docking_mean"] for row in observations]
    axes[0].plot(x, means, marker="o", markersize=12, linewidth=3, color="#315B87")
    for index, (mean, row) in enumerate(zip(means, observations)):
        axes[0].annotate(f"{mean:.2f}\nn={row['docking_n']}", (index, mean),
                         xytext=(0, 12), textcoords="offset points", ha="center", fontsize=18)
    axes[0].set_xticks(x, ["ep1", "ep79"])
    axes[0].set_xlim(-0.4, 1.4)
    axes[0].set_ylim(-12.2, -8.9)
    axes[0].set_ylabel("对接评分均值（低值优）", fontsize=18)
    axes[0].set_title("训练样本的对接变化", fontsize=21, pad=18)
    width = 0.34
    for index, (field, label) in enumerate((("three_member_percent", "含三元环"), ("spiro_percent", "螺环"))):
        values = [row[field] for row in observations]
        bars = axes[1].bar(x + (index - 0.5) * width, values, width=width,
                           color=colors[index], label=label)
        axes[1].bar_label(bars, labels=[f"{value:.1f}%" for value in values], padding=5, fontsize=18)
    axes[1].set_xticks(x, ["ep1", "ep79"])
    axes[1].set_ylim(0, 53)
    axes[1].set_ylabel("已报告的有效样本比例（%）", fontsize=18)
    axes[1].set_title("训练样本的环系变化", fontsize=21, pad=18)
    axes[1].legend(loc="upper left", fontsize=16, frameon=False)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=17)
        axis.grid(axis="y", alpha=0.16)
        axis.set_axisbelow(True)
    fig.suptitle("DiffGui：性质变化与结构变化需要共同解释", fontsize=25, y=1.03)
    fig.text(0.5, -0.04, "历史训练样本观察，非独立生成对照；环系存在不等于分子不可用；无因果检验",
             ha="center", fontsize=16, color="#52657A")
    fig.tight_layout()
    save_figure(fig, output_dir, "rl_property_structure_observation")


def plot_argument_routes(output_dir: Path) -> None:
    """Map literature arguments to their evidence endpoints without empirical scores."""
    fig, ax = plt.subplots(figsize=(16, 10))
    fig.patch.set_facecolor("#F5F7FA")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.text(0.45, 9.55, "论文用什么证据把问题接到结论", fontsize=28, fontweight="bold", color="#17324D")
    for x, heading in ((0.55, "问题起点"), (5.65, "采取方法"), (10.75, "证据与结论")):
        ax.text(x, 8.9, heading, fontsize=21, fontweight="bold", color="#17324D")
    routes = [
        ("生成质量或可控性不足", "改机制 / 调整生成分布", "横评与消融\n支撑生成性能", "DiffSBDD / DiffGui / AliDiff / DecompDPO"),
        ("常用指标遗漏合理性", "增加互补质量检查", "检查改变判断\n支撑评价与诊断", "PoseBusters / From Theory / JMC 化学合理性"),
        ("通用能力需要靶点信息", "配体 / 结构信息适配", "生成对照与候选再评价\n支撑计算概念验证", "LS-MolGen"),
        ("评价成本高且预算有限", "主动选择 / 定向搜索", "预算曲线与计算命中\n支撑搜索效率", "PMO / RL–AL"),
        ("候选需可获得、有功能", "选择、合成、功能测试", "实验结果与机制线索\n支撑具体候选价值", "A2A / CXCR4 / SyntheMol / RE-CmeB / 酪氨酸酶"),
    ]
    palette = ["#E6EDF8", "#E3F0EC", "#EEE7F4", "#FFF0DB", "#E4EFF2"]
    for index, (problem, approach, evidence, papers) in enumerate(routes):
        y = 7.22 - index * 1.45
        for x, content in ((0.45, problem), (5.55, approach), (10.65, evidence)):
            ax.add_patch(FancyBboxPatch((x, y), 4.8, 0.98,
                                       boxstyle="round,pad=0.08,rounding_size=0.1",
                                       facecolor=palette[index], edgecolor="none"))
            ax.text(x + 0.18, y + 0.70, content, fontsize=18, color="#263A4C", va="top", linespacing=1.4)
        ax.text(0.6, y - 0.19, papers, fontsize=12.5, color="#5B7188")
    ax.text(0.5, 0.1, "分析性归纳，不是实验结果；同一论文可采用多种论证方式", fontsize=14, color="#65758A")
    save_figure(fig, output_dir, "literature_argument_routes")


def plot_thesis_scope(output_dir: Path) -> None:
    """Show the two independent optimization routes and the bounded candidate endpoint."""
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor("#F5F7FA")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.text(0.45, 8.45, "当前课题：共同目的、并行优化、统一候选分析", fontsize=27,
            fontweight="bold", color="#17324D")
    ax.text(0.45, 7.7, "范围：RE-CmeB 口袋下的生成质量与计算候选", fontsize=21, color="#40556D")
    boxes = [
        (0.5, 4.65, 5.7, 2.4, "性质反馈优化", "DiffGui 原始底座 / RL 模型\n评价性质变化及结构代价", "#E6EDF8"),
        (0.5, 1.6, 5.7, 2.4, "结构偏好优化", "DiffSBDD 原始底座 / ESRPO 模型\n评价结构产出及性质、形态代价", "#E3F0EC"),
        (8.1, 3.75, 7.1, 2.3, "各自对照，采用统一评价口径", "固定尝试分母、结构与性质\n区分原始生成与后处理结果", "#EEE7F4"),
        (8.1, 0.95, 7.1, 2.2, "候选筛选与逐分子计算分析", "记录来源与筛选标准\n结构、性质、预测结合模式及必要复核", "#FFF0DB"),
    ]
    for x, y, width, height, title, body, color in boxes:
        ax.add_patch(FancyBboxPatch((x, y), width, height,
                                   boxstyle="round,pad=0.12,rounding_size=0.16",
                                   facecolor=color, edgecolor="none"))
        ax.text(x + 0.2, y + height - 0.52, title, fontsize=22, fontweight="bold", color="#17324D")
        ax.text(x + 0.2, y + height - 1.1, body, fontsize=18, color="#263A4C", va="top", linespacing=1.7)
    for start, end in (((6.35, 5.82), (7.9, 5.15)), ((6.35, 2.78), (7.9, 4.45)),
                       ((11.65, 3.6), (11.65, 3.3))):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=25,
                                    linewidth=2.3, color="#76899B"))
    ax.text(0.5, 0.3, "两条训练路线独立；生成改善与候选价值分别验证；功能活性留待后续研究",
            fontsize=16, color="#52657A")
    save_figure(fig, output_dir, "thesis_parallel_route_logic")


def main() -> None:
    """Export the survey evidence artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--audit-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    plt.rcParams.update({"font.family": "Noto Sans CJK JP", "axes.unicode_minus": False,
                         "svg.fonttype": "path"})
    build_comparison(args.record, args.output_dir)
    plot_observations(args.output_dir, args.audit_csv)
    plot_argument_routes(args.output_dir)
    plot_thesis_scope(args.output_dir)
    LOGGER.info("Generated 15 paper comparisons, figure data, and 3 PNG/SVG figure pairs")


if __name__ == "__main__":
    main()
