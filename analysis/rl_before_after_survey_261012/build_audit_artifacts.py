"""Export a literature evidence table and readable, nonexperimental overview figures."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyBboxPatch


LOGGER = logging.getLogger(__name__)
SOURCE_IDS = ["reinvent_2017", "release_2018", "multiparam_2019", "reinvent4", "mol2mol_rl",
              "augmented_memory", "acegen", "rl_al", "ls_molgen", "a2a", "cxcr4", "tyrosinase",
              "deppa", "rl_diffusion", "synthemol_rl", "gentrl", "drugex", "alidiff", "decompdpo"]
MARK_VALUES = {"?": 0, "…": 1, "◇": 2, "✓": 3}
SHORT_LABELS = ["R01 REINVENT 2017", "R02 ReLeaSE 2018", "R03 多参数 RL 2019",
                "R04 REINVENT4 2024", "R05 Mol2Mol-RL 2024", "R06 Augmented Memory 2024",
                "R07 ACEGEN 2024", "R08 RL–AL 2024", "R09 LS-MolGen 2023", "R10 A2A 2025",
                "R11 CXCR4 2024", "R12 酪氨酸酶 2024", "R13 DePPA 2026 预印本",
                "R14 RL-Diffusion 2025", "R15 SyntheMol-RL 2026", "R16 GENTRL 2019（材料不足）",
                "R17 DrugEx 2023（框架）", "P01 AliDiff（偏好优化）", "P02 DecompDPO（偏好优化）"]


def configure_fonts() -> None:
    """Use an available CJK typeface and presentation-readable text."""
    fonts = fm.findSystemFonts()
    preferred = next((p for p in fonts if "NotoSansCJK-Regular" in p), None)
    if preferred:
        fm.fontManager.addfont(preferred)
        plt.rcParams["font.family"] = fm.FontProperties(fname=preferred).get_name()
    else:
        plt.rcParams["font.family"] = ["Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams.update({"font.size": 14, "axes.unicode_minus": False, "svg.fonttype": "none"})


def parse_evidence(record: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Extract the checked table, retaining uncertainty and exact card locators."""
    document = record.read_text()
    section = document.split("## 4.", 1)[1].split("## 5.", 1)[0]
    lines = [line for line in section.splitlines() if line.startswith("| ")]
    headers = [part.strip() for part in lines[0].strip("|").split("|")]
    rows = []
    for line in lines[2:]:
        values = [part.strip() for part in line.strip("|").split("|")]
        if len(values) != len(headers):
            raise ValueError(f"Malformed comparison row: {line}")
        rows.append(dict(zip(headers, values)))
    if len(rows) != len(SOURCE_IDS):
        raise ValueError(f"Expected {len(SOURCE_IDS)} rows, found {len(rows)}")
    cards = re.findall(r"### ([RP]\d{2}) ([^\n]+)\n(.*?)(?=\n### |\n## |\Z)", document, re.S)
    card_map = {identifier: (title, body) for identifier, title, body in cards}
    for row, source_id in zip(rows, SOURCE_IDS):
        identifier = row[headers[0]].split()[0]
        title, body = card_map[identifier]
        row["source_id"] = source_id
        row["card_title"] = title
        row["inspection_scope"] = next((x.strip()[len("- 本次检查："):] for x in body.splitlines()
                                        if x.startswith("- 本次检查：")), "")
        row["source_url"] = re.search(r"\]\((https?://[^)]+)\)", body).group(1)
        row["evidence_notes"] = body.strip()
    return headers, rows


def save_figure(fig: plt.Figure, name: str, output: Path, record_dir: Path) -> None:
    """Save figures next to data and copy PNG into the self-contained work record."""
    fig.savefig(output / f"{name}.png", dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(output / f"{name}.svg", bbox_inches="tight", facecolor="white")
    shutil.copy2(output / f"{name}.png", record_dir / f"{name}.png")
    plt.close(fig)


def draw_matrix(headers: list[str], rows: list[dict[str, str]], output: Path, record_dir: Path,
                start: int = 0, stop: int = 10, part: int = 1) -> None:
    """Show confirmed and qualified evidence without inferring universal absence."""
    subset = rows[start:stop]
    values = np.array([[MARK_VALUES[row[column][0]] for column in headers[1:]] for row in subset])
    cmap = ListedColormap(["#fff1d3", "#eef1f5", "#d9e7fb", "#aed5c4"])
    fig, ax = plt.subplots(figsize=(10.0, 7.8))
    ax.imshow(values, cmap=cmap, vmin=0, vmax=3, aspect="auto")
    ax.set_xticks(range(4), ["同一模型\n前后集合", "前后生成\n分子例子", "设计结果\n分子分析", "双方最终\ntop-k 对照"], fontsize=17)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(subset)), SHORT_LABELS[start:stop], fontsize=15)
    for i in range(len(subset)):
        for j, column in enumerate(headers[1:]):
            mark = subset[i][column][0]
            label = {"✓": "已确认", "◇": "不同对照", "…": "未定位", "?": "资料不足"}[mark]
            ax.text(j, i, label, ha="center", va="center", fontsize=17,
                    color="#17324d" if mark != "…" else "#637080")
    ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(subset), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(axis="both", which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.suptitle(f"论文实际展示类型（{part}/2）", fontsize=23, y=0.99)
    fig.text(0.02, 0.016, "“未定位”仅限已读材料；“不同对照”详见表格与逐篇说明。", fontsize=15, color="#586679")
    fig.tight_layout(rect=[0, 0.055, 1, 0.93])
    save_figure(fig, f"literature_comparison_matrix_{part}", output, record_dir)


def draw_bridge(output: Path, record_dir: Path) -> None:
    """Explain the two claims connected by the screened joint-qualified set."""
    fig, ax = plt.subplots(figsize=(13, 6.7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    blocks = [(0.03, "各路线原模型\n与优化模型", "身份清楚的 checkpoint\n同条件生成"),
              (0.35, "同标准筛选\n与联合合格集合", "性质 + 结构同时合格\n数量、分布、独特骨架"),
              (0.67, "代表候选\n计算分析", "具体结构与口袋接触\n性质、姿态及已有验证")]
    for x, title, detail in blocks:
        ax.add_patch(FancyBboxPatch((x, 0.4), 0.28, 0.39, boxstyle="round,pad=0.01,rounding_size=0.025",
                                   facecolor="#e7eef8", edgecolor="#527aa6", linewidth=1.8))
        ax.text(x + 0.14, 0.68, title, ha="center", va="center", fontsize=20, color="#17324d", weight="bold")
        ax.text(x + 0.14, 0.49, detail, ha="center", va="center", fontsize=15, color="#315b87")
    for x in [0.318, 0.638]:
        ax.annotate("", xy=(x + 0.03, 0.59), xytext=(x - 0.01, 0.59),
                    arrowprops={"arrowstyle": "->", "lw": 2.1, "color": "#527aa6"})
    ax.text(0.34, 0.24, "模型贡献：生成结果是否更符合候选要求", ha="center", fontsize=18, color="#2a6b53")
    ax.text(0.8, 0.15, "分子价值：为什么值得后续研究", ha="center", fontsize=17, color="#2a6b53")
    ax.text(0.5, 0.93, "把模型优化落到候选集合，再解释具体分子", ha="center", fontsize=23, weight="bold", color="#17324d")
    ax.text(0.5, 0.035, "示意图；不存在要求每个指标或冠军分子都胜出的逻辑前提。", ha="center", fontsize=13, color="#637080")
    save_figure(fig, "comparison_evidence_bridge", output, record_dir)


def main() -> None:
    """Build the artifacts from the reviewed narrative without modifying experiments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    headers, rows = parse_evidence(args.record)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "literature_comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    audit = {"survey_date": "2026-10-12", "scope": "定向证据类型调查，非系统综述",
             "uncertainty": {"✓": "确认", "◇": "相关但对照不同", "…": "已读材料未定位", "?": "资料不足"},
             "rows": rows, "limitations": "GENTRL正文与四篇独立SI未取得；代码版本不保证与发表时一致"}
    (args.output_dir / "evidence_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    configure_fonts()
    draw_matrix(headers, rows, args.output_dir, args.record.parent, 0, 10, 1)
    draw_matrix(headers, rows, args.output_dir, args.record.parent, 10, 19, 2)
    draw_bridge(args.output_dir, args.record.parent)
    LOGGER.info("Exported %d literature comparisons and three figures", len(rows))


if __name__ == "__main__":
    main()
