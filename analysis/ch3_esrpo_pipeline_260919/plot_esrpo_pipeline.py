#!/usr/bin/env python
"""第 3 章 ESRPO 流程横向流程图。

把原先自上而下的 ASCII 流程图改为**横向（从左到右）**的单行 8 阶段流程图。
数字全部取自已验证的分析产物：

- 候选池三分账：analysis/diffsbdd_structure_pairs_matched_260831/pair_audit.json
- 偏好对与匹配质量：同上 matching_diagnostics / pair_summary
- 训练配置：launch/diffsbdd_global_esrpo_x_only_lr1e5_mainline.yaml
- 停留点与输出指标：analysis/e2po_mainline_reasonable_eval/reasonable_molecules_metrics.csv
  （QED 为合理分子**中位数**，与论文表 3.3 一致）

排版：按显示宽度（CJK 计 2）折行，避免中文溢出方框。

输出：esrpo_pipeline.png / esrpo_pipeline.svg
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT_DIR = Path(__file__).resolve().parent

CJK_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if Path(CJK_FONT).exists():
    font_manager.fontManager.addfont(CJK_FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=CJK_FONT).get_name()
plt.rcParams["axes.unicode_minus"] = False

WRAP_LIMIT = 34  # 显示宽度单位（CJK 字符 = 2）

STAGES: list[dict] = [
    {
        "title": "① DiffSBDD 底座",
        "body": ["采集完整去噪轨迹", "20 万端点候选分子", "（离线，无在线采样）"],
        "color": "#D6E4F0",
    },
    {
        "title": "② RDKit 合法性过滤",
        "body": ["剔除 64,467（32.23%）", "多片段等不合法结构", "另 35 个不可用", "可用候选 135,498"],
        "color": "#D6E4F0",
    },
    {
        "title": "③ R1–R4 病理判定",
        "body": ["winner = R1–R4 全清", "loser = 至少命中 1 条", "R5 仅作护栏"],
        "color": "#DCEEDC",
    },
    {
        "title": "④ 严格条件匹配",
        "body": ["节点数差 = 0", "MW 残差 2.92 Da", "TPSA 差 ≤ 30", "docking 差 ≤ 1.5", "R5 命中率拉平"],
        "color": "#DCEEDC",
    },
    {
        "title": "⑤ 构建偏好对",
        "body": ["train 36,232 对", "val 3,495 对", "test 3,480 对", "R(M) = −v(M)", "v = R1–R4 命中条数"],
        "color": "#DCEEDC",
    },
    {
        "title": "⑥ ESRPO 训练",
        "body": [
            "终点级偏好优化",
            "x-only wx=1.0, wv=0.0",
            "冻结 enc/dec + EGNN 前3块",
            "可训练参数 40.4%",
            "18,116 次更新",
            "每 2000 步存盘",
        ],
        "color": "#FBE7CF",
    },
    {
        "title": "⑦ 停留点选择",
        "body": ["QED 中位 ≥ 基线 85%", "0.647×0.85 = 0.550", "达标中取产率最高者", "→ 推荐 step 4000"],
        "color": "#FBE7CF",
    },
    {
        "title": "⑧ 输出结果",
        "body": [
            "产率 30.5% → 55.6%（1.82×）",
            "R1–R4 48.67% → 21.27%",
            "对接中位 −8.92 → −8.46",
            "sanitize 67.5% → 81.0%",
            "QED 中位 0.647 → 0.561",
        ],
        "color": "#F7D6D6",
    },
]


def display_width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def wrap_by_width(text: str, limit: int) -> list[str]:
    """Greedy wrap measured in display-width units."""
    if display_width(text) <= limit:
        return [text]
    lines: list[str] = []
    current = ""
    for token in _tokens(text):
        candidate = current + token
        if current and display_width(candidate) > limit:
            lines.append(current.rstrip())
            current = token.lstrip()
        else:
            current = candidate
    if current.strip():
        lines.append(current.rstrip())
    return lines


def _tokens(text: str) -> list[str]:
    """Split into CJK single chars and ASCII words, keeping a trailing space."""
    tokens: list[str] = []
    buf = ""
    for ch in text:
        if unicodedata.east_asian_width(ch) in "WF":
            if buf:
                tokens.append(buf)
                buf = ""
            tokens.append(ch)
        elif ch == " ":
            buf += ch
            tokens.append(buf)
            buf = ""
        else:
            buf += ch
    if buf:
        tokens.append(buf)
    return tokens


def draw_box(ax, x: float, y: float, w: float, h: float, stage: dict) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.003,rounding_size=0.018",
            linewidth=1.4,
            edgecolor="#8A8A8A",
            facecolor=stage["color"],
            mutation_aspect=1.0,
        )
    )
    ax.text(
        x + w / 2,
        y + h - 0.055,
        stage["title"],
        ha="center",
        va="top",
        fontsize=12.5,
        fontweight="bold",
        color="#1B1B1B",
    )
    body_y = y + h - 0.165
    for line in stage["body"]:
        for wrapped in wrap_by_width(line, WRAP_LIMIT):
            ax.text(x + 0.006, body_y, wrapped, ha="left", va="top", fontsize=10.0, color="#2B2B2B")
            body_y -= 0.058


def main() -> None:
    n = len(STAGES)
    gap = 0.016
    left = 0.006
    box_w = (1.0 - left * 2 - gap * (n - 1)) / n
    box_h = 0.56
    box_y = 0.20

    fig, ax = plt.subplots(figsize=(30.0, 5.0), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.985,
        "第 3 章 ESRPO 结构偏好优化流程"
        "（Endpoint Structural-Rule Preference Optimization）",
        ha="center",
        va="top",
        fontsize=18,
        fontweight="bold",
        color="#111111",
    )
    ax.text(
        0.5,
        0.930,
        "横向流程：离线轨迹端点 → 结构规则偏好对 → 终点级偏好优化 → 停留点与输出",
        ha="center",
        va="top",
        fontsize=13,
        color="#555555",
    )

    x = left
    centers: list[tuple[float, float]] = []
    for stage in STAGES:
        draw_box(ax, x, box_y, box_w, box_h, stage)
        centers.append((x + box_w, box_y + box_h / 2))
        x += box_w + gap

    for i in range(n - 1):
        x0, y0 = centers[i]
        ax.add_patch(
            FancyArrowPatch(
                (x0 + 0.0008, y0),
                (x0 + gap - 0.0008, y0),
                arrowstyle="-|>,head_width=6,head_length=10",
                mutation_scale=1.0,
                linewidth=2.2,
                color="#4A6FA5",
                shrinkA=0,
                shrinkB=0,
            )
        )

    ax.text(
        0.006,
        0.030,
        "口径：候选池三分账与偏好对规模来自 pair_audit.json；训练配置来自 "
        "launch/diffsbdd_global_esrpo_x_only_lr1e5_mainline.yaml；"
        "产率 / 对接 / sanitize / QED 来自 analysis/e2po_mainline_reasonable_eval/"
        "reasonable_molecules_metrics.csv（固定 1000 次尝试）。QED 为合理分子中位数。",
        ha="left",
        va="bottom",
        fontsize=10.5,
        color="#666666",
    )

    fig.savefig(OUT_DIR / "esrpo_pipeline.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_DIR / "esrpo_pipeline.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", OUT_DIR / "esrpo_pipeline.png")


if __name__ == "__main__":
    main()
