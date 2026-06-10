from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Cm, Pt


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
PPTX_PATH = OUT_DIR / "四条线方向请教_20260610.pptx"
SPEECH_PATH = OUT_DIR / "speech.md"

IMG_SANITIZE = ROOT / "analysis/8gjl_c1101_benchmark_20260603_500mol_500steps/evaluation/six_models/sanitize_rate.png"
IMG_RL_DRIFT = ROOT / "analysis/rl_high_carbon_sparse_reward_20260604/rl_descriptor_trends.png"
IMG_ABLATION = ROOT / "analysis/refinement_ablation_smoke_atomdiag_20260603/sanitize_success_by_method.png"
IMG_XHE = ROOT / "analysis/diffsbdd_xhe_diffgui_scratch_20260606_longtest/longtest_overview.png"
IMG_DIFFGUI_RL = ROOT / "OUT/diffgui-8gpu-ring-last3-lr3e5_20260610_114650/metrics/training_summary.png"


NAVY = RGBColor(22, 42, 69)
TEAL = RGBColor(0, 137, 123)
BLUE = RGBColor(49, 102, 178)
ORANGE = RGBColor(232, 132, 76)
RED = RGBColor(200, 78, 82)
GREEN = RGBColor(77, 150, 98)
GRAY = RGBColor(94, 107, 124)
LIGHT_BG = RGBColor(246, 248, 251)
PANEL = RGBColor(255, 255, 255)
LINE = RGBColor(219, 226, 235)
PALE_TEAL = RGBColor(226, 244, 241)
PALE_BLUE = RGBColor(231, 239, 252)
PALE_RED = RGBColor(252, 235, 235)
PALE_ORANGE = RGBColor(253, 242, 229)

FONT = "Microsoft YaHei"


def add_bg(slide) -> None:
    bg = slide.background.fill
    bg.solid()
    bg.fore_color.rgb = LIGHT_BG


def add_footer(slide, idx: int) -> None:
    box = slide.shapes.add_textbox(Cm(0.65), Cm(18.55), Cm(31.0), Cm(0.35))
    tf = box.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    p.text = f"DiffB110 四条探索线方向请教 | 2026-06-10 | {idx}/10"
    p.font.name = FONT
    p.font.size = Pt(8)
    p.font.color.rgb = GRAY
    p.alignment = PP_ALIGN.RIGHT


def add_title(slide, title: str, subtitle: str | None = None) -> None:
    box = slide.shapes.add_textbox(Cm(0.8), Cm(0.45), Cm(31.5), Cm(1.05))
    tf = box.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    p.text = title
    p.font.name = FONT
    p.font.size = Pt(25)
    p.font.bold = True
    p.font.color.rgb = NAVY
    if subtitle:
        sub = slide.shapes.add_textbox(Cm(0.82), Cm(1.45), Cm(30.5), Cm(0.55))
        stf = sub.text_frame
        stf.clear()
        sp = stf.paragraphs[0]
        sp.text = subtitle
        sp.font.name = FONT
        sp.font.size = Pt(10.5)
        sp.font.color.rgb = GRAY


def add_tag(slide, x, y, text: str, fill=PALE_TEAL, color=TEAL, width=4.0) -> None:
    shape = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, Cm(x), Cm(y), Cm(width), Cm(0.65))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = fill
    tf = shape.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    p.text = text
    p.font.name = FONT
    p.font.size = Pt(9)
    p.font.bold = True
    p.font.color.rgb = color
    p.alignment = PP_ALIGN.CENTER


def add_panel(slide, x, y, w, h, fill=PANEL, line=LINE):
    shape = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, Cm(x), Cm(y), Cm(w), Cm(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = line
    shape.line.width = Pt(1)
    return shape


def add_text(slide, x, y, w, h, text: str, size=13, color=NAVY, bold=False, align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(Cm(x), Cm(y), Cm(w), Cm(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.clear()
    p = tf.paragraphs[0]
    p.text = text
    p.font.name = FONT
    p.font.size = Pt(size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.alignment = align
    return box


def add_bullets(slide, x, y, w, h, bullets: list[str], size=13, color=NAVY, bullet_color=None) -> None:
    box = slide.shapes.add_textbox(Cm(x), Cm(y), Cm(w), Cm(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = Cm(0.1)
    tf.margin_right = Cm(0.1)
    tf.clear()
    for i, item in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = item
        p.level = 0
        p.font.name = FONT
        p.font.size = Pt(size)
        p.font.color.rgb = color
        p.space_after = Pt(5)
        p._p.get_or_add_pPr().set("marL", "250000")
        p._p.get_or_add_pPr().set("hanging", "120000")
        if bullet_color:
            p.font.color.rgb = bullet_color


def add_image_fit(slide, image: Path, x, y, w, h) -> None:
    if not image.exists():
        add_placeholder(slide, x, y, w, h, f"缺失图片\n{image.name}")
        return
    slide.shapes.add_picture(str(image), Cm(x), Cm(y), width=Cm(w), height=Cm(h))


def add_placeholder(slide, x, y, w, h, text: str) -> None:
    shape = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, Cm(x), Cm(y), Cm(w), Cm(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor(250, 252, 255)
    shape.line.color.rgb = RGBColor(168, 181, 196)
    shape.line.dash_style = 2
    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.name = FONT
    p.font.size = Pt(13)
    p.font.color.rgb = GRAY
    p.alignment = PP_ALIGN.CENTER


def add_metric_table(slide, x, y, rows: list[tuple[str, str, str, str]], col_w=(5.0, 3.0, 3.0, 5.2)) -> None:
    table = slide.shapes.add_table(len(rows) + 1, 4, Cm(x), Cm(y), Cm(sum(col_w)), Cm(0.75 * (len(rows) + 1))).table
    headers = ["指标", "epoch1", "epoch9", "解读"]
    for c, label in enumerate(headers):
        cell = table.cell(0, c)
        cell.text = label
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        for p in cell.text_frame.paragraphs:
            p.font.name = FONT
            p.font.size = Pt(9)
            p.font.bold = True
            p.font.color.rgb = RGBColor(255, 255, 255)
    for r, row in enumerate(rows, start=1):
        for c, val in enumerate(row):
            cell = table.cell(r, c)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(255, 255, 255) if r % 2 else RGBColor(244, 247, 250)
            for p in cell.text_frame.paragraphs:
                p.font.name = FONT
                p.font.size = Pt(8.3)
                p.font.color.rgb = NAVY
    for c, width in enumerate(col_w):
        table.columns[c].width = Cm(width)


def slide_cover(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(slide)
    add_tag(slide, 0.9, 0.8, "方向请教 / 非最终成果", fill=PALE_ORANGE, color=ORANGE, width=5.0)
    add_text(slide, 1.0, 2.05, 24.5, 1.3, "DiffB110：RE-CmeB 靶点适配分子生成的四条探索线", size=27, bold=True)
    add_text(slide, 1.05, 3.35, 22.0, 0.7, "生成质量、靶点适配与课题路线选择", size=16, color=GRAY)
    add_text(slide, 1.05, 4.55, 23.0, 0.8, "RE-CmeB / DiffSBDD / DiffGui / DDPO-PPO / x-h-e diffusion", size=12, color=TEAL, bold=True)
    for i, (label, color) in enumerate([("C: DiffSBDD RL", BLUE), ("A: 内部小改", GREEN), ("B: x/h/e 适配", ORANGE), ("D: DiffGui RL", TEAL)]):
        add_panel(slide, 1.0 + i * 7.6, 6.4, 6.8, 2.4, fill=RGBColor(255, 255, 255))
        add_text(slide, 1.35 + i * 7.6, 6.78, 6.0, 0.5, label, size=14, color=color, bold=True, align=PP_ALIGN.CENTER)
        desc = ["成熟但 reward hacking", "低风险但改善有限", "触及根因但高风险", "当前最倾向路线"][i]
        add_text(slide, 1.35 + i * 7.6, 7.55, 6.0, 0.55, desc, size=10, color=GRAY, align=PP_ALIGN.CENTER)
    add_text(slide, 1.1, 10.35, 28.0, 1.0, "目标：请老师判断哪条路线最有希望生成更适配 RE-CmeB 靶点、且化学合理的候选分子。", size=15, color=NAVY)
    add_footer(slide, 1)


def slide_bases(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(slide)
    add_title(slide, "两个底座：DiffSBDD vs DiffGui", "同一个 RL 问题，在不同生成底座上难度完全不同")
    add_panel(slide, 0.9, 2.25, 15.0, 5.0)
    add_panel(slide, 17.0, 2.25, 15.0, 5.0)
    add_text(slide, 1.25, 2.55, 5.5, 0.5, "DiffSBDD", size=18, color=BLUE, bold=True)
    add_bullets(slide, 1.25, 3.2, 13.8, 3.4, [
        "生成 ligand 坐标 x 与原子类型 h",
        "最后依赖 OpenBabel/RDKit 从点云重建键",
        "速度快、RL 工程成熟",
        "成键/合法性弱，RL 容易陷入坏分子惩罚",
    ], size=12)
    add_text(slide, 17.35, 2.55, 5.5, 0.5, "DiffGui", size=18, color=TEAL, bold=True)
    add_bullets(slide, 17.35, 3.2, 13.8, 3.4, [
        "原生建模 atom、bond/halfedge 与坐标",
        "更接近 x/h/e 联合生成",
        "成功写入样本可重建合法性更强",
        "但 raw generation 仍有 reconstruction failure 和复杂环偏差",
    ], size=12)
    add_placeholder(slide, 1.0, 8.0, 14.9, 5.6, "占位图：DiffSBDD 生成分子样貌\n用户后续手动加入\n不要自动补图")
    add_placeholder(slide, 17.0, 8.0, 15.0, 5.6, "占位图：DiffGui 生成分子样貌\n用户后续手动加入\n不要自动补图")
    add_text(slide, 1.0, 14.2, 30.0, 0.55, "关键口径：DiffGui 的 100% sanitize 是“成功写入/进入评估 SDF”口径，不代表原始生成全流程成功。", size=11, color=RED, bold=True)
    add_footer(slide, 2)


def slide_context(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(slide)
    add_title(slide, "共同背景：底座质量决定 RL 难度", "横评说明 DiffGui 更强，但 sanitize 口径必须谨慎")
    add_image_fit(slide, IMG_SANITIZE, 0.9, 2.2, 14.8, 9.1)
    add_panel(slide, 16.5, 2.15, 15.3, 9.5)
    add_bullets(slide, 17.0, 2.65, 14.0, 5.3, [
        "DiffSBDD：评估 SDF sanitize 约 66.6%，速度和 docking 潜力不差",
        "DiffGui：进入评估 SDF sanitize 为 100%，但为得到 500 个成功样本另有 55 个失败样本被丢弃",
        "两者都会生成复杂环/稠环；DiffGui ring mean=4.68，芳香分子率=70.2%，carbon fraction=0.820",
        "底座弱时，RL 学习信号会被坏分子惩罚稀释",
    ], size=12)
    add_text(slide, 17.0, 9.25, 13.8, 1.35, "所以问题不是“哪个 reward 更花哨”，而是：RL 应该建立在什么底座上，才能学到稳定方向？", size=15, color=NAVY, bold=True)
    add_footer(slide, 3)


def slide_c(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(slide)
    add_title(slide, "C 线：DiffSBDD RL 跑通，但 reward hacking 明显", "工程最成熟，但模型会钻 reward 空子")
    add_image_fit(slide, IMG_RL_DRIFT, 0.9, 2.1, 15.4, 9.6)
    add_panel(slide, 17.0, 2.1, 14.9, 9.6)
    add_bullets(slide, 17.5, 2.65, 13.8, 4.6, [
        "DDPO/PPO、多 GPU、reward factory、descriptor logging、checkpoint 已经完整",
        "旧训练暴露出高碳化、TPSA 下降、杂原子减少",
        "classifier reward 太稀疏，docking 与 classifier 目标冲突",
        "有效好样本太少，PPO 容易学到偏离分布的捷径",
    ], size=12)
    add_text(slide, 17.5, 8.3, 12.8, 0.5, "下一步候选修复", size=14, color=BLUE, bold=True)
    add_bullets(slide, 17.5, 9.0, 13.2, 2.2, [
        "KL / prior-agent 约束",
        "ChemistryGuard + descriptor drift 监控",
        "GoodMoleculePool + dense classifier shaping",
    ], size=11)
    add_footer(slide, 4)


def slide_a(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(slide)
    add_title(slide, "A 线：DiffSBDD 小改低风险，但改善有限", "适合作为 ablation 或辅助增强，不宜单独承担药学目标")
    add_image_fit(slide, IMG_ABLATION, 0.95, 2.2, 15.0, 9.2)
    add_panel(slide, 16.8, 2.2, 15.0, 9.2)
    add_bullets(slide, 17.3, 2.7, 13.8, 3.8, [
        "不改变 DiffSBDD 扩散变量，只加 atom prior / geometry refine / local mask / bond head",
        "工程风险低，容易接到现有 RL 流程",
        "但不引入显式 bond diffusion，最终成键仍依赖规则重建",
        "可能缓解现象，但难以解决根因",
    ], size=12)
    add_text(slide, 17.3, 8.0, 12.9, 1.25, "定位：低风险修补方案，可作为对照和 ablation；但不一定显著提高候选分子质量。", size=15, color=GREEN, bold=True)
    add_footer(slide, 5)


def slide_b(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(slide)
    add_title(slide, "B 线：x/h/e 大改触及根因，但不是原创机制", "已有 joint atom-bond-coordinate 思路在 DiffSBDD 上的适配/重构")
    add_image_fit(slide, IMG_XHE, 0.9, 2.2, 15.4, 9.2)
    add_panel(slide, 17.0, 2.2, 14.9, 9.2)
    add_bullets(slide, 17.5, 2.7, 13.6, 4.2, [
        "目标：把 bond / halfedge 作为 e 通道加入扩散过程",
        "已完成 halfedge 数据链、edge denoiser、xhe joint skeleton、DiffGui 对照分析",
        "关键认识：不是最后预测键，而是每一步联合去噪时 edge 参与坐标更新",
        "真实定位：参考已有联合建模思想，做 DiffSBDD bond-aware 适配",
    ], size=11.5)
    add_text(slide, 17.5, 8.0, 12.9, 0.5, "主要风险", size=14, color=ORANGE, bold=True)
    add_bullets(slide, 17.5, 8.65, 13.2, 2.1, [
        "loss scale / edge schedule / joint sampling 难稳定",
        "DDPO logprob 对接工作量大",
        "即使完成，也不确定超过 DiffGui",
    ], size=11)
    add_footer(slide, 6)


def slide_d(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(slide)
    add_title(slide, "D 线：DiffGui RL 已跑通，进入分布稳定化测试", "当前最倾向路线，但不能说已经成功")
    add_image_fit(slide, IMG_DIFFGUI_RL, 0.9, 2.1, 14.3, 8.4)
    rows = [
        ("reward_total", "0.406", "0.471", "中间 e3/e4 更高后回落"),
        ("reconstruction", "0.859", "0.896", "略稳定"),
        ("docking_raw", "-9.96", "-9.42", "未同步改善"),
        ("ring_reward", "0.279", "0.355", "轻微改善"),
        ("ring_count", "4.14", "3.96", "略下降"),
        ("carbon_fraction", "0.784", "0.753", "高碳缓解"),
    ]
    add_metric_table(slide, 16.0, 2.25, rows)
    add_text(slide, 16.0, 8.1, 15.0, 1.0, "配置：8GPU / 200 steps / batch 192 / last3 / lr=3e-5 / 24 epochs", size=12, color=TEAL, bold=True)
    add_bullets(slide, 16.0, 9.2, 14.8, 2.2, [
        "ring/carbon 有早期改善信号",
        "docking 尚未同步改善",
        "下一步重点是 distribution-stabilized RL、多目标权重和 PPO 稳定性",
    ], size=11.5)
    add_footer(slide, 7)


def slide_matrix(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(slide)
    add_title(slide, "四条线质量-可控性判断", "当前建议：主推 D，保留 C，收缩 B，A 做辅助")
    add_panel(slide, 2.0, 2.2, 27.8, 10.5)
    # axes
    slide.shapes.add_connector(1, Cm(4.2), Cm(11.0), Cm(27.0), Cm(11.0)).line.color.rgb = GRAY
    slide.shapes.add_connector(1, Cm(4.2), Cm(11.0), Cm(4.2), Cm(3.0)).line.color.rgb = GRAY
    add_text(slide, 12.0, 11.25, 8.5, 0.5, "短期可控性 →", size=12, color=GRAY, align=PP_ALIGN.CENTER)
    add_text(slide, 0.85, 5.8, 3.0, 1.0, "分子质量\n改善确定性 ↑", size=12, color=GRAY, align=PP_ALIGN.CENTER)
    points = [
        ("A", "小改\n低风险辅助", 9.5, 8.8, GREEN, PALE_TEAL),
        ("C", "DiffSBDD RL\n成熟对照", 20.3, 7.6, BLUE, PALE_BLUE),
        ("B", "x/h/e 适配\n高风险重构", 8.4, 4.2, ORANGE, PALE_ORANGE),
        ("D", "DiffGui RL\n当前主推", 22.0, 4.0, TEAL, PALE_TEAL),
    ]
    for label, text, x, y, color, fill in points:
        add_panel(slide, x, y, 5.2, 1.65, fill=fill, line=color)
        add_text(slide, x + 0.25, y + 0.18, 0.7, 0.5, label, size=16, color=color, bold=True)
        add_text(slide, x + 1.0, y + 0.2, 3.9, 1.0, text, size=11, color=NAVY)
    add_text(slide, 3.0, 13.15, 26.0, 0.8, "注意：B 线不是原创机制最高，而是已有联合建模思想在 DiffSBDD 上的高风险适配/重构。", size=12, color=RED, bold=True, align=PP_ALIGN.CENTER)
    add_footer(slide, 8)


def slide_questions(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(slide)
    add_title(slide, "想请教老师的关键问题", "希望得到的是方向判断，而不是单个实验细节建议")
    qs = [
        ("1", "哪条线最可能改善 RE-CmeB 候选分子质量？", "底座、docking、化学合理性和分布稳定性要一起看。"),
        ("2", "reward hacking 会不会损害药学可用性？", "分数好看但高碳、低极性、复杂环偏移，可能并不适合后续筛选。"),
        ("3", "B 线是否值得继续？", "还是应收缩成 bond-aware adapter / reconstruction 模块？"),
        ("4", "下一步验证应优先补什么？", "docking/LE、化学合理性、scaffold 多样性，还是人工可检查候选分子？"),
    ]
    for i, (num, title, desc) in enumerate(qs):
        x = 1.5 + (i % 2) * 15.2
        y = 2.5 + (i // 2) * 4.6
        add_panel(slide, x, y, 14.0, 3.55)
        add_tag(slide, x + 0.45, y + 0.35, f"Q{num}", fill=PALE_BLUE, color=BLUE, width=1.6)
        add_text(slide, x + 2.3, y + 0.35, 10.7, 0.65, title, size=14, color=NAVY, bold=True)
        add_text(slide, x + 0.7, y + 1.35, 12.2, 1.25, desc, size=12, color=GRAY)
    add_footer(slide, 9)


def slide_stories(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(slide)
    add_title(slide, "两种可收敛课题路线", "当前个人倾向 Story A，但需要老师判断哪条更利于产出可用候选分子")
    add_panel(slide, 1.0, 2.4, 14.8, 8.8, fill=PALE_TEAL, line=TEAL)
    add_panel(slide, 17.0, 2.4, 14.8, 8.8, fill=PALE_ORANGE, line=ORANGE)
    add_text(slide, 1.7, 3.0, 13.0, 0.7, "路线 A：DiffGui 靶点适配生成优化", size=18, color=TEAL, bold=True)
    add_bullets(slide, 1.75, 4.0, 12.8, 4.0, [
        "主线：D 线 DiffGui RL",
        "目标：生成更适配 RE-CmeB、且化学合理的候选分子",
        "重点：docking/LE 与 ring/carbon 分布稳定化",
        "C 线提供 reward hacking 诊断和 guard/pool/KL 工具链",
    ], size=12)
    add_text(slide, 1.75, 9.25, 12.8, 0.8, "当前更推荐：底座更稳，更可能服务药学筛选目标。", size=13, color=TEAL, bold=True)
    add_text(slide, 17.7, 3.0, 13.0, 0.7, "路线 B：DiffSBDD 的 bond-aware 适配", size=18, color=ORANGE, bold=True)
    add_bullets(slide, 17.75, 4.0, 12.8, 4.0, [
        "主线：B 线，或收缩后的轻量 bond-aware 模块",
        "目标：修复 DiffSBDD 成键短板，提高有效分子比例",
        "风险：完整链条重构，收益不确定",
        "RL 作为 RE-CmeB 靶点适配应用验证",
    ], size=12)
    add_text(slide, 17.75, 9.25, 12.8, 0.8, "需要老师判断：是否值得继续押注，还是作为支线保留。", size=13, color=ORANGE, bold=True)
    add_text(slide, 3.2, 12.35, 26.5, 0.9, "结论口径：当前是路线选择与药学目标对齐阶段，不是最终成果展示。", size=15, color=NAVY, bold=True, align=PP_ALIGN.CENTER)
    add_footer(slide, 10)


def write_speech() -> None:
    SPEECH_PATH.write_text(
        """# DiffB110 四条探索线方向请教 speaker notes

## Slide 1: 封面

这次汇报不是最终成果展示，而是想请老师帮我判断哪条技术路线最有希望服务 RE-CmeB 靶点适配分子生成。我目前围绕 3D 分子扩散模型强化学习做了四条探索线：DiffSBDD 强化学习、DiffSBDD 内部小改、DiffSBDD x/h/e 大改，以及 DiffGui 强化学习。核心问题是：如何在 docking、化学合理性、分布稳定性和候选分子可用性之间取舍。

---

注意点：
- 重点：这是方向请教，不是包装结果。
- 节奏：开场直接说明希望老师帮忙判断哪条路线更服务药学目标。

## Slide 2: 两个底座：DiffSBDD vs DiffGui

这里先解释两个底座的差别。DiffSBDD 主要生成坐标和原子类型，最后靠规则工具重建键，所以速度快、工程成熟，但成键质量弱。DiffGui 则原生建模 atom、bond/halfedge 和坐标，更接近联合生成，所以成功样本合法性更强。但这里必须谨慎，DiffGui 的 100% sanitize 是成功写入并进入评估的 SDF 口径，不代表原始生成没有失败。

---

注意点：
- 重点：不要把 DiffGui 说成完全无失败。
- 补充：左右分子图由用户后续手动加入。

## Slide 3: 共同背景：底座质量决定 RL 难度

这页是共同背景。同一靶点横评里，DiffSBDD 的 sanitize 只有约 66.6%，DiffGui 成功写入样本为 100%。但 DiffGui 仍有 raw reconstruction failure，而且它也会生成复杂环和高碳分子。这说明问题不是单纯哪个模型合法性更好，而是 RL 会在什么样的底座上学习。如果底座本身坏分子太多，reward 信号会大量消耗在惩罚坏分子上。

---

注意点：
- 画面引导：先看 sanitize 图，再看右侧口径说明。
- 重点：评估口径必须讲清楚。

## Slide 4: C 线：DiffSBDD RL 跑通，但 reward hacking 明显

C 线是工程最成熟的一条。DDPO/PPO、多 GPU、reward factory、描述子监控这些都跑通了。但旧训练暴露出明显 reward hacking：模型向高碳、低极性、少杂原子方向漂移。classifier 信号稀疏，docking 又偏好某些疏水复杂结构，所以 PPO 很容易学到不理想的捷径。

---

注意点：
- 重点：C 线价值是工具链和诊断，不一定最能产出好分子。

## Slide 5: A 线：DiffSBDD 小改低风险，但改善有限

A 线是低风险修补方案，比如 atom prior、geometry refine、local mask 和 bond head。它的优点是容易接入现有 DiffSBDD 和 RL 流程，但它没有改变扩散变量，最终成键仍然靠规则重建。因此它更适合做 ablation 或辅助增强，不一定能单独显著提高 RE-CmeB 候选分子的质量。

---

注意点：
- 重点：低风险不等于一定能改善药学可用性。

## Slide 6: B 线：x/h/e 大改触及根因，但不是原创机制

B 线结构上最接近 DiffSBDD 的成键根因，因为它尝试把 bond/halfedge 作为 e 通道加入扩散。但这个方向不能写成原创机制，因为 x/h/e 或 atom-bond-coordinate 联合建模已经被 DiffGui 等模型实现过。更准确地说，这是把已有联合建模思想适配到 DiffSBDD 上，工程风险很高，也不确定最终能否超过直接使用 DiffGui。

---

注意点：
- 重点：不要说“创新性最高”。
- 补充：这里的价值是适配、验证和对照。

## Slide 7: D 线：DiffGui RL 已跑通，进入分布稳定化测试

D 线是我目前最倾向的路线。DiffGui 底座更强，RL rollout、logprob、PPO actor 和 reward-only 子进程已经接通。最新 ring-quality 测试中，ring 和 carbon 有一点改善信号，但 docking 没有同步改善。这说明路线有希望，但不能说已经成功。下一步真正要看的是它能不能生成更适配 RE-CmeB、且化学更合理的候选分子。

---

注意点：
- 重点：ring/carbon 有早期信号，docking 未同步改善。
- 节奏：这里要诚实，不要夸大。

## Slide 8: 四条线风险-收益判断

这页是我目前的路线判断。C 线成熟，适合作为工具链和对照；A 线低风险，适合作为 ablation；B 线触及根因但属于高风险适配；D 线底座最好、工程已跑通、成本可控，所以最适合作为当前主推候选。

---

注意点：
- 重点：主推 D，不是否定其它线，而是给它们重新定位。

## Slide 9: 想请教老师的关键问题

这里我想请老师帮忙判断几个关键问题：哪条路线最可能改善 RE-CmeB 候选分子质量？reward hacking 会不会让分子分数好看但药学可用性变差？B 线是否值得继续投入，还是收缩成适配模块？下一步更应该补 docking/LE、化学合理性、scaffold 多样性，还是人工可检查候选分子？

---

注意点：
- 重点：问题要聚焦方向选择。

## Slide 10: 两种可收敛课题路线

最后我把可能路线收敛成两个。路线 A 是 DiffGui 上的靶点适配生成优化，这是我当前更倾向的方向。路线 B 是 DiffSBDD 的 bond-aware adaptation，风险更高，更适合作为支线或后续扩展。我希望老师判断的是：哪条路线更可能产出可用候选分子，以及 B 线是否还值得继续押注。

---

注意点：
- 重点：给出个人倾向，但保留老师判断空间。
""",
        encoding="utf-8",
    )


def build() -> None:
    prs = Presentation()
    prs.slide_width = Cm(33.867)
    prs.slide_height = Cm(19.05)
    slide_cover(prs)
    slide_bases(prs)
    slide_context(prs)
    slide_c(prs)
    slide_a(prs)
    slide_b(prs)
    slide_d(prs)
    slide_matrix(prs)
    slide_questions(prs)
    slide_stories(prs)
    write_speech()
    prs.save(PPTX_PATH)
    print(PPTX_PATH)
    print(SPEECH_PATH)


if __name__ == "__main__":
    build()
