/**
 * RE-CmeB 靶点分子生成（第 2–3 章）汇报 PPT 生成脚本
 *
 * 遵循 academic-pptx skill：
 *   - 每个内容页的标题都是"行动标题"（完整结论句），幽灵 deck 测试通过
 *   - 每页一张图；结果页图在左、解读在右（宽幅图改为全宽 + 下方要点）
 *   - 正文 ≥20 pt、每页 ≤40 词、单一字体、白底
 *   - 以 Conclusions 页收尾（非 "Thank You"），附参考文献页与附录
 *
 * 技术栈：pptxgenjs（skill 原套）；配色沿用仓库既有 build_ppt.py 方案（选项 ②）。
 */

const pptxgen = require("pptxgenjs");
const path = require("path");

// --------------------------------------------------------------------------
// 配色：与仓库 build_ppt.py 保持一致（选项 ②）
// --------------------------------------------------------------------------
const C = {
  navy: "162A45", teal: "00897B", blue: "3166B2", orange: "E8844C",
  red: "C84E52", green: "4D9662", gray: "5E6B7C",
  bg: "F6F8FB", panel: "FFFFFF", line: "DBE2EB",
  paleTeal: "E2F4F1", paleBlue: "E7EFFC", paleRed: "FCEBEB", paleOrange: "FDF2E5",
};

const F = { face: "Microsoft YaHei", title: 26, h2: 20, body: 20, label: 16, cite: 13 };
const M = 0.5;                 // 页边距（英寸）
const SLIDE_W = 10, SLIDE_H = 5.625;
const CW = SLIDE_W - 2 * M;    // 内容宽度 9.0

const IMG = (f) => path.join(__dirname, "..", "images", f);

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.title = "RE-CmeB 靶点分子生成：从代理奖励套利到独立口径的偏好优化";
pres.author = "谢东宸";

// --------------------------------------------------------------------------
// 组件
// --------------------------------------------------------------------------
function newSlide(bg = C.panel) {
  const s = pres.addSlide();
  s.background = { color: bg };
  return s;
}

/** 行动标题：必须是一句完整结论句 */
function title(slide, text, h = 0.9) {
  slide.addText(text, {
    x: M, y: 0.2, w: CW, h, fontFace: F.face, fontSize: F.title,
    color: C.navy, bold: true, valign: "top", lineSpacingMultiple: 1.05,
  });
}

function divider(slide, y) {
  slide.addShape(pres.ShapeType.rect, {
    x: M, y, w: CW, h: 0.02, fill: { color: C.line }, line: { color: C.line },
  });
}

/** 正文要点（skill：每页 ≤40 词、3–5 条） */
function bullets(slide, items, x, y, w, h) {
  slide.addText(
    items.map((t) => (typeof t === "string"
      ? { text: t, options: { breakLine: true } }
      : { ...t, options: { ...(t.options || {}), breakLine: true } })),
    {
      x, y, w, h, fontFace: F.face, fontSize: F.body, color: C.navy,
      bullet: { indent: 14 }, paraSpaceAfter: 7, valign: "top",
    }
  );
}

function cite(slide, text, y = 5.2) {
  slide.addText(text, {
    x: M, y, w: CW, h: 0.3, fontFace: F.face, fontSize: F.cite,
    color: C.gray, valign: "middle",
  });
}

/** 右对齐页码 */
function pageNo(slide, n, total) {
  slide.addText(`${n} / ${total}`, {
    x: SLIDE_W - 1.3, y: 5.24, w: 0.9, h: 0.28,
    fontFace: F.face, fontSize: 11, color: C.gray, align: "right",
  });
}

/** 全宽图（用于 3.6:1 的宽幅图） */
function wideFigure(slide, img, aspect, y = 1.2, w = CW) {
  slide.addImage({ path: img, x: M, y, w, h: w / aspect });
  return y + w / aspect;
}

/** 左图右文（用于 1.83:1 的常规图） */
function leftFigure(slide, img, aspect, bulletItems) {
  const w = 5.15, h = w / aspect;
  slide.addImage({ path: img, x: M, y: 1.25, w, h });
  bullets(slide, bulletItems, 5.85, 1.2, 3.65, 3.8);
}

/** 提示条（用于强调关键数字） */
function callout(slide, text, x, y, w, kind = "teal") {
  const map = {
    teal: { bg: C.paleTeal, fg: C.teal },
    red: { bg: C.paleRed, fg: C.red },
    orange: { bg: C.paleOrange, fg: C.orange },
  };
  const c = map[kind];
  slide.addShape(pres.ShapeType.roundRect, {
    x, y, w, h: 0.5, fill: { color: c.bg }, line: { color: c.fg, width: 1 }, rectRadius: 0.06,
  });
  slide.addText(text, {
    x, y, w, h: 0.5, fontFace: F.face, fontSize: F.label, color: c.fg,
    bold: true, align: "center", valign: "middle",
  });
}

const TOTAL = 17;

// ==========================================================================
// Slide 1 — 封面
// ==========================================================================
{
  const s = newSlide(C.navy);
  s.addText("RE-CmeB 靶点分子生成", {
    x: 0.7, y: 1.45, w: 8.6, h: 0.7, fontFace: F.face, fontSize: 34,
    color: "FFFFFF", bold: true,
  });
  s.addText("从代理奖励套利到独立口径的偏好优化", {
    x: 0.7, y: 2.15, w: 8.6, h: 0.6, fontFace: F.face, fontSize: 24,
    color: "A0BBDD", bold: false,
  });
  s.addShape(pres.ShapeType.rect, {
    x: 0.7, y: 2.95, w: 2.0, h: 0.04, fill: { color: C.teal }, line: { color: C.teal },
  });
  s.addText("第 2–3 章工作汇报", {
    x: 0.7, y: 3.2, w: 8.6, h: 0.4, fontFace: F.face, fontSize: 16, color: "CADCFC",
  });
  s.addText("谢东宸\n[培养院系]  ·  2026 年 9 月", {
    x: 0.7, y: 3.7, w: 8.6, h: 0.7, fontFace: F.face, fontSize: 15, color: "CADCFC",
  });
  s.addNotes("本次汇报覆盖论文第 2 章与第 3 章。核心是一条完整的论证链：先用硬指标把模型适配好，结果发现指标改善的来源是结构病理；于是换一把独立于奖励的尺子重选底座，再换优化范式，最后在对接基本持平下把结构问题压下去。");
}

// ==========================================================================
// Slide 2 — 背景（Situation）
// ==========================================================================
{
  const s = newSlide();
  title(s, "预训练生成模型不含靶点偏好，直接用于 RE-CmeB 无法保证分子可用", 0.85);
  divider(s, 1.12);
  bullets(s, [
    { text: "生成模型的训练目标是拟合数据分布", options: { bold: true } },
    "参数中不含任何口袋特异性；换一个靶点必须重新适配",
    { text: "适配手段是奖励，而奖励只是代理量", options: { bold: true } },
    "对接分数、QED、SA score 都不等于真实目标 → 适配结果必须用独立口径检验",
  ], M, 1.35, CW, 3.4);
  cite(s, "RE-CmeB 靶点：PDB 8GJL，口袋 C:1101");
  pageNo(s, 2, TOTAL);
  s.addNotes("开场先立问题：预训练模型学的是数据分布，参数里没有靶点偏好。用到具体靶点上必须做适配，而适配只能靠奖励——奖励又只是代理量。这就埋下了后面 reward hacking 的伏笔。");
}

// ==========================================================================
// Slide 3 — 研究问题（skill 强制独立成页）
// ==========================================================================
{
  const s = newSlide();
  title(s, "核心问题：能否在不牺牲对接的前提下，抑制生成分子的结构病理？", 0.85);
  divider(s, 1.12);
  s.addShape(pres.ShapeType.roundRect, {
    x: 1.4, y: 1.5, w: 7.2, h: 1.5, fill: { color: C.paleBlue },
    line: { color: C.blue, width: 1.5 }, rectRadius: 0.08,
  });
  s.addText(
    "用标量奖励把模型适配好之后，\n能否不以其结构合理性为代价？\n能否用一个独立于训练奖励的口径来度量并抑制这种代价？",
    {
      x: 1.6, y: 1.62, w: 6.8, h: 1.3, fontFace: F.face, fontSize: 18,
      color: C.navy, align: "center", valign: "middle", lineSpacingMultiple: 1.25,
    }
  );
  bullets(s, [
    "第 2 章建立可复用的适配与检验流程，并完成一次完整实现",
    "第 3 章以该口径为起点，更换底座选择依据与优化范式",
    { text: "判据必须独立于训练奖励", options: { bold: true, color: C.red } },
  ], M, 3.3, CW, 1.7);
  pageNo(s, 3, TOTAL);
  s.addNotes("研究问题必须有独立一页。本文用两章回答：第2章建立一套适配与诊断流程并跑一次完整实现；第3章沿着诊断结论改底座、改范式。贯穿全程的原则是：判据要独立于训练奖励。");
}

// ==========================================================================
// Slide 4 — 方法总览（流程图，原生形状）
// ==========================================================================
{
  const s = newSlide();
  title(s, "本文建立两条递进的流程：先用硬指标适配，再用独立口径审查并重选", 0.85);
  divider(s, 1.12);

  const rowY = [1.35, 3.05];
  const labels = [
    { tag: "第 2 章", color: C.blue, steps: ["统一对接口径", "五模型横评", "选 DiffGui", "强化学习适配", "独立口径诊断"] },
    { tag: "第 3 章", color: C.teal, steps: ["该口径下复评", "改以 DiffSBDD", "成对偏好优化", "合理分子 55.6%", "对接基本持平"] },
  ];
  labels.forEach((row, ri) => {
    const y = rowY[ri];
    s.addShape(pres.ShapeType.roundRect, {
      x: M, y: y + 0.12, w: 0.9, h: 0.5, fill: { color: row.color },
      line: { color: row.color }, rectRadius: 0.05,
    });
    s.addText(row.tag, {
      x: M, y: y + 0.12, w: 0.9, h: 0.5, fontFace: F.face, fontSize: 14,
      color: "FFFFFF", bold: true, align: "center", valign: "middle",
    });
    row.steps.forEach((st, i) => {
      const x = 1.55 + i * 1.60;
      s.addShape(pres.ShapeType.roundRect, {
        x, y: y + 0.12, w: 1.5, h: 0.5, fill: { color: C.panel },
        line: { color: row.color, width: 1 }, rectRadius: 0.05,
      });
      s.addText(st, {
        x, y: y + 0.12, w: 1.5, h: 0.5, fontFace: F.face, fontSize: 12,
        color: C.navy, align: "center", valign: "middle",
      });
      if (i < row.steps.length - 1) {
        s.addText("▶", {
          x: x + 1.5, y: y + 0.12, w: 0.12, h: 0.5, fontFace: F.face,
          fontSize: 9, color: C.gray, align: "center", valign: "middle",
        });
      }
    });
  });

  s.addText("诊断结论作为转折：硬指标改善，但结构病理同步恶化", {
    x: M, y: 2.42, w: CW, h: 0.4, fontFace: F.face, fontSize: 15,
    color: C.red, bold: true, align: "center", valign: "middle",
  });
  cite(s, "两条流程为递进关系：第 2 章的诊断界定了第 2 章结论的适用范围，并构成第 3 章的问题起点");
  pageNo(s, 4, TOTAL);
  s.addNotes("这页是全场的地图。上半是第2章：统一口径、五模型横评、选 DiffGui、强化学习适配，最后用独立口径诊断一次。中间的红色一句是转折——正是这个诊断否定了自己的指标口径。下半是第3章：在该口径下复评底座、改用 ESRPO。");
}

// ==========================================================================
// Slide 5 — 第 2 章 硬指标结果
// ==========================================================================
{
  const s = newSlide();
  title(s, "在硬指标口径下，DiffGui 强化学习完成了有效的靶点适配", 0.85);
  divider(s, 1.12);
  leftFigure(s, IMG("fig2_3_training_curve.png"), 1320 / 720, [
    { text: "对接均值 −9.78 → 最优 −11.28（约 15%）", options: { bold: true } },
    "最优出现在 epoch 79",
    { text: "RDKit valid / docking success：0.8906 → 0.9531", options: {} },
    "重复率为 0；8 卡、全模型微调、t=200、等权奖励",
  ]);
  cite(s, "图 2.3 DiffGui 主训练逐 epoch 的奖励、对接与有效性（ep1–89）");
  pageNo(s, 5, TOTAL);
  s.addNotes("先给正面结果。这一页要讲清楚：强化学习确实把模型往高对接推了约 15%，有效性也上去了，重复率为零。这部分是实实在在成立的，而且是后面所有论证的起点。");
}

// ==========================================================================
// Slide 6 — 诊断（关键页）
// ==========================================================================
{
  const s = newSlide();
  title(s, "但同一批分子的环系病理同步恶化：三元环率由 12.3% 升至 44.3%", 0.85);
  divider(s, 1.12);
  leftFigure(s, IMG("fig2_5_proxy_reward_exploitation.png"), 1320 / 720, [
    { text: "三元环率 12.3% → 44.3%（约 3.6 倍）", options: { bold: true, color: C.red } },
    "螺环率 12.3% → 27.9%，桥环率 14.0% → 19.7%，芳香环数下降",
    { text: "重原子数基本不变（18.2 → 19.0）", options: { bold: true } },
    "→ 排除“生成更大分子刷分”的解释",
  ]);
  cite(s, "图 2.5 代理奖励套利诊断（主训练 ep1–89）；表 2.6 主训练前后环系与刚性指标");
  pageNo(s, 6, TOTAL);
  s.addNotes("这是全场的转折点。硬指标改善的同时，环系病理同步恶化。注意强调重原子数基本不变——这一点很关键，它排除了“模型只是把分子做大所以对接分变低”这个最常见的质疑。所以指标改善的来源里，至少有一部分是结构病理的富集。");
}

// ==========================================================================
// Slide 7 — 单分子证据（第 2 章侧）
// ==========================================================================
{
  const s = newSlide();
  title(s, "在对接最好的分子中，含三元环的比例由 3/10 升至 8/10", 0.8);
  divider(s, 1.05);
  wideFigure(s, IMG("fig2_6_diffgui_rl_top_docking.png"), 4600 / 1268, 1.15);
  bullets(s, [
    { text: "选样：对接分数最优的 10 个 —— 奖励函数直接指向的区域，非事后挑选", options: { bold: true } },
    "三元环 / R4b / R4c 命中数 30% → 80%",
    "该子集重原子数 23.6 → 27.1（图中已标注，尺寸混杂透明化）",
  ], M, 3.78, CW, 1.4);
  pageNo(s, 7, TOTAL);
  s.addNotes("上一页是统计证据，这页是单分子视觉证据。选样规则是“对接最优的10个”，也就是奖励函数直接指向的区域——所以这不是挑样本，而是看奖励想要什么。上排是 ep1，下排是 ep89，红色高亮是三元环：命中比例从 3/10 涨到 8/10。同时如实标注该子集重原子数从 23.6 涨到 27.1。");
}

// ==========================================================================
// Slide 8 — 根因
// ==========================================================================
{
  const s = newSlide();
  title(s, "根因是评价口径与训练奖励同源，改善无法自证", 0.85);
  divider(s, 1.12);

  const boxY = 1.45, boxH = 1.35, boxW = 3.9;
  s.addShape(pres.ShapeType.roundRect, {
    x: M, y: boxY, w: boxW, h: boxH, fill: { color: C.paleBlue },
    line: { color: C.blue, width: 1 }, rectRadius: 0.06,
  });
  s.addText("训练奖励", {
    x: M, y: boxY + 0.1, w: boxW, h: 0.35, fontFace: F.face, fontSize: 16,
    color: C.blue, bold: true, align: "center",
  });
  s.addText("docking · QED · SA", {
    x: M, y: boxY + 0.5, w: boxW, h: 0.7, fontFace: F.face, fontSize: 15,
    color: C.navy, align: "center",
  });

  s.addShape(pres.ShapeType.roundRect, {
    x: 5.6, y: boxY, w: boxW, h: boxH, fill: { color: C.paleOrange },
    line: { color: C.orange, width: 1 }, rectRadius: 0.06,
  });
  s.addText("评价口径", {
    x: 5.6, y: boxY + 0.1, w: boxW, h: 0.35, fontFace: F.face, fontSize: 16,
    color: C.orange, bold: true, align: "center",
  });
  s.addText("UniDock + Vina · QED", {
    x: 5.6, y: boxY + 0.5, w: boxW, h: 0.7, fontFace: F.face, fontSize: 15,
    color: C.navy, align: "center",
  });

  s.addText("同一套计算过程", {
    x: 4.45, y: boxY + 0.35, w: 1.1, h: 0.6, fontFace: F.face, fontSize: 12,
    color: C.red, bold: true, align: "center", valign: "middle",
  });

  bullets(s, [
    { text: "指标的改善既可来自分子质量的真实提升，也可来自模型对该口径的针对性利用", options: {} },
    { text: "二者在数值上无法区分", options: { bold: true, color: C.red } },
    "→ 评价与优化必须解耦，这是第 3 章的出发点",
  ], M, 3.1, CW, 2.0);
  pageNo(s, 8, TOTAL);
  s.addNotes("把根因讲透：训练奖励用的是 docking/QED/SA，而评价用的也是 UniDock+Vina 和 QED，是同一套计算过程。同源意味着指标的改善既可能来自真实提升，也可能来自模型针对这个口径的利用——两者在数值上分不开。所以出路不是调奖励权重，而是把评价和优化解耦。");
}

// ==========================================================================
// Slide 9 — 第 3 章 换尺子
// ==========================================================================
{
  const s = newSlide();
  title(s, "改用独立于奖励的结构规则口径后，DiffGui 的高对接优势消失", 0.85);
  divider(s, 1.12);
  leftFigure(s, IMG("fig3_1_model_pathology.png"), 3300 / 1800, [
    { text: "R1–R4 命中率：DiffGui 73.23% vs DiffSBDD 50.12%", options: { bold: true } },
    "限定在合理分子子集后：−8.86 vs −8.92 kcal/mol",
    { text: "→ 全口径的对接优势归零", options: { bold: true, color: C.red } },
    "说明优势来自分子池中的病理结构，而非合理分子本身更好",
  ]);
  cite(s, "图 3.1 五模型结构规则命中率（500 步、3 次重复）；表 3.2 / 表 3.2b");
  pageNo(s, 9, TOTAL);
  s.addNotes("换尺子之后结论翻转。DiffGui 的 R1–R4 命中率 73%，远高于 DiffSBDD 的 50%；但把比较限定在各自的合理分子子集上，两者对接几乎一样（−8.86 vs −8.92）。也就是说，DiffGui 全口径下的对接优势，完全来自它生成了更多病理分子。这句话把“选错模型”升级成了“指标口径有结构性盲区”。");
}

// ==========================================================================
// Slide 10 — 新范式
// ==========================================================================
{
  const s = newSlide();
  title(s, "ESRPO 把“什么更好”从事后标量打分改为事前成对比较", 0.85);
  divider(s, 1.12);

  const y0 = 1.4, hh = 2.35, ww = 4.4;
  s.addShape(pres.ShapeType.roundRect, {
    x: M, y: y0, w: ww, h: hh, fill: { color: C.paleRed },
    line: { color: C.red, width: 1 }, rectRadius: 0.06,
  });
  s.addText("标量奖励（第 2 章）", {
    x: M + 0.15, y: y0 + 0.12, w: ww - 0.3, h: 0.4, fontFace: F.face,
    fontSize: 17, color: C.red, bold: true, align: "center",
  });
  bullets(s, [
    "奖励函数给每个分子打一个分",
    "模型可反向利用该函数",
    "→ 出现套利（reward hacking）",
  ], M + 0.25, y0 + 0.6, ww - 0.5, 1.6);

  s.addShape(pres.ShapeType.roundRect, {
    x: 5.1, y: y0, w: ww, h: hh, fill: { color: C.paleTeal },
    line: { color: C.teal, width: 1 }, rectRadius: 0.06,
  });
  s.addText("成对偏好（ESRPO，第 3 章）", {
    x: 5.25, y: y0 + 0.12, w: ww - 0.3, h: 0.4, fontFace: F.face,
    fontSize: 17, color: C.teal, bold: true, align: "center",
  });
  bullets(s, [
    "winner / loser 在训练前由规则判定并固定",
    "模型无法改变“谁是 winner”",
    "→ 机制上切断套利通道",
  ], 5.35, y0 + 0.6, ww - 0.5, 1.6);

  s.addText("奖励只决定“这一对梯度用多大力”，方向来自模型自身的去噪误差", {
    x: M, y: 3.95, w: CW, h: 0.45, fontFace: F.face, fontSize: 15,
    color: C.navy, align: "center", valign: "middle",
  });
  cite(s, "偏好对定义：R1–R4 命中 0 条为 good，≥1 条为 bad；R5 在配对中保持中性");
  pageNo(s, 10, TOTAL);
  s.addNotes("这页讲机制。左边是第2章的做法——奖励函数给标量分，模型可以反向利用它。右边是 ESRPO：winner 和 loser 在训练之前就由规则判定并固定，模型无论怎么改生成分布都改不了谁是 winner。所以奖励退化成“这一对梯度用多大力”，方向来自模型自己的去噪误差。这是结构性的反套利，不是调参。");
}

// ==========================================================================
// Slide 11 — 主结果
// ==========================================================================
{
  const s = newSlide();
  title(s, "ESRPO 将合理分子产率由 30.5% 提升至 55.6%，对接基本持平", 0.85);
  divider(s, 1.12);
  leftFigure(s, IMG("fig3_3_esrpo_tradeoff.png"), 1320 / 720, [
    { text: "合理分子产率 30.5% → 55.6%（1.82×）", options: { bold: true, color: C.teal } },
    "R1–R4 命中率 48.67% → 21.27%",
    { text: "合理分子对接中位 −8.92 → −8.46（回退 0.46）", options: {} },
    "停留点 step 4000 由预先确定的准则选出，非事后挑选",
  ]);
  cite(s, "表 3.3 / 图 3.3；停留点准则：合理分子 QED 不低于基线 85% 时取产率最高者");
  pageNo(s, 11, TOTAL);
  s.addNotes("主结果。合理分子产率从 30.5% 提到 55.6%，几乎翻倍；对接只回退 0.46 kcal/mol，基本持平。特别要说明停留点是按预先定好的准则选出来的，不是看结果挑的——这一点在答辩时几乎一定会被问。");
}

// ==========================================================================
// Slide 12 — 改善的分布（原生条形图）
// ==========================================================================
{
  const s = newSlide();
  title(s, "改善集中在三元环与环系维度：R2 与 R4c 几乎被清除", 0.85);
  divider(s, 1.12);

  s.addChart(pres.ChartType.bar, [
    { name: "Original", labels: ["R1", "R2", "R3", "R4a", "R4b", "R4c", "R5"], values: [32.45, 12.98, 4.72, 10.47, 17.70, 13.57, 14.01] },
    { name: "ESRPO step 4000", labels: ["R1", "R2", "R3", "R4a", "R4b", "R4c", "R5"], values: [15.04, 0.37, 2.08, 1.59, 4.03, 0.98, 15.16] },
  ], {
    x: M, y: 1.2, w: 5.8, h: 3.9, barDir: "col",
    chartColors: [C.gray, C.teal],
    chartArea: { fill: { color: C.panel } },
    catAxisLabelColor: C.gray, valAxisLabelColor: C.gray,
    valGridLine: { color: C.line, size: 0.5 }, catGridLine: { style: "none" },
    showLegend: true, legendPos: "b", legendColor: C.navy,
    showValue: true, dataLabelColor: C.navy, dataLabelFontSize: 9, dataLabelFormatCode: "0.0",
    valAxisTitle: "命中率（%）", showValAxisTitle: true,
    valAxisTitleColor: C.gray, valAxisTitleFontSize: 12,
    catAxisTitle: "规则", showCatAxisTitle: true,
    catAxisTitleColor: C.gray, catAxisTitleFontSize: 12,
  });

  bullets(s, [
    { text: "R2 12.98% → 0.37%，R4c 13.57% → 0.98%", options: { bold: true, color: C.teal } },
    "R4b 17.70% → 4.03%，R1 32.45% → 15.04%",
    { text: "R5 护栏 14.01% → 15.16%", options: { bold: true, color: C.orange } },
    "它未进入优化目标，故没有改善动力 —— 这反证了“判据覆盖维度决定改善方向”",
  ], 6.5, 1.3, 3.0, 3.6);
  cite(s, "表 3.3 ESRPO 优化前后逐条命中率（推荐停留点 step 4000）");
  pageNo(s, 12, TOTAL);
  s.addNotes("看改善落在哪里。R2 和 R4c 几乎清零，R4b 也降了四倍多。但请特别注意橙色的 R5——它是组成护栏，在偏好对里被匹配成中性、不参与优化，所以完全没有改善。这不是缺陷，而是一个天然对照组：它证明了模型的改善方向严格由判据覆盖的维度决定。");
}

// ==========================================================================
// Slide 13 — 单分子证据（第 3 章侧）
// ==========================================================================
{
  const s = newSlide();
  title(s, "与第 2 章相反：优化后对接最好的分子中，干净分子由 1/10 增至 5/10", 0.8);
  divider(s, 1.05);
  wideFigure(s, IMG("fig3_2_esrpo_top_docking.png"), 4600 / 1268, 1.15);
  bullets(s, [
    { text: "选样规则与第 7 页完全一致（对接分数最优的 10 个）", options: { bold: true } },
    "三元环 2/10 → 0/10；R1–R4 命中 8/10 → 3/10",
    "该子集重原子数 30.6 → 24.6（图中已标注）；全体口径下重原子数基本不变（21.7 → 20.8）",
  ], M, 3.78, CW, 1.4);
  pageNo(s, 13, TOTAL);
  s.addNotes("与第7页口径完全一致的一张图，方向完全相反。优化前对接最好的10个分子里只有1个干净，优化后有5个。这说明结构改善不是靠让出对接优势区换来的——即便在对接最优的子集里，结构也变干净了。这是整个工作里最有说服力的一张图。");
}

// ==========================================================================
// Slide 14 — 与外部参照的距离（原生条形图）
// ==========================================================================
{
  const s = newSlide();
  title(s, "优化后仍未达到真实类药空间水平，差距约 2 倍", 0.85);
  divider(s, 1.12);

  s.addChart(pres.ChartType.bar, [
    { name: "ChEMBL 参照", labels: ["R1–R4", "R1–R5"], values: [10.38, 15.28] },
    { name: "ESRPO step 4000", labels: ["R1–R4", "R1–R5"], values: [21.27, 32.03] },
  ], {
    x: M, y: 1.25, w: 5.2, h: 3.7, barDir: "col",
    chartColors: [C.blue, C.orange],
    chartArea: { fill: { color: C.panel } },
    catAxisLabelColor: C.gray, valAxisLabelColor: C.gray,
    valGridLine: { color: C.line, size: 0.5 }, catGridLine: { style: "none" },
    showLegend: true, legendPos: "b", legendColor: C.navy,
    showValue: true, dataLabelColor: C.navy, dataLabelFontSize: 11, dataLabelFormatCode: "0.0",
    valAxisTitle: "命中率（%）", showValAxisTitle: true,
    valAxisTitleColor: C.gray, valAxisTitleFontSize: 12,
  });

  bullets(s, [
    { text: "R1–R4：21.27% vs 10.38%（2.05×）", options: { bold: true } },
    { text: "R1–R5：32.03% vs 15.28%（2.10×）", options: { bold: true } },
    { text: "两个口径给出同一结论 → 差距不依赖于是否纳入组成护栏", options: { color: C.teal, bold: true } },
    "ChEMBL 参照 1 550 090 个类药分子，与生成分子同一规则实现",
  ], 5.95, 1.3, 3.55, 3.6);
  cite(s, "§3.3.2.3 与基线和外部参照的口径对比；ChEMBL 36，relaxed 分母口径一致");
  pageNo(s, 14, TOTAL);
  s.addNotes("最后把结果放到真实化学空间里定位。优化后 R1–R4 是 21.27%，参照水平是 10.38%——还差两倍。换 R1–R5 口径算同样是两倍，说明这个差距是稳的，不依赖于是否纳入组成护栏。这一页的价值在于诚实：改善是真的，但离真实类药空间还有距离。");
}

// ==========================================================================
// Slide 15 — 边界
// ==========================================================================
{
  const s = newSlide();
  title(s, "结论的三条边界：QED 下降、单一口袋、未达参照水平", 0.85);
  divider(s, 1.12);
  bullets(s, [
    { text: "目标外维度漂移：合理分子 QED 0.647 → 0.561 —— 优化目标未覆盖的维度会退化", options: {} },
    { text: "统计效力有限：单一口袋（8GJL/C1101）、单 seed、无方差估计", options: {} },
    { text: "未达参照水平：与 ChEMBL 相比仍有约 2 倍差距", options: {} },
    { text: "候选分子是否真的更优，最终仍需分子动力学模拟与实验活性验证", options: { bold: true } },
  ], M, 1.35, CW, 3.6);
  cite(s, "§3.3.2.4 放行标准核对；§3.4 本章小结");
  pageNo(s, 15, TOTAL);
  s.addNotes("主动把边界摆出来。QED 下降说明换口径只是把未被覆盖的维度暴露出来，这不是失败而是必然；单口袋单 seed 是统计效力问题；离 ChEMBL 参照还差两倍。最后一句最重要：候选分子是否真的更优，最终要靠 MD 和实验，这是计算工作的固有边界。");
}

// ==========================================================================
// Slide 16 — 结论（Q&A 时保持显示）
// ==========================================================================
{
  const s = newSlide();
  title(s, "结论", 0.75);
  divider(s, 1.0);

  const items = [
    { t: "标量代理奖励可被模型针对性利用", d: "对接均值改善约 15% 的同时，三元环率由 12.3% 升至 44.3%，且重原子数基本不变 —— 改善的来源包含结构病理富集。" },
    { t: "优化目标改为独立于奖励的显式结构规则后，可在对接基本持平下实现结构改善", d: "合理分子产率 30.5% → 55.6%（1.82×），R1–R4 命中率 48.67% → 21.27%，对接中位仅回退 0.46 kcal/mol。" },
    { t: "判据覆盖的维度决定能改善什么", d: "未进入优化目标的 R5 护栏无改善动力；优化后与真实类药空间仍有约 2 倍差距。" },
  ];
  items.forEach((it, i) => {
    const y = 1.15 + i * 1.25;
    s.addShape(pres.ShapeType.roundRect, {
      x: M, y, w: 0.42, h: 0.42, fill: { color: C.teal }, line: { color: C.teal },
      rectRadius: 0.21,
    });
    s.addText(String(i + 1), {
      x: M, y, w: 0.42, h: 0.42, fontFace: F.face, fontSize: 16,
      color: "FFFFFF", bold: true, align: "center", valign: "middle",
    });
    s.addText(it.t, {
      x: 1.1, y: y - 0.02, w: 8.3, h: 0.42, fontFace: F.face, fontSize: 17,
      color: C.navy, bold: true, valign: "middle",
    });
    s.addText(it.d, {
      x: 1.1, y: y + 0.4, w: 8.3, h: 0.72, fontFace: F.face, fontSize: 14,
      color: C.gray, valign: "top", lineSpacingMultiple: 1.15,
    });
  });

  s.addText("谢东宸 · [培养院系] · [邮箱]", {
    x: M, y: 5.02, w: CW, h: 0.32, fontFace: F.face, fontSize: 13,
    color: C.gray, align: "center",
  });
  pageNo(s, 16, TOTAL);
  s.addNotes("结论页在问答环节保持显示。三条：一是奖励可被套利，这是机制性发现；二是把优化目标换成独立口径后能实现“对接不掉、结构变好”；三是判据覆盖的维度决定改善方向——这条是方法学层面的，也是最有推广价值的一条。欢迎提问。");
}

// ==========================================================================
// Slide 17 — 参考文献
// ==========================================================================
{
  const s = newSlide();
  title(s, "参考文献", 0.6);
  divider(s, 0.95);
  const refs = [
    "[1] Zdrazil B, Felix E, Hunter F, et al. The ChEMBL database in 2023[J]. Nucleic Acids Research, 2024, 52(D1): D1180-D1192.",
    "[2] Schneuing A, Harris C, Du Y, et al. Structure-based drug design with equivariant diffusion models[J]. Nature Computational Science, 2024, 4(12): 899-909.",
    "[3] Hu Q, Sun C, He H, et al. Target-aware 3D molecular generation based on guided equivariant diffusion[J]. Nature Communications, 2025, 16(1): 7928.",
    "[4] Schulman J, Wolski F, Dhariwal P, et al. Proximal policy optimization algorithms[J/OL]. arXiv:1707.06347, 2017.",
    "[5] Black K, Janner M, Du Y, et al. Training diffusion models with reinforcement learning[C]//ICLR. 2024.",
    "[6] Rafailov R, Sharma A, Mitchell E, et al. Direct preference optimization: your language model is secretly a reward model[C]//NeurIPS. 2023.",
    "[7] Yu Y, Cai C, Wang J, et al. Uni-Dock: GPU-accelerated docking enables ultralarge virtual screening[J]. JCTC, 2023, 19(11): 3336-3345.",
    "[8] Trott O, Olson A J. AutoDock Vina[J]. Journal of Computational Chemistry, 2010, 31(2): 455-461.",
    "[9] Bickerton G R, Paolini G V, Besnard J, et al. Quantifying the chemical beauty of drugs[J]. Nature Chemistry, 2012, 4(2): 90-98.",
    "[10] Ertl P, Schuffenhauer A. Estimation of synthetic accessibility score[J]. Journal of Cheminformatics, 2009, 1(1): 8.",
  ];
  s.addText(refs.map((r) => ({ text: r, options: { breakLine: true } })), {
    x: M, y: 1.02, w: CW, h: 4.25, fontFace: F.face, fontSize: 10,
    color: C.gray, paraSpaceAfter: 3, valign: "top", lineSpacingMultiple: 1.0,
  });
  pageNo(s, 17, TOTAL);
  s.addNotes("完整参考文献。汇报中引用的每一项来源都在这里。");
}

// ==========================================================================
// 附录（预判 Q&A）
// ==========================================================================
const APX = [
  {
    t: "A1 · 为什么不用 SA score / Brenk 作为判据？",
    b: [
      "Brenk 过滤器：五模型命中率持平（26%–35%），无法区分病理",
      "SA score：能区分但动态范围小（ESRPO 全程仅动 0.23），不适合作主要偏好轴",
      "结构规则口径：仅由拓扑与元素组成判定，不依赖对接或活性计算，与优化目标天然独立",
    ],
  },
  {
    t: "A2 · R1–R5 的阈值（≥7、>3、0.85）是怎么定的？",
    b: [
      "判据用法是相对比较 —— 比较同条件、不同模型或不同训练阶段之间的差异，而非以绝对阈值判定单个分子",
      "参照系为 ChEMBL 36 全量 1 550 090 个类药分子，使用完全相同的规则实现",
      "逐条富集倍数：R4a 31.7×、R4c 20.9×、R4b 16.0×、R1 7.9×、R2 3.9×、R3 1.9×、R5 2.2×",
    ],
  },
  {
    t: "A3 · 与 ChEMBL 参照的口径是否可比？",
    b: [
      "参照集经 chembl_structure_pipeline 标准化，全部 SMILES 均为单片段（多片段 0、解析失败 0）",
      "因此 relaxed 最大片段步骤在参照集上为空操作，relaxed 分子数 = 全量 1 550 090",
      "生成侧与参照侧使用同一分母定义（relaxed 分子数）",
    ],
  },
  {
    t: "A4 · 分子尺寸（HAC）的混杂如何控制？",
    b: [
      "环系病理与分子尺寸正相关：corr(HAC, R1–R4) = +0.64 ~ +0.70",
      "按重原子数分层后，三元环率的上升在每一档内均成立（0→22%、0→31%、7→50%、12→43%）",
      "但 R1–R4 聚合量在各档内基本不变 → 该聚合量的变化主要由尺寸分布驱动",
    ],
  },
  {
    t: "A5 · 偏好对的匹配质量如何？",
    b: [
      "按 (元素组成, 重原子数) 严格匹配：两侧 R1–R4 平均命中条数 0.000 vs 1.978",
      "R5 命中率被拉平（均为 3.29%），对接分均值差 < 0.003 → 两维度差异已被抵消",
      "因此偏好对中唯一系统性差异是 R1–R4 结构病理",
    ],
  },
  {
    t: "A6 · 为什么选 step 4000 而非产率更高的检查点？",
    b: [
      "准则预先确定：合理分子 QED 不低于基线 85%（0.647 × 0.85 = 0.550）时，取产率最高者",
      "继续训练至 step 16000 产率可达约 69%，但合理分子对接中位一路回退至约 −8.0",
      "step 4000 把对接让步限制在 0.46 kcal/mol 以内",
    ],
  },
];
APX.forEach((a, i) => {
  const s = newSlide();
  title(s, a.t, 0.7);
  divider(s, 0.98);
  bullets(s, a.b, M, 1.15, CW, 3.9);
  pageNo(s, 17 + i + 1, TOTAL);
  s.addNotes("附录页，用于预判的问答。");
});

// --------------------------------------------------------------------------
pres.writeFile({ fileName: path.join(__dirname, "RE-CmeB_第2-3章汇报.pptx") })
  .then((f) => console.log("已生成:", f))
  .catch((e) => { console.error("生成失败:", e); process.exit(1); });
