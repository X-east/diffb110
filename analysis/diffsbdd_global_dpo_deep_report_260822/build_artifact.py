"""Build the canonical portable artifact for the DiffSBDD–DPO feasibility report."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent
XV_DIR = PROJECT_ROOT / "analysis/diffsbdd_global_dpo_xv_feasibility_260822"
ADAPT_DIR = PROJECT_ROOT / "analysis/diffsbdd_decompdpo_adaptation_audit_260822"


def _read_csv(path: Path) -> list[dict[str, object]]:
    """Read a CSV and coerce numeric fields for native report charts."""
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, object] = {}
            for key, value in raw.items():
                if value is None:
                    row[key] = None
                    continue
                try:
                    row[key] = float(value)
                except ValueError:
                    row[key] = value
            rows.append(row)
    return rows


def build_artifact() -> dict[str, object]:
    """Return a source-backed technical report artifact."""
    xv_rows = _read_csv(XV_DIR / "native_xv_denoising_summary.csv")
    descriptor_rows = _read_csv(XV_DIR / "descriptor_shortcut_diagnostics.csv")
    descriptor_rows = [
        row for row in descriptor_rows if row["descriptor"] != "umap_nearest_chembl_distance"
    ][:9]
    pairing_rows = _read_csv(ADAPT_DIR / "pairing_coverage.csv")

    x_fraction_rows = [
        {
            "timestep": int(row["timestep"]),
            "label": "好分子" if row["label"] == "good" else "困难坏分子",
            "x_error_fraction": row["x_raw_error_fraction"],
        }
        for row in xv_rows
    ]
    descriptor_labels = {
        "sa_raw": "SA 难度",
        "fused_ring_count": "稠合环数",
        "total_ring_count": "总环数",
        "carbon_fraction": "碳原子比例",
        "hetero_atom_count": "杂原子数",
        "complex_ring_count": "复杂环数",
        "tpsa": "TPSA",
        "three_member_ring_count": "三元环数",
        "logp": "logP",
    }
    descriptor_auc_rows = [
        {
            "descriptor": descriptor_labels.get(str(row["descriptor"]), row["descriptor"]),
            "auc": row["univariate_auc"],
        }
        for row in descriptor_rows
    ]
    pairing_chart_rows = [
        {
            "split": "验证" if row["split"] == "validation" else str(row["split"]),
            "side": "好分子覆盖率",
            "coverage": row["good_coverage"],
        }
        for row in pairing_rows
    ] + [
        {
            "split": "验证" if row["split"] == "validation" else str(row["split"]),
            "side": "坏分子覆盖率",
            "coverage": row["hard_bad_coverage"],
        }
        for row in pairing_rows
    ]

    adaptation_rows = [
        {"module": "终局偏好对", "decompdpo": "全分子或局部分解 pair", "diffsbdd": "同 pocket、精确 N 的终局 x/v pair", "decision": "保留思想，重写数据适配"},
        {"module": "训练样本", "decompdpo": "随机噪声层的终局分子", "diffsbdd": "终局 x/v 重新前向加噪", "decision": "可直接采用；无需 1001 步轨迹"},
        {"module": "坐标扩散", "decompdpo": "连续坐标误差", "diffsbdd": "连续坐标 epsilon MSE", "decision": "可适配，但保持零质心投影"},
        {"module": "原子类型", "decompdpo": "categorical posterior KL", "diffsbdd": "one-hot 连续高斯 epsilon MSE", "decision": "不能复制 v=100"},
        {"module": "键", "decompdpo": "显式 bond categorical diffusion", "diffsbdd": "无 bond 分支；终局重建", "decision": "不能移植 bond loss"},
        {"module": "分解结构", "decompdpo": "decomp_mask、arms/scaffold", "diffsbdd": "无对应语义", "decision": "Global 目标不需要，删除"},
        {"module": "reference-relative DPO", "decompdpo": "current/reference loss 差", "diffsbdd": "x/v 分量 reference-relative margin", "decision": "核心可保留"},
        {"module": "时间范围", "decompdpo": "代码中 1000 步假设", "diffsbdd": "checkpoint 原生 T=500，t=0 特殊", "decision": "首版仅 t=1…500"},
    ]
    risk_rows = [
        {"priority": "P0", "risk": "标签优化错位", "mechanism": "UMAP 低维距离被当作真实 ChEMBL 相似度", "observable": "直接距离 AUC=0.962，但由阈值规则构造", "gate": "原空间 ECFP4 kNN/独立模型复核"},
        {"priority": "P0", "risk": "winner 也被破坏", "mechanism": "DPO margin 可通过增大 rejected loss 获益", "observable": "winner current-reference loss 持续上升", "gate": "winner-preservation + fresh generation"},
        {"priority": "P0", "risk": "x/v 更新失衡", "mechanism": "联合数值尺度与共享参数梯度可能不匹配", "observable": "x 原始误差占 81.6%–99.2%；梯度尚未测", "gate": "分量梯度范数、逐层范数与夹角"},
        {"priority": "P0", "risk": "原子数问题不可学习", "mechanism": "N 在 dynamics 前由冻结 size distribution 采样", "observable": "质量提升只发生在特定 N，整体不动", "gate": "按 N 分层 fresh evaluation"},
        {"priority": "P1", "risk": "拓扑监督不连续", "mechanism": "bond 由 x/v 经 OpenBabel/RDKit 后处理推断", "observable": "loss 改善但重建失败、环病理不改善", "gate": "重建率、PoseBusters、环/键异常"},
        {"priority": "P1", "risk": "描述符捷径与模式坍缩", "mechanism": "模型只减少碳或复杂环以满足标签", "observable": "相似度升高但多样性、docking 或元素分布恶化", "gate": "多目标 guardrails 与新颖性/多样性"},
        {"priority": "P1", "risk": "t=0 语义错误", "mechanism": "DiffSBDD t=0 不是普通 epsilon MSE", "observable": "末端离散原子类型异常", "gate": "首版排除 t=0"},
        {"priority": "P1", "risk": "噪声未配对", "mechanism": "current/reference 或 winner/loser 使用不同随机噪声", "observable": "margin 高方差且复现实验不稳定", "gate": "固定共享 t、epsilon、mask、pocket"},
    ]
    gate_rows = [
        {"stage": "G0 数据", "required": "冻结 scaffold split；同 pocket；精确 N；原空间标签复核", "pass": "pair 覆盖≥90%，无 split 泄露，标签方向一致", "failure_action": "重建 pair/label，禁止训练"},
        {"stage": "G1 梯度", "required": "g_x、g_v 全模型/逐层范数和 cosine", "pass": "无单分量压倒性失控；权重有实测依据", "failure_action": "改分量归一化或权重"},
        {"stage": "G2 小规模优化", "required": "原生联合、分量均衡、梯度标定三组", "pass": "held-out pair 上升且 winner 不退化", "failure_action": "降 beta/LR；考虑 SDPO safeguard"},
        {"stage": "G3 生成", "required": "每检查点固定尝试数 fresh generation", "pass": "valid/PB、原空间相似度、环病理、多样性均过 guardrail", "failure_action": "拒绝该 checkpoint"},
        {"stage": "G4 对照", "required": "Original、filtered BC、AWR、DPO 匹配预算", "pass": "DPO 改善超过重复间方差且无 docking 损失", "failure_action": "不升级为正式路线"},
    ]

    generated_at = datetime.now(timezone.utc).isoformat()
    sources = [
        {"id": "diffsbdd_code", "label": "DiffB110 本地 DiffSBDD 实现", "path": "DiffSBDD/equivariant_diffusion/conditional_model.py", "query": {"engine": "sqlite3", "sql": "SELECT * FROM code_findings", "description": "选择经本地代码审计确认的 DiffSBDD 模型语义。"}},
        {"id": "diffsbdd_lightning", "label": "DiffSBDD 原生损失实现", "path": "DiffSBDD/lightning_modules.py", "query": {"engine": "sqlite3", "sql": "SELECT * FROM loss_findings", "description": "选择经本地代码审计确认的损失语义。"}},
        {"id": "diffsbdd_size", "label": "DiffSBDD 节点数条件分布", "path": "DiffSBDD/equivariant_diffusion/en_diffusion.py", "query": {"engine": "sqlite3", "sql": "SELECT * FROM size_findings", "description": "选择经本地代码审计确认的 size-prior 语义。"}},
        {"id": "decompdpo_code", "label": "本地 DecompDPO 参考代码（commit cb109f7）", "path": "reference project/代码/DecompDPO/train_decompdpo.py", "query": {"engine": "sqlite3", "sql": "SELECT * FROM adaptation", "description": "读取逐模块代码适配矩阵。"}},
        {"id": "xv_audit", "label": "DiffSBDD x/v 数值审计", "path": "analysis/diffsbdd_global_dpo_xv_feasibility_260822/native_xv_denoising_summary.csv", "query": {"engine": "sqlite3", "sql": "SELECT timestep, label, x_raw_error_fraction FROM native_xv_denoising_summary", "description": "读取六个噪声层的 x/v 原始误差尺度审计。"}},
        {"id": "descriptor_audit", "label": "好坏池描述符捷径审计", "path": "analysis/diffsbdd_global_dpo_xv_feasibility_260822/descriptor_shortcut_diagnostics.csv", "query": {"engine": "sqlite3", "sql": "SELECT descriptor, univariate_auc FROM descriptor_shortcut_diagnostics WHERE descriptor <> 'umap_nearest_chembl_distance' ORDER BY univariate_auc DESC LIMIT 9", "description": "排除标签定义距离后选择最强单描述符 AUC。"}},
        {"id": "pairing_audit", "label": "精确原子数配对覆盖率审计", "path": "analysis/diffsbdd_decompdpo_adaptation_audit_260822/pairing_coverage.csv", "query": {"engine": "sqlite3", "sql": "SELECT split, exact_heavy_atom_pairs_without_reuse, good_coverage, hard_bad_coverage FROM pairing_coverage", "description": "读取精确 heavy-atom count 配对数量和两侧覆盖率。"}},
        {"id": "risk_register", "label": "基于代码与数据证据构建的风险登记", "path": "analysis/diffsbdd_global_dpo_deep_report_260822/build_artifact.py", "query": {"engine": "sqlite3", "sql": "SELECT * FROM risks ORDER BY priority", "description": "选择报告中定义的机制性风险与验收门。"}},
        {"id": "gate_register", "label": "DiffSBDD Global DPO Go/No-Go 设计", "path": "analysis/diffsbdd_global_dpo_deep_report_260822/build_artifact.py", "query": {"engine": "sqlite3", "sql": "SELECT * FROM gates", "description": "选择分阶段实验验收门。"}},
        {"id": "diffusion_dpo_paper", "label": "Diffusion-DPO（Wallace et al.）", "url": "https://arxiv.org/abs/2311.12908"},
        {"id": "decompdpo_paper", "label": "DecompDPO", "url": "https://arxiv.org/abs/2407.13981"},
        {"id": "sdpo_paper", "label": "Diffusion-SDPO", "url": "https://arxiv.org/abs/2511.03317"},
    ]

    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "DiffSBDD × Diffusion-DPO：可行性、适配边界与失败机制",
            "description": "基于 DiffSBDD、DecompDPO 本地代码与现有好坏分子池审计的技术决策报告。",
            "generatedAt": generated_at,
            "cards": [
                {"id": "train_pairs", "description": "训练集精确重原子数、无复用可配对数量。", "dataset": "headline", "sourceId": "pairing_audit", "metrics": [{"label": "可用训练 pair", "field": "train_pairs", "format": "number"}]},
                {"id": "coverage", "description": "训练集 good 侧被精确 N 配对覆盖的比例。", "dataset": "headline", "sourceId": "pairing_audit", "metrics": [{"label": "good 配对覆盖", "field": "good_coverage", "format": "percent"}]},
                {"id": "x_fraction", "description": "六个噪声层中坐标项占联合原始平方误差的最高值；不是梯度占比。", "dataset": "headline", "sourceId": "xv_audit", "metrics": [{"label": "x loss 最高占比", "field": "x_fraction_max", "format": "percent"}]},
                {"id": "shortcut_auc", "description": "排除标签定义使用的 UMAP 距离后，最强单描述符 AUC。", "dataset": "headline", "sourceId": "descriptor_audit", "metrics": [{"label": "最大非标签 AUC", "field": "shortcut_auc", "format": "number"}]},
            ],
            "charts": [
                {"id": "x_fraction_by_t", "title": "联合去噪误差在数值上主要来自坐标项", "subtitle": "128 good + 128 hard-bad；六个 checkpoint-native 噪声层。该图只说明 loss 尺度，不说明参数梯度主导。", "type": "line", "dataset": "x_fraction", "sourceId": "xv_audit", "valueFormat": "percent", "encodings": {"x": {"field": "timestep", "type": "quantitative", "label": "扩散 timestep"}, "y": {"field": "x_error_fraction", "type": "quantitative", "label": "坐标原始误差占比", "format": "percent"}, "color": {"field": "label", "type": "nominal", "label": "终局标签"}}},
                {"id": "descriptor_auc", "title": "标签含有可学习的结构捷径，但单一描述符并不能完全分开", "subtitle": "已排除构造标签使用的 UMAP 最近距离；AUC 越高表示单变量越容易区分两池。", "type": "bar", "dataset": "descriptor_auc", "sourceId": "descriptor_audit", "encodings": {"x": {"field": "descriptor", "type": "nominal", "label": "描述符"}, "y": {"field": "auc", "type": "quantitative", "label": "单变量 AUC"}}},
                {"id": "pairing_coverage", "title": "DPO 配对在精确重原子数约束下仍有充足覆盖", "subtitle": "同一固定 pocket 条件；validation/test pair 冻结，训练 pair 可动态重配。", "type": "bar", "dataset": "pairing_coverage", "sourceId": "pairing_audit", "valueFormat": "percent", "encodings": {"x": {"field": "split", "type": "nominal", "label": "数据划分"}, "y": {"field": "coverage", "type": "quantitative", "label": "覆盖率", "format": "percent"}, "color": {"field": "side", "type": "nominal", "label": "pair 侧"}}},
            ],
            "tables": [
                {"id": "adaptation", "title": "DecompDPO 到 DiffSBDD 的逐模块适配边界", "subtitle": "“保留”指保留数学思想；所有模型接口仍必须按 DiffSBDD 原生语义实现。", "dataset": "adaptation", "sourceId": "decompdpo_code", "columns": [{"field": "module", "label": "模块", "type": "text"}, {"field": "decompdpo", "label": "DecompDPO", "type": "text"}, {"field": "diffsbdd", "label": "DiffSBDD", "type": "text"}, {"field": "decision", "label": "结论", "type": "text"}]},
                {"id": "risks", "title": "机制性风险登记表", "subtitle": "这些风险均可从当前代码路径或已测数据导出，不是泛化式风险罗列。", "dataset": "risks", "sourceId": "risk_register", "columns": [{"field": "priority", "label": "优先级", "type": "text"}, {"field": "risk", "label": "风险", "type": "text"}, {"field": "mechanism", "label": "发生机制", "type": "text"}, {"field": "observable", "label": "可观察证据", "type": "text"}, {"field": "gate", "label": "验收/防护", "type": "text"}]},
                {"id": "gates", "title": "从可行性到正式实验的 Go/No-Go 门", "subtitle": "任一 P0 门失败，都不应进入大规模 DPO 训练。", "dataset": "gates", "sourceId": "gate_register", "columns": [{"field": "stage", "label": "阶段", "type": "text"}, {"field": "required", "label": "必须完成", "type": "text"}, {"field": "pass", "label": "通过条件", "type": "text"}, {"field": "failure_action", "label": "失败处理", "type": "text"}]},
            ],
            "sources": sources,
            "blocks": [
                {"id": "title_note", "type": "markdown", "body": "**技术结论：Conditional GO。** 可以做一个 DiffSBDD-native Global Diffusion-DPO pilot；当前证据不支持直接复制 DecompDPO，也不支持立即启动全量正式训练。这里的 DPO 对象是终局分子经随机前向加噪得到的 `(z_t, t, pocket)`，不是保存下来的 1001 步反向轨迹。"},
                {"id": "headline", "type": "metric-strip", "cardIds": ["train_pairs", "coverage", "x_fraction", "shortcut_auc"]},
                {"id": "summary", "type": "markdown", "body": "## 技术摘要\n\n可行性来自三点：第一，DiffSBDD 原生就把坐标 `x∈R³` 与 10 维原子 one-hot `v` 拼成 13 维连续状态并联合做 Gaussian diffusion；第二，Diffusion-DPO 只需要 preferred/rejected 的**终局样本**、随机噪声层和冻结 reference，不要求生成时保存的完整反向轨迹；第三，现有 extreme pool 在同 pocket、精确 heavy-atom count 下可形成 17,289 个训练 pair，数据规模足够做 pilot。\n\n真正困难不是“中间状态分类器能不能分好坏”，而是偏好如何通过 DiffSBDD 的去噪参数传递到终局拓扑。DiffSBDD 不显式生成 bond，原子数 `N` 也由冻结的条件分布在扩散前采样，因此 DPO 只能改变固定 `N` 下的 `x/v` 条件分布。它可能改善元素组成和可重建几何，也可能只学到碳比例、环数等捷径；它无法自动修正错误的 size prior。\n\n推荐名称是 **DiffSBDD-native Global Diffusion-DPO**。DecompDPO 提供 reference-relative preference loss、终局重新加噪和 global objective 的实现依据；它的 decomp mask、arms/scaffold、categorical atom/bond KL、`v=100` 和 1000-step 常数不属于可移植部分。"},
                {"id": "scope", "type": "markdown", "body": "## 范围、数据与判定对象\n\n本报告聚焦固定 8GJL/C1101 pocket 下的 DiffSBDD checkpoint-native `T=500` 模型。候选偏好池共 50,155 个极端终局分子：good 25,211、hard-bad 24,944；中间 17,539 个样本排除。good 由 rank40 preferred、较小 UMAP-nearest-ChEMBL distance 且无结构 hard-bad 构成；hard-bad 来自 nonpreferred 且具有结构病理或较大 UMAP 距离。\n\n因此当前标签更准确的名称是 **UMAP-near normality + structural-hard-bad preference**，不是已经验证的“真实 ChEMBL 似然”。UMAP 距离是低维投影距离；它对标签的高 AUC 是定义性结果。正式实验必须至少用原始 ECFP4 kNN Tanimoto/距离或独立分子模型复核方向，并将 UMAP 保留为分析视图。"},
                {"id": "architecture", "type": "markdown", "body": "## 模型边界：DPO 与 DiffSBDD 是目标函数和生成模型的关系\n\nDPO 不是替换 geometric model 的分类器。DiffSBDD 的 equivariant dynamics 仍是被优化的生成模型；DPO 只规定如何用好/坏终局 pair 更新它。训练时对 winner 和 loser 各自采样噪声层，构造与原模型一致的 `z_t=α_t·xh₀+σ_t·ε`，current 与 frozen reference 比较相对去噪损失。换言之：**DiffSBDD 决定“怎么表示和生成”，DPO 决定“参数朝哪种终局分布移动”。**\n\n这解释了为什么不需要先训练一个能分类所有中间状态的 geometric critic。对同一终局分子随机加噪，监督直接落在原 denoiser 可处理的状态分布上。完整历史轨迹不仅不是必要输入，还会把旧 sampler 的强相关 1001 个状态重复计权，增加存储和分布偏差。"},
                {"id": "adaptation_table", "type": "table", "tableId": "adaptation"},
                {"id": "objective", "type": "markdown", "body": "## 建议的 DiffSBDD-native 目标\n\n对同 pocket、相同原子数 `N` 的 winner/loser，采样 `t∈{1,…,500}`。current/reference 必须共享同一份 `z_t`、`ε`、`t`、mask 和 pocket 条件。分别计算：\n\n`Lx = Σ(εx−ε̂x)²/(3N)`，`Lv = Σ(εv−ε̂v)²/(10N)`。\n\n令 `δx(w)=Lxθ(w)−Lxref(w)`，其余同理，则：\n\n`margin = −βt[λx(δx(w)−δx(l)) + λv(δv(w)−δv(l))]`\n\n`LDPO = −log σ(margin)`。\n\n首版应排除 `t=0`。DiffSBDD 在 `t=0` 使用坐标 likelihood 与离散 atom-type 积分概率，而非普通 epsilon MSE；直接当作一般噪声层会改变原模型的概率语义。reference 必须 `eval()`、冻结参数并只作前向；current/reference 的归一化、centering 和节点 mask 必须完全相同。"},
                {"id": "x_chart", "type": "chart", "chartId": "x_fraction_by_t"},
                {"id": "x_interpretation", "type": "markdown", "sourceId": "xv_audit", "body": "六层实测中，坐标项占原始联合平方误差 81.6%–99.2%。这足以否定“机械照搬 DecompDPO 的 `pos:v:bond=1:100:100`”，但还不足以断言坐标参数梯度占主导：loss 数值和共享网络参数上的梯度范数不是同一个量。正式选 `λx/λv` 前必须在相同 pair、timestep 和噪声下独立反传 `gx=∇θ(Lxw−Lxl)`、`gv=∇θ(Lvw−Lvl)`，记录全模型/逐层范数、加权比和 `cos(gx,gv)`。负 cosine 意味着目标冲突，不只是尺度问题。"},
                {"id": "label_chart", "type": "chart", "chartId": "descriptor_auc"},
                {"id": "label_interpretation", "type": "markdown", "sourceId": "descriptor_audit", "body": "排除标签直接使用的 UMAP 距离后，最强单变量只是 SA AUC 0.633，随后是稠合环 0.631、总环数 0.617、碳比例 0.614。这说明两池确实带有希望抑制的复杂环/高碳模式，但不存在一个完全决定标签的简单描述符。好处是 DPO 有机会学习组合信号；风险是模型仍可能选择最便宜的组成捷径，而不是学习更接近 ChEMBL 的完整拓扑分布。"},
                {"id": "pair_chart", "type": "chart", "chartId": "pairing_coverage"},
                {"id": "pairing_method", "type": "markdown", "sourceId": "pairing_audit", "body": "DPO 在数学上需要 pair，但不要求人工逐对标注，也不要求预枚举所有组合。现有三划分精确 heavy-atom count 配对覆盖均超过 92.8%。训练时可在同标签池内按 `N` 动态重配以提高组合多样性；validation/test pair 必须冻结，并保留 scaffold split、score margin、hard-bad 类型与重复策略，避免评估漂移。"},
                {"id": "pathology", "type": "markdown", "body": "## 最危险的优化病理：pair accuracy 上升但生成更差\n\n标准 DPO 只要求 winner 相对优于 loser。设 winner/loser 的 current 梯度为 `gw, gl`，近似更新为 `Δθ=−η(gw−λgl)`；winner 一阶变化为 `ΔLw≈−η(||gw||²−λ·gw·gl)`。当两侧梯度高度同向且 rejected 推力过强，margin 可以变大，同时 winner reconstruction 也变坏。\n\n这一问题不是 DecompDPO 提出的。后续独立工作 [Diffusion-SDPO](https://arxiv.org/abs/2511.03317) 在文生图 diffusion preference optimization 中系统讨论了类似退化，并用安全系数限制 rejected 梯度。它不能被当作分子体系已验证方案，但给出了必要诊断：分开记录 winner/current-reference、loser/current-reference、pair margin、两侧梯度内积，以及 fresh sample 质量。首个 pilot 仍应先复现标准 objective；只有 winner gate 失败时，再把 SDPO-style safeguard 作为有文献依据的消融。"},
                {"id": "risks_table", "type": "table", "tableId": "risks"},
                {"id": "generation_limits", "type": "markdown", "body": "## 训练 loss 与真正分子质量之间的断层\n\n1. **原子数不在 actor 内。** `N~p_original(N|pocket_size)` 先采样，DPO 只学习 `pθ(x,v|N,pocket)`。必须按 N 分层评价；若异常集中在 size prior，需单独做受控 size-prior 实验。\n2. **bond 不在扩散状态内。** 终局 `x/v` 经 OpenBabel/RDKit 推断键。轻微坐标变化可能跨过成键阈值并离散改变拓扑；训练 loss 下降不能替代 reconstruction、validity、PoseBusters、键长和复杂环检查。\n3. **几何可能被牺牲。** ChEMBL normality 偏好主要来自 2D fingerprint/描述符时，过大的 `λv` 可能推动元素组成而破坏 pocket geometry；过大的 `λx` 又可能只改变可重建构象。必须同时报告 docking/interaction guardrail。\n4. **相似度目标可导致复制与模式坍缩。** 应监测 unique scaffold、pairwise diversity、nearest-neighbor 重复率和训练集近邻，而不能只看平均 ChEMBL 距离。\n5. **单 pocket 结论不可外推。** 当前 evidence 只支持 RE-CmeB 的 8GJL/C1101 条件。跨 pocket 泛化需另建配对和拆分。"},
                {"id": "gates_table", "type": "table", "tableId": "gates"},
                {"id": "experiment", "type": "markdown", "body": "## 最小但能回答问题的实验设计\n\n**阶段 A：无训练梯度审计。** 在 512–2048 个冻结 train pairs 上比较三种 x/v 定义：原生 13 维联合 loss、`mean_x+mean_v` 分量均衡、根据梯度 RMS 标定的权重。跨 low/mid/high timestep 记录范数、cosine 与 batch 方差。\n\n**阶段 B：短 pilot。** 同一初始化、同一 pair stream、同一更新数，运行三组 DPO；建议低学习率起步，beta 只做小网格，不从 DecompDPO 的数值照抄。每个 epoch 保存 checkpoint 和本地 component metrics。关键训练指标不是单独 accuracy，而是 held-out margin、winner delta、loser delta、reference KL proxy、x/v 分量梯度。\n\n**阶段 C：fresh generation。** 每个候选 checkpoint 用固定尝试数和固定 seed 生成，不回放训练终局。至少测 RDKit validity、PB-valid、重建失败、原空间 ChEMBL kNN、碳比例、杂原子数、元素多样性、logP、TPSA、总环/芳环/三元环/复杂环、unique scaffold、内部多样性、docking。\n\n**阶段 D：匹配预算对照。** Original、BC-all、filtered BC、AWR、native-joint DPO、component-balanced DPO 使用相同数据拆分、更新预算和生成尝试数。只有 DPO 的 fresh-generation 改善超过重复间方差，且 docking/geometry guardrail 不退化，才升级为正式路线。"},
                {"id": "decision", "type": "markdown", "body": "## 最终判断\n\n**可行，但不是“拿 DecompDPO 代码改几个输入字段”式可行。** 数学层面的 Global Diffusion-DPO 与 DiffSBDD 匹配：终局分子 pair、随机前向加噪、冻结 reference、current/reference 相对 loss 都成立；完整反向轨迹对这条路线没有必要。模型层面必须原生重写：连续 Gaussian `v`、无 bond diffusion、独立 size prior、`T=500`、特殊 `t=0`、零质心几何约束都改变了实现和风险。\n\n当前最合理决策是先通过 G0/G1，再做受控小规模 pilot。若梯度审计显示 x/v 可稳定平衡，held-out winner 不退化，且 fresh generation 同时改善原空间 ChEMBL normality 与结构/几何 guardrails，则路线成立；若只提升 pair accuracy 或 UMAP 距离，则应判定为标签/优化捷径，而不是分子生成质量提升。"},
                {"id": "limitations", "type": "markdown", "body": "## 限制与未决问题\n\n本报告未实际训练 DPO，因此对学习率、beta、梯度裁剪和最优 x/v 权重不做未经验证的定值推荐。现有 256 分子的六层 probe 足以发现数值尺度问题，但不能替代训练期梯度统计。当前 extreme 标签依赖 UMAP 距离，原空间相似度复核仍是启动门。Diffusion-SDPO 的结果来自文生图体系，只能作为病理诊断与后续 safeguard 依据。\n\n进一步需要回答：偏好信号主要经 x 还是 v 传递？改善是否局限于特定 N？复杂环下降是否伴随 docking/interaction 损失？不同 bad 类型（高碳、复杂环、远离 ChEMBL）是否需要分层 pair，而不是混为一个 rejected pool？这些问题都应由分量梯度和 fresh generation 消融回答。"},
                {"id": "references", "type": "markdown", "body": "## 主要参考\n\n- [Diffusion-DPO](https://arxiv.org/abs/2311.12908)：扩散模型 reference-relative preference optimization 基础。\n- [DecompDPO](https://arxiv.org/abs/2407.13981)：分子扩散中的 global/local preference 与终局随机加噪实现参考。\n- [Diffusion-SDPO](https://arxiv.org/abs/2511.03317)：标准 diffusion preference update 的 winner 退化诊断及 safeguard；对分子体系尚需验证。\n- 本地 DiffSBDD/DecompDPO 代码审计与数值 probe：见报告内来源详情。"},
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "headline": [{"train_pairs": 17289, "good_coverage": 0.9539811289521603, "x_fraction_max": max(float(row["x_error_fraction"]) for row in x_fraction_rows), "shortcut_auc": max(float(row["auc"]) for row in descriptor_auc_rows)}],
                "x_fraction": x_fraction_rows,
                "descriptor_auc": descriptor_auc_rows,
                "pairing_coverage": pairing_chart_rows,
                "adaptation": adaptation_rows,
                "risks": risk_rows,
                "gates": gate_rows,
            },
        },
        "sources": sources,
        "package_info": {"originUrl": "artifact://diffsbdd-global-dpo-feasibility-260822", "controls": {"edit": False, "refresh": False}},
    }


def main() -> None:
    """Write artifact.json beside this script."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "artifact.json"
    output_path.write_text(
        json.dumps(build_artifact(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output_path)


if __name__ == "__main__":
    main()
