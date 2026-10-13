"""Build a DiffGui RL next-test planning report from local artifacts."""

from __future__ import annotations

import csv
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[2]
SINGLE_DELTA = PROJECT_ROOT / "analysis/diffgui_rl/diffgui_single_reward_analysis_20260616/single_reward_start_end_delta.csv"
COMPONENT_DELTA = PROJECT_ROOT / "analysis/diffgui_rl/diffgui_single_reward_analysis_20260616/single_reward_component_delta.csv"
COMBO_DELTA = PROJECT_ROOT / "analysis/diffgui_rl/diffgui_single_reward_analysis_20260616/combination_reward_start_end_delta.csv"
RING_REPORT = PROJECT_ROOT / "analysis/diffgui_rl/diffgui_ring_last3_lr3e5_training_plots_20260611/report.md"
MOLECULE_REPORT = PROJECT_ROOT / "analysis/diffgui_rl/diffgui_ring_last3_lr3e5_training_plots_20260611/molecular_property_report.md"


def main() -> None:
    """Write CSV, figure, and Markdown report."""
    ROOT.mkdir(parents=True, exist_ok=True)
    single_rows = _read_csv(SINGLE_DELTA)
    component_rows = _read_csv(COMPONENT_DELTA)
    combo_rows = _read_csv(COMBO_DELTA)

    evidence_rows = _build_evidence_rows(single_rows, component_rows, combo_rows)
    test_rows = _build_test_matrix()
    _write_csv(ROOT / "evidence_summary.csv", evidence_rows)
    _write_csv(ROOT / "diffgui_rl_next_test_matrix.csv", test_rows)
    _plot_priority_matrix(test_rows, ROOT / "diffgui_rl_next_test_priority.png")
    _write_report(evidence_rows, test_rows, ROOT / "report.md")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, "") or default)
    except ValueError:
        return default


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _build_evidence_rows(
    single_rows: list[dict[str, str]],
    component_rows: list[dict[str, str]],
    combo_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Extract high-signal evidence from previous DiffGui runs."""
    component_by_slug = {row["slug"]: row for row in component_rows}
    selected_slugs = {
        "docking",
        "ring_pathology",
        "ring_complexity",
        "aromatic_scaffold",
        "chembl_classifier",
        "qed",
        "sa",
        "ligand_efficiency",
    }
    rows: list[dict[str, Any]] = []
    for row in single_rows:
        slug = row.get("slug", "")
        if slug not in selected_slugs:
            continue
        component = component_by_slug.get(slug, {})
        rows.append(
            {
                "run_group": "single_reward",
                "slug": slug,
                "last_epoch": row.get("last_epoch", ""),
                "delta_reward_total_mean": _fmt(_float(row, "delta_reward_total_mean")),
                "delta_target_component_mean": _fmt(_float(component, "delta_target_component_mean")),
                "last_valid_ratio": _fmt(_float(row, "last_molecule_rdkit_valid_ratio")),
                "last_descriptor_mmd": _fmt(_float(row, "last_descriptor_mmd_vs_baseline")),
                "last_ring_count": _fmt(_float(row, "last_ring_count_mean")),
                "last_aromatic_ring": _fmt(_float(row, "last_aromatic_ring_mean")),
                "last_three_member_total": _fmt(_float(row, "last_three_member_ring_total")),
                "last_kl": _fmt(_float(row, "last_PPO_ApproxKL")),
                "local_interpretation": _interpret_single(row, component),
            }
        )
    for row in combo_rows:
        rows.append(
            {
                "run_group": "combination_reward",
                "slug": row.get("slug", ""),
                "last_epoch": row.get("last_epoch", ""),
                "delta_reward_total_mean": _fmt(_float(row, "delta_reward_total_mean")),
                "delta_target_component_mean": "",
                "last_valid_ratio": _fmt(_float(row, "last_molecule_rdkit_valid_ratio")),
                "last_descriptor_mmd": _fmt(_float(row, "last_descriptor_mmd_vs_baseline")),
                "last_ring_count": _fmt(_float(row, "last_ring_count_mean")),
                "last_aromatic_ring": _fmt(_float(row, "last_aromatic_ring_mean")),
                "last_three_member_total": _fmt(_float(row, "last_three_member_ring_total")),
                "last_kl": _fmt(_float(row, "last_PPO_ApproxKL")),
                "local_interpretation": _interpret_combo(row),
            }
        )
    return rows


def _interpret_single(row: dict[str, str], component: dict[str, str]) -> str:
    slug = row.get("slug", "")
    target_delta = _float(component, "delta_target_component_mean")
    valid = _float(row, "last_molecule_rdkit_valid_ratio")
    three_total = _float(row, "last_three_member_ring_total")
    mmd = _float(row, "last_descriptor_mmd_vs_baseline")
    if slug in {"ring_pathology", "aromatic_scaffold", "chembl_classifier"} and target_delta > 0:
        return "目标组件有正向信号，但仍要看环拓扑和分布漂移"
    if slug in {"qed", "sa", "ligand_efficiency", "ring_complexity"} and target_delta < 0:
        return "单奖励下目标组件首末下降，需先测 PPO/奖励尺度稳定性"
    if slug == "docking":
        return "总 reward 高但 docking 组件未明显改善，需独立重打分"
    if valid < 0.88 or three_total > 25 or mmd > 0.30:
        return "结构或分布风险较高，不能直接作为主路线"
    return "可作为对照或组合组件"


def _interpret_combo(row: dict[str, str]) -> str:
    slug = row.get("slug", "")
    if slug == "le_ring_aromatic":
        return "分布漂移较低且三元环较少，值得作为环控制组合对照"
    if slug == "dock_ring_aromatic":
        return "有效率高但三元环和 ring_count 仍偏高，需换 ring_pathology v2"
    return "组合奖励可作为权重扫描基线"


def _build_test_matrix() -> list[dict[str, Any]]:
    """Define the recommended next-test matrix."""
    return [
        _row(
            "T01",
            "基线与评估",
            "未训练 DiffGui baseline 等量采样",
            "没有同 pocket、同采样量的 base checkpoint 参照，所有 RL 结论都只能互比。",
            "reference DiffGui `trained.pt`; 现有矩阵每 run 144 mol/epoch。",
            "用同 pocket、同 ligand_atoms/priori、同 gui_strength，固定 3-5 个 seed 生成等量分子，只做 reward-only 重打分。",
            "baseline 生成的 validity、ring、MMD、docking 和 classifier 分布成为所有 RL 的零点。",
            "若 baseline 本身已有同等或更好 top 分子，RL 暂停调参，先修 reward/筛选。",
            priority=5,
            cost=2,
            stage="立即",
            category_score=5,
        ),
        _row(
            "T02",
            "基线与评估",
            "checkpoint 离线采样评估",
            "训练 rollout 样本和策略更新样本混在一起，不能证明 checkpoint 真变好。",
            "每个正式 run 已保存 epoch 5/10 checkpoint，旧 ring-last3 有完整 epoch 1-23 summary。",
            "对 epoch 0/5/10/23 checkpoint 各离线采样 500-1000 mol，统一重打分。",
            "同一 checkpoint 离线样本中 ring_pathology 下降、valid/top docking 不降、unique scaffold 不塌。",
            "训练期 reward 上升但离线采样无改善，判为 rollout 偏差或 reward hacking。",
            priority=5,
            cost=3,
            stage="立即",
            category_score=5,
        ),
        _row(
            "T03",
            "环病理",
            "ring_pathology v2 结构事实审计",
            "用户核心问题是环太多太复杂；旧 ring_complexity 会误伤芳香环并留下大环/复杂拓扑。",
            "ring-last3: aromatic ring 1.350->0.894，large_ring_count 0.667->1.000，complex topology 基本未降。",
            "先离线给已有分子打 v2 facts，再小训练：保留苯环/合理芳香，强扣大环、稠合大环、桥头/螺环、复杂多稠杂环。",
            "large_ring_fused_flag、complex_ring_topology、三元环下降；benzene-like/aromatic 不被继续压低。",
            "只降低总 ring_count 或芳香环而复杂拓扑不降，判为错误方向。",
            priority=5,
            cost=3,
            stage="立即",
            category_score=5,
        ),
        _row(
            "T04",
            "环病理",
            "aromatic-preservation vs ring-suppression ablation",
            "需要证明不是把所有环都压没，而是压坏环、保留合理药物样芳香骨架。",
            "aromatic_scaffold 单奖励目标组件提升；ring_complexity 单奖励目标组件下降。",
            "3 组小训：ring_pathology only、aromatic only、ring_pathology+aromatic；每组 3-5 epoch。",
            "组合组同时满足坏环下降、苯环/芳香环不下降、MMD 不升。",
            "出现去芳香化、脂环/大环替代、卤代小芳香堆积，判为 reward 需要拆项。",
            priority=5,
            cost=3,
            stage="立即",
            category_score=5,
        ),
        _row(
            "T05",
            "环病理",
            "ligand_atoms / priori 分布敏感性",
            "环复杂可能来自生成原子数分布过宽，不一定全是 reward 问题。",
            "当前 priori 下 ligand_atoms 可到 38-40，ring-last3 末端 mol_wt 变大。",
            "固定 18/22/26 atoms 和 priori 各跑 baseline + 小训，比较环病理按 heavy atom 归一化后的变化。",
            "若固定较小尺寸显著降低复杂环且 docking 不变差，后续 reward 需加入尺寸/LE 约束。",
            "若固定尺寸仍复杂，则问题主要在 edge/halfedge 生成偏好。",
            priority=4,
            cost=2,
            stage="第一批",
            category_score=4,
        ),
        _row(
            "T06",
            "奖励函数",
            "docking/LE + ring_pathology v2 权重扫描",
            "已有 dock_ring_aromatic 有效率高但三元环仍多；le_ring_aromatic 分布更稳。",
            "组合结果：LE 组合 MMD=0.232、三元环16；docking组合 valid=0.924 但三元环27。",
            "扫描 docking/LE 主项 0.25-0.40，ring_pathology v2 0.25-0.40，aromatic 0.10-0.25。",
            "选 Pareto：valid>=0.90、MMD<=0.25、three_member_total 下降、top docking 不差于 baseline。",
            "若单一权重只提升一个指标并损坏其他指标，不进入长训。",
            priority=5,
            cost=4,
            stage="第一批",
            category_score=5,
        ),
        _row(
            "T07",
            "奖励函数",
            "D-MPNN/classifier 校准进入 reward",
            "旧三层 MLP classifier 单独奖励仍为负，不能独自承担药物样约束。",
            "chembl_classifier 单奖励目标组件 +0.223，但 last score 仍为负；已有 D-MPNN checkpoint 分析线。",
            "离线比较 MLP v0.3.1、D-MPNN、docking、ring facts 的相关性；再把 D-MPNN 作为小权重软约束。",
            "D-MPNN 高分 top 分子在人工/规则审计中坏环率更低，且不与 docking 完全冲突。",
            "若 D-MPNN 只学习训练集 scaffold 或偏好复杂环，不能作为主 reward。",
            priority=4,
            cost=3,
            stage="第一批",
            category_score=4,
        ),
        _row(
            "T08",
            "PPO稳定性",
            "lr/clip/freeze 小矩阵",
            "多个单奖励目标组件下降，可能不是 reward 错，而是 PPO 更新过猛或解冻策略不合适。",
            "单奖励 KL 多在 1.7-4.6；当前常用 lr=3e-5、last3、clip=0.2。",
            "固定一个组合 reward，扫 lr 1e-5/2e-5/3e-5，clip 0.1/0.2，last1/last3，3 epoch。",
            "目标组件单调或总体上升，KL/ClipFrac 可控，valid/MMD 不恶化。",
            "目标组件仍下降或 KL 高且 reward 波动，先降学习率/收窄训练层。",
            priority=5,
            cost=4,
            stage="立即",
            category_score=5,
        ),
        _row(
            "T09",
            "PPO稳定性",
            "per-timestep KL/halfedge 诊断",
            "DiffGui 的环来自 edge/halfedge，必须看 PPO 主要改的是位置、原子还是边。",
            "`ppo_step_metrics.csv` 已记录 pos/node/halfedge logprob；`kl_by_timestep.csv` 已合并。",
            "对已有 run 出 heatmap：KL、ratio、node/halfedge logprob 按 timestep/epoch；标出异常高 KL 区间。",
            "半边 logits 更新和坏环下降相关，而不是只有 position logprob 改变。",
            "若 halfedge 几乎不动，RL 难以解决环；需要奖励/解冻/学习率针对 edge 分支调整。",
            priority=4,
            cost=1,
            stage="立即",
            category_score=4,
        ),
        _row(
            "T10",
            "PPO稳定性",
            "guided property labels vs RL reward 解耦",
            "DiffGui 自带 property guidance，RL 可能被 gui_strength/property label 主导。",
            "DiffGui README 指出 logp/tpsa/sa/qed/aff 可调；当前 configs 用 gui_strength=3.0。",
            "固定 checkpoint，扫 gui_strength 0/1/3/5 和 batch_lab 目标；再加/不加 PPO 小训。",
            "确认 RL reward 的增益超过 guidance-only，并且不只是 guidance 改变分布。",
            "如果 guidance-only 已达到同等效果，优先做条件引导而不是 RL。",
            priority=4,
            cost=3,
            stage="第一批",
            category_score=4,
        ),
        _row(
            "T11",
            "过滤与回放",
            "不合法/极端坏分子不参与训练 ablation",
            "用户明确希望失败和极端坏分子不参与 PPO；但过度丢弃会造成选择偏差。",
            "`DiffGuiTrainingFilter` 已支持 valid/reconstruction/quantile/quality filters，尚缺正式 ablation。",
            "四组：不过滤、只过滤 invalid、invalid+bottom10%、invalid+bad ring facts；都保留失败记录。",
            "过滤后 PPO 更稳、valid 上升、坏环下降，不降低 diversity/top docking。",
            "训练样本太少导致 DDP crop/跳过 PPO 或模式坍缩，则过滤阈值过重。",
            priority=5,
            cost=3,
            stage="立即",
            category_score=5,
        ),
        _row(
            "T12",
            "过滤与回放",
            "good pool store-only 审计",
            "回放前先确认保存的好分子真好、去重有效、没有把错误结构存成高分。",
            "`DiffGuiGoodMoleculePool` 已实现 top-k、去重、retained pool、SDF 输出。",
            "打开 good_pool、关闭 replay，跑 5 epoch；人工/规则审计 retained top-k。",
            "retained pool 中坏环率低、重复低、top docking/D-MPNN 可信。",
            "如果 retained pool 被 reward hacking 污染，不能开启 replay。",
            priority=4,
            cost=2,
            stage="第一批",
            category_score=4,
        ),
        _row(
            "T13",
            "过滤与回放",
            "短窗口小比例 replay + KL gate",
            "好分子轨迹复用可能提升样本效率，但 off-policy 风险最大。",
            "GPU1 smoke 已证明 replay/gate 功能闭环，epoch2 ratio 0.5->0.25。",
            "在最佳过滤配置上扫 replay_ratio 0/0.05/0.10，max_age 2-3，max KL/clip gate 打开。",
            "相对无 replay，top 分子率提高且 KL/ClipFrac 不升、重复不升。",
            "gate 持续降到 0 或重复率升高，说明 replay 暂不适合正式训练。",
            priority=3,
            cost=3,
            stage="第二批",
            category_score=3,
        ),
        _row(
            "T14",
            "奖励黑客",
            "top 分子独立重打分与人工审计",
            "训练期 docking/reward 容易被构象、质子化、重建异常利用。",
            "每个 epoch 已有 `top_reward.sdf`、`molecules.sdf`、`failures.csv`。",
            "每组取 top50，重新 protonation/3D cleanup 后 docking、D-MPNN、ring facts、相互作用几何审计。",
            "top 分子仍然通过独立审计，并且不是重复 scaffold。",
            "top reward 分子大量失败/结构怪，说明 reward 被黑，需要先改 reward。",
            priority=5,
            cost=2,
            stage="立即",
            category_score=5,
        ),
        _row(
            "T15",
            "奖励黑客",
            "多样性/模式坍缩监控",
            "RL 很容易只生成少数高分 scaffold，尤其有 replay/good pool 时。",
            "主流程已写 unique canonical smiles、duplicate ratio、unique scaffolds、pairwise similarity。",
            "所有实验统一画 duplicate、scaffold_top1、ECFP/FCFP similarity、top-k scaffold 占比。",
            "奖励改善时 duplicate 不升、unique scaffold 不降，top1 scaffold fraction 不膨胀。",
            "奖励上升但 scaffold_top1/duplicate 上升，判为坍缩。",
            priority=4,
            cost=1,
            stage="立即",
            category_score=4,
        ),
        _row(
            "T16",
            "泛化",
            "多 seed 重复与跨靶点复核",
            "现在多数 run 单 seed，不能分清真实趋势和采样噪声。",
            "单奖励分析已经指出每个奖励只有一条 10 epoch 轨迹。",
            "最佳 2-3 个组合各 3 seed；再在 SHP2/8GJL/9PC 等已有靶点上做离线采样评估。",
            "趋势跨 seed 保持，跨靶点不明显增加坏环或 invalid。",
            "只有单 seed 成功或换靶点崩，不能作为主方法结论。",
            priority=4,
            cost=5,
            stage="第二批",
            category_score=4,
        ),
    ]


def _row(
    test_id: str,
    category: str,
    question: str,
    why_now: str,
    existing_basis: str,
    implementation: str,
    success_metric: str,
    failure_signal: str,
    *,
    priority: int,
    cost: int,
    stage: str,
    category_score: int,
) -> dict[str, Any]:
    return {
        "test_id": test_id,
        "category": category,
        "question": question,
        "why_now": why_now,
        "existing_basis": existing_basis,
        "implementation": implementation,
        "success_metric": success_metric,
        "failure_signal": failure_signal,
        "priority_1_5": priority,
        "cost_1_5": cost,
        "stage": stage,
        "decision_value_1_5": category_score,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_priority_matrix(rows: list[dict[str, Any]], path: Path) -> None:
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    colors = {
        "基线与评估": "#3B82F6",
        "环病理": "#EF4444",
        "奖励函数": "#F59E0B",
        "PPO稳定性": "#10B981",
        "过滤与回放": "#8B5CF6",
        "奖励黑客": "#64748B",
        "泛化": "#14B8A6",
    }
    legend_labels = {
        "基线与评估": "Baseline/eval",
        "环病理": "Ring pathology",
        "奖励函数": "Reward design",
        "PPO稳定性": "PPO stability",
        "过滤与回放": "Filter/replay",
        "奖励黑客": "Reward hacking",
        "泛化": "Generalization",
    }
    fig, ax = plt.subplots(figsize=(11, 7), dpi=180)
    for row in rows:
        x = int(row["cost_1_5"])
        y = int(row["priority_1_5"])
        cat = str(row["category"])
        ax.scatter(
            x,
            y,
            s=150 + 50 * int(row["decision_value_1_5"]),
            color=colors.get(cat, "#6B7280"),
            edgecolor="white",
            linewidth=1.2,
            alpha=0.9,
        )
        ax.text(x + 0.04, y + 0.04, row["test_id"], fontsize=8, color="#111827")
    ax.set_xlim(0.5, 5.6)
    ax.set_ylim(2.5, 5.5)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_yticks([3, 4, 5])
    ax.set_xlabel("Cost / runtime burden (1 low, 5 high)")
    ax.set_ylabel("Priority (1 low, 5 high)")
    ax.set_title("DiffGui RL next-test priority matrix")
    ax.grid(True, color="#E5E7EB", linewidth=0.8)
    for cat, color in colors.items():
        if any(row["category"] == cat for row in rows):
            ax.scatter([], [], color=color, label=legend_labels.get(cat, cat), s=80)
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _markdown_table(rows: list[dict[str, Any]], columns: list[str], max_rows: int | None = None) -> str:
    shown = rows if max_rows is None else rows[:max_rows]
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]
    for row in shown:
        values = [str(row.get(column, "")).replace("\n", " ") for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _wrap(text: str, width: int = 88) -> str:
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False, replace_whitespace=False))


def _write_report(evidence_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]], path: Path) -> None:
    immediate = [row for row in test_rows if row["stage"] == "立即"]
    first_batch = [row for row in test_rows if row["stage"] == "第一批"]
    second_batch = [row for row in test_rows if row["stage"] == "第二批"]
    evidence_columns = [
        "run_group",
        "slug",
        "delta_reward_total_mean",
        "delta_target_component_mean",
        "last_valid_ratio",
        "last_descriptor_mmd",
        "last_ring_count",
        "last_aromatic_ring",
        "last_three_member_total",
        "last_kl",
        "local_interpretation",
    ]
    matrix_columns = [
        "test_id",
        "category",
        "question",
        "priority_1_5",
        "cost_1_5",
        "stage",
        "success_metric",
    ]
    report = f"""# DiffGui 强化学习后续测试深度分析

## 结论先行

DiffGui RL 现在最值得继续测，但目标应该收窄为：**能否稳定压制坏环/复杂环，同时保留合理芳香骨架、有效率、多样性和 docking top 分子**。它不适合作为双 R 对称侧链硬机制的主平台；那条路线已经暴露出机制不清晰和生成结果差的问题。DiffGui RL 更合理的定位是分布整形和候选筛选增强。

下一步不建议再直接开一个大训练。推荐顺序是：

1. 先补 `T01/T02/T14/T09`：baseline、checkpoint 离线采样、top 分子独立审计、per-timestep halfedge 诊断。
2. 再做 `T03/T04/T08/T11`：ring_pathology v2、芳香保留 ablation、PPO 稳定性、训练过滤。
3. 只有前两步成立，再做 `T06/T07/T12/T13/T16`：权重扫描、D-MPNN、good pool/replay、多 seed 和跨靶点。

## 现有证据

{_markdown_table(evidence_rows, evidence_columns)}

核心读法：

- `ring-last3` 证明 DiffGui 会响应结构 reward，但旧 ring reward 会把芳香环也压下去；更糟的是大环和复杂拓扑没有同步改善。
- 单奖励矩阵里，`aromatic_scaffold`、`chembl_classifier`、`ring_pathology` 的目标组件有正向信号；`QED/SA/LE/ring_complexity` 的目标组件反而下降，说明当前 PPO/奖励尺度并不总是稳定。
- 组合奖励里 `LE + ring_complexity + aromatic` 的分布漂移和三元环更好，`docking + ring_complexity + aromatic` 的有效率更高，但三元环仍偏多。下一版应把 `ring_complexity` 换成更细的 `ring_pathology v2`。

## 方法参照

- DDPO 把扩散反向去噪过程建模为多步决策过程，用终态 reward 优化下游目标。这支持当前 DiffGui PPO/logprob 设计，但也意味着终态分子 reward 是稀疏信号，必须监控 KL、clip、per-timestep 贡献和 reward hacking。[DDPO](https://arxiv.org/abs/2305.13301)
- DePPA 是 2026 年 pocket-aware diffusion + denoising policy optimization 的直接参照，目标包含 binding affinity、drug-likeness、synthesizability、diversity。这说明 DiffGui RL 的评估不能只看 docking 或总 reward，必须做多目标 Pareto 和多样性审计。[DePPA](https://arxiv.org/abs/2605.17693)
- REINVENT/GCPN/GraphAF 一类分子 RL 工作共同经验是：化学规则、合法性、去重/多样性和目标性质要一起控，不能只奖励一个黑盒分数。[REINVENT](https://arxiv.org/abs/1704.07555)、[GCPN](https://arxiv.org/abs/1806.02473)、[GraphAF](https://arxiv.org/abs/2001.09382)
- 本地 DiffGui 代码是 joint pos/node/edge diffusion：`reference project/代码/DiffGui/models/model.py` 中有 `pos_transition`、`node_transition`、`edge_transition`，并且 `num_edge_types` 是显式 bond type。因此 DiffGui 理论上比 DiffSBDD 更有希望通过 halfedge/edge 分支响应环拓扑 reward，但这要用 per-timestep 和 halfedge 诊断证实。

## 测试矩阵

{_markdown_table(test_rows, matrix_columns)}

![DiffGui RL next-test priority](diffgui_rl_next_test_priority.png)

完整矩阵见 [diffgui_rl_next_test_matrix.csv](diffgui_rl_next_test_matrix.csv)，已有证据摘要见 [evidence_summary.csv](evidence_summary.csv)。

## 立即可做的第一组

{_markdown_table(immediate, matrix_columns)}

这组的目的不是追求最好分子，而是判断“现在看到的 RL 改善到底可信不可信”。如果 baseline 或 checkpoint 离线采样不能支持训练期曲线，就不应该继续开更大训练。

## 第一批训练测试

{_markdown_table(first_batch, matrix_columns)}

这组开始真正处理环问题。关键不是降低平均 ring count，而是降低坏环事实：大环、稠合大环、桥头/螺环、复杂多稠杂环、异常三元环，同时保住苯环/合理芳香骨架。

## 第二批扩展测试

{_markdown_table(second_batch, matrix_columns)}

good pool 和 replay 放到第二批，是因为它们会放大已有 reward 的偏差。只有 retained pool 经过独立审计，且 KL/clip gate 不频繁触发，才值得打开小比例回放。

## 推荐决策标准

一个 DiffGui RL 配置进入长训，至少要同时满足：

- `rdkit_valid_ratio >= 0.90`，`reconstruction_success_ratio >= 0.90`。
- 相对 baseline，`large_ring_fused_flag`、`complex_ring_topology_flag`、`three_member_ring_total` 明显下降。
- `benzene_like_ring_count` 或合理 `aromatic_ring_count` 不继续被压低。
- `descriptor_mmd_vs_baseline <= 0.25` 或至少不高于当前组合奖励最好水平。
- `duplicate_ratio`、`scaffold_top1_fraction` 不上升，top 分子不是单一 scaffold。
- `PPO_ApproxKL`、`ClipFrac` 没有持续走高；若 replay 开启，gate 不应连续把 ratio 衰减到 0。
- top50 经过独立 docking/D-MPNN/RDKit 审计后仍然合理，而不是训练期 reward 虚高。

## 结论

DiffGui RL 后续还有值得测的空间，但不是“再多跑几组奖励”这么简单。最有价值的路线是：**先验证基线和离线采样，再把 ring reward 拆成坏环事实，最后在稳定 PPO 和训练过滤的前提下做小权重组合扫描**。如果这些测试不能证明坏环事实下降，同时芳香骨架、多样性和 top 分子质量不坏，那 DiffGui RL 就只能作为候选过滤/重排序辅助，而不应继续当主方法。
"""
    path.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
