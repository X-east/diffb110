from __future__ import annotations

import csv
import subprocess
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LAUNCH_ROOT = ROOT / "launch"
OUTDIR = ROOT / "analysis" / "launch_cleanup_20260618"


ACTIVE_CORE = {
    "launch/run.sh",
    "launch/smoke.sh",
    "launch/smoke.yaml",
    "launch/7gpu_dock_classifier_v0_1.yaml",
    "launch/8gpu_dock.yaml",
    "launch/8gpu_chembl_only.yaml",
}

ACTIVE_MATRIX = {
    "launch/run_diffsbdd_single_reward_matrix_20260612.sh",
    "launch/generate_diffsbdd_single_reward_matrix_20260612.py",
    "launch/run_diffgui_ring_control_matrix_20260615.sh",
    "launch/generate_diffgui_ring_control_matrix_20260615.py",
}

ACTIVE_DIFFGUI = {
    "launch/run_diffgui_docking_train_ddp_smoke.sh",
    "launch/diffgui_docking_train_ddp_smoke.yaml",
}

DIFFGUI_SUPPORT = {
    "launch/archive/diffgui_historical/diffgui_docking_train_smoke.yaml",
    "launch/archive/diffgui_historical/diffgui_docking_train_stage1_last1_b16.yaml",
    "launch/archive/diffgui_historical/run_diffgui_docking_train_stage1.sh",
    "launch/archive/diffgui_historical/diffgui_reward_only_smoke.yaml",
    "launch/archive/diffgui_historical/diffgui_docking_batch_sweep_smoke.yaml",
    "launch/archive/diffgui_historical/diffgui_8gpu_b24_ring_last3_lr3e5.yaml",
    "launch/archive/diffgui_historical/diffgui_8gpu_b24_aromatic_last3_lr3e5.yaml",
    "launch/archive/diffgui_historical/diffgui_8gpu_b24_docking_only_lr1e5_last2.yaml",
    "launch/archive/diffgui_historical/diffgui_8gpu_b24_chembl21_dock.yaml",
    "launch/archive/diffgui_historical/diffgui_8gpu_b36_classifier_v0_3_1_200step_ppo4.yaml",
    "launch/archive/diffgui_historical/run_diffgui_8gpu_ring_last3_lr3e5.sh",
    "launch/archive/diffgui_historical/run_diffgui_8gpu_aromatic_last3_lr3e5.sh",
    "launch/archive/diffgui_historical/run_diffgui_8gpu_docking_only_lr1e5_last2.sh",
    "launch/archive/diffgui_historical/run_diffgui_8gpu_train.sh",
    "launch/archive/diffgui_historical/run_diffgui_8gpu_b36_classifier_v0_3_1_ppo4.sh",
    "launch/archive/diffgui_historical/run_diffgui_ring_control_matrix_20260613.sh",
    "launch/archive/diffgui_historical/generate_diffgui_ring_control_matrix_20260613.py",
}

STALE_WRAPPERS = {
    "launch/archive/legacy_diffsbdd_wrappers/run_6gpu.sh",
    "launch/archive/legacy_diffsbdd_wrappers/run_6gpu_dock.sh",
    "launch/archive/legacy_diffsbdd_wrappers/run_8gpu_dock.sh",
    "launch/archive/legacy_diffsbdd_wrappers/run_6gpu_chembl_only.sh",
    "launch/archive/legacy_diffsbdd_wrappers/run_8gpu_chembl_only.sh",
    "launch/archive/legacy_diffsbdd_wrappers/run_6gpu_dock_kl_sweep.sh",
}

RESEARCH_OR_HISTORICAL = {
    "launch/archive/legacy_diffsbdd_wrappers/6gpu_chembl_only.yaml",
    "launch/archive/legacy_diffsbdd_wrappers/6gpu_dock.yaml",
    "launch/archive/legacy_diffsbdd_wrappers/7gpu_classifier_only.yaml",
    "launch/archive/legacy_diffsbdd_wrappers/run.yaml",
    "launch/archive/research_lines/molskill_test.yaml",
    "launch/archive/research_lines/smoke_docking_classifier_v0_1.yaml",
    "launch/archive/research_lines/diffsbdd_8gpu_anticarbon_guard_20260611.yaml",
    "launch/archive/research_lines/run_diffsbdd_8gpu_anticarbon_guard_20260611.sh",
    "launch/archive/research_lines/edge_diffusion_10k_6gpu.yaml",
    "launch/archive/research_lines/edge_diffusion_ddp_smoke.yaml",
    "launch/archive/research_lines/run_edge_diffusion_ddp.sh",
    "launch/archive/research_lines/diffsbdd_xhe_diffgui_scratch_8gpu.yaml",
    "launch/archive/research_lines/diffsbdd_xhe_diffgui_scratch_pocket700_8gpu.yaml",
    "launch/archive/research_lines/diffsbdd_xhe_diffgui_scratch_smoke.yaml",
    "launch/archive/research_lines/smoke_xhe_scratch_pocket700.yaml",
}


def _git_output(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True)


def _tracked_files() -> set[str]:
    return set(_git_output(["ls-files", "launch"]).splitlines())


def _ignored_files() -> set[str]:
    ignored: set[str] = set()
    for line in _git_output(["status", "--short", "--ignored", "--", "launch"]).splitlines():
        if not line.startswith("!! "):
            continue
        rel = line[3:].strip()
        path = ROOT / rel
        if rel.endswith("/") and path.exists():
            ignored.update(str(p.relative_to(ROOT)) for p in path.rglob("*") if p.is_file())
        else:
            ignored.add(rel)
    return ignored


def _classify(path: str) -> tuple[str, str]:
    if path in {"launch/README.md", "launch/archive/README.md"}:
        return "active_documentation_keep", "保留；launch 目录入口说明"
    if path.startswith("launch/archive/cache/"):
        return "archived_cache", "已归档 cache；可清理"
    if path.startswith("launch/archive/logs/") or path.startswith("launch/archive/runtime_logs/"):
        return "archived_runtime_logs", "已归档运行日志；不属于入口层"
    if path.startswith("launch/archive/generated/"):
        return "archived_generated_regenerable", "已归档自动生成配置"
    if path in ACTIVE_CORE:
        return "active_core_keep", "保留；当前通用正式入口或核心配置"
    if path in ACTIVE_MATRIX:
        return "active_matrix_keep", "保留；当前仍有复现价值的矩阵生成/启动入口"
    if path in ACTIVE_DIFFGUI:
        return "active_diffgui_keep", "保留；当前 DiffGui smoke 入口"
    if path in STALE_WRAPPERS:
        return "archived_stale_wrapper", "已归档；不建议作为新入口"
    if path in DIFFGUI_SUPPORT:
        return "archived_diffgui_historical", "已归档；DiffGui 历史实验入口"
    if path in RESEARCH_OR_HISTORICAL:
        return "archived_research_or_legacy", "已归档；研究线或历史配置"
    return "uncategorized_review", "需要人工复核后再处理"


def _rows() -> list[dict[str, object]]:
    tracked = _tracked_files()
    ignored = _ignored_files()
    rows: list[dict[str, object]] = []
    for file_path in sorted(p for p in LAUNCH_ROOT.rglob("*") if p.is_file()):
        rel = str(file_path.relative_to(ROOT))
        category, action = _classify(rel)
        rows.append(
            {
                "path": rel,
                "category": category,
                "action": action,
                "tracked": rel in tracked,
                "ignored": rel in ignored,
                "size_bytes": file_path.stat().st_size,
            }
        )
    return rows


def _write_csv(rows: list[dict[str, object]], summary: Counter[str]) -> None:
    with (OUTDIR / "launch_inventory.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["path", "category", "action", "tracked", "ignored", "size_bytes"],
        )
        writer.writeheader()
        writer.writerows(rows)

    with (OUTDIR / "launch_inventory_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "count"])
        for category, count in sorted(summary.items(), key=lambda item: (-item[1], item[0])):
            writer.writerow([category, count])


def _write_plot(summary: Counter[str]) -> None:
    import matplotlib.pyplot as plt

    items = sorted(summary.items(), key=lambda item: item[1])
    labels = [key for key, _ in items]
    values = [value for _, value in items]
    fig_h = max(4.5, 0.42 * len(labels) + 1.6)
    fig, ax = plt.subplots(figsize=(9.0, fig_h))
    palette = [
        "#4f81bd",
        "#9bbb59",
        "#c0504d",
        "#8064a2",
        "#4bacc6",
        "#f79646",
        "#7f7f7f",
        "#c49a4a",
        "#5d9a8d",
    ]
    ax.barh(labels, values, color=palette[: len(labels)])
    ax.set_xlabel("file count")
    ax.set_title("launch/ file categories, 2026-06-18")
    ax.grid(axis="x", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for index, value in enumerate(values):
        ax.text(value + 0.4, index, str(value), va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTDIR / "launch_file_category_summary.png", dpi=180)
    plt.close(fig)


def _write_report(rows: list[dict[str, object]], summary: Counter[str]) -> None:
    tracked_count = sum(1 for row in rows if row["tracked"])
    ignored_count = sum(1 for row in rows if row["ignored"])
    root_file_count = sum(
        1
        for row in rows
        if len(str(row["path"]).split("/")) == 2
    )

    table_lines = ["| Category | Count |", "|---|---:|"]
    for category, count in sorted(summary.items(), key=lambda item: (-item[1], item[0])):
        table_lines.append(f"| `{category}` | {count} |")

    report = f"""# launch 目录整理审计

> 日期：2026-06-18

## 结论

`launch/` 已整理为“根目录只保留活跃入口，`launch/archive/` 保存历史/生成/日志/cache”。旧 6/8GPU wrapper、DiffGui 历史单次脚本、研究线配置、自动生成 YAML 和运行日志都已从入口层移出。

## 分类统计

![launch 文件分类统计](launch_file_category_summary.png)

{chr(10).join(table_lines)}

## 数量依据

| 指标 | 数量 |
|---|---:|
| `launch/` 文件总数 | {len(rows)} |
| `launch/` 根目录文件数 | {root_file_count} |
| Git 跟踪文件 | {tracked_count} |
| Git 忽略文件 | {ignored_count} |
| 已归档自动生成文件 | {summary.get("archived_generated_regenerable", 0)} |
| 已归档日志文件 | {summary.get("archived_runtime_logs", 0)} |
| 已归档 cache 文件 | {summary.get("archived_cache", 0)} |

## 已从入口层移出的内容

- `launch/archive/cache/`：Python bytecode cache。
- `launch/archive/runtime_logs/`：原先散落在 `launch/` 根目录的 nohup 日志。
- `launch/archive/logs/`：矩阵 launcher 日志。
- `launch/archive/generated/`：自动生成 YAML、matrix CSV/JSON。
- `launch/archive/legacy_diffsbdd_wrappers/`：旧 6/8GPU wrapper。
- `launch/archive/diffgui_historical/`：DiffGui 历史实验入口。
- `launch/archive/research_lines/`：edge diffusion、x/h/e、anti-carbon guard、MolSkill 等研究线配置。

## 不建议继续作为新入口的脚本

- `launch/archive/legacy_diffsbdd_wrappers/run_6gpu.sh`
- `launch/archive/legacy_diffsbdd_wrappers/run_6gpu_dock.sh`
- `launch/archive/legacy_diffsbdd_wrappers/run_8gpu_dock.sh`
- `launch/archive/legacy_diffsbdd_wrappers/run_6gpu_chembl_only.sh`
- `launch/archive/legacy_diffsbdd_wrappers/run_8gpu_chembl_only.sh`
- `launch/archive/legacy_diffsbdd_wrappers/run_6gpu_dock_kl_sweep.sh`

这些脚本的共同问题是：功能已经可由 `CONFIG_PATH=... bash launch/run.sh` 覆盖，并且多处仍引用不存在的 `launch/yaml_launch_env.py`。如要保留，应修复 helper 路径并移入历史/兼容区；如不保留，应先确认没有正在复现实验依赖它们。

## 建议的新规则

新运行优先使用：

```bash
bash launch/smoke.sh
CONFIG_PATH=launch/<config>.yaml bash launch/run.sh
```

DiffGui 矩阵实验保留专用 wrapper；单次历史实验脚本已移入 archive，只用于复现，不作为默认入口。

## 支持文件

- [launch_inventory.csv](launch_inventory.csv)
- [launch_inventory_summary.csv](launch_inventory_summary.csv)
"""
    (OUTDIR / "report.md").write_text(report, encoding="utf-8")



def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows = _rows()
    summary: Counter[str] = Counter(str(row["category"]) for row in rows)
    _write_csv(rows, summary)
    _write_plot(summary)
    _write_report(rows, summary)
    print(f"wrote {OUTDIR}")
    print(f"rows={len(rows)}")
    for category, count in sorted(summary.items(), key=lambda item: (-item[1], item[0])):
        print(f"{category}\t{count}")


if __name__ == "__main__":
    main()
