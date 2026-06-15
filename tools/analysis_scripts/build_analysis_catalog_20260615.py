#!/usr/bin/env python
"""Build a non-destructive catalog for the DiffB110 analysis directory."""

from __future__ import annotations

import csv
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-diffb110")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = ROOT / "analysis"
OUT_DIR = ANALYSIS_DIR / "_index"
DATE_RE = re.compile(r"(20\d{6}|26\d{4})")
ORGANIZED_CATEGORY_DIRS = {
    "benchmark",
    "classifier",
    "diffgui_rl",
    "diffsbdd_rl",
    "diffsbdd_small_mod",
    "diffsbdd_xhe",
    "preprocessing",
    "reconstruction",
}


@dataclass(frozen=True)
class CategoryRule:
    """Keyword-based category rule for analysis artifacts."""

    category: str
    keywords: tuple[str, ...]


CATEGORY_RULES: tuple[CategoryRule, ...] = (
    CategoryRule(
        "classifier",
        (
            "classifier",
            "chembl",
            "good_bad",
            "named_sdf",
            "molecule_classification",
            "score_diffgui_diffsbdd",
        ),
    ),
    CategoryRule(
        "diffsbdd_xhe",
        (
            "xhe",
            "edge_",
            "edge-",
            "halfedge",
            "denoiser",
            "semantic_bundle",
            "transition",
        ),
    ),
    CategoryRule(
        "diffgui_rl",
        (
            "diffgui",
            "ring_last",
            "ring_reward",
            "timestep",
            "batch_sweep",
        ),
    ),
    CategoryRule(
        "diffsbdd_rl",
        (
            "diffsbdd_single_reward",
            "diffsbdd_rl",
            "diffsbdd_8gpu_5000_score",
            "reward_hacking",
            "rl_high_carbon",
            "kl_",
            "logprob",
            "good_pool",
            "reward_loophole",
            "rl_next",
            "rl_data",
        ),
    ),
    CategoryRule(
        "diffsbdd_small_mod",
        (
            "atom_score",
            "atom_bias",
            "hetero",
            "aromatic_benzene",
            "local_atom_mask",
            "final_refine",
            "refinement_ablation",
            "local_bond_predictor",
        ),
    ),
    CategoryRule(
        "benchmark",
        (
            "benchmark",
            "8gjl",
            "5xzr",
            "8j6",
            "formal_xh_xhe_comparison",
            "reference_project_generation",
            "reference_project_supplement",
            "model_strategy",
            "modern_sbdd",
        ),
    ),
    CategoryRule(
        "preprocessing",
        (
            "preprocess",
            "dataset",
            "selection",
            "crossdock",
            "train_analysis",
        ),
    ),
    CategoryRule(
        "reconstruction",
        (
            "reconstruction",
            "reconstruct",
            "oracle",
            "bond_predictor",
            "valid_sdf",
            "sanitize",
        ),
    ),
)


def _format_size(num_bytes: int) -> str:
    """Return a compact human-readable size string."""
    size = float(num_bytes)
    for unit in ("B", "K", "M", "G", "T"):
        if size < 1024.0 or unit == "T":
            if unit == "B":
                return f"{int(size)}{unit}"
            return f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{size:.1f}T"


def _classify(name: str) -> str:
    """Classify an analysis artifact from its name."""
    lowered = name.lower()
    if lowered in {"_index", "_archive", "_compat_links", "_loose", "__pycache__"}:
        return "internal"
    if "smoke" in lowered or "dryrun" in lowered or "debug" in lowered:
        return "smoke_or_debug"
    for rule in CATEGORY_RULES:
        if any(keyword in lowered for keyword in rule.keywords):
            return rule.category
    return "unclassified"


def _suggest_action(name: str, category: str, has_report: bool, size_bytes: int) -> str:
    """Suggest a conservative action without moving anything."""
    lowered = name.lower()
    if category == "internal":
        return "keep_internal"
    if category == "smoke_or_debug":
        return "archive_smoke"
    if re.search(r"(^|[_-])(obsolete|legacy|old)([_-]|$)", lowered):
        return "archive_candidate"
    if has_report or size_bytes >= 50 * 1024 * 1024:
        return "keep_review_before_move"
    if category == "unclassified":
        return "manual_review"
    return "move_candidate_with_symlink"


def _date_token(name: str) -> str:
    """Extract the first visible date token from an artifact name."""
    match = DATE_RE.search(name)
    return match.group(1) if match else ""


def _scan_directory(path: Path, forced_category: str | None = None) -> dict[str, object]:
    """Collect metadata for one top-level analysis directory."""
    size_bytes = 0
    file_count = 0
    dir_count = 0
    extension_counts: Counter[str] = Counter()
    report_paths: list[str] = []
    readme_paths: list[str] = []
    image_paths: list[str] = []
    latest_mtime = path.stat().st_mtime

    for root, dirs, files in os.walk(path):
        root_path = Path(root)
        dir_count += len(dirs)
        for filename in files:
            file_path = root_path / filename
            try:
                stat = file_path.stat()
            except OSError:
                continue
            size_bytes += stat.st_size
            file_count += 1
            latest_mtime = max(latest_mtime, stat.st_mtime)
            suffix = file_path.suffix.lower() or "<none>"
            extension_counts[suffix] += 1
            rel = file_path.relative_to(ROOT).as_posix()
            if file_path.name.lower() in {"report.md", "readme.md"} or file_path.suffix.lower() == ".md":
                if file_path.name.lower() == "readme.md":
                    readme_paths.append(rel)
                if file_path.name.lower() == "report.md" or "report" in file_path.stem.lower():
                    report_paths.append(rel)
            if suffix in {".png", ".svg", ".pdf"} and len(image_paths) < 12:
                image_paths.append(rel)

    category = forced_category or _classify(path.name)
    has_report = bool(report_paths or readme_paths)
    return {
        "kind": "directory",
        "name": path.name,
        "path": path.relative_to(ROOT).as_posix(),
        "category": category,
        "date_token": _date_token(path.name),
        "suggested_action": _suggest_action(path.name, category, has_report, size_bytes),
        "size_bytes": size_bytes,
        "size_human": _format_size(size_bytes),
        "file_count": file_count,
        "dir_count": dir_count,
        "latest_mtime": datetime.fromtimestamp(latest_mtime).isoformat(timespec="seconds"),
        "md_count": extension_counts[".md"],
        "png_count": extension_counts[".png"],
        "svg_count": extension_counts[".svg"],
        "pdf_count": extension_counts[".pdf"],
        "csv_count": extension_counts[".csv"],
        "json_count": extension_counts[".json"],
        "sdf_count": extension_counts[".sdf"],
        "pt_count": extension_counts[".pt"],
        "log_count": extension_counts[".log"],
        "py_count": extension_counts[".py"],
        "report_paths": "|".join(report_paths[:8]),
        "readme_paths": "|".join(readme_paths[:8]),
        "sample_figure_paths": "|".join(image_paths[:8]),
    }


def _scan_top_level_file(path: Path) -> dict[str, object]:
    """Collect metadata for one top-level analysis file."""
    stat = path.stat()
    category = _classify(path.name)
    if path.suffix.lower() == ".py":
        suggested_action = "move_to_analysis_scripts"
    elif path.suffix.lower() in {".png", ".svg", ".csv", ".md", ".json"}:
        suggested_action = "keep_or_attach_to_nearest_report"
    else:
        suggested_action = "manual_review"
    return {
        "kind": "file",
        "name": path.name,
        "path": path.relative_to(ROOT).as_posix(),
        "category": category,
        "date_token": _date_token(path.name),
        "suggested_action": suggested_action,
        "size_bytes": stat.st_size,
        "size_human": _format_size(stat.st_size),
        "latest_mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        "suffix": path.suffix.lower() or "<none>",
    }


def _iter_catalog_directories() -> list[tuple[Path, str | None]]:
    """Return real analysis directories to catalog.

    Supports both the old flat layout and the organized layout where experiments
    live one level below category folders such as analysis/classifier/.
    """
    entries: list[tuple[Path, str | None]] = []
    for path in sorted(ANALYSIS_DIR.iterdir(), key=lambda item: item.name):
        if not path.is_dir() or path.is_symlink():
            continue
        if path.name in ORGANIZED_CATEGORY_DIRS:
            for child in sorted(path.iterdir(), key=lambda item: item.name):
                if child.is_dir() and not child.is_symlink():
                    entries.append((child, path.name))
            continue
        entries.append((path, None))
    return entries


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    """Write rows as CSV."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _category_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Summarize directory rows by category."""
    summary: dict[str, dict[str, object]] = {}
    for row in rows:
        category = str(row["category"])
        bucket = summary.setdefault(
            category,
            {
                "category": category,
                "directory_count": 0,
                "size_bytes": 0,
                "file_count": 0,
                "report_dir_count": 0,
                "png_count": 0,
                "sdf_count": 0,
                "csv_count": 0,
            },
        )
        bucket["directory_count"] = int(bucket["directory_count"]) + 1
        bucket["size_bytes"] = int(bucket["size_bytes"]) + int(row["size_bytes"])
        bucket["file_count"] = int(bucket["file_count"]) + int(row["file_count"])
        bucket["report_dir_count"] = int(bucket["report_dir_count"]) + int(bool(row["report_paths"] or row["readme_paths"]))
        bucket["png_count"] = int(bucket["png_count"]) + int(row["png_count"])
        bucket["sdf_count"] = int(bucket["sdf_count"]) + int(row["sdf_count"])
        bucket["csv_count"] = int(bucket["csv_count"]) + int(row["csv_count"])

    output = list(summary.values())
    for row in output:
        row["size_human"] = _format_size(int(row["size_bytes"]))
        row["size_gb"] = round(int(row["size_bytes"]) / (1024**3), 3)
    return sorted(output, key=lambda item: int(item["size_bytes"]), reverse=True)


def _write_plots(directory_rows: list[dict[str, object]], summary_rows: list[dict[str, object]]) -> None:
    """Write compact overview plots for the catalog."""
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    categories = [str(row["category"]) for row in summary_rows]
    sizes = [float(row["size_gb"]) for row in summary_rows]
    counts = [int(row["directory_count"]) for row in summary_rows]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].barh(categories[::-1], sizes[::-1], color="#3268a8")
    axes[0].set_xlabel("Size (GB)")
    axes[0].set_title("analysis/ size by category")
    axes[0].grid(axis="x", alpha=0.25)

    axes[1].barh(categories[::-1], counts[::-1], color="#2f7f5f")
    axes[1].set_xlabel("Top-level directory count")
    axes[1].set_title("analysis/ directory count by category")
    axes[1].grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "category_overview.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / "category_overview.svg", bbox_inches="tight")
    plt.close(fig)

    top_dirs = sorted(directory_rows, key=lambda row: int(row["size_bytes"]), reverse=True)[:25]
    fig, ax = plt.subplots(figsize=(11, 7))
    names = [str(row["name"]) for row in top_dirs][::-1]
    values = [int(row["size_bytes"]) / (1024**3) for row in top_dirs][::-1]
    ax.barh(names, values, color="#7c3aed")
    ax.set_xlabel("Size (GB)")
    ax.set_title("Largest analysis directories")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "largest_directories.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / "largest_directories.svg", bbox_inches="tight")
    plt.close(fig)

    action_counts = Counter(str(row["suggested_action"]) for row in directory_rows)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    labels = list(action_counts.keys())
    values = [action_counts[label] for label in labels]
    ax.barh(labels[::-1], values[::-1], color="#d97706")
    ax.set_xlabel("Top-level directory count")
    ax.set_title("Conservative suggested actions")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "suggested_action_counts.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / "suggested_action_counts.svg", bbox_inches="tight")
    plt.close(fig)


def _write_markdown(
    directory_rows: list[dict[str, object]],
    file_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
) -> None:
    """Write human-readable index documents."""
    total_size = sum(int(row["size_bytes"]) for row in directory_rows) + sum(int(row["size_bytes"]) for row in file_rows)
    action_counts = Counter(str(row["suggested_action"]) for row in directory_rows)
    category_counts = Counter(str(row["category"]) for row in directory_rows)

    largest = sorted(directory_rows, key=lambda row: int(row["size_bytes"]), reverse=True)[:20]
    smoke = [row for row in directory_rows if row["suggested_action"] == "archive_smoke"]
    scripts = [row for row in file_rows if row["suggested_action"] == "move_to_analysis_scripts"]
    unclassified = [row for row in directory_rows if row["category"] == "unclassified"]

    lines = [
        "# analysis 目录索引",
        "",
        f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}",
        f"- 分类后真实结果目录数：{len(directory_rows)}",
        f"- 顶层散文件数：{len(file_rows)}",
        f"- 估算总大小：{_format_size(total_size)}",
        "- 整理原则：按模型线和任务线分类，旧顶层路径保留软链接，避免历史 Markdown / PPT / 命令记录断链。",
        "- 当前主目录：`classifier/`、`diffsbdd_rl/`、`diffsbdd_xhe/`、`diffsbdd_small_mod/`、`diffgui_rl/`、`benchmark/`、`preprocessing/`、`reconstruction/`。",
        "",
        "## 输出文件",
        "",
        "- `analysis_catalog.csv`：分类后真实结果目录完整索引",
        "- `top_level_files.csv`：顶层散文件索引",
        "- `category_summary.csv`：按类别汇总",
        "- `category_overview.png`：类别大小和数量概览",
        "- `largest_directories.png`：最大目录清单",
        "- `suggested_action_counts.png`：建议操作数量",
        "",
        "## 类别汇总",
        "",
        "| 类别 | 目录数 | 大小 | 有报告/README的目录 | PNG | SDF | CSV |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['category']} | {row['directory_count']} | {row['size_human']} | "
            f"{row['report_dir_count']} | {row['png_count']} | {row['sdf_count']} | {row['csv_count']} |"
        )
    lines.extend(
        [
            "",
            "![类别概览](category_overview.png)",
            "",
            "## 建议操作统计",
            "",
            "| 建议操作 | 目录数 | 说明 |",
            "|---|---:|---|",
        ]
    )
    action_descriptions = {
        "keep_internal": "索引目录或内部缓存，不作为实验结果移动。",
        "archive_smoke": "smoke/debug/dryrun 类，适合后续归到 `_archive/smoke/`。",
        "archive_candidate": "旧版或已被替代结果，移动前仍需确认是否被文档引用。",
        "keep_review_before_move": "较大或已有报告，先保留原路径，移动时建议建软链接。",
        "move_candidate_with_symlink": "可按主题搬迁，但为避免断链建议保留旧路径软链接。",
        "manual_review": "分类不确定，需要人工看一眼。",
    }
    for action, count in sorted(action_counts.items()):
        lines.append(f"| {action} | {count} | {action_descriptions.get(action, '')} |")
    lines.extend(
        [
            "",
            "![建议操作统计](suggested_action_counts.png)",
            "",
            "## 最大目录 Top 20",
            "",
            "| 目录 | 类别 | 大小 | 建议 | 报告/README |",
            "|---|---|---:|---|---|",
        ]
    )
    for row in largest:
        has_report = "是" if row["report_paths"] or row["readme_paths"] else "否"
        lines.append(
            f"| `{row['path']}` | {row['category']} | {row['size_human']} | "
            f"{row['suggested_action']} | {has_report} |"
        )
    lines.extend(
        [
            "",
            "![最大目录](largest_directories.png)",
            "",
            "## 优先可归档 smoke/debug 目录",
            "",
        ]
    )
    if smoke:
        lines.extend(["| 目录 | 类别 | 大小 |", "|---|---|---:|"])
        for row in smoke[:80]:
            lines.append(f"| `{row['path']}` | {row['category']} | {row['size_human']} |")
    else:
        lines.append("- 暂无。")
    lines.extend(
        [
            "",
            "## 顶层散脚本",
            "",
        ]
    )
    if scripts:
        lines.extend(["| 文件 | 大小 | 建议 |", "|---|---:|---|"])
        for row in scripts:
            lines.append(f"| `{row['path']}` | {row['size_human']} | {row['suggested_action']} |")
    else:
        lines.append("- 暂无。")
    lines.extend(
        [
            "",
            "## 需要人工判断的未分类目录",
            "",
        ]
    )
    if unclassified:
        lines.extend(["| 目录 | 大小 | 文件数 |", "|---|---:|---:|"])
        for row in unclassified:
            lines.append(f"| `{row['path']}` | {row['size_human']} | {row['file_count']} |")
    else:
        lines.append("- 暂无。")
    lines.extend(
        [
            "",
        "## 当前目录结构",
            "",
            "```text",
            "analysis/",
            "  _index/",
            "  classifier/",
            "  diffsbdd_rl/",
            "  diffgui_rl/",
            "  diffsbdd_xhe/",
            "  diffsbdd_small_mod/",
            "  benchmark/",
            "  preprocessing/",
            "  reconstruction/",
            "  _archive/smoke/",
            "  _compat_links/top_level/",
            "  _loose/",
            "tools/analysis_scripts/legacy_top_level/",
            "```",
            "",
            "说明：`analysis/` 顶层只保留分类目录、索引目录和归档目录；旧路径软链接已集中移动到 `analysis/_compat_links/top_level/`。",
            "",
            "如果后续某个旧 Markdown / PPT 仍依赖旧路径，可从 `_compat_links/top_level/` 中把对应软链接临时移回 `analysis/` 顶层。",
            "",
            "## 类别计数快照",
            "",
            "```json",
            json.dumps(dict(sorted(category_counts.items())), ensure_ascii=False, indent=2),
            "```",
        ]
    )
    (OUT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    figure_lines = [
        "# analysis 关键图索引",
        "",
        "这里只列出每个目录扫描到的前若干张图，方便后续找 PPT/报告素材。",
        "",
        "| 目录 | 类别 | 示例图 |",
        "|---|---|---|",
    ]
    for row in sorted(directory_rows, key=lambda item: (str(item["category"]), str(item["name"]))):
        figures = str(row["sample_figure_paths"])
        if not figures:
            continue
        first = figures.split("|")[0]
        rel_from_index = os.path.relpath(ROOT / first, OUT_DIR)
        figure_lines.append(f"| `{row['path']}` | {row['category']} | ![]({rel_from_index}) |")
    (OUT_DIR / "key_figures.md").write_text("\n".join(figure_lines) + "\n", encoding="utf-8")


def main() -> None:
    """Build catalog files under analysis/_index."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    directory_entries = _iter_catalog_directories()
    files = sorted(
        [path for path in ANALYSIS_DIR.iterdir() if path.is_file() and not path.is_symlink()],
        key=lambda item: item.name,
    )

    directory_rows = [_scan_directory(path, forced_category=forced_category) for path, forced_category in directory_entries]
    file_rows = [_scan_top_level_file(path) for path in files]
    summary_rows = _category_summary(directory_rows)

    directory_fields = [
        "kind",
        "name",
        "path",
        "category",
        "date_token",
        "suggested_action",
        "size_bytes",
        "size_human",
        "file_count",
        "dir_count",
        "latest_mtime",
        "md_count",
        "png_count",
        "svg_count",
        "pdf_count",
        "csv_count",
        "json_count",
        "sdf_count",
        "pt_count",
        "log_count",
        "py_count",
        "report_paths",
        "readme_paths",
        "sample_figure_paths",
    ]
    file_fields = [
        "kind",
        "name",
        "path",
        "category",
        "date_token",
        "suggested_action",
        "size_bytes",
        "size_human",
        "latest_mtime",
        "suffix",
    ]
    summary_fields = [
        "category",
        "directory_count",
        "size_bytes",
        "size_human",
        "size_gb",
        "file_count",
        "report_dir_count",
        "png_count",
        "sdf_count",
        "csv_count",
    ]

    _write_csv(OUT_DIR / "analysis_catalog.csv", directory_rows, directory_fields)
    _write_csv(OUT_DIR / "top_level_files.csv", file_rows, file_fields)
    _write_csv(OUT_DIR / "category_summary.csv", summary_rows, summary_fields)
    _write_plots(directory_rows, summary_rows)
    _write_markdown(directory_rows, file_rows, summary_rows)

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "analysis_dir": str(ANALYSIS_DIR),
        "directory_count": len(directory_rows),
        "top_level_file_count": len(file_rows),
        "total_size_bytes": sum(int(row["size_bytes"]) for row in directory_rows)
        + sum(int(row["size_bytes"]) for row in file_rows),
        "category_summary": summary_rows,
        "suggested_action_counts": dict(Counter(str(row["suggested_action"]) for row in directory_rows)),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(OUT_DIR.relative_to(ROOT))


if __name__ == "__main__":
    main()
