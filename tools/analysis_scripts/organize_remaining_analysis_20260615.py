#!/usr/bin/env python
"""Move remaining analysis artifacts into category folders with compatibility links."""

from __future__ import annotations

import csv
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = ROOT / "analysis"
CATALOG_PATH = ANALYSIS_DIR / "_index" / "analysis_catalog.csv"
TOP_FILES_PATH = ANALYSIS_DIR / "_index" / "top_level_files.csv"
OUT_DIR = ANALYSIS_DIR / "_index"
MANIFEST_CSV = OUT_DIR / "organize_remaining_manifest_20260615.csv"
MANIFEST_JSON = OUT_DIR / "organize_remaining_manifest_20260615.json"
SUMMARY_MD = OUT_DIR / "organize_remaining_summary_20260615.md"

CATEGORY_DIRS = {
    "benchmark",
    "classifier",
    "diffgui_rl",
    "diffsbdd_rl",
    "diffsbdd_small_mod",
    "diffsbdd_xhe",
    "preprocessing",
    "reconstruction",
}
INTERNAL_DIRS = {"_archive", "_index", "_loose", "__pycache__"}


@dataclass(frozen=True)
class MoveItem:
    """One artifact selected for organization."""

    kind: str
    name: str
    old_rel: str
    target_rel: str
    category: str
    size_human: str


def _safe_category(category: str) -> str:
    """Return a usable category for destination paths."""
    if category in CATEGORY_DIRS:
        return category
    if category in {"internal", "smoke_or_debug"}:
        return category
    return "misc"


def _relative_symlink_target(old_path: Path, target_path: Path) -> Path:
    """Compute relative symlink target from old path's parent."""
    return Path(os.path.relpath(target_path, start=old_path.parent))


def _load_directory_items() -> list[MoveItem]:
    """Load remaining non-internal real analysis directories from the catalog."""
    items: list[MoveItem] = []
    with CATALOG_PATH.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            category = row["category"]
            if category == "internal":
                continue
            old_rel = row["path"]
            old_path = ROOT / old_rel
            if old_path.is_symlink() or not old_path.exists():
                continue
            if Path(old_rel).parent != Path("analysis"):
                continue
            target_rel = (Path("analysis") / _safe_category(category) / row["name"]).as_posix()
            if old_rel == target_rel:
                continue
            items.append(
                MoveItem(
                    kind="directory",
                    name=row["name"],
                    old_rel=old_rel,
                    target_rel=target_rel,
                    category=category,
                    size_human=row["size_human"],
                )
            )
    return items


def _load_file_items() -> list[MoveItem]:
    """Load remaining real top-level analysis files."""
    if not TOP_FILES_PATH.exists():
        return []
    items: list[MoveItem] = []
    with TOP_FILES_PATH.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            old_rel = row["path"]
            old_path = ROOT / old_rel
            if old_path.is_symlink() or not old_path.exists():
                continue
            category = _safe_category(row["category"])
            suffix = row.get("suffix", "")
            if suffix == ".py":
                target_rel = (Path("tools/analysis_scripts/legacy_top_level") / row["name"]).as_posix()
            else:
                target_rel = (Path("analysis/_loose") / category / row["name"]).as_posix()
            items.append(
                MoveItem(
                    kind="file",
                    name=row["name"],
                    old_rel=old_rel,
                    target_rel=target_rel,
                    category=category,
                    size_human=row["size_human"],
                )
            )
    return items


def _move_item(item: MoveItem) -> dict[str, object]:
    """Move one file/directory and leave a compatibility symlink."""
    old_path = ROOT / item.old_rel
    target_path = ROOT / item.target_rel
    action = "missing"
    symlink_created = False
    note = ""

    if old_path.is_symlink():
        action = "already_symlink"
        note = os.readlink(old_path)
    elif target_path.exists() and not old_path.exists():
        action = "already_moved"
    elif target_path.exists() and old_path.exists():
        action = "target_exists_skipped"
        note = "target and old path both exist"
    elif old_path.exists():
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old_path), str(target_path))
        old_path.symlink_to(
            _relative_symlink_target(old_path, target_path),
            target_is_directory=target_path.is_dir(),
        )
        action = "moved"
        symlink_created = True
    else:
        note = "old path not found"

    return {
        "kind": item.kind,
        "name": item.name,
        "category": item.category,
        "old_path": item.old_rel,
        "target_path": item.target_rel,
        "size_human": item.size_human,
        "action": action,
        "symlink_created": symlink_created,
        "note": note,
    }


def _write_csv(rows: list[dict[str, object]]) -> None:
    """Write move manifest CSV."""
    fields = [
        "kind",
        "name",
        "category",
        "old_path",
        "target_path",
        "size_human",
        "action",
        "symlink_created",
        "note",
    ]
    with MANIFEST_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(rows: list[dict[str, object]]) -> None:
    """Write Markdown summary."""
    moved_dirs = [row for row in rows if row["kind"] == "directory" and row["action"] == "moved"]
    moved_files = [row for row in rows if row["kind"] == "file" and row["action"] == "moved"]
    skipped = [row for row in rows if row["action"] not in {"moved", "already_moved", "already_symlink"}]
    categories = sorted({str(row["category"]) for row in rows})
    lines = [
        "# analysis 剩余内容分类整理记录",
        "",
        f"- 执行时间：{datetime.now().isoformat(timespec='seconds')}",
        f"- 移动目录：{len(moved_dirs)}",
        f"- 移动顶层文件：{len(moved_files)}",
        f"- 创建兼容软链接：{sum(bool(row['symlink_created']) for row in rows)}",
        f"- 跳过/需复查：{len(skipped)}",
        "",
        "## 目标结构",
        "",
        "```text",
        "analysis/",
        "  benchmark/",
        "  classifier/",
        "  diffgui_rl/",
        "  diffsbdd_rl/",
        "  diffsbdd_small_mod/",
        "  diffsbdd_xhe/",
        "  preprocessing/",
        "  reconstruction/",
        "  _archive/",
        "  _index/",
        "  _loose/",
        "tools/analysis_scripts/legacy_top_level/",
        "```",
        "",
        "## 说明",
        "",
        "- 本次没有删除任何内容。",
        "- 所有被移动的顶层目录/文件，旧位置都保留软链接，旧报告和旧命令仍可打开。",
        "- `analysis/_index/analysis_catalog.csv` 会按分类后的真实目录重新生成。",
        "",
        "## 分类目录",
        "",
        "| 类别 | 是否处理 |",
        "|---|---|",
    ]
    for category in categories:
        lines.append(f"| {category} | 是 |")
    lines.extend(["", "## 移动目录清单", "", "| 旧路径 | 新路径 | 类别 | 大小 |", "|---|---|---|---:|"])
    for row in moved_dirs:
        lines.append(f"| `{row['old_path']}` | `{row['target_path']}` | {row['category']} | {row['size_human']} |")
    lines.extend(["", "## 移动文件清单", "", "| 旧路径 | 新路径 | 类别 | 大小 |", "|---|---|---|---:|"])
    for row in moved_files:
        lines.append(f"| `{row['old_path']}` | `{row['target_path']}` | {row['category']} | {row['size_human']} |")
    if skipped:
        lines.extend(["", "## 需要复查", "", "| 旧路径 | 目标路径 | 动作 | 说明 |", "|---|---|---|---|"])
        for row in skipped:
            lines.append(f"| `{row['old_path']}` | `{row['target_path']}` | {row['action']} | {row['note']} |")
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Organize remaining analysis artifacts."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    items = _load_directory_items() + _load_file_items()
    if not items:
        print("No remaining real top-level analysis artifacts to organize.")
        return
    rows = [_move_item(item) for item in items]
    _write_csv(rows)
    _write_summary(rows)
    MANIFEST_JSON.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "rows": rows,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(SUMMARY_MD.relative_to(ROOT))


if __name__ == "__main__":
    main()
