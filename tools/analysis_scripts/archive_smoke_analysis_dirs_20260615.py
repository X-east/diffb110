#!/usr/bin/env python
"""Archive smoke/debug analysis directories without deleting evidence."""

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
ARCHIVE_ROOT = ANALYSIS_DIR / "_archive" / "smoke"
MANIFEST_CSV = ARCHIVE_ROOT / "archive_smoke_manifest_20260615.csv"
MANIFEST_JSON = ARCHIVE_ROOT / "archive_smoke_manifest_20260615.json"
SUMMARY_MD = ARCHIVE_ROOT / "archive_smoke_summary_20260615.md"

REFERENCE_ROOTS = (
    ROOT / "docs",
    ROOT / "work_records",
    ROOT / "work_days",
    ROOT / "launch",
    ROOT / "tools",
)
TEXT_SUFFIXES = {
    ".md",
    ".txt",
    ".py",
    ".sh",
    ".yaml",
    ".yml",
    ".json",
    ".csv",
    ".html",
}


@dataclass(frozen=True)
class ArchiveItem:
    """One analysis directory selected for smoke archive."""

    name: str
    old_rel: str
    target_rel: str
    size_human: str


def _load_archive_items() -> list[ArchiveItem]:
    """Load archive_smoke entries from the generated catalog."""
    items: list[ArchiveItem] = []
    with CATALOG_PATH.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["suggested_action"] != "archive_smoke":
                continue
            old_rel = row["path"]
            name = Path(old_rel).name
            items.append(
                ArchiveItem(
                    name=name,
                    old_rel=old_rel,
                    target_rel=(Path("analysis/_archive/smoke") / name).as_posix(),
                    size_human=row["size_human"],
                )
            )
    return items


def _iter_reference_files() -> list[Path]:
    """Return text files that may contain references to old analysis paths."""
    files: list[Path] = []
    for root in REFERENCE_ROOTS:
        if not root.exists():
            continue
        for path, dirs, filenames in os.walk(root):
            path_obj = Path(path)
            dirs[:] = [
                name
                for name in dirs
                if name not in {".git", "__pycache__"}
                and not (path_obj / name).is_symlink()
            ]
            for filename in filenames:
                file_path = path_obj / filename
                if file_path.suffix.lower() in TEXT_SUFFIXES:
                    files.append(file_path)
    return files


def _find_references(items: list[ArchiveItem]) -> dict[str, list[str]]:
    """Find text references to the selected smoke analysis paths."""
    refs: dict[str, list[str]] = {item.old_rel: [] for item in items}
    needles = {item.old_rel: (item.old_rel, item.name) for item in items}
    for file_path in _iter_reference_files():
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        rel_file = file_path.relative_to(ROOT).as_posix()
        for old_rel, (full_path, dirname) in needles.items():
            if full_path in text or dirname in text:
                refs[old_rel].append(rel_file)
    return refs


def _relative_symlink_target(old_path: Path, target_path: Path) -> Path:
    """Compute a relative symlink target from old path's parent."""
    return Path(os.path.relpath(target_path, start=old_path.parent))


def _archive_item(item: ArchiveItem, references: list[str]) -> dict[str, object]:
    """Move one directory and optionally create a compatibility symlink."""
    old_path = ROOT / item.old_rel
    target_path = ROOT / item.target_rel
    symlink_created = False
    action = "missing"
    note = ""

    if old_path.is_symlink():
        action = "already_symlink"
        note = os.readlink(old_path)
    elif target_path.exists() and not old_path.exists():
        action = "already_archived"
    elif target_path.exists() and old_path.exists():
        action = "target_exists_skipped"
        note = "target and old path both exist"
    elif old_path.exists():
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old_path), str(target_path))
        action = "moved"
        if references:
            old_path.symlink_to(_relative_symlink_target(old_path, target_path), target_is_directory=True)
            symlink_created = True
    else:
        note = "old path not found"

    return {
        "name": item.name,
        "old_path": item.old_rel,
        "archive_path": item.target_rel,
        "size_human": item.size_human,
        "action": action,
        "symlink_created": symlink_created,
        "reference_count": len(references),
        "reference_samples": "|".join(references[:12]),
        "note": note,
    }


def _write_csv(rows: list[dict[str, object]]) -> None:
    """Write archive manifest CSV."""
    fields = [
        "name",
        "old_path",
        "archive_path",
        "size_human",
        "action",
        "symlink_created",
        "reference_count",
        "reference_samples",
        "note",
    ]
    with MANIFEST_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(rows: list[dict[str, object]]) -> None:
    """Write a Markdown summary for the archive operation."""
    moved = [row for row in rows if row["action"] == "moved"]
    linked = [row for row in rows if row["symlink_created"]]
    skipped = [row for row in rows if row["action"] not in {"moved", "already_archived", "already_symlink"}]
    lines = [
        "# smoke/debug analysis 目录归档记录",
        "",
        f"- 执行时间：{datetime.now().isoformat(timespec='seconds')}",
        f"- 来源清单：`{CATALOG_PATH.relative_to(ROOT).as_posix()}`",
        f"- 归档目标：`{ARCHIVE_ROOT.relative_to(ROOT).as_posix()}`",
        f"- 处理条目：{len(rows)}",
        f"- 已移动：{len(moved)}",
        f"- 创建旧路径软链接：{len(linked)}",
        f"- 需要复查/跳过：{len(skipped)}",
        "",
        "## 说明",
        "",
        "- 本次只归档 `archive_smoke` 目录，不移动正式结果目录。",
        "- 若目录在 `docs/work_records/work_days/launch/tools` 中被文本引用，会在旧位置保留软链接。",
        "- 后续重建 `analysis/_index` 时，索引脚本会跳过顶层软链接，避免重复统计。",
        "",
        "## 移动清单",
        "",
        "| 旧路径 | 新路径 | 大小 | 软链接 | 引用数 |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['old_path']}` | `{row['archive_path']}` | {row['size_human']} | "
            f"{'是' if row['symlink_created'] else '否'} | {row['reference_count']} |"
        )
    if skipped:
        lines.extend(["", "## 需要复查", "", "| 路径 | 动作 | 说明 |", "|---|---|---|"])
        for row in skipped:
            lines.append(f"| `{row['old_path']}` | {row['action']} | {row['note']} |")
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Archive smoke/debug directories and write a manifest."""
    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
    items = _load_archive_items()
    if not items:
        print("No archive_smoke entries found in current catalog; existing archive manifest was not modified.")
        return
    references = _find_references(items)
    rows = [_archive_item(item, references[item.old_rel]) for item in items]
    _write_csv(rows)
    _write_summary(rows)
    MANIFEST_JSON.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "archive_root": ARCHIVE_ROOT.relative_to(ROOT).as_posix(),
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
