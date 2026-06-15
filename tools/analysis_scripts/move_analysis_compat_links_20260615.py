#!/usr/bin/env python
"""Move top-level compatibility links out of analysis/."""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = ROOT / "analysis"
OUT_DIR = ANALYSIS_DIR / "_index"
COMPAT_DIR = ANALYSIS_DIR / "_compat_links" / "top_level"
CACHE_DIR = ANALYSIS_DIR / "_archive" / "cache"
MANIFEST_CSV = OUT_DIR / "move_compat_links_manifest_20260615.csv"
MANIFEST_JSON = OUT_DIR / "move_compat_links_manifest_20260615.json"
SUMMARY_MD = OUT_DIR / "move_compat_links_summary_20260615.md"

KEEP_TOP_LEVEL = {
    "_archive",
    "_compat_links",
    "_index",
    "_loose",
    "benchmark",
    "classifier",
    "diffgui_rl",
    "diffsbdd_rl",
    "diffsbdd_small_mod",
    "diffsbdd_xhe",
    "preprocessing",
    "reconstruction",
}


def _readlink(path: Path) -> str:
    """Return a symlink target as text."""
    return os.readlink(path)


def _move_symlink(path: Path) -> dict[str, object]:
    """Move one top-level symlink into the compatibility link directory."""
    target = _readlink(path)
    destination = COMPAT_DIR / path.name
    row: dict[str, object] = {
        "kind": "symlink",
        "name": path.name,
        "old_path": path.relative_to(ROOT).as_posix(),
        "new_path": destination.relative_to(ROOT).as_posix(),
        "target": target,
        "action": "pending",
        "note": "",
    }
    if destination.exists() or destination.is_symlink():
        row["action"] = "skipped_destination_exists"
        return row
    destination.parent.mkdir(parents=True, exist_ok=True)
    path.rename(destination)
    row["action"] = "moved"
    return row


def _move_pycache() -> dict[str, object] | None:
    """Move analysis/__pycache__ into the archive cache folder if present."""
    path = ANALYSIS_DIR / "__pycache__"
    if not path.exists() or path.is_symlink():
        return None
    destination = CACHE_DIR / "analysis_pycache_20260615"
    row: dict[str, object] = {
        "kind": "directory",
        "name": "__pycache__",
        "old_path": path.relative_to(ROOT).as_posix(),
        "new_path": destination.relative_to(ROOT).as_posix(),
        "target": "",
        "action": "pending",
        "note": "",
    }
    if destination.exists():
        row["action"] = "skipped_destination_exists"
        return row
    destination.parent.mkdir(parents=True, exist_ok=True)
    path.rename(destination)
    row["action"] = "moved"
    return row


def _repair_compat_links() -> list[dict[str, object]]:
    """Repair moved symlinks so they resolve from the compatibility folder."""
    rows: list[dict[str, object]] = []
    if not COMPAT_DIR.exists():
        return rows
    for path in sorted(COMPAT_DIR.iterdir(), key=lambda item: item.name):
        if not path.is_symlink():
            continue
        current_target = _readlink(path)
        current_resolved = (path.parent / current_target).resolve(strict=False)
        row: dict[str, object] = {
            "kind": "symlink_repair",
            "name": path.name,
            "old_path": path.relative_to(ROOT).as_posix(),
            "new_path": path.relative_to(ROOT).as_posix(),
            "target": current_target,
            "action": "already_ok",
            "note": "",
        }
        if current_resolved.exists():
            rows.append(row)
            continue

        old_resolved = (ANALYSIS_DIR / current_target).resolve(strict=False)
        repaired_target = os.path.relpath(old_resolved, start=path.parent)
        path.unlink()
        path.symlink_to(repaired_target, target_is_directory=old_resolved.is_dir())
        row["target"] = repaired_target
        row["action"] = "repaired"
        row["note"] = f"old_target={current_target}"
        rows.append(row)
    return rows


def _write_csv(rows: list[dict[str, object]]) -> None:
    """Write a manifest CSV."""
    fields = ["kind", "name", "old_path", "new_path", "target", "action", "note"]
    with MANIFEST_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(rows: list[dict[str, object]]) -> None:
    """Write a short Markdown summary."""
    moved_links = [row for row in rows if row["kind"] == "symlink" and row["action"] == "moved"]
    moved_dirs = [row for row in rows if row["kind"] == "directory" and row["action"] == "moved"]
    repaired_links = [row for row in rows if row["kind"] == "symlink_repair" and row["action"] == "repaired"]
    skipped = [row for row in rows if row["action"] not in {"moved", "repaired", "already_ok"}]
    lines = [
        "# analysis 顶层兼容软链接移动记录",
        "",
        f"- 执行时间：{datetime.now().isoformat(timespec='seconds')}",
        f"- 移动顶层软链接：{len(moved_links)}",
        f"- 修正兼容软链接：{len(repaired_links)}",
        f"- 移动缓存目录：{len(moved_dirs)}",
        f"- 跳过/需复查：{len(skipped)}",
        "",
        "## 目标",
        "",
        "让 `analysis/` 顶层只保留分类目录、索引目录和归档目录；旧路径软链接集中放到 `analysis/_compat_links/top_level/`。",
        "",
        "## 说明",
        "",
        "- 本次没有删除真实结果文件。",
        "- 被移动的是旧路径软链接，因此历史旧路径不再直接可用。",
        "- 如需恢复某个旧路径，可从 `_compat_links/top_level/` 移回对应软链接。",
        "",
        "## 移动清单",
        "",
        "| 类型 | 旧路径 | 新路径 | 原指向 | 动作 |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['kind']} | `{row['old_path']}` | `{row['new_path']}` | "
            f"`{row['target']}` | {row['action']} |"
        )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Move top-level compatibility links and cache artifacts."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for path in sorted(ANALYSIS_DIR.iterdir(), key=lambda item: item.name):
        if path.name in KEEP_TOP_LEVEL:
            continue
        if path.is_symlink():
            rows.append(_move_symlink(path))
    cache_row = _move_pycache()
    if cache_row is not None:
        rows.append(cache_row)
    rows.extend(_repair_compat_links())
    _write_csv(rows)
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
    _write_summary(rows)
    moved = sum(1 for row in rows if row["action"] == "moved")
    repaired = sum(1 for row in rows if row["action"] == "repaired")
    already_ok = sum(1 for row in rows if row["action"] == "already_ok")
    skipped = len(rows) - moved - repaired - already_ok
    print(
        f"moved={moved} repaired={repaired} already_ok={already_ok} "
        f"skipped={skipped} summary={SUMMARY_MD.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
