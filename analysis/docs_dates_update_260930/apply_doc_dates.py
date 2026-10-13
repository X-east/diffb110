#!/usr/bin/env python
"""Normalise the date metadata of every document under ``docs/``.

Convention requested on 2026-09-30:

* every **document date metadata** value becomes ``2026-09-29`` (yesterday);
* the ``创建：<date>`` fragment is removed;
* only a last-updated stamp remains, spelled ``最后更新``.

Deliberately NOT touched: dates inside the running text that record facts
("E2PO 已于 2026-09-18 更名", "本文正文截至 2026-09-02", "命名说明（2026-09-30）",
experiment tables, references to ``work_records/260930/``).  Rewriting those would
falsify the record.  Run with ``--check`` to see the plan without writing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DOCS = Path("/home/md666/data/disk6/DiffB110/docs")
NEW_DATE = "2026-09-29"

# (relative path, exact old fragment, new fragment) — each must occur exactly once.
LINE_EDITS: list[tuple[str, str, str]] = [
    (
        "archive/todo_cleanup.md",
        "> **最后更新**: 2026-06-30",
        f"> **最后更新**: {NEW_DATE}",
    ),
    (
        "DiffSBDD-IQL模型训练与使用说明.md",
        "> 更新：2026-08-24。",
        f"> 最后更新：{NEW_DATE}。",
    ),
    (
        "overview/DiffGui-RL训练汇总.md",
        "> 更新：2026-09-30。以下为",
        f"> 最后更新：{NEW_DATE}。以下为",
    ),
    (
        "overview/DiffSBDD-ESRPO-ChEMBL版总审计.md",
        "> 创建：2026-09-30　状态：",
        f"> 最后更新：{NEW_DATE}　状态：",
    ),
    (
        "overview/DiffSBDD-IQL中期整体梳理.md",
        "> 日期：2026-08-17",
        f"> 最后更新：{NEW_DATE}",
    ),
    (
        "project_context.md",
        "> **最后更新**: 2026-09-30（",
        f"> **最后更新**: {NEW_DATE}（",
    ),
    (
        "project_paths.md",
        "> **最后更新**: 2026-06-18",
        f"> **最后更新**: {NEW_DATE}",
    ),
    (
        "README.md",
        "> 更新：2026-09-30　维护规则：",
        f"> 最后更新：{NEW_DATE}　维护规则：",
    ),
    (
        "workflow_commands.md",
        "> **最后更新**: 2026-08-24",
        f"> **最后更新**: {NEW_DATE}",
    ),
    (
        "管线总览.md",
        "> 更新：2026-09-30　定位：",
        f"> 最后更新：{NEW_DATE}　定位：",
    ),
    (
        "评估口径规范.md",
        "> 创建：2026-09-30　最后更新：**2026-09-30（对接流程整理 + 加氢口径确立）**",
        f"> 最后更新：**{NEW_DATE}（对接流程整理 + 加氢口径确立）**",
    ),
]

TABLE_HEADER = "| 文件 | 定位 | 最后更新 | 状态 |"
TABLE_END_MARKER = "> `work_records/` 根目录的 6 份总审计"


def rewrite_table(path: Path, text: str) -> tuple[str, int]:
    """Set the 最后更新 column of the docs index table to NEW_DATE."""
    lines = text.splitlines(keepends=True)
    start = next((i for i, line in enumerate(lines) if line.startswith(TABLE_HEADER)), None)
    if start is None:
        return text, 0
    changed = 0
    i = start + 2  # skip header + separator
    while i < len(lines):
        line = lines[i]
        if line.startswith(TABLE_END_MARKER):
            break
        cells = line.split("|")
        if len(cells) >= 5:
            value = cells[3].strip()
            if value not in ("", "—", "-", NEW_DATE):
                cells[3] = f" {NEW_DATE} "
                lines[i] = "|".join(cells)
                changed += 1
        i += 1
    return "".join(lines), changed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="dry run")
    args = parser.parse_args()

    failures: list[str] = []
    for rel, old, new in LINE_EDITS:
        path = DOCS / rel
        text = path.read_text(encoding="utf-8")
        count = text.count(old)
        if count != 1:
            failures.append(f"{rel}: expected 1 occurrence of {old!r}, found {count}")
            continue
        if args.check:
            print(f"[plan] {rel}: {old!r} -> {new!r}")
            continue
        path.write_text(text.replace(old, new), encoding="utf-8")
        print(f"[edit] {rel}")

    readme = DOCS / "README.md"
    text = readme.read_text(encoding="utf-8")
    updated, changed = rewrite_table(readme, text)
    if args.check:
        print(f"[plan] README.md: {changed} table rows -> {NEW_DATE}")
    elif changed:
        readme.write_text(updated, encoding="utf-8")
        print(f"[edit] README.md: {changed} table rows -> {NEW_DATE}")
    else:
        failures.append("README.md: index table not found or already normalised")

    if failures:
        print("\nFAILURES:")
        for item in failures:
            print(" -", item)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
