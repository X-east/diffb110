#!/usr/bin/env python
"""Apply the 260928 ChEMBL reference-number update to the thesis draft.

Replaces the stale "84.72% / n = 1 550 090" reference everywhere in
``docs/毕业论文_RE-CmeB_DiffGui强化学习_正式初稿.md`` with the recomputed
small-molecule reference (80.94% / n = 1 790 085), and rewrites the
"generative models generally fall below this level" wording to state that the
optimized generative model reaches a comparable level.

Each replacement is verified to hit exactly once; the script raises if any
anchor is missing so a partial edit is impossible.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
THESIS = REPO_ROOT / "docs" / "毕业论文_RE-CmeB_DiffGui强化学习_正式初稿.md"

# (description, old, new) — plain string replacements, must each hit exactly once.
PLAIN = [
    (
        "中文摘要",
        "以 ChEMBL 数据库（样本量 1 550 090）作为类药化学空间的外部参照，其 R1–R5 全清比例为 84.72%，而待评估的生成模型普遍低于该水平。",
        "以 ChEMBL 数据库（仅保留 Small molecule，样本量 1 790 085）作为类药化学空间的外部参照，其 R1–R5 全清比例为 80.94%，优化后的生成模型在该结构规则口径下已达到与参照相当的水平。",
    ),
    (
        "英文摘要",
        "Using the ChEMBL database (n = 1 550 090) as an external reference for drug-like chemical space, its R1-R5 all-clear fraction is 84.72%, whereas the evaluated generative models generally fall below this level.",
        "Using the ChEMBL database (small molecules only, n = 1 790 085) as an external reference for drug-like chemical space, its R1-R5 all-clear fraction is 80.94%, and the optimized generative model reaches a level comparable to this reference.",
    ),
    (
        "第三章正文-参照集规模",
        "参照集包含 1 550 090 个类药分子，",
        "参照集仅保留官方标记为 Small molecule 的记录，共 1 790 085 个类药分子，",
    ),
    (
        "第三章正文-single fragment 分母",
        "参照集的 relaxed 分子数等于全量 1 550 090，",
        "参照集的 relaxed 分子数等于全量 1 790 085，",
    ),
    (
        "结论章",
        "其 R1–R5 全清比例为 84.72%（样本量 1 550 090），而待评估的生成模型普遍低于该水平。",
        "其 R1–R5 全清比例为 80.94%（仅保留 Small molecule，样本量 1 790 085），优化后的生成模型在该结构规则口径下已达到与参照相当的水平。",
    ),
    (
        "创新点",
        "引入 ChEMBL 数据库（1 550 090 个类药分子）作为类药化学空间的外部参照",
        "引入 ChEMBL 数据库（仅保留 Small molecule，1 790 085 个类药分子）作为类药化学空间的外部参照",
    ),
]

# Table row in §3.3.1 — spaces are hard to match, so replace the whole row by regex.
TABLE_OLD = r"\| ChEMBL36 参照 \| 1 550 090 \|[^\n]*"
TABLE_NEW = ("| ChEMBL 参照    | 1 790 085 |              — |                   — |"
             "   6.19% |   6.33% |     6.76% |  14.22% | 19.06% |")


def main() -> None:
    """Apply all replacements with exactly-once verification."""
    text = THESIS.read_text(encoding="utf-8")
    original = text

    for desc, old, new in PLAIN:
        count = text.count(old)
        if count != 1:
            sys.exit(f"[FAIL] '{desc}': expected 1 match, found {count}")
        text = text.replace(old, new)
        print(f"[ok] {desc}: replaced 1")

    table_matches = re.findall(TABLE_OLD, text)
    if len(table_matches) != 1:
        sys.exit(f"[FAIL] '§3.3.1 table row': expected 1 match, found {len(table_matches)}")
    text = re.sub(TABLE_OLD, TABLE_NEW, text)
    print("[ok] §3.3.1 table row: replaced 1")

    if text == original:
        sys.exit("[FAIL] no change produced")
    THESIS.write_text(text, encoding="utf-8")
    print(f"[done] wrote {THESIS}")

    # Post-condition: no stale reference numbers remain.
    for stale in ("84.72", "1 550 090", "1,550,090"):
        n = text.count(stale)
        print(f"[check] remaining '{stale}': {n}")
        if n:
            sys.exit(f"[FAIL] stale token still present: {stale}")


if __name__ == "__main__":
    main()
