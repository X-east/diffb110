"""Build a deduplicated thematic reference library with dated names and provenance."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
from urllib.parse import quote

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


THEMES = {
    "D": "01_扩散生成与结构设计",
    "P": "02_强化学习与偏好优化",
    "R": "03_表示学习与序列生成",
    "E": "04_评价指标与奖励设计",
    "B": "05_数据集与基准评测",
    "S": "06_综述与研究资料",
    "A": "07_应用案例与靶点设计",
    "G": "08_框架图与演示素材",
}
THEME_DESCRIPTIONS = {
    "D": "结构条件生成、等变扩散、药效团/形状/相互作用引导，以及结构生成对照模型。",
    "P": "分子强化学习、扩散模型微调、偏好优化、样本效率与多目标优化。",
    "R": "三维表征、SMILES/片段表示、化学语言模型与迁移学习。",
    "E": "独立评价、奖励设计、多样性、奖励黑客与偏好打分。",
    "B": "数据集、泛化能力、基准和分析工具。",
    "S": "领域综述、学位论文和检索报告；检索报告与论文分开注明。",
    "A": "具体靶点、循环肽、抗生素与先导化合物应用。",
    "G": "参考项目里的框架图素材；图名年份沿用配套方法年，不代表图的制作日期。",
}


def read_tsv(path: Path) -> list[dict]:
    """Read a reviewed classification table."""
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def safe_name(name: str, max_bytes: int = 235) -> str:
    """Keep names within Linux component limits without breaking UTF-8."""
    name = re.sub(r"[/\\\x00]", "_", name)
    while len(name.encode("utf-8")) > max_bytes:
        name = name[:-1]
    return name


def link(label: str, target: str | Path) -> str:
    """Create an escaped relative Markdown link."""
    return f"[{label.replace('|', '／')}]({quote(str(target), safe='/')})"


def digest(path: Path) -> str:
    """Hash a local file without loading large archives into memory."""
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def archived_path(source: str, moves: list[dict]) -> str:
    """Resolve an original path through the planned archival moves."""
    for move in moves:
        if source == move["old"] or source.startswith(move["old"] + "/"):
            return move["new"] + source[len(move["old"]):]
    return source


def build_plan(root: Path, audit: Path) -> dict:
    """Prepare a reviewable layout and catalog using explicit local year evidence."""
    inventory = json.loads((audit / "inventory_before.json").read_text(encoding="utf-8"))
    paper_specs = read_tsv(audit / "papers.tsv")
    unique = {}
    for row in inventory["pdfs"]:
        unique.setdefault(row["sha256"], row)
    ordered = list(unique.values())
    if len(paper_specs) != len(ordered):
        raise ValueError("Classification table does not cover every distinct PDF")
    specs_by_hash = {}
    for spec in paper_specs:
        row = ordered[int(spec["id"]) - 1]
        specs_by_hash[row["sha256"]] = {**spec, "sha256":row["sha256"]}
    archive = "99_原始资料归档"
    moves = [{"old":"AI药物设计-XDC", "new":f"{archive}/知识库原始解压/AI药物设计-XDC"}]
    for row in inventory["top_level_files"]:
        suffix = Path(row["path"]).suffix.lower()
        if suffix not in {".pdf", ".zip"}:
            continue
        bucket = "原始压缩包" if suffix == ".zip" else "原根目录散放论文"
        moves.append({"old":row["path"], "new":f"{archive}/{bucket}/{row['path']}"})
    groups = defaultdict(list)
    for row in inventory["pdfs"]:
        spec = specs_by_hash[row["sha256"]]
        doi = row["doi_candidates"][0].rstrip(".;,)").lower() if row["doi_candidates"] else ""
        kind = "supplement" if spec["kind"] == "补充材料" else "article"
        key = f"doi:{doi}:{kind}" if doi and spec["theme"] != "G" else "sha:" + row["sha256"]
        groups[key].append({**row, "spec":spec, "doi":doi})
    papers = []
    for key, rows in groups.items():
        spec = rows[0]["spec"]
        for row in rows:
            if (row["spec"]["label"], row["spec"]["year"], row["spec"]["theme"]) != (spec["label"], spec["year"], spec["theme"]):
                raise ValueError(f"Conflicting classifications for {key}")
        # All variants are retained; a parseable large copy is the default reading entry.
        selected = max(rows, key=lambda row: (row["parse_ok"], row["bytes"], -len(row["path"])))
        prefix = "年份待核实" if spec["year"] == "待核实" else spec["year"]
        bucket = "素材" if spec["theme"] == "G" else "论文"
        output = f"{THEMES[spec['theme']]}/{bucket}/{safe_name(prefix + '_' + spec['label'])}.pdf"
        sources = [{"original_path":r["path"], "archive_path":archived_path(r["path"], moves),
                    "sha256":r["sha256"], "bytes":r["bytes"], "scan_id":r["spec"]["id"]} for r in rows]
        paper = {"id":f"P{spec['id'].zfill(3)}", "theme":spec["theme"], "year":spec["year"],
                 "kind":spec["kind"], "label":spec["label"], "doi":selected["doi"],
                 "reading_path":output, "source_path":archived_path(selected["path"], moves),
                 "sha256":selected["sha256"], "bytes":selected["bytes"], "sources":sources,
                 "unique_byte_versions":len({r["sha256"] for r in rows}),
                 "year_evidence":selected["year_evidence_lines"],
                 "evidence_text":f"pdf_first_pages/{selected['sha256']}.txt"}
        papers.append(paper)
    by_scan_id = {source["scan_id"]:paper["id"] for paper in papers for source in paper["sources"]}
    project_specs = read_tsv(audit / "projects.tsv")
    if {r["path"] for r in project_specs} != {r["path"] for r in inventory["repositories"]}:
        raise ValueError("Project classifications do not match the full inventory")
    projects = []
    for row in project_specs:
        name = Path(row["path"]).name
        prefix = "年份待核实" if row["year"] == "待核实" else row["year"]
        reading = f"{THEMES[row['theme']]}/参考代码/{prefix}_{name}"
        projects.append({**row, "name":name, "reading_path":reading,
                         "paper_links":[by_scan_id[s] for s in row["paper_ids"].split(",") if s]})
    old_audit = json.loads((audit.parent / "knowledgebase_import_261012/integrity.json").read_text(encoding="utf-8"))
    notes = []
    for i, row in enumerate((r for r in old_audit["files_detail"] if r["path"].endswith(".md")), 1):
        broken = not row.get("markdown_content_ok", False)
        original = "AI药物设计-XDC/" + row["path"]
        if broken:
            reading = "90_待补正文/" + safe_name(f"{i:02d}_缺正文_" + Path(row["path"]).stem, 225) + ".md"
        else:
            reading = f"{THEMES['D']}/笔记/2024_KGDiff_知乎提取笔记.md"
        notes.append({"original_path":original, "source_path":archived_path(original, moves),
                      "reading_path":reading, "sha256":row["sha256"], "missing_body":broken})
    outputs = [r["reading_path"] for r in papers + projects + notes]
    if len(outputs) != len(set(outputs)):
        raise ValueError("Reading-entry filename collision")
    counts = Counter(p["theme"] for p in papers if p["theme"] != "G")
    summary = {"original_pdf_files":len(inventory["pdfs"]),
               "original_document_files":sum(len(p["sources"]) for p in papers if p["theme"] != "G"),
               "document_reading_entries":sum(p["theme"] != "G" for p in papers),
               "figure_reading_entries":sum(p["theme"] == "G" for p in papers),
               "removed_duplicate_reading_entries":sum(len(p["sources"]) - 1 for p in papers if p["theme"] != "G"),
               "duplicate_document_groups":sum(len(p["sources"]) > 1 and p["theme"] != "G" for p in papers),
               "different_byte_same_doi_groups":sum(p["unique_byte_versions"] > 1 for p in papers),
               "code_projects":len(projects), "missing_article_bodies":sum(r["missing_body"] for r in notes),
               "documents_with_year":sum(p["theme"] != "G" and p["year"] != "待核实" for p in papers),
               "documents_year_pending":sum(p["theme"] != "G" and p["year"] == "待核实" for p in papers),
               "projects_year_pending":sum(p["year"] == "待核实" for p in projects),
               "document_counts_by_theme":dict(counts)}
    return {"root":str(root), "library":"主题资料库", "archive":archive,
            "moves":moves, "papers":papers, "projects":projects, "notes":notes, "summary":summary}


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write a spreadsheet-readable provenance table."""
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = list(rows[0]) if rows else []
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def create_link(staging: Path, final: Path, reading: str, source: Path) -> None:
    """Create a relative symlink computed for the final location."""
    destination = staging / reading
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(os.path.relpath(source, (final / reading).parent), target_is_directory=source.is_dir())


def write_indexes(plan: dict, staging: Path, audit: Path) -> None:
    """Write thematic indexes, duplicate lists, pending years, and a root entry point."""
    root = Path(plan["root"])
    final = root / plan["library"]
    summary = plan["summary"]
    papers_by_id = {p["id"]:p for p in plan["papers"]}
    for theme, folder in THEMES.items():
        directory = staging / folder
        directory.mkdir(parents=True, exist_ok=True)
        paper_rows = []
        for p in sorted((p for p in plan["papers"] if p["theme"] == theme), key=lambda p:(p["year"], p["label"])):
            relative = str(Path(p["reading_path"]).relative_to(folder))
            associated = [r["name"] for r in plan["projects"] if p["id"] in r["paper_links"]]
            paper_rows.append(f"| {p['year']} | {link(p['label'], relative)} | {p['kind']} | {len(p['sources'])} | {'、'.join(associated) or '—'} |")
        code_rows = []
        for project in sorted((r for r in plan["projects"] if r["theme"] == theme), key=lambda r:(r["year"], r["name"])):
            references = [link(papers_by_id[p]["label"].split("_")[0], "../" + papers_by_id[p]["reading_path"]) for p in project["paper_links"]]
            code_rows.append(f"| {project['year']} | {link(project['name'], str(Path(project['reading_path']).relative_to(folder)))} | {'、'.join(references) or '见项目README'} | {project['note']} |")
        text = f"# {folder[3:]}\n\n{THEME_DESCRIPTIONS[theme]}\n\n[返回总索引](../README.md)\n\n"
        if paper_rows:
            text += "## 论文与资料\n\n| 年份 | 条目 | 版本/年份口径 | 原件数 | 关联代码 |\n|---|---|---|---:|---|\n" + "\n".join(paper_rows) + "\n\n"
        if code_rows:
            text += "## 参考代码\n\n| 年份 | 项目 | 关联论文 | 说明 |\n|---|---|---|---|\n" + "\n".join(code_rows) + "\n\n"
        if theme == "D":
            text += "## 网页笔记\n\n[2024 KGDiff 知乎提取笔记](笔记/2024_KGDiff_知乎提取笔记.md)：年份为笔记声明的发布日期；图表和超链接已省略。\n"
        (directory / "README.md").write_text(text, encoding="utf-8")
    catalog_rows = []
    versions = []
    manifest = []
    for p in plan["papers"]:
        catalog_rows.append({"id":p["id"], "theme":THEMES[p["theme"]], "year":p["year"], "year_kind":p["kind"],
                             "label":p["label"], "reading_path":p["reading_path"], "doi":p["doi"],
                             "source_count":len(p["sources"]), "byte_versions":p["unique_byte_versions"],
                             "sha256":p["sha256"], "year_evidence_text":p["evidence_text"]})
        for source in p["sources"]:
            item = {"catalog_id":p["id"], "label":p["label"], "year":p["year"],
                    "reading_path":p["reading_path"], **source}
            manifest.append(item)
            if len(p["sources"]) > 1:
                versions.append(item)
    write_csv(staging / "论文总表.csv", catalog_rows)
    write_csv(staging / "原件路径对照.csv", manifest)
    write_csv(staging / "重复与版本对照.csv", versions)
    write_csv(staging / "代码项目总表.csv", [{"name":r["name"], "year":r["year"], "theme":THEMES[r["theme"]],
                                             "source_path":r["path"], "reading_path":r["reading_path"], "note":r["note"]} for r in plan["projects"]])
    duplicates = ["# 重复与版本对照", "", "SHA-256 相同表示字节完全相同；同 DOI 且同为正文的资料归为一个论文条目。补充材料独立保留。预印本与正式版在本次目录中分别保留。原件全部保留，未删除重复文件。", ""]
    for p in plan["papers"]:
        if len(p["sources"]) <= 1:
            continue
        duplicates += [f"## {p['year']} {p['label']}", "", f"阅读入口：{link(p['label'], p['reading_path'])}。原件 {len(p['sources'])} 份，{p['unique_byte_versions']} 个不同字节版本。", ""]
        for source in p["sources"]:
            duplicates.append(f"- {link(source['original_path'], '../' + source['archive_path'])}；SHA-256 `{source['sha256'][:12]}`。")
        duplicates.append("")
    (staging / "重复与版本对照.md").write_text("\n".join(duplicates), encoding="utf-8")
    pending = ["# 年份与内容待核实", "", "本索引年份以现有 PDF 的期刊/会议引文、明确日期或项目 README 为依据；未开展网络逐篇引文核验。依赖版本、目录日期、文件修改时间和 DOI 投稿编号不作发表年份。", "", "## 未确认年份", "", "| 类型 | 条目 | 原因 |", "|---|---|---|"]
    for p in plan["papers"]:
        if p["year"] == "待核实":
            pending.append(f"| 文献 | {link(p['label'], p['reading_path'])} | 文件首页缺少可确认的年份 |")
    for project in plan["projects"]:
        if project["year"] == "待核实":
            pending.append(f"| 源码 | {link(project['name'], project['reading_path'])} | {project['note']} |")
    pending += ["", "## 原始导出缺正文", "", "以下 31 个 MD 实际是微信验证 HTML 页面，不能作为文章正文使用。", ""]
    broken_dir = staging / "90_待补正文"
    broken_dir.mkdir(exist_ok=True)
    broken_lines = ["# 待补正文", "", "这些链接指向原始导出中的验证页面，仅保留文件线索。需要从源知识库重新导出正文。", ""]
    for note in plan["notes"]:
        if note["missing_body"]:
            pending.append("- " + link(Path(note["original_path"]).name, note["reading_path"]))
            broken_lines.append("- " + link(Path(note["original_path"]).name, Path(note["reading_path"]).name))
    (broken_dir / "README.md").write_text("\n".join(broken_lines) + "\n", encoding="utf-8")
    (staging / "年份与内容待核实.md").write_text("\n".join(pending) + "\n", encoding="utf-8")
    corrections = """# 年份与名称校正说明

仅规范阅读入口名称，原始文件名和内容留在归档中。年份证据来自本地PDF/README，首页提取文本保存在整理审计目录。

| 条目 | 原名称问题 | 统一口径 |
|---|---|---|
| Uni-Mol | 一份文件标为2022 | 本地文件明确为ICLR2023会议版；不混用首次预印本年 |
| 从头药物设计中的扩散模型综述 | 一份标为2025-JCIM | PDF引文为JCIM2024，64，7238–7256 |
| 化学中的深度强化学习综述 | 一份标为2023 | 两份同DOI正文均为2024年；2023为接收前的投稿时间 |
| From Theory to Therapy | 原名含2025-arxiv | 本地文件为2024-06-13的arXiv v1，以该版本2024标年 |
| DiffBP预印本 | 文件名标为2023 | 本地v4修订日期2024-07-14；正式期刊版另列2025 |
| KGDiff | PDF含2023版权/接收日期 | 期刊卷期为2024，25(1)；用2024，保留早期日期说明 |
| 3DSMILES-GPT | 线上发表于2024，卷期2025 | 用ChemSci2025卷期年，在线发布日期2024-12-04 |
| DiffIUPAC | DOI和版权含2024 | 期刊卷期为2025，第15卷；用2025 |
| DecompDPO | PDF同时出现2025与2026 | 正式发表TMLR09/2025；本地arXiv修订2026-02-10，名称注明修订 |
| Reward Shaping | arXiv首次编号为2502 | 本地v5修订日期2026-01-21，以版本年2026标名 |
| MOLER | 在线发表2021，当前卷期2022 | 用TKDE2022卷期年，另记在线发表2021 |
| Scopus AI报告 | 不是学术论文 | 三份按报告生成日2025标年；图像PDF已渲染核对首页Generated by Scopus AI, Mon Sep 22 2025，不声称发表年 |
| RL4SBDD | ICLR2026首页写under review | 标为2026会议投稿稿，不能写成已接收会议论文 |
| ExSelfRL | 现有稿件是Journal Pre-proof | 标为2024预发表稿年份，不能据此认定最终卷期年份 |
| Pharmacophore生成论文 | 两份首页日期为xx xx xxxx | 标为2025校样年份（PDF DOI/接受日期）；完整正式日期仍需核实 |
| 利用强化学习从头药物设计 | 一份原名写基于量子力学 | 对应DOI10.1007/s10994-024-06519-w，正文标题没有量子力学，规范为SMILES-RL关联论文 |

CrossDocked的预印本和正式稿、图生成强化学习的2021预印本和2022正式稿、DiffBP的2024预印本和2025正式版均分别保留。DiffSBDD正文与补充材料独立保留。
"""
    (staging / "年份与名称校正.md").write_text(corrections, encoding="utf-8")
    all_rows = ["# 论文与资料总索引", "", "默认阅读入口按主题与年份排列；完整来源见原件路径对照表。", "", "| 主题 | 年份 | 条目 | 口径/版本 |", "|---|---|---|---|"]
    for p in sorted(plan["papers"], key=lambda p:(THEMES[p["theme"]], p["year"], p["label"])):
        all_rows.append(f"| {THEMES[p['theme']][3:]} | {p['year']} | {link(p['label'], p['reading_path'])} | {p['kind']} |")
    (staging / "论文总索引.md").write_text("\n".join(all_rows) + "\n", encoding="utf-8")
    counts_rows = []
    for theme, folder in THEMES.items():
        counts_rows.append(f"| {link(folder[3:], folder + '/README.md')} | {sum(p['theme'] == theme for p in plan['papers'])} | {sum(p['theme'] == theme for p in plan['projects'])} |")
    main = f"""# 参考资料主题库

整理日期：2026-10-12。按研究主题组织论文、源码和笔记；论文/代码阅读入口以年份开头。

{summary['original_document_files']} 份原始文献资料归并为 **{summary['document_reading_entries']} 个阅读条目**；另有 {summary['figure_reading_entries']} 个独立框架图条目、**{summary['code_projects']} 个源码项目**、1 个提取笔记。{summary['missing_article_bodies']} 个缺正文网页导出单列。

## 按主题浏览

| 主题 | 文献/素材入口 | 源码项目 |
|---|---:|---:|
{chr(10).join(counts_rows)}

![文献按研究主题统计](reference_theme_counts.png)

## 查找与核对

- [论文总索引](论文总索引.md) / [论文总表CSV](论文总表.csv)
- [源码项目总表](代码项目总表.csv)
- [重复与版本对照](重复与版本对照.md) / [CSV](重复与版本对照.csv)
- [年份与名称校正](年份与名称校正.md)
- [年份与内容待核实](年份与内容待核实.md)
- [全部原件路径对照CSV](原件路径对照.csv)
- [原始资料归档](../99_原始资料归档/README.md)

## 阅读规则

相同SHA-256的文件合并阅读入口；同DOI、同为正文的不同字节版本归入一个条目，原始版本均保留。正文与补充材料分别列出，已有预印本与正式版分别保留。跨主题关联通过表格链接完成，不再复制论文。

年份优先使用期刊卷期/会议年；预印本、投稿稿、校样、报告和素材的年份口径在表中单独注明。{summary['documents_with_year']} 个文献条目已有年份，{summary['documents_year_pending']} 个文献及 {summary['projects_year_pending']} 个代码项目待核实。ScopusAI是检索报告，不能作为原始研究证据。

阅读入口是相对符号链接，文档源文件在归档或原项目目录。现有代码项目、权重、数据和运行输出保持原位置；可从“参考代码”入口浏览，但运行项目时沿用原源码路径。归档后的原始知识库、ZIP及散放PDF在旧路径保留兼容链接。
"""
    (staging / "README.md").write_text(main, encoding="utf-8")
    root_index = f"""# Reference project 统一入口

请从 **[主题资料库](主题资料库/README.md)** 开始阅读。论文、源码和笔记已按研究主题关联，阅读入口名称统一补上年份或“年份待核实”。

- **文献**：{summary['original_document_files']} 份原件 → {summary['document_reading_entries']} 个阅读条目，合并 {summary['removed_duplicate_reading_entries']} 个重复阅读入口；原件保留。
- **源码**：{summary['code_projects']} 个项目按主题建立入口；根目录原源码目录继续用于运行。
- **内容问题**：31 个网页导出缺正文，见[待补清单](主题资料库/年份与内容待核实.md)。

常用入口：[论文总索引](主题资料库/论文总索引.md)、[重复与版本](主题资料库/重复与版本对照.md)、[年份校正](主题资料库/年份与名称校正.md)、[原始归档](99_原始资料归档/README.md)。

`AI药物设计-XDC`、根目录旧PDF及ZIP是兼容入口，实际原件已归档。代码目录没有搬迁。原始导出不作为默认阅读目录，避免再次浏览重复副本或缺正文页面。
"""
    (audit / "root_README.preview.md").write_text(root_index, encoding="utf-8")


def apply_plan(plan: dict, audit: Path, record: Path) -> None:
    """Apply non-overwriting archival moves and verify every new navigation link."""
    root = Path(plan["root"])
    final = root / plan["library"]
    staging = root / ".主题资料库整理中_261012"
    for path in [final, staging, root / plan["archive"], root / "README.md", record]:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing to overwrite: {path}")
    for move in plan["moves"]:
        old, new = root / move["old"], root / move["new"]
        if not old.exists() or old.is_symlink() or new.exists():
            raise ValueError(f"Archival precondition failed: {move}")
    baseline = {}
    for project in plan["projects"]:
        source = root / project["path"]
        for base, dirs, names in os.walk(source):
            dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", ".venv"}]
            for name in names:
                path = Path(base) / name
                if path.is_file() and not path.is_symlink() and path.suffix.lower() in {".py", ".sh", ".yaml", ".yml", ".md"}:
                    baseline[str(path.relative_to(root))] = digest(path)
    (audit / "code_baseline_sha256.json").write_text(json.dumps(baseline, ensure_ascii=False, indent=2), encoding="utf-8")
    for paper in plan["papers"]:
        for source in paper["sources"]:
            if digest(root / source["original_path"]) != source["sha256"]:
                raise ValueError(f"Original changed after inventory: {source['original_path']}")
    staging.mkdir()
    write_indexes(plan, staging, audit)
    for paper in plan["papers"]:
        create_link(staging, final, paper["reading_path"], root / paper["source_path"])
    for project in plan["projects"]:
        create_link(staging, final, project["reading_path"], root / project["path"])
    for note in plan["notes"]:
        create_link(staging, final, note["reading_path"], root / note["source_path"])
    move_log = []
    for move in plan["moves"]:
        old, new = root / move["old"], root / move["new"]
        before = old.stat()
        new.parent.mkdir(parents=True, exist_ok=True)
        old.rename(new)
        old.symlink_to(os.path.relpath(new, old.parent), target_is_directory=new.is_dir())
        move_log.append({**move, "inode_before":before.st_ino, "inode_after":new.stat().st_ino})
        (audit / "moves_applied.json").write_text(json.dumps(move_log, ensure_ascii=False, indent=2), encoding="utf-8")
        logging.info("Archived with compatibility link: %s -> %s", move["old"], move["new"])
    archive_text = "# 原始资料归档\n\n保留原始导出、全部重复副本和散放论文原件，用于追溯。日常阅读请用[主题资料库](../主题资料库/README.md)。\n\n| 原路径（现为兼容链接） | 实际归档位置 |\n|---|---|\n"
    for move in plan["moves"]:
        archive_text += f"| {link(move['old'], '../' + move['old'])} | {link(move['new'].split('/', 1)[1], move['new'].split('/', 1)[1])} |\n"
    (root / plan["archive"] / "README.md").write_text(archive_text, encoding="utf-8")
    fig, ax = plt.subplots(figsize=(11, 6))
    labels = ["Diffusion / structure", "RL / preference", "Representation", "Evaluation / reward", "Data / benchmark", "Reviews / reports", "Target applications"]
    keys = ["D", "P", "R", "E", "B", "S", "A"]
    values = [plan["summary"]["document_counts_by_theme"].get(key, 0) for key in keys]
    bars = ax.barh(labels, values, color="#4b839e")
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) + 5)
    ax.set_title("Reference documents by research topic", fontsize=21, pad=16)
    ax.set_xlabel("Deduplicated reading entries", fontsize=16)
    ax.tick_params(labelsize=15)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, values):
        ax.text(value + 0.5, bar.get_y() + bar.get_height() / 2, str(value), va="center", fontsize=16)
    fig.tight_layout()
    fig.savefig(staging / "reference_theme_counts.png", dpi=160)
    plt.close(fig)
    staging.rename(final)
    with (root / "README.md").open("x", encoding="utf-8") as handle:
        handle.write((audit / "root_README.preview.md").read_text(encoding="utf-8"))
    validation = validate_plan(plan, baseline)
    (audit / "verification.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    write_record(plan, audit, record, validation)
    logging.info("COMPLETE %s", json.dumps(plan["summary"], ensure_ascii=False))


def validate_plan(plan: dict, baseline: dict) -> dict:
    """Verify originals, compatible paths, generated links, and untouched source code."""
    root = Path(plan["root"])
    final = root / plan["library"]
    failures = []
    originals = 0
    for paper in plan["papers"]:
        entry = final / paper["reading_path"]
        if not entry.is_symlink() or not entry.exists() or digest(entry) != paper["sha256"]:
            failures.append(str(entry))
        for source in paper["sources"]:
            old, archived = root / source["original_path"], root / source["archive_path"]
            if not old.exists() or not archived.exists() or digest(archived) != source["sha256"]:
                failures.append(source["original_path"])
            originals += 1
    for note in plan["notes"]:
        if digest(final / note["reading_path"]) != note["sha256"]:
            failures.append(note["original_path"])
    for project in plan["projects"]:
        if (final / project["reading_path"]).resolve() != (root / project["path"]).resolve():
            failures.append(project["path"])
    for source, expected in baseline.items():
        if digest(root / source) != expected:
            failures.append(source)
    for move in plan["moves"]:
        if not (root / move["old"]).is_symlink() or (root / move["old"]).resolve() != (root / move["new"]).resolve():
            failures.append(move["old"])
    broken = [str(path.relative_to(root)) for path in final.rglob("*") if path.is_symlink() and not path.exists()]
    failures.extend(broken)
    result = {"original_pdf_files_verified":originals, "note_files_verified":len(plan["notes"]),
              "code_files_hash_verified":len(baseline), "source_projects_verified":len(plan["projects"]),
              "reading_symlinks_verified":sum(path.is_symlink() for path in final.rglob("*")),
              "compatibility_links_verified":len(plan["moves"]), "broken_symlinks":broken,
              "failures":failures, "passed":not failures}
    if failures:
        raise ValueError(f"Layout validation failed: {result}")
    return result


def write_record(plan: dict, audit: Path, record: Path, validation: dict) -> None:
    """Write one durable report with local figures and exact provenance links."""
    root = Path(plan["root"])
    final = root / plan["library"]
    record.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(final / "reference_theme_counts.png", record.parent / "reference_theme_counts.png")
    rel = os.path.relpath(audit, record.parent)
    library_rel = os.path.relpath(final, record.parent)
    s = plan["summary"]
    rows = "\n".join(f"| {THEMES[t][3:]} | {sum(p['theme'] == t for p in plan['papers'])} | {sum(p['theme'] == t for p in plan['projects'])} |" for t in THEMES)
    text = f"""# 参考资料按主题去重与年份整理

日期：2026-10-12。原始问题：`reference project` 混有代码、论文、压缩包和知识库导出；同一论文跨分类重复，名称年份不一致。用户选择按研究主题整合，并强调去重、清晰和标年。

## 最终结构

阅读入口：`{final}`；根目录入口：`{root / 'README.md'}`。

建立8个主题目录，论文和参考代码分别在对应主题的子目录。根目录25个源码目录不搬迁，聚合目录中的子项目也保留原路径。共44个源码项目建立主题入口。

原知识库解压目录、3个ZIP、5个散放PDF归入 `99_原始资料归档`；9个旧入口建立相对兼容链接。保留每份原件，未删除重复文件。阅读入口均为相对符号链接，不新增论文副本。

| 指标 | 数量 |
|---|---:|
| 扫描PDF（含项目框架图） | {s['original_pdf_files']} |
| 原始文献资料（不含框架图） | {s['original_document_files']} |
| 文献阅读条目 | {s['document_reading_entries']} |
| 合并的重复阅读入口 | {s['removed_duplicate_reading_entries']} |
| 文献重复组 | {s['duplicate_document_groups']} |
| 同DOI但不同字节版本组 | {s['different_byte_same_doi_groups']} |
| 独立框架图入口 | {s['figure_reading_entries']} |
| 源码项目 | {s['code_projects']} |
| 已标年份的文献条目 | {s['documents_with_year']} |
| 文献年份待核实 | {s['documents_year_pending']} |
| 源码项目年份待核实 | {s['projects_year_pending']} |
| 缺正文导出 | {s['missing_article_bodies']} |

## 分类统计

| 主题 | 文献/素材 | 源码 |
|---|---:|---:|
{rows}

![主题资料统计](reference_theme_counts.png)

## 判定与年份边界

文件SHA-256相同者合并阅读入口。同DOI且同为正文的文件归为同一条目，保留不同字节版本的原始记录；补充材料独立保留。已有预印本和正式版分别保留。默认阅读副本在可解析原件中选大小较大的文件；不声称较大者是最新版，所有来源都可打开。

年份按本地PDF的期刊/会议引文与明确日期、项目README核对，不用文件修改时间、依赖版本或代码下载目录日期替代。预印本、投稿稿、校样、报告和配套素材均注明口径。本次没有联网核验所有引文，也没有读完每篇论文的科学内容。

重要校正包括：Uni-Mol会议版2023、扩散JCIM综述2024、化学强化学习综述2024、FromTheoryToTherapy本地预印本2024、DiffBP本地v4为2024（正式版2025）、KGDiff卷期2024、DecompDPO正式年2025/修订2026，以及MOLER卷期2022/在线2021。详见[全部年份与名称校正]({library_rel}/年份与名称校正.md)。

## 操作与验证

使用`dif`环境、18线程扫描，`pdftotext`提取PDF前两页形成可追溯年份证据。两个新增Python脚本执行`py_compile`，通过。先生成并检查`layout_plan.json`，再用`organize_references.py --apply`执行受控归档、创建链接、索引与统计图。

| 验证 | 结果 |
|---|---|
| PDF原件归档后SHA-256 | {validation['original_pdf_files_verified']}/{s['original_pdf_files']}通过 |
| MD原件及入口SHA-256 | {validation['note_files_verified']}/32通过 |
| 源码/配置/README文件前后SHA-256 | {validation['code_files_hash_verified']}个文件通过 |
| 项目主题入口指向原源码 | {validation['source_projects_verified']}/44通过 |
| 资料库相对链接 | {validation['reading_symlinks_verified']}个可用，无断链 |
| 原路径兼容入口 | {validation['compatibility_links_verified']}/9通过 |

不运行模型、生成或训练；这是资料目录整理，不涉及GPU和训练烟测。原工作树已有608个未跟踪条目、96个修改及4个删除，本次未覆盖这些变更。

## 产物与追溯

- [主题总入口]({library_rel}/README.md)
- [论文总索引]({library_rel}/论文总索引.md)
- [重复与版本对照]({library_rel}/重复与版本对照.md)
- [原件路径CSV]({library_rel}/原件路径对照.csv)
- [年份/正文待核实]({library_rel}/年份与内容待核实.md)
- [整理前完整盘点JSON]({rel}/inventory_before.json)
- [可复核分类与移动计划JSON]({rel}/layout_plan.json)
- [实际移动清单JSON]({rel}/moves_applied.json)
- [验证结果JSON]({rel}/verification.json)
- [处理日志]({rel}/organize.log)

剩余问题为{ s['documents_year_pending']}个文献和{ s['projects_year_pending']}个源码项目的年份，以及31篇原始导出缺正文。它们在总入口明确列出，不伪装成已确认条目。代码项目的状态、依赖和论文实现一致性不属于本次整理结论。
"""
    with record.open("x", encoding="utf-8") as handle:
        handle.write(text)


def main() -> None:
    """Prepare the plan, or apply a previously reviewed plan to original sources."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(args.audit_dir / "organize.log"), logging.StreamHandler()])
    if args.apply:
        plan = json.loads((args.audit_dir / "layout_plan.json").read_text(encoding="utf-8"))
        if str(args.root) != plan["root"]:
            raise ValueError("Requested root does not match saved plan")
        apply_plan(plan, args.audit_dir, args.record)
    else:
        plan = build_plan(args.root, args.audit_dir)
        (args.audit_dir / "layout_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        logging.info("Plan ready: %s", json.dumps(plan["summary"], ensure_ascii=False))
    print(json.dumps(plan["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
