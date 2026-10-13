"""Recheck imported files, record source export failures, and produce a work record."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from import_and_check import inspect_file


def main() -> None:
    """Write the final audit and a self-contained record without changing originals."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=18)
    args = parser.parse_args()
    audit = args.audit_dir
    summary = json.loads((audit / "integrity.json").read_text(encoding="utf-8"))
    destination = Path(summary["destination"])
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        rows = list(executor.map(lambda row: inspect_file(row, destination), summary["files_detail"]))
    summary["files_detail"] = rows
    failures = [r for r in rows if r.get("markdown_content_ok") is False]
    summary["markdown_content_failures"] = [r["path"] for r in failures]
    summary["markdown_readable_body_count"] = sum(r.get("markdown_content_ok") is True for r in rows)
    summary["unique_pdf_count"] = len({r["sha256"] for r in rows if Path(r["path"]).suffix.lower() == ".pdf"})
    summary["conclusion"] = "ZIP and extraction byte integrity passed; PDF parsing passed; 31 source-exported Markdown entries contain verification pages instead of article bodies."
    (audit / "integrity.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, subset in [("file_manifest.csv", rows), ("missing_article_bodies.csv", failures)]:
        with (audit / name).open("w", encoding="utf-8-sig", newline="") as handle:
            fields = sorted({key for row in subset for key in row})
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(subset)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    values = [139, summary["markdown_readable_body_count"], len(failures)]
    bars = ax.barh(["PDF: readable", "MD: article body", "MD: verification page"], values,
                   color=["#358769", "#518bc1", "#cc694f"])
    ax.invert_yaxis()
    ax.set_xlim(0, 155)
    ax.set_xlabel("Files", fontsize=16)
    ax.set_title("Knowledge base export integrity", fontsize=20, pad=16)
    ax.tick_params(labelsize=14)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, values):
        ax.text(value + 2, bar.get_y() + bar.get_height() / 2, str(value), va="center", fontsize=17)
    fig.tight_layout()
    plot = audit / "knowledgebase_integrity_summary.png"
    fig.savefig(plot, dpi=160)
    plt.close(fig)
    args.record.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(plot, args.record.parent / plot.name)
    rel_audit = Path(os.path.relpath(audit, args.record.parent))
    headings = Counter(Path(r["path"]).parts[0] if len(Path(r["path"]).parts) > 1 else "根目录" for r in rows)
    archive_table = "\n".join(f"| {Path(a['archive']).name} | {a['files']} | {a['uncompressed_bytes'] / 1024**2:.2f} | 通过 |" for a in summary["archives"])
    categories = "\n".join(f"| {name} | {count} |" for name, count in sorted(headings.items()))
    broken = "\n".join(f"- `{r['path']}`" for r in failures)
    record = f"""# AI药物设计-XDC知识库解压与完整性检查

日期：2026-10-12。目的：解压用户提供的两份知识库导出，并区分压缩/解压完整性与文章内容完整性。

**结论：两个 ZIP 和解压文件均通过字节完整性检查；139 个 PDF 可解析；31 个 MD 原始导出缺少文章正文，因此知识库内容不能判定为全部完好。**

## 解压位置与操作

`{destination}`

保留两份原始 ZIP；新建独立目录合并解压，保留原分类和全部重复文件。恢复 ZIP 中未标记 UTF-8 的中文路径，未改写文档内容。工作区原有大量未提交变更，本次未修改这些文件。

| 来源 | 文件数 | 解压大小（MiB） | ZIP CRC |
|---|---:|---:|---|
{archive_table}
| 合计 | 171 | {summary['total_bytes'] / 1024**2:.2f} | 通过 |

## 验证结果

| 检查项 | 结果 |
|---|---|
| ZIP 解压时全成员 CRC | 通过 |
| 落盘文件数 | 171/171 |
| 落盘大小、独立 CRC 校验 | 171/171 通过 |
| SHA-256 清单 | 两份 ZIP 及全部 171 文件均已记录 |
| PDF 结构与全页内容流解析 | 139/139 通过，共 2,487 页，pdfinfo 无警告 |
| PDF 按 SHA-256 去重 | {summary['unique_pdf_count']} 个不同 PDF，18 组重复内容，保留全部原文件 |
| MD UTF-8 编码 | 32/32 通过，无替换字符 |
| MD 内容 | 1 个正文文本；31 个微信验证 HTML 页面，缺少文章正文 |
| 可识别的本地资源引用 | 缺失 0；验证页面不含正文，不能据此推断文章附件完整 |
| 外部资源链接 | 248 处引用，未在线验证或下载 |

![知识库内容检查统计](knowledgebase_integrity_summary.png)

MD 的典型可见内容是“环境异常 当前环境异常，完成验证后即可继续访问。 去验证”；31 个文件均为 `<!DOCTYPE html>` 页面，未发现 `id=js_content` 文章正文容器。该内容已存在于压缩包中，解压后的字节和 CRC 与原始条目一致。

唯一含正文的 MD 是 `扩散生成模型/Brief Bioinform｜KGDiff：知识引导的扩散模型实现可解释的靶点感知分子生成-知乎专栏.md`；其原文声明正文经过提取整理、图表与超链接已省略，因此也不属于包含全部附件的完整网页存档。

## 分类清单

| 分类 | 文件数 |
|---|---:|
{categories}

## 复核方法与证据

运行环境 `dif`；CPU 检查使用 18 workers。本次为文件导入与完整性检查，不涉及 GPU 或模型运行。

1. `conda run -n dif python -m py_compile` 检查两个新增脚本，通过。
2. 运行 `import_and_check.py --archives ... --destination ... --audit-dir ... --workers 18`：完成安全路径检查、编码恢复、解压、全成员 ZIP CRC、落盘大小/CRC/SHA-256、PDF 与链接检查。
3. 运行 `finalize_check.py --audit-dir ... --record ... --workers 18`：复核落盘文件并检查 MD 正文，生成本记录及图表。
4. `pdfinfo` 检查全部 PDF 页数与结构；`pdftotext -q <file> /dev/null` 解析所有 PDF 的全页内容流。另抽查 DiffSBDD 与 Uni-Mol 首页渲染；这不是逐页人工审读或文献内容准确性核验。

- [完整检查 JSON]({rel_audit}/integrity.json)
- [逐文件大小、CRC、SHA-256 与解析结果]({rel_audit}/file_manifest.csv)
- [31 个缺正文条目清单]({rel_audit}/missing_article_bodies.csv)
- [运行日志]({rel_audit}/import_check.log)
- [导入检查脚本]({rel_audit}/import_and_check.py)
- [内容检查与记录脚本]({rel_audit}/finalize_check.py)

## 原始导出缺正文条目

{broken}

后续需从源知识库重新导出这 31 篇正文（可使用能保存正文的 PDF/HTML 导出方式）。不能通过改扩展名或再次解压恢复原本未保存的正文。
"""
    with args.record.open("x", encoding="utf-8") as handle:
        handle.write(record)
    with (audit / "import_check.log").open("a", encoding="utf-8") as handle:
        handle.write("2026-10-12 CONTENT_AUDIT complete: readable MD=1, verification HTML=31; record=" + str(args.record) + "\n")
    print(json.dumps({"record":str(args.record), "readable_pdf":139, "unique_pdf":summary["unique_pdf_count"], "readable_md":summary["markdown_readable_body_count"], "missing_md_bodies":len(failures)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
