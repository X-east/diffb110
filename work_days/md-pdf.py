#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将多个 Markdown 合并为一个 PDF（含图片自适应、页码、目录）。

核心功能：
1) 批量读取目录中的 .md/.markdown 文件并合并输出 PDF。
2) 保留标题、列表、代码块、表格等常见 Markdown 样式。
3) 每个源文件之间自动分页，底部显示页码。
4) 自动处理图片路径（相对路径、绝对路径、Obsidian 图片语法）。
5) 图片按宽高上限等比缩放，避免单张图片占满整页。
6) 表格启用“严格识别”，降低普通文本被误识别为表格的概率。

输入输出：
- 输入目录：`INPUT_DIR`
- 输出 PDF：`OUTPUT_PDF`

依赖：
- markdown
- weasyprint

安装：
    pip install markdown weasyprint

运行：
    python md2pdf_with_png_autofit_tablefix_cn.py
"""

from __future__ import annotations

import html
import argparse
import re
import sys
from pathlib import Path
from typing import List, Sequence

import markdown
from weasyprint import HTML


# =========================================================
# Default configuration
# =========================================================
DEFAULT_INPUT_DIR = r"D:\Desktop\记录\日实验记录-文件夹\26年5月\第四周"
DEFAULT_OUTPUT_PDF = r"D:\Desktop\记录\日实验记录-文件夹\26年5月\第四周\260525-260530_日实验记录汇总.pdf"

# Whether to scan subfolders recursively
RECURSIVE = False

# Sorting:
# "name"  -> sort by filename (natural sort)
# "mtime" -> sort by file modified time
SORT_MODE = "name"

# Page settings
PAGE_SIZE = "A4"
PAGE_MARGIN_TOP = "20mm"
PAGE_MARGIN_RIGHT = "18mm"
PAGE_MARGIN_BOTTOM = "18mm"
PAGE_MARGIN_LEFT = "18mm"

# Optional document title shown at beginning
DOCUMENT_TITLE = ""

# Whether to show each source filename before its rendered markdown content
SHOW_FILENAME_HEADER = True

# If True, add a title page-like first heading for the merged document
SHOW_DOCUMENT_TITLE = bool(DOCUMENT_TITLE.strip())

# Image adaptive layout:
# - max width avoids horizontal overflow
# - max height prevents one image from visually occupying an entire page
IMAGE_MAX_WIDTH_PERCENT = 88
IMAGE_MAX_HEIGHT_MM = 210

# Strict table parsing:
# - only keep Markdown tables that match header + separator pattern
# - escape "|" in suspicious table-like blocks to avoid false positives
STRICT_TABLE_PARSING = True


# =========================================================
# CSS template
# =========================================================
BASE_CSS = f"""
@page {{
    size: {PAGE_SIZE};
    margin: {PAGE_MARGIN_TOP} {PAGE_MARGIN_RIGHT} {PAGE_MARGIN_BOTTOM} {PAGE_MARGIN_LEFT};

    @bottom-center {{
        content: counter(page);
        font-family: "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", "SimSun", sans-serif;
        font-size: 10pt;
        color: #666;
    }}
}}

html {{
    -weasy-hyphens: auto;
}}

body {{
    font-family: "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", "SimSun", sans-serif;
    font-size: 11pt;
    line-height: 1.65;
    color: #222;
    font-weight: 400;
    word-wrap: break-word;
    overflow-wrap: anywhere;
}}

h1, h2, h3, h4, h5, h6 {{
    font-family: "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", "SimSun", sans-serif;
    color: #111;
    margin-top: 1.2em;
    margin-bottom: 0.5em;
    font-weight: 700;
    page-break-after: avoid;
}}

h1 {{
    font-size: 20pt;
    border-bottom: 1px solid #ccc;
    padding-bottom: 0.2em;
}}

h2 {{
    font-size: 16pt;
}}

h3 {{
    font-size: 14pt;
}}

h4 {{
    font-size: 12.5pt;
}}

p {{
    margin: 0.55em 0;
    font-weight: 400;
}}

ul, ol {{
    margin: 0.45em 0 0.8em 1.6em;
}}

li {{
    margin: 0.2em 0;
}}

blockquote {{
    border-left: 4px solid #ccc;
    padding-left: 1em;
    color: #555;
    margin: 0.8em 0;
}}

code {{
    font-family: "Consolas", "Courier New", "Microsoft YaHei UI", monospace;
    font-size: 0.95em;
    background: #f5f5f5;
    padding: 0.1em 0.25em;
    border-radius: 3px;
    word-break: break-all;
}}

pre {{
    background: #f7f7f7;
    border: 1px solid #ddd;
    border-radius: 6px;
    padding: 12px;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-word;
    overflow-wrap: anywhere;
    page-break-inside: avoid;
}}

pre code {{
    background: none;
    padding: 0;
}}

table {{
    border-collapse: collapse;
    width: 100%;
    margin: 1em 0;
    font-size: 10pt;
    table-layout: fixed;
}}

th, td {{
    border: 1px solid #bbb;
    padding: 6px 8px;
    text-align: left;
    vertical-align: top;
    word-break: break-all;
    overflow-wrap: anywhere;
    white-space: normal;
}}

th {{
    background: #f0f0f0;
    font-weight: 700;
}}

td {{
    font-weight: 400;
}}

hr {{
    border: none;
    border-top: 1px solid #ccc;
    margin: 1.5em 0;
}}

img {{
    display: block;
    max-width: {IMAGE_MAX_WIDTH_PERCENT}%;
    max-height: {IMAGE_MAX_HEIGHT_MM}mm;
    width: auto;
    height: auto;
    margin: 0.35em auto;
    object-fit: contain;
    page-break-inside: avoid;
    break-inside: avoid;
}}

p > img {{
    margin-top: 0.45em;
    margin-bottom: 0.45em;
}}

.doc-title {{
    text-align: center;
    font-size: 24pt;
    font-weight: 700;
    margin-top: 1.5em;
    margin-bottom: 1.5em;
}}

.file-header {{
    font-size: 18pt;
    font-weight: 700;
    color: #1a1a1a;
    margin-bottom: 0.8em;
    border-bottom: 1px solid #ddd;
    padding-bottom: 0.2em;
}}

.page-break {{
    break-after: page;
    page-break-after: always;
}}

.section {{
    margin-bottom: 1em;
}}

.path-like {{
    font-family: "Consolas", "Courier New", "Microsoft YaHei UI", monospace;
    font-size: 9.5pt;
    word-break: break-all;
    overflow-wrap: anywhere;
}}
"""


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <style>
{css}
    </style>
</head>
<body>
{body}
</body>
</html>
"""


# =========================================================
# Utilities
# =========================================================
def natural_sort_key(path: Path):
    """
    Natural sort by filename, e.g.:
    1.md, 2.md, 10.md
    """
    name = path.name.lower()
    return [
        int(text) if text.isdigit() else text
        for text in re.split(r"(\d+)", name)
    ]


def collect_markdown_files(input_dir: Path) -> List[Path]:
    """Collect markdown files from input directory."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    patterns = ["*.md", "*.markdown", "*.MD", "*.MARKDOWN"]

    files: List[Path] = []
    if RECURSIVE:
        for pattern in patterns:
            files.extend(input_dir.rglob(pattern))
    else:
        for pattern in patterns:
            files.extend(input_dir.glob(pattern))

    # Remove duplicates while preserving logical uniqueness
    files = list(set(files))

    if SORT_MODE == "mtime":
        files.sort(key=lambda p: p.stat().st_mtime)
    else:
        files.sort(key=natural_sort_key)

    return files


def resolve_markdown_files(
    input_dir: Path,
    selected_files: Sequence[str] | None = None,
) -> List[Path]:
    """Collect all markdown files or resolve an explicit ordered file list."""
    if not selected_files:
        return collect_markdown_files(input_dir)

    files: List[Path] = []
    for item in selected_files:
        candidate = Path(item)
        if not candidate.is_absolute():
            candidate = input_dir / candidate
        candidate = candidate.resolve()

        if not candidate.exists():
            raise FileNotFoundError(f"Markdown file does not exist: {candidate}")
        if candidate.suffix.lower() not in {".md", ".markdown"}:
            raise ValueError(f"Selected file is not Markdown: {candidate}")
        files.append(candidate)

    return files


def read_text_with_fallback(path: Path) -> str:
    """Read text file with common encoding fallbacks."""
    encodings = ["utf-8", "utf-8-sig", "gb18030", "gbk", "latin-1"]
    for enc in encodings:
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(
        "unknown", b"", 0, 1,
        f"Could not decode file with tried encodings: {path}"
    )


TABLE_SEPARATOR_RE = re.compile(
    r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$"
)


def _unescaped_pipe_count(line: str) -> int:
    """Count non-escaped `|` characters."""
    count = 0
    escaped = False
    for ch in line:
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == "|":
            count += 1
    return count


def _is_table_separator_line(line: str) -> bool:
    """Check whether a line is a valid Markdown table separator."""
    return bool(TABLE_SEPARATOR_RE.match(line))


def _column_count(line: str) -> int:
    """Estimate column count from unescaped pipes."""
    pipes = _unescaped_pipe_count(line)
    if pipes == 0:
        return 1
    trimmed = line.strip()
    # Leading/trailing pipes add one extra delimiter.
    if trimmed.startswith("|") and trimmed.endswith("|"):
        return max(1, pipes - 1)
    return pipes + 1


def _escape_pipes(line: str) -> str:
    """Escape unescaped `|` so parser won't treat it as table syntax."""
    # Keep existing escaped pipes as-is.
    return re.sub(r"(?<!\\)\|", r"\\|", line)


def strict_table_preprocess(md_text: str) -> str:
    """
    Reduce false-positive table rendering.

    Strategy:
    - Detect candidate blocks with `header + separator` shape.
    - Keep the block only when column structure is reasonable.
    - Otherwise escape `|` in the candidate lines to force plain text rendering.
    """
    lines = md_text.split("\n")
    out: List[str] = []
    i = 0

    while i < len(lines):
        current = lines[i]
        next_line = lines[i + 1] if i + 1 < len(lines) else ""

        has_pipe = _unescaped_pipe_count(current) >= 1
        if has_pipe and _is_table_separator_line(next_line):
            header_cols = _column_count(current)
            sep_cols = _column_count(next_line)

            # Collect consecutive table-like body rows.
            j = i + 2
            body_rows: List[str] = []
            while j < len(lines) and _unescaped_pipe_count(lines[j]) >= 1 and lines[j].strip():
                body_rows.append(lines[j])
                j += 1

            valid = (
                header_cols >= 2
                and sep_cols >= 2
                and header_cols == sep_cols
            )

            if valid:
                for row in body_rows:
                    if _column_count(row) != header_cols:
                        valid = False
                        break

            if valid:
                out.extend(lines[i:j])
            else:
                out.append(_escape_pipes(current))
                out.append(_escape_pipes(next_line))
                out.extend(_escape_pipes(row) for row in body_rows)
            i = j
            continue

        out.append(current)
        i += 1

    return "\n".join(out)


def preprocess_markdown(md_text: str) -> str:
    """
    Light preprocessing to reduce some table / path rendering issues.

    What this does:
    - normalizes line endings
    - strips trailing spaces
    - keeps content mostly unchanged
    """
    md_text = md_text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in md_text.split("\n")]
    normalized = "\n".join(lines)
    if STRICT_TABLE_PARSING:
        return strict_table_preprocess(normalized)
    return normalized


def path_to_file_uri(path: Path) -> str:
    """Convert local path to file:// URI, safe for spaces/Chinese characters."""
    return path.resolve().as_uri()


def resolve_image_path(raw_path: str, md_file: Path) -> str:
    """
    Resolve image path against markdown file directory and return usable URI/path.

    - Keep http/https/data/file/anchor links unchanged
    - Convert local relative/absolute filesystem paths to file:// URI
    """
    cleaned = raw_path.strip()
    if not cleaned:
        return cleaned

    lowered = cleaned.lower()
    if lowered.startswith(("http://", "https://", "data:", "file://", "#")):
        return cleaned

    # Remove optional title part in Markdown image: ![alt](path "title")
    path_part = cleaned
    title_part = ""
    m = re.match(r'^(.*?)(\s+"[^"]*"\s*)$', cleaned)
    if m:
        path_part = m.group(1).strip()
        title_part = m.group(2)

    candidate = Path(path_part)
    if not candidate.is_absolute():
        candidate = (md_file.parent / candidate).resolve()
    else:
        candidate = candidate.resolve()

    # Keep original if path doesn't exist, to preserve previous behavior
    if not candidate.exists():
        return cleaned

    return f"{path_to_file_uri(candidate)}{title_part}"


def normalize_image_links(md_text: str, md_file: Path) -> str:
    """
    Normalize image links for PDF export:
    1) Obsidian syntax: ![[image.png]] -> ![](resolved_path)
    2) Standard Markdown image syntax path resolution per source file
    """
    # 1) Obsidian style images
    def obsidian_repl(match: re.Match) -> str:
        inner = match.group(1).strip()
        # Support possible aliases like ![[path/to/img.png|caption]]
        img_path = inner.split("|", 1)[0].strip()
        resolved = resolve_image_path(img_path, md_file)
        return f"![]({resolved})"

    md_text = re.sub(r"!\[\[(.*?)\]\]", obsidian_repl, md_text)

    # 2) Standard Markdown images
    # Matches: ![alt](path) and ![alt](<path with spaces>)
    image_pattern = re.compile(r'!\[([^\]]*)\]\((.*?)\)')

    def markdown_img_repl(match: re.Match) -> str:
        alt = match.group(1)
        raw = match.group(2).strip()
        if raw.startswith("<") and raw.endswith(">"):
            raw_path = raw[1:-1].strip()
            resolved = resolve_image_path(raw_path, md_file)
            return f"![{alt}](<{resolved}>)"
        resolved = resolve_image_path(raw, md_file)
        return f"![{alt}]({resolved})"

    return image_pattern.sub(markdown_img_repl, md_text)


def markdown_to_html(md_text: str) -> str:
    """
    Convert Markdown to HTML.
    Keep extensions conservative for PDF stability.
    """
    return markdown.markdown(
        md_text,
        extensions=[
            "extra",      # tables, fenced code, etc.
            "nl2br",
            "sane_lists",
        ],
        output_format="html5",
    )


def build_body(md_files: List[Path]) -> str:
    """Build the full HTML body."""
    parts: List[str] = []

    if SHOW_DOCUMENT_TITLE and DOCUMENT_TITLE.strip():
        parts.append(f'<div class="doc-title">{html.escape(DOCUMENT_TITLE)}</div>')
        if md_files:
            parts.append('<div class="page-break"></div>')

    for i, md_file in enumerate(md_files):
        raw = read_text_with_fallback(md_file)
        cleaned = preprocess_markdown(raw)
        normalized = normalize_image_links(cleaned, md_file)
        rendered = markdown_to_html(normalized)

        section: List[str] = ['<div class="section">']

        if SHOW_FILENAME_HEADER:
            section.append(
                f'<div class="file-header">{html.escape(md_file.name)}</div>'
            )

        section.append(rendered)
        section.append("</div>")

        # Force next markdown file to start on a new page
        if i < len(md_files) - 1:
            section.append('<div class="page-break"></div>')

        parts.append("\n".join(section))

    return "\n".join(parts)


def infer_output_pdf(input_dir: Path) -> Path:
    """Infer a weekly PDF path from a week folder."""
    md_files = collect_markdown_files(input_dir)
    stems = [p.stem for p in md_files if re.fullmatch(r"\d{6}", p.stem)]
    if stems:
        start = min(stems)
        end = max(stems)
        filename = f"谢东宸{start}-{end}.pdf"
    else:
        filename = f"{input_dir.name}.pdf"

    return input_dir.parent / filename


def generate_pdf(
    input_dir: Path,
    output_pdf: Path,
    selected_files: Sequence[str] | None = None,
) -> None:
    """Generate the merged PDF."""
    md_files = resolve_markdown_files(input_dir, selected_files)

    if not md_files:
        raise FileNotFoundError(f"No Markdown files found in: {input_dir}")

    print("Found Markdown files:")
    for f in md_files:
        print(f"  - {f}")

    body_html = build_body(md_files)
    full_html = HTML_TEMPLATE.format(
        title=html.escape(DOCUMENT_TITLE or "Merged Markdown Document"),
        css=BASE_CSS,
        body=body_html,
    )

    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    # base_url is important for resolving relative resources such as images
    HTML(string=full_html, base_url=str(input_dir.resolve())).write_pdf(str(output_pdf))

    print(f"\nPDF generated successfully:\n{output_pdf}")


def main() -> int:
    global RECURSIVE, SORT_MODE, DOCUMENT_TITLE, SHOW_DOCUMENT_TITLE

    parser = argparse.ArgumentParser(
        description="Merge Markdown files in a folder into a weekly PDF."
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default=DEFAULT_INPUT_DIR,
        help="Markdown folder to merge. Default: current configured week folder.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=DEFAULT_OUTPUT_PDF,
        help="Output PDF path. If omitted, infer from date-like markdown filenames.",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Scan subfolders recursively.",
    )
    parser.add_argument(
        "--sort",
        choices=["name", "mtime"],
        default=SORT_MODE,
        help="Sort markdown files by filename or modified time.",
    )
    parser.add_argument(
        "--title",
        default=DOCUMENT_TITLE,
        help="Optional document title shown on the first page.",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=None,
        help="Explicit ordered Markdown files to merge, relative to input_dir.",
    )

    try:
        args = parser.parse_args()

        RECURSIVE = args.recursive
        SORT_MODE = args.sort
        DOCUMENT_TITLE = args.title
        SHOW_DOCUMENT_TITLE = bool(DOCUMENT_TITLE.strip())

        input_dir = Path(args.input_dir)
        output_pdf = Path(args.output) if args.output else infer_output_pdf(input_dir)

        generate_pdf(input_dir, output_pdf, selected_files=args.files)
        return 0

    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
