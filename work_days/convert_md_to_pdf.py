#!/usr/bin/env python3
"""Convert the 两周工作总结 markdown to PDF with embedded images."""

import base64, os, re, sys
from pathlib import Path

ROOT = Path("/home/md666/data/disk6/DiffB110")
MD_FILE = ROOT / "work_days/260526-260605 两周工作总结（五线梳理）.md"
OUT_HTML = ROOT / "work_days/260526-260605 两周工作总结（五线梳理）.html"
OUT_PDF = ROOT / "work_days/260526-260605 两周工作总结（五线梳理）.pdf"


def img_to_data_uri(img_path: Path) -> str:
    """Convert an image file to base64 data URI."""
    ext = img_path.suffix.lower().lstrip(".")
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "svg": "image/svg+xml", "gif": "image/gif"}.get(ext, "image/png")
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"


def convert_md_to_html(md_path: Path) -> str:
    """Read markdown, resolve image paths, produce standalone HTML."""
    import markdown_it
    md = (
        markdown_it.MarkdownIt("commonmark", {"breaks": True, "html": True})
        .enable("table")
    )
    
    lines = md_path.read_text(encoding="utf-8").splitlines()
    resolved = []
    missing = []
    
    for line in lines:
        # Match ![alt](relative_path)
        m = re.match(r'^!\[(.*?)\]\((.*?)\)', line)
        if m:
            alt, img_rel = m.groups()
            md_dir = md_path.parent
            img_abs = (md_dir / img_rel).resolve()
            if img_abs.exists():
                data_uri = img_to_data_uri(img_abs)
                resolved.append(f'![{alt}]({data_uri})')
            else:
                missing.append(str(img_abs))
                resolved.append(f'> *[图片缺失: {img_rel}]*')
        else:
            resolved.append(line)
    
    if missing:
        print(f"警告: {len(missing)} 个图片文件缺失")
    
    body = md.render("\n".join(resolved))
    
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="utf-8"/><title>两周工作总结</title>
<style>
  body {{
    font-family: "SimSun", "Noto Sans CJK SC", sans-serif;
    max-width: 900px; margin: 0 auto; padding: 20px;
    font-size: 14px; line-height: 1.7; color: #222;
  }}
  h1 {{ text-align: center; font-size: 22px; }}
  h2 {{ font-size: 18px; border-bottom: 1px solid #333; padding-bottom: 4px; margin-top: 28px; }}
  h3 {{ font-size: 15px; margin-top: 20px; }}
  h4 {{ font-size: 14px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 12px; table-layout: auto; page-break-inside: auto; }}
  tr {{ page-break-inside: avoid; page-break-after: auto; }}
  thead {{ display: table-header-group; }}
  tfoot {{ display: table-footer-group; }}
  th, td {{ border: 1px solid #555; padding: 4px 6px; vertical-align: top; }}
  th {{ background: #ddd; font-weight: bold; font-size: 12px; }}
  td {{ font-size: 12px; }}
  blockquote {{ border-left: 3px solid #888; padding-left: 12px; color: #555; margin: 8px 0; }}
  code {{ background: #f4f4f4; padding: 1px 4px; border-radius: 3px; font-size: 13px; }}
  img {{ max-width: 100%; height: auto; display: block; margin: 10px auto; }}
  hr {{ border: none; border-top: 1px solid #ccc; margin: 24px 0; }}
  .page-break {{ page-break-after: always; }}
  @media print {{
    body {{ font-size: 12px; }}
    h2 {{ page-break-before: always; }}
    h2:first-of-type {{ page-break-before: avoid; }}
    table {{ page-break-inside: auto; }}
    tr {{ page-break-inside: avoid; page-break-after: auto; }}
    thead {{ display: table-header-group; }}
  }}
</style></head>
<body>{body}</body></html>"""
    return html


if __name__ == "__main__":
    print(f"转换: {MD_FILE}")
    html = convert_md_to_html(MD_FILE)
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"HTML 已生成: {OUT_HTML} ({len(html)//1024} KB)")
    
    # 用 Playwright (Chromium) 生成 PDF，表格跨页不分裂
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.set_content(html, timeout=30000)
        page.pdf(
            path=str(OUT_PDF),
            format="A4",
            margin={"top": "15mm", "bottom": "15mm", "left": "12mm", "right": "12mm"},
            print_background=True,
        )
        browser.close()
    print(f"PDF 已生成: {OUT_PDF} ({OUT_PDF.stat().st_size//1024} KB)")
