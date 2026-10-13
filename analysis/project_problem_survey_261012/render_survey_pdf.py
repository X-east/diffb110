"""Render one survey using the repository's existing Markdown/PDF helpers."""

from __future__ import annotations

import argparse
import importlib.util
import logging
from pathlib import Path

from weasyprint import HTML


LOGGER = logging.getLogger(__name__)


def render_survey(record: Path, output: Path, converter: Path) -> None:
    """Reuse Markdown conversion while applying document-specific readable typography."""
    spec = importlib.util.spec_from_file_location("project_markdown_pdf", converter)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load the configured converter: {converter}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.SHOW_FILENAME_HEADER = False
    body = module.build_body([record.resolve()])
    # Keep archived HTML images local to the report directory.
    body = body.replace(record.resolve().parent.as_uri() + "/", "")
    css = module.BASE_CSS + """
    body { font-size: 11.3pt; line-height: 1.62; }
    h1 { font-size: 23pt; color: #17324d; }
    h2 { font-size: 16pt; color: #17324d; }
    h3 { font-size: 13pt; color: #315b87; }
    table { table-layout: fixed; width: 100%; font-size: 9.5pt; }
    th, td { overflow-wrap: anywhere; word-break: normal; padding: 5px; }
    th { background: #e6edf8; }
    img { max-width: 100%; max-height: 170mm; }
    a { color: #315b87; text-decoration: none; }
    blockquote { border-left: 3px solid #6f91be; background: #f5f7fa; }
    """
    html = f"<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><style>{css}</style></head><body>{body}</body></html>"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".html").write_text(html)
    HTML(string=html, base_url=str(record.resolve().parent)).write_pdf(output)
    LOGGER.info("Saved PDF %s", output)


def main() -> None:
    """Render the selected survey only, leaving all existing reports unchanged."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--converter", type=Path,
                        default=Path(__file__).resolve().parents[2] / "work_days" / "md-pdf.py")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    logging.getLogger("weasyprint").setLevel(logging.WARNING)
    render_survey(args.record, args.output, args.converter)


if __name__ == "__main__":
    main()
