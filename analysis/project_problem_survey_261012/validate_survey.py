"""Check survey links and PDF bounds, and render pages for visual review."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote
from xml.etree import ElementTree

from PIL import Image, ImageDraw


LOGGER = logging.getLogger(__name__)


def validate_survey(record: Path, pdf: Path, analysis_dir: Path) -> dict[str, object]:
    """Check local evidence links and render every PDF page using Poppler."""
    links = re.findall(r"\]\(([^)]+)\)", record.read_text())
    local = [unquote(link) for link in links if not link.startswith(("https://", "http://", "#"))]
    missing = [link for link in local if not (record.parent / link).exists()]
    images = re.findall(r"!\[[^]]*\]\(([^)]+)\)", record.read_text())
    if missing:
        raise ValueError(f"Missing local links: {missing}")
    if any("/" in image for image in images):
        raise ValueError("Work-record images must be stored in the same directory")
    with (analysis_dir / "literature_argument_comparison.csv").open(encoding="utf-8-sig") as stream:
        comparisons = list(csv.DictReader(stream))
    if len(comparisons) != 15:
        raise ValueError(f"Expected 15 paper comparisons, found {len(comparisons)}")
    if any(not row[field] for row in comparisons for field in ("问题与做法", "证据结构", "结论边界与借鉴")):
        raise ValueError("Incomplete paper comparison")
    html = pdf.with_suffix(".html").read_text()
    image_sources = re.findall(r'<img[^>]+src="([^"]+)"', html)
    if any("/" in source for source in image_sources):
        raise ValueError("Archived HTML images must use local filenames")
    qa_dir = analysis_dir / "pdf_qa"
    qa_dir.mkdir(exist_ok=True)
    subprocess.run(["pdftotext", "-bbox-layout", str(pdf), str(qa_dir / "text_bounds.html")],
                   check=True, capture_output=True)
    root = ElementTree.parse(qa_dir / "text_bounds.html").getroot()
    namespace = {"x": "http://www.w3.org/1999/xhtml"}
    pages = root.findall(".//x:page", namespace)
    outside = []
    for index, page in enumerate(pages):
        width, height = float(page.attrib["width"]), float(page.attrib["height"])
        for word in page.findall(".//x:word", namespace):
            bounds = [float(word.attrib[key]) for key in ("xMin", "yMin", "xMax", "yMax")]
            if bounds[0] < -1 or bounds[1] < -1 or bounds[2] > width + 1 or bounds[3] > height + 1:
                outside.append({"page": index + 1, "text": word.text, "bounds": bounds})
    subprocess.run(["pdftoppm", "-r", "65", "-png", str(pdf), str(qa_dir / "page")],
                   check=True, capture_output=True)
    page_images = sorted(qa_dir.glob("page-*.png"))
    if len(page_images) != len(pages):
        raise ValueError("PDF page count and rendered image count differ")
    for start in range(0, len(pages), 12):
        sheet = Image.new("RGB", (1260, 1395), "#E7EBF0")
        draw = ImageDraw.Draw(sheet)
        for offset, path in enumerate(page_images[start:start + 12]):
            image = Image.open(path)
            image.thumbnail((300, 425))
            x = (offset % 4) * 315 + (315 - image.width) // 2
            y = (offset // 4) * 465 + 28
            sheet.paste(image, (x, y))
            draw.text((x, y - 20), str(start + offset + 1), fill="black")
        sheet.save(qa_dir / f"contact_{start + 1:02d}.png")
    summary = {"pdf_pages": len(pages), "local_links": len(local), "missing_links": missing,
               "embedded_images": len(images), "paper_comparisons": len(comparisons),
               "html_image_sources": image_sources, "text_outside_pages": outside}
    (analysis_dir / "verification.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    if outside:
        raise ValueError(f"Text extends beyond the PDF page: {outside}")
    return summary


def main() -> None:
    """Validate one report and print a compact summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    summary = validate_survey(args.record, args.pdf, args.analysis_dir)
    LOGGER.info("Validated %s pages, %s local links, %s images, %s paper comparisons",
                summary["pdf_pages"], summary["local_links"], summary["embedded_images"],
                summary["paper_comparisons"])


if __name__ == "__main__":
    main()
