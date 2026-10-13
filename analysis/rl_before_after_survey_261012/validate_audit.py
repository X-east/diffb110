"""Validate audit provenance and local links, then render every exported PDF page."""

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


def validate(record: Path, pdf: Path, output_dir: Path) -> dict[str, object]:
    """Check artifacts and produce page images for subsequent visual inspection."""
    verification = output_dir / "verification.json"
    document = record.read_text()
    local = [unquote(x) for x in re.findall(r"\]\(([^)]+)\)", document)
             if not x.startswith(("http://", "https://", "#"))]
    missing = [x for x in local if (record.parent / x).resolve() != verification.resolve()
               and not (record.parent / x).exists()]
    if missing:
        raise ValueError(f"Missing local links: {missing}")
    images = re.findall(r"!\[[^]]*\]\(([^)]+)\)", document)
    if any("/" in x for x in images):
        raise ValueError("Work-record images must have same-directory paths")
    with (output_dir / "literature_comparison.csv").open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    audit = json.loads((output_dir / "evidence_audit.json").read_text())
    if len(rows) != 19 or len(audit["rows"]) != 19:
        raise ValueError("Expected 19 paper audit entries")
    if any(not row[field] for row in rows for field in ["inspection_scope", "source_url", "evidence_notes"]):
        raise ValueError("Incomplete paper evidence")
    repos = json.loads((output_dir / "repository_source_index.json").read_text())
    code = json.loads((output_dir / "code_source_index.json").read_text())
    if len(repos) != 10 or sum(item["status"] == "ok" for item in code) != 11:
        raise ValueError("Repository or code inventory count mismatch")
    if any(not Path(item["local_file"]).exists() for item in code if item["status"] == "ok"):
        raise ValueError("A downloaded code file is missing")
    html = pdf.with_suffix(".html").read_text()
    html_images = re.findall(r'<img[^>]+src="([^"]+)"', html)
    if any("/" in x for x in html_images):
        raise ValueError("Archived HTML images must have local filenames")
    qa = output_dir / "pdf_qa"
    qa.mkdir(exist_ok=True)
    subprocess.run(["pdftotext", "-bbox-layout", str(pdf), str(qa / "text_bounds.html")],
                   check=True, capture_output=True)
    ns = {"x": "http://www.w3.org/1999/xhtml"}
    pages = ElementTree.parse(qa / "text_bounds.html").getroot().findall(".//x:page", ns)
    outside = []
    for page_number, page in enumerate(pages, 1):
        width, height = float(page.get("width")), float(page.get("height"))
        for word in page.findall(".//x:word", ns):
            bounds = [float(word.get(key)) for key in ["xMin", "yMin", "xMax", "yMax"]]
            if bounds[0] < -1 or bounds[1] < -1 or bounds[2] > width + 1 or bounds[3] > height + 1:
                outside.append({"page": page_number, "word": word.text, "bounds": bounds})
    # Remove only this validator's previous page previews after a shorter re-export.
    for preview in qa.glob("page-[0-9]*.png"):
        preview.unlink()
    subprocess.run(["pdftoppm", "-r", "75", "-png", str(pdf), str(qa / "page")],
                   check=True, capture_output=True)
    page_images = sorted(qa.glob("page-*.png"))
    if len(page_images) != len(pages):
        raise ValueError("Rendered page count mismatch")
    for start in range(0, len(pages), 12):
        sheet = Image.new("RGB", (1260, 1395), "#E7EBF0")
        draw = ImageDraw.Draw(sheet)
        for offset, path in enumerate(page_images[start:start + 12]):
            page_image = Image.open(path)
            page_image.thumbnail((300, 425))
            x = (offset % 4) * 315 + (315 - page_image.width) // 2
            y = (offset // 4) * 465 + 28
            sheet.paste(page_image, (x, y))
            draw.text((x, y - 20), str(start + offset + 1), fill="black")
        sheet.save(qa / f"contact_{start + 1:02d}.png")
    result = {"paper_entries": len(rows), "repositories": len(repos), "code_files": len(code),
              "local_links": len(local), "missing_local_links": missing, "embedded_images": images,
              "pdf_pages": len(pages), "rendered_pages": len(page_images), "html_images": html_images,
              "text_outside_pages": outside, "visual_review": "rendered; inspect contact sheets separately"}
    verification.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    if outside:
        raise ValueError(f"Text extends beyond pages: {outside}")
    return result


def main() -> None:
    """Validate the new audit only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = validate(args.record, args.pdf, args.output_dir)
    LOGGER.info("Verified %d paper entries, %d PDF pages, %d local links",
                result["paper_entries"], result["pdf_pages"], result["local_links"])


if __name__ == "__main__":
    main()
