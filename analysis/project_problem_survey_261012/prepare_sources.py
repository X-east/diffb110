"""Extract local papers and retrieve open primary sources for the survey."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from xml.etree import ElementTree

import requests
from bs4 import BeautifulSoup


LOGGER = logging.getLogger(__name__)
DEFAULT_WORKERS = 18
LOCAL_PAPERS = {
    "diffsbdd": "01_扩散生成与结构设计/2024_DiffSBDD_等变扩散结构药物设计.pdf",
    "diffsbdd_supplement": "01_扩散生成与结构设计/2024_DiffSBDD_补充材料.pdf",
    "diffgui": "01_扩散生成与结构设计/2025_DiffGui_引导等变扩散靶点分子生成.pdf",
    "alidiff": "02_强化学习与偏好优化/2024_AliDiff_精确能量优化与扩散模型对齐.pdf",
    "decompdpo": "02_强化学习与偏好优化/2025_DecompDPO_分解偏好优化_2026修订版.pdf",
    "rl_al": "02_强化学习与偏好优化/2024_主动学习辅助的样本高效强化学习.pdf",
    "ls_molgen": "07_应用案例与靶点设计/2023_LS-MolGen_配体与结构双驱动靶点生成.pdf",
    "cxcr4": "07_应用案例与靶点设计/2024_CXCR4_合成可行性与强化学习先导设计.pdf",
    "tyrosinase": "07_应用案例与靶点设计/2024_酪氨酸酶抑制剂_AI生成与优化.pdf",
    "from_theory": "04_评价指标与奖励设计/2024_FromTheoryToTherapy_SBDD实用评价指标_预印本.pdf",
}
OPEN_XML_PAPERS = {
    "a2a": "PMC12216625",
    "posebusters": "PMC10901501",
    "chemical_plausibility": "PMC13138256",
}
OPEN_HTML_PAPERS = {
    "re_cmeb": "https://link.springer.com/article/10.1186/s44280-025-00102-1",
    "synthemol": "https://www.nature.com/articles/s42256-024-00809-7",
}
OPEN_PDF_PAPERS = {
    "pmo": "https://proceedings.neurips.cc/paper_files/paper/2022/file/8644353f7d307baaf29bc1e56fe8e0ec-Paper-Datasets_and_Benchmarks.pdf",
}


def extract_local(paper_id: str, source: Path, output_dir: Path) -> dict[str, object]:
    """Extract a supplied PDF without changing the original."""
    destination = output_dir / f"{paper_id}.txt"
    subprocess.run(["pdftotext", "-layout", str(source), str(destination)], check=True)
    return {"id": paper_id, "source": str(source), "type": "local_pdf", "status": "ok",
            "text_file": str(destination), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "characters": len(destination.read_text(errors="replace"))}


def retrieve_open(paper_id: str, source: str, kind: str, output_dir: Path) -> dict[str, object]:
    """Retrieve an open primary-source article and preserve its provenance."""
    url = (f"https://www.ebi.ac.uk/europepmc/webservices/rest/{source}/fullTextXML"
           if kind == "xml" else source)
    response = requests.get(url, timeout=35)
    response.raise_for_status()
    destination = output_dir / f"{paper_id}.txt"
    if kind == "xml":
        xml_path = output_dir / f"{paper_id}.xml"
        xml_path.write_bytes(response.content)
        root = ElementTree.fromstring(response.content)
        parts = []
        for node in root.iter():
            if node.tag in {"article-title", "title", "p", "caption", "table"}:
                parts.append(" ".join("".join(node.itertext()).split()))
        destination.write_text("\n\n".join(parts))
    elif kind == "pdf":
        if not response.content.startswith(b"%PDF"):
            raise ValueError(f"The endpoint did not return a PDF: {url}")
        pdf_path = output_dir / f"{paper_id}.pdf"
        pdf_path.write_bytes(response.content)
        subprocess.run(["pdftotext", "-layout", str(pdf_path), str(destination)], check=True)
    else:
        soup = BeautifulSoup(response.content, "html.parser")
        article = soup.find("article") or soup.find("main")
        if article is None:
            raise ValueError(f"No article body was found: {url}")
        destination.write_text(article.get_text("\n", strip=True))
    return {"id": paper_id, "source": url, "type": f"open_{kind}", "status": "ok",
            "http_status": response.status_code, "text_file": str(destination),
            "characters": len(destination.read_text(errors="replace"))}


def main() -> None:
    """Prepare the selected corpus and report any unavailable source."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references-root", type=Path, default=Path("reference project/论文"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "sources")
    parser.add_argument("--num-workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    records = []
    with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        jobs = [(paper_id, pool.submit(extract_local, paper_id, args.references_root / name,
                                      args.output_dir)) for paper_id, name in LOCAL_PAPERS.items()]
        for mapping, kind in ((OPEN_XML_PAPERS, "xml"), (OPEN_HTML_PAPERS, "html"),
                              (OPEN_PDF_PAPERS, "pdf")):
            jobs.extend((paper_id, pool.submit(retrieve_open, paper_id, source, kind, args.output_dir))
                        for paper_id, source in mapping.items())
        for paper_id, future in jobs:
            try:
                record = future.result()
                LOGGER.info("Prepared %s: %s characters", paper_id, record["characters"])
            except Exception as exc:
                record = {"id": paper_id, "status": "failed", "error": str(exc)}
                LOGGER.warning("Source unavailable: %s: %s", paper_id, exc)
            records.append(record)
    index_path = args.output_dir.parent / "literature_source_index.json"
    index_path.write_text(json.dumps(records, ensure_ascii=False, indent=2))
    LOGGER.info("Prepared %s/%s sources; index=%s",
                sum(record["status"] == "ok" for record in records), len(records), index_path)


if __name__ == "__main__":
    main()
