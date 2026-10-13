"""Inventory reference projects and collect local bibliographic evidence for PDFs."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


def inspect_pdf(path: Path, root: Path, evidence: Path) -> dict:
    """Hash a PDF and extract its first two pages for year/title verification."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    proc = subprocess.run(["pdftotext", "-f", "1", "-l", "2", "-layout", str(path), "-"],
                          capture_output=True, text=True, errors="replace", timeout=90)
    text = proc.stdout
    (evidence / f"{digest}.txt").write_text(text, encoding="utf-8")
    year_lines = [line.strip() for line in text.splitlines()
                  if re.search(r"(?:20[12]\d|19\d\d)", line) and
                  re.search(r"Published|publication|Received|Accepted|Copyright|©|arXiv|ICLR|ICML|NeurIPS|20[12]\d\s*[,;:]|日期|年|学位", line, re.I)]
    dois = list(dict.fromkeys(re.findall(r"10\.\d{4,9}/[^\s<>]+", text, re.I)))
    return {"path":str(path.relative_to(root)), "bytes":path.stat().st_size,
            "sha256":digest, "parse_ok":proc.returncode == 0,
            "first_page_lines":[line.strip() for line in text.splitlines() if line.strip()][:18],
            "year_evidence_lines":year_lines[:12], "doi_candidates":dois[:4]}


def main() -> None:
    """Write the pre-change inventory and print compact bibliographic evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=18)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    evidence = args.output / "pdf_first_pages"
    evidence.mkdir(exist_ok=True)
    files = []
    directories = []
    pdfs = []
    leaves = []
    for item in sorted(args.root.iterdir()):
        if item.is_symlink():
            continue
        if item.is_file():
            files.append({"path":item.name, "bytes":item.stat().st_size})
            if item.suffix.lower() == ".pdf":
                pdfs.append(item)
            continue
        count = Counter()
        total = 0
        for base, dirs, names in os.walk(item, followlinks=False):
            for name in names:
                path = Path(base) / name
                if path.is_symlink():
                    continue
                total += path.stat().st_size
                count[path.suffix.lower() or "[no suffix]"] += 1
                if path.suffix.lower() == ".pdf" and ".git" not in path.parts:
                    pdfs.append(path)
        directories.append({"path":item.name, "files":sum(count.values()),
                            "bytes":total, "suffix_counts":dict(count)})
        if item.name == "AI药物设计-XDC":
            continue
        if item.name in {"modern_sbdd_20260531", "pharmacophore"}:
            leaves.extend(p for p in sorted(item.iterdir()) if p.is_dir() and not p.name.startswith("."))
        else:
            leaves.append(item)
    repos = []
    for path in leaves:
        readmes = [p for p in sorted(path.iterdir()) if p.is_file() and p.name.lower().startswith("readme")]
        text = "\n".join(p.read_text(errors="replace") for p in readmes)
        repos.append({"path":str(path.relative_to(args.root)), "name":path.name,
                      "readmes":[str(p.relative_to(args.root)) for p in readmes],
                      "readme_excerpt":text[:1800],
                      "readme_citation_lines":[line for line in text.splitlines() if re.search(r"year\s*=|@article|@inproceedings|doi.org|arxiv.org|paper.*20[12]\d|ICML.24|PNAS 2025|AAAI 2026", line, re.I)][:16]})
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        rows = list(executor.map(lambda p: inspect_pdf(p, args.root, evidence), sorted(pdfs)))
    groups = defaultdict(list)
    for row in rows:
        groups[row["sha256"]].append(row["path"])
    payload = {"root":str(args.root), "top_level_directories":directories, "top_level_files":files,
               "repositories":repos, "pdfs":rows,
               "exact_duplicate_groups":[paths for paths in groups.values() if len(paths) > 1]}
    (args.output / "inventory_before.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"pdfs":len(rows), "unique_pdf_bytes":len(groups), "code_projects":len(repos),
                      "top_level_directories":len(directories),
                      "exact_duplicate_groups":payload["exact_duplicate_groups"]}, ensure_ascii=False, indent=2))
    for index, paths in enumerate(groups.values(), 1):
        row = next(row for row in rows if row["path"] == paths[0])
        print(f"\n[{index:03d}] {row['sha256'][:10]} {row['path']}")
        print(" / ".join(row["first_page_lines"][:7])[:650])
        print("YEARS: " + " | ".join(row["year_evidence_lines"][:5])[:600])


if __name__ == "__main__":
    main()
