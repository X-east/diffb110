"""Extract the two knowledge-base ZIPs and audit bytes, PDFs, and note links."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
from html import unescape
import json
import logging
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
from urllib.parse import unquote, urlsplit
import zipfile
import zlib


def restored_name(info: zipfile.ZipInfo) -> str:
    """Recover UTF-8 names from ZIP entries missing the UTF-8 flag."""
    if info.flag_bits & 0x800:
        return info.filename
    try:
        return info.filename.encode("cp437").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return info.filename


def inspect_file(row: dict, destination: Path) -> dict:
    """Verify an extracted file independently and check its document format."""
    path = destination / row["path"]
    data = path.read_bytes()
    row.update(sha256=hashlib.sha256(data).hexdigest(),
               size_ok=len(data) == row["bytes"],
               crc_ok=f"{zlib.crc32(data) & 0xffffffff:08x}" == row["crc32"])
    if path.suffix.lower() == ".pdf":
        proc = subprocess.run(["pdfinfo", str(path)], capture_output=True,
                              text=True, errors="replace", timeout=60)
        pages = re.search(r"^Pages:\s+(\d+)", proc.stdout, re.M)
        row.update(pdf_ok=proc.returncode == 0 and bool(pages),
                   pages=int(pages.group(1)) if pages else 0,
                   pdf_diagnostics=proc.stderr.strip())
        if row["pdf_ok"]:
            parsed = subprocess.run(["pdftotext", "-q", str(path), "/dev/null"],
                                    capture_output=True, text=True,
                                    errors="replace", timeout=120)
            row["pdf_content_parse_ok"] = parsed.returncode == 0
            row["pdf_content_diagnostics"] = parsed.stderr.strip()
    elif path.suffix.lower() == ".md":
        try:
            content = data.decode("utf-8-sig")
            row.update(markdown_utf8_ok=True, replacement_characters=content.count("\ufffd"))
            is_html = content.lstrip().lower().startswith("<!doctype html>")
            visible = re.sub(r"<(script|style)\b[^>]*>.*?</\1>", "", content, flags=re.S | re.I)
            visible = re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", visible))).strip()
            blocked = is_html and "当前环境异常，完成验证后即可继续访问" in visible
            row.update(markdown_is_html=is_html, markdown_content_ok=not blocked,
                       markdown_visible_excerpt=visible[:300])
        except UnicodeDecodeError:
            row.update(markdown_utf8_ok=False, replacement_characters=0)
    return row


def main() -> None:
    """Import without overwriting an existing destination and write audit evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", nargs="+", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=18)
    args = parser.parse_args()
    args.audit_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(args.audit_dir / "import_check.log"),
                                  logging.StreamHandler()])
    logging.info("environment=dif destination=%s workers=%s", args.destination, args.workers)
    if args.destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing destination: {args.destination}")
    planned = {}
    archive_results = []
    for archive in args.archives:
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                name = restored_name(info)
                parts = PurePosixPath(name).parts
                if not parts or name.startswith("/") or ".." in parts or "\\" in name:
                    raise ValueError(f"Invalid archive path: {name}")
                if stat.S_ISLNK(info.external_attr >> 16):
                    raise ValueError(f"Unsupported symlink: {name}")
                if not info.is_dir():
                    if name in planned:
                        raise ValueError(f"Duplicate output path: {name}")
                    planned[name] = archive.name
    args.destination.mkdir(parents=True)
    rows = []
    for archive in args.archives:
        result = {"archive":str(archive), "archive_bytes":archive.stat().st_size,
                  "archive_sha256":hashlib.sha256(archive.read_bytes()).hexdigest(),
                  "files":0, "uncompressed_bytes":0, "restored_names":0}
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                name = restored_name(info)
                path = args.destination / name
                result["restored_names"] += name != info.filename
                if info.is_dir():
                    path.mkdir(parents=True, exist_ok=True)
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                # Reading to EOF validates the archive member's CRC.
                data = zf.read(info)
                with path.open("xb") as handle:
                    handle.write(data)
                rows.append({"archive":archive.name, "path":name, "bytes":info.file_size,
                             "crc32":f"{info.CRC:08x}"})
                result["files"] += 1
                result["uncompressed_bytes"] += info.file_size
        result["zip_crc_ok"] = True
        archive_results.append(result)
        logging.info("Extracted and ZIP CRC checked: %s files=%s", archive.name, result["files"])
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        rows = list(executor.map(lambda r: inspect_file(r, args.destination), rows))
    missing = []
    remote_links = 0
    for row in rows:
        if not row.get("markdown_utf8_ok"):
            continue
        path = args.destination / row["path"]
        content = path.read_text(encoding="utf-8-sig")
        # Standard inline links, reference definitions, and embedded HTML resources.
        targets = re.findall(r"!?\[[^\]]*\]\(\s*(<[^>]+>|[^\s)]+)", content)
        targets += re.findall(r"(?m)^\s*\[[^\]]+\]:\s*(<[^>]+>|\S+)", content)
        targets += re.findall(r"(?:src|href)=[\"']([^\"']+)", content)
        for target in targets:
            target = target.strip("<>")
            url = urlsplit(target)
            if url.scheme or url.netloc:
                remote_links += 1
                continue
            if not url.path:
                continue
            resolved = path.parent / unquote(url.path)
            if not resolved.exists():
                missing.append({"source":row["path"], "target":target})
    by_hash = defaultdict(list)
    for row in rows:
        by_hash[row["sha256"]].append(row["path"])
    duplicates = [paths for paths in by_hash.values() if len(paths) > 1]
    summary = {"destination":str(args.destination), "archives":archive_results,
               "files":len(rows), "total_bytes":sum(r["bytes"] for r in rows),
               "file_types":dict(Counter(Path(r["path"]).suffix.lower() for r in rows)),
               "size_crc_failures":[r["path"] for r in rows if not r["size_ok"] or not r["crc_ok"]],
               "pdf_failures":[r["path"] for r in rows if r.get("pdf_ok") is False or r.get("pdf_content_parse_ok") is False],
               "pdf_total_pages":sum(r.get("pages", 0) for r in rows),
               "pdf_warnings":[{"path":r["path"], "diagnostics":r["pdf_diagnostics"]} for r in rows if r.get("pdf_diagnostics")],
               "markdown_encoding_failures":[r["path"] for r in rows if r.get("markdown_utf8_ok") is False],
               "markdown_content_failures":[r["path"] for r in rows if r.get("markdown_content_ok") is False],
               "markdown_replacement_characters":sum(r.get("replacement_characters", 0) for r in rows),
               "missing_local_links":missing, "external_link_occurrences":remote_links,
               "duplicate_content_groups":duplicates,
               "actual_file_count":sum(p.is_file() for p in args.destination.rglob("*")),
               "files_detail":rows}
    (args.audit_dir / "integrity.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = sorted({key for r in rows for key in r})
    with (args.audit_dir / "file_manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    logging.info("Audit complete files=%s pdf_pages=%s missing_local_links=%s", len(rows), summary["pdf_total_pages"], len(missing))
    print(json.dumps({k:v for k,v in summary.items() if k != "files_detail"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
