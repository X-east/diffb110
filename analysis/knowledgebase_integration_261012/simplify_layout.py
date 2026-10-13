"""Replace the link-heavy reference layout with four plain directories."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
from urllib.parse import quote, unquote, urlsplit


def sha256(path: Path) -> str:
    """Hash a file for migration verification."""
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def map_path(path: Path, moves: list[dict]) -> Path:
    """Map a pre-migration absolute path to its final location."""
    for row in sorted(moves, key=lambda row:len(row["old"]), reverse=True):
        old = Path(row["old"])
        try:
            relative = path.relative_to(old)
        except ValueError:
            continue
        return Path(row["new"]) / relative
    return path


def path_patches(repo: Path, top_names: list[str], audit: Path) -> list[dict]:
    """Collect narrowly matched path substitutions, preserving all other bytes."""
    names = "(?:" + "|".join(re.escape(name) for name in top_names) + ")"
    direct = re.compile(r"reference project/(?=" + names + r"(?![A-Za-z0-9_.-]))")
    joined = re.compile(r"(?P<q>[\"'])reference project(?P=q)(?=\s*/\s*[\"']" + names + r"[\"'])")
    tupled = re.compile(r"(?P<q>[\"'])reference project(?P=q)(?=\s*,\s*[\"']" + names + r"[\"'])")
    candidates = set()
    for name in ["ddpo", "scripts", "launch", "configs", "test", "tools", "chembl_classifier", "DiffSBDD", "docs", "reference project"]:
        base = repo / name
        if not base.exists():
            continue
        for current, dirs, files in os.walk(base, followlinks=False):
            dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", ".venv", "主题资料库", "99_原始资料归档"}]
            for filename in files:
                path = Path(current) / filename
                if path.suffix.lower() in {".py", ".sh", ".yaml", ".yml", ".json", ".md"} and not path.is_symlink():
                    candidates.add(path)
    # Analysis programs are executable references; immutable run/config snapshots are not edited.
    for current, dirs, files in os.walk(repo / "analysis", followlinks=False):
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", "knowledgebase_import_261012", "knowledgebase_integration_261012"} and "snapshot" not in d.lower()]
        for filename in files:
            path = Path(current) / filename
            if path.suffix.lower() in {".py", ".sh"} and not path.is_symlink():
                candidates.add(path)
    for path in repo.iterdir():
        if path.is_file() and path.suffix.lower() in {".py", ".sh", ".md", ".yaml", ".yml"}:
            candidates.add(path)
    patches = []
    for path in sorted(candidates):
        try:
            if path.stat().st_size > 2 * 1024**2:
                continue
            original = path.read_bytes()
            text = original.decode("utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if "reference project" not in text:
            continue
        replacement = direct.sub("reference project/代码/", text)
        replacement = joined.sub(lambda match:match["q"] + "reference project" + match["q"] + " / " + match["q"] + "代码" + match["q"], replacement)
        replacement = tupled.sub(lambda match:match["q"] + "reference project" + match["q"] + ", " + match["q"] + "代码" + match["q"], replacement)
        if replacement == text:
            continue
        relative = str(path.relative_to(repo))
        backup = audit / "before_path_edits" / relative
        backup.parent.mkdir(parents=True, exist_ok=True)
        backup.write_bytes(original)
        preview = audit / "after_path_edits" / relative
        preview.parent.mkdir(parents=True, exist_ok=True)
        preview.write_bytes(replacement.encode("utf-8"))
        patches.append({"path":relative, "before_sha256":hashlib.sha256(original).hexdigest(),
                        "after_sha256":hashlib.sha256(replacement.encode("utf-8")).hexdigest(),
                        "preview":str(preview), "backup":str(backup)})
    return patches


def make_plan(repo: Path, audit: Path) -> dict:
    """Prepare the complete move and path-edit plan without changing project files."""
    root = repo / "reference project"
    old = json.loads((audit / "layout_plan.json").read_text(encoding="utf-8"))
    inventory = json.loads((audit / "inventory_before.json").read_text(encoding="utf-8"))
    top_names = [r["path"] for r in inventory["top_level_directories"] if r["path"] != "AI药物设计-XDC"]
    moves = []
    papers = []
    for row in old["papers"]:
        if row["theme"] == "G":
            continue
        parts = Path(row["reading_path"]).parts
        target = root / "论文" / parts[0] / parts[-1]
        source = root / row["source_path"]
        moves.append({"old":str(source), "new":str(target), "kind":"paper", "sha256":row["sha256"]})
        papers.append({**row, "new_path":str(target.relative_to(root))})
    notes = []
    for row in old["notes"]:
        if row["missing_body"]:
            continue
        target = root / "笔记" / Path(row["reading_path"]).name
        moves.append({"old":str(root / row["source_path"]), "new":str(target), "kind":"note", "sha256":row["sha256"]})
        notes.append(str(target))
    for name in top_names:
        moves.append({"old":str(root / name), "new":str(root / "代码" / name), "kind":"code"})
    moves += [
        {"old":str(root / "99_原始资料归档"), "new":str(root / "归档/原始导出"), "kind":"archive"},
        {"old":str(root / "主题资料库"), "new":str(root / "归档/旧版入口/主题资料库"), "kind":"old_index"},
        {"old":str(root / "README.md"), "new":str(root / "归档/旧版入口/README_第一版.md"), "kind":"old_index"},
    ]
    for path in sorted(root.iterdir()):
        if path.is_symlink():
            moves.append({"old":str(path), "new":str(root / "归档/旧版入口/兼容链接" / path.name), "kind":"old_alias"})
    symlinks = []
    for path in list((root / "主题资料库").rglob("*")) + list(root.iterdir()):
        if path.is_symlink():
            symlinks.append({"old":str(path), "target":str(path.resolve()), "link_text":os.readlink(path)})
    docs = [root / "README.md", root / "99_原始资料归档/README.md"]
    docs += [p for p in (root / "主题资料库").rglob("*.md") if not p.is_symlink()]
    doc_texts = [{"old":str(p), "text":p.read_text(encoding="utf-8")} for p in docs]
    baseline = json.loads((audit / "code_baseline_sha256.json").read_text(encoding="utf-8"))
    baseline_now = {name:sha256(root / name) for name in baseline}
    patches = path_patches(repo, top_names, audit)
    return {"repo":str(repo), "root":str(root), "moves":moves, "papers":papers,
            "notes":notes, "projects":old["projects"], "old_plan":old,
            "symlinks":symlinks, "doc_texts":doc_texts, "code_baseline":baseline_now,
            "path_patches":patches, "top_level_before":len(list(root.iterdir()))}


def md_link(label: str, target: str) -> str:
    """Create a local link with escaped spaces and Chinese path components."""
    return f"[{label}]({quote(target, safe='/')})"


def apply_plan(plan: dict, audit: Path) -> dict:
    """Move real files, patch references, and replace the root index with a short one."""
    repo = Path(plan["repo"])
    root = Path(plan["root"])
    moves = plan["moves"]
    for name in ["论文", "代码", "笔记", "归档"]:
        if (root / name).exists():
            raise FileExistsError(root / name)
    for row in moves:
        source, target = Path(row["old"]), Path(row["new"])
        if not source.exists() or target.exists():
            raise ValueError(f"Move precondition failed: {row}")
        if row.get("sha256") and sha256(source) != row["sha256"]:
            raise ValueError(f"Source changed: {source}")
    for patch in plan["path_patches"]:
        if sha256(repo / patch["path"]) != patch["before_sha256"]:
            raise ValueError(f"Code changed after preflight: {patch['path']}")
    for name in ["论文", "代码", "笔记", "归档"]:
        (root / name).mkdir()
    # Figures are embedded project assets and remain in their source repositories.
    for row in plan["old_plan"]["papers"]:
        if row["theme"] != "G":
            continue
        target = root / "笔记/框架图" / Path(row["reading_path"]).name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / row["source_path"], target)
    applied = []
    for row in moves:
        source, target = Path(row["old"]), Path(row["new"])
        target.parent.mkdir(parents=True, exist_ok=True)
        before = source.lstat().st_ino
        source.rename(target)
        applied.append({**row, "inode_preserved":before == target.lstat().st_ino})
        (audit / "simplify_moves_applied.json").write_text(json.dumps(applied, ensure_ascii=False, indent=2), encoding="utf-8")
        logging.info("Move %s -> %s", row["old"], row["new"])
    for patch in plan["path_patches"]:
        original = repo / patch["path"]
        path = map_path(original, moves)
        if sha256(path) != patch["before_sha256"]:
            raise ValueError(f"Unexpected code contents at {path}")
        path.write_bytes(Path(patch["preview"]).read_bytes())
    for row in plan["symlinks"]:
        path = map_path(Path(row["old"]), moves)
        target = map_path(Path(row["target"]), moves)
        temp = path.with_name(path.name + ".migration-link")
        temp.symlink_to(os.path.relpath(target, path.parent), target_is_directory=target.is_dir())
        temp.replace(path)
    archive_moves = [row for row in moves if row["kind"] not in {"paper", "note"}]
    for row in moves:
        if row["kind"] not in {"paper", "note"}:
            continue
        archived_name = map_path(Path(row["old"]), archive_moves)
        archived_name.parent.mkdir(parents=True, exist_ok=True)
        archived_name.symlink_to(os.path.relpath(Path(row["new"]), archived_name.parent))
    # Rebase the archived indexes so historical provenance remains accessible.
    for row in plan["doc_texts"]:
        original = Path(row["old"])
        target = map_path(original, moves)
        def rebase(match: re.Match) -> str:
            url = urlsplit(match[1])
            if url.scheme or url.netloc or not url.path:
                return match[0]
            old_destination = Path(os.path.abspath(original.parent / unquote(url.path)))
            destination = map_path(old_destination, moves)
            relative = os.path.relpath(destination, target.parent)
            return "(" + quote(relative, safe="/") + ("#" + url.fragment if url.fragment else "") + ")"
        text = re.sub(r"(?<=\])\(([^)]+)\)", rebase, row["text"])
        target.write_text(text, encoding="utf-8")
    categories = {}
    for row in plan["papers"]:
        folder = Path(row["new_path"]).parts[1]
        categories[folder] = categories.get(folder, 0) + 1
    short = ["# 参考资料", "", "日常阅读从论文目录开始。每篇文献只保留一个阅读文件，名称以年份开头；重复原件和旧版索引放在归档。", "", "| 目录 | 内容 |", "|---|---|",
             "| [论文](论文/) | 114个去重文献条目，按研究主题分类 |",
             "| [代码](代码/README.md) | 44个参考项目及对应年份 |",
             "| [笔记](笔记/) | 网页摘录、框架图和待补正文清单 |",
             "| [归档](归档/) | 原始ZIP、重复副本及上一版索引 |", "", "论文分类：", ""]
    short += [f"- {md_link(folder[3:], '论文/' + folder + '/')}：{count}篇" for folder, count in sorted(categories.items())]
    short += ["", "4个文献和8个代码项目年份待核实，31篇网页导出缺正文，见[待核实清单](笔记/待核实与待补正文.md)。", ""]
    (root / "README.md").write_text("\n".join(short), encoding="utf-8")
    projects = ["# 参考代码", "", "这里保留项目原名称及原内部结构。年份按配套论文或README核对，无法确认的标为待核实。", "", "| 项目 | 年份 | 方向 |", "|---|---|---|"]
    for row in sorted(plan["projects"], key=lambda row:(row["theme"], row["name"])):
        projects.append(f"| {md_link(row['name'], row['path'] + '/')} | {row['year']} | {Path(row['reading_path']).parts[0][3:]} |")
    (root / "代码/README.md").write_text("\n".join(projects) + "\n", encoding="utf-8")
    pending = ["# 待核实与待补正文", "", "## 文献年份待核实", ""]
    for row in plan["papers"]:
        if row["year"] == "待核实":
            pending.append("- " + md_link(row["label"], "../" + row["new_path"]))
    pending += ["", "## 代码年份待核实", ""]
    for row in plan["projects"]:
        if row["year"] == "待核实":
            pending.append("- " + md_link(row["name"], "../代码/" + row["path"] + "/"))
    pending += ["", "## 31篇缺正文导出", "", "原始文件为微信验证页面，仅保留文章名线索；需要从源知识库重新导出正文。", ""]
    for row in plan["old_plan"]["notes"]:
        if row["missing_body"]:
            path = map_path(root / row["source_path"], moves)
            pending.append("- " + md_link(Path(row["original_path"]).name, os.path.relpath(path, root / "笔记")))
    (root / "笔记/待核实与待补正文.md").write_text("\n".join(pending) + "\n", encoding="utf-8")
    verification = verify(plan, applied)
    (audit / "simplify_verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8")
    return verification


def verify(plan: dict, applied: list[dict]) -> dict:
    """Validate the four-directory layout, documents, source edits, and archived links."""
    root = Path(plan["root"])
    repo = Path(plan["repo"])
    moves = plan["moves"]
    failures = []
    expected_root = {"论文", "代码", "笔记", "归档", "README.md"}
    actual_root = {p.name for p in root.iterdir()}
    if expected_root != actual_root:
        failures.append(f"Unexpected root entries: {actual_root}")
    for row in plan["papers"]:
        path = root / row["new_path"]
        if path.is_symlink() or sha256(path) != row["sha256"]:
            failures.append(str(path))
    expected_code = {r["path"]:r["after_sha256"] for r in plan["path_patches"]}
    for source, original_hash in plan["code_baseline"].items():
        original = root / source
        path = map_path(original, moves)
        expected = expected_code.get(str(original.relative_to(repo)), original_hash)
        if sha256(path) != expected:
            failures.append(str(path))
    for patch in plan["path_patches"]:
        path = map_path(repo / patch["path"], moves)
        if sha256(path) != patch["after_sha256"]:
            failures.append(str(path))
    broken_links = []
    for path in (root / "归档/旧版入口").rglob("*"):
        if path.is_symlink() and not path.exists():
            broken_links.append(str(path.relative_to(root)))
    failures.extend(broken_links)
    markdown_links = 0
    for doc in [root / "README.md", root / "代码/README.md", root / "笔记/待核实与待补正文.md"]:
        for value in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", doc.read_text(encoding="utf-8")):
            url = urlsplit(value)
            if url.scheme or not url.path:
                continue
            markdown_links += 1
            if not (doc.parent / unquote(url.path)).exists():
                failures.append(str(doc) + ":" + value)
    result = {"root_before":plan["top_level_before"], "root_after":len(actual_root),
              "root_entries":sorted(actual_root), "physical_paper_files":len(plan["papers"]),
              "code_files_verified":len(plan["code_baseline"]), "path_files_updated":len(plan["path_patches"]),
              "same_inode_moves":sum(row["inode_preserved"] for row in applied),
              "new_index_links_checked":markdown_links, "broken_archived_symlinks":broken_links,
              "failures":failures, "passed":not failures}
    if failures:
        raise ValueError(json.dumps(result, ensure_ascii=False))
    return result


def main() -> None:
    """Prepare or apply a migration whose inputs and outputs are preserved locally."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(args.audit_dir / "simplify.log"), logging.StreamHandler()])
    if args.apply:
        plan = json.loads((args.audit_dir / "simplify_plan.json").read_text(encoding="utf-8"))
        result = apply_plan(plan, args.audit_dir)
    else:
        plan = make_plan(args.repo, args.audit_dir)
        (args.audit_dir / "simplify_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        result = {"moves":len(plan["moves"]), "path_files":len(plan["path_patches"]),
                  "path_files_by_area":{}, "paper_files":len(plan["papers"])}
        for row in plan["path_patches"]:
            area = Path(row["path"]).parts[0]
            result["path_files_by_area"][area] = result["path_files_by_area"].get(area, 0) + 1
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
