"""Record local reference source URLs, commits, and source changes without fetching."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import urlsplit, urlunsplit


DEFAULT_WORKERS = 18
SOURCE_SUFFIXES = {".py", ".sh", ".yaml", ".yml", ".json", ".md", ".toml"}
SOURCE_PATHSPECS = [f":(glob)**/*{suffix}" for suffix in sorted(SOURCE_SUFFIXES)]


def run_git(root: Path, *arguments: str) -> subprocess.CompletedProcess:
    """Read a reference repository using a command-local ownership allowance."""
    return subprocess.run(["git", "-c", f"safe.directory={root.resolve()}", "-C", str(root), *arguments],
                          capture_output=True, text=True, errors="replace", check=False)


def safe_source_url(value: str) -> str:
    """Remove authentication information from a remote URL before recording it."""
    if re.match(r"^[^/]+@[^:]+:", value):
        host, path = value.split("@", 1)[1].split(":", 1)
        return f"https://{host}/{path}"
    parsed = urlsplit(value)
    if parsed.scheme in {"https", "http", "ssh", "git"} and parsed.hostname:
        host = parsed.hostname
        if parsed.port:
            host += f":{parsed.port}"
        return urlunsplit((parsed.scheme, host, parsed.path, "", ""))
    return ""


def project_directories(root: Path) -> list[Path]:
    """Find top-level reference projects and projects inside collection directories."""
    if not root.is_dir():
        raise FileNotFoundError(f"Reference source directory not found: {root}")
    result = []
    for directory in sorted(root.iterdir()):
        if not directory.is_dir() or directory.name.startswith("."):
            continue
        has_readme = any(p.is_file() and p.name.lower().startswith("readme") for p in directory.iterdir())
        if (directory / ".git").exists() or has_readme:
            result.append(directory)
        else:
            for child in sorted(directory.iterdir()):
                if child.is_dir() and not child.name.startswith("."):
                    result.append(child)
    return result


def inspect_project(root: Path, source_root: Path) -> dict:
    """Inspect only this project's Git metadata and source paths, not its parent repo."""
    readmes = [p.name for p in sorted(root.iterdir()) if p.is_file() and p.name.lower().startswith("readme")]
    environment_files = [p.name for p in sorted(root.iterdir()) if p.is_file() and
                         ("environment" in p.name.lower() or p.name.lower().startswith(("env.", "requirements")))]
    row = {"project":root.name, "relative_path":str(root.relative_to(source_root)),
           "source_url":"", "commit_sha":"", "source_changes":"unknown",
           "untracked_source_files":"unknown", "status":"needs_source_version",
           "readme_files":";".join(readmes), "environment_files":";".join(environment_files)}
    if not (root / ".git").exists():
        return row
    top = run_git(root, "rev-parse", "--show-toplevel")
    if top.returncode or Path(top.stdout.strip()).resolve() != root.resolve():
        return row
    remote = run_git(root, "remote", "get-url", "origin")
    commit = run_git(root, "rev-parse", "HEAD")
    if remote.returncode == 0:
        row["source_url"] = safe_source_url(remote.stdout.strip())
    if commit.returncode == 0 and re.fullmatch(r"[0-9a-f]{40,64}", commit.stdout.strip()):
        row["commit_sha"] = commit.stdout.strip()
    status = run_git(root, "status", "--porcelain=v1", "--untracked-files=all", "--", *SOURCE_PATHSPECS)
    if status.returncode == 0:
        lines = status.stdout.splitlines()
        row["source_changes"] = sum(not line.startswith("??") for line in lines)
        row["untracked_source_files"] = sum(line.startswith("??") for line in lines)
    if row["source_url"] and row["commit_sha"]:
        row["status"] = "source_version_recorded"
    return row


def main() -> None:
    """Print the local reference inventory, optionally saving a TSV manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parents[1]
    parser.add_argument("--root", type=Path, default=repo / "reference project/代码")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    root = args.root.resolve()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        rows = list(executor.map(lambda p:inspect_project(p, root), project_directories(root)))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [], delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        print(json.dumps({"projects":len(rows), "versions_recorded":sum(bool(r["commit_sha"] and r["source_url"]) for r in rows),
                          "needs_source_version":[r["project"] for r in rows if r["status"] == "needs_source_version"],
                          "source_changes":[r["project"] for r in rows if isinstance(r["source_changes"], int) and
                                            (r["source_changes"] or r["untracked_source_files"])],
                          "output":str(args.output)}, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
