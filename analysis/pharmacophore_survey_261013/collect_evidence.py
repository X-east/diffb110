"""Archive public LS-MolGen and Pharmit evidence without executing external code."""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from analysis.project_problem_survey_261012.prepare_sources import retrieve_open
from analysis.rl_before_after_survey_261012.collect_sources import collect_repo


LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent
TEXT_EXTENSIONS = {".py", ".md", ".ipynb", ".json", ".yml", ".yaml", ".pbs", ".txt"}
PHARMIT_PATHS = ["web/help.html", "README.md", "src/pharmit.cpp", "src/PharmerQuery.cpp",
                 "src/pharmarec.cpp", "src/pharmarec.h"]


def fetch_file(repository: dict[str, object], filename: str) -> dict[str, object]:
    """Preserve an immutable source file and its permalink."""
    name, sha = repository["repository"], repository["sha"]
    response = requests.get(f"https://raw.githubusercontent.com/{name}/{sha}/{filename}", timeout=25)
    response.raise_for_status()
    destination = Path(str(repository["folder"])) / "files" / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(response.text)
    return {"repository": name, "path": filename, "local_file": str(destination), "sha": sha,
            "url": f"https://github.com/{name}/blob/{sha}/{filename}", "status": "ok"}


def main() -> None:
    """Collect version-pinned repositories, selected files, and lawful open sources."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    (ROOT / "sources").mkdir(parents=True, exist_ok=True)
    repos, sources, files = [], [], []
    with ThreadPoolExecutor(max_workers=18) as pool:
        repo_jobs = [(name, pool.submit(collect_repo, name, ROOT / "repositories"))
                     for name in ["songleee/LS-MolGen", "dkoes/pharmit"]]
        paper_job = pool.submit(retrieve_open, "pharmit_paper", "PMC4987880", "xml", ROOT / "sources")
        for name, job in repo_jobs:
            try:
                repos.append(job.result())
                LOGGER.info("Archived repository %s", name)
            except Exception as exc:
                repos.append({"repository": name, "status": "failed", "error": str(exc)})
        jobs = []
        for repo in repos:
            if repo["status"] != "ok":
                continue
            tree = json.loads((Path(str(repo["folder"])) / "tree.json").read_text())
            blobs = [x for x in tree["tree"] if x["type"] == "blob"]
            if repo["repository"] == "songleee/LS-MolGen":
                selected = [x["path"] for x in blobs if Path(x["path"]).suffix.lower() in TEXT_EXTENSIONS
                            and x.get("size", 0) < 2_000_000 and not x["path"].startswith("data/")]
            else:
                selected = [x["path"] for x in blobs if x["path"] in PHARMIT_PATHS]
            jobs.extend((repo["repository"], filename, pool.submit(fetch_file, repo, filename)) for filename in selected)
        for name, filename, job in jobs:
            try:
                files.append(job.result())
            except Exception as exc:
                files.append({"repository": name, "path": filename, "status": "failed", "error": str(exc)})
        try:
            sources.append(paper_job.result())
        except Exception as exc:
            sources.append({"id": "pharmit_paper", "status": "failed", "error": str(exc)})
    # Figshare's public API exposes lawful supporting-file metadata, when indexed.
    try:
        response = requests.post("https://api.figshare.com/v2/articles/search",
                                 json={"search_for": "10.1021/acs.jcim.3c00587"}, timeout=25)
        response.raise_for_status()
        items = response.json()
        sources.append({"id": "ls_molgen_si_search", "status": "ok", "articles": items})
        LOGGER.info("SI metadata search returned %d articles", len(items))
        for item in items[:3]:
            metadata = requests.get(f"https://api.figshare.com/v2/articles/{item['id']}", timeout=25)
            metadata.raise_for_status()
            detail = metadata.json()
            (ROOT / "sources" / f"figshare_{item['id']}.json").write_text(json.dumps(detail, ensure_ascii=False, indent=2))
            for file in detail.get("files", []):
                if file["name"].endswith(".pdf"):
                    try:
                        sources.append(retrieve_open("ls_molgen_si", file["download_url"], "pdf", ROOT / "sources"))
                    except Exception as exc:
                        sources.append({"id": "ls_molgen_si", "status": "failed", "error": str(exc)})
    except Exception as exc:
        sources.append({"id": "ls_molgen_si_search", "status": "failed", "error": str(exc)})
        LOGGER.warning("SI metadata retrieval failed: %s", exc)
    for name, records in [("repository_index.json", repos), ("code_index.json", files), ("source_index.json", sources)]:
        (ROOT / name).write_text(json.dumps(records, ensure_ascii=False, indent=2))
    LOGGER.info("Archived %d repositories, %d/%d source files", len(repos),
                sum(x["status"] == "ok" for x in files), len(files))


if __name__ == "__main__":
    main()
