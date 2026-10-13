"""Recover public repository evidence via Git metadata when the API is rate limited."""

from __future__ import annotations

import json
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from analysis.pharmacophore_survey_261013.collect_evidence import fetch_file


ROOT = Path(__file__).resolve().parent
LOGGER = logging.getLogger(__name__)
PHARMIT_FILES = ["README.md", "web/help.html", "web/examples/4pps.json", "web/src/feature.js",
                 "src/PharmerQuery.cpp", "src/RMSD.cpp", "src/pharmarec.cpp", "src/main.cpp"]


def main() -> None:
    """Archive only selected text blobs; inspect the full filename inventory."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    old = json.loads((ROOT / "repository_index.json").read_text())
    old_errors = {row["repository"]: row.get("error") for row in old}
    repos, files = [], []
    with ThreadPoolExecutor(max_workers=18) as pool:
        jobs = []
        for name, git_folder in [("songleee/LS-MolGen", "ls_molgen_git_index"),
                                 ("dkoes/pharmit", "pharmit_git_index")]:
            index = ROOT / "repositories" / git_folder
            sha = subprocess.check_output(["git", "-C", str(index), "rev-parse", "HEAD"], text=True).strip()
            # Do not request blob sizes: doing so triggers full blob downloads in partial clones.
            paths = subprocess.check_output(["git", "-C", str(index), "ls-tree", "-r", "--name-only", "HEAD"], text=True).splitlines()
            destination = ROOT / "repositories" / name.replace("/", "__")
            destination.mkdir(parents=True, exist_ok=True)
            record = {"repository": name, "sha": sha, "status": "ok", "url": f"https://github.com/{name}",
                      "folder": str(destination), "retrieval_method": "git metadata plus immutable raw files",
                      "api_error": old_errors.get(name), "files": len(paths), "paths": paths}
            (destination / "metadata.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))
            repos.append(record)
            if name == "songleee/LS-MolGen":
                chosen = [p for p in paths if Path(p).suffix.lower() in {".py", ".md", ".ipynb", ".json", ".pbs", ".yml", ".yaml"}
                          and not p.startswith("data/")]
            else:
                chosen = [p for p in PHARMIT_FILES if p in paths]
            jobs.extend((name, path, pool.submit(fetch_file, record, path)) for path in chosen)
            LOGGER.info("Indexed %s at %s with %d files", name, sha[:12], len(paths))
        for name, path, job in jobs:
            try:
                files.append(job.result())
            except Exception as exc:
                files.append({"repository": name, "path": path, "status": "failed", "error": str(exc)})
    (ROOT / "repository_index.json").write_text(json.dumps(repos, ensure_ascii=False, indent=2))
    (ROOT / "code_index.json").write_text(json.dumps(files, ensure_ascii=False, indent=2))
    LOGGER.info("Archived %d/%d selected files", sum(x["status"] == "ok" for x in files), len(files))


if __name__ == "__main__":
    main()
