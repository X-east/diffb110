"""Archive selected public code and supplementary sources for an evidence audit."""

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
CODE_PATHS = {
    "MarcusOlivecrona/REINVENT": ["train_agent.py"],
    "MolecularAI/REINVENT4": ["configs/sampling.toml", "reinvent/runmodes/samplers/run_sampling.py"],
    "isayev/ReLeaSE": ["release/reinforcement.py"],
    "MorganCThomas/SMILES-RNN": ["scripts/sample_model.py", "scripts/reinforcement_learning.py"],
    "xy9485/DePPA": ["generate_ligands.py", "batch_generate_ligands_rl.py", "analysis/eval_qvina2.py"],
    "Kyle4490/RL-Diffusion": ["scripts/eval_sample.py", "scripts/eval_analyze.py"],
}
SUPPLEMENTS = {
    "gentrl_si": "https://static-content.springer-cdn.com/esm/art%3A10.1038%2Fs41587-019-0224-x/MediaObjects/41587_2019_224_MOESM1_ESM.pdf",
    "a2a_si": "https://static-content.springer-cdn.com/esm/art%3A10.1038%2Fs41467-025-60629-0/MediaObjects/41467_2025_60629_MOESM1_ESM.pdf",
    "cxcr4_si": "https://acs.figshare.com/doi/suppl/10.1021/acs.jmedchem.4c00184",
    "tyrosinase_si": "https://acs.figshare.com/doi/suppl/10.1021/acs.jmedchem.4c00091",
}


def archive_code(repo: dict[str, object], filename: str) -> dict[str, object]:
    """Download a version-pinned public file without executing it."""
    name, sha = repo["repository"], repo["sha"]
    url = f"https://raw.githubusercontent.com/{name}/{sha}/{filename}"
    response = requests.get(url, timeout=25)
    response.raise_for_status()
    destination = Path(str(repo["folder"])) / "code" / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(response.text)
    return {"repository": name, "sha": sha, "path": filename, "local_file": str(destination),
            "permalink": f"https://github.com/{name}/blob/{sha}/{filename}", "status": "ok"}


def main() -> None:
    """Collect supplementary evidence and extend the repository inventory."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    repos = json.loads((ROOT / "repository_source_index.json").read_text())
    existing = {item["repository"] for item in repos}
    code, supplements = [], []
    with ThreadPoolExecutor(max_workers=18) as pool:
        additional = [(name, pool.submit(collect_repo, name, ROOT / "repositories"))
                      for name in ["insilicomedicine/gentrl", "swansonk14/SyntheMol", "MolecularAI/transformer_rl"]
                      if name not in existing]
        for name, job in additional:
            try:
                repos.append(job.result())
            except Exception as exc:
                LOGGER.warning("Repository %s unavailable: %s", name, exc)
        jobs = [(name, pool.submit(retrieve_open, name, url, "pdf" if url.endswith("pdf") else "html", ROOT / "sources"))
                for name, url in SUPPLEMENTS.items()]
        code_jobs = [(repo["repository"], path, pool.submit(archive_code, repo, path))
                     for repo in repos for path in CODE_PATHS.get(repo["repository"], [])]
        for name, job in jobs:
            try:
                result = job.result()
                result["url"] = SUPPLEMENTS[name]
                supplements.append(result)
                LOGGER.info("Archived supplement %s", name)
            except Exception as exc:
                supplements.append({"id": name, "url": SUPPLEMENTS[name], "status": "failed", "error": str(exc)})
                LOGGER.warning("Supplement %s unavailable: %s", name, exc)
        for name, path, job in code_jobs:
            try:
                code.append(job.result())
            except Exception as exc:
                code.append({"repository": name, "path": path, "status": "failed", "error": str(exc)})
    for filename, records in [("repository_source_index.json", repos), ("code_source_index.json", code),
                              ("supplement_source_index.json", supplements)]:
        (ROOT / filename).write_text(json.dumps(records, ensure_ascii=False, indent=2))
    LOGGER.info("Archived %d repositories, %d/%d code files, %d/%d supplementary sources",
                len(repos), sum(x["status"] == "ok" for x in code), len(code),
                sum(x["status"] == "ok" for x in supplements), len(supplements))


if __name__ == "__main__":
    main()
