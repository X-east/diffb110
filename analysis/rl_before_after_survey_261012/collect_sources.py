"""Collect primary paper sources and public repository inventories for RL comparison audits."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from analysis.project_problem_survey_261012.prepare_sources import extract_local, retrieve_open


LOGGER = logging.getLogger(__name__)
LOCAL_PAPERS = {
    "reinvent_2017": "02_强化学习与偏好优化/2017_REINVENT_深度强化学习从头分子设计.pdf",
    "release_2018": "02_强化学习与偏好优化/2018_ReLeaSE_深度强化学习从头药物设计.pdf",
    "multiparam_2019": "02_强化学习与偏好优化/2019_深度强化学习_多参数药物设计优化.pdf",
    "reinvent4": "02_强化学习与偏好优化/2024_REINVENT4_生成式分子设计框架.pdf",
    "acegen": "02_强化学习与偏好优化/2024_ACEGEN_生成化学智能体强化学习.pdf",
    "augmented_memory": "04_评价指标与奖励设计/2024_AugmentedMemory_样本高效分子设计.pdf",
    "deppa": "02_强化学习与偏好优化/2026_DePPA_去噪策略优化微调口袋扩散_预印本.pdf",
    "rl_diffusion": "02_强化学习与偏好优化/2025_RL-Diffusion_不确定性感知多目标强化学习.pdf",
    "drugex": "02_强化学习与偏好优化/2023_DrugEx_类药化学空间探索工具.pdf",
    "mol2mol_rl": "04_评价指标与奖励设计/2024_Transformer分子设计_强化学习评估.pdf",
}
REUSE_PAPERS = ["a2a", "rl_al", "ls_molgen", "cxcr4", "tyrosinase", "alidiff", "decompdpo"]
REPOSITORIES = [
    "MarcusOlivecrona/REINVENT", "MolecularAI/REINVENT4", "isayev/ReLeaSE",
    "MorganCThomas/SMILES-RNN", "xy9485/DePPA", "Kyle4490/RL-Diffusion",
    "MolecularAI/reinforcement-learning-active-learning",
]


def collect_repo(repository: str, output_dir: Path) -> dict[str, object]:
    """Preserve public README and immutable tree provenance without running external code."""
    api = f"https://api.github.com/repos/{repository}"
    metadata_response = requests.get(api, timeout=25)
    metadata_response.raise_for_status()
    metadata = metadata_response.json()
    branch = metadata["default_branch"]
    tree_response = requests.get(f"{api}/git/trees/{branch}?recursive=1", timeout=30)
    tree_response.raise_for_status()
    tree = tree_response.json()
    folder = output_dir / repository.replace("/", "__")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "tree.json").write_text(json.dumps(tree, ensure_ascii=False, indent=2))
    paths = [item["path"] for item in tree.get("tree", []) if item["type"] == "blob"]
    readme_path = next((path for path in paths if path.lower() in {"readme.md", "readme.rst"}), None)
    if readme_path:
        response = requests.get(f"https://raw.githubusercontent.com/{repository}/{tree['sha']}/{readme_path}", timeout=25)
        response.raise_for_status()
        (folder / "README.txt").write_text(response.text)
    record = {"repository": repository, "url": metadata["html_url"], "sha": tree["sha"],
              "default_branch": branch, "files": len(paths), "tree_truncated": tree.get("truncated", False),
              "status": "ok", "folder": str(folder)}
    (folder / "metadata.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))
    return record


def main() -> None:
    """Collect the selected corpus, reporting incomplete sources explicitly."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references-root", type=Path, default=Path("reference project/论文"))
    parser.add_argument("--previous-dir", type=Path, default=Path("analysis/project_problem_survey_261012"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--num-workers", type=int, default=18)
    args = parser.parse_args()
    papers_dir, repos_dir = args.output_dir / "sources", args.output_dir / "repositories"
    papers_dir.mkdir(parents=True, exist_ok=True)
    repos_dir.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    paper_records = []
    for paper_id in REUSE_PAPERS:
        source = args.previous_dir / "sources" / f"{paper_id}.txt"
        destination = papers_dir / source.name
        shutil.copy2(source, destination)
        paper_records.append({"id": paper_id, "source": str(source), "type": "reused_primary_text",
                              "text_file": str(destination), "status": "ok"})
    repo_records = []
    with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        jobs = [(paper_id, pool.submit(extract_local, paper_id, args.references_root / filename, papers_dir))
                for paper_id, filename in LOCAL_PAPERS.items()]
        jobs += [("synthemol_rl", pool.submit(retrieve_open, "synthemol_rl", "PMC13230741", "xml", papers_dir)),
                 ("gentrl", pool.submit(retrieve_open, "gentrl", "https://www.nature.com/articles/s41587-019-0224-x.pdf", "pdf", papers_dir))]
        repo_jobs = [(repo, pool.submit(collect_repo, repo, repos_dir)) for repo in REPOSITORIES]
        for paper_id, future in jobs:
            try:
                record = future.result()
                LOGGER.info("Prepared paper %s", paper_id)
            except Exception as exc:
                record = {"id": paper_id, "status": "failed", "error": str(exc)}
                LOGGER.warning("Paper source unavailable: %s: %s", paper_id, exc)
            paper_records.append(record)
        for repo, future in repo_jobs:
            try:
                record = future.result()
                LOGGER.info("Prepared repository %s at %s", repo, record["sha"][:10])
            except Exception as exc:
                record = {"repository": repo, "status": "failed", "error": str(exc)}
                LOGGER.warning("Repository unavailable: %s: %s", repo, exc)
            repo_records.append(record)
    (args.output_dir / "paper_source_index.json").write_text(json.dumps(paper_records, ensure_ascii=False, indent=2))
    (args.output_dir / "repository_source_index.json").write_text(json.dumps(repo_records, ensure_ascii=False, indent=2))
    LOGGER.info("Prepared %s/%s paper sources, %s/%s repositories",
                sum(item["status"] == "ok" for item in paper_records), len(paper_records),
                sum(item["status"] == "ok" for item in repo_records), len(repo_records))


if __name__ == "__main__":
    main()
