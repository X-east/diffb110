"""Check staged file content against the project's Git upload rules without writing."""
from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path, PurePosixPath
import re
import subprocess


DEFAULT_MAX_FILE_MIB = 10.0
BLOCKED_ROOTS = {"OUT", "output", "tmp", "runs", "data", "checkpoints", "saved_models",
                 "reference project", "work_records", "wandb", "mlruns", ".envs", ".codex", ".agents"}
BLOCKED_PARTS = {"__pycache__", ".pytest_cache", "before_path_edits", "after_path_edits",
                 "pdf_first_pages", "script_snapshot", "code_snapshot", "runtime_snapshot", "snapshots"}
BLOCKED_SUFFIXES = {".ckpt", ".pth", ".pt", ".bin", ".pkl", ".pickle", ".npz", ".npy", ".h5", ".hdf5",
                    ".safetensors", ".zip", ".tar", ".gz", ".so", ".log", ".pem", ".key", ".pyc"}
PUBLISHED_SUFFIXES = {".md", ".csv", ".json", ".png", ".svg", ".pdf", ".yaml", ".yml", ".py", ".sh"}
CREDENTIAL_MARKERS = (
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),
    re.compile(rb"\bgithub_pat_[A-Za-z0-9_]{40,}\b"),
    re.compile(rb"\bsk-(?:proj-)?[A-Za-z0-9_-]{32,}\b"),
)
LOGGER = logging.getLogger(__name__)


def git_bytes(repo: Path, *arguments: str) -> bytes:
    """Read Git output without using the shell or printing staged content."""
    return subprocess.run(["git", "-C", str(repo), *arguments], capture_output=True, check=True).stdout


def path_problem(name: str) -> str | None:
    """Reject generated artifacts and credential paths, allowing small test fixtures."""
    path = PurePosixPath(name)
    if path.parts[0] in BLOCKED_ROOTS or any(p in BLOCKED_PARTS for p in path.parts):
        return "local output, data, archive, snapshot, or backup directory"
    if path.name == ".env" or (path.name.startswith(".env.") and path.name != ".env.example"):
        return "local environment credentials"
    if path.suffix.lower() in BLOCKED_SUFFIXES or ".bak" in path.name or path.name.endswith(".tar.gz"):
        return "model, binary data, log, credential file, or backup"
    if path.suffix.lower() in {".sdf", ".pdb", ".pdbqt", ".smi"} and not name.startswith("test/fixtures/"):
        return "raw molecule/protein data (use local evidence storage)"
    if name.startswith("analysis/published/") and path.suffix.lower() not in PUBLISHED_SUFFIXES:
        return "publication area accepts reviewed summaries, figures, configs, and source only"
    if name.startswith("analysis/") and path.suffix.lower() not in {".py", ".sh"} and not name.startswith("analysis/published/"):
        return "analysis artifact outside the reviewed publication area"
    return None


def inspect_staged(repo: Path, max_file_mib: float) -> tuple[int, list[tuple[str, str]]]:
    """Check staged blobs rather than possibly different working-tree files."""
    conflicted = git_bytes(repo, "diff", "--name-only", "--diff-filter=U", "-z")
    if conflicted:
        return 0, [("<index>", "unresolved merge conflicts")]
    raw = git_bytes(repo, "diff", "--cached", "--name-only", "--diff-filter=ACMRT", "-z")
    names = [p.decode("utf-8", errors="surrogateescape") for p in raw.split(b"\0") if p]
    failures = []
    for name in names:
        problem = path_problem(name)
        if problem:
            failures.append((name, problem))
            continue
        ignored = subprocess.run(["git", "-C", str(repo), "check-ignore", "--no-index", "-q", "--", name],
                                 capture_output=True, check=False)
        if ignored.returncode not in {0, 1}:
            raise ValueError("Cannot read the current Git ignore rules")
        if ignored.returncode == 0:
            failures.append((name, "path matches current Git ignore rules"))
            continue
        size = int(git_bytes(repo, "cat-file", "-s", ":" + name).strip())
        if size > max_file_mib * 1024**2:
            failures.append((name, f"staged file exceeds {max_file_mib:g} MiB project limit"))
            continue
        content = git_bytes(repo, "show", ":" + name)
        if any(pattern.search(content) for pattern in CREDENTIAL_MARKERS):
            failures.append((name, "possible credential marker in staged content"))
    return len(names), failures


def main() -> None:
    """Report staged upload eligibility without changing the index or remote."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--max-file-mib", type=float, default=DEFAULT_MAX_FILE_MIB)
    args = parser.parse_args()
    if not math.isfinite(args.max_file_mib) or args.max_file_mib <= 0:
        parser.error("--max-file-mib must be positive and finite")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        count, failures = inspect_staged(args.repo, args.max_file_mib)
    except (subprocess.CalledProcessError, ValueError):
        LOGGER.error("Cannot inspect staged Git files; check repository and index state.")
        raise SystemExit(2) from None
    for name, reason in failures:
        LOGGER.error("%s: %s", name, reason)
    if failures:
        LOGGER.error("Upload check failed: %d files checked, %d issues.", count, len(failures))
        raise SystemExit(1)
    LOGGER.info("Upload check passed: %d staged non-deletion files checked.", count)
    if not count:
        LOGGER.info("No source/artifact content is staged; this does not validate unstaged files.")


if __name__ == "__main__":
    main()
