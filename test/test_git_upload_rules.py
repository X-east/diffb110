"""Exercise upload checks against real isolated Git indexes and source inventories."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECK_SCRIPT = PROJECT_ROOT / "scripts/check_git_upload.py"
INVENTORY_SCRIPT = PROJECT_ROOT / "scripts/inventory_reference_projects.py"


def git(repo: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    """Run a Git operation in a disposable fixture repository."""
    return subprocess.run(["git", "-C", str(repo), *arguments], capture_output=True, text=True, check=True)


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    """Create an independent Git index without touching the project index."""
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q")
    return root


def run_check(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    """Run the actual upload-check CLI with a fixture repository."""
    return subprocess.run([sys.executable, str(CHECK_SCRIPT), "--repo", str(root), *arguments],
                          capture_output=True, text=True, check=False)


def test_check_reads_staged_blob_and_does_not_print_secret(repository: Path) -> None:
    """A clean working file cannot hide a credential already staged in the index."""
    file = repository / "settings.txt"
    token = "ghp_" + "a" * 40
    file.write_text("token=" + token + "\n")
    git(repository, "add", "--", "settings.txt")
    staged_tree = git(repository, "write-tree").stdout
    file.write_text("token=placeholder\n")
    result = run_check(repository)
    assert result.returncode == 1
    assert "credential marker" in result.stderr
    assert token not in result.stdout + result.stderr
    assert git(repository, "write-tree").stdout == staged_tree
    assert file.read_text() == "token=placeholder\n"


@pytest.mark.parametrize("relative_path", ["OUT/metrics.csv", "analysis/published/model.bin", "analysis/published/report.csv"])
def test_check_rejects_force_added_output(repository: Path, relative_path: str) -> None:
    """Ignoring an output directory must not let force-added output pass the check."""
    (repository / ".gitignore").write_text("OUT/\nanalysis/published/report.csv\n")
    path = repository / relative_path
    path.parent.mkdir(parents=True)
    path.write_text("epoch,reward\n1,0\n")
    git(repository, "add", "-f", "--", relative_path)
    result = run_check(repository)
    assert result.returncode == 1
    assert relative_path in result.stderr
    assert path.exists()


def test_check_enforces_configured_size_limit(repository: Path) -> None:
    """A normal text artifact exceeding the configured limit is rejected."""
    (repository / "summary.txt").write_bytes(b"x" * 512)
    git(repository, "add", "--", "summary.txt")
    result = run_check(repository, "--max-file-mib", "0.0001")
    assert result.returncode == 1
    assert "exceeds" in result.stderr


def test_cancelled_tracking_is_allowed_and_keeps_local_file(repository: Path) -> None:
    """Deletion-only index changes used for local-output cleanup are accepted."""
    path = repository / "tmp"
    path.mkdir()
    (path / "preview.png").write_bytes(b"local preview")
    git(repository, "add", "--", "tmp/preview.png")
    git(repository, "-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "fixture")
    git(repository, "rm", "--cached", "--", "tmp/preview.png")
    result = run_check(repository)
    assert result.returncode == 0
    assert "0 staged non-deletion" in result.stderr
    assert (path / "preview.png").read_bytes() == b"local preview"


def test_inventory_uses_child_metadata_and_handles_custom_root(repository: Path) -> None:
    """A project without Git metadata must not inherit its parent's source identity."""
    git(repository, "remote", "add", "origin", "https://example.invalid/parent.git")
    root = repository / "references"
    child = root / "pinned"
    missing = root / "unversioned"
    child.mkdir(parents=True)
    missing.mkdir()
    (child / "README.md").write_text("# Pinned reference\n")
    (missing / "README.md").write_text("# Local copy\n")
    git(child, "init", "-q")
    git(child, "add", "--", "README.md")
    git(child, "-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "fixture")
    git(child, "remote", "add", "origin", "https://username:private-value@example.invalid/child.git")
    result = subprocess.run([sys.executable, str(INVENTORY_SCRIPT), "--root", str(root), "--workers", "2"],
                            capture_output=True, text=True, check=True)
    rows = {r["project"]:r for r in json.loads(result.stdout)}
    assert rows["pinned"]["commit_sha"] == git(child, "rev-parse", "HEAD").stdout.strip()
    assert rows["pinned"]["source_url"] == "https://example.invalid/child.git"
    assert rows["pinned"]["relative_path"] == "pinned"
    assert "private-value" not in result.stdout
    assert rows["unversioned"]["commit_sha"] == ""
    assert rows["unversioned"]["source_url"] == ""
    assert rows["unversioned"]["status"] == "needs_source_version"
