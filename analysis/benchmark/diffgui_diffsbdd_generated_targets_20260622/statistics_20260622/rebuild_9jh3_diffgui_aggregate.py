"""Rebuild the 9JH3 DiffGui aggregate SDF while preserving SDF headers."""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
SOURCE_DIR = REPO_ROOT / "analysis/benchmark/9jh3_benchmark_20260617/outputs/diffgui/diffgui_denovo_20260617_114329/home_SDF"
OUTPUTS = [
    REPO_ROOT / "analysis/benchmark/9jh3_benchmark_20260617/outputs/diffgui_9jh3_200steps.sdf",
    REPO_ROOT / "analysis/benchmark/diffgui_diffsbdd_generated_targets_20260622/9JH3_R401_DiffGui_500mol_200steps.sdf",
]
SUMMARY = REPO_ROOT / "analysis/benchmark/9jh3_benchmark_20260617/outputs/diffgui_9jh3_200steps.summary.json"


def _numeric_sdf_files(source_dir: Path) -> list[Path]:
    return sorted(source_dir.glob("*.sdf"), key=lambda path: int(path.stem))


def _normalized_record(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    while text.endswith("\n"):
        text = text[:-1]
    return f"{text}\n$$$$\n"


def main() -> None:
    files = _numeric_sdf_files(SOURCE_DIR)
    if not files:
        raise FileNotFoundError(f"No SDF files found under {SOURCE_DIR}")
    payload = "".join(_normalized_record(path) for path in files)
    for output in OUTPUTS:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    summary = {
        "source_dir": str(SOURCE_DIR),
        "outfile": str(OUTPUTS[0]),
        "curated_outfile": str(OUTPUTS[1]),
        "input_files": len(files),
        "written": len(files),
        "empty_files_skipped": 0,
        "timesteps": 200,
        "seed": 20260617,
        "format_note": "Rebuilt from original one-molecule SDF files without stripping leading title/header lines; appended $$$$ after each record.",
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
