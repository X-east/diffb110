from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from rdkit import Chem


FINISHED_RE = re.compile(
    r"Finished generation\. total_generated=(?P<total>\d+), "
    r"sdf_written=(?P<written>\d+), reconstruction_failed=(?P<failed>\d+)"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", type=Path, required=True)
    return parser.parse_args()


def _steps_from_config(config_path: Path) -> int:
    text = config_path.read_text(encoding="utf-8")
    match = re.search(r"num_steps:\s*(\d+)", text)
    if match is None:
        raise ValueError(f"num_steps not found in {config_path}")
    return int(match.group(1))


def _count_sdf(sdf_path: Path) -> tuple[int, int, int, int]:
    if not sdf_path.exists():
        return 0, 0, 0, 0
    text = sdf_path.read_text(errors="ignore")
    sdf_records = text.count("$$$$")
    supplier = Chem.SDMolSupplier(str(sdf_path), sanitize=False, removeHs=False)
    readable = 0
    sanitize_valid = 0
    unique_smiles: set[str] = set()
    for mol in supplier:
        if mol is None:
            continue
        readable += 1
        try:
            mol_copy = Chem.Mol(mol)
            Chem.SanitizeMol(mol_copy)
        except Exception:
            continue
        sanitize_valid += 1
        smiles = Chem.MolToSmiles(mol_copy, canonical=True)
        if smiles:
            unique_smiles.add(smiles)
    return sdf_records, readable, sanitize_valid, len(unique_smiles)


def _parse_finished_line(log_path: Path) -> tuple[int, int, int]:
    if not log_path.exists():
        return 0, 0, 0
    text = log_path.read_text(errors="ignore")
    matches = list(FINISHED_RE.finditer(text))
    if not matches:
        return 0, 0, 0
    match = matches[-1]
    return int(match.group("total")), int(match.group("written")), int(match.group("failed"))


def _write_plot(summary_csv: Path, output_png: Path) -> None:
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(summary_csv.open()))
    steps = [int(row["steps"]) for row in rows]
    sdf_rate = [float(row["sdf_written_rate"]) for row in rows]
    sanitize_rate = [float(row["sanitize_valid_rate"]) for row in rows]
    recon_fail_rate = [float(row["reconstruction_failed_rate"]) for row in rows]

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(steps, sdf_rate, marker="o", label="SDF written")
    ax.plot(steps, sanitize_rate, marker="o", label="sanitize-valid")
    ax.plot(steps, recon_fail_rate, marker="o", label="reconstruction failed")
    ax.set_xlabel("Diffusion steps")
    ax.set_ylabel("Fraction of requested samples")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_png, dpi=200)


def _write_report(summary_csv: Path, plot_path: Path, report_path: Path) -> None:
    rows = list(csv.DictReader(summary_csv.open()))
    headers = [
        "steps",
        "total_generated",
        "sdf_written",
        "reconstruction_failed",
        "sdf_records",
        "rdkit_readable",
        "sanitize_valid",
        "unique_smiles",
    ]
    lines = [
        "# TargetDiff diffusion-step sweep",
        "",
        "Fixed protocol: 8GJL/C1101, `num_samples=100`, `seed=1`, `batch_size=5`; only `sample.num_steps` changes.",
        "",
        "![TargetDiff success rates](targetdiff_step_success_rates.png)",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[h] for h in headers) + " |")
    lines.extend(
        [
            "",
            f"CSV: [{summary_csv.name}]({summary_csv.name})",
            f"Plot: [{plot_path.name}]({plot_path.name})",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    exp = args.exp.resolve()
    rows: list[dict[str, str]] = []
    for config_path in sorted((exp / "configs").glob("targetdiff_c1101_100mol_steps_*_seed*.yml")):
        steps = _steps_from_config(config_path)
        output_dir = exp / "outputs" / f"steps_{steps}"
        log_path = exp / "logs" / f"targetdiff_steps_{steps}.log"
        status_path = exp / "logs" / f"targetdiff_steps_{steps}.status"
        sdf_path = output_dir / "generated_molecules.sdf"
        total_generated, sdf_written, reconstruction_failed = _parse_finished_line(log_path)
        sdf_records, rdkit_readable, sanitize_valid, unique_smiles = _count_sdf(sdf_path)
        requested = 100
        row = {
            "steps": str(steps),
            "status": status_path.read_text(encoding="utf-8").strip() if status_path.exists() else "unknown",
            "requested": str(requested),
            "total_generated": str(total_generated),
            "sdf_written": str(sdf_written),
            "reconstruction_failed": str(reconstruction_failed),
            "sdf_records": str(sdf_records),
            "rdkit_readable": str(rdkit_readable),
            "sanitize_valid": str(sanitize_valid),
            "unique_smiles": str(unique_smiles),
            "sdf_written_rate": f"{sdf_written / requested:.6f}",
            "sanitize_valid_rate": f"{sanitize_valid / requested:.6f}",
            "reconstruction_failed_rate": f"{reconstruction_failed / requested:.6f}",
            "config": str(config_path),
            "sdf": str(sdf_path),
            "log": str(log_path),
        }
        rows.append(row)

    rows.sort(key=lambda item: int(item["steps"]))
    summary_dir = exp / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = summary_dir / "targetdiff_step_sweep_summary.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    plot_path = summary_dir / "targetdiff_step_success_rates.png"
    report_path = summary_dir / "report.md"
    if rows:
        _write_plot(summary_csv, plot_path)
        _write_report(summary_csv, plot_path, report_path)
    print(f"wrote {summary_csv}")
    print(f"wrote {plot_path}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
