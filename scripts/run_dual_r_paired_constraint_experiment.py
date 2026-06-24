"""Run DiffSBDD paired atom-type/root-distance dual-R inpaint experiments."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ExperimentSetting:
    """One paired-constraint generation setting."""

    name: str
    atom_sync_strength: float
    distance_strength: float
    max_distance_step: float
    enabled: bool = True
    target_atom_types: str = ""  # space-separated, e.g. "C O C O"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--gpu", default="2")
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--timesteps", type=int, default=50)
    parser.add_argument("--resamplings", type=int, default=5)
    parser.add_argument("--add-n-nodes", type=int, default=4)
    parser.add_argument("--pair-count", type=int, default=2)
    parser.add_argument("--checkpoint", type=Path, default=Path("DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt"))
    parser.add_argument("--pdbfile", type=Path, default=Path("data/4dxd_9pc.pdb"))
    parser.add_argument(
        "--ref-ligand",
        type=Path,
        default=Path("analysis/honokiol_9pc_docking_20260617/docking/successful_docked_outputs.sdf"),
    )
    parser.add_argument(
        "--fixed-core-sdf",
        type=Path,
        default=Path("analysis/dual_sidechain_honokiol_9pc_20260617/inpaint_fixed_core_inputs/fixed_core_no_oh.sdf"),
    )
    parser.add_argument("--site-reference-index", type=int, default=0)
    parser.add_argument("--left-site-index", type=int, default=0)
    parser.add_argument("--right-site-index", type=int, default=1)
    parser.add_argument("--left-core-idx", type=int, default=0)
    parser.add_argument("--right-core-idx", type=int, default=11)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Run generation and audit settings."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Target atom types: carbon chain ending with heteroatom
    # pair_count=n → n atoms per side: "C C ... C O"
    target_types = " ".join(["C"] * (args.pair_count - 1) + ["O"])

    settings = [
        ExperimentSetting("baseline", 0.0, 0.0, 0.0, enabled=False),
        ExperimentSetting("atom_sync_only", 1.0, 0.0, 0.0, enabled=True, target_atom_types=target_types),
        ExperimentSetting("atom_sync_distance", 1.0, 0.35, 0.30, enabled=True, target_atom_types=target_types),
    ]
    rows = []
    commands = []
    for setting in settings:
        setting_dir = args.out_dir / setting.name
        setting_dir.mkdir(parents=True, exist_ok=True)
        sdf_path = setting_dir / f"{setting.name}.sdf"
        gen_log = setting_dir / "generation.log"
        audit_dir = setting_dir / "audit"
        generation_cmd = _generation_command(args, setting, sdf_path)
        audit_cmd = _audit_command(args, sdf_path, audit_dir)
        commands.append({"setting": setting.name, "generation": generation_cmd, "audit": audit_cmd})
        if not args.dry_run:
            _run_command(generation_cmd, gen_log, args.gpu)
            _run_command(audit_cmd, setting_dir / "audit.log", args.gpu)
        summary_path = audit_dir / "summary.json"
        summary = _read_summary(summary_path) if summary_path.exists() else {}
        rows.append(_summary_row(setting, sdf_path, gen_log, audit_dir, summary))

    _write_json(args.out_dir / "commands.json", commands)
    _write_csv(args.out_dir / "experiment_summary.csv", rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


def _generation_command(
    args: argparse.Namespace,
    setting: ExperimentSetting,
    sdf_path: Path,
) -> list[str]:
    """Build one generation command."""
    cmd = [
        sys.executable,
        "DiffSBDD/inpaint.py",
        str(args.checkpoint),
        "--pdbfile",
        str(args.pdbfile),
        "--ref_ligand",
        str(args.ref_ligand),
        "--fix_atoms",
        str(args.fixed_core_sdf),
        "--center",
        "ligand",
        "--add_n_nodes",
        str(args.add_n_nodes),
        "--n_samples",
        str(args.n_samples),
        "--timesteps",
        str(args.timesteps),
        "--resamplings",
        str(args.resamplings),
        "--sanitize",
        "--outfile",
        str(sdf_path),
    ]
    if setting.enabled:
        cmd.extend([
            "--dual_r_constraint",
            "--dual_r_pair_count",
            str(args.pair_count),
            "--dual_r_left_core_idx",
            str(args.left_core_idx),
            "--dual_r_right_core_idx",
            str(args.right_core_idx),
            "--dual_r_atom_sync_strength",
            str(setting.atom_sync_strength),
            "--dual_r_distance_strength",
            str(setting.distance_strength),
            "--dual_r_max_distance_step",
            str(setting.max_distance_step),
        ])
        if setting.target_atom_types:
            cmd.extend([
                "--dual_r_target_atom_types",
                setting.target_atom_types,
            ])
    return cmd


def _audit_command(
    args: argparse.Namespace,
    sdf_path: Path,
    audit_dir: Path,
) -> list[str]:
    """Build one generated-candidate audit command."""
    return [
        sys.executable,
        "scripts/analyze_generated_dual_r_candidates.py",
        "--sdf",
        str(sdf_path),
        "--site-reference-sdf",
        str(args.ref_ligand),
        "--site-reference-index",
        str(args.site_reference_index),
        "--out-dir",
        str(audit_dir),
        "--left-site-index",
        str(args.left_site_index),
        "--right-site-index",
        str(args.right_site_index),
        "--write-success-sdf",
        "--write-failure-sdf",
        "--write-audited-sdf",
    ]


def _run_command(cmd: list[str], log_path: Path, gpu: str) -> None:
    """Run a command and write a durable log file."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-dual-r-paired")
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write("$ " + " ".join(cmd) + "\n\n")
        log_handle.flush()
        completed = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log_handle.write(f"\n[exit_code] {completed.returncode}\n")
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(cmd)}")


def _read_summary(path: Path) -> dict[str, Any]:
    """Read generated audit summary."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload.get("summary", {})


def _summary_row(
    setting: ExperimentSetting,
    sdf_path: Path,
    gen_log: Path,
    audit_dir: Path,
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Return a flat experiment summary row."""
    return {
        "setting": setting.name,
        "atom_sync_strength": setting.atom_sync_strength,
        "distance_strength": setting.distance_strength,
        "max_distance_step": setting.max_distance_step,
        "sdf": str(sdf_path),
        "generation_log": str(gen_log),
        "audit_dir": str(audit_dir),
        "total": summary.get("total", 0),
        "mapping_rdkit_valid": summary.get("mapping_rdkit_valid", 0),
        "core_found": summary.get("core_found", 0),
        "dual_r_success": summary.get("dual_r_success", 0),
        "dual_r_success_rate": summary.get("dual_r_success_rate", 0.0),
        "left_extractable": summary.get("audit_summary", {}).get("left_extractable", 0),
        "right_extractable": summary.get("audit_summary", {}).get("right_extractable", 0),
        "atom_symbols_match": summary.get("audit_summary", {}).get("atom_symbols_match", 0),
        "graph_signature_match": summary.get("audit_summary", {}).get("graph_signature_match", 0),
        "failure_counts": json.dumps(summary.get("audit_summary", {}).get("failure_counts", {}), ensure_ascii=False),
    }


def _write_json(path: Path, payload: Any) -> None:
    """Write JSON payload."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write summary CSV."""
    if not rows:
        raise ValueError("rows must not be empty")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
