"""Validate CrossDocked edge labels for local bond predictor training."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from rdkit import Chem, RDLogger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIFFSBDD_ROOT = Path(__file__).resolve().parents[1]
for extra in (PROJECT_ROOT, DIFFSBDD_ROOT):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import constants
from dataset import ProcessedLigandPocketDataset
from edge_utils import BOND_TYPE_NAMES
from analysis.edge_data_oracle_report import (  # noqa: E402
    _atom_symbols,
    _build_oracle_mol,
    _lookup_reference_length,
)

LOGGER = logging.getLogger(__name__)
SPLITS = ("train", "val", "test")

RDLogger.DisableLog("rdApp.*")


@dataclass
class SplitReport:
    """Aggregated validation statistics for one processed data split."""

    split: str
    npz_path: Path
    dataset_samples: int
    analyzed_samples: int
    elapsed_sec: float
    total_atoms: int
    total_halfedges: int
    total_real_bonds: int
    sanitize_success: int
    unsupported_atom_type: int
    sample_error_count: int
    halfedge_mismatch_count: int
    bond_count_mismatch_count: int
    bad_length_count: int
    checked_length_count: int
    mean_bond_length: float
    p05_bond_length: float
    p50_bond_length: float
    p95_bond_length: float

    @property
    def no_bond_fraction(self) -> float:
        """Return the fraction of halfedges labelled as no-bond."""
        if self.total_halfedges == 0:
            return 0.0
        return 1.0 - self.total_real_bonds / self.total_halfedges

    @property
    def sanitize_success_rate(self) -> float:
        """Return oracle reconstruction sanitize success rate."""
        if self.analyzed_samples == 0:
            return 0.0
        return self.sanitize_success / self.analyzed_samples

    @property
    def bad_length_rate(self) -> float:
        """Return bad bond length rate among checked reference pairs."""
        if self.checked_length_count == 0:
            return 0.0
        return self.bad_length_count / self.checked_length_count

    @property
    def samples_per_sec(self) -> float:
        """Return analyzed molecule throughput."""
        if self.elapsed_sec <= 0:
            return 0.0
        return self.analyzed_samples / self.elapsed_sec

    def to_row(self) -> dict[str, object]:
        """Return a CSV-ready row."""
        return {
            "split": self.split,
            "npz_path": str(self.npz_path),
            "dataset_samples": self.dataset_samples,
            "analyzed_samples": self.analyzed_samples,
            "elapsed_sec": f"{self.elapsed_sec:.3f}",
            "samples_per_sec": f"{self.samples_per_sec:.3f}",
            "total_atoms": self.total_atoms,
            "mean_atoms": f"{self.total_atoms / max(1, self.analyzed_samples):.3f}",
            "total_halfedges": self.total_halfedges,
            "total_real_bonds": self.total_real_bonds,
            "no_bond_fraction": f"{self.no_bond_fraction:.8f}",
            "sanitize_success": self.sanitize_success,
            "sanitize_success_rate": f"{self.sanitize_success_rate:.8f}",
            "unsupported_atom_type": self.unsupported_atom_type,
            "sample_error_count": self.sample_error_count,
            "halfedge_mismatch_count": self.halfedge_mismatch_count,
            "bond_count_mismatch_count": self.bond_count_mismatch_count,
            "bad_length_count": self.bad_length_count,
            "checked_length_count": self.checked_length_count,
            "bad_length_rate": f"{self.bad_length_rate:.8f}",
            "mean_bond_length": f"{self.mean_bond_length:.6f}",
            "p05_bond_length": f"{self.p05_bond_length:.6f}",
            "p50_bond_length": f"{self.p50_bond_length:.6f}",
            "p95_bond_length": f"{self.p95_bond_length:.6f}",
        }


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    """Write CSV rows using a stable column order."""
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _choose_indices(length: int, max_samples: int | None, seed: int) -> list[int]:
    """Choose a reproducible subset of sample indices."""
    if max_samples is None or max_samples >= length:
        return list(range(length))
    rng = random.Random(seed)
    return sorted(rng.sample(range(length), max_samples))


def _quantile(values: list[float], q: float) -> float:
    """Return a numeric quantile or zero when no values are available."""
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _iter_bond_lengths(
    split: str,
    sample_idx: int,
    atom_symbols: list[str],
    coords: torch.Tensor,
    halfedge_index: torch.Tensor,
    halfedge_type: torch.Tensor,
    length_tolerance: float,
) -> Iterable[dict[str, object]]:
    """Yield per-bond length rows for one molecule."""
    for edge_pos, edge_type_tensor in enumerate(halfedge_type):
        edge_type = int(edge_type_tensor)
        if edge_type <= 0:
            continue
        atom_i = int(halfedge_index[0, edge_pos])
        atom_j = int(halfedge_index[1, edge_pos])
        atom_a, atom_b = atom_symbols[atom_i], atom_symbols[atom_j]
        length = float(torch.linalg.norm(coords[atom_i] - coords[atom_j]).item())
        ref_length = _lookup_reference_length(atom_a, atom_b, edge_type)
        is_checked = ref_length is not None
        is_bad = is_checked and abs(length - ref_length) > length_tolerance
        yield {
            "split": split,
            "sample_idx": sample_idx,
            "atom_i": atom_i,
            "atom_j": atom_j,
            "atom_a": atom_a,
            "atom_b": atom_b,
            "bond_type_id": edge_type,
            "bond_type": BOND_TYPE_NAMES[edge_type],
            "length": f"{length:.6f}",
            "reference_length": "" if ref_length is None else f"{ref_length:.6f}",
            "is_checked": int(is_checked),
            "is_bad_length": int(is_bad),
        }


def _plot_grouped_rates(
    path: Path,
    split_rows: list[dict[str, object]],
    metric: str,
    ylabel: str,
) -> None:
    """Plot one rate-like metric by split."""
    labels = [str(row["split"]) for row in split_rows]
    values = [float(row[metric]) for row in split_rows]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, values, color="#4C78A8")
    plt.ylabel(ylabel)
    plt.ylim(0, max(1.0, max(values, default=0.0) * 1.15))
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_bond_type_fractions(
    path: Path,
    rows: list[dict[str, object]],
    include_no_bond: bool,
) -> None:
    """Plot class fractions by split for halfedge labels."""
    bond_ids = sorted(BOND_TYPE_NAMES)
    if not include_no_bond:
        bond_ids = [bond_id for bond_id in bond_ids if bond_id > 0]
    splits = [split for split in SPLITS if any(row["split"] == split for row in rows)]
    x = np.arange(len(splits))
    width = 0.8 / max(1, len(bond_ids))

    plt.figure(figsize=(8, 4.5))
    for offset, bond_id in enumerate(bond_ids):
        values = []
        for split in splits:
            match = next(
                (
                    row for row in rows
                    if row["split"] == split and int(row["bond_type_id"]) == bond_id
                ),
                None,
            )
            values.append(0.0 if match is None else float(match["fraction"]))
        plt.bar(
            x + (offset - (len(bond_ids) - 1) / 2) * width,
            values,
            width=width,
            label=BOND_TYPE_NAMES[bond_id],
        )
    plt.xticks(x, splits)
    plt.ylabel("fraction")
    plt.legend(frameon=False, ncols=3)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_bond_length_histogram(path: Path, rows: list[dict[str, object]]) -> None:
    """Plot bond length distributions by bond type."""
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["bond_type"])].append(float(row["length"]))

    plt.figure(figsize=(8, 4.5))
    for bond_type, values in sorted(grouped.items()):
        if not values:
            continue
        plt.hist(values, bins=50, alpha=0.45, label=bond_type)
    plt.xlabel("bond length (Angstrom)")
    plt.ylabel("count")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _validate_split(
    split: str,
    npz_path: Path,
    dataset_name: str,
    max_samples: int | None,
    seed: int,
    length_tolerance: float,
) -> tuple[
    SplitReport,
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    Counter[int],
]:
    """Validate one processed split and return report rows."""
    dataset_info = constants.dataset_params[dataset_name]
    dataset = ProcessedLigandPocketDataset(npz_path, center=False)
    indices = _choose_indices(len(dataset), max_samples, seed)

    bond_type_counts: Counter[int] = Counter()
    molecule_rows: list[dict[str, object]] = []
    length_rows: list[dict[str, object]] = []
    pair_lengths: dict[str, list[float]] = defaultdict(list)
    error_counts: Counter[str] = Counter()

    total_atoms = 0
    total_halfedges = 0
    total_real_bonds = 0
    sanitize_success = 0
    unsupported_atom_type = 0
    sample_error_count = 0
    halfedge_mismatch_count = 0
    bond_count_mismatch_count = 0
    bad_length_count = 0
    checked_length_count = 0
    length_values: list[float] = []

    start = time.perf_counter()
    for sample_idx in indices:
        try:
            sample = dataset[sample_idx]
            atom_symbols = _atom_symbols(sample["lig_one_hot"], dataset_info["atom_decoder"])
            halfedge_type = sample["ligand_halfedge_type"]
            halfedge_index = sample["ligand_halfedge_index"]
            coords = sample["lig_coords"].float()
        except Exception as exc:  # noqa: BLE001 - collect data-chain failures
            sample_error_count += 1
            error_counts[type(exc).__name__] += 1
            molecule_rows.append({
                "split": split,
                "sample_idx": sample_idx,
                "name": "",
                "num_atoms": "",
                "num_bonds": "",
                "num_halfedges": "",
                "oracle_sanitize_success": 0,
                "sanitize_error": type(exc).__name__,
                "smiles": "",
            })
            continue

        num_atoms = int(sample["num_lig_atoms"])
        expected_halfedges = num_atoms * (num_atoms - 1) // 2
        actual_halfedges = int(len(halfedge_type))
        halfedge_mismatch = int(actual_halfedges != expected_halfedges)
        halfedge_mismatch_count += halfedge_mismatch

        real_bonds = int((halfedge_type > 0).sum().item())
        num_lig_bonds = int(sample.get("num_lig_bonds", torch.tensor(real_bonds)))
        bond_count_mismatch = int(real_bonds != num_lig_bonds)
        bond_count_mismatch_count += bond_count_mismatch

        total_atoms += num_atoms
        total_halfedges += actual_halfedges
        total_real_bonds += real_bonds
        bond_type_counts.update(int(x) for x in halfedge_type.tolist())

        for row in _iter_bond_lengths(
            split,
            sample_idx,
            atom_symbols,
            coords,
            halfedge_index,
            halfedge_type,
            length_tolerance,
        ):
            length_rows.append(row)
            length = float(row["length"])
            length_values.append(length)
            atom_pair = "-".join(sorted((str(row["atom_a"]), str(row["atom_b"]))))
            pair_key = f"{atom_pair}|{row['bond_type']}"
            pair_lengths[pair_key].append(length)
            if int(row["is_checked"]):
                checked_length_count += 1
                bad_length_count += int(row["is_bad_length"])

        mol, sanitize_error = _build_oracle_mol(atom_symbols, coords, halfedge_index, halfedge_type)
        if sanitize_error is None:
            sanitize_success += 1
        else:
            error_counts[sanitize_error] += 1
        if sanitize_error == "unsupported_atom_type":
            unsupported_atom_type += 1

        molecule_rows.append({
            "split": split,
            "sample_idx": sample_idx,
            "name": str(sample["names"]),
            "num_atoms": num_atoms,
            "num_bonds": real_bonds,
            "num_halfedges": actual_halfedges,
            "halfedge_mismatch": halfedge_mismatch,
            "bond_count_mismatch": bond_count_mismatch,
            "oracle_sanitize_success": int(sanitize_error is None),
            "sanitize_error": "" if sanitize_error is None else sanitize_error,
            "smiles": "" if mol is None or sanitize_error is not None else Chem.MolToSmiles(mol),
        })

    elapsed_sec = time.perf_counter() - start
    report = SplitReport(
        split=split,
        npz_path=npz_path,
        dataset_samples=len(dataset),
        analyzed_samples=len(indices),
        elapsed_sec=elapsed_sec,
        total_atoms=total_atoms,
        total_halfedges=total_halfedges,
        total_real_bonds=total_real_bonds,
        sanitize_success=sanitize_success,
        unsupported_atom_type=unsupported_atom_type,
        sample_error_count=sample_error_count,
        halfedge_mismatch_count=halfedge_mismatch_count,
        bond_count_mismatch_count=bond_count_mismatch_count,
        bad_length_count=bad_length_count,
        checked_length_count=checked_length_count,
        mean_bond_length=float(np.mean(length_values)) if length_values else 0.0,
        p05_bond_length=_quantile(length_values, 5),
        p50_bond_length=_quantile(length_values, 50),
        p95_bond_length=_quantile(length_values, 95),
    )

    pair_rows = [
        {
            "split": split,
            "pair_bond_type": key,
            "count": len(values),
            "mean_length": f"{float(np.mean(values)):.6f}",
            "p05_length": f"{_quantile(values, 5):.6f}",
            "p50_length": f"{_quantile(values, 50):.6f}",
            "p95_length": f"{_quantile(values, 95):.6f}",
            "min_length": f"{min(values):.6f}",
            "max_length": f"{max(values):.6f}",
        }
        for key, values in sorted(pair_lengths.items())
    ]
    error_rows = [
        {"split": split, "error": error, "count": count}
        for error, count in sorted(error_counts.items())
    ]
    return report, molecule_rows, length_rows, pair_rows, error_rows, bond_type_counts


def _write_manifest(
    path: Path,
    processed_dir: Path,
    split_reports: list[SplitReport],
    args: argparse.Namespace,
) -> None:
    """Write a manifest that describes the predictor data source."""
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "processed_dir": str(processed_dir),
        "dataset": args.dataset,
        "length_tolerance": args.length_tolerance,
        "sample_seed": args.sample_seed,
        "representation": {
            "source_bond_keys": ["lig_bond_index", "lig_bond_type", "num_lig_bonds"],
            "on_the_fly_halfedge_keys": [
                "ligand_halfedge_index",
                "ligand_halfedge_type",
                "num_ligand_halfedges",
            ],
            "class_labels": {str(key): value for key, value in BOND_TYPE_NAMES.items()},
        },
        "splits": [report.to_row() for report in split_reports],
    }
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


def _write_markdown_report(
    path: Path,
    split_rows: list[dict[str, object]],
    bond_type_rows: list[dict[str, object]],
    error_rows: list[dict[str, object]],
    args: argparse.Namespace,
) -> None:
    """Write a Markdown report with links to generated evidence files."""
    real_bond_rows = [row for row in bond_type_rows if int(row["bond_type_id"]) > 0]
    with path.open("w") as handle:
        handle.write("# Local Bond Predictor Data Report\n\n")
        handle.write("## Purpose\n\n")
        handle.write(
            "Validate that CrossDocked processed edge labels can provide a stable "
            "`ligand_halfedge_index / ligand_halfedge_type` data chain for a local "
            "bond predictor baseline.\n\n"
        )
        handle.write("## Inputs\n\n")
        handle.write(f"- processed_dir: `{args.processed_dir}`\n")
        handle.write(f"- dataset: `{args.dataset}`\n")
        handle.write(f"- train_max_samples: `{args.train_max_samples}`\n")
        handle.write(f"- val_max_samples: `{args.val_max_samples}`\n")
        handle.write(f"- test_max_samples: `{args.test_max_samples}`\n")
        handle.write(f"- length_tolerance: `{args.length_tolerance}` Angstrom\n")
        handle.write(f"- sample_seed: `{args.sample_seed}`\n\n")

        handle.write("## Split Summary\n\n")
        handle.write(
            "| split | dataset | analyzed | sanitize | bad length | no-bond | "
            "halfedge mismatch | bond mismatch | samples/s |\n"
        )
        handle.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in split_rows:
            handle.write(
                f"| {row['split']} | {row['dataset_samples']} | {row['analyzed_samples']} | "
                f"{row['sanitize_success_rate']} | {row['bad_length_rate']} | "
                f"{row['no_bond_fraction']} | {row['halfedge_mismatch_count']} | "
                f"{row['bond_count_mismatch_count']} | {row['samples_per_sec']} |\n"
            )
        handle.write("\n")

        handle.write("## Real Bond Type Fractions\n\n")
        handle.write("| split | single | double | triple | aromatic |\n")
        handle.write("|---|---:|---:|---:|---:|\n")
        for split in SPLITS:
            rows = [row for row in real_bond_rows if row["split"] == split]
            if not rows:
                continue
            by_id = {int(row["bond_type_id"]): row for row in rows}
            handle.write(
                f"| {split} | "
                f"{by_id.get(1, {}).get('real_bond_fraction', '0.00000000')} | "
                f"{by_id.get(2, {}).get('real_bond_fraction', '0.00000000')} | "
                f"{by_id.get(3, {}).get('real_bond_fraction', '0.00000000')} | "
                f"{by_id.get(4, {}).get('real_bond_fraction', '0.00000000')} |\n"
            )
        handle.write("\n")

        if error_rows:
            handle.write("## Sanitize Errors\n\n")
            handle.write("| split | error | count |\n")
            handle.write("|---|---|---:|\n")
            for row in error_rows:
                handle.write(f"| {row['split']} | {row['error']} | {row['count']} |\n")
            handle.write("\n")

        handle.write("## Files\n\n")
        handle.write("- [split_summary.csv](split_summary.csv)\n")
        handle.write("- [bond_type_counts_by_split.csv](bond_type_counts_by_split.csv)\n")
        handle.write("- [bond_length_by_split.csv](bond_length_by_split.csv)\n")
        handle.write("- [bond_pair_length_summary.csv](bond_pair_length_summary.csv)\n")
        handle.write("- [oracle_reconstruction_by_split.csv](oracle_reconstruction_by_split.csv)\n")
        handle.write("- [sanitize_errors_by_split.csv](sanitize_errors_by_split.csv)\n")
        handle.write("- [local_bond_predictor_manifest.json](local_bond_predictor_manifest.json)\n\n")

        handle.write("## Plots\n\n")
        handle.write("![halfedge class fraction](halfedge_class_fraction_by_split.png)\n\n")
        handle.write("![real bond class fraction](real_bond_type_fraction_by_split.png)\n\n")
        handle.write("![oracle sanitize](oracle_sanitize_success_by_split.png)\n\n")
        handle.write("![bad bond length](bad_bond_length_rate_by_split.png)\n\n")
        handle.write("![bond length distribution](bond_length_distribution_by_type.png)\n")


def generate_report(args: argparse.Namespace) -> Path:
    """Generate split-level local bond predictor data validation artifacts."""
    args.outdir.mkdir(parents=True, exist_ok=True)
    split_reports: list[SplitReport] = []
    molecule_rows: list[dict[str, object]] = []
    length_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    error_rows: list[dict[str, object]] = []
    bond_type_rows: list[dict[str, object]] = []

    max_by_split = {
        "train": args.train_max_samples,
        "val": args.val_max_samples,
        "test": args.test_max_samples,
    }
    for split in SPLITS:
        npz_path = args.processed_dir / f"{split}.npz"
        LOGGER.info("validating split=%s path=%s", split, npz_path)
        (
            split_report,
            split_molecule_rows,
            split_length_rows,
            split_pair_rows,
            split_error_rows,
            split_bond_type_counts,
        ) = _validate_split(
            split,
            npz_path,
            args.dataset,
            max_by_split[split],
            args.sample_seed,
            args.length_tolerance,
        )
        split_reports.append(split_report)
        molecule_rows.extend(split_molecule_rows)
        length_rows.extend(split_length_rows)
        pair_rows.extend(split_pair_rows)
        error_rows.extend(split_error_rows)

        total_halfedges = max(1, sum(split_bond_type_counts.values()))
        total_real_bonds = max(
            1,
            sum(count for bond_id, count in split_bond_type_counts.items() if bond_id > 0),
        )
        for bond_id in sorted(BOND_TYPE_NAMES):
            count = split_bond_type_counts.get(bond_id, 0)
            real_fraction = 0.0 if bond_id == 0 else count / total_real_bonds
            bond_type_rows.append({
                "split": split,
                "bond_type_id": bond_id,
                "bond_type": BOND_TYPE_NAMES[bond_id],
                "count": count,
                "fraction": f"{count / total_halfedges:.8f}",
                "real_bond_fraction": "" if bond_id == 0 else f"{real_fraction:.8f}",
            })

    split_rows = [report.to_row() for report in split_reports]
    _write_csv(
        args.outdir / "split_summary.csv",
        split_rows,
        list(split_rows[0].keys()),
    )
    _write_csv(
        args.outdir / "bond_type_counts_by_split.csv",
        bond_type_rows,
        ["split", "bond_type_id", "bond_type", "count", "fraction", "real_bond_fraction"],
    )
    _write_csv(
        args.outdir / "bond_length_by_split.csv",
        length_rows,
        [
            "split", "sample_idx", "atom_i", "atom_j", "atom_a", "atom_b",
            "bond_type_id", "bond_type", "length", "reference_length",
            "is_checked", "is_bad_length",
        ],
    )
    _write_csv(
        args.outdir / "bond_pair_length_summary.csv",
        pair_rows,
        [
            "split", "pair_bond_type", "count", "mean_length", "p05_length",
            "p50_length", "p95_length", "min_length", "max_length",
        ],
    )
    _write_csv(
        args.outdir / "oracle_reconstruction_by_split.csv",
        molecule_rows,
        [
            "split", "sample_idx", "name", "num_atoms", "num_bonds", "num_halfedges",
            "halfedge_mismatch", "bond_count_mismatch", "oracle_sanitize_success",
            "sanitize_error", "smiles",
        ],
    )
    _write_csv(
        args.outdir / "sanitize_errors_by_split.csv",
        error_rows,
        ["split", "error", "count"],
    )
    _write_manifest(
        args.outdir / "local_bond_predictor_manifest.json",
        args.processed_dir,
        split_reports,
        args,
    )
    _plot_bond_type_fractions(
        args.outdir / "halfedge_class_fraction_by_split.png",
        bond_type_rows,
        include_no_bond=True,
    )
    _plot_bond_type_fractions(
        args.outdir / "real_bond_type_fraction_by_split.png",
        bond_type_rows,
        include_no_bond=False,
    )
    _plot_grouped_rates(
        args.outdir / "oracle_sanitize_success_by_split.png",
        split_rows,
        "sanitize_success_rate",
        "sanitize success rate",
    )
    _plot_grouped_rates(
        args.outdir / "bad_bond_length_rate_by_split.png",
        split_rows,
        "bad_length_rate",
        "bad bond length rate",
    )
    _plot_bond_length_histogram(args.outdir / "bond_length_distribution_by_type.png", length_rows)

    report_path = args.outdir / "local_bond_predictor_data_report.md"
    _write_markdown_report(report_path, split_rows, bond_type_rows, error_rows, args)
    return report_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/crossdocked_raw/processed_edge_noH_full"),
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(constants.dataset_params.keys()),
        default="crossdock_full",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("analysis") / f"local_bond_predictor_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument("--train-max-samples", type=int, default=10000)
    parser.add_argument("--val-max-samples", type=int, default=None)
    parser.add_argument("--test-max-samples", type=int, default=None)
    parser.add_argument("--length-tolerance", type=float, default=0.25)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    """Run the local bond predictor data validation CLI."""
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    report_path = generate_report(args)
    LOGGER.info("wrote report to %s", report_path)


if __name__ == "__main__":
    main()
