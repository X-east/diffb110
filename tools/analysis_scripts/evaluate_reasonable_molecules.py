"""Evaluate "reasonable molecules" selected by structural rules R1-R4.

Definition (aligned with the preference-pair construction in
``analysis/diffsbdd_structure_pairs_matched_260831``, where
``source_rule = structure_r1_r4_exact_formula_tpsa_dock_nn_v2``):

  reasonable molecule = molecule that can be sanitized AND does NOT violate
  ANY of the five screening rules R1-R5 (``is_pathological == False``).

    R1 large ring (>=7 atoms),  R2 ring system (>3 rings),  R3 four-membered
    ring,  R4 three-membered ring that is hetero / over-attached /
    spiro-or-fused,  R5 carbon fraction drift (carbon_fraction > 0.85).

  Reasonableness is decided ONLY by these five structural rules -- properties
  such as QED/SA/docking are NOT used to decide reasonableness; they are merely
  measured on the reasonable subset.

  The R1-R4-only count is additionally reported as a reference口径 because the
  preference-pair construction used R1-R4 as the discriminative signal (R5 was
  matched on both sides there).

All models (base / Original / E2PO intermediate steps) are judged with the SAME
rule, then only the reasonable subset is scored for properties and docking.

The success criterion is a yield-vs-quality trade-off:
  reasonable-molecule yield goes UP while the quality (properties + docking)
  of the reasonable subset stays comparable  ->  training succeeded.

Inputs are the relaxed SDFs (largest-fragment, "normal" molecules) together
with their docking result CSVs, because docking was run on the relaxed SDF and
``benchmark_index`` indexes that same SDF.

Usage:
  python tools/analysis_scripts/evaluate_reasonable_molecules.py \
      --input "Original:path/to/molecules_relaxed_largest.sdf:path/to/docking_results.csv:1000" \
      --input "E2PO_step2000:...:...:1000" \
      --out-dir analysis/e2po_reasonable_eval
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, FilterCatalog, Lipinski, QED, rdMolDescriptors

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "DiffSBDD" / "analysis" / "SA_Score"))

from DiffSBDD.analysis.SA_Score import sascorer  # noqa: E402
from scripts.offline_rl.structure_pathology_rules import (  # noqa: E402
    RULE_FIELDS,
    compute_structure_pathology,
)

RDLogger.DisableLog("rdApp.*")


def _brenk_catalog() -> FilterCatalog.FilterCatalog:
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
    return FilterCatalog.FilterCatalog(params)


def _bond_quality(mol: Chem.Mol) -> tuple[int, int, float]:
    """Return (bad_bonds, checked_bonds, bad_fraction) from 3D geometry."""
    if mol.GetNumConformers() == 0:
        return 0, 0, 0.0
    conformer = mol.GetConformer()
    table = Chem.GetPeriodicTable()
    bad = 0
    checked = 0
    for bond in mol.GetBonds():
        atom_a = bond.GetBeginAtom()
        atom_b = bond.GetEndAtom()
        if atom_a.GetAtomicNum() == 1 or atom_b.GetAtomicNum() == 1:
            continue
        ideal = table.GetRcovalent(atom_a.GetAtomicNum()) + table.GetRcovalent(atom_b.GetAtomicNum())
        if ideal <= 0:
            continue
        distance = conformer.GetAtomPosition(atom_a.GetIdx()).Distance(
            conformer.GetAtomPosition(atom_b.GetIdx())
        )
        checked += 1
        bad += int(distance / ideal < 0.70 or distance / ideal > 1.45)
    return bad, checked, bad / checked if checked else 0.0

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.edgecolor"] = "#d4d4d4"
plt.rcParams["axes.linewidth"] = 1.2

DIFFSBDD_RED = "#E74C3C"
ORIGINAL_BLUE = "#3498DB"

PROPERTY_KEYS = (
    "mw",
    "logp",
    "tpsa",
    "hba",
    "hbd",
    "qed",
    "sa_score",
    "heavy_atom_count",
    "ring_count",
    "aromatic_ring_count",
    "carbon_fraction",
)


def _finite(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def load_docking_scores(path: Path) -> dict[int, float]:
    """Map benchmark_index -> docking score (successful, finite, negative)."""
    scores: dict[int, float] = {}
    if not path.exists():
        return scores
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("success", "")).strip().lower() != "true":
                continue
            try:
                index = int(row["benchmark_index"])
                score = float(row["score"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(score) and score < 0:
                scores[index] = score
    return scores


def _raw_sdf_stats(raw_sdf: Path | None, attempts: int) -> dict:
    """Fragment rate and strict sanitize rate measured on the raw generated SDF.

    These must be measured before largest-fragment extraction, because the
    relaxed SDF has already had fragments removed.
    """
    if raw_sdf is None or str(raw_sdf) == "-" or not Path(raw_sdf).exists():
        return {
            "strict_sanitize_pct_of_attempts": math.nan,
            "fragment_pct_of_attempts": math.nan,
        }
    supplier = Chem.SDMolSupplier(str(raw_sdf), sanitize=False, removeHs=False)
    n_raw = 0
    strict_ok = 0
    multi_fragment = 0
    for parsed in supplier:
        n_raw += 1
        if parsed is None:
            continue
        try:
            mol = Chem.Mol(parsed)
            Chem.SanitizeMol(mol)
            strict_ok += 1
        except Exception:
            pass
        try:
            frags = Chem.GetMolFrags(parsed, asMols=True, sanitizeFrags=False)
            if len(frags) >= 2:
                multi_fragment += 1
        except Exception:
            pass
    denominator = attempts if attempts else n_raw
    return {
        "strict_sanitize_pct_of_attempts": round(100.0 * strict_ok / denominator, 2)
        if denominator
        else math.nan,
        "fragment_pct_of_attempts": round(100.0 * multi_fragment / denominator, 2)
        if denominator
        else math.nan,
    }


def analyze_run(
    label: str,
    raw_sdf: Path | None,
    relaxed_sdf: Path,
    docking_csv: Path,
    attempts: int,
) -> dict:
    """Select reasonable molecules by the five rules R1-R5 and score them.

    Only reasonable molecules (R1-R5 all clean) contribute to the property and
    docking statistics; unreasonable ones are skipped entirely.

    Violation rates for the individual rules R1-R5 (plus Brenk alerts and bad
    bond lengths) are reported over the relaxed molecules, because those are
    the readable/normal molecules before the reasonableness filter.
    """
    docking_scores = load_docking_scores(docking_csv)
    supplier = Chem.SDMolSupplier(str(relaxed_sdf), sanitize=False, removeHs=False)
    brenk = _brenk_catalog()

    n_relaxed = 0
    reasonable_indices: list[int] = []
    r5_violations = 0
    r1r4_only_count = 0  # 参考口径: R1-R4 通过的数量 (含 R5 违反者)
    rule_counts: dict[str, int] = {key: 0 for key in RULE_FIELDS}
    brenk_hits = 0
    bad_bond_hits = 0
    properties: dict[str, list[float]] = {key: [] for key in PROPERTY_KEYS}

    for index, parsed in enumerate(supplier):
        n_relaxed += 1
        if parsed is None:
            continue
        try:
            mol = Chem.Mol(parsed)
            Chem.SanitizeMol(mol)
        except Exception:
            continue
        facts = compute_structure_pathology(mol)
        for key in RULE_FIELDS:
            if getattr(facts, key):
                rule_counts[key] += 1
        if facts.r5_carbon_fraction_violation:
            r5_violations += 1
        # 参考口径: 仅 R1-R4 通过 (R5 单独报, 与偏好对构建口径一致)
        if not facts.is_structurally_pathological:
            r1r4_only_count += 1
        if brenk.HasMatch(mol):
            brenk_hits += 1
        _bad, checked, bad_fraction = _bond_quality(mol)
        if checked > 0 and bad_fraction > 1e-15:
            bad_bond_hits += 1
        # 主判据: 五条筛选标准 R1-R5 全部不违反 = 合理分子
        if facts.is_pathological:
            continue
        reasonable_indices.append(index)
        clean = Chem.RemoveHs(mol)
        try:
            heavy_atoms = [a for a in clean.GetAtoms() if a.GetAtomicNum() >= 2]
            carbon_count = sum(1 for a in heavy_atoms if a.GetSymbol() == "C")
            carbon_fraction = carbon_count / len(heavy_atoms) if heavy_atoms else math.nan
            properties["mw"].append(_finite(Descriptors.MolWt(clean)))
            properties["logp"].append(_finite(Crippen.MolLogP(clean)))
            properties["tpsa"].append(_finite(Descriptors.TPSA(clean)))
            properties["hba"].append(float(Lipinski.NumHAcceptors(clean)))
            properties["hbd"].append(float(Lipinski.NumHDonors(clean)))
            properties["qed"].append(_finite(QED.qed(clean)))
            properties["sa_score"].append(_finite(sascorer.calculateScore(clean)))
            properties["heavy_atom_count"].append(float(clean.GetNumHeavyAtoms()))
            properties["ring_count"].append(float(rdMolDescriptors.CalcNumRings(clean)))
            properties["aromatic_ring_count"].append(
                float(rdMolDescriptors.CalcNumAromaticRings(clean))
            )
            properties["carbon_fraction"].append(_finite(carbon_fraction))
        except Exception:
            continue

    # Docking restricted to the reasonable subset.
    dock_values = [docking_scores[i] for i in reasonable_indices if i in docking_scores]

    row: dict = {
        "label": label,
        "attempts": attempts,
        "relaxed_molecules": n_relaxed,
        "reasonable_count": len(reasonable_indices),
        "reasonable_yield_per_attempt_pct": round(100.0 * len(reasonable_indices) / attempts, 2)
        if attempts
        else math.nan,
        "reasonable_pct_of_relaxed": round(100.0 * len(reasonable_indices) / n_relaxed, 2)
        if n_relaxed
        else math.nan,
        "r1r4_only_count": r1r4_only_count,  # 参考: R1-R4 通过(含 R5 违反)
        "r1r4_only_yield_per_attempt_pct": round(100.0 * r1r4_only_count / attempts, 2)
        if attempts
        else math.nan,
        "r5_carbon_guard_violations": r5_violations,
        "dock_n_reasonable": len(dock_values),
    }
    # 各条判据违规率 (分母 = relaxed 正常分子数)
    for key in RULE_FIELDS:
        row[f"{key}_rate_of_relaxed_pct"] = (
            round(100.0 * rule_counts[key] / n_relaxed, 2) if n_relaxed else math.nan
        )
    row["r1r4_pathology_rate_of_relaxed_pct"] = (
        round(100.0 * (n_relaxed - r1r4_only_count) / n_relaxed, 2) if n_relaxed else math.nan
    )
    row["r1r5_pathology_rate_of_relaxed_pct"] = (
        round(100.0 * (n_relaxed - len(reasonable_indices)) / n_relaxed, 2)
        if n_relaxed
        else math.nan
    )
    row["brenk_alert_rate_of_relaxed_pct"] = (
        round(100.0 * brenk_hits / n_relaxed, 2) if n_relaxed else math.nan
    )
    row["bad_bond_rate_of_relaxed_pct"] = (
        round(100.0 * bad_bond_hits / n_relaxed, 2) if n_relaxed else math.nan
    )

    # 原始 SDF 上的碎片率与严格 sanitize 率 (分母 = attempts)
    row.update(_raw_sdf_stats(raw_sdf, attempts))

    if dock_values:
        row["dock_median_reasonable"] = round(float(np.median(dock_values)), 3)
        row["dock_mean_reasonable"] = round(float(np.mean(dock_values)), 3)
        row["dock_best_reasonable"] = round(float(np.min(dock_values)), 3)
    else:
        row["dock_median_reasonable"] = math.nan
        row["dock_mean_reasonable"] = math.nan
        row["dock_best_reasonable"] = math.nan

    for key in PROPERTY_KEYS:
        values = np.asarray([v for v in properties[key] if math.isfinite(v)], dtype=float)
        row[f"{key}_median_reasonable"] = round(float(np.median(values)), 3) if values.size else math.nan
        row[f"{key}_mean_reasonable"] = round(float(np.mean(values)), 3) if values.size else math.nan
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_figure(rows: list[dict], out_dir: Path) -> None:
    """Yield vs quality: the decisive trade-off plot for the paper."""
    labels = [r["label"] for r in rows]
    x = np.arange(len(labels))
    yield_pct = [r["reasonable_yield_per_attempt_pct"] for r in rows]
    dock_med = [r["dock_median_reasonable"] for r in rows]
    qed_med = [r["qed_median_reasonable"] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))

    ax = axes[0]
    ax.bar(x, yield_pct, color=DIFFSBDD_RED, alpha=0.85)
    ax.set_title("Reasonable-molecule yield\n(R1-R4 clean, per 1000 attempts)")
    ax.set_ylabel("%")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    for xi, v in zip(x, yield_pct):
        ax.text(xi, v + 0.6, f"{v:.1f}", ha="center", fontsize=10)

    ax = axes[1]
    ax.bar(x, dock_med, color=ORIGINAL_BLUE, alpha=0.85)
    ax.set_title("Docking median of reasonable molecules\n(lower = better)")
    ax.set_ylabel("kcal/mol")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    for xi, v in zip(x, dock_med):
        if np.isfinite(v):
            ax.text(xi, v - 0.25, f"{v:.2f}", ha="center", fontsize=10)

    ax = axes[2]
    ax.bar(x, qed_med, color="#2ECC71", alpha=0.85)
    ax.set_title("QED median of reasonable molecules")
    ax.set_ylabel("QED")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    for xi, v in zip(x, qed_med):
        if np.isfinite(v):
            ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_dir / "reasonable_yield_vs_quality.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="LABEL:RAW_SDF:RELAXED_SDF:DOCKING_CSV:ATTEMPTS",
        help="One evaluated run. RAW_SDF may be '-' to skip fragment stats. "
        "Repeat the flag for each model.",
    )
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for spec in args.input:
        parts = spec.split(":")
        if len(parts) == 5:
            label, raw, sdf, docking, attempts = parts
            raw_path: Path | None = None if raw == "-" else Path(raw)
        elif len(parts) == 4:
            # Backwards-compatible: LABEL:RELAXED_SDF:DOCKING_CSV:ATTEMPTS
            label, sdf, docking, attempts = parts
            raw_path = None
        else:
            raise SystemExit(
                "bad --input spec (need LABEL:RAW_SDF:RELAXED_SDF:DOCKING_CSV:ATTEMPTS "
                f"or LABEL:RELAXED_SDF:DOCKING_CSV:ATTEMPTS): {spec}"
            )
        row = analyze_run(label, raw_path, Path(sdf), Path(docking), int(attempts))
        rows.append(row)
        print(
            f"{label:22s} relaxed={row['relaxed_molecules']:4d} "
            f"reasonable={row['reasonable_count']:4d} "
            f"yield={row['reasonable_yield_per_attempt_pct']:5.2f}% "
            f"dock_med={row['dock_median_reasonable']:.2f} "
            f"qed_med={row['qed_median_reasonable']:.3f}"
        )

    write_csv(out_dir / "reasonable_molecules_metrics.csv", rows)
    make_figure(rows, out_dir)
    (out_dir / "reasonable_eval_manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "rule": "reasonable = five screening rules R1-R5 all clean; "
                "properties/docking measured on the reasonable subset only",
                "inputs": args.input,
                "rows": len(rows),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[ok] wrote {out_dir / 'reasonable_molecules_metrics.csv'}")
    print(f"[ok] wrote {out_dir / 'reasonable_yield_vs_quality.png'}")


if __name__ == "__main__":
    main()
