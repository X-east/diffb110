"""Analyze the fixed-attempt DiffSBDD Global E2PO generation matrix."""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, FilterCatalog, Lipinski, QED, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
GEN_ROOT = ROOT / "generation_comparison"
VARIANTS = ("original", "x_only", "v_only", "native_joint")
OUT_CSV = ROOT / "generation_molecule_metrics.csv"
SUMMARY_CSV = ROOT / "generation_summary.csv"
ELEMENT_CSV = ROOT / "generation_element_distribution.csv"
PATHOLOGY_CSV = ROOT / "generation_pathology_summary.csv"
REPORT_MD = ROOT / "generation_analysis_report.md"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "DiffSBDD" / "analysis" / "SA_Score"))

from DiffSBDD.analysis.SA_Score import sascorer  # noqa: E402
from ddpo.rewards.chemistry_guard_reward import ChemistryGuardReward  # noqa: E402
from ddpo.rewards.ring_complexity_reward import RingComplexityReward  # noqa: E402
from ddpo.rewards.ring_pathology_reward import RingPathologyReward  # noqa: E402

RDLogger.DisableLog("rdApp.*")


def _catalog() -> FilterCatalog.FilterCatalog:
    """Build the project-standard Brenk structural-alert catalog."""
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
    return FilterCatalog.FilterCatalog(params)


def _safe_float(value: object) -> float:
    """Convert a numeric value to a finite float."""
    result = float(value)
    return result if math.isfinite(result) else float("nan")


def _scaffold_smiles(mol: Chem.Mol) -> str:
    """Return a canonical Murcko scaffold, preserving empty scaffolds."""
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold, canonical=True, isomericSmiles=True)


def _sanitize(mol: Chem.Mol) -> Chem.Mol | None:
    """Copy and sanitize one parsed SDF molecule."""
    try:
        sanitized = Chem.Mol(mol)
        Chem.SanitizeMol(sanitized)
        return sanitized
    except Exception:
        return None


def _row(
    variant: str,
    attempt: int,
    parsed_mol: Chem.Mol | None,
    mol: Chem.Mol | None,
    brenk: FilterCatalog.FilterCatalog,
    guard: ChemistryGuardReward,
    rings: RingComplexityReward,
    pathology: RingPathologyReward,
) -> dict[str, object]:
    """Extract all per-attempt metrics from one SDF record."""
    base: dict[str, object] = {
        "variant": variant,
        "attempt": attempt,
        "parsed": int(parsed_mol is not None),
        "sanitize_success": int(mol is not None),
        "rdkit_valid": int(mol is not None),
    }
    if mol is None:
        base.update(
            {
                "canonical_smiles": "",
                "murcko_scaffold": "",
                "failure_reason": "parse_failed" if mol is None else "sanitize_failed",
            }
        )
        return base

    guard_facts = guard._compute_facts(mol)
    ring_facts = rings._compute_facts(mol)
    pathology_facts = pathology._compute_facts(mol)
    canonical = Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True, isomericSmiles=True)
    scaffold = _scaffold_smiles(mol)
    base.update(
        {
            "canonical_smiles": canonical,
            "murcko_scaffold": scaffold,
            "failure_reason": "",
            "heavy_atom_count": mol.GetNumHeavyAtoms(),
            "atom_count": mol.GetNumAtoms(),
            "fragment_count": guard_facts.fragment_count,
            "mw": _safe_float(Descriptors.MolWt(mol)),
            "logp": _safe_float(Crippen.MolLogP(mol)),
            "tpsa": _safe_float(Descriptors.TPSA(mol)),
            "hba": int(Lipinski.NumHAcceptors(mol)),
            "hbd": int(Lipinski.NumHDonors(mol)),
            "qed": _safe_float(QED.qed(mol)),
            "sa_score": _safe_float(sascorer.calculateScore(mol)),
            "brenk_alert": int(brenk.HasMatch(mol)),
            "carbon_fraction": guard_facts.carbon_fraction,
            "hetero_atom_count": guard_facts.hetero_atom_count,
            "hetero_atom_fraction": guard_facts.hetero_atom_fraction,
            "aromatic_ring_count": ring_facts.aromatic_ring_count,
            "ring_count": ring_facts.ring_count,
            "three_member_ring_count": ring_facts.three_member_ring_count,
            "max_fused_ring_system_size": ring_facts.max_fused_ring_system_size,
            "spiro_atom_count": ring_facts.spiro_atom_count,
            "bridgehead_atom_count": ring_facts.bridgehead_atom_count,
            "large_ring_count": pathology_facts.large_ring_count,
            "large_ring_fused_six_count": pathology_facts.large_ring_fused_six_count,
            "three_member_fused_or_spiro_count": pathology_facts.three_member_fused_or_spiro_count,
            "bad_bond_length_fraction": guard_facts.bad_bond_length_fraction,
            "halogen_bad_position_count": guard_facts.halogen_bad_position_count,
            "chemistry_guard_violation_count": guard_facts.violation_count,
        }
    )
    return base


def analyze_variant(
    variant: str,
    brenk: FilterCatalog.FilterCatalog,
    guard: ChemistryGuardReward,
    rings: RingComplexityReward,
    pathology: RingPathologyReward,
) -> tuple[list[dict[str, object]], Counter[str], dict[str, object]]:
    """Read and analyze every record from one fixed-attempt SDF."""
    sdf_path = GEN_ROOT / variant / "molecules.sdf"
    supplier = Chem.SDMolSupplier(str(sdf_path), sanitize=False, removeHs=False)
    rows: list[dict[str, object]] = []
    elements: Counter[str] = Counter()
    for attempt, parsed in enumerate(supplier, start=1):
        mol = _sanitize(parsed) if parsed is not None else None
        row = _row(variant, attempt, parsed, mol, brenk, guard, rings, pathology)
        rows.append(row)
        if mol is not None:
            elements.update(atom.GetSymbol() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)

    valid = [row for row in rows if row["rdkit_valid"] == 1]
    unique_smiles = {str(row["canonical_smiles"]) for row in valid}
    nonempty_scaffolds = {str(row["murcko_scaffold"]) for row in valid if row["murcko_scaffold"]}
    all_scaffolds = {str(row["murcko_scaffold"]) for row in valid}
    summary: dict[str, object] = {
        "variant": variant,
        "attempts": 1000,
        "sdf_records": len(rows),
        "parsed_records": sum(int(row["parsed"]) for row in rows),
        "sanitize_valid": len(valid),
        "sanitize_rate_attempts": len(valid) / 1000,
        "unique_smiles": len(unique_smiles),
        "unique_smiles_rate_valid": len(unique_smiles) / len(valid) if valid else 0.0,
        "unique_murcko_scaffolds_nonempty": len(nonempty_scaffolds),
        "unique_murcko_scaffolds_all": len(all_scaffolds),
        "empty_murcko_scaffold_count": sum(not bool(row["murcko_scaffold"]) for row in valid),
        "brenk_alert_rate_valid": _mean(valid, "brenk_alert"),
        "fragment_rate_valid": _fraction_at_least(valid, "fragment_count", 2),
        "guard_violation_rate_valid": _fraction_at_least(valid, "chemistry_guard_violation_count", 1),
        "three_member_ring_rate_valid": _fraction_at_least(valid, "three_member_ring_count", 1),
        "large_ring_rate_valid": _fraction_at_least(valid, "large_ring_count", 1),
        "carbon_fraction_mean": _mean(valid, "carbon_fraction"),
        "hetero_atom_mean": _mean(valid, "hetero_atom_count"),
        "heavy_atom_mean": _mean(valid, "heavy_atom_count"),
        "mw_mean": _mean(valid, "mw"),
        "logp_mean": _mean(valid, "logp"),
        "tpsa_mean": _mean(valid, "tpsa"),
        "hba_mean": _mean(valid, "hba"),
        "hbd_mean": _mean(valid, "hbd"),
        "qed_mean": _mean(valid, "qed"),
        "sa_mean": _mean(valid, "sa_score"),
        "ring_count_mean": _mean(valid, "ring_count"),
        "aromatic_ring_mean": _mean(valid, "aromatic_ring_count"),
        "three_member_ring_mean": _mean(valid, "three_member_ring_count"),
        "fused_system_size_mean": _mean(valid, "max_fused_ring_system_size"),
        "bad_bond_length_fraction_mean": _mean(valid, "bad_bond_length_fraction"),
        "elements_total": int(sum(elements.values())),
    }
    return rows, elements, summary


def _mean(rows: list[dict[str, object]], key: str) -> float:
    """Compute a finite mean over valid molecule rows."""
    values = [float(row[key]) for row in rows if key in row and isinstance(row[key], (int, float))]
    return float(np.mean(values)) if values else float("nan")


def _fraction_at_least(rows: list[dict[str, object]], key: str, threshold: float) -> float:
    """Compute the valid-molecule fraction meeting a threshold."""
    values = [float(row[key]) for row in rows if key in row and isinstance(row[key], (int, float))]
    return float(sum(value >= threshold for value in values) / len(values)) if values else float("nan")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write rows with a stable union of fields."""
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(summaries: list[dict[str, object]]) -> None:
    """Create readable validity/diversity, descriptor, and pathology plots."""
    labels = [str(row["variant"]) for row in summaries]
    colors = ["#4d4d4d", "#d73027", "#1a9850", "#4575b4"]
    positions = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    axes[0].bar(positions, [float(row["sanitize_rate_attempts"]) for row in summaries], color=colors)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Rate / fixed 1000 attempts")
    axes[0].set_title("RDKit sanitize validity")
    axes[0].set_xticks(positions, labels)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(positions - 0.2, [float(row["unique_smiles_rate_valid"]) for row in summaries], 0.4, label="unique SMILES / valid", color="#762a83")
    axes[1].bar(positions + 0.2, [float(row["unique_murcko_scaffolds_nonempty"]) / 1000 for row in summaries], 0.4, label="nonempty scaffolds / attempts", color="#af8dc3")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Rate")
    axes[1].set_title("Diversity and scaffold retention")
    axes[1].set_xticks(positions, labels)
    axes[1].legend(frameon=False, fontsize=9)
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("DiffSBDD Global E2PO generation quality (T=500, 1000 attempts)")
    fig.tight_layout()
    fig.savefig(ROOT / "generation_validity_diversity.png", dpi=200)
    plt.close(fig)

    descriptor_keys = ["qed_mean", "sa_mean", "logp_mean", "tpsa_mean", "mw_mean", "carbon_fraction_mean"]
    descriptor_labels = ["QED", "SA", "LogP", "TPSA", "MW", "Carbon fraction"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for axis, key, title in zip(axes.flat, descriptor_keys, descriptor_labels):
        axis.bar(positions, [float(row[key]) for row in summaries], color=colors)
        axis.set_title(title)
        axis.set_xticks(positions, labels, rotation=20)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Valid-molecule descriptor comparison")
    fig.tight_layout()
    fig.savefig(ROOT / "generation_descriptor_comparison.png", dpi=200)
    plt.close(fig)

    pathology_keys = ["brenk_alert_rate_valid", "fragment_rate_valid", "guard_violation_rate_valid", "three_member_ring_rate_valid", "large_ring_rate_valid"]
    pathology_labels = ["Brenk alert", "Disconnected fragments", "Guard violation", "3-member ring", "Large ring"]
    fig, axis = plt.subplots(figsize=(12, 5.5))
    width = 0.18
    for index, (key, title) in enumerate(zip(pathology_keys, pathology_labels)):
        axis.bar(positions + (index - 2) * width, [float(row[key]) for row in summaries], width, label=title)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Fraction among valid molecules")
    axis.set_title("Structural-alert and pathology rates")
    axis.set_xticks(positions, labels)
    axis.legend(frameon=False, ncol=3)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(ROOT / "generation_pathology_comparison.png", dpi=200)
    plt.close(fig)


def write_element_csv(element_rows: list[dict[str, object]]) -> None:
    """Write per-method element prevalence and atom fractions."""
    write_csv(ELEMENT_CSV, element_rows)


def write_report(summaries: list[dict[str, object]]) -> None:
    """Write a compact evidence-grounded Markdown report."""
    columns = [
        ("variant", "方法"),
        ("attempts", "attempts"),
        ("sanitize_valid", "valid"),
        ("sanitize_rate_attempts", "valid/attempts"),
        ("unique_smiles", "unique SMILES"),
        ("unique_murcko_scaffolds_nonempty", "unique scaffolds"),
        ("qed_mean", "QED"),
        ("sa_mean", "SA"),
        ("logp_mean", "LogP"),
        ("tpsa_mean", "TPSA"),
        ("carbon_fraction_mean", "C fraction"),
        ("brenk_alert_rate_valid", "Brenk"),
        ("guard_violation_rate_valid", "guard violation"),
    ]
    lines = [
        "# DiffSBDD Global E2PO 生成分析",
        "",
        "固定条件：T=500、seed=1、每方法 1000 attempts、batch=25、resamplings=10、OpenBabel、all_frags。所有率的分母明确标注；有效分子指标只在 sanitize-valid 分子上计算。",
        "",
        "## Summary",
        "",
        "| " + " | ".join(label for _, label in columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in summaries:
        values = []
        for key, _label in columns:
            value = row[key]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    lines += [
        "",
        "## Figures",
        "",
        "![Validity and diversity](generation_validity_diversity.png)",
        "",
        "![Descriptors](generation_descriptor_comparison.png)",
        "",
        "![Pathology](generation_pathology_comparison.png)",
        "",
        "## Evidence files",
        "",
        f"- Per-attempt metrics: `{OUT_CSV.name}`",
        f"- Summary CSV: `{SUMMARY_CSV.name}`",
        f"- Element distribution: `{ELEMENT_CSV.name}`",
        f"- Pathology summary: `{PATHOLOGY_CSV.name}`",
        "",
        "## Interpretation boundary",
        "",
        "训练 pair accuracy 只证明 endpoint x/v loss 对偏好对的拟合能力；本报告的生成统计才检验该信号是否转化为有效性、分子多样性和化学结构质量。docking/ligand-efficiency 需结合后续 UniDock CSV 解释，不能由本表推断。",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Analyze all completed variants and write CSV, PNG, and Markdown evidence."""
    brenk = _catalog()
    guard = ChemistryGuardReward()
    rings = RingComplexityReward()
    pathology = RingPathologyReward()
    all_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    element_rows: list[dict[str, object]] = []
    pathology_rows: list[dict[str, object]] = []
    for variant in VARIANTS:
        rows, elements, summary = analyze_variant(variant, brenk, guard, rings, pathology)
        all_rows.extend(rows)
        summaries.append(summary)
        total_atoms = sum(elements.values())
        for element, count in sorted(elements.items()):
            element_rows.append(
                {
                    "variant": variant,
                    "element": element,
                    "atom_count": count,
                    "atom_fraction": count / total_atoms if total_atoms else 0.0,
                }
            )
        valid = [row for row in rows if row["rdkit_valid"] == 1]
        for key in ("brenk_alert", "fragment_count", "chemistry_guard_violation_count", "three_member_ring_count", "large_ring_count"):
            pathology_rows.append(
                {
                    "variant": variant,
                    "metric": key,
                    "valid_molecules": len(valid),
                    "count": sum(
                        int(float(row.get(key, 0)) > 0) if key != "fragment_count" else int(float(row.get(key, 0)) >= 2)
                        for row in valid
                    ),
                    "rate_valid": summary[
                        {
                            "brenk_alert": "brenk_alert_rate_valid",
                            "fragment_count": "fragment_rate_valid",
                            "chemistry_guard_violation_count": "guard_violation_rate_valid",
                            "three_member_ring_count": "three_member_ring_rate_valid",
                            "large_ring_count": "large_ring_rate_valid",
                        }[key]
                    ],
                }
            )
    write_csv(OUT_CSV, all_rows)
    write_csv(SUMMARY_CSV, summaries)
    write_element_csv(element_rows)
    write_csv(PATHOLOGY_CSV, pathology_rows)
    plot_summary(summaries)
    write_report(summaries)
    print(json.dumps({"summary": str(SUMMARY_CSV), "rows": len(all_rows), "variants": len(summaries)}, indent=2))


if __name__ == "__main__":
    main()
