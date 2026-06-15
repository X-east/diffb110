from __future__ import annotations

"""Export scored SDFs with classifier probabilities embedded in molecule names."""

import csv
from pathlib import Path

from rdkit import Chem


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "analysis/classifier_named_sdfs_20260609"

INPUTS = [
    {
        "label": "diffgui",
        "input_sdf": ROOT / "analysis/diffgui_classifier_score_20260609/diffgui_500_scored.sdf",
        "output_sdf": OUT_DIR / "diffgui_500_name_with_classifier_scores.sdf",
    },
    {
        "label": "diffsbdd",
        "input_sdf": ROOT / "analysis/diffsbdd_8gpu_5000_score_20260608/generated_5000_scored.sdf",
        "output_sdf": OUT_DIR / "diffsbdd_5000_name_with_classifier_scores.sdf",
    },
]


def _prop(mol: Chem.Mol, key: str, default: str = "NA") -> str:
    """Return a stripped SDF property value or a default marker."""
    if not mol.HasProp(key):
        return default
    value = mol.GetProp(key).strip()
    if value == "" or value.lower() == "nan":
        return default
    return value


def _short_float(value: str, digits: int = 4) -> str:
    """Compact numeric property values for molecule names."""
    if value == "NA":
        return value
    try:
        return f"{float(value):.{digits}f}"
    except ValueError:
        return value


def _build_name(mol: Chem.Mol, label: str, index: int) -> str:
    """Build a name that carries the two classifier scores."""
    mol_id = _prop(mol, "mol_id", f"{label}_{index:05d}")
    p_v2 = _short_float(_prop(mol, "p_good_v02"))
    p_v21 = _short_float(_prop(mol, "p_good_v021"))
    score = _short_float(_prop(mol, "score"))
    valid = _prop(mol, "scoring_valid")
    status = _prop(mol, "sanitize_status")
    return f"{mol_id}|v2={p_v2}|v2.1={p_v21}|score={score}|valid={valid}|{status}"


def export_one(label: str, input_sdf: Path, output_sdf: Path) -> dict[str, int | str]:
    """Rewrite one SDF with score-bearing molecule names."""
    supplier = Chem.SDMolSupplier(str(input_sdf), sanitize=False, removeHs=False)
    output_sdf.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(output_sdf))
    writer.SetKekulize(False)

    total = 0
    readable = 0
    written = 0
    scored = 0
    for index, mol in enumerate(supplier):
        total += 1
        if mol is None:
            continue
        readable += 1
        out = Chem.Mol(mol)
        out.SetProp("_Name", _build_name(out, label, index))
        if _prop(out, "p_good_v02") != "NA" and _prop(out, "p_good_v021") != "NA":
            scored += 1
        writer.write(out)
        written += 1
    writer.close()
    return {
        "label": label,
        "input_sdf": str(input_sdf),
        "output_sdf": str(output_sdf),
        "total_records": total,
        "readable_records": readable,
        "written_records": written,
        "records_with_both_classifier_scores": scored,
    }


def main() -> None:
    """Export all configured SDFs and write a manifest."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [export_one(**item) for item in INPUTS]
    manifest_path = OUT_DIR / "manifest.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    readme_path = OUT_DIR / "README.md"
    readme_path.write_text(
        "# Classifier-score named SDFs\n\n"
        "Molecule `_Name` format:\n\n"
        "`mol_id|v2=<p_good_v02>|v2.1=<p_good_v021>|score=<guarded_score>|valid=<0/1>|<sanitize_status>`\n\n"
        "Files:\n\n"
        "- `diffgui_500_name_with_classifier_scores.sdf`\n"
        "- `diffsbdd_5000_name_with_classifier_scores.sdf`\n"
        "- `manifest.csv`\n",
        encoding="utf-8",
    )
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
