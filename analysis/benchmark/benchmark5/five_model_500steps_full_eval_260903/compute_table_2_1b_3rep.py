"""Recompute the thesis table 2.1(b) five-model benchmark on the 3-repeat protocol.

Table 2.1(b) originally mixed a single-run batch (2026-06) with different
repeats than table 3.4. This script produces every column of table 2.1(b)
from the same 3-repeat subset (rep4/rep5/rep6) used by table 3.4, so the two
tables share one data source.

Outputs a printed summary; no files are written.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

RDLogger.DisableLog("rdApp.*")

BASE = Path(__file__).resolve().parents[1]
REPEATS = ("rep4", "rep5", "rep6")
ATTEMPTS_PER_REP = 1000


def main() -> None:
    """Compute per-model benchmark metrics on the 3-repeat subset."""
    records = pd.read_csv(
        BASE / "five_model_completed_comparison_260827" / "per_record_quality.csv.gz"
    )
    subset = records[
        (records["step"] == 500)
        & (records["repeat"].isin(REPEATS))
        & (records["sanitize_valid"])
    ]

    ring_rows = []
    for model, group in subset.groupby("model"):
        aromatic = three = 0
        for smiles in group["canonical_smiles"]:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            aromatic += 1 if Descriptors.NumAromaticRings(mol) > 0 else 0
            three += 1 if any(len(r) == 3 for r in mol.GetRingInfo().AtomRings()) else 0
        ring_rows.append(
            {
                "model": model,
                "n_valid": len(group),
                "valid_rate": len(group) / (ATTEMPTS_PER_REP * len(REPEATS)),
                "aromatic_rate": aromatic / len(group),
                "three_member_rate": three / len(group),
            }
        )
    rings = pd.DataFrame(ring_rows).set_index("model")

    dock_rows = []
    for repeat in REPEATS:
        for model_dir in sorted((Path(__file__).parent / "docking" / repeat).glob("*")):
            summary_path = model_dir / "run_summary.json"
            if not summary_path.exists():
                continue
            payload = json.loads(summary_path.read_text())
            scores = pd.read_csv(model_dir / "docking_results.csv")
            column = "score" if "score" in scores.columns else scores.columns[-1]
            values = pd.to_numeric(scores[column], errors="coerce").dropna()
            dock_rows.append(
                {
                    "repeat": repeat,
                    "model": payload["model"],
                    "dock_mean": values.mean(),
                    "dock_median": values.median(),
                    "le_minus10": (values <= -10.0).mean(),
                }
            )
    dock = pd.DataFrame(dock_rows).groupby("model").mean(numeric_only=True)

    descriptors = pd.read_csv(
        BASE / "five_model_completed_comparison_260827" / "per_repeat_descriptor_means.csv"
    )
    desc_subset = descriptors[
        (descriptors["step"] == 500) & (descriptors["repeat"].isin(REPEATS))
    ]
    qed = desc_subset.groupby("model")["qed"].mean()

    table = rings.join(dock[["dock_mean", "dock_median", "le_minus10"]]).join(qed)
    table = table[
        [
            "n_valid",
            "valid_rate",
            "dock_mean",
            "dock_median",
            "le_minus10",
            "qed",
            "aromatic_rate",
            "three_member_rate",
        ]
    ]
    pd.set_option("display.width", 200)
    print(f"repeats={REPEATS} attempts/rep={ATTEMPTS_PER_REP}")
    print(table.round(4).to_string())


if __name__ == "__main__":
    main()
