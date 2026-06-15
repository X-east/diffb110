from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
POS_V0_3 = BASE_DIR / "selected_positives_v0_3_scaffold.smi"
NEG_V0_3 = BASE_DIR / "selected_negatives_v0_3_scaffold_labeled.smi"
POS_V0_3_1 = BASE_DIR / "selected_positives_v0_3_1_scaffold_strongpoly.smi"
NEG_V0_3_1 = BASE_DIR / "selected_negatives_v0_3_1_scaffold_strongpoly_labeled.smi"
SUMMARY_PATH = BASE_DIR / "v0_3_1_scaffold_strongpoly_training_lists_summary.json"
CHEMBL_PROPERTIES = PROJECT_ROOT / "data/chembl/chembl_36_properties.csv.gz"
TARGET_POSITIVE_ROWS = 150_000
SEED = 45


def _add_balancing_chembl_positives(positives: pd.DataFrame, negatives: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Add mildly filtered ChEMBL positives to make positives slightly more abundant."""
    n_needed = max(TARGET_POSITIVE_ROWS - len(positives), 0)
    if n_needed == 0:
        return positives, 0

    existing = set(positives["smiles"].astype(str))
    existing.update(negatives["smiles"].astype(str))

    chembl = pd.read_csv(CHEMBL_PROPERTIES, compression="gzip")
    filtered = chembl[
        chembl["canonical_smiles"].notna()
        & chembl["formal_charge"].eq(0)
        & chembl["heavy_atoms"].between(12, 55)
        & chembl["num_heteroatoms"].ge(1)
        & chembl["mol_log_p"].between(-1.0, 6.0)
        & chembl["tpsa"].between(10.0, 180.0)
        & chembl["carbon_fraction"].le(0.90)
        & chembl["lipinski_heavy_violations"].le(1)
    ].copy()
    filtered["smiles"] = filtered["canonical_smiles"].astype(str)
    filtered = filtered[~filtered["smiles"].isin(existing)]
    filtered = filtered.drop_duplicates("smiles")

    n_sample = min(n_needed, len(filtered))
    extra = filtered.sample(n=n_sample, random_state=SEED)
    extra_pos = pd.DataFrame(
        {
            "smiles": extra["smiles"].tolist(),
            "label": [1] * n_sample,
            "source": ["chembl_balancing_positive_v0_3_1"] * n_sample,
            "all_labels": ["chembl_balancing_positive_v0_3_1"] * n_sample,
            "weight": [1.0] * n_sample,
        }
    )
    return pd.concat([positives, extra_pos], ignore_index=True), int(n_sample)


def main() -> None:
    """Create v0.3.1 classifier training lists with stronger poly-fused negatives."""
    positives = pd.read_csv(POS_V0_3, sep="\t")
    negatives = pd.read_csv(NEG_V0_3, sep="\t")

    positives, n_added_positives = _add_balancing_chembl_positives(positives, negatives)

    if "weight" not in negatives.columns:
        negatives["weight"] = 1.0
    negatives["weight"] = negatives["weight"].astype(float)

    label_text = (
        negatives.get("label", pd.Series([""] * len(negatives))).astype(str)
        + "|"
        + negatives.get("all_labels", pd.Series([""] * len(negatives))).astype(str)
    )
    poly_mask = label_text.str.contains("bad_poly_fused_or_bridge_ring", regex=False, na=False)
    before = negatives.loc[poly_mask, "weight"].copy()
    negatives.loc[poly_mask, "weight"] = negatives.loc[poly_mask, "weight"].clip(lower=3.0)

    positives.to_csv(POS_V0_3_1, sep="\t", index=False)
    negatives.to_csv(NEG_V0_3_1, sep="\t", index=False)

    summary = {
        "positive_input": str(POS_V0_3),
        "negative_input": str(NEG_V0_3),
        "positive_output": str(POS_V0_3_1),
        "negative_output": str(NEG_V0_3_1),
        "n_positive_rows": int(len(positives)),
        "target_positive_rows": TARGET_POSITIVE_ROWS,
        "n_added_balancing_chembl_positives": n_added_positives,
        "n_negative_rows": int(len(negatives)),
        "bad_poly_fused_or_bridge_ring_rows": int(poly_mask.sum()),
        "bad_poly_weight_before": {
            "min": float(before.min()) if len(before) else 0.0,
            "mean": float(before.mean()) if len(before) else 0.0,
            "max": float(before.max()) if len(before) else 0.0,
        },
        "bad_poly_weight_after": {
            "min": float(negatives.loc[poly_mask, "weight"].min()) if int(poly_mask.sum()) else 0.0,
            "mean": float(negatives.loc[poly_mask, "weight"].mean()) if int(poly_mask.sum()) else 0.0,
            "max": float(negatives.loc[poly_mask, "weight"].max()) if int(poly_mask.sum()) else 0.0,
        },
        "policy": "bad_poly_fused_or_bridge_ring is treated as strong negative; weight floor raised to 3.0.",
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
