from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from rdkit import Chem


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "analysis" / "classifier_v0_2_1_weighted_20260608"

BASE_POS = ROOT / "analysis/final_classifier_dataset_selection_20260527/selected_positives_100k.smi"
BASE_NEG = ROOT / "analysis/final_classifier_dataset_selection_20260527/selected_negatives_70k_labeled.smi"
CURATED = ROOT / "analysis/good_bad_classifier_curation_20260608/curated_molecules.csv.gz"

POS_OUT = OUT_DIR / "selected_positives_v0_2_1_weighted.smi"
NEG_OUT = OUT_DIR / "selected_negatives_v0_2_1_weighted_labeled.smi"


def canonical_smiles(smiles: str | None) -> str | None:
    if smiles is None or pd.isna(smiles):
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=False)


def hard_negative_weight(row: pd.Series) -> float:
    reasons = set(str(row.get("bad_reasons", "")).split("|"))
    primary = str(row.get("primary_reason", ""))
    if "bad_classifier_false_positive" in reasons or primary == "bad_classifier_false_positive":
        return 8.0
    if "bad_bad_bond_length" in reasons or primary == "bad_bad_bond_length":
        return 3.0
    if "bad_low_tpsa_or_high_logp" in reasons or primary == "bad_low_tpsa_or_high_logp":
        return 3.0
    if "bad_ring_complex_failed_rl" in reasons or primary == "bad_ring_complex_failed_rl":
        return 2.0
    if "bad_top_docking_exploit" in reasons or primary == "bad_top_docking_exploit":
        return 2.0
    if "bad_three_member_ring" in reasons or primary == "bad_three_member_ring":
        return 2.0
    if "bad_non_aromatic_polycycle" in reasons or primary == "bad_non_aromatic_polycycle":
        return 2.0
    return 1.5


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    base_pos = pd.read_csv(BASE_POS, sep=None, engine="python")
    base_neg = pd.read_csv(BASE_NEG, sep=None, engine="python")
    curated = pd.read_csv(CURATED)

    base_pos["canonical_smiles"] = base_pos["smiles"].map(canonical_smiles)
    base_pos["source"] = "selected_chembl_v0_1"
    base_pos["all_labels"] = "selected_chembl_v0_1"
    base_pos["weight"] = 1.0

    base_neg["canonical_smiles"] = base_neg["smiles"].map(canonical_smiles)
    base_neg["source"] = "selected_negative_v0_1"
    base_neg["weight"] = 1.0

    generated_pos = curated[
        curated["label"].eq("good_high_conf")
        & curated["parse_ok"].eq(1)
        & curated["rdkit_valid"].eq(1)
        & curated["canonical_smiles"].notna()
    ].copy()
    generated_pos["smiles"] = generated_pos["canonical_smiles"]
    generated_pos["label"] = 1
    generated_pos["source"] = "generated_good_high_conf"
    generated_pos["all_labels"] = generated_pos["good_reasons"].fillna(generated_pos["primary_reason"])
    generated_pos["weight"] = 2.0

    generated_neg = curated[
        curated["label"].eq("bad")
        & curated["parse_ok"].eq(1)
        & curated["rdkit_valid"].eq(1)
        & curated["canonical_smiles"].notna()
        & ~curated["primary_reason"].eq("bad_invalid_or_unparseable")
    ].copy()
    generated_neg["smiles"] = generated_neg["canonical_smiles"]
    generated_neg["label"] = generated_neg["primary_reason"]
    generated_neg["all_labels"] = generated_neg["bad_reasons"].fillna(generated_neg["primary_reason"])
    generated_neg["source"] = "generated_failed_rl_hard_negative"
    generated_neg["weight"] = generated_neg.apply(hard_negative_weight, axis=1)

    pos_rows = pd.concat(
        [
            base_pos[["smiles", "label", "source", "all_labels", "weight", "canonical_smiles"]],
            generated_pos[["smiles", "label", "source", "all_labels", "weight", "canonical_smiles"]],
        ],
        ignore_index=True,
    )
    pos_rows = pos_rows[pos_rows["canonical_smiles"].notna()].drop_duplicates("canonical_smiles", keep="first")

    neg_rows = pd.concat(
        [
            base_neg[["smiles", "label", "source", "all_labels", "weight", "canonical_smiles"]],
            generated_neg[["smiles", "label", "source", "all_labels", "weight", "canonical_smiles"]],
        ],
        ignore_index=True,
    )
    neg_rows = neg_rows[neg_rows["canonical_smiles"].notna()].drop_duplicates("canonical_smiles", keep="first")

    positive_smiles = set(pos_rows["canonical_smiles"])
    overlap = neg_rows["canonical_smiles"].isin(positive_smiles)
    overlap_count = int(overlap.sum())
    neg_rows = neg_rows[~overlap].copy()

    pos_rows[["smiles", "label", "source", "all_labels", "weight"]].to_csv(POS_OUT, sep="\t", index=False)
    neg_rows[["smiles", "label", "source", "all_labels", "weight"]].to_csv(NEG_OUT, sep="\t", index=False)

    summary = {
        "base_positive_rows": int(len(base_pos)),
        "base_negative_rows": int(len(base_neg)),
        "generated_good_high_conf_before_dedup": int(len(generated_pos)),
        "generated_negative_rows_before_dedup": int(len(generated_neg)),
        "positive_rows_after_dedup": int(len(pos_rows)),
        "negative_rows_after_dedup_overlap_removed": int(len(neg_rows)),
        "negative_positive_overlap_removed": overlap_count,
        "positive_output": str(POS_OUT.relative_to(ROOT)),
        "negative_output": str(NEG_OUT.relative_to(ROOT)),
        "negative_weight_counts": neg_rows["weight"].value_counts().sort_index().to_dict(),
        "generated_negative_primary_reason_counts": generated_neg["primary_reason"].value_counts().to_dict(),
        "generated_positive_primary_reason_counts": generated_pos["primary_reason"].value_counts().to_dict(),
    }
    (OUT_DIR / "sample_list_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    pd.DataFrame(
        [
            {"split": "positive", "source": source, "n": int(n)}
            for source, n in pos_rows["source"].value_counts().items()
        ]
        + [
            {"split": "negative", "source": source, "n": int(n)}
            for source, n in neg_rows["source"].value_counts().items()
        ]
    ).to_csv(OUT_DIR / "sample_source_summary.csv", index=False)

    neg_rows.groupby(["source", "weight"]).size().rename("n").reset_index().to_csv(
        OUT_DIR / "sample_weight_summary.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
