from __future__ import annotations

"""Score generated DiffSBDD molecules and export an annotated SDF.

The script keeps the scoring path offline and reviewable:

- reads one generated SDF;
- assigns stable molecule IDs;
- scores each molecule with ChEMBL classifier v0.2 and v0.2.1;
- computes dynamic rank scores and simple chemistry/geometry flags;
- writes one annotated SDF plus CSV/summary files.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MOLSKILL_ROOT = PROJECT_ROOT / "reference project" / "代码" / "molskill"
if str(MOLSKILL_ROOT) not in sys.path:
    sys.path.insert(0, str(MOLSKILL_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molskill.data.featurizers import get_featurizer  # noqa: E402

from chembl_classifier.model import ChemblClassifier  # noqa: E402


RDLogger.DisableLog("rdApp.*")

DEFAULT_V02_MODEL = PROJECT_ROOT / "chembl_classifier/models/chembl_classifier_selected_v0_2_hardneg.pt"
DEFAULT_V021_MODEL = PROJECT_ROOT / "chembl_classifier/models/chembl_classifier_selected_v0_2_1_weighted.pt"
HALOGENS = frozenset({"F", "Cl", "Br", "I"})


def _float_str(value: float | int | None, precision: int = 6) -> str:
    """Format numeric values for SDF/CSV."""
    if value is None:
        return ""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{precision}f}"


def _safe_set_prop(mol: Chem.Mol, key: str, value: Any) -> None:
    """Set an SDF property using a stable string representation."""
    if value is None:
        mol.SetProp(key, "")
    elif isinstance(value, float):
        mol.SetProp(key, _float_str(value))
    else:
        mol.SetProp(key, str(value))


def _sanitize_copy(mol: Chem.Mol) -> tuple[Chem.Mol | None, str]:
    """Return a sanitized molecule copy when possible."""
    candidate = Chem.Mol(mol)
    try:
        candidate.UpdatePropertyCache(strict=False)
        status = Chem.SanitizeMol(candidate, catchErrors=True)
    except Exception as exc:
        return None, f"sanitize_exception:{type(exc).__name__}"
    if status != Chem.SanitizeFlags.SANITIZE_NONE:
        return None, f"sanitize_failed:{int(status)}"
    return candidate, "ok"


def _bad_bond_length_fraction(mol: Chem.Mol) -> float:
    """Return fraction of heavy-atom bonds outside covalent-radius tolerance."""
    if mol.GetNumBonds() == 0 or mol.GetNumConformers() == 0:
        return 0.0
    conformer = mol.GetConformer()
    table = Chem.GetPeriodicTable()
    bad_count = 0
    checked_count = 0
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtom()
        end = bond.GetEndAtom()
        if begin.GetAtomicNum() <= 1 or end.GetAtomicNum() <= 1:
            continue
        expected = table.GetRcovalent(begin.GetAtomicNum()) + table.GetRcovalent(end.GetAtomicNum())
        tolerance = max(0.20, expected * 0.30)
        distance = conformer.GetAtomPosition(begin.GetIdx()).Distance(conformer.GetAtomPosition(end.GetIdx()))
        checked_count += 1
        if distance < expected - tolerance or distance > expected + tolerance:
            bad_count += 1
    return float(bad_count / checked_count) if checked_count else 0.0


def _halogen_bad_position_count(mol: Chem.Mol) -> int:
    """Count suspicious halogen graph placements."""
    count = 0
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in HALOGENS:
            continue
        heavy_neighbors = [nbr for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() > 1]
        if len(heavy_neighbors) != 1:
            count += 1
    return count


def _chemistry_facts(mol: Chem.Mol | None) -> dict[str, float | int | str]:
    """Compute per-molecule chemistry facts used for scoring review."""
    if mol is None:
        return {
            "scoring_valid": 0,
            "canonical_smiles": "",
            "heavy_atom_count": 0,
            "carbon_fraction": 0.0,
            "hetero_atom_count": 0,
            "hetero_atom_fraction": 0.0,
            "logp": float("nan"),
            "tpsa": float("nan"),
            "qed": float("nan"),
            "hba": 0,
            "hbd": 0,
            "rotatable_bonds": 0,
            "ring_count": 0,
            "aromatic_ring_count": 0,
            "three_member_ring_count": 0,
            "fragment_count": 0,
            "bad_bond_length_fraction": 0.0,
            "halogen_bad_position_count": 0,
            "guard_violation_count": 1,
        }

    heavy_atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
    heavy_atom_count = len(heavy_atoms)
    carbon_count = sum(1 for atom in heavy_atoms if atom.GetSymbol() == "C")
    hetero_atom_count = heavy_atom_count - carbon_count
    carbon_fraction = carbon_count / heavy_atom_count if heavy_atom_count else 0.0
    hetero_atom_fraction = hetero_atom_count / heavy_atom_count if heavy_atom_count else 0.0
    ring_info = mol.GetRingInfo()
    three_member_ring_count = sum(1 for ring in ring_info.AtomRings() if len(ring) == 3)
    fragment_count = len(Chem.GetMolFrags(mol, sanitizeFrags=False))
    bad_bond_length_fraction = _bad_bond_length_fraction(mol)
    halogen_bad_position_count = _halogen_bad_position_count(mol)
    guard_violation_count = 0
    guard_violation_count += int(carbon_fraction > 0.85)
    guard_violation_count += int(hetero_atom_fraction < 0.05)
    guard_violation_count += int(three_member_ring_count > 0)
    guard_violation_count += int(fragment_count > 1)
    guard_violation_count += int(bad_bond_length_fraction > 0.05)
    guard_violation_count += int(halogen_bad_position_count > 0)

    return {
        "scoring_valid": 1,
        "canonical_smiles": Chem.MolToSmiles(mol, canonical=True),
        "heavy_atom_count": heavy_atom_count,
        "carbon_fraction": carbon_fraction,
        "hetero_atom_count": hetero_atom_count,
        "hetero_atom_fraction": hetero_atom_fraction,
        "logp": float(Crippen.MolLogP(mol)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "qed": float(QED.qed(mol)),
        "hba": int(Lipinski.NumHAcceptors(mol)),
        "hbd": int(Lipinski.NumHDonors(mol)),
        "rotatable_bonds": int(Lipinski.NumRotatableBonds(mol)),
        "ring_count": int(Descriptors.RingCount(mol)),
        "aromatic_ring_count": int(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "three_member_ring_count": int(three_member_ring_count),
        "fragment_count": int(fragment_count),
        "bad_bond_length_fraction": bad_bond_length_fraction,
        "halogen_bad_position_count": int(halogen_bad_position_count),
        "guard_violation_count": int(guard_violation_count),
    }


def _percentile_rank(values: np.ndarray) -> np.ndarray:
    """Return percentile ranks in [0, 1], with high values ranked high."""
    if values.size == 0:
        return values
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    ranks[order] = np.arange(values.shape[0], dtype=np.float64)
    denom = max(values.shape[0] - 1, 1)
    return ranks / denom


def _load_mols(path: Path) -> list[tuple[int, Chem.Mol | None, Chem.Mol | None, str]]:
    """Load molecules as original and sanitized-copy pairs."""
    supplier = Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False)
    rows: list[tuple[int, Chem.Mol | None, Chem.Mol | None, str]] = []
    for idx, mol in enumerate(supplier):
        if mol is None:
            rows.append((idx, None, None, "supplier_none"))
            continue
        sanitized, status = _sanitize_copy(mol)
        rows.append((idx, mol, sanitized, status))
    return rows


def _predict_probs(
    mols: list[Chem.Mol | None],
    model_path: Path,
    *,
    batch_size: int,
    device: str,
) -> np.ndarray:
    """Predict classifier probabilities for sanitized molecules."""
    probs = np.full((len(mols),), np.nan, dtype=np.float64)
    valid_items = [(idx, mol) for idx, mol in enumerate(mols) if mol is not None]
    if not valid_items:
        return probs

    featurizer = get_featurizer("morgan_count_rdkit_2d")
    model = ChemblClassifier.from_checkpoint(str(model_path), device=device)
    features: list[np.ndarray] = []
    indices: list[int] = []
    for idx, mol in valid_items:
        try:
            feat = featurizer.get_feat(mol)
        except Exception:
            continue
        if feat is None:
            continue
        features.append(feat.astype(np.float32, copy=False))
        indices.append(idx)

    if not features:
        return probs

    stacked = np.stack(features, axis=0)
    out: list[np.ndarray] = []
    for start in range(0, stacked.shape[0], batch_size):
        batch = torch.tensor(stacked[start : start + batch_size], dtype=torch.float32, device=device)
        pred = model.predict_proba(batch).detach().cpu().numpy().reshape(-1)
        out.append(pred)
    pred_all = np.concatenate(out, axis=0).astype(np.float64)
    for idx, prob in zip(indices, pred_all):
        probs[idx] = float(prob)
    return probs


def _write_outputs(
    rows: list[dict[str, Any]],
    original_mols: list[Chem.Mol | None],
    *,
    out_sdf: Path,
    out_csv: Path,
) -> int:
    """Write annotated SDF and CSV."""
    out_sdf.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(out_sdf))
    writer.SetKekulize(False)
    written = 0
    for row, mol in zip(rows, original_mols):
        if mol is None:
            continue
        out_mol = Chem.Mol(mol)
        out_mol.SetProp("_Name", str(row["mol_id"]))
        for key, value in row.items():
            _safe_set_prop(out_mol, key, value)
        writer.write(out_mol)
        written += 1
    writer.close()

    fieldnames = list(rows[0].keys()) if rows else []
    with out_csv.open("w", newline="") as f:
        csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
        csv_writer.writeheader()
        csv_writer.writerows(rows)
    return written


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a compact scoring summary."""
    valid_rows = [row for row in rows if int(row["scoring_valid"]) == 1]

    def vals(key: str) -> np.ndarray:
        arr = np.array([float(row[key]) for row in valid_rows if row.get(key) not in ("", None)], dtype=np.float64)
        return arr[np.isfinite(arr)]

    summary: dict[str, Any] = {
        "input_records": len(rows),
        "scoring_valid": len(valid_rows),
        "scoring_invalid": len(rows) - len(valid_rows),
    }
    for key in ("p_good_v02", "p_good_v021", "dynamic_score_v02", "guarded_score"):
        arr = vals(key)
        if arr.size == 0:
            continue
        summary[f"{key}_mean"] = float(np.mean(arr))
        summary[f"{key}_median"] = float(np.median(arr))
        summary[f"{key}_q10"] = float(np.quantile(arr, 0.10))
        summary[f"{key}_q20"] = float(np.quantile(arr, 0.20))
        summary[f"{key}_q80"] = float(np.quantile(arr, 0.80))
        summary[f"{key}_q90"] = float(np.quantile(arr, 0.90))
    for key in ("carbon_fraction", "hetero_atom_fraction", "logp", "tpsa", "qed", "bad_bond_length_fraction"):
        arr = vals(key)
        if arr.size == 0:
            continue
        summary[f"{key}_mean"] = float(np.mean(arr))
        summary[f"{key}_median"] = float(np.median(arr))
    if valid_rows:
        summary["guard_violation_any_ratio"] = float(np.mean([int(row["guard_violation_count"]) > 0 for row in valid_rows]))
        summary["three_member_ring_any_ratio"] = float(np.mean([int(row["three_member_ring_count"]) > 0 for row in valid_rows]))
        summary["fragment_gt1_ratio"] = float(np.mean([int(row["fragment_count"]) > 1 for row in valid_rows]))
    return summary


def build_argparser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-sdf", type=Path, required=True)
    parser.add_argument("--out-sdf", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--v02-model", type=Path, default=DEFAULT_V02_MODEL)
    parser.add_argument("--v021-model", type=Path, default=DEFAULT_V021_MODEL)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mol-id-prefix", type=str, default="diffsbdd5000")
    parser.add_argument("--guard-penalty", type=float, default=0.5)
    return parser


def main() -> None:
    """Score molecules and export annotated files."""
    args = build_argparser().parse_args()
    rows_loaded = _load_mols(args.input_sdf)
    original_mols = [item[1] for item in rows_loaded]
    sanitized_mols = [item[2] for item in rows_loaded]
    statuses = [item[3] for item in rows_loaded]

    p_v02 = _predict_probs(sanitized_mols, args.v02_model, batch_size=args.batch_size, device=args.device)
    p_v021 = _predict_probs(sanitized_mols, args.v021_model, batch_size=args.batch_size, device=args.device)

    valid_mask = np.isfinite(p_v02)
    v02_rank = np.full_like(p_v02, np.nan, dtype=np.float64)
    v021_rank = np.full_like(p_v021, np.nan, dtype=np.float64)
    if np.any(valid_mask):
        v02_rank[valid_mask] = _percentile_rank(p_v02[valid_mask])
    valid_mask_021 = np.isfinite(p_v021)
    if np.any(valid_mask_021):
        v021_rank[valid_mask_021] = _percentile_rank(p_v021[valid_mask_021])

    q20_v021 = float(np.nanquantile(p_v021, 0.20)) if np.any(valid_mask_021) else float("nan")

    output_rows: list[dict[str, Any]] = []
    for loaded_idx, original, sanitized, status in rows_loaded:
        facts = _chemistry_facts(sanitized)
        mol_id = f"{args.mol_id_prefix}_{loaded_idx:05d}"
        guard_flag = int(facts["guard_violation_count"]) > 0
        low_v021_flag = bool(np.isfinite(p_v021[loaded_idx]) and p_v021[loaded_idx] <= q20_v021)
        guard_penalty = float(args.guard_penalty) if low_v021_flag and guard_flag else 0.0
        dynamic_score = float(v02_rank[loaded_idx]) if np.isfinite(v02_rank[loaded_idx]) else float("nan")
        guarded_score = dynamic_score - guard_penalty if np.isfinite(dynamic_score) else float("nan")
        row: dict[str, Any] = {
            "mol_id": mol_id,
            "source_index": loaded_idx,
            "sanitize_status": status,
            **facts,
            "p_good_v02": float(p_v02[loaded_idx]) if np.isfinite(p_v02[loaded_idx]) else float("nan"),
            "p_good_v021": float(p_v021[loaded_idx]) if np.isfinite(p_v021[loaded_idx]) else float("nan"),
            "rank_v02": float(v02_rank[loaded_idx]) if np.isfinite(v02_rank[loaded_idx]) else float("nan"),
            "rank_v021": float(v021_rank[loaded_idx]) if np.isfinite(v021_rank[loaded_idx]) else float("nan"),
            "dynamic_score_v02": dynamic_score,
            "v021_low_q20_flag": int(low_v021_flag),
            "guard_penalty": guard_penalty,
            "guarded_score": guarded_score,
            "score": guarded_score,
            "score_definition": "guarded_score=rank_v02-guard_penalty",
        }
        output_rows.append(row)

    written = _write_outputs(output_rows, original_mols, out_sdf=args.out_sdf, out_csv=args.out_csv)
    summary = _summary(output_rows)
    summary.update(
        {
            "input_sdf": str(args.input_sdf),
            "out_sdf": str(args.out_sdf),
            "out_csv": str(args.out_csv),
            "sdf_written": written,
            "v02_model": str(args.v02_model),
            "v021_model": str(args.v021_model),
            "v021_q20_threshold": q20_v021,
        }
    )
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
