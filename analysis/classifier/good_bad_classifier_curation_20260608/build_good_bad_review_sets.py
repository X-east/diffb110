from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "analysis" / "good_bad_classifier_curation_20260608"
REVIEW_BY_REASON_DIR = OUT_DIR / "sdf_review_by_reason"
REVIEW_BY_LABEL_DIR = OUT_DIR / "sdf_review_by_label"
PRIMARY_CATEGORY_DIR = OUT_DIR / "sdf_review_by_primary_category"

FAILED_RUN_DIRS = [
    ROOT / "OUT/7gpu-dock-classifier-v0-1_20260529_133609",
    ROOT / "OUT/7gpu-dock-classifier-v0-1_20260528_000841",
]
BASE_SDF_SOURCES = [
    ROOT / "OUT/260407_generate_ligands_10000/260407generated_10000_filtered.sdf",
]

REVIEW_MAX_PER_REASON = 200
REVIEW_MAX_PER_LABEL = 300
REVIEW_MAX_PER_PRIMARY_CATEGORY = 250


HALOGENS = frozenset({"F", "Cl", "Br", "I"})


@dataclass(frozen=True)
class MolFacts:
    parse_ok: int
    canonical_smiles: str | None
    heavy_atoms: int = 0
    carbon_atoms: int = 0
    hetero_atoms: int = 0
    oxygen_count: int = 0
    non_o_hetero_count: int = 0
    halogen_count: int = 0
    element_type_count: int = 0
    carbon_fraction: float = 0.0
    hetero_fraction: float = 0.0
    mw: float = 0.0
    logp: float = 0.0
    tpsa: float = 0.0
    qed: float = 0.0
    hba: int = 0
    hbd: int = 0
    rot_bonds: int = 0
    ring_count: int = 0
    aromatic_ring_count: int = 0
    max_fused_ring_cluster: int = 0
    max_aromatic_fused_cluster: int = 0
    bridgehead_atoms: int = 0
    spiro_atoms: int = 0
    three_member_ring_count: int = 0
    fragment_count: int = 0
    bad_bond_length_fraction: float = 0.0
    halogen_bad_position_count: int = 0


def _largest_component_size(edges: list[set[int]], n_nodes: int) -> int:
    if n_nodes == 0:
        return 0
    seen: set[int] = set()
    best = 1
    for start in range(n_nodes):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            for nxt in edges[node]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        best = max(best, size)
    return best


def _bad_bond_length_fraction(mol: Chem.Mol | None) -> float:
    if mol is None or mol.GetNumBonds() == 0 or mol.GetNumConformers() == 0:
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
    count = 0
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in HALOGENS:
            continue
        heavy_neighbors = [nbr for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() > 1]
        if len(heavy_neighbors) != 1:
            count += 1
    return count


def compute_facts(smiles: str | None, sdf_mol: Chem.Mol | None) -> MolFacts:
    mol = Chem.MolFromSmiles(str(smiles)) if smiles and str(smiles) != "nan" else None
    if mol is None and sdf_mol is not None:
        try:
            smiles = Chem.MolToSmiles(sdf_mol)
            mol = Chem.MolFromSmiles(smiles)
        except Exception:
            mol = None
    if mol is None:
        return MolFacts(parse_ok=0, canonical_smiles=None)

    canonical = Chem.MolToSmiles(mol, isomericSmiles=False)
    heavy_atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
    heavy = len(heavy_atoms)
    if heavy == 0:
        return MolFacts(parse_ok=0, canonical_smiles=None)

    carbon = sum(1 for atom in heavy_atoms if atom.GetSymbol() == "C")
    hetero = heavy - carbon
    oxygen = sum(1 for atom in heavy_atoms if atom.GetSymbol() == "O")
    halogens = sum(1 for atom in heavy_atoms if atom.GetSymbol() in HALOGENS)
    non_o_hetero = hetero - oxygen
    symbols = {atom.GetSymbol() for atom in heavy_atoms}
    ring_info = mol.GetRingInfo()
    rings = list(ring_info.AtomRings())
    ring_sets = [set(ring) for ring in rings]
    aromatic_ring_ids = [
        idx
        for idx, ring in enumerate(rings)
        if all(mol.GetAtomWithIdx(atom_idx).GetIsAromatic() for atom_idx in ring)
    ]
    aromatic_ring_id_set = set(aromatic_ring_ids)
    fused_edges = [set() for _ in rings]
    aromatic_fused_edges = [set() for _ in rings]
    for i in range(len(rings)):
        for j in range(i + 1, len(rings)):
            if len(ring_sets[i].intersection(ring_sets[j])) >= 2:
                fused_edges[i].add(j)
                fused_edges[j].add(i)
                if i in aromatic_ring_id_set and j in aromatic_ring_id_set:
                    aromatic_fused_edges[i].add(j)
                    aromatic_fused_edges[j].add(i)

    return MolFacts(
        parse_ok=1,
        canonical_smiles=canonical,
        heavy_atoms=heavy,
        carbon_atoms=carbon,
        hetero_atoms=hetero,
        oxygen_count=oxygen,
        non_o_hetero_count=non_o_hetero,
        halogen_count=halogens,
        element_type_count=len(symbols),
        carbon_fraction=carbon / heavy,
        hetero_fraction=hetero / heavy,
        mw=float(Descriptors.MolWt(mol)),
        logp=float(Crippen.MolLogP(mol)),
        tpsa=float(rdMolDescriptors.CalcTPSA(mol)),
        qed=float(QED.qed(mol)),
        hba=int(Lipinski.NumHAcceptors(mol)),
        hbd=int(Lipinski.NumHDonors(mol)),
        rot_bonds=int(Lipinski.NumRotatableBonds(mol)),
        ring_count=len(rings),
        aromatic_ring_count=len(aromatic_ring_ids),
        max_fused_ring_cluster=_largest_component_size(fused_edges, len(rings)),
        max_aromatic_fused_cluster=_largest_component_size(
            aromatic_fused_edges, len(aromatic_ring_ids)
        ),
        bridgehead_atoms=int(rdMolDescriptors.CalcNumBridgeheadAtoms(mol)),
        spiro_atoms=int(rdMolDescriptors.CalcNumSpiroAtoms(mol)),
        three_member_ring_count=sum(1 for ring in rings if len(ring) == 3),
        fragment_count=len(Chem.GetMolFrags(mol, sanitizeFrags=False)),
        bad_bond_length_fraction=_bad_bond_length_fraction(sdf_mol),
        halogen_bad_position_count=_halogen_bad_position_count(mol),
    )


def _float_or_nan(value: object) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out


def _safe_str(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def iter_failed_run_records() -> Iterable[tuple[dict, Chem.Mol | None]]:
    for run_dir in FAILED_RUN_DIRS:
        epoch_dirs = sorted((run_dir / "epochs").glob("epoch_*"))
        for epoch_dir in epoch_dirs:
            csv_path = epoch_dir / "molecules.csv"
            sdf_path = epoch_dir / "molecules.sdf"
            if not csv_path.exists() or not sdf_path.exists():
                continue
            df = pd.read_csv(csv_path)
            mols = list(Chem.SDMolSupplier(str(sdf_path), sanitize=False, removeHs=False))
            n = min(len(df), len(mols))
            for i in range(n):
                row = df.iloc[i].to_dict()
                row["source_kind"] = "failed_rl"
                row["source_run"] = run_dir.name
                row["source_epoch_dir"] = epoch_dir.name
                row["source_sdf"] = str(sdf_path.relative_to(ROOT))
                row["source_index"] = i
                yield row, mols[i]


def iter_base_sdf_records() -> Iterable[tuple[dict, Chem.Mol | None]]:
    for sdf_path in BASE_SDF_SOURCES:
        mols = list(Chem.SDMolSupplier(str(sdf_path), sanitize=False, removeHs=False))
        for i, mol in enumerate(mols):
            smiles = ""
            if mol is not None:
                try:
                    smiles = Chem.MolToSmiles(mol)
                except Exception:
                    smiles = ""
            yield {
                "source_kind": "base_diffsbdd",
                "source_run": sdf_path.stem,
                "source_epoch_dir": "",
                "source_sdf": str(sdf_path.relative_to(ROOT)),
                "source_index": i,
                "epoch": "",
                "batch_idx": "",
                "rank": "",
                "molecule_id": i,
                "smiles": smiles,
                "rdkit_valid": int(mol is not None),
                "docking_raw": float("nan"),
                "chembl_classifier_reward": float("nan"),
                "reward_total_raw": float("nan"),
            }, mol


def is_ring_complex(row: dict) -> bool:
    return (
        int(row["ring_count"]) >= 5
        or int(row["max_fused_ring_cluster"]) >= 4
        or int(row["bridgehead_atoms"]) >= 2
        or int(row["spiro_atoms"]) >= 2
    )


def strict_chemistry_pass(row: dict) -> bool:
    return (
        bool(row["parse_ok"])
        and int(row["rdkit_valid"]) == 1
        and 12 <= int(row["heavy_atoms"]) <= 32
        and 180 <= float(row["mw"]) <= 450
        and 0.0 <= float(row["logp"]) <= 4.0
        and 35 <= float(row["tpsa"]) <= 130
        and 1 <= int(row["hba"]) <= 10
        and int(row["hbd"]) <= 5
        and int(row["rot_bonds"]) <= 8
        and float(row["carbon_fraction"]) <= 0.78
        and int(row["hetero_atoms"]) >= 3
        and int(row["non_o_hetero_count"]) >= 1
        and int(row["three_member_ring_count"]) == 0
        and int(row["fragment_count"]) == 1
        and int(row["ring_count"]) <= 4
        and not is_ring_complex(row)
        and float(row["bad_bond_length_fraction"]) <= 0.05
        and int(row["halogen_bad_position_count"]) == 0
    )


def assign_reasons(row: dict) -> tuple[list[str], list[str]]:
    good: list[str] = []
    bad: list[str] = []
    parse_ok = int(row["parse_ok"]) == 1
    rdkit_valid = int(row["rdkit_valid"]) == 1
    docking = _float_or_nan(row.get("docking_raw"))
    classifier_reward = _float_or_nan(row.get("chembl_classifier_reward"))

    if not rdkit_valid or not parse_ok:
        bad.append("bad_invalid_or_unparseable")
        return good, bad

    if int(row["fragment_count"]) > 1:
        bad.append("bad_fragmented")
    if float(row["bad_bond_length_fraction"]) > 0.05:
        bad.append("bad_bad_bond_length")
    if int(row["three_member_ring_count"]) > 0:
        bad.append("bad_three_member_ring")
    if is_ring_complex(row):
        bad.append("bad_ring_complex_failed_rl")
    if int(row["aromatic_ring_count"]) == 0 and int(row["ring_count"]) >= 3:
        bad.append("bad_non_aromatic_polycycle")
    if (
        float(row["carbon_fraction"]) >= 0.90
        or (float(row["carbon_fraction"]) >= 0.80 and int(row["hetero_atoms"]) <= 3)
        or (float(row["carbon_fraction"]) >= 0.75 and int(row["non_o_hetero_count"]) == 0)
    ):
        bad.append("bad_high_carbon_low_hetero")
    if float(row["tpsa"]) < 40 or float(row["logp"]) > 5.0 or (
        float(row["tpsa"]) < 60 and float(row["logp"]) > 4.0
    ):
        bad.append("bad_low_tpsa_or_high_logp")
    if int(row["halogen_bad_position_count"]) > 0:
        bad.append("bad_halogen_position")

    base_bad_count = len(bad)
    if row["source_kind"] == "failed_rl" and docking <= -8.0 and base_bad_count > 0:
        bad.append("bad_top_docking_exploit")
    if (
        row["source_kind"] == "failed_rl"
        and classifier_reward >= 0.5
        and base_bad_count > 0
    ):
        bad.append("bad_classifier_false_positive")

    if strict_chemistry_pass(row):
        good.append("good_strict_chemistry")
        if row["source_kind"] == "failed_rl" and docking <= -7.5:
            good.append("good_strict_with_docking")
        if not math.isnan(classifier_reward) and classifier_reward > 0:
            good.append("good_classifier_accepted")

    return sorted(set(good)), sorted(set(bad))


def primary_category(good_reasons: list[str], bad_reasons: list[str]) -> tuple[str, str]:
    if bad_reasons:
        priority = [
            "bad_classifier_false_positive",
            "bad_top_docking_exploit",
            "bad_bad_bond_length",
            "bad_fragmented",
            "bad_three_member_ring",
            "bad_ring_complex_failed_rl",
            "bad_non_aromatic_polycycle",
            "bad_high_carbon_low_hetero",
            "bad_low_tpsa_or_high_logp",
            "bad_halogen_position",
            "bad_invalid_or_unparseable",
        ]
        for reason in priority:
            if reason in bad_reasons:
                return "bad", reason
        return "bad", bad_reasons[0]
    if "good_strict_with_docking" in good_reasons:
        return "good_high_conf", "good_strict_with_docking"
    if good_reasons:
        return "good_candidate", good_reasons[0]
    return "borderline_or_exclude", "no_clear_good_or_bad_reason"


def severity_score(row: dict) -> float:
    score = 0.0
    score += max(0.0, float(row["carbon_fraction"]) - 0.78) * 5.0
    score += max(0.0, 60.0 - float(row["tpsa"])) / 30.0
    score += max(0.0, float(row["logp"]) - 4.0) / 2.0
    score += int(row["three_member_ring_count"]) * 1.5
    score += int(row["ring_count"]) * 0.25
    score += int(row["bridgehead_atoms"]) * 0.4
    score += int(row["spiro_atoms"]) * 0.4
    score += float(row["bad_bond_length_fraction"]) * 5.0
    docking = _float_or_nan(row.get("docking_raw"))
    classifier_reward = _float_or_nan(row.get("chembl_classifier_reward"))
    if docking <= -8:
        score += 1.0
    if not math.isnan(classifier_reward) and classifier_reward >= 0.5:
        score += 1.0
    return float(score)


def good_score(row: dict) -> float:
    score = 0.0
    score += float(row["qed"])
    score += max(0.0, 1.0 - abs(float(row["logp"]) - 2.0) / 3.0)
    score += max(0.0, 1.0 - abs(float(row["tpsa"]) - 80.0) / 80.0)
    docking = _float_or_nan(row.get("docking_raw"))
    classifier_reward = _float_or_nan(row.get("chembl_classifier_reward"))
    if docking <= -8:
        score += 1.0
    if not math.isnan(classifier_reward) and classifier_reward > 0:
        score += 0.5
    return float(score)


def _mol_with_props(mol: Chem.Mol | None, row: pd.Series) -> Chem.Mol | None:
    out = Chem.Mol(mol) if mol is not None else None
    if out is None:
        smiles = _safe_str(row.get("canonical_smiles")) or _safe_str(row.get("smiles"))
        out = Chem.MolFromSmiles(smiles)
        if out is None:
            return None
    for key in [
        "label",
        "primary_reason",
        "good_reasons",
        "bad_reasons",
        "source_kind",
        "source_run",
        "source_epoch_dir",
        "source_index",
        "canonical_smiles",
        "docking_raw",
        "chembl_classifier_reward",
        "reward_total_raw",
        "severity_score",
        "good_score",
        "carbon_fraction",
        "hetero_atoms",
        "non_o_hetero_count",
        "logp",
        "tpsa",
        "ring_count",
        "aromatic_ring_count",
        "three_member_ring_count",
        "bad_bond_length_fraction",
        "fragment_count",
    ]:
        if key in row:
            out.SetProp(key, _safe_str(row.get(key)))
    out.SetProp("_Name", f"{row.get('label', '')}|{row.get('primary_reason', '')}|{row.get('canonical_smiles', '')}")
    return out


def write_sdf(path: Path, rows: pd.DataFrame, mols_by_record_id: dict[int, Chem.Mol | None]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(path))
    written = 0
    seen: set[str] = set()
    for _, row in rows.iterrows():
        smi = _safe_str(row.get("canonical_smiles"))
        if smi and smi in seen:
            continue
        seen.add(smi)
        mol = _mol_with_props(mols_by_record_id.get(int(row["record_id"])), row)
        if mol is None:
            continue
        writer.write(mol)
        written += 1
    writer.close()
    return written


def build_records() -> tuple[pd.DataFrame, dict[int, Chem.Mol | None]]:
    records: list[dict] = []
    mols_by_record_id: dict[int, Chem.Mol | None] = {}
    for row, mol in list(iter_failed_run_records()) + list(iter_base_sdf_records()):
        record_id = len(records)
        facts = compute_facts(row.get("smiles"), mol)
        rec = {
            "record_id": record_id,
            "source_kind": row.get("source_kind", ""),
            "source_run": row.get("source_run", ""),
            "source_epoch_dir": row.get("source_epoch_dir", ""),
            "source_sdf": row.get("source_sdf", ""),
            "source_index": row.get("source_index", ""),
            "epoch": row.get("epoch", ""),
            "batch_idx": row.get("batch_idx", ""),
            "rank": row.get("rank", ""),
            "molecule_id": row.get("molecule_id", ""),
            "smiles": row.get("smiles", ""),
            "rdkit_valid": int(row.get("rdkit_valid", 0) == 1),
            "docking_raw": _float_or_nan(row.get("docking_raw")),
            "chembl_classifier_reward": _float_or_nan(row.get("chembl_classifier_reward")),
            "reward_total_raw": _float_or_nan(row.get("reward_total_raw")),
            **asdict(facts),
        }
        good_reasons, bad_reasons = assign_reasons(rec)
        label, primary_reason = primary_category(good_reasons, bad_reasons)
        rec["good_reasons"] = "|".join(good_reasons)
        rec["bad_reasons"] = "|".join(bad_reasons)
        rec["label"] = label
        rec["primary_reason"] = primary_reason
        rec["strict_chemistry_pass"] = int("good_strict_chemistry" in good_reasons)
        rec["severity_score"] = severity_score(rec)
        rec["good_score"] = good_score(rec)
        records.append(rec)
        mols_by_record_id[record_id] = mol
    return pd.DataFrame(records), mols_by_record_id


def write_review_sets(df: pd.DataFrame, mols_by_record_id: dict[int, Chem.Mol | None]) -> pd.DataFrame:
    written_rows: list[dict] = []
    reason_to_ids: dict[str, set[int]] = defaultdict(set)
    for _, row in df.iterrows():
        for reason in filter(None, str(row["good_reasons"]).split("|")):
            reason_to_ids[reason].add(int(row["record_id"]))
        for reason in filter(None, str(row["bad_reasons"]).split("|")):
            reason_to_ids[reason].add(int(row["record_id"]))

    for reason, ids in sorted(reason_to_ids.items()):
        rows = df[df["record_id"].isin(ids)].copy()
        sort_col = "good_score" if reason.startswith("good_") else "severity_score"
        rows = rows.sort_values(sort_col, ascending=False).head(REVIEW_MAX_PER_REASON)
        n_written = write_sdf(REVIEW_BY_REASON_DIR / f"{reason}.sdf", rows, mols_by_record_id)
        written_rows.append({"group_type": "reason", "name": reason, "n_source": len(ids), "n_sdf": n_written})

    for label, rows in df.groupby("label"):
        n_source = len(rows)
        sort_col = "good_score" if label.startswith("good") else "severity_score"
        rows = rows.sort_values(sort_col, ascending=False).head(REVIEW_MAX_PER_LABEL)
        n_written = write_sdf(REVIEW_BY_LABEL_DIR / f"{label}.sdf", rows, mols_by_record_id)
        written_rows.append({"group_type": "label", "name": label, "n_source": n_source, "n_sdf": n_written})

    for reason, rows in df.groupby("primary_reason"):
        n_source = len(rows)
        sort_col = "good_score" if reason.startswith("good_") else "severity_score"
        rows = rows.sort_values(sort_col, ascending=False).head(REVIEW_MAX_PER_PRIMARY_CATEGORY)
        n_written = write_sdf(PRIMARY_CATEGORY_DIR / f"{reason}.sdf", rows, mols_by_record_id)
        written_rows.append({"group_type": "primary_category", "name": reason, "n_source": n_source, "n_sdf": n_written})

    return pd.DataFrame(written_rows)


def write_summary_plot(df: pd.DataFrame, category_summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    label_counts = df["label"].value_counts().sort_values(ascending=True)
    axes[0].barh(label_counts.index, label_counts.values, color="#4E79A7")
    axes[0].set_title("Label counts")
    axes[0].set_xlabel("molecules")

    bad_reason_counts = []
    for reasons in df["bad_reasons"].fillna(""):
        for reason in filter(None, str(reasons).split("|")):
            bad_reason_counts.append(reason)
    bad_counts = pd.Series(bad_reason_counts).value_counts().head(12).sort_values(ascending=True)
    axes[1].barh(bad_counts.index, bad_counts.values, color="#E15759")
    axes[1].set_title("Top bad reasons")
    axes[1].set_xlabel("molecules")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "curation_label_distribution.png", dpi=180)
    fig.savefig(OUT_DIR / "curation_label_distribution.svg")


def write_readme(df: pd.DataFrame, sdf_manifest: pd.DataFrame) -> None:
    summary = {
        "n_records": int(len(df)),
        "n_unique_canonical_smiles": int(df["canonical_smiles"].dropna().nunique()),
        "label_counts": df["label"].value_counts().to_dict(),
        "source_counts": df["source_run"].value_counts().head(20).to_dict(),
        "review_max_per_reason": REVIEW_MAX_PER_REASON,
        "review_max_per_label": REVIEW_MAX_PER_LABEL,
        "review_max_per_primary_category": REVIEW_MAX_PER_PRIMARY_CATEGORY,
    }
    (OUT_DIR / "curation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    manifest_preview = sdf_manifest.head(30).to_string(index=False)
    lines = [
        "# 好坏分子分类审阅集",
        "",
        "本目录用于调优 ChEMBL/good-bad 分子分类器前的人工审阅。",
        "",
        "## 主要产物",
        "",
        "- `curated_molecules.csv.gz`：全量候选分子、标签、理由和描述子。",
        "- `category_summary.csv`：按 label / primary reason / source 的计数汇总。",
        "- `sdf_review_by_reason/`：按每个好/坏理由抽样导出的 SDF，每类最多 200 个，分子可重复进入多个理由文件。",
        "- `sdf_review_by_label/`：按总标签抽样导出的 SDF。",
        "- `sdf_review_by_primary_category/`：按主理由抽样导出的 SDF，每条分子只有一个主理由。",
        "- `curation_label_distribution.png`：标签和坏理由分布图。",
        "",
        "## 标签含义",
        "",
        "- `good_high_conf`：严格化学护栏通过，并且 failed RL 中 docking_raw <= -7.5。",
        "- `good_candidate`：严格化学护栏通过，但不一定有强 docking 证据。",
        "- `bad`：触发至少一个坏理由，例如复杂环、高碳低杂、三元环、bad bond length、top docking exploit 或 classifier false positive。",
        "- `borderline_or_exclude`：没有明确好/坏理由，先不进入训练。",
        "",
        "## 注意",
        "",
        "SDF 是审阅用抽样文件，不是最终训练集；最终训练应先人工检查这些类别，再决定哪些 reason 进入 hard negative、哪些作为 soft/borderline。",
        "",
        "## 汇总",
        "",
        "```json",
        json.dumps(summary, ensure_ascii=False, indent=2),
        "```",
        "",
        "## SDF manifest preview",
        "",
        "```text",
        manifest_preview,
        "```",
        "",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(lines))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REVIEW_BY_REASON_DIR.mkdir(parents=True, exist_ok=True)
    REVIEW_BY_LABEL_DIR.mkdir(parents=True, exist_ok=True)
    PRIMARY_CATEGORY_DIR.mkdir(parents=True, exist_ok=True)

    df, mols_by_record_id = build_records()
    df.to_csv(OUT_DIR / "curated_molecules.csv.gz", index=False)

    label_summary = df.groupby(["label"]).size().rename("n").reset_index().sort_values("n", ascending=False)
    primary_reason_summary = (
        df.groupby(["label", "primary_reason"])
        .size()
        .rename("n")
        .reset_index()
        .sort_values("n", ascending=False)
    )
    source_label_summary = (
        df.groupby(["source_run", "label"])
        .size()
        .rename("n")
        .reset_index()
        .sort_values(["source_run", "n"], ascending=[True, False])
    )
    reason_rows: list[dict] = []
    for col in ["good_reasons", "bad_reasons"]:
        reason_type = "good" if col.startswith("good") else "bad"
        for reasons in df[col].fillna(""):
            for reason in filter(None, str(reasons).split("|")):
                reason_rows.append({"reason_type": reason_type, "reason": reason})
    reason_summary = (
        pd.DataFrame(reason_rows)
        .groupby(["reason_type", "reason"])
        .size()
        .rename("n")
        .reset_index()
        .sort_values("n", ascending=False)
    )
    label_summary.to_csv(OUT_DIR / "label_summary.csv", index=False)
    primary_reason_summary.to_csv(OUT_DIR / "primary_reason_summary.csv", index=False)
    source_label_summary.to_csv(OUT_DIR / "source_label_summary.csv", index=False)
    reason_summary.to_csv(OUT_DIR / "reason_summary.csv", index=False)
    category_summary = pd.concat(
        [
            label_summary.assign(summary_type="label"),
            primary_reason_summary.assign(summary_type="primary_reason"),
            source_label_summary.assign(summary_type="source_label"),
        ],
        ignore_index=True,
        sort=False,
    )
    category_summary.to_csv(OUT_DIR / "category_summary.csv", index=False)

    sdf_manifest = write_review_sets(df, mols_by_record_id)
    sdf_manifest.to_csv(OUT_DIR / "sdf_manifest.csv", index=False)
    write_summary_plot(df, category_summary)
    write_readme(df, sdf_manifest)


if __name__ == "__main__":
    main()
