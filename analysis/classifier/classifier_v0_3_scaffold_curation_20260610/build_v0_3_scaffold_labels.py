"""Build v0.3 classifier labels from old sets and recent generated molecules.

This script is intentionally rule-first: it separates explicit chemical failure
modes into inspectable SDF buckets before any classifier is trained.
"""

from __future__ import annotations

import csv
import json
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-diffb110")

import matplotlib.pyplot as plt
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.logger().setLevel(RDLogger.ERROR)


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "analysis" / "classifier_v0_3_scaffold_curation_20260610"

OLD_POS = ROOT / "analysis/classifier_v0_2_1_weighted_20260608/selected_positives_v0_2_1_weighted.smi"
OLD_NEG = ROOT / "analysis/classifier_v0_2_1_weighted_20260608/selected_negatives_v0_2_1_weighted_labeled.smi"

RUN_SOURCES = [
    ("diffgui_ring_rl", ROOT / "OUT/diffgui-8gpu-ring-last3-lr3e5_20260610_114650"),
    ("diffgui_dock_classifier", ROOT / "OUT/diffgui-8gpu-chembl21-dock_20260609_224313"),
    ("diffgui_docking_only", ROOT / "OUT/diffgui-8gpu-docking-only-lr1e5-last2_20260610_094719"),
]

STATIC_SOURCES = [
    (
        "diffgui_500_scored",
        ROOT / "analysis/diffgui_classifier_score_20260609/diffgui_500_scored.sdf",
    ),
    (
        "diffsbdd_5000_scored",
        ROOT / "analysis/diffsbdd_8gpu_5000_score_20260608/generated_5000_scored.sdf",
    ),
    (
        "diffsbdd_260407_10000_1_filtered",
        ROOT / "OUT/260407_generate_ligands_10000/260407generated_10000_filtered.sdf",
    ),
    (
        "diffsbdd_260407_10000_2_filtered",
        ROOT / "OUT/260407_generate_ligands_10000/260407generated_10000-2_filtered.sdf",
    ),
    (
        "diffsbdd_260407_10000_3_filtered",
        ROOT / "OUT/260407_generate_ligands_10000/260407generated_10000-3_filtered.sdf",
    ),
    (
        "diffsbdd_260407_10000_4_raw",
        ROOT / "OUT/260407_generate_ligands_10000/260407generated_10000-4.sdf",
    ),
    (
        "diffsbdd_260407_10000_5_raw",
        ROOT / "OUT/260407_generate_ligands_10000/260407generated_10000-5.sdf",
    ),
]


HALOGEN_ATOMS = {9, 17, 35, 53}
HEAVY_HALOGEN_ATOMS = {35, 53}


@dataclass(frozen=True)
class ScaffoldFacts:
    """Rule features for one molecule."""

    heavy_atom_count: int
    carbon_fraction: float
    hetero_atom_count: int
    aromatic_ring_count: int
    benzene_ring_count: int
    heterocycle_ring_count: int
    aromatic_heterocycle_ring_count: int
    aliphatic_carbocycle_ring_count: int
    ring_count: int
    max_ring_size: int
    large_ring_count: int
    large_ring_fused_with_six_count: int
    max_fused_ring_system_size: int
    three_member_ring_count: int
    three_member_fused_ring_count: int
    three_member_spiro_ring_count: int
    three_member_high_degree_carbon_count: int
    spiro_atom_count: int
    bridgehead_atom_count: int
    total_halogen_count: int
    aromatic_halogen_count: int
    heavy_halogen_count: int
    br_i_count: int
    logp: float
    tpsa: float
    hba: int
    hbd: int
    fragment_count: int
    murcko_scaffold: str


@dataclass(frozen=True)
class LabelDecision:
    """Rule label and all reasons for one molecule."""

    label: str
    primary_reason: str
    all_reasons: tuple[str, ...]
    weight: float


def canonical_smiles(mol: Chem.Mol) -> str | None:
    try:
        return Chem.MolToSmiles(mol, isomericSmiles=False)
    except Exception:
        return None


def load_sdf_molecules(source: str, path: Path) -> list[tuple[str, Chem.Mol]]:
    """Read molecules from an SDF while preserving a stable source id."""
    if not path.exists():
        return []
    supplier = Chem.SDMolSupplier(str(path), sanitize=True, removeHs=False)
    out: list[tuple[str, Chem.Mol]] = []
    for idx, mol in enumerate(supplier):
        if mol is None:
            continue
        mol_id = mol.GetProp("_Name") if mol.HasProp("_Name") else f"{source}_{idx:06d}"
        out.append((mol_id, mol))
    return out


def compute_facts(mol: Chem.Mol) -> ScaffoldFacts:
    heavy_atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
    heavy_atom_count = len(heavy_atoms)
    carbon_count = sum(1 for atom in heavy_atoms if atom.GetAtomicNum() == 6)
    hetero_atom_count = sum(1 for atom in heavy_atoms if atom.GetAtomicNum() != 6)
    carbon_fraction = carbon_count / heavy_atom_count if heavy_atom_count else 0.0

    ring_info = mol.GetRingInfo()
    atom_rings = [tuple(ring) for ring in ring_info.AtomRings()]
    aromatic_ring_count = int(rdMolDescriptors.CalcNumAromaticRings(mol))
    benzene_ring_count = 0
    heterocycle_ring_count = 0
    aromatic_heterocycle_ring_count = 0
    aliphatic_carbocycle_ring_count = 0
    three_member_ring_count = 0
    three_member_fused_ring_count = 0
    three_member_spiro_ring_count = 0
    three_member_high_degree_carbon_atoms: set[int] = set()
    ring_sets = [set(ring) for ring in atom_rings]
    max_ring_size = max((len(ring) for ring in atom_rings), default=0)
    large_ring_count = 0
    large_ring_fused_with_six_count = 0

    for ring_index, ring in enumerate(atom_rings):
        atoms = [mol.GetAtomWithIdx(atom_idx) for atom_idx in ring]
        is_aromatic = all(atom.GetIsAromatic() for atom in atoms)
        is_all_carbon = all(atom.GetAtomicNum() == 6 for atom in atoms)
        has_hetero = any(atom.GetAtomicNum() not in (1, 6) for atom in atoms)
        if len(ring) == 3:
            three_member_ring_count += 1
            ring_set = set(ring)
            shared_atom_counts = [
                len(ring_set & other)
                for other_index, other in enumerate(ring_sets)
                if other_index != ring_index
            ]
            if any(shared_count >= 2 for shared_count in shared_atom_counts):
                three_member_fused_ring_count += 1
            elif any(shared_count == 1 for shared_count in shared_atom_counts):
                three_member_spiro_ring_count += 1
            for atom in atoms:
                if atom.GetAtomicNum() == 6 and atom.GetDegree() >= 4:
                    three_member_high_degree_carbon_atoms.add(atom.GetIdx())
        if len(ring) >= 7:
            large_ring_count += 1
            ring_set = set(ring)
            if any(
                other_index != ring_index
                and len(other) == 6
                and len(ring_set & other) >= 2
                for other_index, other in enumerate(ring_sets)
            ):
                large_ring_fused_with_six_count += 1
        if is_aromatic and len(ring) == 6 and is_all_carbon:
            benzene_ring_count += 1
        if has_hetero:
            heterocycle_ring_count += 1
            aromatic_heterocycle_ring_count += int(is_aromatic)
        elif not is_aromatic:
            aliphatic_carbocycle_ring_count += 1

    total_halogen_count = 0
    aromatic_halogen_count = 0
    heavy_halogen_count = 0
    br_i_count = 0
    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        if atomic_num not in HALOGEN_ATOMS:
            continue
        total_halogen_count += 1
        heavy_halogen_count += int(atomic_num in HEAVY_HALOGEN_ATOMS)
        br_i_count += int(atomic_num in HEAVY_HALOGEN_ATOMS)
        aromatic_halogen_count += int(any(neighbor.GetIsAromatic() for neighbor in atom.GetNeighbors()))

    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:
        scaffold = ""

    return ScaffoldFacts(
        heavy_atom_count=heavy_atom_count,
        carbon_fraction=carbon_fraction,
        hetero_atom_count=hetero_atom_count,
        aromatic_ring_count=aromatic_ring_count,
        benzene_ring_count=benzene_ring_count,
        heterocycle_ring_count=heterocycle_ring_count,
        aromatic_heterocycle_ring_count=aromatic_heterocycle_ring_count,
        aliphatic_carbocycle_ring_count=aliphatic_carbocycle_ring_count,
        ring_count=len(atom_rings),
        max_ring_size=max_ring_size,
        large_ring_count=large_ring_count,
        large_ring_fused_with_six_count=large_ring_fused_with_six_count,
        max_fused_ring_system_size=max_fused_ring_system_size(atom_rings),
        three_member_ring_count=three_member_ring_count,
        three_member_fused_ring_count=three_member_fused_ring_count,
        three_member_spiro_ring_count=three_member_spiro_ring_count,
        three_member_high_degree_carbon_count=len(three_member_high_degree_carbon_atoms),
        spiro_atom_count=int(rdMolDescriptors.CalcNumSpiroAtoms(mol)),
        bridgehead_atom_count=int(rdMolDescriptors.CalcNumBridgeheadAtoms(mol)),
        total_halogen_count=total_halogen_count,
        aromatic_halogen_count=aromatic_halogen_count,
        heavy_halogen_count=heavy_halogen_count,
        br_i_count=br_i_count,
        logp=float(Crippen.MolLogP(mol)),
        tpsa=float(rdMolDescriptors.CalcTPSA(mol)),
        hba=int(Lipinski.NumHAcceptors(mol)),
        hbd=int(Lipinski.NumHDonors(mol)),
        fragment_count=len(Chem.GetMolFrags(mol, sanitizeFrags=False)),
        murcko_scaffold=scaffold,
    )


def max_fused_ring_system_size(atom_rings: list[tuple[int, ...]]) -> int:
    """Return largest fused ring component size."""
    if not atom_rings:
        return 0
    ring_sets = [set(ring) for ring in atom_rings]
    adjacency: list[set[int]] = [set() for _ in ring_sets]
    for i, left in enumerate(ring_sets):
        for j in range(i + 1, len(ring_sets)):
            if len(left & ring_sets[j]) >= 2:
                adjacency[i].add(j)
                adjacency[j].add(i)
    seen: set[int] = set()
    largest = 1
    for start in range(len(ring_sets)):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            for nxt in adjacency[node]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        largest = max(largest, size)
    return largest


def label_molecule(facts: ScaffoldFacts) -> LabelDecision:
    """Assign v0.3 rule label and reasons.

    Good labels are deliberately stricter than bad labels. Ambiguous molecules
    should be reviewed rather than used as positives by default.
    """
    bad: list[str] = []
    nonhalogen_hetero_count = max(facts.hetero_atom_count - facts.total_halogen_count, 0)
    halogen_fraction = facts.total_halogen_count / facts.heavy_atom_count if facts.heavy_atom_count else 0.0
    heavy_halogen_fraction = facts.heavy_halogen_count / facts.heavy_atom_count if facts.heavy_atom_count else 0.0

    if facts.fragment_count > 1:
        bad.append("bad_fragmented")
    complex_three_member_ring = (
        facts.three_member_ring_count >= 2
        or facts.three_member_fused_ring_count > 0
        or facts.three_member_spiro_ring_count > 0
        or facts.three_member_high_degree_carbon_count > 0
    )
    if complex_three_member_ring:
        bad.append("bad_three_member_ring")

    complex_fused_heterocycle = (
        facts.heterocycle_ring_count >= 2
        and (
            facts.max_fused_ring_system_size >= 3
            or facts.ring_count >= 5
            or facts.bridgehead_atom_count > 0
            or facts.spiro_atom_count > 0
        )
    )
    if complex_fused_heterocycle:
        bad.append("bad_fused_heterocycle")

    excessive_aromatic_heterocycles = (
        facts.aromatic_heterocycle_ring_count >= 4
        or (
            facts.aromatic_heterocycle_ring_count >= 3
            and (
                facts.max_fused_ring_system_size >= 3
                or facts.ring_count >= 6
                or facts.heterocycle_ring_count >= 4
            )
        )
    )
    if excessive_aromatic_heterocycles:
        bad.append("bad_multi_aromatic_heterocycle")

    if facts.large_ring_count > 0:
        bad.append("bad_large_ring")

    if facts.aromatic_ring_count == 0 and (
        facts.ring_count >= 3
        or facts.max_fused_ring_system_size >= 3
        or facts.bridgehead_atom_count > 0
        or facts.spiro_atom_count > 0
    ):
        bad.append("bad_saturated_polycycle_no_aromatic")
    if facts.max_fused_ring_system_size >= 5 or facts.bridgehead_atom_count >= 2:
        bad.append("bad_poly_fused_or_bridge_ring")

    small_halogen_dominated = (
        facts.total_halogen_count >= 4
        and facts.heavy_atom_count <= 18
        and halogen_fraction >= 0.24
        and nonhalogen_hetero_count <= 2
        and facts.tpsa < 45.0
    )
    many_heavy_halogens = (
        facts.br_i_count >= 3
        and (facts.heavy_atom_count <= 24 or heavy_halogen_fraction >= 0.12)
        and nonhalogen_hetero_count <= 3
    )
    many_direct_aromatic_halogens = (
        facts.aromatic_halogen_count >= 4
        and (facts.heavy_atom_count <= 24 or halogen_fraction >= 0.22 or nonhalogen_hetero_count <= 1)
    )
    extreme_halogen_dominated = (
        facts.total_halogen_count >= 7
        and halogen_fraction >= 0.22
        and nonhalogen_hetero_count <= 3
    )
    if small_halogen_dominated or many_heavy_halogens or many_direct_aromatic_halogens or extreme_halogen_dominated:
        bad.append("bad_many_halogen")

    small_bare_polyhalo_aromatic = (
        facts.benzene_ring_count >= 1
        and facts.heavy_atom_count <= 12
        and facts.aromatic_halogen_count >= 3
        and nonhalogen_hetero_count == 0
        and facts.tpsa < 15.0
    )
    halogen_dominated_polyhalo_aromatic = (
        facts.benzene_ring_count >= 1
        and facts.heavy_atom_count <= 16
        and facts.aromatic_halogen_count >= 4
        and halogen_fraction >= 0.30
        and nonhalogen_hetero_count <= 1
        and facts.tpsa < 20.0
    )
    if small_bare_polyhalo_aromatic or halogen_dominated_polyhalo_aromatic:
        bad.append("bad_overhalogenated_small_aromatic")

    high_carbon_hydrophobic = (
        facts.carbon_fraction >= 0.90
        and nonhalogen_hetero_count <= 2
        and (facts.logp >= 3.0 or facts.tpsa < 25.0)
    )
    if high_carbon_hydrophobic:
        bad.append("bad_high_carbon_low_hetero")

    if bad:
        primary = choose_primary_bad_reason(bad)
        return LabelDecision(label="bad", primary_reason=primary, all_reasons=tuple(bad), weight=bad_weight(bad))

    good: list[str] = []
    single_benzene_reasonable = (
        facts.benzene_ring_count == 1
        and facts.total_halogen_count <= 2
        and facts.aromatic_halogen_count <= 2
        and 14 <= facts.heavy_atom_count <= 45
        and facts.hetero_atom_count >= 1
        and facts.max_fused_ring_system_size <= 3
        and facts.heterocycle_ring_count <= 1
        and facts.carbon_fraction <= 0.86
        and facts.logp <= 5.5
        and facts.tpsa >= 15.0
    )
    aromatic_reasonable = (
        1 <= facts.aromatic_ring_count <= 2
        and facts.total_halogen_count <= 2
        and 16 <= facts.heavy_atom_count <= 50
        and facts.hetero_atom_count >= 2
        and facts.ring_count <= 5
        and facts.max_fused_ring_system_size <= 3
        and facts.logp <= 5.5
        and facts.tpsa >= 20.0
    )
    if single_benzene_reasonable:
        good.append("good_single_benzene_low_halogen")
    if aromatic_reasonable:
        good.append("good_aromatic_reasonable")

    if good:
        return LabelDecision(label="good", primary_reason=good[0], all_reasons=tuple(good), weight=1.5)

    return LabelDecision(label="ambiguous", primary_reason="ambiguous_manual_review", all_reasons=(), weight=0.0)


def choose_primary_bad_reason(reasons: list[str]) -> str:
    priority = [
        "bad_fragmented",
        "bad_many_halogen",
        "bad_overhalogenated_small_aromatic",
        "bad_fused_heterocycle",
        "bad_multi_aromatic_heterocycle",
        "bad_large_ring",
        "bad_saturated_polycycle_no_aromatic",
        "bad_poly_fused_or_bridge_ring",
        "bad_three_member_ring",
        "bad_high_carbon_low_hetero",
    ]
    for reason in priority:
        if reason in reasons:
            return reason
    return reasons[0]


def bad_weight(reasons: list[str]) -> float:
    if "bad_many_halogen" in reasons or "bad_overhalogenated_small_aromatic" in reasons:
        return 4.0
    if (
        "bad_fused_heterocycle" in reasons
        or "bad_saturated_polycycle_no_aromatic" in reasons
        or "bad_poly_fused_or_bridge_ring" in reasons
        or "bad_large_ring" in reasons
    ):
        return 3.0
    if "bad_multi_aromatic_heterocycle" in reasons:
        return 2.5
    if "bad_three_member_ring" in reasons:
        return 2.0
    return 1.5


def read_old_list(path: Path, default_source: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    if "smiles" not in df.columns:
        raise ValueError(f"Missing smiles column: {path}")
    if "source" not in df.columns:
        df["source"] = default_source
    if "all_labels" not in df.columns:
        df["all_labels"] = df.get("label", default_source)
    if "weight" not in df.columns:
        df["weight"] = 1.0
    df["canonical_smiles"] = df["smiles"].map(canonical_from_smiles)
    df = df[df["canonical_smiles"].notna()].copy()
    return df


def canonical_from_smiles(smiles: str | None) -> str | None:
    if smiles is None or pd.isna(smiles):
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=False)


def write_sdf(path: Path, rows: list[dict[str, object]]) -> None:
    writer = Chem.SDWriter(str(path))
    try:
        for row in rows:
            mol = row.get("_mol")
            if not isinstance(mol, Chem.Mol):
                continue
            mol = Chem.Mol(mol)
            mol.SetProp("_Name", str(row["global_id"]))
            for key, value in row.items():
                if key == "_mol":
                    continue
                mol.SetProp(str(key), "" if value is None else str(value))
            writer.write(mol)
    finally:
        writer.close()


def collect_new_rows() -> tuple[list[dict[str, object]], Counter]:
    rows: list[dict[str, object]] = []
    source_counts: Counter = Counter()
    seen: set[str] = set()
    sources = expanded_sources()
    for source, path in sources:
        molecules = load_sdf_molecules(source, path)
        source_counts[source] = len(molecules)
        for local_idx, (mol_id, mol) in enumerate(molecules):
            smiles = canonical_smiles(mol)
            if not smiles or smiles in seen:
                continue
            seen.add(smiles)
            facts = compute_facts(mol)
            decision = label_molecule(facts)
            row = {
                "global_id": f"{source}_{local_idx:06d}",
                "source": source,
                "source_path": str(path.relative_to(ROOT)) if path.exists() else str(path),
                "source_mol_id": mol_id,
                "smiles": smiles,
                "label": decision.label,
                "primary_reason": decision.primary_reason,
                "all_reasons": "|".join(decision.all_reasons),
                "weight": decision.weight,
                "_mol": mol,
            }
            row.update(asdict(facts))
            rows.append(row)
    return rows, source_counts


def expanded_sources() -> list[tuple[str, Path]]:
    """Return static SDF sources plus every epoch molecules.sdf in target runs."""
    sources: list[tuple[str, Path]] = []
    for run_name, run_dir in RUN_SOURCES:
        epoch_dir = run_dir / "epochs"
        if not epoch_dir.exists():
            continue
        for sdf_path in sorted(epoch_dir.glob("epoch_*/molecules.sdf")):
            source_name = f"{run_name}_{sdf_path.parent.name}"
            sources.append((source_name, sdf_path))
    sources.extend(STATIC_SOURCES)
    return sources


def write_sample_lists(new_rows: list[dict[str, object]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    old_pos = read_old_list(OLD_POS, "old_v0_2_1_positive")
    old_neg = read_old_list(OLD_NEG, "old_v0_2_1_negative")

    new_df = pd.DataFrame([{k: v for k, v in row.items() if k != "_mol"} for row in new_rows])
    new_pos = new_df[new_df["label"].eq("good")].copy()
    new_neg = new_df[new_df["label"].eq("bad")].copy()

    new_pos_list = pd.DataFrame(
        {
            "smiles": new_pos["smiles"],
            "label": 1,
            "source": new_pos["source"],
            "all_labels": new_pos["all_reasons"].where(new_pos["all_reasons"].ne(""), new_pos["primary_reason"]),
            "weight": new_pos["weight"],
            "canonical_smiles": new_pos["smiles"],
        }
    )
    new_neg_list = pd.DataFrame(
        {
            "smiles": new_neg["smiles"],
            "label": new_neg["primary_reason"],
            "source": new_neg["source"],
            "all_labels": new_neg["all_reasons"].where(new_neg["all_reasons"].ne(""), new_neg["primary_reason"]),
            "weight": new_neg["weight"],
            "canonical_smiles": new_neg["smiles"],
        }
    )

    pos = pd.concat(
        [old_pos[["smiles", "label", "source", "all_labels", "weight", "canonical_smiles"]], new_pos_list],
        ignore_index=True,
    )
    pos = pos.drop_duplicates("canonical_smiles", keep="first")

    neg = pd.concat(
        [old_neg[["smiles", "label", "source", "all_labels", "weight", "canonical_smiles"]], new_neg_list],
        ignore_index=True,
    )
    neg = neg.drop_duplicates("canonical_smiles", keep="first")

    overlap = neg["canonical_smiles"].isin(set(pos["canonical_smiles"]))
    neg = neg[~overlap].copy()

    pos[["smiles", "label", "source", "all_labels", "weight"]].to_csv(
        OUT_DIR / "selected_positives_v0_3_scaffold.smi",
        sep="\t",
        index=False,
    )
    neg[["smiles", "label", "source", "all_labels", "weight"]].to_csv(
        OUT_DIR / "selected_negatives_v0_3_scaffold_labeled.smi",
        sep="\t",
        index=False,
    )
    return pos, neg


def write_reason_sdfs(rows: list[dict[str, object]]) -> None:
    sdf_dir = OUT_DIR / "sdf_by_reason"
    sdf_dir.mkdir(parents=True, exist_ok=True)
    reason_rows: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        label = str(row["label"])
        primary = str(row["primary_reason"])
        reason_rows[f"{label}__{primary}"].append(row)
        if label == "bad":
            for reason in str(row.get("all_reasons", "")).split("|"):
                if reason:
                    reason_rows[f"all_bad_reason__{reason}"].append(row)

    for reason, reason_bucket in sorted(reason_rows.items()):
        write_sdf(sdf_dir / f"{reason}.sdf", reason_bucket)

    top_review = [
        row
        for row in rows
        if row["primary_reason"] in {
            "bad_saturated_polycycle_no_aromatic",
            "bad_overhalogenated_small_aromatic",
            "bad_many_halogen",
            "bad_fused_heterocycle",
        }
    ]
    write_sdf(OUT_DIR / "manual_review_priority_negative_examples.sdf", top_review[:1000])


def write_plots(classification: pd.DataFrame) -> None:
    counts = classification["primary_reason"].value_counts().sort_values(ascending=True)
    plt.figure(figsize=(9, max(4, 0.34 * len(counts))))
    counts.plot(kind="barh", color="#4c78a8")
    plt.xlabel("molecules")
    plt.ylabel("primary reason")
    plt.title("v0.3 scaffold label primary reasons")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "primary_reason_counts.png", dpi=180)
    plt.close()

    source_label = classification.groupby(["source", "label"]).size().unstack(fill_value=0)
    source_label = source_label.sort_index()
    plt.figure(figsize=(11, max(4, 0.34 * len(source_label))))
    source_label.plot(kind="barh", stacked=True, ax=plt.gca(), color=["#d62728", "#ffbf00", "#2ca02c"])
    plt.xlabel("unique molecules")
    plt.ylabel("source")
    plt.title("v0.3 scaffold labels by source")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "source_label_counts.png", dpi=180)
    plt.close()


def markdown_count_table(series: pd.Series, index_name: str, value_name: str = "n") -> str:
    """Format a small count table without requiring optional tabulate."""
    rows = [(str(index), int(value)) for index, value in series.items()]
    lines = [f"| {index_name} | {value_name} |", "|---|---:|"]
    lines.extend(f"| {index} | {value} |" for index, value in rows)
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, source_counts = collect_new_rows()
    classification = pd.DataFrame([{k: v for k, v in row.items() if k != "_mol"} for row in rows])
    classification.to_csv(OUT_DIR / "new_molecule_classification.csv", index=False)

    write_reason_sdfs(rows)
    pos, neg = write_sample_lists(rows)

    label_counts = classification["label"].value_counts().to_dict()
    reason_counts = classification["primary_reason"].value_counts().to_dict()
    source_label_counts = (
        classification.groupby(["source", "label"]).size().rename("n").reset_index()
    )
    source_label_counts.to_csv(OUT_DIR / "source_label_counts.csv", index=False)
    classification.groupby(["label", "primary_reason"]).size().rename("n").reset_index().to_csv(
        OUT_DIR / "label_reason_counts.csv",
        index=False,
    )
    write_plots(classification)

    summary = {
        "new_source_input_counts": dict(source_counts),
        "new_unique_molecules": int(len(classification)),
        "new_label_counts": label_counts,
        "new_primary_reason_counts": reason_counts,
        "v0_3_positive_rows": int(len(pos)),
        "v0_3_negative_rows": int(len(neg)),
        "outputs": {
            "classification_csv": "new_molecule_classification.csv",
            "positive_smi": "selected_positives_v0_3_scaffold.smi",
            "negative_smi": "selected_negatives_v0_3_scaffold_labeled.smi",
            "reason_sdf_dir": "sdf_by_reason/",
            "priority_review_sdf": "manual_review_priority_negative_examples.sdf",
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    report = [
        "# Classifier v0.3 scaffold curation",
        "",
        "## 输入",
        "",
        f"- 旧正样本：`{OLD_POS.relative_to(ROOT)}`",
        f"- 旧负样本：`{OLD_NEG.relative_to(ROOT)}`",
        f"- 新分子来源数：{len(expanded_sources())}",
        "",
        "## 新分子标签统计",
        "",
        markdown_count_table(classification["label"].value_counts(), "label"),
        "",
        "## 主要原因统计",
        "",
        markdown_count_table(classification["primary_reason"].value_counts(), "primary_reason"),
        "",
        "## 输出",
        "",
        "- `new_molecule_classification.csv`：所有新分子规则特征和标签",
        "- `sdf_by_reason/`：按标签原因拆分的 SDF，供人工复查",
        "- `manual_review_priority_negative_examples.sdf`：优先人工看的 hard negative",
        "- `selected_positives_v0_3_scaffold.smi` / `selected_negatives_v0_3_scaffold_labeled.smi`：可用于下一步训练",
        "",
        "![primary reason counts](primary_reason_counts.png)",
        "",
        "![source label counts](source_label_counts.png)",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
