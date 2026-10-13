"""Rank fixed-checkpoint molecules by transparent, typed 3D feature matching.

The isolated analysis borrows RDKit feature definitions and rigid alignment,
and Pharmit's typed-point matching concept. It does not reproduce unpublished
LS-MolGen query parameters or infer experimentally validated pharmacophores.
"""
from __future__ import annotations

import itertools
import json
import logging
import math
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDConfig, RDLogger, rdBase
from rdkit.Chem import ChemicalFeatures, Crippen, Descriptors, FilterCatalog, QED, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Numerics import rdAlignment
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(RDConfig.RDContribDir) / "SA_Score"))
from ddpo.utils.ligand_prep import extract_largest_fragment
from scripts.offline_rl.structure_pathology_rules import compute_structure_pathology
import sascorer

LOGGER = logging.getLogger("epi-pharmacophore")
CONFIG = json.loads((OUT / "screen_config.json").read_text())
FACTORY = ChemicalFeatures.BuildFeatureFactory(str(Path(RDConfig.RDDataDir) / "BaseFeatures.fdef"))
FEATURE_FAMILIES = set(CONFIG["families"])
CATALOG_PARAMS = FilterCatalog.FilterCatalogParams()
CATALOG_PARAMS.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
CATALOG_PARAMS.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
CATALOG = FilterCatalog.FilterCatalog(CATALOG_PARAMS)
REFERENCES: dict = {}


def features(mol: Chem.Mol) -> list[dict]:
    """Return typed reference points; atom-level Hydrophobe is not double counted."""
    result = []
    for feat in FACTORY.GetFeaturesForMol(mol):
        if feat.GetFamily() in FEATURE_FAMILIES:
            result.append({"family": feat.GetFamily(), "atoms": list(feat.GetAtomIds()), "xyz": list(feat.GetPos())})
    return result


def anchors(feats: list[dict]) -> list[tuple[int, ...]]:
    """Select deterministic noncollinear reference triples before seeing probes."""
    xyz = np.asarray([x["xyz"] for x in feats])
    triples = []
    for triple in itertools.combinations(range(len(feats)), 3):
        a, b, c = xyz[list(triple)]
        area = float(np.linalg.norm(np.cross(b - a, c - a)))
        diversity = len({feats[i]["family"] for i in triple})
        if area > 0.25 and diversity >= 2:
            triples.append((triple, area, diversity))
    chosen, covered = [], set()
    while triples and len(chosen) < CONFIG["reference_anchor_limit"]:
        item = max(triples, key=lambda x: (len(set(x[0]) - covered), x[2], x[1], tuple(-i for i in x[0])))
        triples.remove(item)
        chosen.append(item[0])
        covered.update(item[0])
    return chosen


def assignment(ref: list[dict], probe: list[dict], xyz: np.ndarray, radius: float) -> dict:
    """Maximize typed, one-to-one Gaussian overlaps with optional unmatched points."""
    if not ref or not probe:
        return {"similarity": 0.0, "coverage": 0.0, "matched": 0, "rmsd": math.nan, "pairs": []}
    target = np.asarray([x["xyz"] for x in ref])
    distances = np.linalg.norm(target[:, None, :] - xyz[None, :, :], axis=2)
    same = np.asarray([[a["family"] == b["family"] for b in probe] for a in ref])
    weights = np.exp(-distances**2 / (2 * CONFIG["gaussian_sigma_angstrom"]**2)) * same * (distances <= radius)
    matrix = np.concatenate([weights, np.zeros((len(ref), len(ref)))], axis=1)
    rows, cols = linear_sum_assignment(-matrix)
    pairs = [(int(i), int(j), float(distances[i, j])) for i, j in zip(rows, cols) if j < len(probe) and weights[i, j] > 0]
    overlap = sum(float(weights[i, j]) for i, j, _ in pairs)
    return {"similarity": 2 * overlap / (len(ref) + len(probe)), "coverage": len(pairs) / len(ref), "matched": len(pairs),
            "rmsd": float(np.sqrt(np.mean([d*d for _, _, d in pairs]))) if pairs else math.nan, "pairs": pairs}


def align_features(ref: list[dict], probe: list[dict], ref_anchors: list[tuple[int, ...]]) -> tuple[dict, np.ndarray]:
    """Search typed triple seeds and refine a proper rigid transformation."""
    identity = np.eye(4)
    if not ref or not probe:
        return assignment(ref, probe, np.empty((0, 3)), CONFIG["radius_angstrom"]), identity
    ref_xyz = np.asarray([x["xyz"] for x in ref], dtype=float)
    probe_xyz = np.asarray([x["xyz"] for x in probe], dtype=float)
    type_groups = {family: [i for i, x in enumerate(probe) if x["family"] == family] for family in FEATURE_FAMILIES}
    seeds = []
    radius = CONFIG["radius_angstrom"]
    for ri in ref_anchors:
        groups = [type_groups[ref[i]["family"]] for i in ri]
        if any(not group for group in groups):
            continue
        pi = np.asarray([p for p in itertools.product(*groups) if len(set(p)) == 3], dtype=int)
        if not len(pi):
            continue
        r = ref_xyz[list(ri)]
        p = probe_xyz[pi]
        edges = [(0, 1), (0, 2), (1, 2)]
        diffs = np.asarray([np.abs(np.linalg.norm(p[:, a] - p[:, b], axis=1) - np.linalg.norm(r[a] - r[b])) for a, b in edges]).T
        mask = np.max(diffs, axis=1) <= 2 * radius
        for index in np.flatnonzero(mask):
            seeds.append((float(np.sum(diffs[index]**2)), tuple(ri), tuple(int(x) for x in pi[index])))
    seeds.sort()
    seeds = seeds[:CONFIG["alignment_seed_limit"]]
    transforms = [identity.copy()]
    # Include translation seeds so small-feature probes still receive partial matches.
    for ri, r in enumerate(ref):
        for pi in type_groups[r["family"]]:
            matrix = identity.copy()
            matrix[:3, 3] = ref_xyz[ri] - probe_xyz[pi]
            transforms.append(matrix)
    for _, ri, pi in seeds:
        _, matrix = rdAlignment.GetAlignmentTransform(ref_xyz[list(ri)], probe_xyz[list(pi)], reflect=False)
        transforms.append(matrix)
    best = {"similarity": -1.0}
    best_matrix = identity
    for matrix in transforms:
        for iteration in range(CONFIG["alignment_refinements"] + 1):
            moved = probe_xyz @ matrix[:3, :3].T + matrix[:3, 3]
            result = assignment(ref, probe, moved, radius)
            if result["similarity"] > best["similarity"] + 1e-12:
                best, best_matrix = result, matrix.copy()
            pairs = result["pairs"]
            if len(pairs) < 3 or iteration == CONFIG["alignment_refinements"]:
                break
            _, next_matrix = rdAlignment.GetAlignmentTransform(ref_xyz[[i for i, _, _ in pairs]], probe_xyz[[j for _, j, _ in pairs]], reflect=False)
            if np.max(np.abs(next_matrix - matrix)) < 1e-7:
                break
            matrix = next_matrix
    return best, best_matrix


def init_worker(reference_data: dict) -> None:
    """Initialize immutable reference points in each CPU worker."""
    global REFERENCES
    REFERENCES = reference_data
    RDLogger.DisableLog("rdApp.*")


def evaluate(item: tuple[dict, bytes]) -> tuple[dict, list[dict], bytes]:
    """Calculate existing quality gates and all four independent reference scores."""
    row, binary = item
    mol = Chem.Mol(binary)
    mol = Chem.RemoveHs(mol)
    Chem.SanitizeMol(mol)
    feats = features(mol)
    facts = compute_structure_pathology(mol)
    row.update({"smiles": Chem.MolToSmiles(mol, isomericSmiles=True), "ha": mol.GetNumHeavyAtoms(), "mw": Descriptors.MolWt(mol),
                "logp": Crippen.MolLogP(mol), "qed": QED.qed(mol), "sa": sascorer.calculateScore(mol),
                "hba": rdMolDescriptors.CalcNumHBA(mol), "hbd": rdMolDescriptors.CalcNumHBD(mol),
                "rotatable": rdMolDescriptors.CalcNumRotatableBonds(mol), "rings": rdMolDescriptors.CalcNumRings(mol),
                "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol), "tpsa": rdMolDescriptors.CalcTPSA(mol),
                "carbon_fraction": facts.carbon_fraction, "clean": not facts.is_pathological,
                "feature_count": len(feats), "feature_families": json.dumps(Counter(x["family"] for x in feats), sort_keys=True),
                "scaffold": MurckoScaffold.MurckoScaffoldSmiles(mol=mol),
                "alerts": ";".join(sorted({entry.GetDescription() for entry in CATALOG.GetMatches(mol)}))})
    rules = CONFIG["candidate_screen"]
    row["physchem_pass"] = bool(row["mw"] < rules["mw_max_exclusive"] and row["logp"] <= rules["logp_max_inclusive"] and row["hbd"] <= rules["hbd_max_inclusive"] and row["hba"] <= rules["hba_max_inclusive"] and row["rotatable"] <= rules["rotatable_bonds_max_inclusive"])
    row["docking_valid"] = bool(math.isfinite(row["docking_score"]) and row["docking_score"] <= 0)
    row["eligible"] = bool(row["clean"] and row["physchem_pass"] and row["docking_valid"] and row["docking_score"] <= rules["docking_max_inclusive"])
    matches = []
    for reference_id, reference in REFERENCES.items():
        result, matrix = align_features(reference["features"], feats, reference["anchors"])
        match = {"route": row["route"], "record_id": row["record_id"], "reference": reference_id,
                 "feature_score": result["similarity"], "reference_coverage": result["coverage"], "matched_points": result["matched"],
                 "reference_points": len(reference["features"]), "candidate_points": len(feats), "feature_rmsd": result["rmsd"],
                 "matched_families": json.dumps(Counter(reference["features"][i]["family"] for i, _, _ in result["pairs"]), sort_keys=True),
                 "matched_pairs": json.dumps(result["pairs"]), "transform": json.dumps(matrix.tolist())}
        moved = np.asarray([x["xyz"] for x in feats]) @ matrix[:3, :3].T + matrix[:3, 3] if feats else np.empty((0, 3))
        for radius in CONFIG["sensitivity_radii_angstrom"]:
            match[f"score_radius_{radius:g}"] = assignment(reference["features"], feats, moved, radius)["similarity"]
        matches.append(match)
    return row, matches, mol.ToBinary()


def load_pool() -> list[tuple[dict, bytes]]:
    """Load independently docked molecules with checkpoint and source provenance."""
    sources = [
        ("DiffGui_RL", "epoch80_batch1", ROOT / "analysis/diffgui_epoch80_independent_eval_261010/docking/successful_docked_outputs.sdf"),
        ("DiffGui_RL", "epoch80_batch2", OUT / "docking_batch2/successful_docked_outputs.sdf"),
        ("ESRPO", "step4000", OUT / "docking_esrpo/successful_docked_outputs.sdf")]
    pool = []
    for route, batch, path in sources:
        supplier = Chem.SDMolSupplier(str(path), removeHs=False)
        count = 0
        for index, mol in enumerate(supplier):
            if mol is None:
                LOGGER.warning("Unreadable docked molecule %s record %s", path, index)
                continue
            fragment, nfrags, reason = extract_largest_fragment(mol)
            if fragment is None:
                LOGGER.warning("Fragment failed %s record %s: %s", path, index, reason)
                continue
            record_id = mol.GetProp("record_id") if mol.HasProp("record_id") else f"s4000_{mol.GetIntProp('benchmark_index'):06d}"
            if batch == "epoch80_batch1":
                record_id = "b1_" + record_id
            row = {"route": route, "batch": batch, "record_id": record_id,
                   "source_pose_sdf": str(path.relative_to(ROOT)), "source_pose_ordinal_0based": index,
                   "benchmark_index": mol.GetIntProp("benchmark_index"), "fragments_in_docking_pose": nfrags,
                   "original_source_sdf": mol.GetProp("source_sdf") if mol.HasProp("source_sdf") else "OUT/diffsbdd_e2po_mainline_intermediate_gen/step_004000/molecules_relaxed_largest.sdf",
                   "docking_score": float(mol.GetProp("DockingScore"))}
            pool.append((row, fragment.ToBinary()))
            count += 1
        LOGGER.info("Loaded %s %s: %d molecules", route, batch, count)
    return pool


def receptor_tree() -> cKDTree:
    """Load receptor heavy-atom coordinates for a simple severe-overlap check."""
    coords = []
    for line in (ROOT / "data/8gjl.pdbqt").read_text().splitlines():
        if line.startswith(("ATOM  ", "HETATM")) and line.split()[-1] not in {"H", "HD", "HS"}:
            coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return cKDTree(np.asarray(coords))


def export_top(frame: pd.DataFrame, molecules: dict, tree: cKDTree, tier: str) -> pd.DataFrame:
    """Export unique-structure Top N per route/reference and separate pose states."""
    top_rows = []
    for (route, reference), group in frame.groupby(["route", "reference"], sort=False):
        group = group.sort_values(["feature_score", "reference_coverage", "feature_rmsd", "docking_score", "record_id"], ascending=[False, False, True, True, True])
        if tier == "screened":
            group = group[group["eligible"]]
        group = group.drop_duplicates("smiles").head(CONFIG["top_n"]).copy()
        group["rank"] = np.arange(1, len(group) + 1)
        group["tier"] = tier
        stem = f"{route}_{reference}_{tier}_top10"
        with Chem.SDWriter(str(OUT / f"{stem}_docked.sdf")) as dw, Chem.SDWriter(str(OUT / f"{stem}_aligned.sdf")) as aw:
            for _, series in group.iterrows():
                row = series.to_dict()
                mol = Chem.Mol(molecules[(route, row["record_id"])])
                xyz = np.asarray(mol.GetConformer().GetPositions())
                matrix = np.asarray(json.loads(row["transform"]))
                aligned = Chem.Mol(mol)
                moved = xyz @ matrix[:3, :3].T + matrix[:3, 3]
                for atom_index, point in enumerate(moved):
                    aligned.GetConformer().SetAtomPosition(atom_index, point)
                for state, obj, coords in [("docked", mol, xyz), ("free_aligned", aligned, moved)]:
                    distances = tree.query(coords)[0]
                    row[f"{state}_min_receptor_distance"] = float(distances.min())
                    row[f"{state}_severe_overlap_atoms"] = int((distances < 1.5).sum())
                    obj.SetProp("_Name", f"{row['record_id']} | {reference} | rank {row['rank']} | {state}")
                    obj.SetProp("coordinate_state", state)
                    for key in ["record_id", "reference", "rank", "feature_score", "reference_coverage", "feature_rmsd", "docking_score", "smiles", "alerts", "matched_pairs", "source_pose_sdf", "source_pose_ordinal_0based"]:
                        obj.SetProp(key, str(row[key]))
                dw.write(mol)
                aw.write(aligned)
                top_rows.append(row)
        pd.DataFrame([r for r in top_rows if r["route"] == route and r["reference"] == reference]).to_csv(OUT / f"{stem}.csv", index=False)
    return pd.DataFrame(top_rows)


def verify_references(references: dict) -> dict:
    """Verify exact self-match and invariance under a known rigid movement."""
    theta = 0.73
    rotation = np.array([[math.cos(theta), -math.sin(theta), 0], [math.sin(theta), math.cos(theta), 0], [0, 0, 1]])
    results = {}
    for name, ref in references.items():
        moved = [{**x, "xyz": (np.asarray(x["xyz"]) @ rotation.T + [19, -27, 13]).tolist()} for x in ref["features"]]
        original, _ = align_features(ref["features"], ref["features"], ref["anchors"])
        transformed, _ = align_features(ref["features"], moved, ref["anchors"])
        results[name] = {"self": original["similarity"], "rigidly_moved": transformed["similarity"], "passed": original["similarity"] > 0.999999 and transformed["similarity"] > 0.999999}
    if not all(x["passed"] for x in results.values()):
        raise RuntimeError(f"Alignment control failed: {results}")
    return results


def main() -> None:
    """Screen completed independent libraries and export reproducible rankings."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    RDLogger.DisableLog("rdApp.*")
    LOGGER.info("Environment=dif RDKit=%s workers=%d config=%s", rdBase.rdkitVersion, CONFIG["workers"], OUT / "screen_config.json")
    reference_data = {}
    for mol in Chem.SDMolSupplier(str(OUT / "docking_references/successful_docked_outputs.sdf"), removeHs=False):
        name = mol.GetProp("reference_id")
        feats = features(Chem.RemoveHs(mol))
        reference_data[name] = {"features": feats, "anchors": anchors(feats), "reference_docking_score": float(mol.GetProp("DockingScore"))}
    (OUT / "reference_queries.json").write_text(json.dumps(reference_data, indent=2))
    controls = verify_references(reference_data)
    pool = load_pool()
    rows, matches, molecules = [], [], {}
    with ProcessPoolExecutor(max_workers=CONFIG["workers"], initializer=init_worker, initargs=(reference_data,)) as executor:
        for index, (row, pair_matches, binary) in enumerate(executor.map(evaluate, pool, chunksize=8), 1):
            rows.append(row)
            matches.extend(pair_matches)
            molecules[(row["route"], row["record_id"])] = binary
            if index % 100 == 0:
                LOGGER.info("Evaluated %d/%d molecules (%d reference comparisons)", index, len(pool), len(matches))
    metrics = pd.DataFrame(rows)
    frame = pd.DataFrame(matches).merge(metrics, on=["route", "record_id"], validate="many_to_one")
    metrics.to_csv(OUT / "molecule_metrics.csv", index=False)
    frame.to_csv(OUT / "all_reference_matches.csv", index=False)
    tree = receptor_tree()
    unfiltered = export_top(frame, molecules, tree, "similarity_only")
    screened = export_top(frame, molecules, tree, "screened")
    unfiltered.to_csv(OUT / "top10_similarity_only.csv", index=False)
    screened.to_csv(OUT / "top10_screened.csv", index=False)
    summary = {"controls": controls, "counts": [], "reference_queries": reference_data,
               "screened_top_records": len(screened), "screened_unique_structures": int(screened["smiles"].nunique()),
               "ranking": "Descending normalized Gaussian one-to-one feature overlap; coverage, RMSD, docking and ID break ties"}
    for route, group in metrics.groupby("route"):
        summary["counts"].append({"route": route, "docked": len(group), "physchem_pass": int(group["physchem_pass"].sum()),
                                  "r1r5_clean": int(group["clean"].sum()), "eligible": int(group["eligible"].sum()),
                                  "eligible_unique": int(group[group["eligible"]]["smiles"].nunique())})
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    LOGGER.info("COMPLETE: %d matches, %d screened Top records, %d unique structures", len(frame), len(screened), summary["screened_unique_structures"])


if __name__ == "__main__":
    main()
