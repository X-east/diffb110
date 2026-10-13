#!/usr/bin/env python3
"""Analyze IQL generation batches on the frozen portable UMAP reference."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import pickle
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, QED, rdMolDescriptors
from scipy import sparse
from scipy.stats import wasserstein_distance

RDLogger.DisableLog("rdApp.*")
LOGGER = logging.getLogger(__name__)


def bitvector_to_row(fp: object, n_bits: int = 2048) -> sparse.csr_matrix:
    """Convert one RDKit bit vector to a sparse CSR row."""
    bits = np.fromiter(fp.GetOnBits(), dtype=np.int32, count=fp.GetNumOnBits())
    return sparse.csr_matrix((np.ones(len(bits), dtype=np.float32), (np.zeros(len(bits), dtype=np.int32), bits)), shape=(1, n_bits))


def fingerprints_to_csr(fps: list[object], n_bits: int = 2048) -> sparse.csr_matrix:
    """Convert fingerprints to a sparse binary CSR matrix."""
    rows = [bitvector_to_row(fp, n_bits) for fp in fps]
    return sparse.vstack(rows, format="csr") if rows else sparse.csr_matrix((0, n_bits), dtype=np.float32)


def molecule_facts(mol: Chem.Mol) -> dict[str, float | int]:
    """Extract reproducible RDKit descriptors from one valid molecule."""
    return {
        "mw": float(Descriptors.MolWt(mol)),
        "logp": float(Crippen.MolLogP(mol)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "qed": float(QED.qed(mol)),
        "hba": int(Lipinski.NumHAcceptors(mol)),
        "hbd": int(Lipinski.NumHDonors(mol)),
        "heavy_atoms": int(mol.GetNumHeavyAtoms()),
        "rings": int(rdMolDescriptors.CalcNumRings(mol)),
        "aromatic_rings": int(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "three_member_rings": int(sum(1 for ring in mol.GetRingInfo().AtomRings() if len(ring) == 3)),
        "fraction_c": float(rdMolDescriptors.CalcFractionCSP3(mol)),
        "formal_charge": int(sum(atom.GetFormalCharge() for atom in mol.GetAtoms())),
        "fragments": int(len(Chem.GetMolFrags(mol))),
    }


def load_chembl(path: Path) -> tuple[pd.DataFrame, sparse.csr_matrix]:
    """Load the frozen 100k ChEMBL ECFP4 cache."""
    with gzip.open(path, "rb") as handle:
        payload = pickle.load(handle)
    records = pd.DataFrame(payload["records"])
    matrix = fingerprints_to_csr(payload["fingerprints"])
    if len(records) != 100_000 or matrix.shape != (100_000, 2048):
        raise ValueError(f"unexpected ChEMBL cache shape: records={len(records)} matrix={matrix.shape}")
    return records, matrix


def load_batch(path: Path) -> tuple[pd.DataFrame, sparse.csr_matrix]:
    """Read one SDF, retaining attempt index and valid molecule facts."""
    rows: list[dict[str, object]] = []
    fps: list[object] = []
    supplier = Chem.SDMolSupplier(str(path), sanitize=True, removeHs=False)
    for attempt, mol in enumerate(supplier):
        row: dict[str, object] = {"attempt": attempt, "sdf_record": attempt, "valid": int(mol is not None)}
        if mol is None:
            rows.append(row)
            continue
        facts = molecule_facts(mol)
        smiles = Chem.MolToSmiles(mol, canonical=True)
        row.update(facts)
        row["canonical_smiles"] = smiles
        row["inchikey"] = ""
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))
        rows.append(row)
    valid_rows = [row for row in rows if row["valid"] == 1]
    return pd.DataFrame(rows), fingerprints_to_csr(fps)


def similarity_stats(query: sparse.csr_matrix, reference: sparse.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest and twentieth-neighbor ECFP4 Tanimoto similarities."""
    ref_counts = np.asarray(reference.sum(axis=1)).ravel().astype(np.float32)
    query_counts = np.asarray(query.sum(axis=1)).ravel().astype(np.float32)
    nearest = np.empty(query.shape[0], dtype=np.float32)
    twentieth = np.empty(query.shape[0], dtype=np.float32)
    for start in range(0, query.shape[0], 64):
        stop = min(start + 64, query.shape[0])
        intersections = (query[start:stop] @ reference.T).toarray().astype(np.float32)
        unions = query_counts[start:stop, None] + ref_counts[None, :] - intersections
        sims = np.divide(intersections, unions, out=np.zeros_like(intersections), where=unions > 0)
        nearest[start:stop] = np.max(sims, axis=1)
        twentieth[start:stop] = np.partition(sims, -20, axis=1)[:, -20]
    return nearest, twentieth


def main() -> int:
    """Run the frozen-reference batch analysis."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--chembl", type=Path, required=True)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--batch",
        action="append",
        default=None,
        metavar="LABEL=SDF_PATH",
        help=(
            "Explicit batch to analyze; may be passed repeatedly. When omitted, "
            "uses the historical baseline/epoch paths below --batch-root."
        ),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    started = time.time()
    reducer = joblib.load(args.model)
    chembl_records, chembl_matrix = load_chembl(args.chembl)
    if args.batch is None:
        batch_paths = [
            ("baseline_original", args.batch_root / "baseline_original/generated_1000.sdf")
        ]
        batch_paths += [
            (f"epoch_{epoch:04d}", args.batch_root / f"epoch_{epoch:04d}/generated_1000.sdf")
            for epoch in range(9, 100, 10)
        ]
    else:
        batch_paths = []
        seen_labels: set[str] = set()
        for value in args.batch:
            label, separator, raw_path = value.partition("=")
            if not separator or not label or not raw_path:
                raise ValueError("--batch must use LABEL=SDF_PATH")
            if label in seen_labels:
                raise ValueError(f"duplicate --batch label: {label}")
            sdf_path = Path(raw_path).expanduser().resolve()
            if not sdf_path.is_file():
                raise FileNotFoundError(sdf_path)
            seen_labels.add(label)
            batch_paths.append((label, sdf_path))
        if "baseline_original" not in seen_labels:
            raise ValueError("explicit --batch inputs must include baseline_original")
    all_rows: list[pd.DataFrame] = []
    summary: list[dict[str, object]] = []
    umap_reference = np.asarray(reducer.embedding_, dtype=np.float32)
    for batch, sdf_path in batch_paths:
        LOGGER.info("processing %s", batch)
        rows, query = load_batch(sdf_path)
        valid_mask = rows["valid"].eq(1).to_numpy()
        valid_rows = rows.loc[valid_mask].copy().reset_index(drop=True)
        if len(valid_rows):
            nearest, twentieth = similarity_stats(query, chembl_matrix)
            projected = reducer.transform(query)
            valid_rows["nearest_chembl_tanimoto"] = nearest
            valid_rows["chembl_tanimoto_20th"] = twentieth
            valid_rows["umap1"] = projected[:, 0]
            valid_rows["umap2"] = projected[:, 1]
            valid_rows["batch"] = batch
            valid_rows["nearest_umap_reference_distance"] = np.sqrt(((projected[:, None, :] - umap_reference[:100_000][None, :, :]) ** 2).sum(axis=2)).min(axis=1)
            all_rows.append(valid_rows)
            values = {name: valid_rows[name].to_numpy(dtype=float) for name in ["mw", "logp", "tpsa", "qed", "hba", "hbd", "heavy_atoms", "rings", "aromatic_rings", "three_member_rings", "fraction_c", "nearest_chembl_tanimoto", "chembl_tanimoto_20th", "nearest_umap_reference_distance"]}
            stats = {f"{name}_mean": float(np.mean(value)) for name, value in values.items()}
            stats.update({f"{name}_median": float(np.median(value)) for name, value in values.items()})
            stats["umap_centroid_norm"] = float(np.linalg.norm(projected.mean(axis=0)))
            stats["umap1_std"] = float(np.std(projected[:, 0]))
            stats["umap2_std"] = float(np.std(projected[:, 1]))
        else:
            stats = {}
        stats.update({"batch": batch, "sdf_records": int(len(rows)), "valid": int(valid_mask.sum()), "valid_rate": float(valid_mask.mean()), "unique_valid": int(valid_rows["canonical_smiles"].nunique()) if len(valid_rows) else 0})
        summary.append(stats)
    molecules = pd.concat(all_rows, ignore_index=True)
    baseline = molecules[molecules["batch"].eq("baseline_original")]
    descriptor_names = ["mw", "logp", "tpsa", "qed", "hba", "hbd", "heavy_atoms", "rings", "aromatic_rings", "three_member_rings", "fraction_c", "nearest_chembl_tanimoto", "chembl_tanimoto_20th", "nearest_umap_reference_distance"]
    summary_frame = pd.DataFrame(summary)
    for _, row in summary_frame.iterrows():
        batch = row["batch"]
        current = molecules[molecules["batch"].eq(batch)]
        for name in descriptor_names:
            summary_frame.loc[summary_frame["batch"].eq(batch), f"{name}_wasserstein_vs_baseline"] = wasserstein_distance(baseline[name], current[name])
    molecules.to_csv(args.output / "per_molecule_portable_umap_metrics.csv.gz", index=False, compression="gzip")
    summary_frame.to_csv(args.output / "batch_summary.csv", index=False)
    np.save(args.output / "chembl_reference_umap_coordinates.npy", umap_reference[:100_000])
    for name in descriptor_names:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for batch in summary_frame["batch"]:
            values = molecules.loc[molecules["batch"].eq(batch), name]
            ax.hist(values, bins=35, alpha=0.25, density=True, label=batch)
        ax.set_title(f"{name} across IQL generation batches")
        ax.set_xlabel(name)
        ax.set_ylabel("density")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(args.output / f"distribution_{name}.png", dpi=180)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(umap_reference[:100_000, 0], umap_reference[:100_000, 1], s=1, alpha=0.08, c="lightgray", label="ChEMBL reference")
    for batch in summary_frame["batch"]:
        current = molecules[molecules["batch"].eq(batch)]
        ax.scatter(current["umap1"], current["umap2"], s=8, alpha=0.45, label=batch)
    ax.set_title("IQL generation batches on frozen 100k ChEMBL UMAP")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(args.output / "all_batches_portable_exact_umap.png", dpi=220)
    plt.close(fig)
    manifest = {"status": "completed", "model": str(args.model), "chembl": str(args.chembl), "batches": [str(path) for _, path in batch_paths], "elapsed_seconds": time.time() - started, "reference_rows": int(umap_reference.shape[0]), "valid_rows": int(len(molecules))}
    (args.output / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("completed valid=%d elapsed=%.1fs", len(molecules), time.time() - started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
