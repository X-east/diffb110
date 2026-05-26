#!/usr/bin/env python3
"""Extract negative samples from OUT training outputs.

Two-tier extraction:
  Tier 1: Pain-point targeted molecules (~4K) — specific crash patterns
  Tier 2: Property + structural-alert sweep — drug-like-but-bad molecules
          Replaces the old broad fingerprint-threshold sweep.
          Categories:
            - prop_all_carbon       : carbon_fraction >= 0.95 (no heteroatoms)
            - prop_too_hydrophilic  : logP < -2 AND TPSA > 100
            - prop_too_complex      : MW > 600 OR RotBond > 12
            - prop_sulfur_excess    : S_count >= 3
            - alert_pains_brenk     : (disabled — kept for backward compat)

Output:
  - all_negatives_labeled.smi    : all negatives with pain label
  - all_negatives.smi            : SMILES only (for training)
  - prop_negatives.smi           : property/alert sweep only
  - pain1_sulfur.smi ...         : per-pain-point subsets
"""

import argparse
import csv
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Descriptors


# ── Helpers ──────────────────────────────────────────────────────────


def canonical_smiles(smi: str) -> str | None:
    """Return canonical SMILES, or None if RDKit fails."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def _float(row: dict, key: str, default: float = 0.0) -> float:
    """Safely parse float from CSV row."""
    try:
        return float(row.get(key, default))
    except (ValueError, TypeError):
        return default


# ── Property computation on the fly ─────────────────────────────────


def _compute_mol_properties(mol: Chem.Mol) -> dict:
    """Compute molecular properties from an RDKit Mol.

    Returns dict with keys: logp, tpsa, mw, rotb, carbon_fraction,
    s_count, num_heavy_atoms.
    """
    ha = mol.GetNumHeavyAtoms()
    carbon_count = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6)
    sulfur_count = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 16)

    return {
        "logp": Descriptors.MolLogP(mol),
        "tpsa": Descriptors.TPSA(mol),
        "mw": Descriptors.MolWt(mol),
        "rotb": Descriptors.NumRotatableBonds(mol),
        "carbon_fraction": carbon_count / ha if ha > 0 else 0.0,
        "s_count": sulfur_count,
        "num_heavy_atoms": ha,
    }


def _classify_molecule(props: dict) -> str | None:
    """Classify a molecule into a negative category, or None if it passes all.

    Checks in priority order (first match wins):
      - prop_all_carbon
      - prop_too_hydrophilic
      - prop_too_complex
      - prop_sulfur_excess
    """
    # 1. All-carbon collapse: virtually no heteroatoms
    if props["carbon_fraction"] >= 0.95:
        return "prop_all_carbon"

    # 2. Too hydrophilic: membrane penetration poor
    if props["logp"] < -2.0 and props["tpsa"] > 100.0:
        return "prop_too_hydrophilic"

    # 3. Too complex: MW exceeds typical oral range, or too flexible
    if props["mw"] > 600.0 or props["rotb"] > 12:
        return "prop_too_complex"

    # 4. Sulfur excess: >= 3 S atoms (drugs typically ≤ 2)
    if props["s_count"] >= 3:
        return "prop_sulfur_excess"

    return None


def iter_csv_files(
    root: Path,
    experiment: str,
    epoch_range: tuple[int, int] | None = None,
) -> list[Path]:
    """Find molecules_rank*.csv files under an experiment's epochs dir.

    Args:
        epoch_range: If set, only include epochs in [min, max] inclusive.
    """
    epochs_dir = root / experiment / "epochs"
    if not epochs_dir.exists():
        return []
    files = sorted(epochs_dir.glob("epoch_*/molecules*.csv"))
    if epoch_range is None:
        return files
    lo, hi = epoch_range
    return [
        f
        for f in files
        if lo <= int(f.parent.name.split("_")[1]) <= hi
    ]


# ── Tier 1: Pain-point targeted extraction ──────────────────────────


def _read_valid_smiles_and_cols(
    files: list[Path],
    required_cols: list[str],
) -> list[tuple[str, dict]]:
    """Read all rdkit_valid=1 rows from a list of CSV files.

    Returns list of (smiles, row_dict) for valid molecules.
    Skips rows where any required_col is missing.
    """
    results: list[tuple[str, dict]] = []
    for f in files:
        with open(f) as fh:
            reader = csv.DictReader(fh)
            fieldnames = set(reader.fieldnames or [])
            for row in reader:
                smi = row.get("smiles", "").strip()
                if row.get("rdkit_valid", "0").strip() != "1":
                    continue
                if not smi or smi == "None":
                    continue
                # Check required cols exist
                missing = [
                    c for c in required_cols
                    if c not in fieldnames or row.get(c, "").strip() == ""
                ]
                if missing:
                    continue
                results.append((smi, row))
    return results


def extract_pain1_sulfur(root: Path) -> list[tuple[str, str]]:
    """Pain 1: Sulfur invasion — [SH]/S infiltrating organic scaffolds.

    Source: crash experiment epoch 24-28, the turning point.
    Rule: rdkit_valid=1 AND (has [SH] or ≥2 S) AND ecfp4_sim < 0.25
    """
    exp = "7gpu-dock-ecfp4-fcfp4_20260521_212927"
    files = iter_csv_files(root, exp, epoch_range=(24, 28))
    rows = _read_valid_smiles_and_cols(
        files, ["chembl_prior_ecfp4_similarity"]
    )

    results = []
    for smi, row in rows:
        ecfp4 = _float(row, "chembl_prior_ecfp4_similarity", 2.0)
        if ecfp4 >= 0.25:
            continue
        # Count S atoms (including [SH])
        s_count = smi.count("S")
        has_sh = "[SH]" in smi
        if not has_sh and s_count < 2:
            continue
        can = canonical_smiles(smi)
        if can:
            results.append((can, "pain1_sulfur"))
    return results


def extract_pain2_pseudo_normal(root: Path) -> list[tuple[str, str]]:
    """Pain 2: Pseudo-normal PAH variants.

    Late-stage convergence: good docking but borderline chembl similarity.
    Source: main stable experiment epoch 50-62.
    Rule: rdkit_valid=1 AND 0.18 <= ecfp4 <= 0.30 AND docking < -8
    """
    exp = "7gpu-dock-ecfp4-fcfp4_20260525_212140"
    files = iter_csv_files(root, exp, epoch_range=(50, 62))
    rows = _read_valid_smiles_and_cols(
        files, ["chembl_prior_ecfp4_similarity", "docking_raw"]
    )

    results = []
    for smi, row in rows:
        ecfp4 = _float(row, "chembl_prior_ecfp4_similarity", 2.0)
        docking = _float(row, "docking_raw", 999.0)
        if not (0.18 <= ecfp4 <= 0.30):
            continue
        if docking >= -8:
            continue
        can = canonical_smiles(smi)
        if can:
            results.append((can, "pain2_pseudo_normal"))
    return results


def extract_pain3_dock_collapse(root: Path) -> list[tuple[str, str]]:
    """Pain 3: Docking backend collapse molecules.

    Source: dock-crashed experiment epoch 5-9.
    Rule: rdkit_valid=1 AND (docking_raw > -4 OR docking_reward < -0.5)
    """
    exp = "6gpu-dock-chembl_20260519_180540"
    files = iter_csv_files(root, exp, epoch_range=(5, 9))
    rows = _read_valid_smiles_and_cols(
        files, ["docking_raw", "docking_reward"]
    )

    results = []
    for smi, row in rows:
        docking_raw = _float(row, "docking_raw", 999.0)
        docking_reward = _float(row, "docking_reward", 1.0)
        if not (docking_raw > -4 or docking_reward < -0.5):
            continue
        can = canonical_smiles(smi)
        if can:
            results.append((can, "pain3_dock_collapse"))
    return results


def extract_pain4_extreme_outliers(root: Path) -> list[tuple[str, str]]:
    """Pain 4: Extreme ChEMBL outliers.

    Dual fingerprint confirmed distant molecules from crash phase.
    Source: crash experiment epoch 30-40.
    Rule: rdkit_valid=1 AND ecfp4 < 0.12 AND fcfp4 < 0.14
    """
    exp = "7gpu-dock-ecfp4-fcfp4_20260521_212927"
    files = iter_csv_files(root, exp, epoch_range=(30, 40))
    req_cols = ["chembl_prior_ecfp4_similarity", "chembl_prior_fcfp4_similarity"]
    rows = _read_valid_smiles_and_cols(files, req_cols)

    results = []
    for smi, row in rows:
        ecfp4 = _float(row, "chembl_prior_ecfp4_similarity", 2.0)
        fcfp4 = _float(row, "chembl_prior_fcfp4_similarity", 2.0)
        if not (ecfp4 < 0.12 and fcfp4 < 0.14):
            continue
        can = canonical_smiles(smi)
        if can:
            results.append((can, "pain4_extreme_outlier"))
    return results


# ── Tier 2: Property + structural-alert sweep ──────────────────────


def _process_file_chunk(file_paths: list[str]) -> dict:
    """Worker: scan a chunk of CSV files and extract negatives.

    Runs in a subprocess. Must re-import RDKit internally.

    Returns dict with keys: 'results' (list of (smiles, category)) and
    'total_mols' (int) and 'match_counts' (dict).
    """
    import csv as _csv
    from rdkit import Chem as _Chem
    from rdkit.Chem import Descriptors as _Desc

    results: list[tuple[str, str]] = []
    total_mols = 0
    match_counts: dict[str, int] = defaultdict(int)

    for fpath in file_paths:
        with open(fpath) as fh:
            reader = _csv.DictReader(fh)
            for row in reader:
                smi = row.get("smiles", "").strip()
                if row.get("rdkit_valid", "0").strip() != "1":
                    continue
                if not smi or smi == "None":
                    continue

                total_mols += 1

                mol = _Chem.MolFromSmiles(smi)
                if mol is None:
                    continue

                ha = mol.GetNumHeavyAtoms()
                carbon_count = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6)
                sulfur_count = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 16)

                logp = _Desc.MolLogP(mol)
                tpsa = _Desc.TPSA(mol)
                mw = _Desc.MolWt(mol)
                rotb = _Desc.NumRotatableBonds(mol)
                carbon_fraction = carbon_count / ha if ha > 0 else 0.0

                # Classify (4 property rules, first-match wins)
                category = None
                if carbon_fraction >= 0.95:
                    category = "prop_all_carbon"
                elif logp < -2.0 and tpsa > 100.0:
                    category = "prop_too_hydrophilic"
                elif mw > 600.0 or rotb > 12:
                    category = "prop_too_complex"
                elif sulfur_count >= 3:
                    category = "prop_sulfur_excess"

                if category is None:
                    continue

                match_counts[category] += 1

                # Canonicalize
                can = _Chem.MolToSmiles(mol, canonical=True)
                if can:
                    results.append((can, category))

    return {
        "results": results,
        "total_mols": total_mols,
        "match_counts": dict(match_counts),
    }


def extract_property_negatives(
    root: Path,
    experiments: list[tuple[str, tuple[int, int] | None]],
    n_workers: int = 1,
) -> list[tuple[str, str]]:
    """Tier 2: Drug-like-but-bad molecules via 4 property rules.

    Rules (any one match, first-match wins):
      - prop_all_carbon       : carbon_fraction >= 0.95
      - prop_too_hydrophilic  : logP < -2 AND TPSA > 100
      - prop_too_complex      : MW > 600 OR RotBond > 12
      - prop_sulfur_excess    : S_count >= 3

    Only rdkit_valid=1 molecules are considered.
    Properties are computed on-the-fly via RDKit (no CSV column dependency).

    Args:
        n_workers: Number of parallel subprocesses (default 1 = serial).
    """
    # Gather all file paths
    all_files: list[str] = []
    for exp, epoch_range in experiments:
        files = iter_csv_files(root, exp, epoch_range=epoch_range)
        all_files.extend(str(f) for f in files)

    if not all_files:
        print("  No files found.")
        return []

    # Split files across workers
    n_workers = max(1, n_workers)
    chunks: list[list[str]] = [[] for _ in range(n_workers)]
    for i, fp in enumerate(all_files):
        chunks[i % n_workers].append(fp)
    chunks = [c for c in chunks if c]  # Remove empty

    print(f"  Scanning {len(all_files)} files across {len(experiments)} experiments "
          f"with {len(chunks)} workers...")

    # Run in parallel or serial
    all_results: list[tuple[str, str]] = []
    total_mols = 0
    combined_counts: dict[str, int] = defaultdict(int)

    if n_workers == 1:
        # Serial path (no multiprocessing overhead for small jobs)
        for chunk in chunks:
            out = _process_file_chunk(chunk)
            all_results.extend(out["results"])
            total_mols += out["total_mols"]
            for k, v in out["match_counts"].items():
                combined_counts[k] += v
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_process_file_chunk, c): i for i, c in enumerate(chunks)}
            for future in as_completed(futures):
                out = future.result()
                all_results.extend(out["results"])
                total_mols += out["total_mols"]
                for k, v in out["match_counts"].items():
                    combined_counts[k] += v

    print(f"  Scanned {total_mols:,} valid molecules across {len(all_files)} files")
    print(f"  Raw matches (before dedup):")
    for cat in [
        "prop_all_carbon", "prop_too_hydrophilic", "prop_too_complex",
        "prop_sulfur_excess",
    ]:
        print(f"    {cat:<30s} {combined_counts.get(cat, 0):>6,d}")
    return all_results


# ── Deduplication ────────────────────────────────────────────────────


def deduplicate(
    items: list[tuple[str, str]],
    priority_order: list[str] | None = None,
) -> list[tuple[str, str]]:
    """Dedup by canonical SMILES, keeping highest-priority label."""
    if priority_order is None:
        priority_order = [
            "pain1_sulfur",
            "pain2_pseudo_normal",
            "pain3_dock_collapse",
            "pain4_extreme_outlier",
            "prop_all_carbon",
            "prop_too_hydrophilic",
            "prop_too_complex",
            "prop_sulfur_excess",
        ]

    prio_map = {label: i for i, label in enumerate(priority_order)}
    seen: dict[str, str] = {}
    for smi, label in items:
        if smi in seen:
            old_p = prio_map.get(seen[smi], 999)
            new_p = prio_map.get(label, 999)
            if new_p < old_p:
                seen[smi] = label
        else:
            seen[smi] = label

    return [(smi, label) for smi, label in seen.items()]


# ── Main ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Extract negative samples from OUT")
    parser.add_argument(
        "--out-dir",
        default="/home/md666/data/disk6/DiffB110/OUT",
        help="Path to OUT directory",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/md666/data/disk6/DiffB110/chembl_classifier/data/negatives",
        help="Output directory for extracted negatives",
    )
    parser.add_argument(
        "--skip-broad",
        action="store_true",
        help="Skip Tier 2 property/alert sweep",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=1,
        help="Number of parallel workers for Tier 2 (default: 1 = serial)",
    )
    parser.add_argument(
        "--skip-pain",
        action="store_true",
        help="Skip Tier 1 pain-point extraction",
    )
    args = parser.parse_args()

    root = Path(args.out_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[tuple[str, str]] = []

    # ── Tier 1: Pain-point extraction ────────────────────────────────
    if not args.skip_pain:
        print("=" * 60)
        print("Tier 1: Pain-point targeted extraction")
        print("=" * 60)

        print("\n[Pain 1] Sulfur invasion (crash epoch 24-28)...")
        pain1 = extract_pain1_sulfur(root)
        print(f"  -> {len(pain1)} molecules")
        all_results.extend(pain1)

        print("\n[Pain 2] Pseudo-normal PAH (stable epoch 50-62)...")
        pain2 = extract_pain2_pseudo_normal(root)
        print(f"  -> {len(pain2)} molecules")
        all_results.extend(pain2)

        print("\n[Pain 3] Docking backend collapse (epoch 5-9)...")
        pain3 = extract_pain3_dock_collapse(root)
        print(f"  -> {len(pain3)} molecules")
        all_results.extend(pain3)

        print("\n[Pain 4] Extreme ChEMBL outliers (crash epoch 30-40)...")
        pain4 = extract_pain4_extreme_outliers(root)
        print(f"  -> {len(pain4)} molecules")
        all_results.extend(pain4)

    # ── Tier 2: Property + structural-alert sweep ────────────────────
    if not args.skip_broad:
        print("\n" + "=" * 60)
        print("Tier 2: Property + structural-alert negative sweep")
        print("=" * 60)
        print("Rules: all-carbon | too-hydrophilic | too-complex | sulfur-excess")

        stable_experiments: list[tuple[str, tuple[int, int] | None]] = [
            ("7gpu-dock-ecfp4-fcfp4_20260525_212140", None),    # 91 epochs
            ("7gpu-dock-ecfp4-fcfp4_20260526_115341", None),    # 16 epochs, newest
            ("8gpu-chembl-only_20260519_214913", None),          # 30 epochs
            ("8gpu-chembl-only_20260520_101340", None),          # 30 epochs
            ("6gpu-chembl-only_20260521_154957", None),          # 20 epochs
            ("6gpu-dock_20260513_110544", None),                 # 22 epochs
            ("7gpu-dock-ecfp4-fcfp4_20260521_212927", (1, 23)), # pre-crash only
        ]

        broad = extract_property_negatives(root, stable_experiments, n_workers=args.n_workers)
        print(f"  -> {len(broad)} molecules (before dedup)")
        all_results.extend(broad)

    # ── Deduplication ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Cross-tier deduplication (pain-point priority)")
    print("=" * 60)
    print(f"  Before dedup: {len(all_results)}")
    deduped = deduplicate(all_results)
    print(f"  After dedup:  {len(deduped)} (-{len(all_results) - len(deduped)} dupes)")

    # ── Statistics ───────────────────────────────────────────────────
    label_counts: dict[str, int] = defaultdict(int)
    for _, label in deduped:
        label_counts[label] += 1

    print("\nLabel distribution:")
    for label in sorted(label_counts, key=label_counts.get, reverse=True):  # type: ignore[arg-type]
        print(f"  {label:.<35s} {label_counts[label]:>6d}")

    pain_total = sum(
        v for k, v in label_counts.items()
        if k.startswith("pain")
    )
    prop_total = sum(
        v for k, v in label_counts.items()
        if k.startswith("prop")
    )
    print(f"  {'pain total':.<35s} {pain_total:>6d}")
    print(f"  {'property/alert total':.<35s} {prop_total:>6d}")
    print(f"  {'TOTAL':.<35s} {len(deduped):>6d}")

    # ── Write output files ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Writing output files")
    print("=" * 60)

    # Combined labeled
    path = output_dir / "all_negatives_labeled.smi"
    with open(path, "w") as f:
        f.write("smiles\tlabel\n")
        for smi, label in deduped:
            f.write(f"{smi}\t{label}\n")
    print(f"  all_negatives_labeled.smi  : {len(deduped)} entries")

    # All SMILES only
    path = output_dir / "all_negatives.smi"
    with open(path, "w") as f:
        f.write("smiles\n")
        for smi, _ in deduped:
            f.write(f"{smi}\n")
    print(f"  all_negatives.smi          : {len(deduped)} entries")

    # Broad only (property-rule negatives)
    prop_only = [(s, l) for s, l in deduped if l.startswith("prop")]
    path = output_dir / "prop_negatives.smi"
    with open(path, "w") as f:
        f.write("smiles\n")
        for smi, _ in prop_only:
            f.write(f"{smi}\n")
    print(f"  prop_negatives.smi         : {len(prop_only)} entries")

    # Per-pain-point
    for pain_label in [
        "pain1_sulfur", "pain2_pseudo_normal",
        "pain3_dock_collapse", "pain4_extreme_outlier",
    ]:
        subset = [(s, l) for s, l in deduped if l == pain_label]
        if not subset:
            continue
        path = output_dir / f"{pain_label}.smi"
        with open(path, "w") as f:
            f.write("smiles\n")
            for smi, _ in subset:
                f.write(f"{smi}\n")
        print(f"  {pain_label}.smi{' ' * (25 - len(pain_label))}: {len(subset)} entries")

    print("\nDone! Output directory:", output_dir)


if __name__ == "__main__":
    main()
