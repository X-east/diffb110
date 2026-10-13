"""Audit existing midterm evidence without generating molecules or redocking."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import statistics
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from xml.etree import ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from rdkit.Chem import QED

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from scripts.offline_rl.structure_pathology_rules import compute_structure_pathology

LOGGER = logging.getLogger(__name__)
BENCH = ROOT / "analysis/benchmark/benchmark5/five_model_500steps_full_eval_260903"
REPEATS = ("rep4_seed22_29", "rep5_seed30_37", "rep6_seed38_45")
MODELS = {"DiffGui": "diffgui", "DiffSBDD": "diffsbdd", "GCDM-SBDD": "gcdm", "PMDM": "pmdm", "TargetDiff": "targetdiff"}
ORIGINAL = ROOT / "analysis/diffsbdd_global_e2po_structure_matched_x_only_generation_1000_260901"
ESRPO = ROOT / "OUT/diffsbdd_e2po_mainline_intermediate_gen"
RL = ROOT / "runs/diffgui-deppa-full-lr1e5-wd1e4-b8x8-t200-100ep_20260626_165809"


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a possibly BOM-prefixed CSV."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(name: str, rows: list[dict]) -> None:
    """Write heterogeneous audit rows with a stable union of fields."""
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def sdf_audit(spec: tuple[str, str, int]) -> dict:
    """Recompute validity, fragmentation, and LF rule counts in one dataset."""
    RDLogger.DisableLog("rdApp.*")
    label, filename, attempts = spec
    path = Path(filename)
    counts = dict(records=0, readable=0, raw_sanitize=0, raw_multifragment=0,
                  lf_sanitize=0, lf_clean=0, lf_clean_ha10_30=0, lf_ha10_30=0,
                  raw_single_clean=0, raw_clean=0, lf_ring0=0, explicit_h_records=0)
    clean_qed = []
    for mol in Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False):
        counts["records"] += 1
        if mol is None:
            continue
        counts["readable"] += 1
        counts["explicit_h_records"] += int(any(a.GetAtomicNum() == 1 for a in mol.GetAtoms()))
        fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        counts["raw_multifragment"] += int(len(fragments) > 1)
        raw = Chem.Mol(mol)
        try:
            Chem.SanitizeMol(raw)
            counts["raw_sanitize"] += 1
            raw_clean = not compute_structure_pathology(raw).is_pathological
            counts["raw_clean"] += int(raw_clean)
            counts["raw_single_clean"] += int(len(fragments) == 1 and raw_clean)
        except Exception:
            pass
        if not fragments:
            continue
        lf = max(fragments, key=lambda m: m.GetNumHeavyAtoms())
        try:
            Chem.SanitizeMol(lf)
            facts = compute_structure_pathology(lf)
            counts["lf_sanitize"] += 1
            in_window = 10 <= lf.GetNumHeavyAtoms() <= 30
            counts["lf_ha10_30"] += int(in_window)
            counts["lf_ring0"] += int(lf.GetRingInfo().NumRings() == 0)
            if not facts.is_pathological:
                counts["lf_clean"] += 1
                counts["lf_clean_ha10_30"] += int(in_window)
                clean_qed.append(QED.qed(lf))
        except Exception:
            continue
    return {"dataset": label, "sdf": str(path), "attempts_declared": attempts, **counts,
            "lf_clean_per_attempt_pct": counts["lf_clean"] / attempts * 100,
            "lf_clean_per_lf_pct": counts["lf_clean"] / counts["lf_sanitize"] * 100,
            "raw_single_clean_per_attempt_pct": counts["raw_single_clean"] / attempts * 100,
            "qed_median_lf_clean": statistics.median(clean_qed) if clean_qed else None}


def dock_audit(path: Path) -> dict:
    """Inspect stored scores and saved input samples; do not rerun docking."""
    rows = read_csv(path)
    values = []
    for row in rows:
        try:
            value = float(row["score"])
            if math.isfinite(value) and str(row.get("success", "")).lower() in ("true", "1"):
                values.append(value)
        except (KeyError, ValueError):
            continue
    neg = [v for v in values if v <= 0]
    usable = [v for v in values if -25 <= v <= 100]
    sample = []
    for sdf in sorted((path.parent / "split_ligands").glob("*.sdf"))[:10]:
        mol = next(iter(Chem.SDMolSupplier(str(sdf), sanitize=False, removeHs=False)), None)
        if mol is None:
            continue
        pos = mol.GetConformer().GetPositions()
        sample.append({"path": str(sdf), "explicit_h": sum(a.GetAtomicNum() == 1 for a in mol.GetAtoms()),
                       "centroid": pos.mean(axis=0).tolist()})
    summary_path = path.parent / "run_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    return {"csv": str(path), "n_records": len(rows), "n_success_finite": len(values),
            "n_nonpositive": len(neg), "n_positive_le100": sum(0 < v <= 100 for v in values),
            "n_suspect_gt100_or_ltminus25": sum(v > 100 or v < -25 for v in values),
            "mean_all_success": statistics.mean(values) if values else None,
            "median_all_success": statistics.median(values) if values else None,
            "mean_nonpositive": statistics.mean(neg) if neg else None,
            "median_nonpositive": statistics.median(neg) if neg else None,
            "mean_provisional_usable": statistics.mean(usable) if usable else None,
            "protocol_id_present": "docking_protocol_id" in summary or "docking_protocol_id" in (rows[0] if rows else {}),
            "saved_input_sample_n": len(sample), "saved_input_sample_h_present": sum(s["explicit_h"] > 0 for s in sample),
            "sample_details": json.dumps(sample), "summary": json.dumps(summary, ensure_ascii=False)}


def asset_audit() -> list[dict]:
    """Match embedded slide image bytes against existing figure assets."""
    sources = list((ROOT / "中期/图").rglob("*.png")) + list(BENCH.glob("*.png"))
    known: dict[str, list[str]] = {}
    for path in sources:
        known.setdefault(hashlib.sha256(path.read_bytes()).hexdigest(), []).append(str(path))
    ns = {"p": "http://schemas.openxmlformats.org/presentationml/2006/main", "a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
    result = []
    with zipfile.ZipFile(ROOT / "中期/谢东宸中期_备份261006_184203.pptx") as archive:
        for number in (9, 10, 13, 14, 16):
            relfile = f"ppt/slides/_rels/slide{number}.xml.rels"
            rels = {el.get("Id"): el.get("Target") for el in ET.fromstring(archive.read(relfile))}
            tree = ET.fromstring(archive.read(f"ppt/slides/slide{number}.xml"))
            for blip in tree.findall(".//a:blip", ns):
                key = blip.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed")
                member = "ppt/" + rels[key].removeprefix("../")
                digest = hashlib.sha256(archive.read(member)).hexdigest()
                result.append({"slide": number, "member": member, "sha256": digest, "matches": " | ".join(known.get(digest, []))})
    return result


def main() -> None:
    """Run bounded deterministic checks and write additive audit artifacts."""
    OUT.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(OUT / "audit.log")])
    RDLogger.DisableLog("rdApp.*")
    specs = []
    base = ROOT / "analysis/benchmark/benchmark5/five_model_repeats_500steps_260826"
    for repeat in REPEATS:
        for name, slug in MODELS.items():
            specs.append((f"{name}_{repeat[:4]}", str(base / repeat / f"filtered_sanitize_valid/{slug}.sanitize_valid.sdf"), 1000))
    for name, folder in (("Original", ORIGINAL), ("ESRPO_step4000", ESRPO / "step_004000"), ("ESRPO_final", ESRPO / "step_final")):
        specs.extend([(name + "_raw", str(folder / "molecules.sdf"), 1000),
                      (name + "_relaxed", str(folder / "molecules_relaxed_largest.sdf"), 1000)])
    genuine_original = ROOT / "analysis/diffsbdd_global_e2po_good_bad_xv_smoke_260825/generation_comparison/original"
    specs.append(("Official_original_candidate_260825", str(genuine_original / "molecules.sdf"), 1000))
    LOGGER.info("Auditing %d SDF datasets with 18 CPU workers", len(specs))
    with ProcessPoolExecutor(max_workers=18) as pool:
        sdf_rows = list(pool.map(sdf_audit, specs))
    write_csv("sdf_recomputed_counts.csv", sdf_rows)
    paths = sorted(BENCH.glob("docking/rep[456]/*/docking_results.csv"))
    paths += [ORIGINAL / "docking_relaxed/docking_results.csv"]
    paths += sorted(ESRPO.glob("step_*/docking_relaxed/docking_results.csv"))
    paths.append(genuine_original / "docking/docking_results.csv")
    dock_rows = [dock_audit(path) for path in paths]
    write_csv("docking_score_audit.csv", dock_rows)
    write_csv("ppt_image_provenance.csv", asset_audit())
    rl_rows = []
    for epoch in (1, 79, 89):
        path = RL / f"epochs/epoch_{epoch:04d}/molecules.csv"
        rows = read_csv(path)
        selected = [r for r in rows if r.get("canonical_smiles") and r.get("docking_raw")]
        score = [float(r["docking_raw"]) for r in selected]
        seeds = sorted({int(r["seed"]) for r in rows})
        rl_rows.append({"epoch": epoch, "records": len(rows), "canonical_and_docking_n": len(selected),
                        "n_rdkit_valid_flag": sum(r.get("rdkit_valid") == "1" for r in rows),
                        "n_docking_success_flag": sum(r.get("docking_success") == "1" for r in rows),
                        "n_positive": sum(v > 0 for v in score), "docking_mean_selected": statistics.mean(score),
                        "docking_median_selected": statistics.median(score), "seeds": str(seeds), "csv": str(path)})
    write_csv("rl_epoch_sampling_audit.csv", rl_rows)
    LOGGER.info("SDF counts: %s", [{k: r[k] for k in ("dataset", "records", "lf_sanitize", "lf_clean", "raw_single_clean")} for r in sdf_rows[-6:]])
    LOGGER.info("positive/suspect score datasets: %s", [{k: r[k] for k in ("csv", "n_positive_le100", "n_suspect_gt100_or_ltminus25")} for r in dock_rows if r["n_positive_le100"] or r["n_suspect_gt100_or_ltminus25"]])
    plt.rcParams.update({"font.size": 15, "axes.titlesize": 18, "axes.labelsize": 16})
    selected_rows = [r for r in sdf_rows if r["dataset"] in ("Original_raw", "ESRPO_step4000_raw", "ESRPO_final_raw")]
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    x = list(range(3))
    for shift, field, label, color in ((-.25, "lf_clean_per_attempt_pct", "LF rule-clean / attempts", "#277da1"),
                                     (0, "lf_clean_per_lf_pct", "LF rule-clean / LF valid", "#f8961e"),
                                     (.25, "raw_single_clean_per_attempt_pct", "Raw single-fragment clean / attempts", "#43aa8b")):
        vals = [r[field] for r in selected_rows]
        bars = ax.bar([v + shift for v in x], vals, width=.24, label=label, color=color)
        ax.bar_label(bars, fmt="%.1f", fontsize=13, padding=3)
    ax.set_xticks(x, ["Reported baseline*", "ESRPO step4000", "ESRPO final"])
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=12, loc="upper left")
    ax.set_title("Same samples, different denominators and fragment policies")
    fig.savefig(OUT / "denominator_and_fragment_audit.png", dpi=180)
    fig.savefig(OUT / "denominator_and_fragment_audit.svg")
    plt.close(fig)
    (OUT / "audit_manifest.json").write_text(json.dumps({"workers": 18, "sdf_datasets": len(specs), "docking_datasets": len(paths), "rl_epochs": 3, "scope": "read existing artifacts; deterministic local recomputation; no docking/training/generation", "rdkit_version": __import__("rdkit").__version__}, indent=2))
    LOGGER.info("Audit completed: %s", OUT)


if __name__ == "__main__":
    main()
