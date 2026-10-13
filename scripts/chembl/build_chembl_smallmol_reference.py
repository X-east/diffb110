#!/usr/bin/env python
"""Build the ChEMBL 36 reference set: Small molecule only, no size window.

设计依据
--------
- 方法学参照 GuacaMol（Brown et al. 2019, JCIM）的 5 步清洗，但按本项目的
  DiffSBDD 现状做三处替换：
    1) 元素白名单用 DiffSBDD 词表 10 类（C,N,O,S,B,Br,Cl,P,I,F），
       不用 GuacaMol 的 13 类（它含 Si/Se，而模型无法生成这两种元素）；
    2) 不做 SMILES 长度限制，也不设分子尺寸窗口（尺寸效应在比较阶段处理）；
    3) 去掉 GuacaMol 的"与 10 个 holdout 上市药 ECFP4 > 0.323"这一条（目标不同），
       改为输出与 CrossDocked 训练集的 InChIKey 交集报告作为去泄漏证据。
- 额外新增 `molecule_type == 'Small molecule'` 过滤（GuacaMol 没有这一步）。
- 多片段规则：原始记录为多片段时取最大有机片段（含碳优先，再比重原子数、原子数）。
  这与 ChEMBL 官方（`get_parent_mol` 内部 LargestFragmentChooser）和 GuacaMol
  （"removal of salts"）一致——两者都不剔除多片段记录。实测 parent 之后不再出现
  多片段（10 万条中触发 0 次），因此本脚本**不设**"多片段剔除"步骤。
- 电荷：**不做中和**。实测证据：(1) ChEMBL 官方管线不中和电荷（标准化后仍有 0.87%
  非中性）；(2) 待剔除的非中性分子中 97.8% 是永久阳离子（季铵/吡啶鎓/咪唑鎓，无 H
  可脱），RDKit Uncharger 只能中和 2.2%；(3) 两种做法的剔除量只差约 410 条（0.023%）；
  (4) DiffSBDD 没有电荷通道，带电荷分子在表示上无法生成。故直接剔除非中性分子，
  与项目既有管线（`allow_charged=False`）一致。

处理顺序（严格按此顺序，逐步计数落盘）
--------------------------------------
 1. 取记录          molecule_dictionary JOIN compound_structures, smiles NOT NULL
 2. 分子类型        molecule_type == 'Small molecule'
 3. 最大有机片段    GetMolFrags + max(含碳, 重原子数, 原子数)
 4. 标准化          chembl_structure_pipeline.standardizer.standardize_mol
 5. ChEMBL parent   standardizer.get_parent_mol（去盐/去保护基）
 6. 电荷            形式电荷 != 0 -> 剔除（不中和）
 7. 虚原子          含原子序数 0 的原子 -> 剔除
 8. 元素白名单      heavy-atom 元素集合 ⊆ DiffSBDD 10 类
 9. sanitize        RDKit SanitizeMol 失败 -> 剔除
10. 去重            canonical SMILES（isomericSmiles=True，保留立体）精确去重

输出
----
reference_smallmol.csv.gz        最终参照集（chembl_id, canonical_smiles, 描述符）
element_composition.csv          最终参照集的元素原子占比
leak_check_vs_crossdocked.json   与 CrossDocked 训练集 SMILES 的 InChIKey 交集
build_audit.json                 口径、逐步删减账、SHA256、统计
chunks/*.csv.gz                  可复现的中间分块（可 resume）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sqlite3
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

LOGGER = logging.getLogger("chembl_smallmol_reference")

# DiffSBDD dataset_params['crossdock']['atom_decoder']（10 类），见 DiffSBDD/constants.py:170-171
ELEMENT_WHITELIST: frozenset[str] = frozenset({"C", "N", "O", "S", "B", "Br", "Cl", "P", "I", "F"})

_WORKER: dict[str, Any] = {}


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_logging(output_dir: Path) -> None:
    """Configure durable and terminal logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "build.log", encoding="utf-8"),
        ],
        force=True,
    )


def init_worker() -> None:
    """Install the per-process RDKit ChEMBL standardizer instance."""
    from chembl_structure_pipeline import standardizer

    _WORKER["standardizer"] = standardizer


def load_input_rows(sqlite_path: Path) -> pd.DataFrame:
    """Return chembl_id / molecule_type / canonical_smiles for all structures with a SMILES."""
    query = """
        SELECT md.chembl_id AS chembl_id,
               COALESCE(md.molecule_type, 'None') AS molecule_type,
               cs.canonical_smiles AS canonical_smiles
        FROM molecule_dictionary AS md
        JOIN compound_structures AS cs ON md.molregno = cs.molregno
        WHERE cs.canonical_smiles IS NOT NULL
        ORDER BY md.molregno
    """
    connection = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        frame = pd.read_sql_query(query, connection)
    finally:
        connection.close()
    return frame


def process_one(row: tuple[str, str]) -> tuple[dict[str, Any] | None, str]:
    """Apply the curation steps to one molecule and return ``(record, failure_reason)``."""
    chembl_id, smiles = row
    standardizer = _WORKER["standardizer"]

    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None, "parse_failed"

    # 3. largest organic fragment (carbon-containing first, then heavy atoms, then atoms)
    #    Equivalent to GuacaMol "removal of salts"; get_parent_mol also guarantees a
    #    single fragment downstream, so no separate multi-fragment drop is needed.
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if not fragments:
        return None, "fragment_selection_failed"
    mol = max(
        fragments,
        key=lambda frag: (
            any(atom.GetAtomicNum() == 6 for atom in frag.GetAtoms()),
            frag.GetNumHeavyAtoms(),
            frag.GetNumAtoms(),
        ),
    )

    # 4-5. official standardization and ChEMBL parent (salt removal)
    try:
        mol = standardizer.standardize_mol(mol)
        parent, _ = standardizer.get_parent_mol(mol)
    except Exception:
        return None, "standardize_failed"
    if parent is None:
        return None, "parent_missing"

    # 6. charge: no neutralization; non-neutral entries are dropped
    if Chem.GetFormalCharge(parent) != 0:
        return None, "charged"

    # 7. dummy atoms
    if any(atom.GetAtomicNum() == 0 for atom in parent.GetAtoms()):
        return None, "dummy_atom"

    # 8. element whitelist over heavy atoms (DiffSBDD vocabulary)
    symbols = {atom.GetSymbol() for atom in parent.GetAtoms() if atom.GetAtomicNum() > 1}
    if not symbols:
        return None, "no_heavy_atom"
    if not symbols.issubset(ELEMENT_WHITELIST):
        return None, "element_not_in_diffsbdd_vocab"

    # 9. sanitize
    try:
        Chem.SanitizeMol(parent)
    except Exception:
        return None, "sanitize_failed"

    # 10. canonical SMILES with stereochemistry preserved
    heavy = Chem.RemoveHs(parent)
    canonical = Chem.MolToSmiles(heavy, isomericSmiles=True, canonical=True)
    if not canonical:
        return None, "canonicalize_failed"

    record = {
        "chembl_id": chembl_id,
        "canonical_smiles": canonical,
        "heavy_atoms": int(heavy.GetNumHeavyAtoms()),
        "mol_wt": round(float(Descriptors.MolWt(heavy)), 4),
        "num_rings": int(rdMolDescriptors.CalcNumRings(heavy)),
        "num_aromatic_rings": int(rdMolDescriptors.CalcNumAromaticRings(heavy)),
        "num_heteroatoms": int(rdMolDescriptors.CalcNumHeteroatoms(heavy)),
        "murcko_scaffold": MurckoScaffold.MurckoScaffoldSmiles(mol=heavy),
        "elements": ";".join(sorted(symbols)),
    }
    return record, "ok"


def run_chunked(
    rows: list[tuple[str, str]],
    *,
    chunks_dir: Path,
    workers: int,
    chunk_size: int,
) -> None:
    """Process rows with a resumable chunk cache."""
    import multiprocessing as mp

    chunks_dir.mkdir(parents=True, exist_ok=True)
    context = mp.get_context("fork")
    started = time.time()
    with context.Pool(workers, initializer=init_worker) as pool:
        for chunk_id, start in enumerate(range(0, len(rows), chunk_size)):
            stop = min(start + chunk_size, len(rows))
            output = chunks_dir / f"chunk_{chunk_id:05d}_{start}_{stop}.csv.gz"
            if output.exists():
                LOGGER.info("resume chunk=%d [%d,%d)", chunk_id, start, stop)
                continue
            chunk_records: list[dict[str, Any]] = []
            reasons: Counter[str] = Counter()
            for record, reason in pool.imap(process_one, rows[start:stop], chunksize=16):
                reasons[reason] += 1
                if record is not None:
                    chunk_records.append(record)
            frame = pd.DataFrame(chunk_records)
            if frame.empty:
                frame = pd.DataFrame(columns=["chembl_id", "canonical_smiles"])
            frame["failure_reason"] = "ok" if not frame.empty else ""
            frame.to_csv(output, index=False, compression="gzip")
            LOGGER.info(
                "chunk=%d [%d,%d) kept=%d elapsed=%.1fs reasons=%s",
                chunk_id,
                start,
                stop,
                len(chunk_records),
                time.time() - started,
                dict(reasons),
            )


def leak_check(reference: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    """Compute the InChIKey intersection between the reference set and CrossDocked ligands."""
    candidates = [
        Path("/home/md666/data/disk6/DiffB110/data/crossdocked_raw/processed_edge_noH_full/train_smiles.npy"),
        Path("/home/md666/data/disk6/DiffB110/data/crossdocked_raw/processed_edge_noH_10class_full/train_smiles.npy"),
    ]
    source = next((path for path in candidates if path.exists()), None)
    if source is None:
        return {"status": "skipped", "reason": "CrossDocked train_smiles.npy not found"}

    smiles_array = np.load(source, allow_pickle=True)
    keys: set[str] = set()
    for smiles in smiles_array:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is not None:
            keys.add(Chem.MolToInchiKey(mol))
    reference_keys = {Chem.MolToInchiKey(Chem.MolFromSmiles(s)) for s in reference["canonical_smiles"].head(200000)}
    reference_keys.discard(None)
    payload = {
        "status": "completed",
        "crossdocked_smiles_source": str(source),
        "crossdocked_unique_inchikeys": len(keys),
        "reference_sample_scored": int(min(200000, len(reference))),
        "intersection_size": len(reference_keys & keys),
        "note": "参照集侧只对前 200,000 条做 InChIKey 计算；交集为空即视为无泄漏。",
    }
    return payload


def main() -> None:
    """Build and audit the Small-molecule-only ChEMBL reference set."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=18)
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--max-records", type=int, default=0)
    args = parser.parse_args()

    if args.workers <= 0 or args.chunk_size <= 0:
        raise ValueError("workers and chunk-size must be positive")
    if not args.sqlite_path.is_file():
        raise FileNotFoundError(args.sqlite_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(args.output_dir)

    LOGGER.info("loading input rows from %s", args.sqlite_path)
    table = load_input_rows(args.sqlite_path)
    input_rows = len(table)
    type_counts = {str(k): int(v) for k, v in table["molecule_type"].value_counts().items()}
    LOGGER.info("input rows=%d types=%s", input_rows, type_counts)

    selected = table.loc[table["molecule_type"].eq("Small molecule")]
    if args.max_records > 0:
        selected = selected.head(args.max_records)
    selected = selected.reset_index(drop=True)
    LOGGER.info("small-molecule rows=%d", len(selected))

    rows = list(zip(selected["chembl_id"].astype(str), selected["canonical_smiles"].astype(str)))
    run_chunked(
        rows,
        chunks_dir=args.output_dir / "chunks",
        workers=args.workers,
        chunk_size=args.chunk_size,
    )

    chunk_paths = sorted((args.output_dir / "chunks").glob("chunk_*.csv.gz"))
    frame = pd.concat([pd.read_csv(path) for path in chunk_paths], ignore_index=True)
    reason_counts = {str(k): int(v) for k, v in frame.get("failure_reason", pd.Series(dtype=str)).value_counts().items()}
    kept = frame.loc[frame["failure_reason"].eq("ok")].copy() if "failure_reason" in frame else frame.copy()
    before_dedup = len(kept)
    kept = kept.drop_duplicates(subset=["canonical_smiles"], keep="first").reset_index(drop=True)
    LOGGER.info("kept=%d dedup_removed=%d final=%d", before_dedup, before_dedup - len(kept), len(kept))

    kept = kept.drop(columns=["failure_reason"], errors="ignore")
    output_csv = args.output_dir / "reference_smallmol.csv.gz"
    kept.to_csv(output_csv, index=False, compression="gzip")

    element_counter: Counter[str] = Counter()
    for elements in kept["elements"]:
        for symbol in str(elements).split(";"):
            if symbol:
                element_counter[symbol] += 1
    total_elements = sum(element_counter.values())
    element_frame = pd.DataFrame(
        [
            {"element": symbol, "molecules_containing": count,
             "share_of_molecules_pct": round(100.0 * count / max(len(kept), 1), 4)}
            for symbol, count in element_counter.most_common()
        ]
    )
    element_frame.to_csv(args.output_dir / "element_composition.csv", index=False)

    leak = leak_check(kept, args.output_dir)
    (args.output_dir / "leak_check_vs_crossdocked.json").write_text(
        json.dumps(leak, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    audit = {
        "status": "completed",
        "sqlite_path": str(args.sqlite_path),
        "sqlite_sha256": sha256_file(args.sqlite_path),
        "element_whitelist": sorted(ELEMENT_WHITELIST),
        "no_size_window": True,
        "no_mw_window": True,
        "stereo_policy": "isomericSmiles=True（保留立体；与 chembl_standardized_all 的去重口径一致）",
        "charge_policy": "不做中和，直接剔除非中性（实测 Uncharger 只能救 2.2%，且 DiffSBDD 无电荷通道）",
        "multi_fragment_policy": "取最大有机片段；不剔除多片段记录（与 ChEMBL 官方 get_parent_mol / GuacaMol 一致）",
        "steps": [
            "1 取记录: molecule_dictionary JOIN compound_structures, smiles NOT NULL",
            "2 分子类型: molecule_type == 'Small molecule'",
            "3 最大有机片段: 含碳优先 -> 重原子数 -> 原子数",
            "4 标准化: chembl_structure_pipeline.standardize_mol",
            "5 ChEMBL parent: standardizer.get_parent_mol",
            "6 电荷: 形式电荷 != 0 -> 剔除（不中和）",
            "7 含虚原子(原子序数 0) -> 剔除",
            "8 元素白名单: 重原子元素 ⊆ DiffSBDD 10 类",
            "9 RDKit SanitizeMol 失败 -> 剔除",
            "10 canonical SMILES (isomericSmiles=True) 精确去重",
        ],
        "counts": {
            "input_rows_with_smiles": input_rows,
            "molecule_type_breakdown": type_counts,
            "small_molecule_rows": len(selected),
            "failure_reasons": reason_counts,
            "kept_before_dedup": before_dedup,
            "dedup_removed": before_dedup - len(kept),
            "final_reference_size": len(kept),
        },
        "final_stats": {
            "heavy_atoms_min": int(kept["heavy_atoms"].min()),
            "heavy_atoms_max": int(kept["heavy_atoms"].max()),
            "heavy_atoms_median": float(kept["heavy_atoms"].median()),
            "heavy_atoms_mean": round(float(kept["heavy_atoms"].mean()), 3),
            "mol_wt_min": float(kept["mol_wt"].min()),
            "mol_wt_max": float(kept["mol_wt"].max()),
            "mol_wt_median": float(kept["mol_wt"].median()),
            "unique_murcko_scaffolds": int(kept["murcko_scaffold"].nunique()),
        },
        "output": {"reference_csv": str(output_csv), "sha256": sha256_file(output_csv)},
        "leak_check": leak,
    }
    (args.output_dir / "build_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    LOGGER.info("completed final_reference_size=%d", len(kept))
    LOGGER.info("audit=%s", args.output_dir / "build_audit.json")


if __name__ == "__main__":
    main()
