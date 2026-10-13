"""PB-valid filtering and best-docking selection for de novo drug-design benchmarks.

Logic (matches the standard cross-model benchmark convention):
1. For each model's generated+docked molecules, run PoseBusters (``dock`` config)
   to flag geometrically reasonable poses (PB-valid).
2. Compute the PB-valid rate = #PB-valid / #molecules.
3. Among PB-valid molecules, select the one with the best (lowest) docking score
   for visualization.

Usage (dif env):
    python pb_valid_selection.py --limit 100
"""

from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path

import pandas as pd
from rdkit import Chem

import posebusters.posebusters as _pbmod
from posebusters import PoseBusters

# RDKit 2025 changed GetProp signature; monkeypatch PoseBusters' name reader.
def _get_name(self, mol_pred: Chem.Mol) -> str:  # type: ignore[no-untyped-def]
    try:
        return str(mol_pred.GetProp("_Name"))
    except Exception:
        return "mol"


_pbmod.PoseBusters._get_name = _get_name

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parent
RECEPTOR = Path(
    "/home/md666/data/disk6/DiffB110/analysis/benchmark/8gjl_c1101_benchmark_20260603_500mol_500steps/inputs/8GJL_C1101_pocket8A.pdb"
)
MODEL_ORDER = ["DiffGui", "DiffSBDD", "PMDM", "GCDM-SBDD", "TargetDiff"]
MODEL_SLUG = {
    "DiffGui": "diffgui",
    "DiffSBDD": "diffsbdd",
    "PMDM": "pmdm",
    "GCDM-SBDD": "gcdm",
    "TargetDiff": "targetdiff",
}
REPEAT_ORDER = ["rep1", "rep2", "rep3"]
DOCKING_PROP = "DockingScore"


def _pb_valid_row(df_row: pd.Series) -> bool:
    """Return True if all boolean PoseBusters checks pass for one molecule."""
    bool_cols = [c for c in df_row.index if isinstance(df_row[c], (bool,)) or df_row[c] in (True, False)]
    if not bool_cols:
        return False
    return all(bool(df_row[c]) for c in bool_cols)


def _process_sdf(sdf_path: Path, receptor: Path, buster: PoseBusters, limit: int | None) -> list[dict]:
    """Run PoseBusters per molecule and collect PB-valid + docking score."""
    sup = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    records: list[dict] = []
    count = 0
    for idx, mol in enumerate(sup):
        if mol is None:
            continue
        if limit is not None and count >= limit:
            break
        count += 1
        tmp = tempfile.NamedTemporaryFile(suffix=".sdf", delete=False)
        try:
            writer = Chem.SDWriter(tmp.name)
            writer.write(mol)
            writer.close()
            df = buster.bust(mol_pred=tmp.name, mol_cond=str(receptor))
        finally:
            tmp.close()
            Path(tmp.name).unlink(missing_ok=True)
        if df.empty:
            continue
        row = df.iloc[0]
        pb_valid = _pb_valid_row(row)
        try:
            docking = float(mol.GetProp(DOCKING_PROP))
        except Exception:
            docking = float("nan")
        records.append(
            {
                "source_index": idx,
                "name": mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol{idx}",
                "pb_valid": bool(pb_valid),
                "docking_score": docking,
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receptor", default=str(RECEPTOR), help="Receptor PDB for PoseBusters.")
    parser.add_argument("--limit", type=int, default=None, help="Max molecules per model/repeat (None = all).")
    parser.add_argument("--out-dir", default=str(BASE / "pb_valid_selection"), help="Output directory.")
    parser.add_argument("--models", nargs="*", default=MODEL_ORDER, help="Model names to process.")
    parser.add_argument("--repeats", nargs="*", default=REPEAT_ORDER, help="Repeat names to process.")
    args = parser.parse_args()

    receptor = Path(args.receptor)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    buster = PoseBusters(config="dock")

    summary_rows: list[dict] = []
    for rep in args.repeats:
        for model in args.models:
            slug = MODEL_SLUG.get(model, model.lower())
            sdf_path = BASE / "docking" / rep / slug / "successful_docked_outputs.sdf"
            if not sdf_path.exists():
                logger.warning("Missing docked SDF: %s", sdf_path)
                continue
            logger.info("Processing %s / %s (limit=%s)", rep, model, args.limit)
            records = _process_sdf(sdf_path, receptor, buster, args.limit)
            if not records:
                continue
            df = pd.DataFrame(records)
            total = len(df)
            n_valid = int(df["pb_valid"].sum())
            rate = n_valid / total if total else 0.0
            valid_df = df[df["pb_valid"]]
            if not valid_df.empty and valid_df["docking_score"].notna().any():
                best = valid_df.loc[valid_df["docking_score"].idxmin()]
                best_docking = float(best["docking_score"])
                best_name = str(best["name"])
            else:
                best_docking = float("nan")
                best_name = ""
            summary_rows.append(
                {
                    "repeat": rep,
                    "model": model,
                    "sdf": str(sdf_path),
                    "total": total,
                    "pb_valid_count": n_valid,
                    "pb_valid_rate": round(rate, 6),
                    "best_pb_valid_docking_score": best_docking,
                    "best_pb_valid_name": best_name,
                }
            )
            # Write selected best-PB-valid molecule SDF for visualization.
            if best_name:
                src_sup = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
                for mol in src_sup:
                    if mol is None:
                        continue
                    if (mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol{0}") == best_name:
                        out_sdf = out_dir / f"{rep}_{slug}_best_pb_valid.sdf"
                        w = Chem.SDWriter(str(out_sdf))
                        w.write(mol)
                        w.close()
                        break

    summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / "pb_valid_summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info("Wrote %s", summary_path)
    logger.info("PB-valid summary:\n%s", summary.to_string(index=False))


if __name__ == "__main__":
    main()
