"""PB-valid trend across a DiffGui DePPA training run (parametrized).

Runs PoseBusters (`dock` config) on each epoch's generated 3D molecules and
tracks whether geometrically/chemically valid pose rate degrades over training.

For speed we write each epoch's molecules into ONE combined SDF and call
`bust` once per epoch (instead of one temp file per molecule), and process
epochs in parallel with a process pool.

Usage:
    python pb_valid_training_trend.py --run-dir <run> --receptor <pdb> \
        --out-dir <out> --name <tag> --n-jobs 8
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


def _pb_valid_series(df: pd.DataFrame) -> pd.Series:
    """Return a boolean series: PB-valid for each molecule row."""
    bool_cols: list[str] = []
    for c in df.columns:
        col = df[c].dropna()
        if len(col) == 0:
            continue
        if col.map(lambda v: isinstance(v, bool)).all():
            bool_cols.append(c)
    if not bool_cols:
        return pd.Series([False] * len(df), index=df.index)
    return df[bool_cols].fillna(False).all(axis=1).astype(bool)


def _process_epoch(args: tuple[Path, str, int]) -> tuple[dict, pd.DataFrame]:
    """Run PoseBusters on one epoch's merged molecules.sdf (worker)."""
    epoch_dir, receptor, _ = args
    sdf_path = epoch_dir / "molecules.sdf"
    csv_path = epoch_dir / "molecules.csv"
    epoch = int(epoch_dir.name.split("_")[-1])

    n_total = None
    if csv_path.exists():
        try:
            n_total = int(pd.read_csv(csv_path).shape[0])
        except Exception:
            n_total = None

    empty = {"epoch": epoch, "n_sdf": 0, "n_total": n_total, "n_pb_valid": 0,
             "pb_valid_rate_valid": float("nan"), "pb_valid_rate_total": float("nan")}
    if not sdf_path.exists():
        return empty, pd.DataFrame()

    sup = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    mols = [m for m in sup if m is not None]
    if not mols:
        return empty, pd.DataFrame()

    # Write all mols into one combined temp SDF with _Name set from mol_id.
    names: list[str] = []
    tmp = tempfile.NamedTemporaryFile(suffix=".sdf", delete=False)
    tmp_path = tmp.name
    tmp.close()
    w = Chem.SDWriter(tmp_path)
    for i, m in enumerate(mols):
        mid = m.GetProp("mol_id") if m.HasProp("mol_id") else f"mol{i}"
        m.SetProp("_Name", mid)
        names.append(mid)
        w.write(m)
    w.close()

    try:
        buster = PoseBusters(config="dock")
        df = buster.bust(mol_pred=tmp_path, mol_cond=receptor)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    flags = _pb_valid_series(df).values
    # Map by position (write order preserved on read-back).
    if len(flags) != len(names):
        logger.warning("epoch %s: flag count %d != mol count %d, truncating/padding",
                       epoch, len(flags), len(names))
        if len(flags) > len(names):
            flags = flags[: len(names)]
        else:
            flags = list(flags) + [False] * (len(names) - len(flags))

    per_mol = pd.DataFrame({"epoch": epoch, "mol_id": names, "pb_valid": flags})
    n_sdf = len(names)
    n_pb_valid = int(sum(flags))
    rate_valid = n_pb_valid / n_sdf if n_sdf else float("nan")
    rate_total = n_pb_valid / n_total if (n_total and n_total > 0) else float("nan")
    summary = {
        "epoch": epoch,
        "n_sdf": n_sdf,
        "n_total": n_total,
        "n_pb_valid": n_pb_valid,
        "pb_valid_rate_valid": round(rate_valid, 6),
        "pb_valid_rate_total": round(rate_total, 6),
    }
    return summary, per_mol


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--receptor", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name", default="run")
    ap.add_argument("--n-jobs", type=int, default=8)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    receptor = str(Path(args.receptor).resolve())

    epoch_dirs = sorted((run_dir / "epochs").glob("epoch_*"))
    logger.info("[%s] Found %d epoch dirs", args.name, len(epoch_dirs))

    import multiprocessing as mp

    tasks = [(d, receptor, args.n_jobs) for d in epoch_dirs]
    rows: list[dict] = []
    per_mol_frames: list[pd.DataFrame] = []
    if args.n_jobs > 1 and len(tasks) > 1:
        with mp.Pool(args.n_jobs) as pool:
            for summary, per_mol in pool.imap_unordered(_process_epoch, tasks):
                rows.append(summary)
                if len(per_mol):
                    per_mol_frames.append(per_mol)
                logger.info("[%s] epoch %s done (pb_valid=%s)", args.name, summary["epoch"], summary["n_pb_valid"])
    else:
        for t in tasks:
            summary, per_mol = _process_epoch(t)
            rows.append(summary)
            if len(per_mol):
                per_mol_frames.append(per_mol)
            logger.info("[%s] epoch %s done (pb_valid=%s)", args.name, summary["epoch"], summary["n_pb_valid"])

    summary_df = pd.DataFrame(rows).sort_values("epoch").reset_index(drop=True)
    summary_df.to_csv(out_dir / "per_epoch_pb_valid.csv", index=False)
    if per_mol_frames:
        pd.concat(per_mol_frames, ignore_index=True).to_csv(out_dir / "per_molecule_pb_valid.csv", index=False)

    early = summary_df[summary_df["epoch"] <= 10]
    late_lo = summary_df[(summary_df["epoch"] >= 80) & (summary_df["epoch"] <= 89)]
    # For the 1000-step run (starts at epoch 4, ~34 epochs) compare first vs last window.
    if len(late_lo) == 0:
        late_lo = summary_df.tail(10)
    print(f"\n=== [{args.name}] PB-valid rate (among valid reconstructed molecules) ===")
    print(f"early mean = {early['pb_valid_rate_valid'].mean():.4f}")
    print(f"late  mean = {late_lo['pb_valid_rate_valid'].mean():.4f}")
    print(f"early rate_total mean = {early['pb_valid_rate_total'].mean():.4f}")
    print(f"late  rate_total mean = {late_lo['pb_valid_rate_total'].mean():.4f}")
    if not summary_df["pb_valid_rate_valid"].isna().all():
        min_i = summary_df["pb_valid_rate_valid"].idxmin()
        max_i = summary_df["pb_valid_rate_valid"].idxmax()
        print(f"worst epoch {int(summary_df.loc[min_i,'epoch'])}: {summary_df.loc[min_i,'pb_valid_rate_valid']:.4f}")
        print(f"best  epoch {int(summary_df.loc[max_i,'epoch'])}: {summary_df.loc[max_i,'pb_valid_rate_valid']:.4f}")
    print(f"overall mean = {summary_df['pb_valid_rate_valid'].mean():.4f}")
    logger.info("[%s] Wrote %s", args.name, out_dir / "per_epoch_pb_valid.csv")


if __name__ == "__main__":
    main()
