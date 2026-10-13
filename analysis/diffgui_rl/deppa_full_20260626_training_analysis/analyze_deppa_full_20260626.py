#!/usr/bin/env python3
"""Analyze the 2026-06-26 DiffGui DePPA full fine-tuning run."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen

RDLogger.DisableLog("rdApp.*")


ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_DIR = Path(__file__).resolve().parent
WORK_RECORD_DIR = ROOT / "work_records" / "26-6" / "260629"
WORK_RECORD_PATH = WORK_RECORD_DIR / "260629-01 DiffGui DePPA full 6月26日训练分析.md"

MAIN_RUN = ROOT / "runs" / "diffgui-deppa-full-lr1e5-wd1e4-b8x8-t200-100ep_20260626_165809"
MAIN_LOG = (
    ROOT
    / "launch"
    / "archive"
    / "logs"
    / "diffgui_deppa_full_20260626"
    / "diffgui_deppa_full_lr1e5_wd1e4_b8x8_t200_100ep_20260626_165809.log"
)

REFERENCE_RUNS = [
    (
        "last3_25ep_20260624",
        ROOT
        / "runs"
        / "diffgui"
        / "diffgui-deppa-last3-lr1e5-b18x8-t200-100ep_20260624_102756",
    ),
    (
        "last3_continue35_20260625",
        ROOT
        / "runs"
        / "diffgui"
        / "diffgui-deppa-last3-lr1e5-b18x8-t200-100ep_continue_from25_20260625_110625",
    ),
    (
        "full_trial_100829",
        ROOT
        / "runs"
        / "diffgui"
        / "diffgui-deppa-full-lr1e5-wd1e4-b8x8-t200-100ep_20260626_100829",
    ),
    (
        "full_trial_113145",
        ROOT
        / "runs"
        / "diffgui"
        / "diffgui-deppa-full-lr1e5-wd1e4-b8x8-t200-100ep_20260626_113145",
    ),
    ("full_main_165809", MAIN_RUN),
]

NUMERIC_COLS = [
    "reward_for_advantage_mean",
    "deppa_reward_mean",
    "reward_total_mean",
    "reward_total_scored_mean",
    "docking_raw_mean",
    "docking_reward_mean",
    "ligand_efficiency_raw_mean",
    "reconstruction_success_ratio",
    "docking_success_ratio_total",
    "molecule_rdkit_valid_ratio",
    "duplicate_ratio",
    "unique_canonical_smiles",
    "unique_scaffolds",
    "descriptor_mmd_vs_baseline",
    "ecfp4_mean_pairwise_similarity",
    "fcfp4_mean_pairwise_similarity",
    "carbon_fraction",
    "hetero_atom_mean",
    "logp_mean",
    "tpsa_mean",
    "hba_mean",
    "hbd_mean",
    "ring_count_mean",
    "aromatic_ring_mean",
    "three_member_ring_total",
    "valence_violation_total",
    "fragment_multi_ratio",
    "PPO_Loss",
    "PPO_ApproxKL",
    "ClipFrac",
    "RatioMean",
    "GradNorm",
    "total_seconds",
    "seconds_per_graph",
    "rollout_seconds",
    "reward_seconds",
    "ppo_seconds",
    "gpu_monitor_peak_used_mb",
]


def _rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _to_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    if isinstance(value, (float, np.floating)):
        if math.isfinite(float(value)):
            return f"{float(value):.{digits}f}"
        return str(value)
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def _md_table(df: pd.DataFrame, digits: int = 3, max_rows: int | None = None) -> str:
    if df.empty:
        return "_无数据_"
    shown = df.copy()
    if max_rows is not None:
        shown = shown.head(max_rows)
    columns = list(shown.columns)
    rows = []
    rows.append("| " + " | ".join(columns) + " |")
    rows.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for _, row in shown.iterrows():
        values = [_fmt(row[col], digits=digits).replace("\n", " ") for col in columns]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def _select_existing(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df[[column for column in columns if column in df.columns]].copy()


def _load_official_summary() -> pd.DataFrame:
    summary = _read_csv(MAIN_RUN / "training_summary.csv")
    if summary.empty:
        raise FileNotFoundError(MAIN_RUN / "training_summary.csv")
    summary = _to_numeric(summary, NUMERIC_COLS + ["epoch"])
    summary["source"] = "official_merged"
    return summary


def _load_epoch_molecules(summary: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for epoch in summary["epoch"].dropna().astype(int).tolist():
        path = MAIN_RUN / "epochs" / f"epoch_{epoch:04d}" / "molecules.csv"
        if path.exists():
            frame = pd.read_csv(path)
            frame["source"] = "official_merged"
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    molecules = pd.concat(frames, ignore_index=True)
    molecules = _to_numeric(
        molecules,
        [
            "epoch",
            "rank",
            "rdkit_valid",
            "docking_success",
            "reward_total",
            "reward_for_advantage",
            "deppa_reward",
            "qed_raw",
            "sa_raw",
            "docking_raw",
            "docking_reward",
            "ligand_efficiency_raw",
            "reward_total_raw",
            "advantage",
            "qed_reward",
            "sa_reward",
            "deppa_valid",
            "deppa_metric_count",
            "deppa_qed_raw",
            "deppa_qed_normal",
            "deppa_sa_raw",
            "deppa_sa_normal",
            "deppa_vina_score_raw",
            "deppa_vina_score_normal",
            "deppa_distance_raw",
            "deppa_distance_normal",
            "num_atoms",
            "duplicate_count",
            "scaffold_count",
        ],
    )
    return molecules


def _load_partial_epoch90() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted((MAIN_RUN / "epochs" / "epoch_0090").glob("rank_*/molecules_with_rewards.csv")):
        frame = pd.read_csv(path)
        frame["source"] = "rank_partial_unmerged"
        frame["rank_dir"] = path.parent.name
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    partial = pd.concat(frames, ignore_index=True)
    partial = _to_numeric(
        partial,
        [
            "epoch",
            "rank",
            "rdkit_valid",
            "docking_success",
            "reward_total",
            "reward_for_advantage",
            "deppa_reward",
            "qed_raw",
            "sa_raw",
            "docking_raw",
            "docking_reward",
            "ligand_efficiency_raw",
            "reward_total_raw",
            "advantage",
            "qed_reward",
            "sa_reward",
            "deppa_valid",
            "deppa_metric_count",
            "deppa_qed_raw",
            "deppa_qed_normal",
            "deppa_sa_raw",
            "deppa_sa_normal",
            "deppa_vina_score_raw",
            "deppa_vina_score_normal",
            "deppa_distance_raw",
            "deppa_distance_normal",
            "num_atoms",
        ],
    )
    return partial


def _summarize_partial_molecules(molecules: pd.DataFrame) -> pd.DataFrame:
    if molecules.empty:
        return pd.DataFrame()
    n = len(molecules)
    canonical = molecules.get("canonical_smiles", pd.Series(dtype=object)).dropna()
    scaffolds = molecules.get("murcko_scaffold", pd.Series(dtype=object)).dropna()
    rdkit_valid_ratio = molecules.get("rdkit_valid", pd.Series(dtype=float)).mean()
    docking_success_ratio = molecules.get("docking_success", pd.Series(dtype=float)).mean()
    reward_for_advantage_mean = molecules.get("reward_for_advantage", pd.Series(dtype=float)).mean()
    raw_reward_total_mean = molecules.get("reward_total", pd.Series(dtype=float)).mean()
    row = {
        "epoch": 90,
        "source": "rank_partial_unmerged",
        "molecule_records_total": n,
        "molecule_rdkit_valid_ratio": rdkit_valid_ratio,
        "molecule_docking_success_ratio": docking_success_ratio,
        "docking_success_ratio_total": docking_success_ratio,
        "reward_for_advantage_mean": reward_for_advantage_mean,
        "deppa_reward_mean": molecules.get("deppa_reward", pd.Series(dtype=float)).mean(),
        "reward_total_mean": reward_for_advantage_mean,
        "reward_total_scored_mean": raw_reward_total_mean,
        "docking_raw_mean": molecules.get("docking_raw", pd.Series(dtype=float)).mean(),
        "docking_reward_mean": molecules.get("docking_reward", pd.Series(dtype=float)).mean(),
        "qed_raw_mean": molecules.get("qed_raw", pd.Series(dtype=float)).mean(),
        "sa_raw_mean": molecules.get("sa_raw", pd.Series(dtype=float)).mean(),
        "ligand_efficiency_raw_mean": molecules.get(
            "ligand_efficiency_raw", pd.Series(dtype=float)
        ).mean(),
        "unique_canonical_smiles": int(canonical.nunique()) if not canonical.empty else 0,
        "duplicate_ratio": 1.0 - float(canonical.nunique()) / float(n) if n else np.nan,
        "unique_scaffolds": int(scaffolds.nunique()) if not scaffolds.empty else 0,
    }
    return pd.DataFrame([row])


def _epoch_molecule_summary(molecules: pd.DataFrame, partial90: pd.DataFrame) -> pd.DataFrame:
    frames = []
    if not molecules.empty:
        grouped = molecules.groupby("epoch", as_index=False)
        official = grouped.agg(
            molecule_records_total=("epoch", "size"),
            rdkit_valid_ratio=("rdkit_valid", "mean"),
            docking_success_ratio=("docking_success", "mean"),
            reward_for_advantage_mean=("reward_for_advantage", "mean"),
            deppa_reward_mean=("deppa_reward", "mean"),
            reward_total_mean=("reward_total", "mean"),
            docking_raw_mean=("docking_raw", "mean"),
            docking_reward_mean=("docking_reward", "mean"),
            qed_raw_mean=("qed_raw", "mean"),
            sa_raw_mean=("sa_raw", "mean"),
            ligand_efficiency_raw_mean=("ligand_efficiency_raw", "mean"),
            unique_canonical_smiles=("canonical_smiles", pd.Series.nunique),
            unique_scaffolds=("murcko_scaffold", pd.Series.nunique),
        )
        official["duplicate_ratio"] = 1.0 - official["unique_canonical_smiles"] / official[
            "molecule_records_total"
        ]
        official["source"] = "official_merged"
        frames.append(official)
    partial_summary = _summarize_partial_molecules(partial90)
    if not partial_summary.empty:
        partial_summary = partial_summary.rename(
            columns={
                "molecule_rdkit_valid_ratio": "rdkit_valid_ratio",
                "molecule_docking_success_ratio": "docking_success_ratio",
            }
        )
        frames.append(partial_summary)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _window_summary(summary: pd.DataFrame) -> pd.DataFrame:
    windows = [
        ("early_01_10", 1, 10),
        ("mid_11_30", 11, 30),
        ("mid_31_60", 31, 60),
        ("late_61_89", 61, 89),
        ("last10_80_89", 80, 89),
    ]
    rows = []
    cols = [
        "reward_for_advantage_mean",
        "deppa_reward_mean",
        "reward_total_mean",
        "reward_total_scored_mean",
        "docking_raw_mean",
        "docking_success_ratio_total",
        "molecule_rdkit_valid_ratio",
        "duplicate_ratio",
        "unique_canonical_smiles",
        "descriptor_mmd_vs_baseline",
        "PPO_ApproxKL",
        "ClipFrac",
        "RatioMean",
        "seconds_per_graph",
    ]
    for label, start, end in windows:
        part = summary[(summary["epoch"] >= start) & (summary["epoch"] <= end)]
        row = {"window": label, "start_epoch": start, "end_epoch": end, "n_epochs": len(part)}
        for col in cols:
            if col in part.columns:
                row[f"{col}_mean"] = part[col].mean()
                row[f"{col}_std"] = part[col].std()
        rows.append(row)
    return pd.DataFrame(rows)


def _key_epoch_summary(summary: pd.DataFrame, partial90_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    labels: list[tuple[str, pd.Series]] = []
    if not summary.empty:
        labels.append(("epoch_0001", summary.loc[summary["epoch"].idxmin()]))
        labels.append(("last_official_epoch_0089", summary.loc[summary["epoch"].idxmax()]))
        if "reward_for_advantage_mean" in summary.columns:
            labels.append(("best_deppa_reward", summary.loc[summary["reward_for_advantage_mean"].idxmax()]))
        if "reward_total_mean" in summary.columns:
            labels.append(("best_reward_total", summary.loc[summary["reward_total_mean"].idxmax()]))
        if "docking_raw_mean" in summary.columns:
            labels.append(("best_mean_docking", summary.loc[summary["docking_raw_mean"].idxmin()]))
        if "descriptor_mmd_vs_baseline" in summary.columns:
            labels.append(("lowest_descriptor_mmd", summary.loc[summary["descriptor_mmd_vs_baseline"].idxmin()]))
    keep = [
        "label",
        "epoch",
        "source",
        "reward_for_advantage_mean",
        "deppa_reward_mean",
        "reward_total_mean",
        "reward_total_scored_mean",
        "docking_raw_mean",
        "docking_success_ratio_total",
        "molecule_rdkit_valid_ratio",
        "duplicate_ratio",
        "unique_canonical_smiles",
        "unique_scaffolds",
        "descriptor_mmd_vs_baseline",
        "PPO_ApproxKL",
        "ClipFrac",
        "RatioMean",
        "checkpoint",
    ]
    for label, series in labels:
        row = {"label": label}
        for col in keep:
            if col != "label" and col in series.index:
                row[col] = series[col]
        rows.append(row)
    if not partial90_summary.empty:
        row = {"label": "partial_epoch_0090_unmerged"}
        for col in keep:
            if col != "label" and col in partial90_summary.columns:
                row[col] = partial90_summary.iloc[0][col]
        rows.append(row)
    return pd.DataFrame(rows)


def _reference_comparison() -> pd.DataFrame:
    rows = []
    for label, run_dir in REFERENCE_RUNS:
        summary_path = run_dir / "training_summary.csv"
        if not summary_path.exists():
            rows.append({"run": label, "exists": False, "run_dir": _rel(run_dir)})
            continue
        df = _read_csv(summary_path)
        df = _to_numeric(df, NUMERIC_COLS + ["epoch"])
        if df.empty:
            rows.append({"run": label, "exists": True, "run_dir": _rel(run_dir), "n_epochs": 0})
            continue
        last = df.loc[df["epoch"].idxmax()]
        last5 = df.sort_values("epoch").tail(min(5, len(df)))
        row = {
            "run": label,
            "exists": True,
            "run_dir": _rel(run_dir),
            "n_epochs": len(df),
            "last_epoch": int(last["epoch"]),
            "last_status": last.get("status", ""),
        }
        for col in [
            "reward_for_advantage_mean",
            "deppa_reward_mean",
            "reward_total_mean",
            "reward_total_scored_mean",
            "docking_raw_mean",
            "docking_success_ratio_total",
            "molecule_rdkit_valid_ratio",
            "duplicate_ratio",
            "descriptor_mmd_vs_baseline",
            "PPO_ApproxKL",
            "ClipFrac",
            "RatioMean",
        ]:
            if col in df.columns:
                row[f"final_{col}"] = last[col]
                row[f"last5_{col}"] = last5[col].mean()
        if "reward_for_advantage_mean" in df.columns:
            best = df.loc[df["reward_for_advantage_mean"].idxmax()]
            row["best_deppa_epoch"] = int(best["epoch"])
            row["best_deppa_reward_for_advantage_mean"] = best["reward_for_advantage_mean"]
        if "docking_raw_mean" in df.columns:
            best_dock = df.loc[df["docking_raw_mean"].idxmin()]
            row["best_docking_epoch"] = int(best_dock["epoch"])
            row["best_docking_raw_mean"] = best_dock["docking_raw_mean"]
        rows.append(row)
    return pd.DataFrame(rows)


def _top_molecules(molecules: pd.DataFrame, partial90: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    if not molecules.empty:
        frames.append(molecules)
    if not partial90.empty:
        frames.append(partial90)
    if not frames:
        return pd.DataFrame(), pd.DataFrame()
    all_molecules = pd.concat(frames, ignore_index=True, sort=False)
    valid = all_molecules.copy()
    if "rdkit_valid" in valid.columns:
        valid = valid[valid["rdkit_valid"] == 1]
    if "docking_success" in valid.columns:
        valid = valid[valid["docking_success"] == 1]
    keep = [
        "source",
        "epoch",
        "rank",
        "global_mol_id",
        "canonical_smiles",
        "murcko_scaffold",
        "reward_for_advantage",
        "deppa_reward",
        "reward_total",
        "qed_raw",
        "sa_raw",
        "docking_raw",
        "docking_reward",
        "ligand_efficiency_raw",
        "num_atoms",
    ]
    existing = [col for col in keep if col in valid.columns]
    by_deppa = valid.sort_values("reward_for_advantage", ascending=False, na_position="last")
    by_docking = valid.sort_values("docking_raw", ascending=True, na_position="last")
    return by_deppa[existing].head(30), by_docking[existing].head(30)


def _ppo_timestep_extremes(epoch: int) -> pd.DataFrame:
    path = MAIN_RUN / "epochs" / f"epoch_{epoch:04d}" / "ppo_step_metrics.csv"
    df = _read_csv(path)
    if df.empty:
        return pd.DataFrame()
    df = _to_numeric(df, ["timestep_mean", "approx_kl", "clipfrac", "ratio_mean"])
    grouped = (
        df.groupby("timestep_mean", as_index=False)
        .agg(
            approx_kl_mean=("approx_kl", "mean"),
            approx_kl_max=("approx_kl", "max"),
            clipfrac_mean=("clipfrac", "mean"),
            ratio_mean=("ratio_mean", "mean"),
        )
        .sort_values("approx_kl_max", ascending=False)
    )
    return grouped.head(25)


def _combine_molecules(molecules: pd.DataFrame, partial90: pd.DataFrame) -> pd.DataFrame:
    frames = []
    if not molecules.empty:
        frames.append(molecules)
    if not partial90.empty:
        frames.append(partial90)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = _to_numeric(
        combined,
        [
            "epoch",
            "rank",
            "rdkit_valid",
            "docking_success",
            "reward_total",
            "reward_for_advantage",
            "deppa_reward",
            "qed_raw",
            "sa_raw",
            "docking_raw",
            "docking_reward",
            "ligand_efficiency_raw",
            "num_atoms",
            "deppa_qed_normal",
            "deppa_sa_normal",
            "deppa_vina_score_normal",
            "deppa_distance_normal",
            "deppa_qed_raw",
            "deppa_sa_raw",
            "deppa_vina_score_raw",
            "deppa_distance_raw",
        ],
    )
    return combined


def _component_epoch_summary(all_molecules: pd.DataFrame) -> pd.DataFrame:
    if all_molecules.empty:
        return pd.DataFrame()
    columns = [
        "reward_for_advantage",
        "reward_total",
        "qed_raw",
        "sa_raw",
        "docking_raw",
        "docking_reward",
        "ligand_efficiency_raw",
        "deppa_qed_normal",
        "deppa_sa_normal",
        "deppa_vina_score_normal",
        "deppa_distance_normal",
        "deppa_qed_raw",
        "deppa_sa_raw",
        "deppa_vina_score_raw",
        "deppa_distance_raw",
    ]
    rows = []
    for (epoch, source), group in all_molecules.groupby(["epoch", "source"], dropna=False):
        row = {"epoch": int(epoch), "source": source, "n_molecules": len(group)}
        for column in columns:
            if column in group.columns:
                row[f"{column}_mean"] = group[column].mean()
                row[f"{column}_median"] = group[column].median()
                row[f"{column}_std"] = group[column].std()
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["epoch", "source"])


def _molecule_property_quantiles(all_molecules: pd.DataFrame) -> pd.DataFrame:
    if all_molecules.empty:
        return pd.DataFrame()
    properties = [
        "reward_for_advantage",
        "reward_total",
        "docking_raw",
        "docking_reward",
        "ligand_efficiency_raw",
        "qed_raw",
        "sa_raw",
        "num_atoms",
        "deppa_qed_normal",
        "deppa_sa_normal",
        "deppa_vina_score_normal",
        "deppa_distance_normal",
    ]
    rows = []
    for (epoch, source), group in all_molecules.groupby(["epoch", "source"], dropna=False):
        for prop in properties:
            if prop not in group.columns:
                continue
            values = group[prop].dropna()
            if values.empty:
                continue
            row = {
                "epoch": int(epoch),
                "source": source,
                "property": prop,
                "n": int(values.shape[0]),
                "mean": values.mean(),
                "std": values.std(),
                "min": values.min(),
                "q10": values.quantile(0.10),
                "q25": values.quantile(0.25),
                "median": values.quantile(0.50),
                "q75": values.quantile(0.75),
                "q90": values.quantile(0.90),
                "max": values.max(),
            }
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["property", "epoch", "source"])


def _failure_mode_stats(all_molecules: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if all_molecules.empty:
        return pd.DataFrame(), pd.DataFrame()
    df = all_molecules.copy()
    reconstruction = df.get("reconstruction_status", pd.Series([""] * len(df))).fillna("")
    rdkit_valid = df.get("rdkit_valid", pd.Series([np.nan] * len(df)))
    docking_success = df.get("docking_success", pd.Series([np.nan] * len(df)))
    modes = []
    for rec, valid, dock in zip(reconstruction, rdkit_valid, docking_success):
        if rec and rec != "ok":
            modes.append(f"reconstruction:{rec}")
        elif valid != 1:
            modes.append("rdkit_invalid")
        elif dock != 1:
            modes.append("docking_failed")
        else:
            modes.append("success")
    df["failure_mode"] = modes
    counts = (
        df.groupby(["epoch", "source", "failure_mode"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["epoch", "source", "count"], ascending=[True, True, False])
    )
    totals = counts.groupby(["epoch", "source"], as_index=False)["count"].sum().rename(
        columns={"count": "epoch_total"}
    )
    counts = counts.merge(totals, on=["epoch", "source"], how="left")
    counts["ratio"] = counts["count"] / counts["epoch_total"]
    overall = (
        counts.groupby("failure_mode", as_index=False)["count"]
        .sum()
        .sort_values("count", ascending=False)
    )
    overall["ratio"] = overall["count"] / overall["count"].sum()
    return counts, overall


def _rank_epoch_summary(all_molecules: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if all_molecules.empty or "rank" not in all_molecules.columns:
        return pd.DataFrame(), pd.DataFrame()
    grouped = all_molecules.groupby(["epoch", "source", "rank"], dropna=False)
    rank_epoch = grouped.agg(
        n_molecules=("epoch", "size"),
        rdkit_valid_ratio=("rdkit_valid", "mean"),
        docking_success_ratio=("docking_success", "mean"),
        reward_for_advantage_mean=("reward_for_advantage", "mean"),
        reward_total_mean=("reward_total", "mean"),
        docking_raw_mean=("docking_raw", "mean"),
        qed_raw_mean=("qed_raw", "mean"),
        sa_raw_mean=("sa_raw", "mean"),
        unique_smiles=("canonical_smiles", pd.Series.nunique),
        unique_scaffolds=("murcko_scaffold", pd.Series.nunique),
    ).reset_index()
    rank_epoch["duplicate_ratio"] = 1.0 - rank_epoch["unique_smiles"] / rank_epoch["n_molecules"]
    rank_overall = (
        rank_epoch.groupby(["source", "rank"], dropna=False)
        .agg(
            n_epochs=("epoch", "nunique"),
            rdkit_valid_ratio_mean=("rdkit_valid_ratio", "mean"),
            docking_success_ratio_mean=("docking_success_ratio", "mean"),
            reward_for_advantage_mean=("reward_for_advantage_mean", "mean"),
            reward_for_advantage_std=("reward_for_advantage_mean", "std"),
            docking_raw_mean=("docking_raw_mean", "mean"),
            docking_raw_std=("docking_raw_mean", "std"),
            duplicate_ratio_mean=("duplicate_ratio", "mean"),
        )
        .reset_index()
    )
    return rank_epoch.sort_values(["epoch", "rank"]), rank_overall.sort_values(["source", "rank"])


def _topk_epoch_summary(all_molecules: pd.DataFrame) -> pd.DataFrame:
    if all_molecules.empty:
        return pd.DataFrame()
    rows = []
    for (epoch, source), group in all_molecules.groupby(["epoch", "source"], dropna=False):
        valid = group.copy()
        if "rdkit_valid" in valid.columns:
            valid = valid[valid["rdkit_valid"] == 1]
        if "docking_success" in valid.columns:
            valid = valid[valid["docking_success"] == 1]
        row = {
            "epoch": int(epoch),
            "source": source,
            "n_molecules": len(group),
            "n_valid_docked": len(valid),
            "valid_docked_ratio": len(valid) / len(group) if len(group) else np.nan,
        }
        if not valid.empty:
            reward_sorted = valid.sort_values("reward_for_advantage", ascending=False, na_position="last")
            docking_sorted = valid.sort_values("docking_raw", ascending=True, na_position="last")
            for k in [1, 5, 10]:
                top_reward = reward_sorted.head(k)
                top_docking = docking_sorted.head(k)
                row[f"top{k}_reward_mean"] = top_reward["reward_for_advantage"].mean()
                row[f"top{k}_raw_reward_mean"] = top_reward["reward_total"].mean()
                row[f"top{k}_docking_raw_mean"] = top_docking["docking_raw"].mean()
                row[f"top{k}_docking_reward_mean"] = top_docking["docking_reward"].mean()
            row["best_reward"] = reward_sorted["reward_for_advantage"].iloc[0]
            row["best_docking_raw"] = docking_sorted["docking_raw"].iloc[0]
            row["dock_le_-12_count"] = int((valid["docking_raw"] <= -12.0).sum())
            row["dock_le_-14_count"] = int((valid["docking_raw"] <= -14.0).sum())
            row["balanced_qed065_sa5_dock10_count"] = int(
                (
                    (valid["qed_raw"] >= 0.65)
                    & (valid["sa_raw"] <= 5.0)
                    & (valid["docking_raw"] <= -10.0)
                ).sum()
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["epoch", "source"])


def _selected_ppo_timestep_summary(epochs: list[int]) -> pd.DataFrame:
    frames = []
    for epoch in epochs:
        path = MAIN_RUN / "epochs" / f"epoch_{epoch:04d}" / "ppo_step_metrics.csv"
        df = _read_csv(path)
        if df.empty:
            continue
        df = _to_numeric(df, ["timestep_mean", "approx_kl", "clipfrac", "ratio_mean"])
        grouped = (
            df.groupby("timestep_mean", as_index=False)
            .agg(
                approx_kl_mean=("approx_kl", "mean"),
                approx_kl_max=("approx_kl", "max"),
                clipfrac_mean=("clipfrac", "mean"),
                ratio_mean=("ratio_mean", "mean"),
            )
            .sort_values("timestep_mean")
        )
        grouped["epoch"] = epoch
        frames.append(grouped)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _ring_flags_from_smiles(smiles: object) -> tuple[float, float, float]:
    if not isinstance(smiles, str) or not smiles:
        return np.nan, np.nan, np.nan
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.nan, np.nan, np.nan
    rings = mol.GetRingInfo().AtomRings()
    has_three_member_ring = any(len(ring) == 3 for ring in rings)
    has_benzene_ring = any(
        len(ring) == 6
        and all(
            mol.GetAtomWithIdx(atom_idx).GetAtomicNum() == 6
            and mol.GetAtomWithIdx(atom_idx).GetIsAromatic()
            for atom_idx in ring
        )
        for ring in rings
    )
    return (
        float(Crippen.MolLogP(mol)),
        float(has_three_member_ring),
        float(has_benzene_ring),
    )


def _requested_property_group_summary(all_molecules: pd.DataFrame) -> pd.DataFrame:
    if all_molecules.empty:
        return pd.DataFrame()
    df = all_molecules.copy()
    smiles_col = "canonical_smiles" if "canonical_smiles" in df.columns else "smiles"
    valid_mask = df.get("rdkit_valid", pd.Series([np.nan] * len(df))) == 1
    flags = df.loc[valid_mask, smiles_col].map(_ring_flags_from_smiles)
    parsed = pd.DataFrame(
        flags.tolist(),
        index=df.loc[valid_mask].index,
        columns=["computed_logp", "has_three_member_ring", "has_benzene_ring"],
    )
    for column in parsed.columns:
        df[column] = np.nan
        df.loc[parsed.index, column] = parsed[column]

    rows = []
    for (epoch, source), group in df.groupby(["epoch", "source"], dropna=False):
        valid = group[group["rdkit_valid"] == 1] if "rdkit_valid" in group.columns else group
        docked = group[group["docking_success"] == 1] if "docking_success" in group.columns else group
        parsed_valid = valid[valid["computed_logp"].notna()]
        row = {
            "epoch": int(epoch),
            "source": source,
            "n_molecules": int(len(group)),
            "n_valid": int(len(valid)),
            "n_parsed_valid": int(len(parsed_valid)),
            "valid_ratio": group["rdkit_valid"].mean() if "rdkit_valid" in group.columns else np.nan,
            "docking_raw_mean": docked["docking_raw"].mean() if "docking_raw" in docked.columns else np.nan,
            "logp_mean": parsed_valid["computed_logp"].mean(),
            "sa_raw_mean": valid["sa_raw"].mean() if "sa_raw" in valid.columns else np.nan,
            "three_member_ring_ratio": parsed_valid["has_three_member_ring"].mean(),
            "benzene_ring_ratio": parsed_valid["has_benzene_ring"].mean(),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["epoch", "source"])


def _plot_training_reward(summary: pd.DataFrame, partial90_summary: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax = axes[0]
    x = summary["epoch"]
    if "reward_for_advantage_mean" in summary.columns:
        ax.plot(x, summary["reward_for_advantage_mean"], label="DePPA reward for PPO", lw=2)
    if "reward_total_mean" in summary.columns:
        reward_total = summary["reward_total_mean"]
        if (
            "reward_for_advantage_mean" not in summary.columns
            or not np.allclose(
                reward_total.fillna(0).to_numpy(),
                summary["reward_for_advantage_mean"].fillna(0).to_numpy(),
            )
        ):
            ax.plot(x, reward_total, label="summary reward_total_mean", lw=1.8)
    if not partial90_summary.empty:
        p = partial90_summary.iloc[0]
        if "reward_for_advantage_mean" in p.index:
            ax.scatter([90], [p["reward_for_advantage_mean"]], s=70, marker="x", label="epoch90 partial")
    ax.axhline(0, color="#999999", lw=0.8)
    ax.set_ylabel("reward")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)

    ax = axes[1]
    for col, label in [
        ("reconstruction_success_ratio", "reconstruction"),
        ("docking_success_ratio_total", "docking"),
        ("molecule_rdkit_valid_ratio", "RDKit valid"),
    ]:
        if col in summary.columns:
            ax.plot(x, summary[col], label=label, lw=2)
    if not partial90_summary.empty:
        p = partial90_summary.iloc[0]
        if "molecule_rdkit_valid_ratio" in p.index:
            ax.scatter([90], [p["molecule_rdkit_valid_ratio"]], s=70, marker="x", color="#2ca02c")
    ax.set_ylim(0.78, 1.02)
    ax.set_xlabel("epoch")
    ax.set_ylabel("ratio")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    fig.suptitle("DiffGui DePPA full 20260626: reward and validity")
    fig.tight_layout()
    path = ANALYSIS_DIR / "01_training_reward_validity.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_docking_quality(epoch_molecule_summary: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    data = epoch_molecule_summary.sort_values("epoch")
    official = data[data["source"] == "official_merged"]
    x = official["epoch"]
    ax = axes[0]
    ax.plot(x, official["docking_raw_mean"], color="#1f77b4", lw=2, label="mean docking raw")
    if (data["source"] == "rank_partial_unmerged").any():
        p = data[data["source"] == "rank_partial_unmerged"].iloc[0]
        ax.scatter([90], [p["docking_raw_mean"]], marker="x", s=70, color="#1f77b4")
    ax.set_ylabel("docking raw")
    ax.invert_yaxis()
    ax.legend()
    ax.grid(alpha=0.25)

    ax = axes[1]
    if "qed_raw_mean" in official.columns:
        ax.plot(x, official["qed_raw_mean"], label="QED", lw=2)
    if "sa_raw_mean" in official.columns:
        ax.plot(x, official["sa_raw_mean"], label="SA raw", lw=2)
    ax.set_ylabel("drug-likeness")
    ax.legend()
    ax.grid(alpha=0.25)

    ax = axes[2]
    if "ligand_efficiency_raw_mean" in official.columns:
        ax.plot(x, official["ligand_efficiency_raw_mean"], label="ligand efficiency raw", lw=2)
    ax.set_xlabel("epoch")
    ax.set_ylabel("LE")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.suptitle("Docking and molecule-level quality")
    fig.tight_layout()
    path = ANALYSIS_DIR / "02_docking_and_quality.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_descriptor_diversity(summary: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    x = summary["epoch"]
    ax = axes[0]
    for col, label in [
        ("descriptor_mmd_vs_baseline", "MMD vs baseline"),
        ("ecfp4_mean_pairwise_similarity", "ECFP4 pairwise sim"),
        ("fcfp4_mean_pairwise_similarity", "FCFP4 pairwise sim"),
    ]:
        if col in summary.columns:
            ax.plot(x, summary[col], label=label, lw=2)
    ax.set_ylabel("distribution")
    ax.legend()
    ax.grid(alpha=0.25)

    ax = axes[1]
    for col, label in [
        ("duplicate_ratio", "duplicate ratio"),
        ("fragment_multi_ratio", "multi-fragment ratio"),
    ]:
        if col in summary.columns:
            ax.plot(x, summary[col], label=label, lw=2)
    ax.set_ylabel("ratio")
    ax.legend()
    ax.grid(alpha=0.25)

    ax = axes[2]
    for col, label in [
        ("ring_count_mean", "rings"),
        ("aromatic_ring_mean", "aromatic rings"),
        ("three_member_ring_total", "3-member total"),
        ("valence_violation_total", "valence violations"),
    ]:
        if col in summary.columns:
            ax.plot(x, summary[col], label=label, lw=2)
    ax.set_xlabel("epoch")
    ax.set_ylabel("structure facts")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.suptitle("Descriptor drift and structural diagnostics")
    fig.tight_layout()
    path = ANALYSIS_DIR / "03_descriptor_drift_diversity.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_ppo(summary: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    x = summary["epoch"]
    ax = axes[0]
    if "PPO_ApproxKL" in summary.columns:
        ax.plot(x, summary["PPO_ApproxKL"], lw=2, color="#d62728", label="ApproxKL")
        ax.set_yscale("symlog", linthresh=1.0)
    ax.set_ylabel("ApproxKL")
    ax.legend()
    ax.grid(alpha=0.25)

    ax = axes[1]
    if "ClipFrac" in summary.columns:
        ax.plot(x, summary["ClipFrac"], lw=2, label="ClipFrac")
    if "RatioMean" in summary.columns:
        ax.plot(x, summary["RatioMean"], lw=2, label="RatioMean")
    ax.axhline(1.0, color="#888888", lw=0.8)
    ax.set_ylabel("PPO ratio")
    ax.legend()
    ax.grid(alpha=0.25)

    ax = axes[2]
    for col, label in [
        ("rollout_seconds", "rollout"),
        ("reward_seconds", "reward"),
        ("ppo_seconds", "PPO"),
    ]:
        if col in summary.columns:
            ax.plot(x, summary[col], label=label, lw=2)
    ax.set_xlabel("epoch")
    ax.set_ylabel("seconds")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.suptitle("PPO stability and runtime")
    fig.tight_layout()
    path = ANALYSIS_DIR / "04_ppo_stability_runtime.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_window_summary(window_summary: pd.DataFrame) -> Path:
    cols = [
        ("reward_for_advantage_mean_mean", "DePPA reward"),
        ("docking_raw_mean_mean", "docking raw"),
        ("molecule_rdkit_valid_ratio_mean", "RDKit valid"),
        ("descriptor_mmd_vs_baseline_mean", "MMD"),
    ]
    existing = [(col, label) for col, label in cols if col in window_summary.columns]
    n = len(existing)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.4), squeeze=False)
    labels = window_summary["window"].tolist()
    for ax, (col, label) in zip(axes[0], existing):
        ax.bar(labels, window_summary[col], color="#4c78a8")
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Window means by training stage")
    fig.tight_layout()
    path = ANALYSIS_DIR / "05_window_summary.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_deppa_component_trends(component_summary: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    official = component_summary[component_summary["source"] == "official_merged"].sort_values("epoch")
    partial = component_summary[component_summary["source"] != "official_merged"].sort_values("epoch")

    ax = axes[0]
    for col, label in [
        ("deppa_qed_normal_mean", "QED normal"),
        ("deppa_sa_normal_mean", "SA normal"),
        ("deppa_vina_score_normal_mean", "vina normal"),
        ("deppa_distance_normal_mean", "distance normal"),
    ]:
        if col in official.columns:
            ax.plot(official["epoch"], official[col], lw=2, label=label)
            if not partial.empty and col in partial.columns:
                ax.scatter(partial["epoch"], partial[col], marker="x", s=70)
    ax.set_ylabel("normalized component")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)

    ax = axes[1]
    for col, label in [
        ("qed_raw_mean", "QED raw"),
        ("sa_raw_mean", "SA raw"),
        ("deppa_distance_raw_mean", "distance raw"),
    ]:
        if col in official.columns:
            ax.plot(official["epoch"], official[col], lw=2, label=label)
            if not partial.empty and col in partial.columns:
                ax.scatter(partial["epoch"], partial[col], marker="x", s=70)
    ax.set_ylabel("raw QED / SA / distance")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)

    ax = axes[2]
    if "docking_raw_mean" in official.columns:
        ax.plot(official["epoch"], official["docking_raw_mean"], lw=2, label="docking raw")
        if not partial.empty and "docking_raw_mean" in partial.columns:
            ax.scatter(partial["epoch"], partial["docking_raw_mean"], marker="x", s=70)
    if "deppa_vina_score_raw_mean" in official.columns:
        aligned_vina = -official["deppa_vina_score_raw_mean"]
        ax.plot(official["epoch"], aligned_vina, lw=1.5, ls="--", label="-DePPA vina raw")
        if not partial.empty and "deppa_vina_score_raw_mean" in partial.columns:
            ax.scatter(partial["epoch"], -partial["deppa_vina_score_raw_mean"], marker="x", s=70)
    ax.invert_yaxis()
    ax.set_xlabel("epoch")
    ax.set_ylabel("aligned vina score")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)

    fig.suptitle("DePPA component trends")
    fig.tight_layout()
    path = ANALYSIS_DIR / "06_deppa_component_trends.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_property_boxplots(all_molecules: pd.DataFrame) -> Path:
    selected_epochs = [1, 10, 25, 59, 79, 89, 90]
    properties = [
        ("reward_for_advantage", "DePPA reward"),
        ("reward_total", "raw scored reward"),
        ("docking_raw", "docking raw"),
        ("qed_raw", "QED"),
        ("sa_raw", "SA"),
        ("num_atoms", "num atoms"),
    ]
    data = all_molecules[all_molecules["epoch"].isin(selected_epochs)].copy()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=False)
    for ax, (column, label) in zip(axes.flat, properties):
        grouped = []
        labels = []
        for epoch in selected_epochs:
            if column not in data.columns:
                continue
            values = data.loc[data["epoch"] == epoch, column].dropna()
            if values.empty:
                continue
            grouped.append(values.to_numpy())
            labels.append(str(epoch))
        if grouped:
            ax.boxplot(grouped, tick_labels=labels, showfliers=False, patch_artist=True)
            if column == "docking_raw":
                ax.invert_yaxis()
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(label)
        ax.set_xlabel("epoch")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Molecule-level property distributions at selected epochs")
    fig.tight_layout()
    path = ANALYSIS_DIR / "07_property_distribution_boxplots.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_failure_modes(failure_by_epoch: pd.DataFrame) -> Path:
    data = failure_by_epoch.copy()
    if data.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "no failure-mode data", ha="center", va="center")
    else:
        non_success = data[data["failure_mode"] != "success"]
        top_failures = (
            non_success.groupby("failure_mode")["count"].sum().sort_values(ascending=False).head(5).index
        )
        keep_modes = ["success", *top_failures.tolist()]
        data["failure_mode_plot"] = data["failure_mode"].where(
            data["failure_mode"].isin(keep_modes), "other_failure"
        )
        compact = data.groupby(["epoch", "failure_mode_plot"], as_index=False)["count"].sum()
        totals = compact.groupby("epoch")["count"].transform("sum")
        compact["ratio"] = compact["count"] / totals
        pivot = (
            compact.pivot_table(
                index="epoch", columns="failure_mode_plot", values="ratio", aggfunc="sum", fill_value=0.0
            )
            .sort_index()
            .sort_index(axis=1)
        )
        ordered = [mode for mode in keep_modes if mode in pivot.columns]
        if "other_failure" in pivot.columns:
            ordered.append("other_failure")
        pivot = pivot[ordered]
        fig, ax = plt.subplots(figsize=(12, 5))
        bottom = np.zeros(len(pivot))
        colors = plt.get_cmap("tab20").colors
        for idx, column in enumerate(pivot.columns):
            ax.bar(pivot.index, pivot[column], bottom=bottom, label=column, color=colors[idx % len(colors)])
            bottom += pivot[column].to_numpy()
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("epoch")
        ax.set_ylabel("ratio")
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Failure-mode composition by epoch")
    fig.tight_layout()
    path = ANALYSIS_DIR / "08_failure_modes.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_rank_variability(rank_epoch: pd.DataFrame) -> Path:
    metrics = [
        ("reward_for_advantage_mean", "rank mean DePPA reward", "viridis"),
        ("docking_raw_mean", "rank mean docking raw", "magma_r"),
        ("rdkit_valid_ratio", "rank RDKit-valid ratio", "YlGnBu"),
    ]
    official = rank_epoch[rank_epoch["source"] == "official_merged"].copy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 7), sharey=True)
    for ax, (column, title, cmap) in zip(axes, metrics):
        if official.empty or column not in official.columns:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            continue
        pivot = official.pivot_table(index="epoch", columns="rank", values=column, aggfunc="mean").sort_index()
        matrix = np.ma.masked_invalid(pivot.to_numpy(dtype=float))
        im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("rank")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([str(int(rank)) if pd.notna(rank) else "NA" for rank in pivot.columns])
        y_positions = np.linspace(0, len(pivot.index) - 1, min(10, len(pivot.index)), dtype=int)
        ax.set_yticks(y_positions)
        ax.set_yticklabels([str(int(pivot.index[pos])) for pos in y_positions])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axes[0].set_ylabel("epoch")
    fig.suptitle("Rank-level variability across official merged epochs")
    fig.tight_layout()
    path = ANALYSIS_DIR / "09_rank_variability_heatmaps.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_topk(topk_summary: pd.DataFrame) -> Path:
    data = topk_summary.sort_values("epoch").copy()
    official = data[data["source"] == "official_merged"]
    partial = data[data["source"] != "official_merged"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

    ax = axes[0]
    for col, label in [("best_reward", "best reward"), ("top5_reward_mean", "top5 reward mean")]:
        if col in official.columns:
            ax.plot(official["epoch"], official[col], lw=2, label=label)
            if not partial.empty and col in partial.columns:
                ax.scatter(partial["epoch"], partial[col], marker="x", s=70)
    ax.set_ylabel("reward")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)

    ax = axes[1]
    for col, label in [("best_docking_raw", "best docking"), ("top5_docking_raw_mean", "top5 docking mean")]:
        if col in official.columns:
            ax.plot(official["epoch"], official[col], lw=2, label=label)
            if not partial.empty and col in partial.columns:
                ax.scatter(partial["epoch"], partial[col], marker="x", s=70)
    ax.invert_yaxis()
    ax.set_ylabel("docking raw")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)

    ax = axes[2]
    for col, label in [
        ("dock_le_-12_count", "dock <= -12"),
        ("dock_le_-14_count", "dock <= -14"),
        ("balanced_qed065_sa5_dock10_count", "QED>=0.65 & SA<=5 & dock<=-10"),
    ]:
        if col in official.columns:
            ax.plot(official["epoch"], official[col], lw=2, label=label)
            if not partial.empty and col in partial.columns:
                ax.scatter(partial["epoch"], partial[col], marker="x", s=70)
    ax.set_xlabel("epoch")
    ax.set_ylabel("count / epoch")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)

    fig.suptitle("Top-k and threshold enrichment")
    fig.tight_layout()
    path = ANALYSIS_DIR / "10_topk_enrichment.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_ppo_timestep_heatmaps(ppo_selected: pd.DataFrame) -> Path:
    metrics = [
        ("approx_kl_mean", "log10 mean ApproxKL", "inferno"),
        ("clipfrac_mean", "ClipFrac mean", "YlOrRd"),
        ("ratio_mean", "RatioMean", "coolwarm"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, (column, title, cmap) in zip(axes, metrics):
        if ppo_selected.empty or column not in ppo_selected.columns:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            continue
        pivot = ppo_selected.pivot_table(
            index="epoch", columns="timestep_mean", values=column, aggfunc="mean"
        ).sort_index()
        values = pivot.to_numpy(dtype=float)
        if column == "approx_kl_mean":
            values = np.log10(np.clip(values, 1e-6, None))
        matrix = np.ma.masked_invalid(values)
        im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("timestep")
        x_positions = np.linspace(0, len(pivot.columns) - 1, min(8, len(pivot.columns)), dtype=int)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(int(pivot.columns[pos])) for pos in x_positions], rotation=45)
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels([str(int(epoch)) for epoch in pivot.index])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axes[0].set_ylabel("epoch")
    fig.suptitle("PPO timestep diagnostics at selected epochs")
    fig.tight_layout()
    path = ANALYSIS_DIR / "11_ppo_timestep_heatmaps.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_property_epoch_quantile_trends(property_quantiles: pd.DataFrame) -> Path:
    properties = [
        ("qed_raw", "QED"),
        ("sa_raw", "SA"),
        ("num_atoms", "num atoms"),
        ("docking_raw", "docking raw"),
        ("reward_for_advantage", "DePPA reward"),
        ("reward_total", "raw scored reward"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for ax, (prop, title) in zip(axes.flat, properties):
        data = property_quantiles[property_quantiles["property"] == prop].sort_values("epoch")
        official = data[data["source"] == "official_merged"]
        partial = data[data["source"] != "official_merged"]
        if official.empty:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            continue
        x = official["epoch"].to_numpy(dtype=float)
        mean = official["mean"].to_numpy(dtype=float)
        q25 = official["q25"].to_numpy(dtype=float)
        q75 = official["q75"].to_numpy(dtype=float)
        ax.plot(x, mean, lw=2, label="mean")
        ax.fill_between(x, q25, q75, alpha=0.22, label="IQR")
        if not partial.empty:
            ax.scatter(partial["epoch"], partial["mean"], marker="x", s=70, label="epoch90 partial")
        if prop == "docking_raw":
            ax.invert_yaxis()
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[1, 0].set_xlabel("epoch")
    axes[1, 1].set_xlabel("epoch")
    axes[1, 2].set_xlabel("epoch")
    axes[0, 0].legend(loc="best")
    fig.suptitle("Molecule property trends by epoch: mean and IQR")
    fig.tight_layout()
    path = ANALYSIS_DIR / "12_property_epoch_quantile_trends.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_descriptor_epoch_trends(summary: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(3, 2, figsize=(15, 11), sharex=True)
    x = summary["epoch"]

    ax = axes[0, 0]
    for col, label in [
        ("molecule_rdkit_valid_ratio", "RDKit valid"),
        ("docking_success_ratio_total", "docking success"),
        ("duplicate_ratio", "duplicate"),
        ("fragment_multi_ratio", "multi-fragment"),
    ]:
        if col in summary.columns:
            ax.plot(x, summary[col], lw=2, label=label)
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("validity and uniqueness")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    for col, label in [
        ("descriptor_mmd_vs_baseline", "MMD vs baseline"),
        ("ecfp4_mean_pairwise_similarity", "ECFP4 pairwise sim"),
        ("fcfp4_mean_pairwise_similarity", "FCFP4 pairwise sim"),
    ]:
        if col in summary.columns:
            ax.plot(x, summary[col], lw=2, label=label)
    ax.set_title("distribution drift and similarity")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    for col, label in [
        ("logp_mean", "logP"),
        ("tpsa_mean", "TPSA / 10"),
        ("hetero_atom_mean", "hetero atoms"),
    ]:
        if col not in summary.columns:
            continue
        values = summary[col] / 10.0 if col == "tpsa_mean" else summary[col]
        ax.plot(x, values, lw=2, label=label)
    ax.set_title("physicochemical means")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    if "carbon_fraction" in summary.columns:
        ax.plot(x, summary["carbon_fraction"], lw=2, label="carbon fraction")
    ax.set_ylim(0, 1)
    ax.set_title("element composition")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)

    ax = axes[2, 0]
    for col, label in [
        ("ring_count_mean", "ring count"),
        ("aromatic_ring_mean", "aromatic rings"),
    ]:
        if col in summary.columns:
            ax.plot(x, summary[col], lw=2, label=label)
    ax.set_xlabel("epoch")
    ax.set_title("ring means")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)

    ax = axes[2, 1]
    for col, label in [
        ("three_member_ring_total", "3-member rings total"),
        ("valence_violation_total", "valence violations"),
    ]:
        if col in summary.columns:
            ax.plot(x, summary[col], lw=2, label=label)
    ax.set_xlabel("epoch")
    ax.set_title("structural warning counts")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)

    fig.suptitle("Descriptor and structural property trends by epoch")
    fig.tight_layout()
    path = ANALYSIS_DIR / "13_descriptor_epoch_trends.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_threshold_epoch_trends(topk_summary: pd.DataFrame) -> Path:
    data = topk_summary.sort_values("epoch")
    official = data[data["source"] == "official_merged"]
    partial = data[data["source"] != "official_merged"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

    ax = axes[0]
    if "valid_docked_ratio" in official.columns:
        ax.plot(official["epoch"], official["valid_docked_ratio"], lw=2, label="valid + docked ratio")
        if not partial.empty and "valid_docked_ratio" in partial.columns:
            ax.scatter(partial["epoch"], partial["valid_docked_ratio"], marker="x", s=70)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("ratio")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)

    ax = axes[1]
    for col, label in [
        ("dock_le_-12_count", "dock <= -12"),
        ("dock_le_-14_count", "dock <= -14"),
        ("balanced_qed065_sa5_dock10_count", "QED>=0.65 & SA<=5 & dock<=-10"),
    ]:
        if col in official.columns:
            ax.plot(official["epoch"], official[col], lw=2, label=label)
            if not partial.empty and col in partial.columns:
                ax.scatter(partial["epoch"], partial[col], marker="x", s=70)
    ax.set_ylabel("count / epoch")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)

    ax = axes[2]
    ax2 = ax.twinx()
    if "top5_reward_mean" in official.columns:
        reward_line = ax.plot(
            official["epoch"], official["top5_reward_mean"], lw=2, color="#1f77b4", label="top5 reward mean"
        )
        if not partial.empty and "top5_reward_mean" in partial.columns:
            ax.scatter(partial["epoch"], partial["top5_reward_mean"], marker="x", s=70, color="#1f77b4")
    else:
        reward_line = []
    if "top5_docking_raw_mean" in official.columns:
        docking_line = ax2.plot(
            official["epoch"],
            -official["top5_docking_raw_mean"],
            lw=2,
            color="#ff7f0e",
            label="-top5 docking raw mean",
        )
        if not partial.empty and "top5_docking_raw_mean" in partial.columns:
            ax2.scatter(partial["epoch"], -partial["top5_docking_raw_mean"], marker="x", s=70, color="#ff7f0e")
    else:
        docking_line = []
    ax.set_xlabel("epoch")
    ax.set_ylabel("top5 reward", color="#1f77b4")
    ax2.set_ylabel("-top5 docking raw", color="#ff7f0e")
    lines = reward_line + docking_line
    ax.legend(lines, [line.get_label() for line in lines], loc="best")
    ax.grid(alpha=0.25)

    fig.suptitle("Quality guardrail and top-k threshold trends by epoch")
    fig.tight_layout()
    path = ANALYSIS_DIR / "14_threshold_epoch_trends.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_requested_property_group(group_summary: pd.DataFrame) -> Path:
    panels = [
        ("valid_ratio", "Validity", "ratio", False),
        ("docking_raw_mean", "Docking raw", "score", True),
        ("logp_mean", "logP", "mean", False),
        ("sa_raw_mean", "SA", "mean", False),
        ("three_member_ring_ratio", "Three-member ring ratio", "ratio", False),
        ("benzene_ring_ratio", "Benzene-ring-containing ratio", "ratio", False),
    ]
    official = group_summary[group_summary["source"] == "official_merged"].sort_values("epoch")
    partial = group_summary[group_summary["source"] != "official_merged"].sort_values("epoch")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for ax, (column, title, ylabel, invert_y) in zip(axes.flat, panels):
        if official.empty or column not in official.columns:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            continue
        ax.plot(official["epoch"], official[column], lw=2)
        if not partial.empty and column in partial.columns:
            ax.scatter(partial["epoch"], partial[column], marker="x", s=80)
        if "ratio" in column:
            ax.set_ylim(-0.03, 1.03)
        if invert_y:
            ax.invert_yaxis()
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    for ax in axes[1]:
        ax.set_xlabel("epoch")
    fig.suptitle("Requested property group: validity, docking, logP, SA, 3-ring and benzene ratios")
    fig.tight_layout()
    path = ANALYSIS_DIR / "15_requested_validity_docking_logp_sa_ring_ratios.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _failure_context() -> str:
    if not MAIN_LOG.exists():
        return "main log not found"
    text = MAIN_LOG.read_text(errors="replace")
    marker = "Invariant Violation"
    index = text.rfind(marker)
    if index < 0:
        return text[-3000:]
    start = max(0, index - 400)
    end = min(len(text), index + 2600)
    return text[start:end]


def _write_report(
    summary: pd.DataFrame,
    window_summary: pd.DataFrame,
    key_epochs: pd.DataFrame,
    partial90_summary: pd.DataFrame,
    run_comparison: pd.DataFrame,
    top_deppa: pd.DataFrame,
    top_docking: pd.DataFrame,
    ppo_extremes: pd.DataFrame,
    component_summary: pd.DataFrame,
    property_quantiles: pd.DataFrame,
    failure_totals: pd.DataFrame,
    rank_overall: pd.DataFrame,
    topk_epoch_summary: pd.DataFrame,
    figure_paths: list[Path],
) -> None:
    official_last = summary.loc[summary["epoch"].idxmax()]
    best_deppa = summary.loc[summary["reward_for_advantage_mean"].idxmax()]
    best_dock = summary.loc[summary["docking_raw_mean"].idxmin()]
    last_window = window_summary[window_summary["window"] == "last10_80_89"].iloc[0]
    partial_line = ""
    if not partial90_summary.empty:
        p = partial90_summary.iloc[0]
        partial_line = (
            f"- epoch 90 rank-level partial: {int(p['molecule_records_total'])} molecules, "
            f"RDKit valid {_fmt(p['molecule_rdkit_valid_ratio'])}, "
            f"docking success {_fmt(p['molecule_docking_success_ratio'])}, "
            f"DePPA mean {_fmt(p['reward_for_advantage_mean'])}; not merged because SDF writing failed.\n"
        )

    focus_epochs = [1, 25, 59, 79, 89, 90]
    topk_focus = (
        topk_epoch_summary[topk_epoch_summary["epoch"].isin(focus_epochs)]
        if not topk_epoch_summary.empty and "epoch" in topk_epoch_summary.columns
        else pd.DataFrame()
    )
    component_focus = (
        component_summary[component_summary["epoch"].isin(focus_epochs)]
        if not component_summary.empty and "epoch" in component_summary.columns
        else pd.DataFrame()
    )
    quantile_focus = pd.DataFrame()
    if not property_quantiles.empty:
        quantile_focus = property_quantiles[
            property_quantiles["epoch"].isin([1, 59, 79, 89, 90])
            & property_quantiles["property"].isin(["reward_for_advantage", "docking_raw", "qed_raw", "sa_raw"])
        ]
    rank_focus = rank_overall.copy()
    if not rank_focus.empty:
        rank_focus = rank_focus.sort_values(["source", "rank"])

    figure_md = "\n".join(f"![{path.stem}]({path.name})" for path in figure_paths)
    report = f"""# DiffGui DePPA Full 2026-06-26 Training Analysis

## Scope

- Main run: `{_rel(MAIN_RUN)}`
- Launch log: `{_rel(MAIN_LOG)}`
- Config: `{_rel(ROOT / 'launch' / 'diffgui_deppa_full_lr1e5_wd1e4_b8x8_t200_100ep_20260626.yaml')}`
- Planned shape: 8 GPUs, batch/rank 8, global batch 64, 100 epochs, 200 timesteps, full fine-tune (`freeze.mode=none`), lr `1e-5`, weight decay `1e-4`.
- Reliable merged summary: epoch 1-89. Epoch 90 has rank-level molecule/reward artifacts but no merged epoch row.

## Main Conclusions

- The run did not complete 100 epochs. It failed at `2026-06-28 02:07:28` while merging epoch 90 SDFs. The immediate RDKit error was `Invariant Violation: no eligible neighbors for chiral center`; the newer crash in this run was amplified by an exception handler that references `Chem.AtomValenceException`, which is absent in the active RDKit 2018.09.1 environment.
- The latest complete checkpoint is epoch 85 (`checkpoints/latest.pt` points to the epoch-85 save). Epochs 86-89 have complete metrics and molecules, but no checkpoint. Epoch 90 has scored rank outputs but is unmerged.
- Training did not produce a monotonic DePPA improvement. Best epoch by `reward_for_advantage_mean` was epoch {int(best_deppa['epoch'])} ({_fmt(best_deppa['reward_for_advantage_mean'])}); final official epoch 89 was {_fmt(official_last['reward_for_advantage_mean'])}. The last-10-epoch mean was {_fmt(last_window['reward_for_advantage_mean_mean'])}.
- Mean docking did improve intermittently but not stably. Best mean docking was epoch {int(best_dock['epoch'])} with raw docking {_fmt(best_dock['docking_raw_mean'])}; epoch 89 was {_fmt(official_last['docking_raw_mean'])}.
- Validity stayed usable but not perfect. Epoch 89 RDKit-valid ratio was {_fmt(official_last['molecule_rdkit_valid_ratio'])}, docking success total was {_fmt(official_last['docking_success_ratio_total'])}, and duplicate ratio was {_fmt(official_last['duplicate_ratio'])}.
- PPO behavior looks unstable for a full-model update: epoch-level `PPO_ApproxKL` has repeated spikes, and the epoch-89 step/timestep table contains very large timestep-local KL outliers. This supports reducing update aggressiveness or re-adding a KL/clip gate before treating this route as stable.
{partial_line}
## Key Epochs

{_md_table(_select_existing(key_epochs, ['label', 'epoch', 'source', 'reward_for_advantage_mean', 'reward_total_scored_mean', 'docking_raw_mean', 'molecule_rdkit_valid_ratio', 'docking_success_ratio_total', 'duplicate_ratio', 'descriptor_mmd_vs_baseline', 'PPO_ApproxKL', 'ClipFrac']), digits=4)}

## Stage Windows

{_md_table(_select_existing(window_summary, ['window', 'n_epochs', 'reward_for_advantage_mean_mean', 'reward_total_mean_mean', 'docking_raw_mean_mean', 'molecule_rdkit_valid_ratio_mean', 'docking_success_ratio_total_mean', 'duplicate_ratio_mean', 'descriptor_mmd_vs_baseline_mean', 'PPO_ApproxKL_mean', 'ClipFrac_mean']), digits=4)}

## Additional Statistics

Selected epoch top-k and threshold enrichment:

{_md_table(_select_existing(topk_focus, ['epoch', 'source', 'n_molecules', 'n_valid_docked', 'valid_docked_ratio', 'best_reward', 'top5_reward_mean', 'best_docking_raw', 'top5_docking_raw_mean', 'dock_le_-12_count', 'dock_le_-14_count', 'balanced_qed065_sa5_dock10_count']), digits=4)}

Selected epoch DePPA component means:

{_md_table(_select_existing(component_focus, ['epoch', 'source', 'n_molecules', 'reward_for_advantage_mean', 'reward_total_mean', 'deppa_qed_normal_mean', 'deppa_sa_normal_mean', 'deppa_vina_score_normal_mean', 'deppa_distance_normal_mean', 'qed_raw_mean', 'sa_raw_mean', 'docking_raw_mean']), digits=4)}

Molecule-level quantiles for key properties:

{_md_table(_select_existing(quantile_focus, ['epoch', 'source', 'property', 'n', 'mean', 'q10', 'q25', 'median', 'q75', 'q90', 'min', 'max']), digits=4, max_rows=24)}

Overall failure modes:

{_md_table(_select_existing(failure_totals, ['failure_mode', 'count', 'ratio']), digits=4, max_rows=12)}

Rank-level long-run means:

{_md_table(_select_existing(rank_focus, ['source', 'rank', 'n_epochs', 'rdkit_valid_ratio_mean', 'docking_success_ratio_mean', 'reward_for_advantage_mean', 'reward_for_advantage_std', 'docking_raw_mean', 'docking_raw_std', 'duplicate_ratio_mean']), digits=4, max_rows=16)}

## Comparison Runs

The reference runs differ in freeze/batch/checkpoint state, so this is directional, not a controlled ablation.

{_md_table(_select_existing(run_comparison, ['run', 'n_epochs', 'last_epoch', 'final_reward_for_advantage_mean', 'last5_reward_for_advantage_mean', 'final_reward_total_scored_mean', 'final_docking_raw_mean', 'last5_docking_raw_mean', 'final_molecule_rdkit_valid_ratio', 'final_duplicate_ratio', 'best_deppa_epoch', 'best_deppa_reward_for_advantage_mean', 'best_docking_epoch', 'best_docking_raw_mean']), digits=4)}

## PPO Timestep Outliers At Epoch 89

{_md_table(_select_existing(ppo_extremes, ['timestep_mean', 'approx_kl_mean', 'approx_kl_max', 'clipfrac_mean', 'ratio_mean']), digits=4, max_rows=10)}

## Figures

{figure_md}

## Top Molecule Tables

- [`top_molecules_by_deppa_reward.csv`](top_molecules_by_deppa_reward.csv)
- [`top_molecules_by_docking.csv`](top_molecules_by_docking.csv)

Top rows by DePPA reward:

{_md_table(_select_existing(top_deppa, ['source', 'epoch', 'rank', 'reward_for_advantage', 'reward_total', 'docking_raw', 'qed_raw', 'sa_raw', 'canonical_smiles']), digits=4, max_rows=8)}

Top rows by docking:

{_md_table(_select_existing(top_docking, ['source', 'epoch', 'rank', 'reward_for_advantage', 'reward_total', 'docking_raw', 'qed_raw', 'sa_raw', 'canonical_smiles']), digits=4, max_rows=8)}

## Output Files

- [`official_epoch_summary_selected.csv`](official_epoch_summary_selected.csv)
- [`molecule_epoch_summary.csv`](molecule_epoch_summary.csv)
- [`key_epoch_summary.csv`](key_epoch_summary.csv)
- [`window_summary.csv`](window_summary.csv)
- [`run_comparison.csv`](run_comparison.csv)
- [`partial_epoch90_molecules.csv`](partial_epoch90_molecules.csv)
- [`partial_epoch90_summary.csv`](partial_epoch90_summary.csv)
- [`component_epoch_summary.csv`](component_epoch_summary.csv)
- [`molecule_property_quantiles.csv`](molecule_property_quantiles.csv)
- [`failure_mode_by_epoch.csv`](failure_mode_by_epoch.csv)
- [`failure_mode_totals.csv`](failure_mode_totals.csv)
- [`rank_epoch_summary.csv`](rank_epoch_summary.csv)
- [`rank_overall_summary.csv`](rank_overall_summary.csv)
- [`topk_epoch_summary.csv`](topk_epoch_summary.csv)
- [`requested_property_group_epoch_summary.csv`](requested_property_group_epoch_summary.csv)
- [`ppo_timestep_selected_epochs.csv`](ppo_timestep_selected_epochs.csv)
- [`epoch89_ppo_timestep_extremes.csv`](epoch89_ppo_timestep_extremes.csv)
- [`failure_context.txt`](failure_context.txt)

## Practical Recommendation

Do not pick epoch 89/90 just because it is later. For continuation/evaluation, prefer checkpoint epoch 85 as the latest saved state, but select molecules/checkpoints by a small ablation: epoch 25 or 35 for better early DePPA/docking tradeoff, epoch 85 for long-run full-tune behavior, and the prior last3 route for stability comparison. Before another 100-epoch run, fix SDF merge exception handling and add a guard for pathological stereo/write failures; then lower update pressure or gate full-model PPO updates by KL/clip statistics.
"""
    (ANALYSIS_DIR / "report.md").write_text(report, encoding="utf-8")


def _write_work_record(
    figure_paths: list[Path],
    key_epochs: pd.DataFrame,
    window_summary: pd.DataFrame,
    topk_epoch_summary: pd.DataFrame,
    failure_totals: pd.DataFrame,
    rank_overall: pd.DataFrame,
) -> None:
    WORK_RECORD_DIR.mkdir(parents=True, exist_ok=True)
    copied = []
    for path in figure_paths:
        target = WORK_RECORD_DIR / path.name
        shutil.copy2(path, target)
        copied.append(target)
    figure_md = "\n".join(f"![{path.stem}]({path.name})" for path in copied)
    focus_topk = (
        topk_epoch_summary[topk_epoch_summary["epoch"].isin([1, 25, 59, 79, 89, 90])]
        if not topk_epoch_summary.empty and "epoch" in topk_epoch_summary.columns
        else pd.DataFrame()
    )
    rank_table = rank_overall.sort_values(["source", "rank"]) if not rank_overall.empty else pd.DataFrame()
    record = f"""# DiffGui DePPA full 6月26日训练分析

## 背景

分析 2026-06-26 启动的 DiffGui DePPA full fine-tune：8 卡、batch/rank 8、100 epoch、200 timestep、全模型更新、lr=1e-5、weight_decay=1e-4。目标是判断训练是否完成、指标是否有效改善、是否值得续跑或作为正式结果。

## 结论

- 训练没有跑满 100 epoch；第 90 轮合并 SDF 时因 RDKit stereo 写出异常失败，正式 merged summary 到 epoch 89。
- 最新 checkpoint 为 epoch 85；epoch 86-89 有完整统计但没有 checkpoint；epoch 90 只有各 rank 的未合并打分产物。
- DePPA 用于 PPO 的 reward 没有单调上升，后期窗口均值并不优于早期最佳点；full fine-tune 的 PPO KL/clip 也偏不稳。
- 分子有效性整体可用，但仍有 reconstruction/RDKit/stereo 异常，说明下一次正式长跑前应先修 SDF merge 异常处理。

## 关键 epoch

{_md_table(_select_existing(key_epochs, ['label', 'epoch', 'source', 'reward_for_advantage_mean', 'reward_total_scored_mean', 'docking_raw_mean', 'molecule_rdkit_valid_ratio', 'docking_success_ratio_total', 'duplicate_ratio', 'PPO_ApproxKL', 'ClipFrac']), digits=4)}

## 窗口统计

{_md_table(_select_existing(window_summary, ['window', 'n_epochs', 'reward_for_advantage_mean_mean', 'reward_total_mean_mean', 'docking_raw_mean_mean', 'molecule_rdkit_valid_ratio_mean', 'docking_success_ratio_total_mean', 'duplicate_ratio_mean', 'PPO_ApproxKL_mean', 'ClipFrac_mean']), digits=4)}

## 额外统计

Top-k 与阈值富集：

{_md_table(_select_existing(focus_topk, ['epoch', 'source', 'n_molecules', 'n_valid_docked', 'valid_docked_ratio', 'best_reward', 'top5_reward_mean', 'best_docking_raw', 'top5_docking_raw_mean', 'dock_le_-12_count', 'dock_le_-14_count', 'balanced_qed065_sa5_dock10_count']), digits=4)}

失败模式汇总：

{_md_table(_select_existing(failure_totals, ['failure_mode', 'count', 'ratio']), digits=4, max_rows=12)}

Rank 长期均值：

{_md_table(_select_existing(rank_table, ['source', 'rank', 'n_epochs', 'rdkit_valid_ratio_mean', 'docking_success_ratio_mean', 'reward_for_advantage_mean', 'reward_for_advantage_std', 'docking_raw_mean', 'docking_raw_std', 'duplicate_ratio_mean']), digits=4, max_rows=16)}

## 图

{figure_md}

## 文件

- 分析报告：`{_rel(ANALYSIS_DIR / 'report.md')}`
- 主要 CSV：`{_rel(ANALYSIS_DIR / 'key_epoch_summary.csv')}`，`{_rel(ANALYSIS_DIR / 'window_summary.csv')}`，`{_rel(ANALYSIS_DIR / 'run_comparison.csv')}`，`{_rel(ANALYSIS_DIR / 'topk_epoch_summary.csv')}`，`{_rel(ANALYSIS_DIR / 'failure_mode_totals.csv')}`
- 崩溃上下文：`{_rel(ANALYSIS_DIR / 'failure_context.txt')}`
"""
    WORK_RECORD_PATH.write_text(record, encoding="utf-8")


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    WORK_RECORD_DIR.mkdir(parents=True, exist_ok=True)

    summary = _load_official_summary()
    selected_cols = ["epoch", "status"] + [col for col in NUMERIC_COLS if col in summary.columns] + [
        "checkpoint"
    ]
    selected = _select_existing(summary, selected_cols)
    selected.to_csv(ANALYSIS_DIR / "official_epoch_summary_selected.csv", index=False)

    molecules = _load_epoch_molecules(summary)
    partial90 = _load_partial_epoch90()
    partial90.to_csv(ANALYSIS_DIR / "partial_epoch90_molecules.csv", index=False)
    all_molecules = _combine_molecules(molecules, partial90)

    partial90_summary = _summarize_partial_molecules(partial90)
    partial90_summary.to_csv(ANALYSIS_DIR / "partial_epoch90_summary.csv", index=False)

    molecule_epoch_summary = _epoch_molecule_summary(molecules, partial90)
    molecule_epoch_summary.to_csv(ANALYSIS_DIR / "molecule_epoch_summary.csv", index=False)

    component_summary = _component_epoch_summary(all_molecules)
    component_summary.to_csv(ANALYSIS_DIR / "component_epoch_summary.csv", index=False)

    property_quantiles = _molecule_property_quantiles(all_molecules)
    property_quantiles.to_csv(ANALYSIS_DIR / "molecule_property_quantiles.csv", index=False)

    failure_by_epoch, failure_totals = _failure_mode_stats(all_molecules)
    failure_by_epoch.to_csv(ANALYSIS_DIR / "failure_mode_by_epoch.csv", index=False)
    failure_totals.to_csv(ANALYSIS_DIR / "failure_mode_totals.csv", index=False)

    rank_epoch, rank_overall = _rank_epoch_summary(all_molecules)
    rank_epoch.to_csv(ANALYSIS_DIR / "rank_epoch_summary.csv", index=False)
    rank_overall.to_csv(ANALYSIS_DIR / "rank_overall_summary.csv", index=False)

    topk_epoch_summary = _topk_epoch_summary(all_molecules)
    topk_epoch_summary.to_csv(ANALYSIS_DIR / "topk_epoch_summary.csv", index=False)

    requested_property_group = _requested_property_group_summary(all_molecules)
    requested_property_group.to_csv(ANALYSIS_DIR / "requested_property_group_epoch_summary.csv", index=False)

    ppo_selected = _selected_ppo_timestep_summary([1, 25, 59, 79, 89])
    ppo_selected.to_csv(ANALYSIS_DIR / "ppo_timestep_selected_epochs.csv", index=False)

    window_summary = _window_summary(summary)
    window_summary.to_csv(ANALYSIS_DIR / "window_summary.csv", index=False)

    key_epochs = _key_epoch_summary(summary, partial90_summary)
    key_epochs.to_csv(ANALYSIS_DIR / "key_epoch_summary.csv", index=False)

    run_comparison = _reference_comparison()
    run_comparison.to_csv(ANALYSIS_DIR / "run_comparison.csv", index=False)

    top_deppa, top_docking = _top_molecules(molecules, partial90)
    top_deppa.to_csv(ANALYSIS_DIR / "top_molecules_by_deppa_reward.csv", index=False)
    top_docking.to_csv(ANALYSIS_DIR / "top_molecules_by_docking.csv", index=False)

    ppo_extremes = _ppo_timestep_extremes(epoch=89)
    ppo_extremes.to_csv(ANALYSIS_DIR / "epoch89_ppo_timestep_extremes.csv", index=False)

    (ANALYSIS_DIR / "failure_context.txt").write_text(_failure_context(), encoding="utf-8")

    figures = [
        _plot_training_reward(summary, partial90_summary),
        _plot_docking_quality(molecule_epoch_summary),
        _plot_descriptor_diversity(summary),
        _plot_ppo(summary),
        _plot_window_summary(window_summary),
        _plot_deppa_component_trends(component_summary),
        _plot_property_boxplots(all_molecules),
        _plot_failure_modes(failure_by_epoch),
        _plot_rank_variability(rank_epoch),
        _plot_topk(topk_epoch_summary),
        _plot_ppo_timestep_heatmaps(ppo_selected),
        _plot_property_epoch_quantile_trends(property_quantiles),
        _plot_descriptor_epoch_trends(summary),
        _plot_threshold_epoch_trends(topk_epoch_summary),
        _plot_requested_property_group(requested_property_group),
    ]
    _write_report(
        summary=summary,
        window_summary=window_summary,
        key_epochs=key_epochs,
        partial90_summary=partial90_summary,
        run_comparison=run_comparison,
        top_deppa=top_deppa,
        top_docking=top_docking,
        ppo_extremes=ppo_extremes,
        component_summary=component_summary,
        property_quantiles=property_quantiles,
        failure_totals=failure_totals,
        rank_overall=rank_overall,
        topk_epoch_summary=topk_epoch_summary,
        figure_paths=figures,
    )
    _write_work_record(
        figures,
        key_epochs,
        window_summary,
        topk_epoch_summary,
        failure_totals,
        rank_overall,
    )

    manifest = {
        "main_run": _rel(MAIN_RUN),
        "analysis_dir": _rel(ANALYSIS_DIR),
        "work_record": _rel(WORK_RECORD_PATH),
        "official_epochs": [int(summary["epoch"].min()), int(summary["epoch"].max())],
        "partial_epoch90_rows": int(len(partial90)),
        "figures": [_rel(path) for path in figures],
        "csv_outputs": [
            _rel(ANALYSIS_DIR / name)
            for name in [
                "official_epoch_summary_selected.csv",
                "molecule_epoch_summary.csv",
                "key_epoch_summary.csv",
                "window_summary.csv",
                "run_comparison.csv",
                "partial_epoch90_molecules.csv",
                "partial_epoch90_summary.csv",
                "component_epoch_summary.csv",
                "molecule_property_quantiles.csv",
                "failure_mode_by_epoch.csv",
                "failure_mode_totals.csv",
                "rank_epoch_summary.csv",
                "rank_overall_summary.csv",
                "topk_epoch_summary.csv",
                "requested_property_group_epoch_summary.csv",
                "ppo_timestep_selected_epochs.csv",
                "epoch89_ppo_timestep_extremes.csv",
            ]
        ],
    }
    (ANALYSIS_DIR / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
