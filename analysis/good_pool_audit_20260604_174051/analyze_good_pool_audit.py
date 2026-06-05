from __future__ import annotations

"""Summarize the 2026-06-04 GoodMoleculePool audit run."""

import ast
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


RUN_V1 = Path("OUT/good_pool_audit_260604_20260604_173430")
RUN_V2 = Path("OUT/good_pool_audit_260604_v2_20260604_174051")
OUT_DIR = Path("analysis/good_pool_audit_20260604_174051")

FILTER_ORDER = [
    ("valid", "RDKit valid"),
    ("carbon_ok", "carbon <= 0.85"),
    ("hetero_ok", "hetero >= 0.05"),
    ("three_ring_ok", "3-ring == 0"),
    ("fragment_ok", "fragment <= 1"),
    ("bad_bond_ok", "bad bond <= 0.05"),
    ("halogen_ok", "bad halogen == 0"),
]


def _read_summary(run_dir: Path) -> pd.DataFrame:
    """Read one run-level training summary."""
    return pd.read_csv(run_dir / "training_summary.csv")


def _read_molecules(run_dir: Path) -> pd.DataFrame:
    """Read all per-epoch molecule CSV files."""
    frames = []
    for csv_path in sorted((run_dir / "epochs").glob("epoch_*/molecules.csv")):
        frames.append(pd.read_csv(csv_path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _read_jsonl(path: Path) -> pd.DataFrame:
    """Read a JSONL file into a DataFrame."""
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def _prepare_molecule_flags(molecules: pd.DataFrame) -> pd.DataFrame:
    """Add configured GoodMoleculePool filter flags."""
    molecules = molecules.copy()
    numeric_cols = [
        "rdkit_valid",
        "reward_total",
        "qed_raw",
        "chembl_classifier_reward",
        "carbon_fraction",
        "hetero_atom_fraction",
        "three_member_ring_count",
        "fragment_count",
        "bad_bond_length_fraction",
        "halogen_bad_position_count",
        "chemistry_guard_violation_count",
        "aromatic_ring_count",
        "heavy_atom_count",
        "hetero_atom_count",
        "chemistry_guard_reward",
    ]
    for col in numeric_cols:
        if col in molecules:
            molecules[col] = pd.to_numeric(molecules[col], errors="coerce")

    molecules["valid"] = molecules["rdkit_valid"].astype(int).eq(1)
    molecules["carbon_ok"] = molecules["carbon_fraction"].le(0.85)
    molecules["hetero_ok"] = molecules["hetero_atom_fraction"].ge(0.05)
    molecules["three_ring_ok"] = molecules["three_member_ring_count"].le(0)
    molecules["fragment_ok"] = molecules["fragment_count"].le(1)
    molecules["bad_bond_ok"] = molecules["bad_bond_length_fraction"].le(0.05)
    molecules["halogen_ok"] = molecules["halogen_bad_position_count"].le(0)
    molecules["passes_all_filters"] = molecules[[name for name, _ in FILTER_ORDER]].all(axis=1)
    return molecules


def _recompute_selected(molecules: pd.DataFrame) -> pd.DataFrame:
    """Recompute per-epoch selected molecules using the audit selector rules."""
    selected_rows = []
    for _epoch, group in molecules.groupby("epoch"):
        candidates = group[group["passes_all_filters"]].copy()
        candidates = candidates.sort_values("reward_total", ascending=False)
        seen = set()
        kept = []
        for _, row in candidates.iterrows():
            smiles = row.get("smiles")
            if pd.notna(smiles) and smiles in seen:
                continue
            if pd.notna(smiles):
                seen.add(smiles)
            kept.append(row)
            if len(kept) >= 16:
                break
        if kept:
            selected_rows.append(pd.DataFrame(kept))
    if not selected_rows:
        return pd.DataFrame(columns=molecules.columns)
    return pd.concat(selected_rows, ignore_index=True)


def _failure_counts(summary: pd.DataFrame) -> dict[str, int]:
    """Aggregate failure_counts strings from training_summary.csv."""
    merged: dict[str, int] = {}
    for raw in summary["failure_counts"].dropna():
        try:
            parsed = ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            parsed = {}
        for key, value in parsed.items():
            merged[key] = merged.get(key, 0) + int(value)
    return merged


def _write_tables(
    *,
    summary_v1: pd.DataFrame,
    summary: pd.DataFrame,
    molecules: pd.DataFrame,
    manifest: pd.DataFrame,
    retained: pd.DataFrame,
    selected: pd.DataFrame,
) -> dict[str, object]:
    """Write CSV/JSON analysis tables and return compact summary facts."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    epoch_cols = [
        "epoch",
        "generated_total",
        "rdkit_valid",
        "rdkit_valid_ratio",
        "reward_mean",
        "avg_qed",
        "avg_chembl_classifier_reward",
        "avg_chemistry_guard_reward",
        "avg_chemistry_guard_violation_count",
        "avg_carbon_fraction",
        "avg_hetero_atom_fraction",
        "avg_three_member_ring_count",
        "avg_bad_bond_length_fraction",
        "good_pool_selected_count",
        "good_pool_retained_count",
        "good_pool_retained_score_max",
        "good_pool_retained_score_mean",
        "time_generate",
        "time_train",
        "time_replay",
    ]
    epoch_summary = summary[[c for c in epoch_cols if c in summary.columns]].copy()
    epoch_summary["epoch_total_time"] = summary[[c for c in summary.columns if c.startswith("time_")]].sum(axis=1)
    epoch_summary.to_csv(OUT_DIR / "epoch_pool_summary.csv", index=False)

    mask = pd.Series(True, index=molecules.index)
    funnel_rows = []
    for key, label in FILTER_ORDER:
        mask = mask & molecules[key].fillna(False)
        funnel_rows.append(
            {
                "step": key,
                "label": label,
                "count": int(mask.sum()),
                "fraction_of_total": float(mask.mean()),
                "fraction_of_valid": float(mask.sum() / max(1, int(molecules["valid"].sum()))),
            }
        )
    funnel = pd.DataFrame(funnel_rows)
    funnel.to_csv(OUT_DIR / "filter_funnel.csv", index=False)

    valid_molecules = molecules[molecules["valid"]].copy()
    independent_rows = []
    for key, label in FILTER_ORDER[1:]:
        ok = valid_molecules[key].fillna(False)
        independent_rows.append(
            {
                "filter": key,
                "label": label,
                "pass_count": int(ok.sum()),
                "fail_count": int((~ok).sum()),
                "pass_rate_valid": float(ok.mean()) if len(ok) else 0.0,
                "fail_rate_valid": float((~ok).mean()) if len(ok) else 0.0,
            }
        )
    independent = pd.DataFrame(independent_rows)
    independent.to_csv(OUT_DIR / "independent_filter_failures.csv", index=False)

    remaining = valid_molecules.copy()
    step_rows = [{"step": "start_valid", "label": "start valid", "removed": 0, "remaining": int(len(remaining))}]
    for key, label in FILTER_ORDER[1:]:
        failed = ~remaining[key].fillna(False)
        step_rows.append({"step": key, "label": label, "removed": int(failed.sum()), "remaining": int((~failed).sum())})
        remaining = remaining[~failed]
    pd.DataFrame(step_rows).to_csv(OUT_DIR / "stepwise_filter_removal.csv", index=False)

    failures = _failure_counts(summary)
    pd.DataFrame([{"failure_type": k, "count": v} for k, v in sorted(failures.items())]).to_csv(
        OUT_DIR / "failure_counts.csv",
        index=False,
    )

    selected.to_csv(OUT_DIR / "selected_molecules_recomputed.csv", index=False)
    manifest.to_csv(OUT_DIR / "good_molecules_manifest.csv", index=False)
    retained.to_csv(OUT_DIR / "good_molecules_retained.csv", index=False)
    molecules.to_csv(OUT_DIR / "all_molecules_with_filter_flags.csv", index=False)

    preview_cols = [
        "epoch",
        "local_index",
        "score",
        "smiles",
        "qed_raw",
        "chembl_classifier_reward",
        "carbon_fraction",
        "hetero_atom_fraction",
        "three_member_ring_count",
        "bad_bond_length_fraction",
        "chemistry_guard_violation_count",
        "heavy_atom_count",
    ]
    if not retained.empty:
        retained[[c for c in preview_cols if c in retained.columns]].to_csv(OUT_DIR / "retained_preview.csv", index=False)

    payload: dict[str, object] = {
        "run_v1": str(RUN_V1),
        "run_v2": str(RUN_V2),
        "config_path": "launch/generated/good_pool_audit_chemistry_guard_10ep_260604.yaml",
        "total_generated": int(summary["generated_total"].sum()),
        "total_valid": int(summary["rdkit_valid"].sum()),
        "overall_valid_ratio": float(summary["rdkit_valid"].sum() / summary["generated_total"].sum()),
        "v1_good_pool_selected_total": float(summary_v1.get("good_pool_selected_count", pd.Series([0])).fillna(0).sum())
        if "good_pool_selected_count" in summary_v1
        else 0.0,
        "v2_good_pool_selected_total": float(summary["good_pool_selected_count"].fillna(0).sum()),
        "v2_good_pool_retained_final": int(summary["good_pool_retained_count"].iloc[-1]),
        "manifest_count": int(len(manifest)),
        "retained_count": int(len(retained)),
        "passes_all_filter_count": int(molecules["passes_all_filters"].sum()),
        "passes_all_filter_rate_total": float(molecules["passes_all_filters"].mean()),
        "passes_all_filter_rate_valid": float(molecules["passes_all_filters"].sum() / max(1, int(molecules["valid"].sum()))),
        "mean_time_generate_sec": float(summary["time_generate"].mean()),
        "mean_time_train_sec": float(summary["time_train"].mean()),
        "mean_epoch_total_time_sec": float(epoch_summary["epoch_total_time"].mean()),
        "wall_clock_seconds_v2": 119.0,
        "failure_counts": failures,
    }

    if not retained.empty:
        for col in [
            "score",
            "qed_raw",
            "chembl_classifier_reward",
            "carbon_fraction",
            "hetero_atom_fraction",
            "bad_bond_length_fraction",
            "three_member_ring_count",
            "heavy_atom_count",
            "aromatic_ring_count",
        ]:
            if col in retained.columns:
                retained[col] = pd.to_numeric(retained[col], errors="coerce")
        payload.update(
            {
                "retained_score_max": float(retained["score"].max()),
                "retained_score_mean": float(retained["score"].mean()),
                "retained_score_median": float(retained["score"].median()),
                "retained_positive_score_count": int((retained["score"] > 0).sum()),
                "retained_mean_carbon_fraction": float(retained["carbon_fraction"].mean()),
                "retained_mean_hetero_fraction": float(retained["hetero_atom_fraction"].mean()),
                "retained_aromatic_nonzero_count": int((retained["aromatic_ring_count"] > 0).sum()),
            }
        )

    with (OUT_DIR / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
    return payload


def _write_plots(summary: pd.DataFrame, retained: pd.DataFrame) -> None:
    """Write simple diagnostic plots."""
    funnel = pd.read_csv(OUT_DIR / "filter_funnel.csv")
    independent = pd.read_csv(OUT_DIR / "independent_filter_failures.csv")

    plt.figure(figsize=(9, 5))
    plt.plot(summary["epoch"], summary["rdkit_valid_ratio"] * 100, marker="o", label="RDKit valid %")
    plt.plot(summary["epoch"], summary["good_pool_selected_count"], marker="s", label="selected count")
    plt.plot(summary["epoch"], summary["good_pool_retained_count"], marker="^", label="retained count")
    plt.xlabel("Epoch")
    plt.ylabel("Percent / Count")
    plt.title("GoodMoleculePool audit by epoch")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "epoch_pool_summary.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.bar(funnel["label"], funnel["count"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Molecule count")
    plt.title("Configured filter funnel")
    for idx, value in enumerate(funnel["count"]):
        plt.text(idx, value + 2, str(int(value)), ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "filter_funnel.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.bar(independent["label"], independent["fail_rate_valid"] * 100)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Fail rate among valid molecules (%)")
    plt.title("Independent quality-filter failure rates")
    for idx, value in enumerate(independent["fail_rate_valid"] * 100):
        plt.text(idx, value + 1, f"{value:.1f}%", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "independent_filter_failures.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    if not retained.empty:
        plt.hist(retained["score"], bins=min(10, max(3, len(retained))), edgecolor="black")
    plt.xlabel("retained score")
    plt.ylabel("count")
    plt.title("Retained score distribution")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "retained_score_distribution.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    if not retained.empty:
        plt.scatter(retained["carbon_fraction"], retained["score"], s=45, alpha=0.8)
    plt.axvline(0.85, color="red", linestyle="--", linewidth=1, label="carbon limit")
    plt.xlabel("carbon fraction")
    plt.ylabel("score")
    plt.title("Retained score vs carbon fraction")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "retained_score_vs_carbon.png", dpi=180)
    plt.close()


def main() -> None:
    """Run the audit analysis."""
    summary_v1 = _read_summary(RUN_V1)
    summary = _read_summary(RUN_V2)
    molecules = _prepare_molecule_flags(_read_molecules(RUN_V2))
    retained = _read_jsonl(RUN_V2 / "good_molecule_pool" / "good_molecules_retained_rank00.jsonl")
    manifest = _read_jsonl(RUN_V2 / "good_molecule_pool" / "good_molecules_rank00.jsonl")
    selected = _recompute_selected(molecules)

    payload = _write_tables(
        summary_v1=summary_v1,
        summary=summary,
        molecules=molecules,
        manifest=manifest,
        retained=retained,
        selected=selected,
    )
    _write_plots(summary, retained)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Wrote analysis to {OUT_DIR}")


if __name__ == "__main__":
    main()
