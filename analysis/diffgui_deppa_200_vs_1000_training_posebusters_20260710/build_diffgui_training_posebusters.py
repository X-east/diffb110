from __future__ import annotations

import json
import math
import tempfile
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem

import posebusters.posebusters as _pbmod
from posebusters import PoseBusters


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
RECEPTOR = ROOT / "analysis/benchmark/8gjl_c1101_benchmark_20260603_500mol_500steps/inputs/8GJL_C1101_pocket8A.pdb"

RUNS = [
    {
        "run_id": "diffgui_t200",
        "label": "DiffGui 200 steps",
        "timesteps": 200,
        "parts": [
            ROOT / "runs/diffgui-deppa-full-lr1e5-wd1e4-b8x8-t200-100ep_20260626_165809",
        ],
        "endpoint_epochs": [1, 89],
    },
    {
        "run_id": "diffgui_t1000",
        "label": "DiffGui 1000 steps",
        "timesteps": 1000,
        "parts": [
            ROOT / "runs/diffgui-deppa-full-lr1e5-wd1e4-b8x8-t1000-300ep_20260704_112615",
            ROOT / "runs/diffgui-deppa-full-lr1e5-wd1e4-b8x8-t1000-300ep-continue-from3_20260704_152703",
        ],
        "endpoint_epochs": [1, 36],
    },
]

METRIC_GROUPS = [
    {
        "filename": "training_reward_docking.png",
        "title": "Reward And Docking",
        "metrics": [
            ("reward_total_mean", "Reward total mean"),
            ("deppa_reward_mean", "DePPA reward mean"),
            ("docking_raw_mean", "Docking raw mean"),
            ("docking_reward_mean", "Docking reward mean"),
        ],
    },
    {
        "filename": "training_validity_success.png",
        "title": "Validity And Success Rates",
        "metrics": [
            ("reconstruction_success_ratio", "Reconstruction"),
            ("molecule_rdkit_valid_ratio", "RDKit valid"),
            ("molecule_docking_success_ratio", "Docking success"),
            ("duplicate_ratio", "Duplicate ratio"),
        ],
    },
    {
        "filename": "training_ppo_drift.png",
        "title": "PPO Drift",
        "metrics": [
            ("PPO_Loss", "PPO loss"),
            ("PPO_ApproxKL", "Approx KL"),
            ("ClipFrac", "Clip frac"),
            ("RatioMean", "Ratio mean"),
        ],
    },
    {
        "filename": "training_descriptor_panel.png",
        "title": "Descriptor Drift",
        "metrics": [
            ("logp_mean", "LogP mean"),
            ("tpsa_mean", "TPSA mean"),
            ("ring_count_mean", "Ring count"),
            ("hetero_atom_mean", "Hetero atoms"),
            ("carbon_fraction", "Carbon fraction"),
            ("aromatic_ring_mean", "Aromatic rings"),
        ],
    },
    {
        "filename": "training_timing_memory.png",
        "title": "Timing And Memory",
        "metrics": [
            ("total_seconds", "Epoch seconds"),
            ("rollout_seconds", "Rollout seconds"),
            ("reward_seconds", "Reward seconds"),
            ("ppo_seconds", "PPO seconds"),
            ("gpu_monitor_peak_used_mb", "Peak GPU used MB"),
        ],
    },
]


def _get_name(self: object, mol_pred: Chem.Mol) -> str:
    try:
        return str(mol_pred.GetProp("_Name"))
    except Exception:
        return "mol"


_pbmod.PoseBusters._get_name = _get_name


def _sdf_count(path: Path) -> int:
    count = 0
    with path.open("r", errors="replace") as handle:
        for line in handle:
            if line.strip() == "$$$$":
                count += 1
    return count


def _pb_valid_row(row: pd.Series) -> bool:
    bool_cols = [col for col, value in row.items() if isinstance(value, (bool, np.bool_))]
    return bool(bool_cols) and all(bool(row[col]) for col in bool_cols)


def _bust_one_molecule(buster: PoseBusters, receptor: Path, mol: Chem.Mol) -> dict[str, Any]:
    tmp = tempfile.NamedTemporaryFile(suffix=".sdf", delete=False)
    tmp_path = Path(tmp.name)
    try:
        writer = Chem.SDWriter(str(tmp_path))
        writer.write(mol)
        writer.close()
        df = buster.bust(mol_pred=str(tmp_path), mol_cond=str(receptor))
    finally:
        tmp.close()
        tmp_path.unlink(missing_ok=True)
    if df.empty:
        return {"pb_valid": False, "posebusters_empty": True}
    row = df.iloc[0]
    result = row.to_dict()
    result["pb_valid"] = _pb_valid_row(row)
    result["posebusters_empty"] = False
    return result


def _load_training_rows() -> pd.DataFrame:
    frames = []
    for run in RUNS:
        for part_index, run_dir in enumerate(run["parts"], start=1):
            csv_path = run_dir / "training_summary.csv"
            if not csv_path.exists():
                raise FileNotFoundError(csv_path)
            df = pd.read_csv(csv_path)
            df["run_id"] = run["run_id"]
            df["run_label"] = run["label"]
            df["timesteps_config"] = run["timesteps"]
            df["run_dir"] = str(run_dir)
            df["part_index"] = part_index
            frames.append(df)
    all_rows = pd.concat(frames, ignore_index=True)
    all_rows["epoch"] = all_rows["epoch"].astype(int)
    all_rows = all_rows.sort_values(["run_id", "epoch", "part_index"]).drop_duplicates(
        ["run_id", "epoch"], keep="last"
    )
    return all_rows


def _endpoint_sdf(run: dict[str, Any], epoch: int) -> Path:
    for run_dir in run["parts"]:
        path = run_dir / "epochs" / f"epoch_{epoch:04d}" / "molecules.sdf"
        if path.exists():
            return path
    raise FileNotFoundError(f"missing molecules.sdf for {run['run_id']} epoch {epoch}")


def _endpoint_rows(training: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for run in RUNS:
        run_df = training[training["run_id"] == run["run_id"]]
        for epoch in run["endpoint_epochs"]:
            row = run_df[run_df["epoch"] == epoch]
            if row.empty:
                raise ValueError(f"missing training row for {run['run_id']} epoch {epoch}")
            item = row.iloc[0].to_dict()
            item["endpoint"] = "epoch1" if epoch == min(run["endpoint_epochs"]) else "last_complete"
            item["sdf_path"] = str(_endpoint_sdf(run, epoch))
            item["sdf_records"] = _sdf_count(Path(item["sdf_path"]))
            rows.append(item)
    return pd.DataFrame(rows)


def _run_posebusters(endpoints: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_mol_dir = OUT_DIR / "posebusters_per_epoch"
    per_mol_dir.mkdir(exist_ok=True)
    buster = PoseBusters(config="dock")
    summary_rows = []
    all_rows = []
    for _, endpoint in endpoints.iterrows():
        run_id = str(endpoint["run_id"])
        run_label = str(endpoint["run_label"])
        epoch = int(endpoint["epoch"])
        sdf_path = Path(str(endpoint["sdf_path"]))
        csv_path = per_mol_dir / f"{run_id}_epoch_{epoch:04d}_posebusters.csv"
        started = time.time()
        rows: list[dict[str, Any]] = []
        print(f"[posebusters] {run_label} epoch={epoch} sdf={sdf_path}", flush=True)
        for source_index, mol in enumerate(Chem.SDMolSupplier(str(sdf_path), removeHs=False)):
            if mol is None:
                continue
            row: dict[str, Any] = {
                "run_id": run_id,
                "run_label": run_label,
                "epoch": epoch,
                "endpoint": endpoint["endpoint"],
                "source_index": source_index,
                "source_sdf": str(sdf_path),
                "name": mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{source_index:06d}",
            }
            try:
                row.update(_bust_one_molecule(buster, RECEPTOR, mol))
            except Exception as exc:
                row.update({"pb_valid": False, "posebusters_error": repr(exc)})
            rows.append(row)
            if (source_index + 1) % 10 == 0:
                print(f"  processed {source_index + 1}", flush=True)
        per_df = pd.DataFrame(rows)
        per_df.to_csv(csv_path, index=False)
        valid_count = int(per_df["pb_valid"].fillna(False).astype(bool).sum()) if not per_df.empty else 0
        processed = len(per_df)
        summary_rows.append(
            {
                "run_id": run_id,
                "run_label": run_label,
                "timesteps_config": int(endpoint["timesteps_config"]),
                "epoch": epoch,
                "endpoint": endpoint["endpoint"],
                "source_sdf": str(sdf_path),
                "sdf_records": int(endpoint["sdf_records"]),
                "processed": processed,
                "pb_valid_count": valid_count,
                "pb_valid_rate": valid_count / processed if processed else math.nan,
                "elapsed_seconds": time.time() - started,
                "per_molecule_csv": str(csv_path),
            }
        )
        all_rows.extend(rows)
    return pd.DataFrame(summary_rows), pd.DataFrame(all_rows)


def _plot_metric_group(training: pd.DataFrame, group: dict[str, Any]) -> Path:
    metrics = [m for m in group["metrics"] if m[0] in training.columns]
    ncols = 2
    nrows = math.ceil(len(metrics) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.2 * nrows), squeeze=False)
    colors = {"diffgui_t200": "#2f6f9f", "diffgui_t1000": "#b35c1e"}
    for ax, (metric, label) in zip(axes.flat, metrics):
        for run_id, run_df in training.groupby("run_id"):
            values = pd.to_numeric(run_df[metric], errors="coerce")
            ax.plot(run_df["epoch"], values, label=run_df["run_label"].iloc[0], color=colors.get(run_id))
        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    for ax in axes.flat[len(metrics):]:
        ax.axis("off")
    fig.suptitle(group["title"], fontsize=14)
    fig.tight_layout()
    path = OUT_DIR / group["filename"]
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_posebusters(pb_summary: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = []
    values = []
    counts = []
    for _, row in pb_summary.iterrows():
        labels.append(f"{row['run_label']}\nE{int(row['epoch'])}")
        values.append(float(row["pb_valid_rate"]))
        counts.append(f"{int(row['pb_valid_count'])}/{int(row['processed'])}")
    bars = ax.bar(labels, values, color=["#88b04b", "#4f8a2f", "#f2a65a", "#c95d1e"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("PB-valid rate")
    ax.set_title("PoseBusters PB-valid Rate At Endpoint Epochs")
    ax.grid(axis="y", alpha=0.25)
    for bar, text in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.025, text, ha="center", fontsize=9)
    fig.tight_layout()
    path = OUT_DIR / "posebusters_endpoint_pbvalid.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_report(
    training: pd.DataFrame,
    endpoints: pd.DataFrame,
    pb_summary: pd.DataFrame,
    plot_paths: list[Path],
) -> None:
    metric_cols = [
        "run_label",
        "epoch",
        "endpoint",
        "num_timesteps",
        "reward_total_mean",
        "docking_raw_mean",
        "molecule_rdkit_valid_ratio",
        "molecule_docking_success_ratio",
        "PPO_ApproxKL",
        "logp_mean",
        "tpsa_mean",
        "ring_count_mean",
        "total_seconds",
    ]
    endpoint_table = endpoints[[c for c in metric_cols if c in endpoints.columns]].copy()
    for col in endpoint_table.columns:
        if col not in {"run_label", "endpoint"}:
            endpoint_table[col] = pd.to_numeric(endpoint_table[col], errors="ignore")
    pb_table = pb_summary[
        ["run_label", "epoch", "endpoint", "processed", "pb_valid_count", "pb_valid_rate", "elapsed_seconds"]
    ].copy()
    pb_table["pb_valid_rate"] = pb_table["pb_valid_rate"].map(lambda x: f"{x:.4f}")
    pb_table["elapsed_seconds"] = pb_table["elapsed_seconds"].map(lambda x: f"{x:.1f}")

    run_ranges = []
    for run_id, run_df in training.groupby("run_id"):
        run_ranges.append(
            {
                "run": run_df["run_label"].iloc[0],
                "first_epoch": int(run_df["epoch"].min()),
                "last_complete_epoch": int(run_df["epoch"].max()),
                "rows": int(len(run_df)),
                "mean_epoch_minutes": float(pd.to_numeric(run_df["total_seconds"], errors="coerce").mean() / 60),
                "total_epoch_hours": float(pd.to_numeric(run_df["total_seconds"], errors="coerce").sum() / 3600),
            }
        )
    ranges_df = pd.DataFrame(run_ranges)
    ranges_df["mean_epoch_minutes"] = ranges_df["mean_epoch_minutes"].map(lambda x: f"{x:.2f}")
    ranges_df["total_epoch_hours"] = ranges_df["total_epoch_hours"].map(lambda x: f"{x:.2f}")

    lines = [
        "# DiffGui 200步与1000步训练曲线及PoseBusters端点统计",
        "",
        "## 训练范围",
        "",
        ranges_df.to_markdown(index=False),
        "",
        "说明：1000步训练由原始1-3 epoch和续训4-36 epoch合并；续训目录中的未完整汇总epoch不纳入曲线和端点统计。",
        "",
        "## 端点PoseBusters PB-valid率",
        "",
        pb_table.to_markdown(index=False),
        "",
        "PB-valid定义沿用benchmark5 PoseBusters脚本：对每个分子运行 `PoseBusters(config=\"dock\")`，所有布尔检查均通过则记为pb_valid。受体文件为 `8GJL_C1101_pocket8A.pdb`。",
        "",
        "![PoseBusters endpoint PB-valid](posebusters_endpoint_pbvalid.png)",
        "",
        "## 端点训练指标",
        "",
        endpoint_table.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 训练曲线",
        "",
    ]
    for path in plot_paths:
        lines.append(f"![{path.stem}]({path.name})")
        lines.append("")
    lines += [
        "## 输出文件",
        "",
        "- [training_metrics_long.csv](training_metrics_long.csv)",
        "- [endpoint_training_summary.csv](endpoint_training_summary.csv)",
        "- [posebusters_endpoint_summary.csv](posebusters_endpoint_summary.csv)",
        "- [posebusters_endpoint_per_molecule.csv](posebusters_endpoint_per_molecule.csv)",
        "",
    ]
    (OUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not RECEPTOR.exists():
        raise FileNotFoundError(RECEPTOR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    training = _load_training_rows()
    endpoints = _endpoint_rows(training)
    training.to_csv(OUT_DIR / "training_metrics_long.csv", index=False)
    endpoints.to_csv(OUT_DIR / "endpoint_training_summary.csv", index=False)
    pb_summary, pb_per_molecule = _run_posebusters(endpoints)
    pb_summary.to_csv(OUT_DIR / "posebusters_endpoint_summary.csv", index=False)
    pb_per_molecule.to_csv(OUT_DIR / "posebusters_endpoint_per_molecule.csv", index=False)
    plot_paths = [_plot_metric_group(training, group) for group in METRIC_GROUPS]
    plot_paths.append(_plot_posebusters(pb_summary))
    _write_report(training, endpoints, pb_summary, plot_paths)
    manifest = {
        "out_dir": str(OUT_DIR),
        "receptor": str(RECEPTOR),
        "created_files": sorted(path.name for path in OUT_DIR.iterdir() if path.is_file()),
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
