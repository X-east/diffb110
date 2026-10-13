"""Score generated 8GJL/C1101 molecules with the trained D-MPNN checkpoint."""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-diffb110")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from chembl_classifier.dmpnn.predictor import DMPNNClassifierPredictor  # noqa: E402

RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.error")

OUT_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = ROOT / "chembl_classifier" / "models" / "chembl_dmpnn_v0_1_scaffold_strongpoly.pt"
INPUT_SDFS = {
    "DiffGui": ROOT
    / "analysis"
    / "benchmark"
    / "8gjl_c1101_benchmark_20260603_500mol_500steps"
    / "aggregated"
    / "diffgui.sdf",
    "DiffSBDD": ROOT
    / "analysis"
    / "benchmark"
    / "8gjl_c1101_benchmark_20260603_500mol_500steps"
    / "aggregated"
    / "diffsbdd.sdf",
}


@dataclass(frozen=True)
class MoleculeRecord:
    """A parsed SDF molecule row plus an optional sanitized molecule."""

    row: dict[str, Any]
    mol: Chem.Mol | None


def _float_or_nan(value: Any) -> float:
    """Return a finite float when possible, otherwise NaN."""
    try:
        output = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return output if math.isfinite(output) else float("nan")


def _safe_qed(mol: Chem.Mol) -> float:
    """Return QED, preserving failures as NaN."""
    try:
        return float(QED.qed(mol))
    except Exception:
        return float("nan")


def _ring_counts(mol: Chem.Mol) -> tuple[int, int, int]:
    """Return total ring, aromatic ring, and three-member ring counts."""
    ring_info = mol.GetRingInfo()
    atom_rings = list(ring_info.AtomRings())
    aromatic = 0
    three_member = 0
    for ring in atom_rings:
        if len(ring) == 3:
            three_member += 1
        if all(mol.GetAtomWithIdx(atom_idx).GetIsAromatic() for atom_idx in ring):
            aromatic += 1
    return len(atom_rings), aromatic, three_member


def _descriptor_row(mol: Chem.Mol) -> dict[str, float | int]:
    """Compute project-relevant RDKit descriptors for one sanitized molecule."""
    heavy_atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
    carbon_count = sum(1 for atom in heavy_atoms if atom.GetAtomicNum() == 6)
    hetero_count = sum(1 for atom in heavy_atoms if atom.GetAtomicNum() not in {1, 6})
    heavy_atom_count = len(heavy_atoms)
    ring_count, aromatic_ring_count, three_member_ring_count = _ring_counts(mol)
    return {
        "qed": _safe_qed(mol),
        "logp": float(Crippen.MolLogP(mol)),
        "tpsa": float(Descriptors.TPSA(mol)),
        "mol_wt": float(Descriptors.MolWt(mol)),
        "heavy_atom_count": int(heavy_atom_count),
        "hetero_atom_count": int(hetero_count),
        "carbon_fraction": float(carbon_count / heavy_atom_count) if heavy_atom_count else float("nan"),
        "hba": int(Lipinski.NumHAcceptors(mol)),
        "hbd": int(Lipinski.NumHDonors(mol)),
        "ring_count": int(ring_count),
        "aromatic_ring_count": int(aromatic_ring_count),
        "three_member_ring_count": int(three_member_ring_count),
    }


def _sanitize_mol(raw_mol: Chem.Mol | None) -> tuple[Chem.Mol | None, str]:
    """Return a sanitized copy of an SDF molecule and a failure reason."""
    if raw_mol is None:
        return None, "unreadable_record"
    mol = Chem.Mol(raw_mol)
    try:
        status = Chem.SanitizeMol(mol, catchErrors=True)
    except Exception:
        return None, "sanitize_exception"
    if status != Chem.SanitizeFlags.SANITIZE_NONE:
        return None, str(status)
    try:
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None, "canonical_smiles_failed"
    return mol, ""


def _load_records(model_name: str, sdf_path: Path) -> list[MoleculeRecord]:
    """Load an SDF into row dictionaries while preserving invalid records."""
    records: list[MoleculeRecord] = []
    supplier = Chem.SDMolSupplier(str(sdf_path), sanitize=False, removeHs=False)
    for index, raw_mol in enumerate(supplier):
        mol, failure_reason = _sanitize_mol(raw_mol)
        row: dict[str, Any] = {
            "model": model_name,
            "source_sdf": str(sdf_path),
            "mol_index": index,
            "readable": int(raw_mol is not None),
            "sanitize_valid": int(mol is not None),
            "sanitize_failure": failure_reason,
            "smiles": "",
        }
        if mol is not None:
            row["smiles"] = Chem.MolToSmiles(mol, canonical=True)
            row.update(_descriptor_row(mol))
        else:
            row.update(
                {
                    "qed": float("nan"),
                    "logp": float("nan"),
                    "tpsa": float("nan"),
                    "mol_wt": float("nan"),
                    "heavy_atom_count": float("nan"),
                    "hetero_atom_count": float("nan"),
                    "carbon_fraction": float("nan"),
                    "hba": float("nan"),
                    "hbd": float("nan"),
                    "ring_count": float("nan"),
                    "aromatic_ring_count": float("nan"),
                    "three_member_ring_count": float("nan"),
                }
            )
        records.append(MoleculeRecord(row=row, mol=mol))
    return records


def _score_records(records: list[MoleculeRecord], predictor: DMPNNClassifierPredictor) -> pd.DataFrame:
    """Attach D-MPNN probabilities to parsed molecule records."""
    valid_positions = [index for index, record in enumerate(records) if record.mol is not None]
    valid_mols = [records[index].mol for index in valid_positions]
    probabilities = predictor.predict_proba_mols(valid_mols, batch_size=128)

    rows = [dict(record.row) for record in records]
    for row in rows:
        row["dmpnn_prob"] = float("nan")
        row["dmpnn_scored"] = 0
    for position, probability in zip(valid_positions, probabilities):
        if probability is None:
            continue
        rows[position]["dmpnn_prob"] = float(probability)
        rows[position]["dmpnn_scored"] = 1
    return pd.DataFrame(rows)


def _summary_for_model(model_name: str, group: pd.DataFrame) -> dict[str, Any]:
    """Summarize one generator's D-MPNN and descriptor distribution."""
    scored = group[group["dmpnn_scored"] == 1].copy()
    probs = scored["dmpnn_prob"].astype(float)
    valid_smiles = group.loc[group["sanitize_valid"] == 1, "smiles"].astype(str)
    summary: dict[str, Any] = {
        "model": model_name,
        "records": int(len(group)),
        "readable": int(group["readable"].sum()),
        "sanitize_valid": int(group["sanitize_valid"].sum()),
        "sanitize_valid_rate": float(group["sanitize_valid"].mean()) if len(group) else 0.0,
        "dmpnn_scored": int(group["dmpnn_scored"].sum()),
        "dmpnn_scored_rate": float(group["dmpnn_scored"].mean()) if len(group) else 0.0,
        "unique_smiles": int(valid_smiles.nunique()),
        "unique_smiles_rate_valid": float(valid_smiles.nunique() / len(valid_smiles)) if len(valid_smiles) else 0.0,
        "dmpnn_prob_mean": float(probs.mean()) if len(probs) else float("nan"),
        "dmpnn_prob_median": float(probs.median()) if len(probs) else float("nan"),
        "dmpnn_prob_std": float(probs.std(ddof=0)) if len(probs) else float("nan"),
        "dmpnn_prob_min": float(probs.min()) if len(probs) else float("nan"),
        "dmpnn_prob_q10": float(probs.quantile(0.10)) if len(probs) else float("nan"),
        "dmpnn_prob_q25": float(probs.quantile(0.25)) if len(probs) else float("nan"),
        "dmpnn_prob_q75": float(probs.quantile(0.75)) if len(probs) else float("nan"),
        "dmpnn_prob_q90": float(probs.quantile(0.90)) if len(probs) else float("nan"),
        "dmpnn_prob_max": float(probs.max()) if len(probs) else float("nan"),
    }
    for threshold in (0.5, 0.7, 0.8, 0.9, 0.95):
        summary[f"dmpnn_ge_{str(threshold).replace('.', '_')}_count"] = int((probs >= threshold).sum())
        summary[f"dmpnn_ge_{str(threshold).replace('.', '_')}_rate_scored"] = (
            float((probs >= threshold).mean()) if len(probs) else 0.0
        )
        summary[f"dmpnn_ge_{str(threshold).replace('.', '_')}_rate_all"] = (
            float((scored["dmpnn_prob"] >= threshold).sum() / len(group)) if len(group) else 0.0
        )

    for column in (
        "qed",
        "logp",
        "tpsa",
        "mol_wt",
        "heavy_atom_count",
        "hetero_atom_count",
        "carbon_fraction",
        "hba",
        "hbd",
        "ring_count",
        "aromatic_ring_count",
        "three_member_ring_count",
    ):
        values = pd.to_numeric(scored[column], errors="coerce").dropna()
        summary[f"{column}_mean"] = float(values.mean()) if len(values) else float("nan")
        summary[f"{column}_median"] = float(values.median()) if len(values) else float("nan")
    return summary


def _write_top_sdf(df: pd.DataFrame, output_sdf: Path, top_n: int = 20) -> None:
    """Write the highest D-MPNN molecules into a small inspection SDF."""
    writer = Chem.SDWriter(str(output_sdf))
    try:
        for _, row in df[df["dmpnn_scored"] == 1].sort_values("dmpnn_prob", ascending=False).head(top_n).iterrows():
            supplier = Chem.SDMolSupplier(str(row["source_sdf"]), sanitize=False, removeHs=False)
            raw_mol = supplier[int(row["mol_index"])]
            mol, _ = _sanitize_mol(raw_mol)
            if mol is None:
                continue
            mol.SetProp("generator", str(row["model"]))
            mol.SetProp("source_sdf", str(row["source_sdf"]))
            mol.SetProp("mol_index", str(row["mol_index"]))
            mol.SetProp("smiles", str(row["smiles"]))
            mol.SetProp("dmpnn_prob", f"{_float_or_nan(row['dmpnn_prob']):.6f}")
            mol.SetProp("qed", f"{_float_or_nan(row['qed']):.6f}")
            mol.SetProp("logp", f"{_float_or_nan(row['logp']):.6f}")
            mol.SetProp("tpsa", f"{_float_or_nan(row['tpsa']):.6f}")
            writer.write(mol)
    finally:
        writer.close()


def _plot_outputs(df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    """Write compact distribution plots for D-MPNN and descriptors."""
    plt.style.use("default")
    colors = {"DiffGui": "#2563eb", "DiffSBDD": "#c2410c"}
    scored = df[df["dmpnn_scored"] == 1].copy()

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0.0, 1.0, 31)
    for model_name, group in scored.groupby("model"):
        ax.hist(
            group["dmpnn_prob"].astype(float),
            bins=bins,
            alpha=0.52,
            label=f"{model_name} (n={len(group)})",
            color=colors.get(model_name),
        )
    ax.set_xlabel("D-MPNN P(good)")
    ax.set_ylabel("Molecule count")
    ax.set_title("8GJL/C1101 generated molecules scored by D-MPNN")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "dmpnn_prob_distribution.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    data = [
        scored.loc[scored["model"] == model_name, "dmpnn_prob"].astype(float).to_numpy()
        for model_name in ["DiffGui", "DiffSBDD"]
    ]
    ax.boxplot(data, tick_labels=["DiffGui", "DiffSBDD"], showmeans=True)
    ax.set_ylabel("D-MPNN P(good)")
    ax.set_title("D-MPNN score distribution")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "dmpnn_prob_boxplot.png", dpi=220)
    plt.close(fig)

    bar_columns = [
        ("sanitize_valid_rate", "Sanitize valid"),
        ("dmpnn_ge_0_8_rate_all", "D-MPNN >= 0.8"),
        ("dmpnn_ge_0_9_rate_all", "D-MPNN >= 0.9"),
        ("unique_smiles_rate_valid", "Unique valid SMILES"),
    ]
    plot_df = summary_df.set_index("model")[[column for column, _ in bar_columns]]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(bar_columns))
    width = 0.34
    for idx, model_name in enumerate(["DiffGui", "DiffSBDD"]):
        ax.bar(
            x + (idx - 0.5) * width,
            plot_df.loc[model_name].to_numpy(dtype=float),
            width=width,
            label=model_name,
            color=colors.get(model_name),
        )
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in bar_columns], rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title("Validity, high-score rate, and uniqueness")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "model_rate_comparison.png", dpi=220)
    plt.close(fig)

    scatter_pairs = [
        ("qed", "QED"),
        ("logp", "LogP"),
        ("tpsa", "TPSA"),
        ("carbon_fraction", "Carbon fraction"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for ax, (column, label) in zip(axes.ravel(), scatter_pairs):
        for model_name, group in scored.groupby("model"):
            ax.scatter(
                group[column].astype(float),
                group["dmpnn_prob"].astype(float),
                s=12,
                alpha=0.55,
                label=model_name,
                color=colors.get(model_name),
            )
        ax.set_xlabel(label)
        ax.set_ylabel("D-MPNN P(good)")
        ax.grid(alpha=0.22)
    axes[0, 0].legend()
    fig.suptitle("Descriptor relationship to D-MPNN score", y=1.02)
    fig.savefig(OUT_DIR / "descriptor_vs_dmpnn.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _markdown_table(df: pd.DataFrame) -> str:
    """Render a small dataframe as a Markdown table without optional deps."""
    headers = [str(column) for column in df.columns]
    rows = [[str(value) for value in row] for row in df.to_numpy().tolist()]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]

    def _format_row(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(values)) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([_format_row(headers), separator, *[_format_row(row) for row in rows]])


def _write_report(summary_df: pd.DataFrame, checkpoint_metadata: dict[str, Any]) -> None:
    """Write a Markdown report next to the analysis artifacts."""
    key_columns = [
        "model",
        "records",
        "sanitize_valid",
        "sanitize_valid_rate",
        "dmpnn_scored",
        "unique_smiles",
        "dmpnn_prob_mean",
        "dmpnn_prob_median",
        "dmpnn_ge_0_8_count",
        "dmpnn_ge_0_8_rate_all",
        "dmpnn_ge_0_9_count",
        "dmpnn_ge_0_9_rate_all",
        "qed_mean",
        "logp_mean",
        "tpsa_mean",
        "carbon_fraction_mean",
        "ring_count_mean",
    ]
    table_df = summary_df[key_columns].copy()
    for column in table_df.columns:
        if column != "model":
            table_df[column] = table_df[column].map(
                lambda value: f"{value:.4f}" if isinstance(value, float) else value
            )

    lines = [
        "# 8GJL/C1101 生成分子 D-MPNN 检查点分析",
        "",
        "## 输入",
        "",
        f"- D-MPNN 检查点：`{CHECKPOINT_PATH.relative_to(ROOT)}`",
        f"- DiffGui 聚合 SDF：`{INPUT_SDFS['DiffGui'].relative_to(ROOT)}`",
        f"- DiffSBDD 聚合 SDF：`{INPUT_SDFS['DiffSBDD'].relative_to(ROOT)}`",
        "- 分析范围：只对上述两份聚合 SDF 中 RDKit sanitize 成功的分子进行 D-MPNN 打分；失败记录保留在明细 CSV。",
        "",
        "## 检查点信息",
        "",
        f"- device：`{checkpoint_metadata.get('device')}`",
        f"- cuda_available_at_runtime：`{checkpoint_metadata.get('cuda_available_at_runtime')}`",
        f"- cuda_device_count_at_runtime：`{checkpoint_metadata.get('cuda_device_count_at_runtime')}`",
        f"- fallback_reason：`{checkpoint_metadata.get('fallback_reason')}`",
        f"- n_params：`{checkpoint_metadata.get('n_params')}`",
        f"- model_type：`{checkpoint_metadata.get('model_type')}`",
        f"- epoch：`{checkpoint_metadata.get('epoch', 'NA')}`",
        f"- val_loss：`{checkpoint_metadata.get('val_loss', 'NA')}`",
        f"- val_accuracy：`{checkpoint_metadata.get('val_accuracy', 'NA')}`",
        f"- val_precision：`{checkpoint_metadata.get('val_precision', 'NA')}`",
        f"- val_recall：`{checkpoint_metadata.get('val_recall', 'NA')}`",
        f"- val_auc：`{checkpoint_metadata.get('val_auc', 'NA')}`",
        "",
        "## 核心结果",
        "",
        _markdown_table(table_df),
        "",
        "## 初步判断",
        "",
    ]

    diffgui = summary_df.set_index("model").loc["DiffGui"]
    diffsbdd = summary_df.set_index("model").loc["DiffSBDD"]
    lines.extend(
        [
            (
                f"- DiffGui：sanitize_valid={int(diffgui['sanitize_valid'])}/{int(diffgui['records'])}"
                f"（{diffgui['sanitize_valid_rate']:.1%}），D-MPNN 均值 {diffgui['dmpnn_prob_mean']:.3f}，"
                f">=0.8 的分子 {int(diffgui['dmpnn_ge_0_8_count'])} 个。"
            ),
            (
                f"- DiffSBDD：sanitize_valid={int(diffsbdd['sanitize_valid'])}/{int(diffsbdd['records'])}"
                f"（{diffsbdd['sanitize_valid_rate']:.1%}），D-MPNN 均值 {diffsbdd['dmpnn_prob_mean']:.3f}，"
                f">=0.8 的分子 {int(diffsbdd['dmpnn_ge_0_8_count'])} 个。"
            ),
            (
                "- 该 D-MPNN 是 ChEMBL-like 分类器/奖励检查点，不等价于 SHP2 docking 或活性预测；"
                "它更适合用来筛掉分布明显不像训练集中好分子的样本，并辅助挑 Top 候选。"
            ),
            "",
            "## 输出文件",
            "",
            "- [per_molecule_dmpnn_scores.csv](per_molecule_dmpnn_scores.csv)",
            "- [model_dmpnn_summary.csv](model_dmpnn_summary.csv)",
            "- [model_dmpnn_summary.json](model_dmpnn_summary.json)",
            "- [top_dmpnn_molecules.csv](top_dmpnn_molecules.csv)",
            "- [top_dmpnn_molecules.sdf](top_dmpnn_molecules.sdf)",
            "",
            "![D-MPNN score distribution](dmpnn_prob_distribution.png)",
            "",
            "![D-MPNN score boxplot](dmpnn_prob_boxplot.png)",
            "",
            "![Model rate comparison](model_rate_comparison.png)",
            "",
            "![Descriptor vs D-MPNN](descriptor_vs_dmpnn.png)",
            "",
        ]
    )
    (OUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run the full checkpoint analysis."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cuda_available = bool(torch.cuda.is_available())
    cuda_device_count = int(torch.cuda.device_count()) if cuda_available else 0
    device = "cuda" if cuda_available else "cpu"
    predictor = DMPNNClassifierPredictor.from_checkpoint(CHECKPOINT_PATH, device=device, cache_smiles=True)

    all_frames: list[pd.DataFrame] = []
    for model_name, sdf_path in INPUT_SDFS.items():
        if not sdf_path.exists():
            raise FileNotFoundError(sdf_path)
        records = _load_records(model_name, sdf_path)
        all_frames.append(_score_records(records, predictor))
    df = pd.concat(all_frames, ignore_index=True)
    df.to_csv(OUT_DIR / "per_molecule_dmpnn_scores.csv", index=False)

    summary_rows = [_summary_for_model(model_name, group) for model_name, group in df.groupby("model")]
    summary_df = pd.DataFrame(summary_rows).sort_values("model").reset_index(drop=True)
    summary_df.to_csv(OUT_DIR / "model_dmpnn_summary.csv", index=False)
    (OUT_DIR / "model_dmpnn_summary.json").write_text(
        json.dumps(summary_rows, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8"
    )

    top_df = df[df["dmpnn_scored"] == 1].sort_values("dmpnn_prob", ascending=False).head(50)
    top_df.to_csv(OUT_DIR / "top_dmpnn_molecules.csv", index=False)
    _write_top_sdf(df, OUT_DIR / "top_dmpnn_molecules.sdf", top_n=50)

    checkpoint_payload = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    val_metrics = checkpoint_payload.get("val_metrics") or {}
    checkpoint_metadata = {
        "device": device,
        "cuda_available_at_runtime": cuda_available,
        "cuda_device_count_at_runtime": cuda_device_count,
        "fallback_reason": "" if cuda_available else "torch.cuda.is_available() was false in this conda run; CPU inference was used for the small 1000-molecule analysis.",
        "n_params": predictor.n_params,
        "model_type": checkpoint_payload.get("model_type"),
        "epoch": checkpoint_payload.get("epoch"),
        "val_loss": val_metrics.get("loss"),
        "val_accuracy": val_metrics.get("accuracy"),
        "val_precision": val_metrics.get("precision"),
        "val_recall": val_metrics.get("recall"),
        "val_auc": val_metrics.get("auc"),
    }
    (OUT_DIR / "checkpoint_metadata.json").write_text(
        json.dumps(checkpoint_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    _plot_outputs(df, summary_df)
    _write_report(summary_df, checkpoint_metadata)
    print(json.dumps({"out_dir": str(OUT_DIR), "summary": summary_rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
