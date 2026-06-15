"""Aggregate edge denoiser ablation reports for 20260603."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


OUTDIR = Path("analysis/edge_denoiser_ablation_summary_20260603")


def _load_json(path: Path) -> dict[str, object]:
    """Load one JSON file."""
    return json.loads(path.read_text())


def _float(value: object) -> float:
    """Convert a JSON value to float."""
    return float(value)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write dictionaries to a CSV file."""
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _bar_plot(path: Path, rows: list[dict[str, object]], keys: list[str], title: str) -> None:
    """Plot grouped bars for selected metrics."""
    labels = [str(row["run_id"]).split("_")[0] for row in rows]
    x_values = list(range(len(rows)))
    width = 0.8 / len(keys)
    plt.figure(figsize=(9, 5))
    for key_index, key in enumerate(keys):
        values = []
        for row in rows:
            raw_value = row.get(key)
            if raw_value in (None, ""):
                values.append(float("nan"))
            else:
                value = float(raw_value)
                values.append(value if math.isfinite(value) else float("nan"))
        offsets = [
            x_value + (key_index - (len(keys) - 1) / 2) * width
            for x_value in x_values
        ]
        plt.bar(offsets, values, width=width, label=key)
    plt.xticks(x_values, labels)
    plt.ylim(0, 1.05)
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _train_rows() -> list[dict[str, object]]:
    """Build train-final rows for A/B/C/D."""
    train_runs = [
        (
            "A_inverse_clean",
            "inverse_sqrt clean",
            Path("analysis/edge_denoiser_ablation_20260603_A_inverse_clean"),
        ),
        (
            "B_diffgui_noise005",
            "diffgui noise0.05 focal1",
            Path("analysis/edge_denoiser_ablation_20260603_B_diffgui_noise005"),
        ),
        (
            "C_diffgui_noise005_focal2",
            "diffgui noise0.05 focal2",
            Path("analysis/edge_denoiser_ablation_20260603_C_diffgui_noise005_focal2_gpu"),
        ),
        (
            "D_diffgui_noise010",
            "diffgui noise0.10 focal1",
            Path("analysis/edge_denoiser_ablation_20260603_D_diffgui_noise010"),
        ),
    ]
    rows: list[dict[str, object]] = []
    for run_id, label, root in train_runs:
        data = _load_json(root / "edge_denoiser_overfit_summary.json")
        final = data["final_epoch"]
        best = data["best_balanced_accuracy_epoch"]
        assert isinstance(final, dict)
        assert isinstance(best, dict)
        rows.append(
            {
                "run_id": run_id,
                "label": label,
                "source_dir": str(root),
                "device": data.get("device", ""),
                "class_weight_mode": data.get("class_weight_mode", ""),
                "no_bond_weight": data.get("no_bond_weight", ""),
                "pos_noise_std": data.get("pos_noise_std", ""),
                "focal_gamma": data.get("focal_gamma", ""),
                "final_epoch": final.get("epoch"),
                "final_eval_loss": final.get("eval_loss"),
                "final_balanced_accuracy": final.get("balanced_accuracy"),
                "final_real_bond_f1": final.get("real_bond_f1"),
                "final_double_f1": final.get("f1_double"),
                "final_triple_f1": final.get("f1_triple"),
                "final_aromatic_f1": final.get("f1_aromatic"),
                "train_recon_semantic_sanitize": final.get(
                    "recon_semantic_sanitize_success_rate"
                ),
                "train_recon_bond_type_map": final.get(
                    "recon_bond_type_map_match_rate"
                ),
                "train_recon_semantic_smiles": final.get(
                    "recon_semantic_smiles_no_stereo_match_rate"
                ),
                "train_recon_bad_bond_fraction": final.get(
                    "recon_mean_bad_bond_length_fraction"
                ),
                "best_epoch": best.get("epoch"),
                "best_balanced_accuracy": best.get("balanced_accuracy"),
                "best_real_bond_recall": best.get("real_bond_recall"),
            }
        )
    return rows


def _eval_rows() -> list[dict[str, object]]:
    """Build t-grid rows for runs with final evaluator output."""
    eval_runs = [
        (
            "B_diffgui_noise005",
            "diffgui noise0.05 focal1",
            Path("analysis/edge_denoiser_ablation_eval_20260603_B_diffgui_noise005"),
        ),
        (
            "C_diffgui_noise005_focal2",
            "diffgui noise0.05 focal2",
            Path(
                "analysis/edge_denoiser_ablation_eval_20260603_C_diffgui_noise005_focal2_gpu"
            ),
        ),
        (
            "D_diffgui_noise010",
            "diffgui noise0.10 focal1",
            Path("analysis/edge_denoiser_ablation_eval_20260603_D_diffgui_noise010"),
        ),
    ]
    rows: list[dict[str, object]] = []
    for run_id, label, root in eval_runs:
        data = _load_json(root / "edge_denoiser_eval_summary.json")
        classification = data["classification"]
        reconstruction = data["reconstruction"]
        assert isinstance(classification, dict)
        assert isinstance(reconstruction, dict)
        rows.append(
            {
                "run_id": run_id,
                "label": label,
                "source_dir": str(root),
                "device": data.get("device", ""),
                "eval_molecules": data.get("eval_molecules"),
                "best_timestep": classification.get("best_timestep"),
                "best_balanced_accuracy": classification.get(
                    "best_balanced_accuracy"
                ),
                "best_real_bond_recall": classification.get(
                    "best_real_bond_recall"
                ),
                "worst_timestep": classification.get("worst_timestep"),
                "worst_balanced_accuracy": classification.get(
                    "worst_balanced_accuracy"
                ),
                "bond_pair_set_match": reconstruction.get(
                    "bond_pair_set_match_rate"
                ),
                "bond_type_map_match": reconstruction.get(
                    "bond_type_map_match_rate"
                ),
                "plain_sanitize": reconstruction.get("plain_sanitize_success_rate"),
                "semantic_sanitize": reconstruction.get(
                    "semantic_sanitize_success_rate"
                ),
                "plain_smiles": reconstruction.get(
                    "plain_smiles_no_stereo_match_rate"
                ),
                "semantic_smiles": reconstruction.get(
                    "semantic_smiles_no_stereo_match_rate"
                ),
                "mean_bad_bond_length_fraction": reconstruction.get(
                    "mean_bad_bond_length_fraction"
                ),
                "total_bad_bond_length_count": reconstruction.get(
                    "total_bad_bond_length_count"
                ),
            }
        )
    return rows


def _row_by_prefix(rows: list[dict[str, object]], prefix: str) -> dict[str, object]:
    """Return the first row whose run id starts with a prefix."""
    for row in rows:
        if str(row["run_id"]).startswith(prefix):
            return row
    raise KeyError(prefix)


def _write_markdown(train_rows: list[dict[str, object]], eval_rows: list[dict[str, object]]) -> None:
    """Write the Markdown summary."""
    b_row = _row_by_prefix(eval_rows, "B_")
    c_row = _row_by_prefix(eval_rows, "C_")
    d_row = _row_by_prefix(eval_rows, "D_")

    lines = [
        "# Edge Denoiser Ablation Summary 20260603",
        "",
        "本汇总加入新补跑的 GPU 版 C 组：`diffgui + pos_noise_std=0.05 + focal_gamma=2.0`。早先误跑的 CPU 版 C 组保留在原目录，但不纳入本结论。",
        "",
        "## Shared Setup",
        "",
        "- 数据：`analysis/edge_data_chain_20260602_n1000/train_edge.npz` + `analysis/edge_semantic_bundle_20260602_n1000/train_edge_semantic.npz`",
        "- split：1000 molecules，train/val = 800/200，split_seed=0，seed=0",
        "- 模型：`node_edge`，hidden_dim=64，num_layers=3，distance_basis_dim=16",
        "- 训练：80 epochs，batch_size=16，reconstruction_every=10，t=500，limit=100",
        "- t-grid 评估：val split 200 molecules，timesteps=0,50,100,250,500,750,950,999，reconstruct_timestep=500，limit=200",
        "",
        "## Train Final Epoch",
        "",
        "| run | device | weight | noise | focal | real F1 | double F1 | aromatic F1 | semantic sanitize | bond type map | semantic SMILES | best BA epoch | best BA |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in train_rows:
        lines.append(
            "| {run_id} | {device} | {class_weight_mode} | {pos_noise_std} | {focal_gamma} | "
            "{final_real_bond_f1:.4f} | {final_double_f1:.4f} | {final_aromatic_f1:.4f} | "
            "{train_recon_semantic_sanitize:.4f} | {train_recon_bond_type_map:.4f} | "
            "{train_recon_semantic_smiles:.4f} | {best_epoch} | {best_balanced_accuracy:.4f} |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "![train_final_key_metrics](train_final_key_metrics.png)",
            "",
            "## T-grid Evaluation",
            "",
            "A 组 t-grid 评估本轮仍未补跑；本表只比较已有同口径 final evaluator 的 B/C/D。",
            "",
            "| run | device | best t | best BA | best real recall | semantic sanitize | bond type map | semantic SMILES | bad bond frac | bad bond count |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in eval_rows:
        lines.append(
            "| {run_id} | {device} | {best_timestep} | {best_balanced_accuracy:.4f} | "
            "{best_real_bond_recall:.4f} | {semantic_sanitize:.4f} | "
            "{bond_type_map_match:.4f} | {semantic_smiles:.4f} | "
            "{mean_bad_bond_length_fraction:.6f} | {total_bad_bond_length_count} |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "![tgrid_eval_available_metrics](tgrid_eval_available_metrics.png)",
            "",
            "## Interpretation",
            "",
            f"- C 组补跑后，t-grid best BA = {_float(c_row['best_balanced_accuracy']):.4f}，是 B/C/D 中最高；说明 focal_gamma=2 对分类层面的困难样本有帮助。",
            f"- 但 C 的 semantic SMILES = {_float(c_row['semantic_smiles']):.4f}，略低于 B 的 {_float(b_row['semantic_smiles']):.4f}；bond type map = {_float(c_row['bond_type_map_match']):.4f}，也略低于 B 的 {_float(b_row['bond_type_map_match']):.4f}。",
            f"- D 组 semantic sanitize = {_float(d_row['semantic_sanitize']):.4f} 仍最高，但 bond type map 和 semantic SMILES 明显弱，说明 0.10 A 噪声更偏向保守可 sanitize，而不是精确还原键型。",
            "- 当前推荐：如果目标是后续接 DiffSBDD 点云并尽量稳，B 仍是最均衡默认；C 可以作为少数键类别/高召回版本并行保留。",
            "",
            "## Files",
            "",
            "- [train_final_summary.csv](train_final_summary.csv)",
            "- [tgrid_eval_summary.csv](tgrid_eval_summary.csv)",
            "- [train_final_key_metrics.png](train_final_key_metrics.png)",
            "- [tgrid_eval_available_metrics.png](tgrid_eval_available_metrics.png)",
        ]
    )
    (OUTDIR / "ablation_summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    """Aggregate CSV, plots, and Markdown."""
    OUTDIR.mkdir(parents=True, exist_ok=True)
    train_rows = _train_rows()
    eval_rows = _eval_rows()
    _write_csv(OUTDIR / "train_final_summary.csv", train_rows)
    _write_csv(OUTDIR / "tgrid_eval_summary.csv", eval_rows)
    _bar_plot(
        OUTDIR / "train_final_key_metrics.png",
        train_rows,
        [
            "final_real_bond_f1",
            "final_double_f1",
            "final_aromatic_f1",
            "train_recon_semantic_sanitize",
            "train_recon_bond_type_map",
            "train_recon_semantic_smiles",
        ],
        "Train final epoch metrics",
    )
    _bar_plot(
        OUTDIR / "tgrid_eval_available_metrics.png",
        eval_rows,
        [
            "best_balanced_accuracy",
            "best_real_bond_recall",
            "semantic_sanitize",
            "bond_type_map_match",
            "semantic_smiles",
        ],
        "T-grid evaluation metrics",
    )
    _write_markdown(train_rows, eval_rows)
    print(OUTDIR / "ablation_summary.md")


if __name__ == "__main__":
    main()
