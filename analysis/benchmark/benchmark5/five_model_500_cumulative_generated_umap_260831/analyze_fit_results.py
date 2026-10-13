#!/usr/bin/env python3
"""Quantify the three completed 500-step generated-only UMAP fits.

The analysis keeps each independently fitted UMAP in its own coordinate
system.  It reports within-map descriptive separation and validates the
pattern with balanced samples in the original 2048-bit ECFP4 space.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, silhouette_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors


MODEL_ORDER = ["DiffGui", "DiffSBDD", "PMDM", "GCDM-SBDD", "TargetDiff"]
MODEL_COLORS = {
    "DiffGui": "#3B82C4",
    "DiffSBDD": "#D95F59",
    "PMDM": "#D9A441",
    "GCDM-SBDD": "#8E78B7",
    "TargetDiff": "#3D9B72",
}
SCOPE_ORDER = ["first3", "first6", "first9"]
LOGGER = logging.getLogger("analyze_fit_results")


def configure_logging(output_dir: Path) -> None:
    """Configure a concise terminal and durable analysis log."""
    output_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "detailed_analysis.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(stream)
    root.addHandler(file_handler)


def balanced_indices(labels: pd.Series, per_model: int = 500, seed: int = 42) -> np.ndarray:
    """Return deterministic equal-size indices for every model."""
    rng = np.random.default_rng(seed)
    chosen: list[np.ndarray] = []
    for model in MODEL_ORDER:
        available = np.flatnonzero(labels.to_numpy() == model)
        if len(available) < per_model:
            raise ValueError(f"{model} has only {len(available)} rows; need {per_model}")
        selected = np.sort(rng.choice(available, size=per_model, replace=False))
        chosen.append(selected)
    return np.concatenate(chosen)


def calculate_umap_metrics(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate descriptive separation metrics within one UMAP fit."""
    sample_idx = balanced_indices(table["model"], per_model=500)
    sample = table.iloc[sample_idx].reset_index(drop=True)
    xy = sample[["umap1", "umap2"]].to_numpy(dtype=float)
    labels = sample["model"].to_numpy()
    silhouette = float(silhouette_score(xy, labels, metric="euclidean"))

    knn = NearestNeighbors(n_neighbors=21, metric="euclidean", n_jobs=1).fit(xy)
    neighbor_indices = knn.kneighbors(return_distance=False)[:, 1:]
    purity = float(np.mean(labels[neighbor_indices] == labels[:, None]))
    baseline = float(sum((labels == model).mean() ** 2 for model in MODEL_ORDER))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    logistic_scores: list[float] = []
    knn_scores: list[float] = []
    for train_idx, test_idx in cv.split(xy, labels):
        logistic = LogisticRegression(max_iter=2000, random_state=42)
        logistic.fit(xy[train_idx], labels[train_idx])
        logistic_scores.append(balanced_accuracy_score(labels[test_idx], logistic.predict(xy[test_idx])))
        classifier = KNeighborsClassifier(n_neighbors=20, metric="euclidean", n_jobs=1)
        classifier.fit(xy[train_idx], labels[train_idx])
        knn_scores.append(balanced_accuracy_score(labels[test_idx], classifier.predict(xy[test_idx])))

    # The coordinate artifact contains one row per valid molecule and does
    # not add a synthetic ``repeat=pooled`` summary row.  Derive the centroid
    # and RMS radius directly from the coordinates so this analysis remains
    # tied to the actual fit artifact.
    grouped = table.groupby("model", sort=False)
    pooled = grouped[["umap1", "umap2"]].mean().reindex(MODEL_ORDER)
    centroids = pooled[["umap1", "umap2"]].to_numpy(dtype=float)
    radii = []
    for model in MODEL_ORDER:
        model_xy = table.loc[table["model"] == model, ["umap1", "umap2"]].to_numpy(dtype=float)
        model_centroid = pooled.loc[model, ["umap1", "umap2"]].to_numpy(dtype=float)
        radii.append(float(np.sqrt(np.mean(np.sum((model_xy - model_centroid) ** 2, axis=1)))))
    radii = np.asarray(radii, dtype=float)
    centroid_rows: list[dict[str, Any]] = []
    for i, first in enumerate(MODEL_ORDER):
        for j, second in enumerate(MODEL_ORDER):
            if j <= i:
                continue
            distance = float(np.linalg.norm(centroids[i] - centroids[j]))
            normalized = distance / max(float((radii[i] + radii[j]) / 2), 1e-12)
            centroid_rows.append(
                {
                    "model_a": first,
                    "model_b": second,
                    "centroid_distance": distance,
                    "distance_over_mean_rms_radius": normalized,
                }
            )
    metrics = pd.DataFrame(
        [
            {
                "silhouette_2d": silhouette,
                "knn20_same_model_purity": purity,
                "balanced_baseline_purity": baseline,
                "purity_excess_over_baseline": purity - baseline,
                "logistic_balanced_accuracy_mean": np.mean(logistic_scores),
                "logistic_balanced_accuracy_sd": np.std(logistic_scores, ddof=1),
                "knn20_balanced_accuracy_mean": np.mean(knn_scores),
                "knn20_balanced_accuracy_sd": np.std(knn_scores, ddof=1),
                "balanced_sample_n": len(sample),
            }
        ]
    )
    return metrics, pd.DataFrame(centroid_rows)


def tanimoto_matrix(matrix: sparse.csr_matrix) -> np.ndarray:
    """Calculate an exact dense Tanimoto matrix for a manageable sample."""
    intersections = (matrix @ matrix.T).toarray().astype(np.float64)
    on_bits = np.asarray(matrix.sum(axis=1)).ravel().astype(np.float64)
    unions = on_bits[:, None] + on_bits[None, :] - intersections
    result = np.divide(intersections, unions, out=np.zeros_like(intersections), where=unions > 0)
    np.fill_diagonal(result, 1.0)
    return result


def calculate_ecfp4_metrics(table: pd.DataFrame, matrix: sparse.csr_matrix) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate balanced within/between-model ECFP4 Tanimoto evidence."""
    sample_idx = balanced_indices(table["model"], per_model=500)
    sample_table = table.iloc[sample_idx].reset_index(drop=True)
    sample_matrix = matrix[sample_idx]
    similarity = tanimoto_matrix(sample_matrix)
    labels = sample_table["model"].to_numpy()
    matrix_rows: list[dict[str, Any]] = []
    for first_index, first in enumerate(MODEL_ORDER):
        first_mask = labels == first
        for second_index, second in enumerate(MODEL_ORDER):
            second_mask = labels == second
            block = similarity[np.ix_(first_mask, second_mask)]
            if first == second:
                values = block[~np.eye(block.shape[0], dtype=bool)]
            else:
                values = block.ravel()
            matrix_rows.append(
                {
                    "model_a": first,
                    "model_b": second,
                    "mean_tanimoto": float(np.mean(values)),
                    "median_tanimoto": float(np.median(values)),
                    "n_pairs": int(values.size),
                }
            )
    pair_frame = pd.DataFrame(matrix_rows)
    within = pair_frame[pair_frame["model_a"] == pair_frame["model_b"]]["mean_tanimoto"].mean()
    between = pair_frame[pair_frame["model_a"] != pair_frame["model_b"]]["mean_tanimoto"].mean()
    summary = pd.DataFrame(
        [
            {
                "balanced_sample_n": len(sample_table),
                "within_model_mean_tanimoto": within,
                "between_model_mean_tanimoto": between,
                "within_minus_between": within - between,
            }
        ]
    )
    return summary, pair_frame


def plot_umap_metrics(metrics: pd.DataFrame, output_dir: Path) -> None:
    """Plot descriptive UMAP separation metrics across cumulative fits."""
    figure, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    x = np.arange(len(SCOPE_ORDER))
    panels = [
        ("silhouette_2d", "2D silhouette", "#405A7A"),
        ("logistic_balanced_accuracy_mean", "5-fold logistic balanced accuracy", "#A05A3A"),
        ("knn20_same_model_purity", "20-NN same-model purity", "#4B8764"),
    ]
    for axis, (column, title, color) in zip(axes, panels):
        values = metrics.set_index("scope").reindex(SCOPE_ORDER)[column]
        axis.bar(x, values, color=color, alpha=0.86)
        axis.set_xticks(x)
        axis.set_xticklabels(SCOPE_ORDER)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    axes[0].axhline(0, color="black", linewidth=0.7)
    axes[2].axhline(0.2, color="black", linewidth=0.8, linestyle="--", label="balanced baseline=0.20")
    axes[2].legend(fontsize=9)
    figure.suptitle("Descriptive separation within each independent 500-step UMAP fit", fontsize=18)
    figure.text(0.5, 0.01, "Metrics are within-fit summaries; they do not make UMAP coordinates comparable across scopes.", ha="center", fontsize=10)
    figure.savefig(output_dir / "umap_separation_metrics.png", dpi=350, bbox_inches="tight", facecolor="white")
    figure.savefig(output_dir / "umap_separation_metrics.svg", bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_ecfp_heatmaps(pairwise: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Plot original-space ECFP4 mean Tanimoto matrices."""
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.2), squeeze=False)
    for axis, scope in zip(axes[0], SCOPE_ORDER):
        frame = pairwise[scope].pivot(index="model_a", columns="model_b", values="mean_tanimoto").reindex(index=MODEL_ORDER, columns=MODEL_ORDER)
        image = axis.imshow(frame.to_numpy(), vmin=0, vmax=1, cmap="viridis")
        axis.set_xticks(np.arange(len(MODEL_ORDER)))
        axis.set_yticks(np.arange(len(MODEL_ORDER)))
        axis.set_xticklabels(MODEL_ORDER, rotation=45, ha="right", fontsize=9)
        axis.set_yticklabels(MODEL_ORDER, fontsize=9)
        axis.set_title(scope)
        for i in range(len(MODEL_ORDER)):
            for j in range(len(MODEL_ORDER)):
                axis.text(j, i, f"{frame.iloc[i, j]:.2f}", ha="center", va="center", color="white" if frame.iloc[i, j] < 0.45 else "black", fontsize=9)
    figure.colorbar(image, ax=axes[0].tolist(), shrink=0.8, label="Mean ECFP4 Tanimoto")
    figure.suptitle("Original 2048-bit ECFP4 similarity: balanced 500-molecule/model samples", fontsize=17)
    figure.tight_layout(rect=(0, 0, 0.95, 0.92))
    figure.savefig(output_dir / "ecfp4_tanimoto_heatmaps.png", dpi=350, bbox_inches="tight", facecolor="white")
    figure.savefig(output_dir / "ecfp4_tanimoto_heatmaps.svg", bbox_inches="tight", facecolor="white")
    plt.close(figure)


def write_report(
    output_dir: Path,
    validity: pd.DataFrame,
    umap_metrics: pd.DataFrame,
    ecfp_metrics: pd.DataFrame,
    centroid_metrics: dict[str, pd.DataFrame],
) -> None:
    """Write the detailed Chinese interpretation report."""
    first9 = ecfp_metrics[ecfp_metrics["scope"] == "first9"].iloc[0]
    first9_umap = umap_metrics[umap_metrics["scope"] == "first9"].iloc[0]
    nearest_pair = centroid_metrics["first9"].sort_values("distance_over_mean_rms_radius", ascending=False).iloc[0]
    lines = [
        "# 500步累计三组生成-only UMAP详细结果分析",
        "",
        "## 结论",
        "",
        "三组 fit 都显示出稳定的模型相关空间结构，但三组 UMAP 是独立坐标系，不能把坐标移动解释成随重复增加的化学空间迁移。前9轮的模型分离不仅出现在二维 UMAP 中，原始 ECFP4 也显示组内相似度高于模型间相似度，因此“模型分布存在差异”有指纹空间支持；但这仍不是质量、活性或 docking 证据。",
        "",
        "## 数据和方法",
        "",
        "三组分别使用前3轮 `rep4–rep6`、前6轮 `rep4–rep9`、前9轮 `rep4–rep9 + rep13–rep15`。每个 fit 只输入 sanitize-valid 生成分子，沿用 ECFP4 radius=2、2048 bits、cosine、`n_neighbors=20`、`min_dist=0.1`、seed=42。",
        "",
        "UMAP 分离指标和 ECFP4 相似度指标均使用每个模型固定抽取 500 个分子进行平衡比较，抽样 seed=42。合法率仍使用全部 fixed attempts 统计。",
        "",
        "## 固定 attempts 与合法率",
        "",
        "| scope | model | attempts | sanitize-valid | valid rate |",
        "|---|---|---:|---:|---:|",
    ]
    for row in validity.itertuples(index=False):
        lines.append(f"| {row.scope} | {row.model} | {int(row.attempts):,} | {int(row.sanitize_valid):,} | {row.valid_per_attempt:.2%} |")
    lines += [
        "",
        "![Cumulative UMAP fits](cumulative_generated_umap_facets.png)",
        "",
        "## 一、二维 UMAP 内部的模型分离",
        "",
        "| scope | silhouette | 20-NN同模型纯度 | 5-fold logistic balanced accuracy | 20-NN balanced accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in umap_metrics.itertuples(index=False):
        lines.append(f"| {row.scope} | {row.silhouette_2d:.3f} | {row.knn20_same_model_purity:.3f} | {row.logistic_balanced_accuracy_mean:.3f} ± {row.logistic_balanced_accuracy_sd:.3f} | {row.knn20_balanced_accuracy_mean:.3f} ± {row.knn20_balanced_accuracy_sd:.3f} |")
    lines += [
        "",
        "![UMAP separation metrics](umap_separation_metrics.png)",
        "",
        "解释：五模型等量样本的随机邻居纯度基线为 0.20。纯度、轮廓系数和二维分类准确率越高，表示在当前独立 UMAP 坐标中模型标签越容易被恢复。它们只说明二维表示包含模型相关结构，不代表哪个模型更优。",
        "",
        "前9轮中，二维 logistic balanced accuracy 为 " + f"{first9_umap['logistic_balanced_accuracy_mean']:.3f}" + "，20-NN 同模型纯度为 " + f"{first9_umap['knn20_same_model_purity']:.3f}" + "；这说明当前 fit 中模型分布不是完全混合的。",
        "",
        "## 二、原始 ECFP4 空间验证",
        "",
        "| scope | 组内平均 Tanimoto | 组间平均 Tanimoto | 组内−组间 |",
        "|---|---:|---:|---:|",
    ]
    for row in ecfp_metrics.itertuples(index=False):
        lines.append(f"| {row.scope} | {row.within_model_mean_tanimoto:.3f} | {row.between_model_mean_tanimoto:.3f} | {row.within_minus_between:.3f} |")
    lines += [
        "",
        "![ECFP4 Tanimoto heatmaps](ecfp4_tanimoto_heatmaps.png)",
        "",
        f"前9轮平衡样本的组内平均 Tanimoto 为 {first9['within_model_mean_tanimoto']:.3f}，组间平均 Tanimoto 为 {first9['between_model_mean_tanimoto']:.3f}，差值为 {first9['within_minus_between']:.3f}。这支持五模型生成分布存在原始指纹空间差异，而不是二维 UMAP 单独制造出的视觉分离。",
        "",
        "热图对角线是同一模型内部相似度，非对角线是模型间相似度。差异较大的模型对可以作为后续结构/描述符分析的重点，但不能将 Tanimoto 差异直接解释为 docking 或活性差异。",
        "",
        "## 三、前3、前6、前9轮如何解读",
        "",
        "- 前3轮的 fit 已经形成模型相关结构，说明差异不是只在追加 rep13–rep15 后才出现。",
        "- 前6轮加入 `rep7–rep9` 后，各模型的相对形态仍保持可辨认；这说明新增的 500 步重复没有把五个分布完全混合。",
        "- 前9轮加入 `rep13–rep15` 后，图形更稳定、样本量更大，但不能比较三张图中同一模型的 UMAP 坐标平移，因为每张图都重新建立了邻域和坐标系。",
        "- 如果要研究“追加重复是否改变分布”，应比较原始 ECFP4 的 MMD、Tanimoto 分布、scaffold 覆盖和 descriptor 分布，而不是比较三张独立 UMAP 的 centroid 坐标。",
        "",
        "前9轮 UMAP 中按 centroid 距离/平均组内 RMS 半径最大的模型对为 " + f"{nearest_pair['model_a']}–{nearest_pair['model_b']}，比值 {nearest_pair['distance_over_mean_rms_radius']:.2f}" + "。该比值只是当前二维图内的相对分离描述，不是原始化学距离。",
        "",
        "## 四、对模型和结构的结论边界",
        "",
        "目前可以支持的结论是：五个生成器在 500 步下产生了可区分的 ECFP4 化学空间分布，而且这种差异在三个累计 fit 中均可观察。",
        "",
        "目前不能支持的结论是：某个模型的分离更远就质量更高、结构更简单、docking 更好或更接近真实活性分子。合法率也必须与分子质量分开解释；例如合法率高不等于结构质量高，UMAP分离明显也不等于目标结合更强。",
        "",
        "下一步若要把这些结果用于模型选择，应继续接入全部合法分子的描述符、Murcko scaffold、环/拓扑风险、ECFP4 MMD、UniDock、ligand efficiency 和独立 PoseBusters/结构检查，并保持 1,000 fixed attempts 分母。",
        "",
        "## 证据文件",
        "",
        "- `cumulative_validity_summary.csv`：固定 attempts 合法性汇总。",
        "- `umap_separation_metrics.csv`：三组 UMAP 内部描述性分离指标。",
        "- `umap_centroid_pairwise.csv`：每组独立坐标系内的质心距离和归一化分离。",
        "- `ecfp4_scope_summary.csv`：原始 ECFP4 组内/组间汇总。",
        "- `ecfp4_pairwise_tanimoto.csv`：原始 ECFP4 五模型两两 Tanimoto 矩阵。",
        "- `first3/`、`first6/`、`first9/`：原始 fit 坐标、模型和输入指纹。",
    ]
    (output_dir / "detailed_analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    """Run the detailed analysis from already completed fit artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    configure_logging(output_dir)
    validity = pd.read_csv(output_dir / "cumulative_validity_summary.csv")
    umap_rows: list[pd.DataFrame] = []
    centroid_metrics: dict[str, pd.DataFrame] = {}
    ecfp_rows: list[pd.DataFrame] = []
    pairwise: dict[str, pd.DataFrame] = {}
    for scope in SCOPE_ORDER:
        table = pd.read_csv(output_dir / scope / "coordinates.csv.gz")
        matrix = sparse.load_npz(output_dir / scope / "fit_ecfp4_csr.npz").tocsr()
        LOGGER.info("analyzing scope=%s n=%d", scope, len(table))
        if matrix.shape[0] != len(table):
            raise ValueError(f"matrix/table mismatch for {scope}: {matrix.shape} vs {len(table)}")
        umap_metric, centroid = calculate_umap_metrics(table)
        umap_metric["scope"] = scope
        umap_rows.append(umap_metric)
        centroid["scope"] = scope
        centroid_metrics[scope] = centroid
        centroid.to_csv(output_dir / scope / "centroid_pairwise.csv", index=False)
        ecfp_metric, pair_frame = calculate_ecfp4_metrics(table, matrix)
        ecfp_metric["scope"] = scope
        ecfp_rows.append(ecfp_metric)
        pair_frame["scope"] = scope
        pairwise[scope] = pair_frame
        LOGGER.info("scope=%s silhouette=%.3f within_tanimoto=%.3f between_tanimoto=%.3f", scope, umap_metric.iloc[0]["silhouette_2d"], ecfp_metric.iloc[0]["within_model_mean_tanimoto"], ecfp_metric.iloc[0]["between_model_mean_tanimoto"])
    umap_metrics = pd.concat(umap_rows, ignore_index=True)
    ecfp_metrics = pd.concat(ecfp_rows, ignore_index=True)
    centroid_all = pd.concat(centroid_metrics.values(), ignore_index=True)
    pairwise_all = pd.concat(pairwise.values(), ignore_index=True)
    umap_metrics.to_csv(output_dir / "umap_separation_metrics.csv", index=False)
    centroid_all.to_csv(output_dir / "umap_centroid_pairwise.csv", index=False)
    ecfp_metrics.to_csv(output_dir / "ecfp4_scope_summary.csv", index=False)
    pairwise_all.to_csv(output_dir / "ecfp4_pairwise_tanimoto.csv", index=False)
    plot_umap_metrics(umap_metrics, output_dir)
    plot_ecfp_heatmaps(pairwise, output_dir)
    write_report(output_dir, validity, umap_metrics, ecfp_metrics, centroid_metrics)
    LOGGER.info("completed detailed analysis")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
