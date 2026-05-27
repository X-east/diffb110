"""MolSkill 区分度分析：生成分子 vs ChEMBL 真实分子。

对比 7gpu-dock-ecfp4-fcfp4 实验所有 epoch 的生成分子与 ChEMBL 采样分子，
用 MolSkill RankNet 打分，评估模型是否能将两者区分开。

输出：
    - KDE 分布对比图
    - 分 epoch 的 MolSkill 分数趋势
    - 统计摘要表（均值、中位数、标准差、P5/P95、KS test）
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

# ── 项目路径 ──
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "reference project" / "molskill"))

from molskill.data.featurizers import get_featurizer  # noqa: E402
from molskill.models.ranknet import LitRankNet  # noqa: E402

import torch  # noqa: E402
from rdkit import Chem  # noqa: E402

# ── 配置 ──────────────────────────────────────────────────────────────────
GENERATED_DIR = str(_PROJECT_ROOT / "OUT" / "7gpu-dock-ecfp4-fcfp4_20260526_115341" / "epochs")
CHEMBL_CSV = str(_PROJECT_ROOT / "data" / "chembl" / "chembl_36_properties.csv.gz")
CHECKPOINT_PATH = str(
    _PROJECT_ROOT / "reference project" / "molskill" / "models" / "default" / "checkpoints" / "last.ckpt"
)
OUTPUT_DIR = str(_PROJECT_ROOT / "OUT" / "7gpu-dock-ecfp4-fcfp4_20260526_115341" / "analysis_figures")
CHEMBL_N_SAMPLE = 10_000  # ChEMBL 采样量
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 256


def load_generated_smiles(base_dir: str) -> dict[int, list[str]]:
    """读取所有 epoch 的有效生成分子 SMILES，返回 {epoch: [smiles, ...]}."""
    epoch_smiles: dict[int, list[str]] = {}
    for ep_name in sorted(os.listdir(base_dir)):
        ep_dir = os.path.join(base_dir, ep_name)
        if not os.path.isdir(ep_dir):
            continue
        epoch = int(ep_name.split("_")[1])
        smiles_list: list[str] = []
        for fname in sorted(os.listdir(ep_dir)):
            if not fname.startswith("molecules_rank") or not fname.endswith(".csv"):
                continue
            with open(os.path.join(ep_dir, fname)) as fh:
                for row in csv.DictReader(fh):
                    if row["rdkit_valid"] == "1":
                        smiles_list.append(row["smiles"])
        epoch_smiles[epoch] = smiles_list
        print(f"  epoch {epoch:4d}: {len(smiles_list):5d} valid molecules")
    return epoch_smiles


def load_chembl_smiles(csv_path: str, n_sample: int) -> list[str]:
    """从 ChEMBL CSV 中随机采样 SMILES（排除 NaN/空值）。"""
    df = pd.read_csv(csv_path, usecols=["canonical_smiles"], dtype=str)
    df = df.dropna()
    df = df[df["canonical_smiles"].str.len() > 0]
    sampled = df.sample(n=min(n_sample, len(df)), random_state=42)
    return sampled["canonical_smiles"].tolist()


def molskill_score_batch(
    smiles_list: list[str],
    featurizer,
    model: LitRankNet,
    device: str = DEVICE,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """批量打分，返回 numpy 数组 [N]."""
    scores: list[float] = []
    for start in range(0, len(smiles_list), batch_size):
        batch_smiles = smiles_list[start : start + batch_size]
        # Featurize
        features_list = []
        valid_idx = []
        for i, smi in enumerate(batch_smiles):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            try:
                feat = torch.from_numpy(
                    featurizer.get_feat(mol).astype(np.float32)
                )
                features_list.append(feat)
                valid_idx.append(i)
            except Exception:
                pass

        if not features_list:
            scores.extend([float("nan")] * len(batch_smiles))
            continue

        features = torch.stack(features_list).to(device)
        with torch.inference_mode():
            raw = model.net.score(features).squeeze(-1).cpu().numpy()

        # 对齐回原始顺序
        batch_scores = [float("nan")] * len(batch_smiles)
        for vi, s in zip(valid_idx, raw):
            batch_scores[vi] = float(s)
        scores.extend(batch_scores)

    return np.array(scores, dtype=np.float32)


def plot_kde_comparison(
    generated_scores: np.ndarray,
    chembl_scores: np.ndarray,
    ks_stat: float,
    ks_pvalue: float,
    output_path: str,
) -> None:
    """绘制生成分子 vs ChEMBL 的 KDE 对比图."""
    from scipy.stats import gaussian_kde

    gen = generated_scores[~np.isnan(generated_scores)]
    chm = chembl_scores[~np.isnan(chembl_scores)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── 左图：全范围 KDE ──
    ax = axes[0]
    xmin = min(gen.min(), chm.min()) - 2
    xmax = max(gen.max(), chm.max()) + 2
    xs = np.linspace(xmin, xmax, 500)

    if len(gen) > 1:
        kde_gen = gaussian_kde(gen)
        ax.fill_between(xs, kde_gen(xs), alpha=0.3, label=f"Generated (n={len(gen):,})", color="#E74C3C")
        ax.plot(xs, kde_gen(xs), color="#E74C3C", lw=1.5)
    if len(chm) > 1:
        kde_chm = gaussian_kde(chm)
        ax.fill_between(xs, kde_chm(xs), alpha=0.3, label=f"ChEMBL (n={len(chm):,})", color="#2980B9")
        ax.plot(xs, kde_chm(xs), color="#2980B9", lw=1.5)

    ax.axvline(gen.mean(), color="#E74C3C", ls="--", lw=1, alpha=0.6)
    ax.axvline(chm.mean(), color="#2980B9", ls="--", lw=1, alpha=0.6)
    ax.set_xlabel("MolSkill Score")
    ax.set_ylabel("Density")
    ax.set_title(
        f"MolSkill Score Distribution\n"
        f"KS stat={ks_stat:.4f}, p={ks_pvalue:.2e}" if ks_pvalue < 0.001
        else f"KS stat={ks_stat:.4f}, p={ks_pvalue:.4f}"
    )
    ax.legend()

    # ── 右图：zoom 到 [-20, 20] ──
    ax = axes[1]
    xs_zoom = np.linspace(-20, 20, 500)
    if len(gen) > 1:
        kde_gen_z = gaussian_kde(gen)
        ax.fill_between(xs_zoom, kde_gen_z(xs_zoom), alpha=0.3, color="#E74C3C")
        ax.plot(xs_zoom, kde_gen_z(xs_zoom), color="#E74C3C", lw=1.5)
    if len(chm) > 1:
        kde_chm_z = gaussian_kde(chm)
        ax.fill_between(xs_zoom, kde_chm_z(xs_zoom), alpha=0.3, color="#2980B9")
        ax.plot(xs_zoom, kde_chm_z(xs_zoom), color="#2980B9", lw=1.5)

    ax.axvline(gen.mean(), color="#E74C3C", ls="--", lw=1, alpha=0.6)
    ax.axvline(chm.mean(), color="#2980B9", ls="--", lw=1, alpha=0.6)
    ax.set_xlabel("MolSkill Score")
    ax.set_ylabel("Density")
    ax.set_title("Zoom: score ∈ [-20, 20]")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] KDE comparison saved: {output_path}")


def plot_epoch_trend(epoch_data: dict[int, np.ndarray], chembl_mean: float, output_path: str) -> None:
    """绘制分 epoch 的 MolSkill 分数趋势."""
    epochs = sorted(epoch_data.keys())
    means = []
    medians = []
    p5s = []
    p95s = []
    for ep in epochs:
        s = epoch_data[ep][~np.isnan(epoch_data[ep])]
        means.append(float(np.mean(s)))
        medians.append(float(np.median(s)))
        p5s.append(float(np.percentile(s, 5)))
        p95s.append(float(np.percentile(s, 95)))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(epochs, p5s, p95s, alpha=0.15, color="#E74C3C", label="P5-P95")
    ax.plot(epochs, means, "o-", color="#E74C3C", lw=2, markersize=4, label="Mean")
    ax.plot(epochs, medians, "s-", color="#C0392B", lw=1.5, markersize=4, alpha=0.7, label="Median")
    ax.axhline(chembl_mean, color="#2980B9", ls="--", lw=2, label=f"ChEMBL mean ({chembl_mean:.2f})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MolSkill Score")
    ax.set_title("Generated Molecule MolSkill Scores by Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] Epoch trend saved: {output_path}")


def print_stats_table(label: str, scores: np.ndarray) -> dict:
    """打印并返回统计量."""
    clean = scores[~np.isnan(scores)]
    stats = {
        "count": len(clean),
        "mean": float(np.mean(clean)),
        "median": float(np.median(clean)),
        "std": float(np.std(clean)),
        "min": float(np.min(clean)),
        "max": float(np.max(clean)),
        "p5": float(np.percentile(clean, 5)),
        "p95": float(np.percentile(clean, 95)),
        "nan_rate": float(np.mean(np.isnan(scores))) * 100,
    }
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  N         : {stats['count']:,}")
    print(f"  NaN rate  : {stats['nan_rate']:.1f}%")
    print(f"  Mean      : {stats['mean']:.4f}")
    print(f"  Median    : {stats['median']:.4f}")
    print(f"  Std       : {stats['std']:.4f}")
    print(f"  Min       : {stats['min']:.4f}")
    print(f"  Max       : {stats['max']:.4f}")
    print(f"  P5        : {stats['p5']:.4f}")
    print(f"  P95       : {stats['p95']:.4f}")
    return stats


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Device: {DEVICE}")

    # ── 1. 加载模型和 featurizer ──────────────────────────────────────────
    print("\n[1/4] Loading MolSkill model ...")
    featurizer = get_featurizer("morgan_count_rdkit_2d")
    model = LitRankNet.load_from_checkpoint(
        checkpoint_path=CHECKPOINT_PATH,
        input_size=featurizer.dim(),
        map_location=DEVICE,
    )
    model.eval()
    model.to(DEVICE)
    print(f"  Model loaded, input dim = {featurizer.dim()}")

    # ── 2. 加载生成分子 SMILES ───────────────────────────────────────────
    print("\n[2/4] Loading generated SMILES ...")
    epoch_smiles = load_generated_smiles(GENERATED_DIR)
    all_generated_smiles: list[str] = []
    for smiles_list in epoch_smiles.values():
        all_generated_smiles.extend(smiles_list)
    print(f"  Total valid generated molecules: {len(all_generated_smiles):,}")

    # ── 3. 加载 ChEMBL SMILES ────────────────────────────────────────────
    print(f"\n[3/4] Loading ChEMBL SMILES (n_sample={CHEMBL_N_SAMPLE}) ...")
    chembl_smiles = load_chembl_smiles(CHEMBL_CSV, CHEMBL_N_SAMPLE)
    print(f"  ChEMBL sampled: {len(chembl_smiles):,}")

    # ── 4. MolSkill 打分 ──────────────────────────────────────────────────
    print("\n[4/4] Scoring with MolSkill ...")
    print(f"  Scoring {len(all_generated_smiles):,} generated molecules ...")
    gen_scores = molskill_score_batch(all_generated_smiles, featurizer, model, DEVICE)
    print(f"  Scoring {len(chembl_smiles):,} ChEMBL molecules ...")
    chembl_scores = molskill_score_batch(chembl_smiles, featurizer, model, DEVICE)

    # ── 统计 ──────────────────────────────────────────────────────────────
    gen_stats = print_stats_table("Generated Molecules", gen_scores)
    chembl_stats = print_stats_table("ChEMBL Molecules", chembl_scores)

    ks_stat, ks_pvalue = ks_2samp(
        gen_scores[~np.isnan(gen_scores)],
        chembl_scores[~np.isnan(chembl_scores)],
    )
    effect_size = (gen_stats["mean"] - chembl_stats["mean"]) / chembl_stats["std"] if chembl_stats["std"] > 0 else 0.0
    print(f"\n{'='*60}")
    print(f"  Distinction Metrics")
    print(f"{'='*60}")
    print(f"  KS statistic          : {ks_stat:.4f}")
    print(f"  KS p-value            : {ks_pvalue:.4e}")
    print(f"  Cohen's d (effect)    : {effect_size:.4f}")
    print(f"  Mean diff (gen-chembl): {gen_stats['mean'] - chembl_stats['mean']:.4f}")

    # ── KDE 图 ────────────────────────────────────────────────────────────
    plot_kde_comparison(
        gen_scores, chembl_scores, ks_stat, ks_pvalue,
        os.path.join(OUTPUT_DIR, "09_molskill_kde_comparison.png"),
    )

    # ── 分 epoch 趋势 ─────────────────────────────────────────────────────
    epoch_data: dict[int, np.ndarray] = {}
    offset = 0
    for ep in sorted(epoch_smiles.keys()):
        n = len(epoch_smiles[ep])
        epoch_data[ep] = gen_scores[offset : offset + n]
        offset += n
    plot_epoch_trend(
        epoch_data, chembl_stats["mean"],
        os.path.join(OUTPUT_DIR, "10_molskill_epoch_trend.png"),
    )

    print(f"\nDone. Figures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
