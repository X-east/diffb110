"""计算 DiffGui ring-last3 RL 训练过程中的 R1-R5 结构病理漂移。

目的：把第二章"RL 提升了打分"与第三章"分子没变好"用同一套 R1-R5 指标锁死。

用法：
    conda run -n dif python analysis/diffgui_rl_r1r5_drift/compute_ring_last3_r1r5_drift.py
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import Crippen, Descriptors, QED  # noqa: E402

from scripts.offline_rl.structure_pathology_rules import (  # noqa: E402
    compute_structure_pathology_from_smiles,
)

RDLogger.DisableLog("rdApp.*")
LOGGER = logging.getLogger("diffgui-rl-r1r5-drift")

DEFAULT_RUN = "runs/diffgui/diffgui-8gpu-ring-last3-lr3e5_20260610_114650"
RULES = ("r1", "r2", "r3", "r4a", "r4b", "r4c", "r5")


def parse_smiles_and_flags(df: pd.DataFrame) -> pd.DataFrame:
    """从 molecules.csv 提取分子及其奖励/对接字段。

    保留成功与失败分子，以便在 evaluate_epoch 内分别统计：
    失败分子不参与 R1-R5 病理率计算（无有效结构），但参与成功率统计。
    """
    out = []
    for _, row in df.iterrows():
        smi = str(row.get("canonical_smiles") or "").strip()
        if not smi or smi.lower() in ("nan", "none"):
            smi = str(row.get("smiles") or "").strip()
        out.append(
            {
                "smiles": "" if smi.lower() in ("nan", "none") else smi,
                "reward_scored": _to_float(row.get("reward_total")),
                "reward_for_advantage": _to_float(row.get("reward_for_advantage")),
                "docking_success": _to_int(row.get("docking_success")),
                "reconstruction_success": _to_int(
                    row.get("reconstruction_success")
                ),
                "rdkit_valid": _to_int(row.get("rdkit_valid")),
            }
        )
    return pd.DataFrame(out)


def _to_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _to_int(v) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return 0


def evaluate_epoch(csv_path: Path) -> dict | None:
    """对单个 epoch 的 molecules.csv 计算 R1-R5 病理率与奖励均值。"""
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("读取失败 %s: %s", csv_path, exc)
        return None

    sub = parse_smiles_and_flags(df)
    if sub.empty:
        LOGGER.warning("无可用数据: %s", csv_path)
        return None

    # R1-R5 病理率仅在有有效 SMILES 的分子上计算，避免把失效分子计入分母
    scored = sub[sub["smiles"] != ""]
    if scored.empty:
        LOGGER.warning("无有效 SMILES: %s", csv_path)
        return None

    flags = {r: [] for r in RULES}
    r1r4_any, r1r5_any = [], []
    desc_rows = []

    for smi in scored["smiles"]:
        # 理化描述子（hacking 诊断用）
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise ValueError("unparsable")
            heavy = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
            carbons = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6)
            hetero = sum(
                1
                for a in mol.GetAtoms()
                if a.GetAtomicNum() not in (1, 6)
            )
            desc_rows.append(
                {
                    "carbon_fraction": carbons / heavy if heavy else float("nan"),
                    "n_heavy": heavy,
                    "n_hetero": hetero,
                    "tpsa": Descriptors.TPSA(mol),
                    "logp": Crippen.MolLogP(mol),
                    "qed": QED.qed(mol),
                    "molwt": Descriptors.MolWt(mol),
                }
            )
        except Exception:  # noqa: BLE001
            desc_rows.append(
                {
                    "carbon_fraction": float("nan"),
                    "n_heavy": 0,
                    "n_hetero": 0,
                    "tpsa": float("nan"),
                    "logp": float("nan"),
                    "qed": float("nan"),
                    "molwt": float("nan"),
                }
            )

        try:
            facts = compute_structure_pathology_from_smiles(smi)
        except Exception:  # noqa: BLE001
            for r in RULES:
                flags[r].append(False)
            r1r4_any.append(False)
            r1r5_any.append(False)
            continue

        vals = {
            "r1": bool(getattr(facts, "r1_large_ring_violation", False)),
            "r2": bool(getattr(facts, "r2_ring_system_violation", False)),
            "r3": bool(getattr(facts, "r3_four_ring_violation", False)),
            "r4a": bool(getattr(facts, "r4a_three_ring_hetero_violation", False)),
            "r4b": bool(
                getattr(facts, "r4b_three_ring_over_attached_violation", False)
            ),
            "r4c": bool(
                getattr(facts, "r4c_three_ring_spiro_or_fused_violation", False)
            ),
            "r5": bool(getattr(facts, "r5_carbon_fraction_violation", False)),
        }
        for r in RULES:
            flags[r].append(vals[r])
        r1r4_any.append(any(vals[r] for r in ("r1", "r2", "r3", "r4a", "r4b", "r4c")))
        r1r5_any.append(any(vals.values()))

    n = len(scored)
    res = {
        "epoch": int(csv_path.parent.name.split("_")[-1]),
        "n_total": len(sub),
        "n_scored": n,
        "reconstruction_success_ratio": float(sub["reconstruction_success"].mean()),
        "rdkit_valid_ratio": float(sub["rdkit_valid"].mean()),
        # 含失败 -1.0 惩罚的口径，对应 training_summary 的 reward_total_mean
        "reward_for_advantage_mean": float(
            np.nanmean(sub["reward_for_advantage"])
        ),
        # 仅成功打分分子的均值，反映单分子质量
        "reward_scored_mean": float(np.nanmean(scored["reward_scored"])),
        "docking_success_ratio": float(sub["docking_success"].mean()),
    }
    for r in RULES:
        res[f"{r}_violation"] = float(np.mean(flags[r]))
    res["r1r4_any"] = float(np.mean(r1r4_any))
    res["r1r5_any"] = float(np.mean(r1r5_any))
    res["reasonable_ratio_r1r5"] = 1.0 - res["r1r5_any"]

    # 理化描述子聚合
    ddf = pd.DataFrame(desc_rows)
    for col in ("carbon_fraction", "n_hetero", "tpsa", "logp", "qed", "molwt"):
        if col in ddf.columns:
            res[f"{col}_mean"] = float(np.nanmean(ddf[col]))
    if "carbon_fraction" in ddf.columns:
        res["frac_carbon_ge_0p90"] = float(
            np.nanmean(ddf["carbon_fraction"].fillna(-1) >= 0.90)
        )
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", default=DEFAULT_RUN)
    ap.add_argument(
        "--epochs",
        default="1,4,8,12,16,20,23",
        help="要评估的 epoch 编号，逗号分隔",
    )
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    epochs_dir = run_dir / "epochs"
    if not epochs_dir.exists():
        LOGGER.error("epochs 目录不存在: %s", epochs_dir)
        return 1

    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for ep in args.epochs.split(","):
        ep = ep.strip()
        if not ep:
            continue
        csv_path = epochs_dir / f"epoch_{int(ep):04d}" / "molecules.csv"
        res = evaluate_epoch(csv_path)
        if res is None:
            LOGGER.warning("跳过 epoch %s", ep)
            continue
        rows.append(res)
        LOGGER.info(
            "epoch %-3s n=%-4d recon=%.4f  reward(惩罚)=%7.4f  reward(成功)=%7.4f"
            "  R1-R4=%.4f  R1-R5=%.4f  合理率=%.4f",
            res["epoch"],
            res["n_scored"],
            res["reconstruction_success_ratio"],
            res["reward_for_advantage_mean"],
            res["reward_scored_mean"],
            res["r1r4_any"],
            res["r1r5_any"],
            res["reasonable_ratio_r1r5"],
        )

    if not rows:
        LOGGER.error("无有效结果")
        return 1

    df = pd.DataFrame(rows).sort_values("epoch").reset_index(drop=True)

    stem = run_dir.name
    csv_out = out_dir / f"{stem}_r1r5_drift.csv"
    df.to_csv(csv_out, index=False, float_format="%.6f")
    LOGGER.info("已写出 %s", csv_out)

    # 首尾变化摘要
    a, b = df.iloc[0], df.iloc[-1]
    summary = {
        "run_dir": str(run_dir),
        "epoch_first": int(a["epoch"]),
        "epoch_last": int(b["epoch"]),
        "n_scored_first": int(a["n_scored"]),
        "n_scored_last": int(b["n_scored"]),
        "reconstruction_success_first": float(a["reconstruction_success_ratio"]),
        "reconstruction_success_last": float(b["reconstruction_success_ratio"]),
        "reward_for_advantage_first": float(a["reward_for_advantage_mean"]),
        "reward_for_advantage_last": float(b["reward_for_advantage_mean"]),
        "reward_scored_first": float(a["reward_scored_mean"]),
        "reward_scored_last": float(b["reward_scored_mean"]),
        "docking_success_ratio_first": float(a["docking_success_ratio"]),
        "docking_success_ratio_last": float(b["docking_success_ratio"]),
        "r1r4_first": float(a["r1r4_any"]),
        "r1r4_last": float(b["r1r4_any"]),
        "r1r5_first": float(a["r1r5_any"]),
        "r1r5_last": float(b["r1r5_any"]),
        "reasonable_r1r5_first": float(a["reasonable_ratio_r1r5"]),
        "reasonable_r1r5_last": float(b["reasonable_ratio_r1r5"]),
    }
    for r in RULES:
        summary[f"{r}_first"] = float(a[f"{r}_violation"])
        summary[f"{r}_last"] = float(b[f"{r}_violation"])
    for col in (
        "carbon_fraction_mean",
        "n_hetero_mean",
        "tpsa_mean",
        "logp_mean",
        "qed_mean",
        "frac_carbon_ge_0p90",
    ):
        if col in df.columns:
            summary[f"{col}_first"] = float(a[col])
            summary[f"{col}_last"] = float(b[col])

    # 奖励增益分解：for_advantage = p*(scored+1) - 1，p = 重建成功率
    p1, p2 = (
        summary["reconstruction_success_first"],
        summary["reconstruction_success_last"],
    )
    s1, s2 = summary["reward_scored_first"], summary["reward_scored_last"]
    total_gain = (p2 * (s2 + 1) - 1) - (p1 * (s1 + 1) - 1)
    from_validity = (p2 - p1) * (s1 + 1)
    from_quality = p2 * (s2 - s1)
    summary["reward_gain_total"] = float(total_gain)
    summary["reward_gain_from_validity"] = float(from_validity)
    summary["reward_gain_from_quality"] = float(from_quality)
    summary["reward_gain_validity_share"] = (
        float(from_validity / total_gain) if abs(total_gain) > 1e-9 else float("nan")
    )
    summary["reward_gain_quality_share"] = (
        float(from_quality / total_gain) if abs(total_gain) > 1e-9 else float("nan")
    )

    json_out = out_dir / f"{stem}_r1r5_drift_summary.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    LOGGER.info("已写出 %s", json_out)

    print("\n=== 首尾变化摘要（epoch %d -> %d）===" % (a["epoch"], b["epoch"]))
    print(
        "重建成功率        %.4f -> %.4f"
        % (p1, p2)
    )
    print(
        "reward(含失败惩罚) %.4f -> %.4f  (%+.4f)"
        % (
            summary["reward_for_advantage_first"],
            summary["reward_for_advantage_last"],
            summary["reward_for_advantage_last"]
            - summary["reward_for_advantage_first"],
        )
    )
    print(
        "reward(仅成功分子) %.4f -> %.4f  (%+.4f)"
        % (s1, s2, s2 - s1)
    )
    print(
        "docking_success   %.4f -> %.4f"
        % (
            summary["docking_success_ratio_first"],
            summary["docking_success_ratio_last"],
        )
    )
    print(
        "R1-R4 病理率      %.4f -> %.4f"
        % (summary["r1r4_first"], summary["r1r4_last"])
    )
    print(
        "R1-R5 病理率      %.4f -> %.4f"
        % (summary["r1r5_first"], summary["r1r5_last"])
    )
    print(
        "R1-R5 合理率      %.4f -> %.4f"
        % (summary["reasonable_r1r5_first"], summary["reasonable_r1r5_last"])
    )
    print("\n--- 理化性质（hacking 诊断）---")
    for label, key in (
        ("碳比例", "carbon_fraction_mean"),
        ("杂原子数", "n_hetero_mean"),
        ("TPSA", "tpsa_mean"),
        ("LogP", "logp_mean"),
        ("QED", "qed_mean"),
        ("C>=0.90 占比", "frac_carbon_ge_0p90"),
    ):
        k1, k2 = f"{key}_first", f"{key}_last"
        if k1 in summary:
            print(
                "%-14s %8.4f -> %8.4f  (%+.4f)"
                % (label, summary[k1], summary[k2], summary[k2] - summary[k1])
            )
    print(
        "\n奖励增益分解：总计 %+.4f = 成功率贡献 %+.4f (%.1f%%) + 单分子质量贡献 %+.4f (%.1f%%)"
        % (
            total_gain,
            from_validity,
            100 * summary["reward_gain_validity_share"],
            from_quality,
            100 * summary["reward_gain_quality_share"],
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
