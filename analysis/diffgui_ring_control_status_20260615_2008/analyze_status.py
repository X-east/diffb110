from __future__ import annotations

import json
import math
import subprocess
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("/home/md666/data/disk6/DiffB110")
OUT_DIR = ROOT / "analysis" / "diffgui_ring_control_status_20260615_2008"


def _fmt(value: object, ndigits: int = 3) -> str:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return "NA"
        return f"{float(value):.{ndigits}f}"
    except Exception:
        return str(value)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    completed = [
        ("09_qed", ROOT / "OUT" / "diffgui-ring-control-20260615-09-qed_20260615_122202"),
        ("10_sa", ROOT / "OUT" / "diffgui-ring-control-20260615-10-sa_20260615_163654"),
    ]
    current = (
        "11_ring_complexity",
        ROOT / "OUT" / "diffgui-ring-control-20260615-11-ring_complexity_20260615_200439",
    )

    cols = [
        "epoch",
        "status",
        "batch_size",
        "total_seconds",
        "rollout_seconds",
        "reward_seconds",
        "ppo_seconds",
        "reward_total_mean",
        "reward_total_scored_mean",
        "reward_for_advantage_mean",
        "docking_raw_mean",
        "docking_reward_mean",
        "ligand_efficiency_raw_mean",
        "molecule_reconstruction_success_ratio",
        "molecule_rdkit_valid_ratio",
        "molecule_docking_success_ratio",
        "unique_canonical_smiles",
        "duplicate_ratio",
        "unique_scaffolds",
        "scaffold_top1_fraction",
        "logp_mean",
        "tpsa_mean",
        "hba_mean",
        "hbd_mean",
        "carbon_fraction",
        "hetero_atom_mean",
        "aromatic_ring_mean",
        "ring_count_mean",
        "three_member_ring_mean",
        "three_member_ring_total",
        "descriptor_mmd_vs_baseline",
        "PPO_ApproxKL",
        "ClipFrac",
        "RatioMean",
        "GradNorm",
        "checkpoint",
        "failure_counts",
    ]

    frames: list[pd.DataFrame] = []
    run_rows: list[dict[str, object]] = []
    for slug, run_dir in completed:
        df = pd.read_csv(run_dir / "training_summary.csv", usecols=lambda c: c in cols)
        df.insert(0, "run", slug)
        frames.append(df)

        first = df.iloc[0]
        last = df.iloc[-1]
        checkpoint = Path(str(last.get("checkpoint", "")))
        run_rows.append(
            {
                "run": slug,
                "run_dir": str(run_dir.relative_to(ROOT)),
                "status": "completed"
                if (run_dir / "summary.json").exists() and checkpoint.exists()
                else "incomplete_artifacts",
                "epochs_recorded": int(df["epoch"].max()),
                "first_reward_total_mean": float(first["reward_total_mean"]),
                "last_reward_total_mean": float(last["reward_total_mean"]),
                "reward_total_delta": float(last["reward_total_mean"] - first["reward_total_mean"]),
                "last_reward_total_scored_mean": float(last["reward_total_scored_mean"]),
                "last_docking_raw_mean": float(last["docking_raw_mean"]),
                "last_reconstruction_ratio": float(last["molecule_reconstruction_success_ratio"]),
                "last_rdkit_valid_ratio": float(last["molecule_rdkit_valid_ratio"]),
                "last_docking_success_ratio": float(last["molecule_docking_success_ratio"]),
                "last_descriptor_mmd": float(last["descriptor_mmd_vs_baseline"]),
                "last_carbon_fraction": float(last["carbon_fraction"]),
                "last_tpsa_mean": float(last["tpsa_mean"]),
                "last_hba_mean": float(last["hba_mean"]),
                "last_ring_count_mean": float(last["ring_count_mean"]),
                "last_three_member_ring_total": float(last["three_member_ring_total"]),
                "last_ppo_approxkl": float(last["PPO_ApproxKL"]),
                "last_clipfrac": float(last["ClipFrac"]),
                "checkpoint_exists": checkpoint.exists(),
                "summary_exists": (run_dir / "summary.json").exists(),
                "manifest_exists": (run_dir / "run_manifest.json").exists(),
            }
        )

    completed_df = pd.concat(frames, ignore_index=True)
    summary_df = pd.DataFrame(run_rows)
    completed_df.to_csv(OUT_DIR / "completed_epoch_metrics.csv", index=False)
    summary_df.to_csv(OUT_DIR / "completed_run_summary.csv", index=False)
    summary_df.to_json(OUT_DIR / "completed_run_summary.json", orient="records", indent=2, force_ascii=False)

    current_slug, current_dir = current
    manifest_path = current_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    rank_csvs = sorted((current_dir / "epochs" / "epoch_0001").glob("rank_*/molecules_with_rewards.csv"))
    rank_frames = []
    for csv_path in rank_csvs:
        rank_df = pd.read_csv(csv_path)
        rank_df["rank_dir"] = csv_path.parent.name
        rank_frames.append(rank_df)
    current_df = pd.concat(rank_frames, ignore_index=True) if rank_frames else pd.DataFrame()

    current_summary: dict[str, object] = {
        "run": current_slug,
        "run_dir": str(current_dir.relative_to(ROOT)),
        "manifest_status": manifest.get("status"),
        "created_at": manifest.get("created_at"),
        "updated_at": manifest.get("updated_at"),
        "world_size": manifest.get("world_size"),
        "global_batch_size": manifest.get("global_batch_size"),
        "per_rank_batch_size": manifest.get("per_rank_batch_size"),
        "config_path": manifest.get("config_path"),
        "rank_reward_csv_count": len(rank_csvs),
    }
    if not current_df.empty:
        current_summary.update(
            {
                "epoch1_reward_rows_now": int(len(current_df)),
                "epoch1_rdkit_valid_ratio_now": float(current_df["rdkit_valid"].mean()),
                "epoch1_reconstruction_success_ratio_now": float(current_df["reconstruction_success"].mean()),
                "epoch1_docking_success_ratio_now": float(current_df["docking_success"].mean()),
                "epoch1_reward_total_mean_now": float(current_df["reward_total"].mean()),
                "epoch1_reward_total_min_now": float(current_df["reward_total"].min()),
                "epoch1_reward_total_max_now": float(current_df["reward_total"].max()),
                "epoch1_docking_raw_mean_now": float(current_df["docking_raw"].mean()),
                "epoch1_ligand_efficiency_mean_now": float(current_df["ligand_efficiency_raw"].mean()),
                "epoch1_ring_complexity_reward_mean_now": float(current_df["ring_complexity_reward"].mean()),
                "epoch1_ring_count_mean_now": float(current_df["ring_complexity_ring_count"].mean()),
                "epoch1_aromatic_ring_count_mean_now": float(
                    current_df["ring_complexity_aromatic_ring_count"].mean()
                ),
                "epoch1_three_member_ring_count_mean_now": float(
                    current_df["ring_complexity_three_member_ring_count"].mean()
                ),
                "epoch1_max_fused_ring_system_size_mean_now": float(
                    current_df["ring_complexity_max_fused_ring_system_size"].mean()
                ),
            }
        )
        current_df.to_csv(OUT_DIR / "current_11_epoch1_molecules_with_rewards_aggregated.csv", index=False)
    else:
        (OUT_DIR / "current_11_epoch1_molecules_with_rewards_aggregated.csv").write_text("", encoding="utf-8")

    try:
        smi = subprocess.check_output(["nvidia-smi"], text=True, stderr=subprocess.STDOUT, timeout=10)
    except Exception as exc:
        smi = f"failed: {exc}"
    (OUT_DIR / "nvidia_smi_snapshot.txt").write_text(smi, encoding="utf-8")
    proc_lines = [line for line in smi.splitlines() if "python" in line.lower() and " C " in line]
    proc_rows = []
    for line in proc_lines:
        tokens = line.replace("|", " ").split()
        if len(tokens) < 7:
            continue
        mem_token = tokens[-1]
        proc_rows.append(
            {
                "gpu": tokens[0],
                "pid": tokens[3],
                "process_name": tokens[5],
                "used_memory_mib": int(mem_token[:-3]) if mem_token.endswith("MiB") else None,
                "raw_line": line.strip(),
            }
        )
    process_source = "internal_nvidia_smi"
    internal_error = smi if smi.startswith("failed:") else ""
    external_proc_path = OUT_DIR / "current_diffgui_processes_external.csv"
    if not proc_rows and external_proc_path.exists() and external_proc_path.stat().st_size > 0:
        external_df = pd.read_csv(
            external_proc_path,
            header=None,
            names=["pid", "process_name", "used_memory_mib", "gpu_uuid"],
            skipinitialspace=True,
        )
        external_df = external_df[external_df["process_name"].astype(str).str.contains("diffgui", case=False, na=False)]
        proc_rows = [
            {
                "gpu": row["gpu_uuid"],
                "pid": str(row["pid"]),
                "process_name": row["process_name"],
                "used_memory_mib": int(row["used_memory_mib"]),
                "raw_line": f"{row['pid']}, {row['process_name']}, {row['used_memory_mib']}, {row['gpu_uuid']}",
            }
            for _, row in external_df.iterrows()
        ]
        process_source = "external_nvidia_smi_query"
    proc_df = pd.DataFrame(proc_rows)
    proc_df.to_csv(OUT_DIR / "current_diffgui_processes.csv", index=False)
    current_summary["nvidia_smi_checked_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    current_summary["nvidia_smi_process_source"] = process_source
    current_summary["nvidia_smi_internal_error"] = internal_error
    current_summary["nvidia_smi_diffgui_process_count"] = int(len(proc_rows))
    current_summary["nvidia_smi_diffgui_gpu_count"] = int(proc_df["gpu"].nunique()) if not proc_df.empty else 0
    current_summary["nvidia_smi_diffgui_pids"] = proc_df["pid"].tolist() if not proc_df.empty else []
    current_summary["nvidia_smi_diffgui_gpu_mem_mib_total"] = (
        int(proc_df["used_memory_mib"].sum()) if not proc_df.empty else 0
    )
    current_summary["nvidia_smi_process_snapshot"] = [row["raw_line"] for row in proc_rows]

    pd.DataFrame([current_summary]).to_csv(OUT_DIR / "current_run_status.csv", index=False)
    (OUT_DIR / "current_run_status.json").write_text(
        json.dumps(current_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    plt.style.use("default")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for run, group in completed_df.groupby("run"):
        axes[0, 0].plot(group["epoch"], group["reward_total_mean"], marker="o", label=f"{run} total")
        axes[0, 0].plot(
            group["epoch"],
            group["reward_total_scored_mean"],
            marker="x",
            linestyle="--",
            label=f"{run} scored",
        )
        axes[0, 1].plot(group["epoch"], group["molecule_reconstruction_success_ratio"], marker="o", label=f"{run} recon")
        axes[0, 1].plot(
            group["epoch"],
            group["molecule_rdkit_valid_ratio"],
            marker="x",
            linestyle="--",
            label=f"{run} valid",
        )
        axes[1, 0].plot(group["epoch"], group["descriptor_mmd_vs_baseline"], marker="o", label=run)
        axes[1, 1].plot(group["epoch"], group["docking_raw_mean"], marker="o", label=run)
    axes[0, 0].set_title("Reward trajectory")
    axes[0, 0].set_xlabel("epoch")
    axes[0, 0].set_ylabel("reward mean")
    axes[0, 1].set_title("Reconstruction and validity")
    axes[0, 1].set_xlabel("epoch")
    axes[0, 1].set_ylabel("ratio")
    axes[1, 0].set_title("Descriptor MMD vs baseline")
    axes[1, 0].set_xlabel("epoch")
    axes[1, 0].set_ylabel("MMD")
    axes[1, 1].set_title("Docking raw mean")
    axes[1, 1].set_xlabel("epoch")
    axes[1, 1].set_ylabel("score")
    for ax in axes.flat:
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.savefig(OUT_DIR / "completed_runs_reward_validity.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    metrics = [
        ("carbon_fraction", "Carbon fraction"),
        ("hetero_atom_mean", "Hetero atom mean"),
        ("tpsa_mean", "TPSA mean"),
        ("hba_mean", "HBA mean"),
        ("ring_count_mean", "Ring count mean"),
        ("three_member_ring_total", "Three-member ring total"),
    ]
    for ax, (metric, title) in zip(axes.flat, metrics):
        for run, group in completed_df.groupby("run"):
            ax.plot(group["epoch"], group[metric], marker="o", label=run)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.savefig(OUT_DIR / "completed_runs_descriptor_trends.png", dpi=180)
    plt.close(fig)

    if not current_df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        axes[0].hist(current_df["reward_total"], bins=12, color="#4C78A8", edgecolor="white")
        axes[0].set_title("Current 11 epoch 1 reward_total")
        axes[0].set_xlabel("reward_total")
        axes[0].set_ylabel("molecules")
        max_ring = int(current_df["ring_complexity_ring_count"].max())
        axes[1].hist(
            current_df["ring_complexity_ring_count"],
            bins=range(0, max_ring + 3),
            color="#F58518",
            edgecolor="white",
            align="left",
        )
        axes[1].set_title("Current 11 epoch 1 ring count")
        axes[1].set_xlabel("ring count")
        axes[1].set_ylabel("molecules")
        for ax in axes:
            ax.grid(True, alpha=0.25)
        fig.savefig(OUT_DIR / "current_11_epoch1_reward_ring_distribution.png", dpi=180)
        plt.close(fig)

    report_lines = [
        "# DiffGui 训练状态快速分析",
        "",
        f"- 检查时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "- 判断：矩阵训练没有全部结束；09_qed 和 10_sa 已完成，11_ring_complexity 正在跑，12_le_ring_aromatic 与 13_dock_ring_aromatic 尚未启动。",
        "- 证据：`launch/logs/diffgui_ring_control_matrix_20260615/_master.log` 显示 09/10 done，11 start；当前 run_manifest 仍为 running。",
        "",
        "## 已完成 run 汇总",
        "",
        "| run | epochs | final reward_total | delta | final scored reward | recon | valid | docking | descriptor MMD | carbon | TPSA | HBA | ring | 3-ring total | KL | checkpoint |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in run_rows:
        report_lines.append(
            f"| {row['run']} | {row['epochs_recorded']} | {_fmt(row['last_reward_total_mean'])} | "
            f"{_fmt(row['reward_total_delta'])} | {_fmt(row['last_reward_total_scored_mean'])} | "
            f"{_fmt(row['last_reconstruction_ratio'])} | {_fmt(row['last_rdkit_valid_ratio'])} | "
            f"{_fmt(row['last_docking_success_ratio'])} | {_fmt(row['last_descriptor_mmd'])} | "
            f"{_fmt(row['last_carbon_fraction'])} | {_fmt(row['last_tpsa_mean'])} | "
            f"{_fmt(row['last_hba_mean'])} | {_fmt(row['last_ring_count_mean'])} | "
            f"{_fmt(row['last_three_member_ring_total'], 1)} | {_fmt(row['last_ppo_approxkl'])} | "
            f"{row['checkpoint_exists']} |"
        )
    report_lines.extend(
        [
            "",
            "## 当前 11_ring_complexity",
            "",
            f"- run_dir：`{current_summary['run_dir']}`",
            f"- manifest_status：`{current_summary.get('manifest_status')}`；world_size={current_summary.get('world_size')}，global_batch_size={current_summary.get('global_batch_size')}。",
            f"- 当前已聚合 epoch 1 reward CSV：{current_summary.get('rank_reward_csv_count')} 个 rank，{current_summary.get('epoch1_reward_rows_now', 0)} 条分子记录。",
        ]
    )
    if current_summary.get("epoch1_reward_rows_now"):
        report_lines.extend(
            [
                f"- epoch 1 当前 reward_total_mean={_fmt(current_summary.get('epoch1_reward_total_mean_now'))}，ring_count_mean={_fmt(current_summary.get('epoch1_ring_count_mean_now'))}，aromatic_ring_count_mean={_fmt(current_summary.get('epoch1_aromatic_ring_count_mean_now'))}，three_member_ring_count_mean={_fmt(current_summary.get('epoch1_three_member_ring_count_mean_now'))}。",
                f"- 当前 rdkit_valid_ratio={_fmt(current_summary.get('epoch1_rdkit_valid_ratio_now'))}，reconstruction_success_ratio={_fmt(current_summary.get('epoch1_reconstruction_success_ratio_now'))}，docking_success_ratio={_fmt(current_summary.get('epoch1_docking_success_ratio_now'))}。",
            ]
        )
    report_lines.extend(
        [
            f"- nvidia-smi 进程快照：{current_summary.get('nvidia_smi_diffgui_process_count')} 个 DiffGui Python 进程，覆盖 {current_summary.get('nvidia_smi_diffgui_gpu_count')} 张 GPU；PID={','.join(current_summary.get('nvidia_smi_diffgui_pids', []))}；source={current_summary.get('nvidia_smi_process_source')}。",
            "- 注：`conda run -n dif` 子进程内调用 `nvidia-smi` 返回 exit 9；GPU 进程判断以外部 shell 的 `nvidia-smi` 快照和 query CSV 为准。",
            "",
            "## 初步分析",
            "",
            "- 09_qed/10_sa 都是正常完成：均到 epoch 10，写出了 `training_summary.csv`、`summary.json` 和 epoch 10 checkpoint。",
            "- 质量上不是简单单调上升：reward_total 有波动；descriptor MMD 在 10_sa 末端低于 09_qed，说明 SA 这段的整体分布漂移略小。",
            "- 分子偏移仍在：末端 carbon_fraction 仍约 0.75-0.78，TPSA 约 45，低 TPSA/低 HBA 比例仍偏高，需要后续 11-13 的环结构/LE+docking 组合来观察能否压住。",
            "- PPO 更新幅度偏大但未明显崩：epoch 10 KL 约 2-3，ClipFrac 约 0.03-0.04；生成/重建成功率约 0.86-0.90。",
            "",
            "## 图表与数据",
            "",
            "- [completed_run_summary.csv](completed_run_summary.csv)",
            "- [completed_epoch_metrics.csv](completed_epoch_metrics.csv)",
            "- [current_run_status.csv](current_run_status.csv)",
            "- [current_diffgui_processes.csv](current_diffgui_processes.csv)",
            "- [current_diffgui_processes_external.csv](current_diffgui_processes_external.csv)",
            "- [nvidia_smi_external_snapshot.txt](nvidia_smi_external_snapshot.txt)",
            "- [current_11_epoch1_molecules_with_rewards_aggregated.csv](current_11_epoch1_molecules_with_rewards_aggregated.csv)",
            "",
            "![completed reward and validity](completed_runs_reward_validity.png)",
            "",
            "![completed descriptor trends](completed_runs_descriptor_trends.png)",
            "",
        ]
    )
    if not current_df.empty:
        report_lines.extend(
            [
                "![current 11 epoch1 reward and ring distribution](current_11_epoch1_reward_ring_distribution.png)",
                "",
            ]
        )
    (OUT_DIR / "report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(OUT_DIR)
    print(summary_df[["run", "epochs_recorded", "last_reward_total_mean", "reward_total_delta", "last_rdkit_valid_ratio", "last_descriptor_mmd"]].to_string(index=False))
    print(json.dumps(current_summary, ensure_ascii=False, indent=2)[:2000])


if __name__ == "__main__":
    main()
