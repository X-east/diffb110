#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
解析 DiffGui step-sweep 各档位(100/200/500/1000) x 3 rep x 7 卡的生成 log.txt，
提取每卡首末时间戳，计算:
  - 墙钟时间(并行) = 同 rep 内 7 卡并行，取最慢单卡耗时
  - 单卡 s/分子     = 单卡耗时 / num_mols (含模型加载)
输出 CSV + Markdown 到 stats 目录。
"""
import os
import re
import csv
import glob
import datetime as dt
from collections import defaultdict

REPO = "/home/md666/data/disk6/DiffB110"
OUT_DIR = os.path.join(
    REPO,
    "analysis/benchmark/diffgui_step_sweep_100_200_500_1000_stats_20260710",
)

# rep 级目录 (每个内部 outputs/diffgui/gpu_1..7/<run>/log.txt)
REP_DIRS = {
    "100": [
        "analysis/benchmark/diffgui_step_sweep_100_1000attempts_20260709/100steps_rep1_seed1_7",
        "analysis/benchmark/diffgui_step_sweep_100_1000attempts_20260709/100steps_rep2_seed8_14",
        "analysis/benchmark/diffgui_step_sweep_100_1000attempts_20260709/100steps_rep3_seed15_21",
    ],
    "200": [
        "analysis/benchmark/diffgui_step_sweep_500_200_1000attempts_20260707_run2/200steps_rep1_seed1_7",
        "analysis/benchmark/diffgui_step_sweep_500_200_1000attempts_20260707_run2/200steps_rep2_seed8_14",
        "analysis/benchmark/diffgui_step_sweep_500_200_1000attempts_20260707_run2/200steps_rep3_seed15_21",
    ],
    "500": [
        "analysis/benchmark/diffgui_step_sweep_500_200_1000attempts_20260707_run2/500steps_rep1_seed1_7",
        "analysis/benchmark/diffgui_step_sweep_500_200_1000attempts_20260707_run2/500steps_rep2_seed8_14",
        "analysis/benchmark/diffgui_step_sweep_500_200_1000attempts_20260707_run2/500steps_rep3_seed15_21",
    ],
    "1000": [
        "analysis/benchmark/8gjl_c1101_benchmark_20260701_diffgui_rep1_1000steps_seed1_7",
        "analysis/benchmark/8gjl_c1101_benchmark_20260701_all_rep2_1000steps_seed8_14",
        "analysis/benchmark/8gjl_c1101_benchmark_20260701_all_rep3_1000steps_seed15_21",
    ],
}

TS_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})::")
NT_RE = re.compile(r"num_timesteps['\"]?\s*[:=]\s*(\d+)")
NM_RE = re.compile(r"num_mols['\"]?\s*[:=]\s*(\d+)")


def parse_log(path):
    ts_list = []
    num_t = num_m = None
    with open(path, "r", errors="replace") as f:
        for line in f:
            m = TS_RE.match(line)
            if m:
                ts_list.append(m.group(1))
            if num_t is None:
                mt = NT_RE.search(line)
                if mt:
                    num_t = int(mt.group(1))
            if num_m is None:
                mm = NM_RE.search(line)
                if mm:
                    num_m = int(mm.group(1))
    if len(ts_list) < 2 or num_t is None or num_m is None:
        return None
    t0 = dt.datetime.strptime(ts_list[0], "%Y-%m-%d %H:%M:%S,%f")
    t1 = dt.datetime.strptime(ts_list[-1], "%Y-%m-%d %H:%M:%S,%f")
    dur = (t1 - t0).total_seconds()
    if dur <= 0:
        return None
    return num_t, num_m, dur


def mean_sd(vals):
    if not vals:
        return float("nan"), float("nan")
    m = sum(vals) / len(vals)
    if len(vals) == 1:
        return m, 0.0
    sd = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
    return m, sd


def main():
    # step -> rep -> list[(gpu, dur_s, num_mols)]
    table = defaultdict(lambda: defaultdict(list))
    detail = []
    for step, roots in REP_DIRS.items():
        for ri, root in enumerate(roots, start=1):
            base = os.path.join(REPO, root)
            for logp in glob.glob(
                os.path.join(base, "outputs", "diffgui", "gpu_*", "*", "log.txt")
            ):
                parsed = parse_log(logp)
                if parsed is None:
                    continue
                num_t, num_m, dur = parsed
                gpu_m = re.search(r"gpu_(\d+)", logp)
                gpu = int(gpu_m.group(1)) if gpu_m else -1
                table[step][ri].append((gpu, dur, num_m))
                detail.append(
                    {
                        "steps": step,
                        "rep": ri,
                        "gpu": gpu,
                        "dur_min": dur / 60.0,
                        "s_per_mol": dur / num_m,
                        "num_mols": num_m,
                        "log": logp,
                    }
                )

    # 每档聚合: 跨 rep 取墙钟(mean±SD) 与 单卡s/mol(mean±SD)
    summary = []
    for step in ["100", "200", "500", "1000"]:
        reps = table.get(step, {})
        wall_per_rep = []
        spm_per_rep = []  # 每 rep 先对 7 卡取均值
        for ri, entries in reps.items():
            durs = [e[1] for e in entries]  # 秒
            nm = entries[0][2]
            wall_per_rep.append(max(durs) / 60.0)  # 并行最慢卡, 转分钟
            spm_per_rep.append(sum(d / nm for d in durs) / len(durs))  # 秒/分子
        w_mean, w_sd = mean_sd(wall_per_rep)
        s_mean, s_sd = mean_sd(spm_per_rep)
        summary.append(
            {
                "steps": step,
                "n_rep": len(reps),
                "wall_mean": w_mean,
                "wall_sd": w_sd,
                "spm_mean": s_mean,
                "spm_sd": s_sd,
            }
        )

    os.makedirs(OUT_DIR, exist_ok=True)

    # CSV 汇总
    sum_csv = os.path.join(OUT_DIR, "diffgui_step_timing_table.csv")
    with open(sum_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "diffusion_steps",
                "n_reps",
                "gpus_per_rep",
                "wall_clock_min_parallel_mean",
                "wall_clock_min_parallel_sd",
                "s_per_mol_gpu_mean",
                "s_per_mol_gpu_sd",
            ]
        )
        for r in summary:
            w.writerow(
                [
                    r["steps"],
                    r["n_rep"],
                    7,
                    f"{r['wall_mean']:.2f}",
                    f"{r['wall_sd']:.2f}",
                    f"{r['spm_mean']:.2f}",
                    f"{r['spm_sd']:.2f}",
                ]
            )

    # CSV 明细
    det_csv = os.path.join(OUT_DIR, "diffgui_step_timing_detail.csv")
    with open(det_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["diffusion_steps", "rep", "gpu", "duration_min", "s_per_mol", "num_mols", "log"]
        )
        for d in sorted(detail, key=lambda x: (int(x["steps"]), x["rep"], x["gpu"])):
            w.writerow(
                [
                    d["steps"],
                    d["rep"],
                    d["gpu"],
                    f"{d['dur_min']:.2f}",
                    f"{d['s_per_mol']:.2f}",
                    d["num_mols"],
                    d["log"],
                ]
            )

    # Markdown 汇总
    md = os.path.join(OUT_DIR, "diffgui_step_timing_table.md")
    with open(md, "w") as f:
        f.write("# DiffGui step-sweep 生成耗时\n\n")
        f.write(
            "墙钟时间(并行) = 同 rep 内 7 卡并行取最慢单卡耗时；"
            "单卡 s/分子 = 单卡耗时 / num_mols (含模型加载)。数值为 3 rep 的 mean ± SD。\n\n"
        )
        f.write(
            "| diffusion_steps | reps | gpus/rep | 墙钟(min,并行) mean±SD | 单卡 s/分子 mean±SD |\n"
        )
        f.write("| --- | --- | --- | --- | --- |\n")
        for r in summary:
            f.write(
                f"| {r['steps']} | {r['n_rep']} | 7 | "
                f"{r['wall_mean']:.2f} ± {r['wall_sd']:.2f} | "
                f"{r['spm_mean']:.2f} ± {r['spm_sd']:.2f} |\n"
            )
        f.write("\n明细见 diffgui_step_timing_detail.csv\n")

    # 终端打印
    print("=== DiffGui step-sweep 生成耗时 (3 rep, 7 gpu/rep) ===\n")
    print(
        f"{'steps':>6} {'reps':>5} {'wall_min(mean±SD)':>22} {'s/mol(mean±SD)':>20}"
    )
    for r in summary:
        print(
            f"{r['steps']:>6} {r['n_rep']:>5} "
            f"{r['wall_mean']:.2f}±{r['wall_sd']:.2f}        "
            f"{r['spm_mean']:.2f}±{r['spm_sd']:.2f}"
        )
    print(f"\n已写出:\n  {sum_csv}\n  {det_csv}\n  {md}")


if __name__ == "__main__":
    main()
