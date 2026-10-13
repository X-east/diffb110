#!/usr/bin/env python3
"""
从 diffgui_step_big_table_all_metrics.csv 提取结构与组成描述子指标，
输出 markdown 与 CSV。
"""
import csv
import os

REPO = "/home/md666/data/disk6/DiffB110"
DIR = os.path.join(
    REPO,
    "analysis/benchmark/diffgui_step_sweep_100_200_500_1000_stats_20260710",
)
SRC = os.path.join(DIR, "diffgui_step_big_table_all_metrics.csv")

METRICS = [
    ("ring_count_mean", "总环数", "Ring count"),
    ("aromatic_ring_count_mean", "芳香环数", "Aromatic ring count"),
    ("has_aromatic_ring_rate", "含芳香环比例", "Has aromatic ring"),
    ("three_member_ring_count_mean", "三元环数", "Three-member ring count"),
    ("has_three_member_ring_rate", "含三元环比例", "Has three-member ring"),
    ("carbon_fraction_mean", "碳比例", "Carbon fraction"),
    ("hetero_atoms_mean", "杂原子数", "Hetero atoms"),
    ("n_fraction_mean", "氮比例", "N fraction"),
    ("o_fraction_mean", "氧比例", "O fraction"),
    ("s_fraction_mean", "硫比例", "S fraction"),
    ("cl_fraction_mean", "氯比例", "Cl fraction"),
    ("f_fraction_mean", "氟比例", "F fraction"),
    ("br_fraction_mean", "溴比例", "Br fraction"),
    ("rotatable_bonds_mean", "可旋转键数", "Rotatable bonds"),
    ("mw_mean", "分子量", "Molecular weight"),
    ("heavy_atoms_mean", "重原子数", "Heavy atoms"),
    ("num_bonds_mean", "键数", "Number of bonds"),
]


def main():
    rows = {}
    with open(SRC, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[row["metric"]] = row

    csv_path = os.path.join(DIR, "diffgui_step_structure_metrics_table.csv")
    md_path = os.path.join(DIR, "diffgui_step_structure_metrics_table.md")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["指标(metric)", "中文名", "100_steps", "200_steps", "500_steps", "1000_steps"])
        for key, cn, en in METRICS:
            row = rows.get(key, {})
            w.writerow([en, cn, row.get("100_steps", ""), row.get("200_steps", ""), row.get("500_steps", ""), row.get("1000_steps", "")])

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# DiffGui step-sweep 结构与组成描述子指标\n\n")
        f.write("数值为 3 rep 的 mean ± SD。\n\n")
        f.write("| 指标 | 中文名 | 100 steps | 200 steps | 500 steps | 1000 steps |\n")
        f.write("| --- | --- | --- | --- | --- | --- |\n")
        for key, cn, en in METRICS:
            row = rows.get(key, {})
            vals = [row.get(c, "") for c in ["100_steps", "200_steps", "500_steps", "1000_steps"]]
            f.write(f"| {en} | {cn} | {vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} |\n")

    print(f"已写出:\n  {csv_path}\n  {md_path}")


if __name__ == "__main__":
    main()
