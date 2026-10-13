"""Extract the AcrB study's input and hypothesis tables and plot survival scores."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


def extract_table(root: ET.Element, table_id: str) -> tuple[list[str], list[list[str]]]:
    """Read one publication table without interpreting its measurement units."""
    table = root.find(f".//table-wrap[@id='{table_id}']")
    if table is None:
        raise ValueError(f"Missing table {table_id}")
    headers = ["".join(cell.itertext()).strip() for cell in table.findall(".//thead/tr/th")]
    rows = [[" ".join("".join(cell.itertext()).split()) for cell in row.findall("td")]
            for row in table.findall(".//tbody/tr")]
    return headers, rows


def main() -> None:
    """Export original table cells, derived counts, and a readable score comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).parent / "sources/acrb_pharmacophore_2024.xml")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "model_details")
    parser.add_argument("--record-dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    root = ET.parse(args.source).getroot()
    tables = {}
    for table_id, filename in [("tab1", "table1_input_molecules.csv"),
                               ("tab2", "table2_hypothesis_scores.csv"),
                               ("tab3", "table3_qsar_metrics.csv")]:
        headers, rows = extract_table(root, table_id)
        tables[table_id] = (headers, rows)
        with (args.output_dir / filename).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(headers)
            writer.writerows(rows)
    inputs = tables["tab1"][1]
    hypotheses = tables["tab2"][1]
    activity_values = [float(row[4].replace("−", "-")) for row in inputs]
    summary = {
        "source_doi": "10.1039/d3md00483j",
        "input_count": len(inputs), "hypothesis_count": len(hypotheses),
        "rounded_table_pmic_gt_0_1": sum(value > 0.1 for value in activity_values),
        "rounded_table_pmic_lt_0_1": sum(value < 0.1 for value in activity_values),
        "rounded_table_pmic_eq_0_1": sum(value == 0.1 for value in activity_values),
        "threshold_caveat": "Counts use rounded Table 1 values; exact labels and the boundary rule were not published in the main text.",
        "model_geometry": "Not reconstructed; numeric distances and tolerances were not inferred from text.",
    }
    assert len(inputs) == 32 and len(hypotheses) == 10
    ranked = sorted(((row[0], float(row[1])) for row in hypotheses), key=lambda item: item[1], reverse=True)
    summary["top_survival"] = ranked[0]
    summary["top_two_survival_gap"] = round(ranked[0][1] - ranked[1][1], 3)
    (args.output_dir / "model_audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    font = font_manager.FontProperties(fname=font_manager.findfont("Noto Sans CJK JP"))
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.subplots_adjust(bottom=0.18)
    values = [value for _, value in ranked]
    ax.barh(range(len(ranked)), values, color=["#267B70"] + ["#B4C5D5"] * (len(ranked) - 1))
    ax.set_yticks(range(len(ranked)), [name for name, _ in ranked], fontsize=16)
    ax.invert_yaxis()
    ax.set_xlim(0, 7.3)
    ax.set_xlabel("Phase survival score（论文表2）", fontproperties=font, fontsize=17)
    ax.set_title("AHHNR.100按模型评分排第一；前两名差0.010", fontproperties=font, fontsize=20, pad=18)
    ax.tick_params(axis="x", labelsize=14)
    ax.spines[["top", "right"]].set_visible(False)
    for index, value in enumerate(values):
        ax.text(value + 0.08, index, f"{value:.3f}", va="center", fontsize=15)
    fig.text(0.13, 0.025, "模型评分不等于活性概率；此图不评价哪个模型更能发现真实抑制剂。",
             fontproperties=font, fontsize=13)
    paths = []
    for suffix in ["png", "svg"]:
        path = args.output_dir / f"acrb_hypothesis_survival_scores.{suffix}"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    if args.record_dir:
        args.record_dir.mkdir(parents=True, exist_ok=True)
        for path in paths:
            shutil.copy2(path, args.record_dir / path.name)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
