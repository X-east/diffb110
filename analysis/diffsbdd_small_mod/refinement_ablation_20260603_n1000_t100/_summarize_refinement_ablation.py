from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
DIFFSBDD_ROOT = PROJECT_ROOT / "DiffSBDD"
if str(DIFFSBDD_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFSBDD_ROOT))

from constants import dataset_params  # noqa: E402


GROUPS = [
    ("baseline", "baseline"),
    ("atom_prior_mask_only", "atom_prior_mask"),
    ("geometry_refine_only", "geometry_refine"),
    ("atom_prior_mask_geometry_refine", "atom_prior_mask+geometry_refine"),
]
ELEMENTS = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B", "others"]
QUALITY_FIELDS = [
    ("sanitize", "mean_sanitize_success"),
    ("bad_bond_length", "mean_bad_bond_length_rate"),
    ("three_member_rings", "mean_three_member_ring_count"),
    ("benzene_like", "mean_benzene_like_ring_count"),
    ("abnormal_aromatic", "mean_abnormal_aromatic_ring_count"),
    ("QED", "mean_qed"),
    ("TPSA", "mean_tpsa"),
    ("sec_per_mol", "mean_reconstruction_seconds"),
]


def _read_summary(group_dir: Path) -> list[dict[str, str]]:
    with (group_dir / "generated_reconstruction_summary.csv").open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], field: str) -> float:
    return float(row[field])


def _element_field(element: str) -> str:
    return "mean_others_fraction" if element == "others" else f"mean_{element.lower()}_fraction"


def _target_distribution() -> dict[str, float]:
    info = dataset_params["crossdock_full"]
    hist = info["atom_hist"]
    total = float(sum(hist.values()))
    return {element: float(hist.get(element, 0.0)) / total for element in ELEMENTS}


def _atom_score_rows(group: str) -> dict[str, object]:
    path = ROOT / group / "atom_score_diagnostics" / "atom_score_summary.json"
    if not path.exists():
        return {
            "atom_score_changed_fraction": "",
            "atom_score_changed_count": "",
            "atom_score_n": "",
        }
    data = json.loads(path.read_text())
    return {
        "atom_score_changed_fraction": data.get("changed_fraction", ""),
        "atom_score_changed_count": data.get("changed_count", ""),
        "atom_score_n": data.get("n_atom_scores", ""),
    }


def build_rows() -> list[dict[str, object]]:
    rows = []
    for group_dir_name, group_label in GROUPS:
        for summary_row in _read_summary(ROOT / group_dir_name):
            row: dict[str, object] = {
                "group": group_label,
                "group_dir": group_dir_name,
                "method": summary_row["method"],
                "samples": summary_row["samples"],
            }
            for output_name, source_field in QUALITY_FIELDS:
                row[output_name] = _float(summary_row, source_field)
            for element in ELEMENTS:
                row[f"{element}_fraction"] = _float(summary_row, _element_field(element))
            if summary_row["method"] == "openbabel":
                row.update(_atom_score_rows(group_dir_name))
            else:
                row.update({
                    "atom_score_changed_fraction": "",
                    "atom_score_changed_count": "",
                    "atom_score_n": "",
                })
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_quality(openbabel_rows: list[dict[str, object]]) -> None:
    groups = [str(row["group"]) for row in openbabel_rows]
    fields = ["sanitize", "bad_bond_length", "three_member_rings", "benzene_like", "abnormal_aromatic", "QED"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, field in zip(axes.flatten(), fields):
        values = [float(row[field]) for row in openbabel_rows]
        ax.bar(groups, values, color="#4C78A8")
        ax.set_title(field)
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(ROOT / "openbabel_quality_metrics.png", dpi=180)
    plt.close(fig)


def plot_elements(openbabel_rows: list[dict[str, object]]) -> None:
    target = _target_distribution()
    x_positions = list(range(len(ELEMENTS)))
    width = 0.8 / (len(openbabel_rows) + 1)
    fig, ax = plt.subplots(figsize=(13, 5))
    for idx, row in enumerate(openbabel_rows):
        values = [float(row[f"{element}_fraction"]) for element in ELEMENTS]
        offsets = [x - 0.4 + width / 2 + idx * width for x in x_positions]
        ax.bar(offsets, values, width=width, label=str(row["group"]))
    target_offsets = [x - 0.4 + width / 2 + len(openbabel_rows) * width for x in x_positions]
    ax.bar(target_offsets, [target[element] for element in ELEMENTS], width=width, label="crossdock_full_target")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(ELEMENTS)
    ax.set_ylabel("atom fraction")
    ax.legend(ncols=2)
    fig.tight_layout()
    fig.savefig(ROOT / "openbabel_element_fraction_vs_target.png", dpi=180)
    plt.close(fig)


def plot_delta(openbabel_rows: list[dict[str, object]]) -> None:
    baseline = openbabel_rows[0]
    fields = ["sanitize", "bad_bond_length", "three_member_rings", "benzene_like", "abnormal_aromatic", "QED", "TPSA"]
    groups = [str(row["group"]) for row in openbabel_rows[1:]]
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.8 / len(fields)
    x_positions = list(range(len(groups)))
    for field_idx, field in enumerate(fields):
        values = [float(row[field]) - float(baseline[field]) for row in openbabel_rows[1:]]
        offsets = [x - 0.4 + width / 2 + field_idx * width for x in x_positions]
        ax.bar(offsets, values, width=width, label=field)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(groups, rotation=15, ha="right")
    ax.set_ylabel("delta vs baseline")
    ax.legend(ncols=2)
    fig.tight_layout()
    fig.savefig(ROOT / "openbabel_delta_vs_baseline.png", dpi=180)
    plt.close(fig)


def _fmt(value: object, digits: int = 3) -> str:
    if value == "":
        return ""
    return f"{float(value):.{digits}f}"


def write_report(rows: list[dict[str, object]]) -> None:
    openbabel_rows = [row for row in rows if row["method"] == "openbabel"]
    target = _target_distribution()
    baseline = openbabel_rows[0]
    report = ROOT / "refinement_ablation_report.md"
    with report.open("w") as handle:
        handle.write("# DiffSBDD generation refinement ablation\n\n")
        handle.write("## Setup\n\n")
        handle.write("- samples per group: `1000`\n")
        handle.write("- timesteps: `100`\n")
        handle.write("- seed: `260603`\n")
        handle.write("- reconstruction methods: `openbabel edm`\n")
        handle.write("- atom prior: CrossDocked target vs baseline generated element fractions, log-ratio bias, strength `0.5`, clamp `0.5`\n")
        handle.write("- local atom mask: terminal-only `F,Cl,Br,I`; globally disabled `P,Br,I,B`\n")
        handle.write("- geometry refine: final-only, `10` steps, lr `0.05`, max total shift `0.30 A`\n\n")

        handle.write("## OpenBabel summary\n\n")
        handle.write("| group | sanitize | bad bond | 3-rings | benzene-like | abnormal arom | C | N | O | F | Cl | QED | TPSA | atom-score changed |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in openbabel_rows:
            handle.write(
                f"| {row['group']} | {_fmt(row['sanitize'])} | {_fmt(row['bad_bond_length'])} "
                f"| {_fmt(row['three_member_rings'])} | {_fmt(row['benzene_like'])} "
                f"| {_fmt(row['abnormal_aromatic'])} | {_fmt(row['C_fraction'])} "
                f"| {_fmt(row['N_fraction'])} | {_fmt(row['O_fraction'])} "
                f"| {_fmt(row['F_fraction'])} | {_fmt(row['Cl_fraction'])} "
                f"| {_fmt(row['QED'])} | {_fmt(row['TPSA'])} "
                f"| {_fmt(row['atom_score_changed_fraction'])} |\n"
            )

        handle.write("\n## Target element fractions\n\n")
        handle.write("| element | target | baseline | delta baseline-target |\n")
        handle.write("|---|---:|---:|---:|\n")
        for element in ELEMENTS:
            baseline_fraction = float(baseline[f"{element}_fraction"])
            handle.write(
                f"| {element} | {target[element]:.4f} | {baseline_fraction:.4f} "
                f"| {baseline_fraction - target[element]:+.4f} |\n"
            )

        handle.write("\n## Interpretation\n\n")
        handle.write("- Atom prior/local mask moved N/O upward and F downward, but did not improve sanitize, bad bond length, rings, or QED in this setting.\n")
        handle.write("- Geometry refine at the current conservative setting barely changed the quality metrics, suggesting the default final-only energy is too weak or the point-cloud errors are not locally correctable by small shifts.\n")
        handle.write("- The combined group behaved like atom prior plus a very small geometry perturbation; there is no evidence yet that this combination should enter the main generation path as-is.\n")
        handle.write("- The result supports keeping these mechanisms as configurable ablation tools, not enabling them by default before tuning.\n\n")

        handle.write("## Atom-score diagnostics\n\n")
        handle.write("The atom prior did not strongly change the model decisions:\n\n")
        handle.write("| item | value |\n")
        handle.write("|---|---:|\n")
        handle.write("| atom scores | 22991 |\n")
        handle.write("| changed top-1 decisions | 362 |\n")
        handle.write("| changed fraction | 0.0157 |\n")
        handle.write("| raw C top-1 count | 17044 |\n")
        handle.write("| C changed to non-C | 8 |\n")
        handle.write("| raw F top-1 count | 585 |\n")
        handle.write("| F kept as F | 243 |\n\n")
        handle.write(
            "This means the current log-ratio bias is too weak relative to the model's atom-score margin. "
            "Raw top-1 minus top-2 margin mean was `0.855`; the C bias was only `-0.048`. "
            "The local mask mostly changed terminal/high-risk atoms such as F and P, but it did not materially reduce carbon dominance.\n\n"
        )
        handle.write(
            "There is also a design conflict: CrossDocked target has about `1.11%` P, "
            "but the current local mask globally disables P. That is safe for early generation "
            "but not chemically faithful if phosphates/phosphonates are expected.\n\n"
        )
        handle.write("## Next tuning plan\n\n")
        handle.write(
            "The current settings should not enter the main generation/RL path by default. "
            "The next useful step is a smaller parameter sweep:\n\n"
        )
        handle.write("| direction | sweep | keep criterion |\n")
        handle.write("|---|---|---|\n")
        handle.write("| atom prior calibration | `strength=1.0/2.0/3.0`, `temperature=1.2/1.5/2.0` | C fraction decreases, N/O approach target, sanitize and QED do not collapse |\n")
        handle.write("| local mask repair | stop globally disabling P; only suppress P in clearly implausible local environments | P is not forced to zero, while valence failures do not increase |\n")
        handle.write("| geometry refine strength | max total shift `0.5/0.8`, steps `20/50`, stronger bond/degree weights | bad bond length and three-member rings decrease without large coordinate drift |\n")
        handle.write("| point-cloud diagnosis | near-neighbor degree, clash, and candidate-bond distance distributions | decide whether final-only refinement has room, or model-internal geometry loss is required |\n\n")

        handle.write("## Plots\n\n")
        handle.write("![quality](openbabel_quality_metrics.png)\n\n")
        handle.write("![elements](openbabel_element_fraction_vs_target.png)\n\n")
        handle.write("![delta](openbabel_delta_vs_baseline.png)\n\n")

        handle.write("## Files\n\n")
        handle.write("- [all_method_refinement_ablation_summary.csv](all_method_refinement_ablation_summary.csv)\n")
        handle.write("- [openbabel_refinement_ablation_summary.csv](openbabel_refinement_ablation_summary.csv)\n")
        for group_dir, group_label in GROUPS:
            handle.write(f"- [{group_label} report]({group_dir}/generated_reconstruction_benchmark_report.md)\n")


def main() -> None:
    rows = build_rows()
    write_csv(ROOT / "all_method_refinement_ablation_summary.csv", rows)
    openbabel_rows = [row for row in rows if row["method"] == "openbabel"]
    write_csv(ROOT / "openbabel_refinement_ablation_summary.csv", openbabel_rows)
    plot_quality(openbabel_rows)
    plot_elements(openbabel_rows)
    plot_delta(openbabel_rows)
    write_report(rows)
    print(f"wrote {ROOT / 'refinement_ablation_report.md'}")


if __name__ == "__main__":
    main()
