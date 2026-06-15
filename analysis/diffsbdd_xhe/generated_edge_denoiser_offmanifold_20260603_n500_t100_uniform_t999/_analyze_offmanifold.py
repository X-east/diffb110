"""Write focused analysis for generated point-cloud edge-denoiser reconstruction."""

from __future__ import annotations

import csv
import os
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


OUTDIR = Path("analysis/generated_edge_denoiser_offmanifold_20260603_n500_t100_uniform_t999")
PER_MOL_CSV = OUTDIR / "generated_reconstruction_per_molecule.csv"


def _read_rows() -> list[dict[str, str]]:
    """Read per-molecule reconstruction rows."""
    with PER_MOL_CSV.open() as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], key: str) -> float:
    """Read a float field."""
    value = row.get(key, "")
    return float(value) if value != "" else 0.0


def _mean(values: list[float]) -> float:
    """Return mean or zero."""
    return sum(values) / len(values) if values else 0.0


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write rows to CSV."""
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _group(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    """Group rows by method."""
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)
    return grouped


def _focus_summary(grouped: dict[str, list[dict[str, str]]]) -> list[dict[str, object]]:
    """Build all-molecule method summary."""
    summary = []
    for method, rows in grouped.items():
        summary.append({
            "method": method,
            "samples": len(rows),
            "sanitize_rate": _mean([_float(row, "sanitize_success") for row in rows]),
            "bad_bond_length_rate": _mean([_float(row, "bad_bond_length_rate") for row in rows]),
            "fragments": _mean([_float(row, "num_fragments") for row in rows]),
            "largest_fragment": _mean([_float(row, "largest_fragment_atom_frac") for row in rows]),
            "aromatic_rings": _mean([_float(row, "aromatic_ring_count") for row in rows]),
            "benzene_like_rings": _mean([_float(row, "benzene_like_ring_count") for row in rows]),
            "abnormal_aromatic_rings": _mean([_float(row, "abnormal_aromatic_ring_count") for row in rows]),
            "three_member_rings": _mean([_float(row, "three_member_ring_count") for row in rows]),
            "qed_all": _mean([_float(row, "qed") for row in rows]),
            "tpsa_all": _mean([_float(row, "tpsa") for row in rows]),
            "seconds_per_mol": _mean([_float(row, "reconstruction_seconds") for row in rows]),
        })
    return summary


def _sanitize_subset_summary(grouped: dict[str, list[dict[str, str]]]) -> list[dict[str, object]]:
    """Build method summary restricted to sanitized molecules."""
    summary = []
    for method, rows in grouped.items():
        sanitized = [row for row in rows if int(float(row["sanitize_success"])) == 1]
        summary.append({
            "method": method,
            "sanitized": len(sanitized),
            "sanitize_rate": len(sanitized) / max(1, len(rows)),
            "qed_sanitized": _mean([_float(row, "qed") for row in sanitized]),
            "sa_sanitized": _mean([_float(row, "sa") for row in sanitized]),
            "tpsa_sanitized": _mean([_float(row, "tpsa") for row in sanitized]),
            "logp_sanitized": _mean([_float(row, "logp") for row in sanitized]),
            "aromatic_rings_sanitized": _mean([_float(row, "aromatic_ring_count") for row in sanitized]),
            "benzene_like_rings_sanitized": _mean([_float(row, "benzene_like_ring_count") for row in sanitized]),
            "abnormal_aromatic_rings_sanitized": _mean([_float(row, "abnormal_aromatic_ring_count") for row in sanitized]),
            "any_aromatic_frac_sanitized": _mean([
                float(_float(row, "aromatic_ring_count") > 0.0) for row in sanitized
            ]),
            "any_benzene_frac_sanitized": _mean([
                float(_float(row, "benzene_like_ring_count") > 0.0) for row in sanitized
            ]),
            "any_abnormal_aromatic_frac_sanitized": _mean([
                float(_float(row, "abnormal_aromatic_ring_count") > 0.0) for row in sanitized
            ]),
            "three_member_rings_sanitized": _mean([_float(row, "three_member_ring_count") for row in sanitized]),
        })
    return summary


def _error_counts(grouped: dict[str, list[dict[str, str]]]) -> list[dict[str, object]]:
    """Count reconstruction errors by method."""
    rows = []
    for method, method_rows in grouped.items():
        counter = Counter(row["error"] or "success" for row in method_rows)
        for error, count in counter.most_common():
            rows.append({
                "method": method,
                "error": error,
                "count": count,
                "fraction": count / max(1, len(method_rows)),
            })
    return rows


def _plot_focus(focus: list[dict[str, object]], subset: list[dict[str, object]]) -> None:
    """Write compact comparison plots."""
    methods = [str(row["method"]) for row in focus]

    plt.figure(figsize=(8, 4))
    x = range(len(methods))
    width = 0.25
    plt.bar([i - width for i in x], [float(row["sanitize_rate"]) for row in focus], width=width, label="sanitize")
    plt.bar([i for i in x], [float(row["qed_all"]) for row in focus], width=width, label="QED all")
    plt.bar([i + width for i in x], [float(row["tpsa_all"]) / 100.0 for row in focus], width=width, label="TPSA / 100")
    plt.xticks(list(x), methods, rotation=15, ha="right")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "focused_sanitize_qed_tpsa.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.bar([i - width for i in x], [float(row["aromatic_rings"]) for row in focus], width=width, label="aromatic")
    plt.bar([i for i in x], [float(row["benzene_like_rings"]) for row in focus], width=width, label="benzene-like")
    plt.bar([i + width for i in x], [float(row["abnormal_aromatic_rings"]) for row in focus], width=width, label="abnormal aromatic")
    plt.xticks(list(x), methods, rotation=15, ha="right")
    plt.ylabel("rings / molecule")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "focused_benzene_aromatic.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.bar(
        methods,
        [float(row["any_abnormal_aromatic_frac_sanitized"]) for row in subset],
        color="#E45756",
    )
    plt.ylabel("fraction among sanitized molecules")
    plt.xticks(rotation=15, ha="right")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(OUTDIR / "focused_abnormal_aromatic_fraction.png", dpi=180)
    plt.close()


def _fmt(value: object, digits: int = 3) -> str:
    """Format numbers for Markdown."""
    if isinstance(value, str):
        return value
    return f"{float(value):.{digits}f}"


def _write_markdown(
    focus: list[dict[str, object]],
    subset: list[dict[str, object]],
    errors: list[dict[str, object]],
) -> None:
    """Write detailed Markdown analysis."""
    focus_by_method = {str(row["method"]): row for row in focus}
    subset_by_method = {str(row["method"]): row for row in subset}
    lines = [
        "# Generated Point Cloud Edge-Denoiser Off-Manifold Analysis",
        "",
        "本分析使用 500 个 DiffSBDD generated point clouds，100 步扩散；edge denoiser 不使用真键 noising，而是 `uniform_random` 初始 edge，`t=999`。",
        "",
        "## Key Takeaway",
        "",
        "- 在 generated point cloud 上，B/C edge denoiser 的 sanitize 只有约 0.26，明显低于 OpenBabel 的 0.654。",
        "- 这说明 clean CrossDocked 上训练出的 edge denoiser 直接接 DiffSBDD 点云会明显 off-manifold 掉点。",
        "- 苯环方面，OpenBabel 的 benzene-like rings 更多，但 abnormal aromatic rings 也更多；B/C 生成的 benzene-like rings 更少，异常芳香环也少一些，主要问题是整体 sanitize 太低。",
        "",
        "## Overall Summary",
        "",
        "| method | sanitize | bad bond | fragments | aromatic | benzene6 | abnormal arom | 3-rings | QED all | TPSA all | sec/mol |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in focus:
        lines.append(
            f"| {row['method']} | {_fmt(row['sanitize_rate'])} | {_fmt(row['bad_bond_length_rate'])} | "
            f"{_fmt(row['fragments'])} | {_fmt(row['aromatic_rings'])} | {_fmt(row['benzene_like_rings'])} | "
            f"{_fmt(row['abnormal_aromatic_rings'])} | {_fmt(row['three_member_rings'])} | "
            f"{_fmt(row['qed_all'])} | {_fmt(row['tpsa_all'])} | {_fmt(row['seconds_per_mol'], 4)} |"
        )
    lines.extend([
        "",
        "![focused_sanitize_qed_tpsa](focused_sanitize_qed_tpsa.png)",
        "",
        "![focused_benzene_aromatic](focused_benzene_aromatic.png)",
        "",
        "## Sanitized Molecules Only",
        "",
        "| method | sanitized | QED | TPSA | aromatic | benzene6 | any benzene frac | abnormal arom | any abnormal frac | 3-rings |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in subset:
        lines.append(
            f"| {row['method']} | {row['sanitized']} | {_fmt(row['qed_sanitized'])} | "
            f"{_fmt(row['tpsa_sanitized'])} | {_fmt(row['aromatic_rings_sanitized'])} | "
            f"{_fmt(row['benzene_like_rings_sanitized'])} | {_fmt(row['any_benzene_frac_sanitized'])} | "
            f"{_fmt(row['abnormal_aromatic_rings_sanitized'])} | {_fmt(row['any_abnormal_aromatic_frac_sanitized'])} | "
            f"{_fmt(row['three_member_rings_sanitized'])} |"
        )
    lines.extend([
        "",
        "![focused_abnormal_aromatic_fraction](focused_abnormal_aromatic_fraction.png)",
        "",
        "## Error Counts",
        "",
        "| method | error | count | fraction |",
        "|---|---|---:|---:|",
    ])
    for row in errors:
        if str(row["error"]) == "success" or float(row["fraction"]) >= 0.01:
            lines.append(
                f"| {row['method']} | {row['error']} | {row['count']} | {_fmt(row['fraction'])} |"
            )
    lines.extend([
        "",
        "## Interpretation",
        "",
        f"- OpenBabel: sanitize={_fmt(focus_by_method['openbabel']['sanitize_rate'])}，benzene-like={_fmt(focus_by_method['openbabel']['benzene_like_rings'])}/mol，abnormal aromatic={_fmt(focus_by_method['openbabel']['abnormal_aromatic_rings'])}/mol。",
        f"- Edge B: sanitize={_fmt(focus_by_method['edge_denoiser_b']['sanitize_rate'])}，benzene-like={_fmt(focus_by_method['edge_denoiser_b']['benzene_like_rings'])}/mol，abnormal aromatic={_fmt(focus_by_method['edge_denoiser_b']['abnormal_aromatic_rings'])}/mol。",
        f"- Edge C: sanitize={_fmt(focus_by_method['edge_denoiser_c']['sanitize_rate'])}，benzene-like={_fmt(focus_by_method['edge_denoiser_c']['benzene_like_rings'])}/mol，abnormal aromatic={_fmt(focus_by_method['edge_denoiser_c']['abnormal_aromatic_rings'])}/mol。",
        f"- 在 sanitize 成功子集中，OpenBabel any abnormal aromatic fraction={_fmt(subset_by_method['openbabel']['any_abnormal_aromatic_frac_sanitized'])}；B={_fmt(subset_by_method['edge_denoiser_b']['any_abnormal_aromatic_frac_sanitized'])}；C={_fmt(subset_by_method['edge_denoiser_c']['any_abnormal_aromatic_frac_sanitized'])}。",
        "",
        "结论：B/C edge denoiser 不是苯环特别异常，而是生成点云上整体化学闭合失败较多。OpenBabel 虽然 sanitize 更高、苯环更多，但 abnormal aromatic 也更多，说明 OpenBabel 的芳香处理也不完全干净。",
        "",
        "## Files",
        "",
        "- [focused_method_summary.csv](focused_method_summary.csv)",
        "- [focused_sanitized_subset_summary.csv](focused_sanitized_subset_summary.csv)",
        "- [focused_error_counts.csv](focused_error_counts.csv)",
        "- [generated_reconstruction_per_molecule.csv](generated_reconstruction_per_molecule.csv)",
        "- [generated_reconstruction_summary.csv](generated_reconstruction_summary.csv)",
    ])
    (OUTDIR / "offmanifold_detailed_analysis.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    """Run focused analysis."""
    rows = _read_rows()
    grouped = _group(rows)
    focus = _focus_summary(grouped)
    subset = _sanitize_subset_summary(grouped)
    errors = _error_counts(grouped)
    _write_csv(OUTDIR / "focused_method_summary.csv", focus)
    _write_csv(OUTDIR / "focused_sanitized_subset_summary.csv", subset)
    _write_csv(OUTDIR / "focused_error_counts.csv", errors)
    _plot_focus(focus, subset)
    _write_markdown(focus, subset, errors)
    print(OUTDIR / "offmanifold_detailed_analysis.md")


if __name__ == "__main__":
    main()
