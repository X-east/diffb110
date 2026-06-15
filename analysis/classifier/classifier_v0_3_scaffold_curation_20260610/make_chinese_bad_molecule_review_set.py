from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from rdkit import Chem, RDLogger


BASE_DIR = Path(__file__).resolve().parent
SOURCE_SDF_DIR = BASE_DIR / "sdf_by_reason"
OUTPUT_DIR = BASE_DIR / "bad_molecule_review_maestro_ascii"


@dataclass(frozen=True)
class ReasonZh:
    code: str
    zh: str
    short: str
    severity: str
    note: str


REASONS: tuple[ReasonZh, ...] = (
    ReasonZh(
        code="bad_saturated_polycycle_no_aromatic",
        zh="无芳香环的多饱和碳环",
        short="无芳香多饱和碳环",
        severity="强负样本",
        note="多个饱和碳环堆叠但没有苯环/芳香环，容易成为当前 DiffGui RL 的错误偏好。",
    ),
    ReasonZh(
        code="bad_overhalogenated_small_aromatic",
        zh="小芳香环过度卤代",
        short="小芳香环过度卤代",
        severity="强负样本",
        note="单个小芳香环上连太多卤素，尤其多 Br/Cl/F，分子过小且不自然。",
    ),
    ReasonZh(
        code="bad_many_halogen",
        zh="卤素数量过多",
        short="卤素过多",
        severity="强负样本",
        note="总卤素或重卤素过多；一两个卤素可以接受，五六个通常应作为负样本。",
    ),
    ReasonZh(
        code="bad_fused_heterocycle",
        zh="稠合杂环/复杂杂环",
        short="稠合复杂杂环",
        severity="强负样本",
        note="杂环稠合过多或结构过复杂，常对应视觉上不合理的复杂环体系。",
    ),
    ReasonZh(
        code="bad_poly_fused_or_bridge_ring",
        zh="多稠合或桥环体系",
        short="多稠合或桥环",
        severity="强负样本",
        note="最大稠合环系统过大，或存在桥头/螺原子等复杂环拓扑。",
    ),
    ReasonZh(
        code="bad_large_ring",
        zh="七元及以上大环",
        short="七元及以上大环",
        severity="强负样本",
        note="存在七元及以上大环；尤其是大环又与六元环稠合时，作为当前生成坏模式强负样本。",
    ),
    ReasonZh(
        code="bad_multi_aromatic_heterocycle",
        zh="多芳香杂环",
        short="多芳香杂环",
        severity="负样本",
        note="芳香杂环数量偏多，可能造成 scaffold 过复杂或偏离目标分布。",
    ),
    ReasonZh(
        code="bad_three_member_ring",
        zh="三元环",
        short="三元环",
        severity="强负样本",
        note="三元环张力大；在当前生成结果中大量出现时通常是不合理模式。",
    ),
    ReasonZh(
        code="bad_high_carbon_low_hetero",
        zh="高碳低杂原子",
        short="高碳低杂原子",
        severity="负样本",
        note="碳比例过高且杂原子过少，提示模型往疏水高碳方向漂移。",
    ),
    ReasonZh(
        code="bad_fragmented",
        zh="碎片化分子",
        short="碎片化",
        severity="强负样本",
        note="分子被拆成多个片段，不能作为合理 ligand。",
    ),
)

REASON_BY_CODE = {item.code: item for item in REASONS}
SEVERITY_ASCII = {
    "强负样本": "strong_negative",
    "负样本": "negative",
    "优先查看": "priority_review",
}
BUCKET_TYPE_ASCII = {
    "主原因不重复分桶": "primary_unique",
    "所有触发原因分桶：允许重复": "all_triggered_reason_duplicate_allowed",
    "全部坏分子：主原因不重复合并": "all_bad_primary_unique_merged",
    "人工优先复核": "manual_priority_review",
}


def _clean_filename(text: str) -> str:
    return (
        text.replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace("|", "_")
        .replace(":", "_")
    )


def _severity_ascii(text: str) -> str:
    return SEVERITY_ASCII.get(text, "review")


def _bucket_type_ascii(text: str) -> str:
    return BUCKET_TYPE_ASCII.get(text, "review_bucket")


def _reason_zh(code: str) -> str:
    return REASON_BY_CODE.get(code, ReasonZh(code, code, code, "待复核", code)).zh


def _all_reasons_zh(reason_text: str) -> str:
    if not isinstance(reason_text, str) or not reason_text:
        return ""
    return "；".join(_reason_zh(code) for code in reason_text.split("|") if code)


def _iter_molecules(path: Path) -> Iterable[Chem.Mol]:
    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
    for mol in supplier:
        if mol is not None:
            yield mol


def _decorate_mol(mol: Chem.Mol, reason: ReasonZh, index: int, bucket_type_zh: str) -> Chem.Mol:
    decorated = Chem.Mol(mol)
    old_name = decorated.GetProp("_Name") if decorated.HasProp("_Name") else ""
    primary_reason_code = decorated.GetProp("primary_reason") if decorated.HasProp("primary_reason") else reason.code
    all_reason_codes = decorated.GetProp("all_reasons") if decorated.HasProp("all_reasons") else reason.code
    reason_count = len([code for code in all_reason_codes.split("|") if code])
    decorated.SetProp("_Name", f"{index:05d}_{reason.code}_{_severity_ascii(reason.severity)}")
    decorated.SetProp("original_name", old_name)
    decorated.SetProp("label_ascii", "bad")
    decorated.SetProp("current_bucket_reason_code", reason.code)
    decorated.SetProp("primary_reason_code", primary_reason_code)
    decorated.SetProp("all_reasons_code", all_reason_codes)
    decorated.SetProp("reason_count", str(reason_count))
    decorated.SetProp("is_multi_reason", "1" if reason_count > 1 else "0")
    decorated.SetProp("severity_ascii", _severity_ascii(reason.severity))
    decorated.SetProp("bucket_type_ascii", _bucket_type_ascii(bucket_type_zh))
    return decorated


def _write_reason_sdf(src: Path, dst: Path, reason: ReasonZh, bucket_type_zh: str) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    writer = Chem.SDWriter(str(dst))
    try:
        for count, mol in enumerate(_iter_molecules(src), start=1):
            writer.write(_decorate_mol(mol, reason, count, bucket_type_zh))
    finally:
        writer.close()
    return count


def _write_all_primary_sdf(primary_files: list[tuple[ReasonZh, Path]], dst: Path) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    writer = Chem.SDWriter(str(dst))
    try:
        for reason, src in primary_files:
            for mol in _iter_molecules(src):
                count += 1
                writer.write(_decorate_mol(mol, reason, count, "全部坏分子：主原因不重复合并"))
    finally:
        writer.close()
    return count


def _build_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(BASE_DIR / "new_molecule_classification.csv", low_memory=False)
    bad_df = df[df["label"] == "bad"].copy()
    bad_df["标签中文"] = "坏分子"
    bad_df["主原因中文"] = bad_df["primary_reason"].map(_reason_zh)
    bad_df["全部原因中文"] = bad_df["all_reasons"].map(_all_reasons_zh)
    bad_df["原因数量"] = bad_df["all_reasons"].map(
        lambda value: len([code for code in str(value).split("|") if code]) if pd.notna(value) else 0
    )
    bad_df["是否多原因"] = bad_df["原因数量"].map(lambda value: "是" if value > 1 else "否")
    bad_df["建议权重解释"] = bad_df["weight"].map(lambda x: f"{x:g}，越高越像强负样本")

    ordered_columns = [
        "global_id",
        "source",
        "source_mol_id",
        "smiles",
        "标签中文",
        "主原因中文",
        "全部原因中文",
        "原因数量",
        "是否多原因",
        "weight",
        "建议权重解释",
        "heavy_atom_count",
        "carbon_fraction",
        "hetero_atom_count",
        "aromatic_ring_count",
        "benzene_ring_count",
        "heterocycle_ring_count",
        "ring_count",
        "max_fused_ring_system_size",
        "three_member_ring_count",
        "spiro_atom_count",
        "bridgehead_atom_count",
        "total_halogen_count",
        "aromatic_halogen_count",
        "heavy_halogen_count",
        "br_i_count",
        "logp",
        "tpsa",
        "fragment_count",
        "primary_reason",
        "all_reasons",
        "source_path",
        "murcko_scaffold",
    ]
    bad_df = bad_df[[column for column in ordered_columns if column in bad_df.columns]]

    primary_counts = (
        bad_df.groupby(["primary_reason", "主原因中文"], dropna=False)
        .size()
        .reset_index(name="数量")
        .sort_values("数量", ascending=False)
    )

    all_reason_rows: list[dict[str, str | int]] = []
    for reasons in df.loc[df["label"] == "bad", "all_reasons"].dropna():
        for code in str(reasons).split("|"):
            if not code:
                continue
            all_reason_rows.append({"reason_code": code, "原因中文": _reason_zh(code)})
    all_counts = (
        pd.DataFrame(all_reason_rows)
        .groupby(["reason_code", "原因中文"], dropna=False)
        .size()
        .reset_index(name="数量")
        .sort_values("数量", ascending=False)
    )
    return bad_df, primary_counts, all_counts


def _write_readme(
    primary_records: list[dict[str, str | int]],
    all_reason_records: list[dict[str, str | int]],
    total_primary_count: int,
) -> None:
    lines = [
        "# 坏分子中文分类查看包",
        "",
        "这个目录用于人工查看 v0.3 scaffold 分类里被标记为坏分子的分子。",
        "",
        "注意：为了兼容 Maestro/Schrödinger，所有 SDF 文件名、SDF molecule name 和新增 SDF property 都使用英文/ASCII。",
        "中文原因只写在 README 和 CSV 对照表里。",
        "",
        "## 怎么看",
        "",
        "- `00_all_bad_primary_unique.sdf`：每个坏分子只出现一次，适合整体浏览。",
        "- `primary_unique/`：每个坏分子只进入一个主原因 SDF，适合建立最终负样本集合。",
        "- `all_triggered_reasons/`：同一个分子如果触发多个规则，会出现在多个 SDF，适合检查某条规则是否过严。",
        "- `坏分子分类总表.csv`：中文原因 + 描述符 + 原始来源，适合筛选和排序。",
        "- 每个 SDF 分子属性里都保留 `primary_reason`、`all_reasons`、`reason_count`；一个分子可以有多个坏原因。",
        "",
        "## 建议优先查看顺序",
        "",
        "1. 无芳香环的多饱和碳环：确认当前 DiffGui RL 里“多个碳环但没有苯环”的坏模式。",
        "2. 小芳香环过度卤代 / 卤素数量过多：确认五六个卤素、过小卤代芳环是否都应强负。",
        "3. 稠合杂环/复杂杂环：确认哪些是明显不合理，哪些只是复杂但可接受。",
        "4. 多稠合或桥环体系：确认桥环/螺环/大稠环是否需要一刀切还是降权。",
        "5. 高碳低杂原子：确认是否只是分布偏差，而不是单分子绝对错误。",
        "",
        f"主原因不重复合并 SDF 分子数：{total_primary_count}",
        "",
        "## 按主原因计数",
        "",
        "| 文件 | 中文原因 | 强度 | 数量 | 说明 |",
        "|---|---:|---:|---:|---|",
    ]
    for row in primary_records:
        lines.append(
            f"| `{row['file']}` | {row['reason_zh']} | {row['severity_zh']} | "
            f"{row['count']} | {row['note']} |"
        )
    lines.extend(["", "## 按所有触发原因计数", "", "| 文件 | 中文原因 | 强度 | 数量 |", "|---|---:|---:|---:|"])
    for row in all_reason_records:
        lines.append(
            f"| `{row['file']}` | {row['reason_zh']} | {row['severity_zh']} | {row['count']} |"
        )
    lines.extend(
        [
            "",
            "## 注意",
            "",
            "这些规则是为了训练分类器和 reward guard 的负样本，不等于化学上绝对禁止。",
            "例如一两个卤素可以是合理分子；这里强扣的是过度卤代、小分子过度卤代和复杂环模式。",
            "",
        ]
    )
    (OUTPUT_DIR / "README_请先看.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    RDLogger.DisableLog("rdApp.*")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    bad_df, primary_counts, all_counts = _build_tables()
    bad_df.to_csv(OUTPUT_DIR / "坏分子分类总表.csv", index=False, encoding="utf-8-sig")
    primary_counts.to_csv(OUTPUT_DIR / "坏分子_主原因计数.csv", index=False, encoding="utf-8-sig")
    all_counts.to_csv(OUTPUT_DIR / "坏分子_所有触发原因计数.csv", index=False, encoding="utf-8-sig")

    primary_dir = OUTPUT_DIR / "primary_unique"
    all_reason_dir = OUTPUT_DIR / "all_triggered_reasons"
    primary_files: list[tuple[ReasonZh, Path]] = []
    primary_records: list[dict[str, str | int]] = []
    all_reason_records: list[dict[str, str | int]] = []

    for order, reason in enumerate(REASONS, start=1):
        filename = f"{order:02d}_{_clean_filename(reason.code)}_{_severity_ascii(reason.severity)}.sdf"

        primary_src = SOURCE_SDF_DIR / f"bad__{reason.code}.sdf"
        if primary_src.exists():
            primary_dst = primary_dir / filename
            count = _write_reason_sdf(primary_src, primary_dst, reason, "主原因不重复分桶")
            primary_files.append((reason, primary_src))
            primary_records.append(
                {
                    "file": str(primary_dst.relative_to(OUTPUT_DIR)),
                    "reason_zh": reason.zh,
                    "severity_zh": reason.severity,
                    "count": count,
                    "note": reason.note,
                }
            )

        all_src = SOURCE_SDF_DIR / f"all_bad_reason__{reason.code}.sdf"
        if all_src.exists():
            all_dst = all_reason_dir / filename
            count = _write_reason_sdf(all_src, all_dst, reason, "所有触发原因分桶：允许重复")
            all_reason_records.append(
                {
                    "file": str(all_dst.relative_to(OUTPUT_DIR)),
                    "reason_zh": reason.zh,
                    "severity_zh": reason.severity,
                    "count": count,
                    "note": reason.note,
                }
            )

    total_primary_count = _write_all_primary_sdf(
        primary_files,
        OUTPUT_DIR / "00_all_bad_primary_unique.sdf",
    )

    manual_src = BASE_DIR / "manual_review_priority_negative_examples.sdf"
    if manual_src.exists():
        manual_reason = ReasonZh(
            code="manual_review_priority_negative_examples",
            zh="人工优先复核负样本",
            short="人工优先复核",
            severity="优先查看",
            note="从规则中挑出的优先人工复核坏样本。",
        )
        _write_reason_sdf(
            manual_src,
            OUTPUT_DIR / "01_manual_review_priority_negative.sdf",
            manual_reason,
            "人工优先复核",
        )

    _write_readme(primary_records, all_reason_records, total_primary_count)

    summary = pd.DataFrame(primary_records)
    summary.to_csv(OUTPUT_DIR / "中文SDF文件清单.csv", index=False, encoding="utf-8-sig")
    print(f"output_dir={OUTPUT_DIR}")
    print(f"bad_rows={len(bad_df)}")
    print(f"all_bad_primary_sdf_count={total_primary_count}")
    print("primary_reason_files=" + str(len(primary_records)))
    print("all_reason_files=" + str(len(all_reason_records)))


if __name__ == "__main__":
    main()
