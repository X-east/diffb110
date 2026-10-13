"""Audit protocol metadata and deliver reference-ranked candidates and figures."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, rdMolTransforms
from rdkit.Chem.Draw import rdMolDraw2D

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
RECORD_DIR = ROOT / "work_records/26-10/261013"
RECORD = RECORD_DIR / "261013-04 四种外排泵抑制剂参照的优化模型候选筛选.md"
sys.path.insert(0, str(ROOT))
from ddpo.utils.docking_protocol import describe_protocol

LOGGER = logging.getLogger("epi-delivery")
COLORS = {"Donor": "#1b9e77", "Acceptor": "#d95f02", "Aromatic": "#7570b3", "PosIonizable": "#e7298a", "NegIonizable": "#66a61e", "LumpedHydrophobe": "#b58c1e"}
REF_ORDER = ["PABN", "NMP", "MBX2319", "D13-9001"]


def load_engine() -> object:
    """Import the frozen alignment implementation for numerical audit."""
    spec = importlib.util.spec_from_file_location("epi_screen", OUT / "screen.py")
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    return engine


def audit_protocols() -> dict:
    """Preserve original unspecified-frame IDs and export verified sidecars."""
    audit = []
    expected = json.loads((ROOT / "analysis/diffgui_epoch80_independent_eval_261010/docking/run_summary.json").read_text())["docking_protocol"]
    for stage in ["references", "batch2", "esrpo"]:
        folder = OUT / f"docking_{stage}"
        summary = json.loads((folder / "run_summary.json").read_text())
        old = summary["docking_protocol"]
        input_path = Path(summary["input_sdf"])
        if not input_path.is_absolute():
            input_path = ROOT / input_path
        centers = [np.mean(mol.GetConformer().GetPositions(), axis=0) for mol in Chem.SDMolSupplier(str(input_path), removeHs=False) if mol is not None]
        median = np.median(centers, axis=0)
        box_center = np.asarray([summary["box"][f"center_{axis}"] for axis in "xyz"])
        frame_verified = float(np.linalg.norm(median - box_center)) < 15
        if not frame_verified:
            raise RuntimeError(f"Input frame unverified for {stage}: median={median}")
        verified = describe_protocol(receptor_pdbqt=ROOT / "data/8gjl.pdbqt", box=summary["box"], scoring=summary["scoring"], search_mode=summary["search_mode"], num_modes=1, seed=42, ligand_prep=old["ligand_prep"], frame="receptor_frame")
        if verified != expected:
            raise RuntimeError(f"Verified protocol differs from existing epoch80: {verified}")
        evidence = {"original_protocol": old, "verified_protocol": verified, "frame_verified": frame_verified,
                    "median_input_center": median.tolist(), "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                    "correction": "Frame label omitted; verified coordinates retained" if old["ligand_frame"] == "unspecified" else "Explicit receptor-frame docking; no metadata correction"}
        (folder / "docking_protocol_verified.json").write_text(json.dumps(evidence, indent=2))
        frame = pd.read_csv(folder / "docking_results.csv")
        frame["original_docking_protocol_id"] = frame["docking_protocol_id"]
        frame["docking_protocol_id"] = verified["docking_protocol_id"]
        frame.to_csv(folder / "docking_results_verified.csv", index=False)
        audit.append({"stage": stage, "success": summary["success_count"], "total": summary["total_ligands"], "seconds": summary["seconds"], "frame_verified": frame_verified, "protocol_id": verified["docking_protocol_id"]})
    return {"protocol_id": expected["docking_protocol_id"], "stages": audit}


def get_molecule(row: pd.Series) -> Chem.Mol:
    """Recover an exact docked pose from its source record."""
    path = ROOT / row["source_pose_sdf"]
    supplier = Chem.SDMolSupplier(str(path), removeHs=False)
    return Chem.RemoveHs(supplier[int(row["source_pose_ordinal_0based"])])


def choose_examples(top: pd.DataFrame) -> pd.DataFrame:
    """Select display examples without replacing or relabelling frozen ranks."""
    examples = []
    for route in ["DiffGui_RL", "ESRPO"]:
        for ref in REF_ORDER:
            group = top[(top.route == route) & (top.reference == ref)].sort_values("rank")
            group = group[group.alerts.fillna("").eq("")]
            if ref == "NMP":
                group = group[group.matched_families.map(lambda v: json.loads(v).get("PosIonizable", 0) > 0)]
            else:
                group = group[group.matched_families.map(lambda v: json.loads(v).get("Aromatic", 0) > 0 and json.loads(v).get("Acceptor", 0) > 0)]
            if group.empty:
                LOGGER.warning("No display example in frozen Top10 for %s %s", route, ref)
                continue
            row = group.iloc[0].copy()
            row["selection_note"] = "原Top10中无Brenk/PAINS警示且匹配可电离胺" if ref == "NMP" else "原Top10中无Brenk/PAINS警示且匹配芳香与受体特征"
            if ref == "PABN":
                row["interpretation"] = "PAβN局部特征匹配；未声称保留完整胍基/阳离子药效团"
            elif ref == "D13-9001":
                row["interpretation"] = "D13-9001局部特征匹配；未声称完整22点覆盖"
            else:
                row["interpretation"] = "部分药效团匹配；不是全点命中"
            examples.append(row)
    return pd.DataFrame(examples)


def molecular_grids(top: pd.DataFrame, examples: pd.DataFrame, ref_mols: dict) -> None:
    """Draw all eight Top10 groups and an eight-example overview."""
    options = rdMolDraw2D.MolDrawOptions()
    options.legendFontSize = 30
    options.legendFraction = 0.32
    options.baseFontSize = 0.7
    for (route, ref), group in top.groupby(["route", "reference"]):
        group = group.sort_values("rank")
        molecules = [Chem.MolFromSmiles(Chem.MolToSmiles(ref_mols[ref]))]
        legends = [f"REFERENCE {ref}\n{len(json.loads((OUT/'reference_queries.json').read_text())[ref]['features'])} feature points"]
        for row in group.itertuples():
            molecules.append(Chem.MolFromSmiles(row.smiles))
            legends.append(f"#{row.rank} {row.record_id}\nF={row.feature_score:.3f} coverage={row.matched_points}/{row.reference_points}\nDock={row.docking_score:.3f}  QED={row.qed:.2f}  SA={row.sa:.2f}" + ("\nALERT: inspect CSV" if pd.notna(row.alerts) else ""))
        image = Draw.MolsToGridImage(molecules, legends=legends, molsPerRow=3, subImgSize=(520, 440), drawOptions=options)
        name = f"{route}_{ref}_screened_top10.png"
        image.save(OUT / name)
        shutil.copy2(OUT / name, RECORD_DIR / name)
    image = Draw.MolsToGridImage([Chem.MolFromSmiles(x) for x in examples.smiles], legends=[f"{r.route} / {r.reference}\n{r.record_id} (original rank {r.rank})\nF={r.feature_score:.3f} coverage={r.matched_points}/{r.reference_points}\nDock={r.docking_score:.3f} QED={r.qed:.2f} SA={r.sa:.2f}" for r in examples.itertuples()], molsPerRow=2, subImgSize=(560, 450), drawOptions=options)
    image.save(OUT / "representative_candidates.png")
    shutil.copy2(OUT / "representative_candidates.png", RECORD_DIR / "representative_candidates.png")


def match_overlays(examples: pd.DataFrame, ref_mols: dict, queries: dict, engine: object) -> None:
    """Plot 3D-match projections, keeping free alignment separate from docking."""
    fig, axes = plt.subplots(2, 4, figsize=(22, 11))
    for row, ax in zip(examples.itertuples(), axes.ravel()):
        mol = get_molecule(pd.Series(row._asdict()))
        reference = ref_mols[row.reference]
        ref_xyz = np.asarray(reference.GetConformer().GetPositions())
        center = ref_xyz.mean(axis=0)
        _, _, vt = np.linalg.svd(ref_xyz - center, full_matrices=False)
        basis = vt[:2].T
        matrix = np.asarray(json.loads(row.transform))
        probe_xyz = np.asarray(mol.GetConformer().GetPositions()) @ matrix[:3, :3].T + matrix[:3, 3]
        for obj, coords, color in [(reference, ref_xyz, "#9ca5b0"), (mol, probe_xyz, "#333333")]:
            xy = (coords - center) @ basis
            for bond in obj.GetBonds():
                a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                ax.plot(xy[[a,b],0], xy[[a,b],1], color=color, alpha=.38, linewidth=1)
        ref_features = queries[row.reference]["features"]
        probe_features = engine.features(mol)
        for kind, feats in [("reference", ref_features), ("candidate", probe_features)]:
            coords = np.asarray([x["xyz"] for x in feats])
            if kind == "candidate":
                coords = coords @ matrix[:3, :3].T + matrix[:3, 3]
            xy = (coords - center) @ basis
            for family in COLORS:
                inds = [i for i,x in enumerate(feats) if x["family"] == family]
                if inds:
                    ax.scatter(xy[inds,0], xy[inds,1], s=95 if kind == "reference" else 40, facecolors="none" if kind == "reference" else COLORS[family], edgecolors=COLORS[family], linewidths=1.8, alpha=.85)
        ref_pos = (np.asarray([x["xyz"] for x in ref_features]) - center) @ basis
        probe_pos = (np.asarray([x["xyz"] for x in probe_features]) @ matrix[:3,:3].T + matrix[:3,3] - center) @ basis
        for i,j,_ in json.loads(row.matched_pairs):
            ax.plot([ref_pos[i,0],probe_pos[j,0]],[ref_pos[i,1],probe_pos[j,1]],"--",color="#aaaaaa",linewidth=1)
        ax.set_title(f"{row.route} / {row.reference}\n{row.record_id}: {row.matched_points}/{row.reference_points}, RMSD {row.feature_rmsd:.2f} Å", fontsize=15)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=11)
        ax.grid(alpha=.15)
    handles = [plt.Line2D([0],[0],marker="o",linestyle="",color=color,label=name) for name,color in COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=13)
    fig.suptitle("Free feature alignment (PCA projection): open points = reference; filled points = candidate", fontsize=19)
    fig.tight_layout(rect=(0,.06,1,.94))
    for suffix in ["png", "svg"]:
        fig.savefig(OUT / f"representative_feature_overlays.{suffix}", dpi=180)
        shutil.copy2(OUT / f"representative_feature_overlays.{suffix}", RECORD_DIR / f"representative_feature_overlays.{suffix}")
    plt.close(fig)


def score_plot(frame: pd.DataFrame) -> None:
    """Show distributions among eligible structures without claiming activity."""
    fig, axes = plt.subplots(1,4,figsize=(21,5))
    for ref, ax in zip(REF_ORDER, axes):
        data = [frame[(frame.reference==ref)&(frame.route==route)&frame.eligible].drop_duplicates("smiles").feature_score.values for route in ["DiffGui_RL","ESRPO"]]
        boxes = ax.boxplot(data, patch_artist=True, widths=.5, showfliers=False)
        for patch,color in zip(boxes["boxes"],["#4277aa","#dc9642"]):patch.set_facecolor(color);patch.set_alpha(.6)
        ax.set_xticks([1,2],["DiffGui RL","ESRPO"],fontsize=13)
        ax.set_title(ref,fontsize=19)
        ax.tick_params(axis="y",labelsize=13)
        ax.set_ylim(0,1)
        ax.grid(axis="y",alpha=.2)
    axes[0].set_ylabel("Normalized 3D feature overlap",fontsize=16)
    fig.tight_layout()
    for suffix in ["png","svg"]:
        fig.savefig(OUT/f"screened_feature_score_distributions.{suffix}",dpi=180)
        shutil.copy2(OUT/f"screened_feature_score_distributions.{suffix}",RECORD_DIR/f"screened_feature_score_distributions.{suffix}")
    plt.close(fig)


def validate_artifacts(top: pd.DataFrame, frame: pd.DataFrame, examples: pd.DataFrame, engine: object, queries: dict) -> dict:
    """Check provenance, score bounds, matching radii and seed stability."""
    checks = {"all_match_count": len(frame), "score_bounds": bool(frame.feature_score.between(0,1).all()),
              "coverage_bounds": bool(frame.reference_coverage.between(0,1).all()),
              "one_to_one_counts": bool((frame.matched_points <= frame[["reference_points","candidate_points"]].min(axis=1)).all()),
              "top_all_eligible": bool(top.eligible.all()), "groups": []}
    for (route,ref), group in top.groupby(["route","reference"]):
        base=OUT/f"{route}_{ref}_screened_top10"
        s=list(Chem.SDMolSupplier(str(base)+"_docked.sdf"))
        checks["groups"].append({"route":route,"reference":ref,"count":len(group),"unique":int(group.smiles.nunique()),"sdf_count":len(s),"source_smiles_match":all(Chem.MolToSmiles(Chem.RemoveHs(mol))==row.smiles for mol,row in zip(s,group.itertuples()))})
    stability=[]
    old_limit=engine.CONFIG["alignment_seed_limit"]
    try:
        engine.CONFIG["alignment_seed_limit"]=256
        for row in examples.itertuples():
            mol=get_molecule(pd.Series(row._asdict()));ref=queries[row.reference]
            result,_=engine.align_features(ref["features"],engine.features(mol),ref["anchors"])
            stability.append({"route":row.route,"reference":row.reference,"record_id":row.record_id,"score_64_seeds":row.feature_score,"score_256_seeds":result["similarity"],"delta":result["similarity"]-row.feature_score})
    finally:
        engine.CONFIG["alignment_seed_limit"]=old_limit
    checks["representative_seed_stability"]=stability
    pd.DataFrame(stability).to_csv(OUT/"representative_seed_stability.csv",index=False)
    sensitivity=[]
    for (route,ref),group in frame[frame.eligible].groupby(["route","reference"]):
        nominal=group.sort_values("feature_score",ascending=False).drop_duplicates("smiles").head(10)
        for radius in [1,2]:
            altered=group.sort_values(f"score_radius_{radius}",ascending=False).drop_duplicates("smiles").head(10)
            sensitivity.append({"route":route,"reference":ref,"radius":radius,"top10_overlap":len(set(nominal.smiles)&set(altered.smiles)),"evaluation":"same fixed alignment; no radius-specific realignment"})
    pd.DataFrame(sensitivity).to_csv(OUT/"radius_sensitivity.csv",index=False)
    checks["radius_sensitivity"]=sensitivity
    if not all(checks[x] for x in ["score_bounds","coverage_bounds","one_to_one_counts","top_all_eligible"]):raise RuntimeError("Artifact checks failed")
    if not all(g["count"]==g["unique"]==g["sdf_count"]==10 and g["source_smiles_match"] for g in checks["groups"]):raise RuntimeError("Top10 exports failed")
    return checks


def make_report(top: pd.DataFrame, examples: pd.DataFrame, frame: pd.DataFrame, summary: dict, audit: dict) -> None:
    """Append actual counts, ranked tables, display examples and evidence links."""
    lines=["## 完成结果", "", "共评价 3,603 个已有对接姿态，得到 14,412 次参考比较。每个参考分别输出两条路线的 Top10，按异构 SMILES 去重；跨参考允许同一分子再次出现。", "", "|路线|固定生成尝试数|本次可评价分子|R1–R5通过|性质通过|同时满足结构、性质及对接≤−10|唯一候选|", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary["counts"]:
        lines.append(f"|{row['route']}|{'3,000' if row['route']=='DiffGui_RL' else '1,000'}|{row['docked']:,}|{row['r1r5_clean']}|{row['physchem_pass']}|{row['eligible']}|{row['eligible_unique']}|")
    lines += ["", "性质通过与R1–R5通过是各自独立计数，不能相加。本次为选例，生成预算不同、ESRPO还有预处理差异，不用于宣称两条路线孰优。", "", "![候选池特征分数分布](screened_feature_score_distributions.png)", "", "### 八组原始第一名", "", "原始排名不因警示或展示需要改名。评分是部分匹配的归一化几何分数；每个参考的点数不同，跨参考不能按同一分数直接拼总榜。", "", "|路线|参考|排名第一|特征分数|匹配点/参考点|匹配点RMSD Å|对接 kcal/mol|复核信息|", "|---|---|---|---:|---:|---:|---:|---|"]
    for row in top[top['rank']==1].itertuples():
        note="无两类目录警示" if pd.isna(row.alerts) else str(row.alerts).replace(";","；")
        if row.reference=="NMP" and json.loads(row.matched_families).get("PosIonizable",0)==0:note+="；未匹配可电离胺"
        lines.append(f"|{row.route}|{row.reference}|{row.record_id}|{row.feature_score:.3f}|{row.matched_points}/{row.reference_points}|{row.feature_rmsd:.3f}|{row.docking_score:.3f}|{note}|")
    lines += ["", "当前姿态、特征定义、1.5 Å半径及有限搜索下，未找到完整覆盖任何一个参考全部特征点的分子。因此应写**局部药效团特征匹配候选**，不能写“通过四种完整药效团模型筛选”。这也不是对所有可能构象或Pharmit查询的不存在性证明。PAβN的胍基/可电离特征、D13-9001的带电尾部不可凭局部重叠宣称已保留。", "", "### 可优先展示的八个例子", "", "在原Top10内按以下复核选择：无Brenk/PAINS目录警示；NMP例子须确实匹配可电离胺，其他参考须匹配芳香与氢键受体特征。该步骤是看完自动排名后的展示复核，未改写原始排序或初筛计数，也不等于化学稳定性已经证实。", "", "|路线|参考|展示候选|原Top10名次|分数|覆盖|对接|QED|SA|匹配到的类别|", "|---|---|---|---:|---:|---:|---:|---:|---:|---|"]
    labels={"Acceptor":"受体","Donor":"供体","Aromatic":"芳香","PosIonizable":"可正电离","NegIonizable":"可负电离","LumpedHydrophobe":"聚集疏水"}
    for row in examples.itertuples():
        families="、".join(f"{labels[k]}{v}" for k,v in json.loads(row.matched_families).items())
        lines.append(f"|{row.route}|{row.reference}|{row.record_id}|{row.rank}|{row.feature_score:.3f}|{row.matched_points}/{row.reference_points}|{row.docking_score:.3f}|{row.qed:.3f}|{row.sa:.2f}|{families}|")
    lines += ["", "![八个展示候选的二维结构](representative_candidates.png)", "", "![自由对齐后的药效团特征投影](representative_feature_overlays.png)", "", "投影图仅用于解释几何匹配，空心为参考、实心为候选，虚线连接已匹配点；灰色/黑色为参考/候选骨架。图是三维结果的PCA二维投影，不能从图上直接读取三维RMSD。", "", "### 坐标和口袋检查", "", f"Top10共80条记录、{summary['screened_unique_structures']}个跨参考唯一异构结构。80个原对接姿态均未出现重原子距受体重原子<1.5 Å的严重重叠；自由对齐后的姿态则有{int((top.free_aligned_severe_overlap_atoms>0).sum())}条出现这一重叠。这是简单几何检查，不是PoseBusters或完整构象验证。", "", "因此两类SDF分别输出：`*_docked.sdf`为独立对接姿态；`*_aligned.sdf`为用于药效团比较的自由刚体对齐姿态。后者不能直接作为已验证RE-CmeB结合姿态，也不能沿用原对接分数解释其结合强度。", "", "### 对接协议核验", "", f"核验后统一协议ID：`{audit['protocols']['protocol_id']}`。首次新增对接启动漏填了坐标系标签，原summary/CSV记录为unspecified；经输入坐标、受体、盒、氢处理、版本、seed等核验，实际是receptor_frame。没有重写原始summary或分数；另存`docking_protocol_verified.json`及`docking_results_verified.csv`保留核验依据，启动脚本已补显式标签。", "", "### 完整Top10与结构图", ""]
    for route in ["DiffGui_RL","ESRPO"]:
        for ref in REF_ORDER:
            stem=f"{route}_{ref}_screened_top10"
            lines += [f"#### {route} / {ref}", "", "|名次|分子ID|分数|覆盖|点RMSD Å|对接|QED|SA|化学警示|", "|---:|---|---:|---:|---:|---:|---:|---:|---|"]
            for row in top[(top.route==route)&(top.reference==ref)].sort_values("rank").itertuples():
                lines.append(f"|{row.rank}|{row.record_id}|{row.feature_score:.3f}|{row.matched_points}/{row.reference_points}|{row.feature_rmsd:.3f}|{row.docking_score:.3f}|{row.qed:.3f}|{row.sa:.2f}|{'无' if pd.isna(row.alerts) else str(row.alerts).replace(';','；')}|")
            lines += ["", f"![{route} {ref} Top10结构]({stem}.png)", "", f"[完整CSV](../../../analysis/epi_pharmacophore_screen_261013/{stem}.csv) · [对接姿态SDF](../../../analysis/epi_pharmacophore_screen_261013/{stem}_docked.sdf) · [自由对齐SDF](../../../analysis/epi_pharmacophore_screen_261013/{stem}_aligned.sdf)", ""]
    max_delta=max(x['delta'] for x in audit['artifacts']['representative_seed_stability'])
    min_overlap=min(x['top10_overlap'] for x in audit['artifacts']['radius_sensitivity'])
    lines += ["## 验证与证据文件", "", f"参考自匹配及已知刚体变换控制均通过；14,412条分数和覆盖率均在0–1内；匹配计数未超过任一方点数；八组筛选榜单均为10个唯一异构结构，CSV与SDF图结构对应。八个展示例子的64→256起点复核最大分数增加为{max_delta:.6f}；1 Å/2 Å半径下，同一冻结对齐的Top10重叠最少为{min_overlap}/10。半径检查不含重新对齐，不是完整参数鲁棒性证明。", "", "执行`py_compile`、分析、交付绘图脚本；18个CPU worker匹配任务完成，原模型、原始分子及既有筛选表未修改。未运行活性实验、MD或PoseBusters。", "", "- [所有候选与性质](../../../analysis/epi_pharmacophore_screen_261013/molecule_metrics.csv)", "- [14,412次三维特征匹配明细](../../../analysis/epi_pharmacophore_screen_261013/all_reference_matches.csv)", "- [仅相似度Top10](../../../analysis/epi_pharmacophore_screen_261013/top10_similarity_only.csv)", "- [筛选后Top10](../../../analysis/epi_pharmacophore_screen_261013/top10_screened.csv)", "- [展示例子CSV](../../../analysis/epi_pharmacophore_screen_261013/representative_candidates.csv)", "- [展示例子对接姿态SDF](../../../analysis/epi_pharmacophore_screen_261013/representative_candidates_docked.sdf)", "- [参考特征点、坐标、锚点](../../../analysis/epi_pharmacophore_screen_261013/reference_queries.json)", "- [固定配置与公式](../../../analysis/epi_pharmacophore_screen_261013/screen_config.json)", "- [运行日志](../../../analysis/epi_pharmacophore_screen_261013/screen.log)", "- [验证汇总](../../../analysis/epi_pharmacophore_screen_261013/verification.json)", "", "## 可以用于论文的表述", "", "“以已知RND外排泵抑制剂PAβN、NMP、MBX2319和D13-9001为结构参考，对优化模型独立生成并通过统一初筛的分子进行三维药效团特征比较。根据局部特征匹配程度，并结合化学结构复核，选择代表性候选开展计算分析。”", "", "若展示NMP样例，可进一步写“候选保留了芳香区域及可电离胺特征的空间关联”；PAβN/D13-9001目前宜写“与参考部分芳香及极性特征较匹配”。当前证据支持候选选择理由，尚不能说明完整抑制机制、靶点活性或优化前后最优单分子的优劣。"]
    text=RECORD.read_text()
    if '<!-- RESULTS_PENDING -->' not in text:
        text=text.split('## 完成结果')[0].rstrip()+"\n\n<!-- RESULTS_PENDING -->\n"
    RECORD.write_text(text.replace('<!-- RESULTS_PENDING -->','\n'.join(lines)))


def main() -> None:
    """Create reviewed delivery artifacts without altering the frozen ranks."""
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(name)s %(message)s")
    RDLogger.DisableLog("rdApp.*")
    top=pd.read_csv(OUT/"top10_screened.csv")
    frame=pd.read_csv(OUT/"all_reference_matches.csv")
    summary=json.loads((OUT/"summary.json").read_text())
    queries=json.loads((OUT/"reference_queries.json").read_text())
    references={mol.GetProp("reference_id"):Chem.RemoveHs(mol) for mol in Chem.SDMolSupplier(str(OUT/"docking_references/successful_docked_outputs.sdf"),removeHs=False)}
    engine=load_engine()
    examples=choose_examples(top)
    examples.to_csv(OUT/"representative_candidates.csv",index=False)
    with Chem.SDWriter(str(OUT/"representative_candidates_docked.sdf")) as writer:
        for _,row in examples.iterrows():
            mol=get_molecule(row)
            for key in ["record_id","route","reference","rank","feature_score","reference_coverage","docking_score","selection_note","interpretation"]:mol.SetProp(key,str(row[key]))
            writer.write(mol)
    audit={"protocols":audit_protocols(),"artifacts":validate_artifacts(top,frame,examples,engine,queries)}
    (OUT/"verification.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    molecular_grids(top,examples,references)
    score_plot(frame)
    match_overlays(examples,references,queries,engine)
    make_report(top,examples,frame,summary,audit)
    LOGGER.info("COMPLETE: %d display examples, 8 Top10 grids, feature overlays and work record",len(examples))


if __name__=="__main__":
    main()
