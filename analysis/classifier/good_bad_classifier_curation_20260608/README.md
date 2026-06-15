# 好坏分子分类审阅集

本目录用于调优 ChEMBL/good-bad 分子分类器前的人工审阅。

## 主要产物

- `curated_molecules.csv.gz`：全量候选分子、标签、理由和描述子。
- `category_summary.csv`：按 label / primary reason / source 的计数汇总。
- `sdf_review_by_reason/`：按每个好/坏理由抽样导出的 SDF，每类最多 200 个，分子可重复进入多个理由文件。
- `sdf_review_by_label/`：按总标签抽样导出的 SDF。
- `sdf_review_by_primary_category/`：按主理由抽样导出的 SDF，每条分子只有一个主理由。
- `curation_label_distribution.png`：标签和坏理由分布图。

## 标签含义

- `good_high_conf`：严格化学护栏通过，并且 failed RL 中 docking_raw <= -7.5。
- `good_candidate`：严格化学护栏通过，但不一定有强 docking 证据。
- `bad`：触发至少一个坏理由，例如复杂环、高碳低杂、三元环、bad bond length、top docking exploit 或 classifier false positive。
- `borderline_or_exclude`：没有明确好/坏理由，先不进入训练。

## 注意

SDF 是审阅用抽样文件，不是最终训练集；最终训练应先人工检查这些类别，再决定哪些 reason 进入 hard negative、哪些作为 soft/borderline。

## 汇总

```json
{
  "n_records": 48052,
  "n_unique_canonical_smiles": 39659,
  "label_counts": {
    "bad": 39643,
    "borderline_or_exclude": 5404,
    "good_candidate": 1781,
    "good_high_conf": 1224
  },
  "source_counts": {
    "7gpu-dock-classifier-v0-1_20260529_133609": 34280,
    "7gpu-dock-classifier-v0-1_20260528_000841": 7943,
    "260407generated_10000_filtered": 5829
  },
  "review_max_per_reason": 200,
  "review_max_per_label": 300,
  "review_max_per_primary_category": 250
}
```

## SDF manifest preview

```text
      group_type                          name  n_source  n_sdf
          reason           bad_bad_bond_length      8108    198
          reason bad_classifier_false_positive      1183    199
          reason    bad_high_carbon_low_hetero      6710    198
          reason    bad_invalid_or_unparseable     10129    200
          reason     bad_low_tpsa_or_high_logp      7185    198
          reason    bad_non_aromatic_polycycle     17409    198
          reason    bad_ring_complex_failed_rl      8482    199
          reason         bad_three_member_ring     16977    199
          reason       bad_top_docking_exploit     14758    199
          reason      good_classifier_accepted      1063    194
          reason         good_strict_chemistry      4112    194
          reason      good_strict_with_docking      1907    194
           label                           bad     39643    277
           label         borderline_or_exclude      5404    299
           label                good_candidate      1781    298
           label                good_high_conf      1224    292
primary_category           bad_bad_bond_length      3695    249
primary_category bad_classifier_false_positive      1183    248
primary_category    bad_high_carbon_low_hetero       763    244
primary_category    bad_invalid_or_unparseable     10129    250
primary_category     bad_low_tpsa_or_high_logp       538    250
primary_category    bad_non_aromatic_polycycle      1226    250
primary_category    bad_ring_complex_failed_rl      1327    249
primary_category         bad_three_member_ring      6526    249
primary_category       bad_top_docking_exploit     14256    249
primary_category      good_classifier_accepted       478    247
primary_category         good_strict_chemistry      1303    250
primary_category      good_strict_with_docking      1224    244
primary_category   no_clear_good_or_bad_reason      5404    249
```
