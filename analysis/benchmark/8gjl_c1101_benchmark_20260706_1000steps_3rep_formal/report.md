# 8GJL/C1101 1000-step 3-repeat formal summary

Protocol: 1000 diffusion steps, 1000 attempts per model per repeat. DiffGui rep1 is replaced by the explicit 1000-step rerun from `8gjl_c1101_benchmark_20260701_diffgui_rep1_1000steps_seed1_7`.

![sanitize-valid mean](sanitize_valid_rate_3rep.png)

![sanitize-valid by repeat](sanitize_valid_rate_by_repeat.png)

## Mean over repeats

| model | sdf_written_mean | sanitize_valid_mean | unique_smiles_mean | sanitize_valid_per_attempt_mean | sanitize_valid_per_attempt_std |
| --- | --- | --- | --- | --- | --- |
| DiffGui | 880.333 | 880.333 | 870.000 | 0.880333 | 0.012503 |
| DiffSBDD | 1000.000 | 666.333 | 666.333 | 0.666333 | 0.006506 |
| PMDM | 575.333 | 575.333 | 575.333 | 0.575333 | 0.009074 |
| GCDM-SBDD | 1000.000 | 523.333 | 519.000 | 0.523333 | 0.012055 |
| TargetDiff | 991.333 | 991.333 | 990.667 | 0.991333 | 0.000577 |

## Per-repeat counts

| repeat | model | sdf_written | sanitize_valid | unique_smiles | sanitize_valid_per_attempt |
| --- | --- | --- | --- | --- | --- |
| rep1 | DiffGui | 886 | 886 | 871 | 0.886000 |
| rep1 | DiffSBDD | 1000 | 673 | 673 | 0.673000 |
| rep1 | PMDM | 565 | 565 | 565 | 0.565000 |
| rep1 | GCDM-SBDD | 1000 | 512 | 511 | 0.512000 |
| rep1 | TargetDiff | 991 | 991 | 990 | 0.991000 |
| rep2 | DiffGui | 889 | 889 | 883 | 0.889000 |
| rep2 | DiffSBDD | 1000 | 666 | 666 | 0.666000 |
| rep2 | PMDM | 579 | 579 | 579 | 0.579000 |
| rep2 | GCDM-SBDD | 1000 | 536 | 530 | 0.536000 |
| rep2 | TargetDiff | 992 | 992 | 991 | 0.992000 |
| rep3 | DiffGui | 866 | 866 | 856 | 0.866000 |
| rep3 | DiffSBDD | 1000 | 660 | 660 | 0.660000 |
| rep3 | PMDM | 582 | 582 | 582 | 0.582000 |
| rep3 | GCDM-SBDD | 1000 | 522 | 516 | 0.522000 |
| rep3 | TargetDiff | 991 | 991 | 991 | 0.991000 |

## Files

- `formal_3rep_per_repeat_counts.csv`
- `formal_3rep_model_summary.csv`
- `formal_3rep_sources.csv`
