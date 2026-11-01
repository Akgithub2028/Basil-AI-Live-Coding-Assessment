# DEV stage

Training subjects: [1, 2, 3, 4, 5, 6]; evaluation: [7, 8]. Trials: 270 / 90. Scaling fitted only on these training subjects.

| Model | Mean BA ± SD | Accuracy best / worst / mean | Mean train accuracy | Mean training | Collapsed seeds |
| --- | ---: | ---: | ---: | ---: | --- |
| EEGNet P1 | 60.67% ± 5.14 pp | 70.00% / 54.44% / 60.67% | 79.19% | 5.51 s | [] |
| Vanilla | 51.78% ± 3.95 pp | 57.78% / 45.56% / 51.78% | 92.44% | 21.08 s | [2] |
| Local stem | 58.22% ± 1.81 pp | 60.00% / 55.56% / 58.22% | 98.74% | 13.68 s | [] |
| Scattered control | 56.44% ± 4.30 pp | 63.33% / 51.11% / 56.44% | 99.70% | 13.71 s | [] |

## Prespecified descriptive support checks

- local_mean_ba_above_vanilla: **True**
- local_mean_ba_above_scattered: **True**
- local_seed_sd_not_above_vanilla: **True**
- local_mean_ba_above_chance: **True**
- no_local_class_collapse: **True**

Full descriptive support: **True**. This is not a statistical significance test.

| Model | Seed | Train acc | Eval acc | Eval BA | Eval loss | Per-subject acc | Predicted counts |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| EEGNet P1 | 0 | 79.26% | 70.00% | 70.00% | 0.660 | {'7': 0.7555555555555555, '8': 0.6444444444444445} | [34, 56] |
| EEGNet P1 | 1 | 77.78% | 58.89% | 58.89% | 0.666 | {'7': 0.6666666666666666, '8': 0.5111111111111111} | [52, 38] |
| EEGNet P1 | 2 | 82.22% | 61.11% | 61.11% | 0.663 | {'7': 0.7333333333333333, '8': 0.4888888888888889} | [60, 30] |
| EEGNet P1 | 3 | 77.04% | 58.89% | 58.89% | 0.673 | {'7': 0.6666666666666666, '8': 0.5111111111111111} | [40, 50] |
| EEGNet P1 | 4 | 79.63% | 54.44% | 54.44% | 0.697 | {'7': 0.6222222222222222, '8': 0.4666666666666667} | [44, 46] |
| Vanilla | 0 | 99.63% | 51.11% | 51.11% | 1.729 | {'7': 0.5333333333333333, '8': 0.4888888888888889} | [53, 37] |
| Vanilla | 1 | 100.00% | 57.78% | 57.78% | 1.799 | {'7': 0.5555555555555556, '8': 0.6} | [47, 43] |
| Vanilla | 2 | 62.59% | 53.33% | 53.33% | 0.712 | {'7': 0.5555555555555556, '8': 0.5111111111111111} | [85, 5] |
| Vanilla | 3 | 100.00% | 51.11% | 51.11% | 1.958 | {'7': 0.5777777777777777, '8': 0.4444444444444444} | [65, 25] |
| Vanilla | 4 | 100.00% | 45.56% | 45.56% | 2.022 | {'7': 0.4666666666666667, '8': 0.4444444444444444} | [32, 58] |
| Local stem | 0 | 99.26% | 55.56% | 55.56% | 2.127 | {'7': 0.5555555555555556, '8': 0.5555555555555556} | [29, 61] |
| Local stem | 1 | 98.15% | 58.89% | 58.89% | 1.744 | {'7': 0.5333333333333333, '8': 0.6444444444444445} | [30, 60] |
| Local stem | 2 | 97.41% | 56.67% | 56.67% | 1.883 | {'7': 0.6222222222222222, '8': 0.5111111111111111} | [36, 54] |
| Local stem | 3 | 99.63% | 60.00% | 60.00% | 1.512 | {'7': 0.5777777777777777, '8': 0.6222222222222222} | [53, 37] |
| Local stem | 4 | 99.26% | 60.00% | 60.00% | 1.534 | {'7': 0.6444444444444445, '8': 0.5555555555555556} | [51, 39] |
| Scattered control | 0 | 100.00% | 58.89% | 58.89% | 1.930 | {'7': 0.6888888888888889, '8': 0.4888888888888889} | [42, 48] |
| Scattered control | 1 | 99.63% | 63.33% | 63.33% | 2.052 | {'7': 0.6888888888888889, '8': 0.5777777777777777} | [36, 54] |
| Scattered control | 2 | 99.63% | 55.56% | 55.56% | 2.310 | {'7': 0.6, '8': 0.5111111111111111} | [59, 31] |
| Scattered control | 3 | 99.26% | 53.33% | 53.33% | 1.966 | {'7': 0.6444444444444445, '8': 0.4222222222222222} | [41, 49] |
| Scattered control | 4 | 100.00% | 51.11% | 51.11% | 2.023 | {'7': 0.6, '8': 0.4222222222222222} | [43, 47] |

## Measured cost

| Model | Parameters | Tokens incl. CLS | Mean training | Mean eval inference | Mean peak GPU allocation |
| --- | ---: | ---: | ---: | ---: | ---: |
| EEGNet P1 | 2,738 | n/a | 5.513 s | 0.0101 s | 202.7 MiB |
| Vanilla | 135,746 | 482 | 21.078 s | 0.0447 s | 639.2 MiB |
| Local stem | 135,746 | 30 | 13.685 s | 0.0341 s | 118.6 MiB |
| Scattered control | 135,746 | 30 | 13.712 s | 0.0323 s | 118.6 MiB |

Inference includes metric calculation and transfer to CPU for the complete evaluation set, measured once after training. Peak memory is PyTorch allocated memory, not total device usage. Per-epoch development diagnostics are excluded from training time and retained in the separate training-and-diagnostics wall-clock field.

## Paired seed differences (local minus comparator, balanced accuracy)

- Vanilla: mean +6.44 pp; by seed: 0: +4.44, 1: +1.11, 2: +3.33, 3: +8.89, 4: +14.44 pp.
- Scattered control: mean +1.78 pp; by seed: 0: -3.33, 1: -4.44, 2: +1.11, 3: +6.67, 4: +8.89 pp.

## Actual determinism checks

- eegnet_part1_config_repeat_check: {'weights_exact': True, 'predictions_exact': True, 'losses_exact': True}
- local_temporal_repeat_check: {'weights_exact': True, 'predictions_exact': True, 'losses_exact': True}
- scattered_control_repeat_check: {'weights_exact': True, 'predictions_exact': True, 'losses_exact': True}
- vanilla_transformer_repeat_check: {'weights_exact': True, 'predictions_exact': True, 'losses_exact': True}

Repeated runs are excluded from summaries. Seed SD reflects optimization variability on the same people, not a confidence interval over new subjects. The fewer attention entries are a structural fact; runtime and accuracy benefits must be measured. Filter responses show learned coefficients, not causal motor features.
