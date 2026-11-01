# FINAL stage

Training subjects: [1, 2, 3, 4, 5, 6, 7, 8]; evaluation: [9, 10]. Trials: 360 / 90. Scaling fitted only on these training subjects.

| Model | Mean BA ± SD | Accuracy best / worst / mean | Mean train accuracy | Mean training | Collapsed seeds |
| --- | ---: | ---: | ---: | ---: | --- |
| EEGNet P1 | 54.95% ± 3.15 pp | 58.89% / 50.00% / 54.67% | 82.39% | 7.77 s | [] |
| Vanilla | 50.65% ± 3.06 pp | 55.56% / 45.56% / 50.44% | 99.56% | 28.31 s | [] |
| Local stem | 54.02% ± 5.32 pp | 62.22% / 47.78% / 54.00% | 99.89% | 18.32 s | [] |
| Scattered control | 51.32% ± 3.01 pp | 55.56% / 47.78% / 51.33% | 99.28% | 18.55 s | [] |

## Prespecified descriptive support checks

- local_mean_ba_above_vanilla: **True**
- local_mean_ba_above_scattered: **True**
- local_seed_sd_not_above_vanilla: **False**
- local_mean_ba_above_chance: **True**
- no_local_class_collapse: **True**

Full descriptive support: **False**. This is not a statistical significance test.

| Model | Seed | Train acc | Eval acc | Eval BA | Eval loss | Per-subject acc | Predicted counts |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| EEGNet P1 | 0 | 85.56% | 55.56% | 55.98% | 0.740 | {'9': 0.4444444444444444, '10': 0.6666666666666666} | [28, 62] |
| EEGNet P1 | 1 | 80.00% | 54.44% | 54.55% | 0.742 | {'9': 0.6, '10': 0.4888888888888889} | [41, 49] |
| EEGNet P1 | 2 | 83.33% | 54.44% | 54.79% | 0.720 | {'9': 0.5333333333333333, '10': 0.5555555555555556} | [31, 59] |
| EEGNet P1 | 3 | 82.22% | 58.89% | 59.63% | 0.748 | {'9': 0.6, '10': 0.5777777777777777} | [15, 75] |
| EEGNet P1 | 4 | 80.83% | 50.00% | 49.80% | 0.939 | {'9': 0.5111111111111111, '10': 0.4888888888888889} | [53, 37] |
| Vanilla | 0 | 99.72% | 55.56% | 55.78% | 1.339 | {'9': 0.5333333333333333, '10': 0.5777777777777777} | [36, 54] |
| Vanilla | 1 | 100.00% | 45.56% | 46.20% | 3.144 | {'9': 0.4444444444444444, '10': 0.4666666666666667} | [19, 71] |
| Vanilla | 2 | 99.17% | 50.00% | 50.05% | 2.092 | {'9': 0.5333333333333333, '10': 0.4666666666666667} | [43, 47] |
| Vanilla | 3 | 98.89% | 51.11% | 51.09% | 1.725 | {'9': 0.4666666666666667, '10': 0.5555555555555556} | [46, 44] |
| Vanilla | 4 | 100.00% | 50.00% | 50.15% | 2.546 | {'9': 0.4666666666666667, '10': 0.5333333333333333} | [39, 51] |
| Local stem | 0 | 100.00% | 62.22% | 62.20% | 1.555 | {'9': 0.6222222222222222, '10': 0.6222222222222222} | [46, 44] |
| Local stem | 1 | 99.72% | 48.89% | 48.72% | 2.220 | {'9': 0.4444444444444444, '10': 0.5333333333333333} | [52, 38] |
| Local stem | 2 | 100.00% | 47.78% | 47.88% | 2.373 | {'9': 0.5111111111111111, '10': 0.4444444444444444} | [41, 49] |
| Local stem | 3 | 100.00% | 56.67% | 56.97% | 2.144 | {'9': 0.5333333333333333, '10': 0.6} | [33, 57] |
| Local stem | 4 | 99.72% | 54.44% | 54.35% | 2.147 | {'9': 0.5555555555555556, '10': 0.5333333333333333} | [49, 41] |
| Scattered control | 0 | 98.89% | 53.33% | 53.71% | 2.627 | {'9': 0.6, '10': 0.4666666666666667} | [30, 60] |
| Scattered control | 1 | 98.06% | 50.00% | 49.60% | 2.550 | {'9': 0.5777777777777777, '10': 0.4222222222222222} | [61, 29] |
| Scattered control | 2 | 99.72% | 50.00% | 50.15% | 2.353 | {'9': 0.5555555555555556, '10': 0.4444444444444444} | [39, 51] |
| Scattered control | 3 | 99.72% | 55.56% | 55.78% | 2.460 | {'9': 0.6888888888888889, '10': 0.4222222222222222} | [36, 54] |
| Scattered control | 4 | 100.00% | 47.78% | 47.38% | 2.737 | {'9': 0.4888888888888889, '10': 0.4666666666666667} | [61, 29] |

## Measured cost

| Model | Parameters | Tokens incl. CLS | Mean training | Mean eval inference | Mean peak GPU allocation |
| --- | ---: | ---: | ---: | ---: | ---: |
| EEGNet P1 | 2,738 | n/a | 7.772 s | 0.0131 s | 213.0 MiB |
| Vanilla | 135,746 | 482 | 28.315 s | 0.0454 s | 649.5 MiB |
| Local stem | 135,746 | 30 | 18.320 s | 0.0254 s | 128.9 MiB |
| Scattered control | 135,746 | 30 | 18.549 s | 0.0359 s | 128.9 MiB |

Inference includes metric calculation and transfer to CPU for the complete evaluation set, measured once after training. Peak memory is PyTorch allocated memory, not total device usage. Per-epoch development diagnostics are excluded from training time and retained in the separate training-and-diagnostics wall-clock field.

## Paired seed differences (local minus comparator, balanced accuracy)

- Vanilla: mean +3.37 pp; by seed: 0: +6.42, 1: +2.52, 2: -2.17, 3: +5.88, 4: +4.20 pp.
- Scattered control: mean +2.70 pp; by seed: 0: +8.50, 1: -0.89, 2: -2.27, 3: +1.19, 4: +6.97 pp.

## Actual determinism checks

- eegnet_part1_config_repeat_check: {'weights_exact': True, 'predictions_exact': True, 'losses_exact': True}
- local_temporal_repeat_check: {'weights_exact': True, 'predictions_exact': True, 'losses_exact': True}
- scattered_control_repeat_check: {'weights_exact': True, 'predictions_exact': True, 'losses_exact': True}
- vanilla_transformer_repeat_check: {'weights_exact': True, 'predictions_exact': True, 'losses_exact': True}

Repeated runs are excluded from summaries. Seed SD reflects optimization variability on the same people, not a confidence interval over new subjects. The fewer attention entries are a structural fact; runtime and accuracy benefits must be measured. Filter responses show learned coefficients, not causal motor features.
