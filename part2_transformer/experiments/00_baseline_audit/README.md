# Experiment 00 — baseline audit on Tesla T4

Original archive: [colab_20261101T083535270691Z.zip](colab_20261101T083535270691Z.zip).
Readable report: [AUDIT_REPORT.md](results/AUDIT_REPORT.md).
Figure: [audit_dashboard.png](results/audit_dashboard.png).
Integrity/source provenance: [PROVENANCE.json](PROVENANCE.json).

Three seeds, 40 epochs, train subjects 1–8 / test 9–10. Models: EEGNet using
Part 1 architecture (kernel 80, dropout 0.5), and unchanged vanilla transformer.
Training is FP16; evaluation is FP32. Exact repeated-seed checks pass for both.

| Model | Mean train accuracy | Test mean ± seed SD | Test best / worst | Mean training time |
| --- | ---: | ---: | ---: | ---: |
| EEGNet, Part 1 settings | 83.15% | 57.04% ± 2.28 pp | 60.00% / 54.44% | 7.86 s |
| Vanilla transformer | 99.07% | 50.74% ± 4.48 pp | 54.44% / 44.44% | 28.46 s |

Vanilla shows a much larger train/test gap and confidently wrong predictions.
Cross-entropy approximated from saved probabilities averages 1.812, versus
0.719 for EEGNet (uniform 50/50 prediction: 0.693). These are descriptive results
on two test people. Changing seeds changes initialization/optimization; it does
not contradict exact deterministic reruns of a fixed seed.

Further calculations: [imported_gpu_audit_analysis.json](results/imported_gpu_audit_analysis.json).
The experiment motivated a temporal-locality hypothesis; it does not establish
that lack of locality is the only cause of the observed overfitting.
