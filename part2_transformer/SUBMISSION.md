# Part 2 hand-in map

| Required item | Location |
| --- | --- |
| Implemented `YourModel` | Exported by `main.py`; implementation in `experiment_models.py` |
| Reproducible entry point, seeds printed | `python main.py`; protocol in `experiment.py` |
| Hypothesis before implementation | `EXPERIMENT_PLAN.md`; identical hash in original Colab metadata |
| What changed, why, best/worst/mean, interpretation | `ANSWERS.md` |
| Final measured records | `results.jsonl`, `summary.json` |
| Saved figure | `figure.png` |
| Two baselines and matched control | `models.py`, `experiment_models.py`; no changes to vanilla |
| Executed notebooks | `Part2_Colab_Audit.ipynb`, `Part2_Controlled_Experiment.ipynb`; outputs retained |
| Complete experiment history and source provenance | `experiments/README.md` and numbered experiment folders |

The submitted result is mixed: local mean balanced accuracy was 54.02%, versus
50.65% for vanilla and 54.95% for EEGNet. Local ordinary accuracy best/worst/mean
was 62.22% / 47.78% / 54.00%. Its seed SD increased to 5.32 pp versus vanilla's
3.06 pp, and train accuracy reached 99.89%. The prior reduced measured runtime
and GPU memory, but did not solve overfitting or meet the stability prediction.

Development and final stages, every seed, failed criteria and controls are
preserved. Repeated use of known test subjects makes this exploratory; five
seeds are not five independent cohorts. No new strategy is part of the final
submission. Root submission copies are the measured final-stage artifacts;
reproduction writes a fresh timestamped `artifacts/study_*/` directory.

## Live walkthrough

1. Start from Part 1's six individually committed fixes and its measured
   66.67% original, 53.33% corrected and 48.89% frontal-only results. Explain
   why the before/after difference does not isolate one bug or one task.
2. Explain why fists/feet is bilateral; no left/right-only prior is imposed.
   Learned spatial mixtures retain every electrode's identity.
3. Derive the local stem: eight 32-sample filters shared across electrodes,
   stride 16, `(481-32)//16+1 = 29` windows, plus CLS = 30 tokens.
4. Explain the unchanged attention backbone: width 64, four heads, three layers;
   head dimension 16, hence QK scores divide by sqrt(16)=4 to control softmax scale.
5. Derive attention score sizes: 482²=232,324 versus 30²=900 entries per head.
   Theoretical attention reduction is 258×; measured total speedup is only 1.55×.
6. Explain the parameter match: all three attention configurations have 135,746
   parameters. The scattered control also matches initialization and token count,
   but changes contiguous grouping and positional meaning.
7. Show the figure, development/final discrepancy and all seeds. Locality alone
   was insufficient to ensure cross-subject stability. Do not interpret learned
   filter responses as cortical localization or an attention map.
8. With 80 subjects, the scarcity advantage may weaken while the compute benefit
   remains. That is a further hypothesis, not a measured result here.

Original Part 1 fix history is preserved without squashing. Part 2 commits
organize work already completed in the workspace and on Colab; they are not
backdated to imply that Git commits preceded those runs. The hypothesis file
was written before implementation and is bound to the run by its recorded hash.
