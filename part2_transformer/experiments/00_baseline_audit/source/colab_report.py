"""Generate the detailed, portable Markdown/HTML/figure audit from measured runs."""
import argparse
import html
import json
from pathlib import Path
import numpy as np


def generate_report(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    meta = json.loads((output / "metadata.json").read_text())
    results = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
    names = meta["models"]
    summary = {}
    for name in names:
        rows = [r for r in results if r["model"] == name]
        scores = np.array([r["test_accuracy"] for r in rows])
        summary[name] = dict(best=float(scores.max()),worst=float(scores.min()),mean=float(scores.mean()),
                             std=float(scores.std()),mean_train_accuracy=float(np.mean([r["train_accuracy"] for r in rows])),
                             mean_train_seconds=float(np.mean([r["train_seconds"] for r in rows])),
                             mean_test_inference_seconds=float(np.mean([r["test_inference_seconds"] for r in rows])))
    (output / "summary.json").write_text(json.dumps(summary,indent=2))
    fig, axes = plt.subplots(2,3,figsize=(16,9))
    for ix,name in enumerate(names):
        rows = [r for r in results if r["model"] == name]
        histories = [json.loads((output/f"{name}_seed{r['seed']}_history.json").read_text()) for r in rows]
        losses = np.array([[h["loss"] for h in hist] for hist in histories])
        epochs = np.arange(1,losses.shape[1]+1)
        axes[0,0].plot(epochs,losses.mean(0),label=name,marker="." if len(epochs)==1 else None)
        axes[0,0].fill_between(epochs,losses.mean(0)-losses.std(0),losses.mean(0)+losses.std(0),alpha=.15)
        axes[0,1].scatter([ix-.1]*len(rows),[r["train_accuracy"] for r in rows],marker="x",label=f"{name} train")
        axes[0,1].scatter([ix+.1]*len(rows),[r["test_accuracy"] for r in rows],label=f"{name} test")
        for s,marker in [("9","o"),("10","^")]:
            axes[0,2].scatter([ix + (-.1 if s=="9" else .1)]*len(rows),
                              [r["per_subject"][s] for r in rows],marker=marker,label=f"{name}, subject {s}")
        cm = np.sum([r["confusion_matrix"] for r in rows],axis=0)
        ax = axes[1,ix]
        ax.imshow(cm,cmap="Blues")
        for a in range(2):
            for b in range(2):
                ax.text(b,a,str(cm[a,b]),ha="center",va="center")
        ax.set(xticks=[0,1],yticks=[0,1],xticklabels=["fists","feet"],yticklabels=["fists","feet"],
               xlabel="Predicted",ylabel="True",title=f"{name}: summed over seeds")
        axes[1,2].bar(ix,summary[name]["mean_train_seconds"],label=name)
    axes[0,0].set(xlabel="Epoch",ylabel="Training loss",title="Mean ± seed SD")
    axes[0,0].legend(fontsize=7)
    for ax in [axes[0,1],axes[0,2]]:
        ax.set(xticks=range(2),xticklabels=["EEGNet P1","Vanilla"],ylim=(0,1))
        ax.axhline(.5,color="gray",linestyle="--",linewidth=1)
        ax.legend(fontsize=6)
    axes[0,1].set_title("Final train and test accuracy")
    axes[0,2].set_title("Held-out subjects separately")
    axes[1,2].set(xticks=range(2),xticklabels=["EEGNet P1","Vanilla"],ylabel="Seconds",title="Mean training time")
    fig.tight_layout()
    fig.savefig(output / "audit_dashboard.png",dpi=160)
    plt.close(fig)
    args = meta["arguments"]
    smoke = args["smoke"]
    lines = ["# " + ("SYNTHETIC SMOKE CHECK — not scientific results" if smoke else "Part 2 GPU baseline audit"),"",
             f"Hardware: {meta['gpu'] or meta['device']}. Training: {args['precision']}; final evaluation: FP32.","",
             f"Budget: {args['epochs']} epochs; seeds {args['seeds']}; effective batch {args['batch']}; "
             f"transformer microbatch {args['microbatch']}. EEGNet uses full effective batches to preserve BN statistics. "
             "Both models use AdamW, lr 0.001, weight decay 0.001.","",
             "## Experiment and data integrity","",
             "Exactly two models: EEGNet with Part 1's F1=8, D=2, F2=16, temporal kernel 80 and "
             "dropout 0.5; unchanged VanillaEEGTransformer with d=64, four heads, three layers and dropout 0.3.","",
             "Real experiment: EEGBCI subjects 1–8 for training (360 trials), subjects 9–10 for test (90 trials). "
             "Imagined both fists versus both feet; runs 6, 10, 14; 64 channels, 160 Hz, 481 samples per trial. "
             "7–30 Hz filtering and training-only channel scaling. Raw data checksums are checked before training.","",
             "The Part 1 architecture is freshly trained on Part 2 preprocessing. This is not a reproduction "
             "of the Part 1 score or a reuse of its checkpoint. This audit does not train the custom model.","",
             "The final epoch is chosen before training; test scores never select checkpoints. "
             "There is no validation-based hyperparameter search. Data preprocessing fits training subjects only.","",
             "## Measured results","",
             "| Model | Params | Test best | Test worst | Test mean ± seed SD | Mean train accuracy | Mean training | Mean test inference |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name,s in summary.items():
        lines.append(f"| {name} | {meta['architectures'][name]['parameters']:,} | {s['best']:.2%} | {s['worst']:.2%} | "
                     f"{s['mean']:.2%} ± {100*s['std']:.2f} pp | {s['mean_train_accuracy']:.2%} | "
                     f"{s['mean_train_seconds']:.2f} s | {s['mean_test_inference_seconds']:.4f} s |")
    lines += ["","Inference time covers the complete test set, including probability transfer to CPU. "
              "It is measured once after training, not a repeated latency benchmark. GPU operations are synchronized for timing.","",
              "| Model | Seed | Train | Test | Balanced test | Subject 9 | Subject 10 | Peak allocated GPU MiB | Skipped AMP steps |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in results:
        memory = "n/a" if r["peak_gpu_allocated_MiB"] is None else f"{r['peak_gpu_allocated_MiB']:.1f}"
        lines.append(f"| {r['model']} | {r['seed']} | {r['train_accuracy']:.2%} | {r['test_accuracy']:.2%} | "
                     f"{r['balanced_accuracy']:.2%} | {r['per_subject']['9']:.2%} | {r['per_subject']['10']:.2%} | "
                     f"{memory} | {r['skipped_optimizer_steps']} |")
    lines += ["","Peak allocated memory includes resident data, model, optimizer and temporary tensors; "
              "it excludes CUDA allocations outside PyTorch's allocator. AMP may skip an update when gradients overflow; "
              "all skipped updates are counted.","","## What the models compute","",
              "EEGNet: shared temporal filtering → frequency-specific full-scalp spatial filtering → "
              "separable temporal filtering → pooled features → linear classifier. Shapes excluding batch: "
              "64×481 → 8×64×482 → 16×1×120 → 16×1×15 → 2 logits. It has 2,738 trainable parameters.","",
              "Vanilla: each 64-electrode instantaneous sample becomes one 64-dimensional token. Add CLS and "
              "learned temporal positions, then three standard encoder layers and a CLS classifier. It has "
              "135,746 parameters, 482 tokens including CLS and 232,324 attention entries per head per layer. "
              "Head dimension is 16, so scaled attention divides QK dot products by 4. It attends across time; "
              "electrode identity enters through fixed slots in the projection, without explicit electrode geometry.","",
              "The transformer is about 50 times larger than this EEGNet. Differences cannot be attributed solely "
              "to attention. Strict mode selects deterministic math SDPA and does not assume T4 supports "
              "FlashAttention. AMP and microbatching are execution choices, not a new inductive bias.","",
              "Transformer microbatching preserves effective batch size and sample-weighted gradients, including short final batches. "
              "EEGNet trains with full effective batches so BN statistics are preserved. Dropout draws and floating-point "
              "reductions differ between GPU and CPU; this is not claimed bitwise equivalent to the CPU run.","",
              "## Reading the outcome","",
              "Compare the train/test gap, seed spread, predicted-class counts and subject-specific results together. "
              "A large training/test gap is compatible with overfitting; low training accuracy also permits optimization "
              "difficulty. This report does not declare a physiological mechanism from accuracy alone.","",
              "The real test majority-class reference is 46/90 = 51.11%; constant predictions have balanced accuracy 50%. "
              "One real test trial is 1.11 percentage points. Repeated seeds share the same two people: their SD is "
              "optimization variability, not a confidence interval over future subjects. Summed confusion matrices "
              "repeat the same test trials and do not increase the independent sample size.","",
              "Both-fists versus both-feet is bilateral. A left/right-only spatial prior misses this contrast. "
              "The cue/eye-position confound from Part 1 is not proven absent by filtering or cross-subject evaluation. "
              "No attention map, source localization or physiological explanation is inferred from these metrics.","",
              "## Artifacts","",
              "- `audit_dashboard.png`: loss curves, seed scores, subject scores, confusion matrices and runtime.",
              "- `metadata.json`: architecture traces, source hashes, packages, GPU, precision and protocol.",
              "- `results.jsonl`, `summary.json`: machine-readable per-seed and aggregate results.",
              "- Per-seed history JSON and prediction CSV files; final checkpoints under `checkpoints/`.",
              "- Dataset manifest and data summary, including SHA-256 checksums for all 30 recordings.","",
              "Reproducibility: strict deterministic algorithms enabled by default, seeded Python/NumPy/PyTorch, "
              "cuDNN benchmark and TF32 disabled, deterministic cuBLAS configured before torch import. "
              "Exact reproducibility applies to the same GPU and software stack, not arbitrary devices or releases.","",
              "## Actual reproducibility checks","",
              "The first seed for each model is repeated independently by default. Repeats do not enter seed averages.",""]
    for path in sorted(output.glob("*_repeat_verification.json")):
        check = json.loads(path.read_text())
        lines.append(f"- {check['model']}, seed {check['seed']}: weights exact={check['weights_exact']}, "
                     f"predictions exact={check['predictions_exact']}, losses exact={check['training_history_exact']}; "
                     f"additional training {check['extra_training_seconds']:.2f} s.")
    if not args["verify_repeat"]:
        lines.append("Repeat verification was explicitly disabled for this run.")
    text = "\n".join(lines)
    (output / "AUDIT_REPORT.md").write_text(text)
    (output / "AUDIT_REPORT.html").write_text("<!doctype html><meta charset='utf-8'><title>EEG audit</title>"
         "<style>body{max-width:1100px;margin:32px auto;font:16px system-ui}pre{white-space:pre-wrap}img{max-width:100%}</style>"
         "<img src='audit_dashboard.png' alt='Audit dashboard'><pre>"+html.escape(text)+"</pre>")
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    generate_report(parser.parse_args().directory)
