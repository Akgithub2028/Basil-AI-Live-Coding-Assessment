"""Report the prespecified hypothesis, including unsupported/negative outcomes."""
import json
from pathlib import Path
import numpy as np

LABELS={"eegnet_part1_config":"EEGNet P1","vanilla_transformer":"Vanilla",
        "local_temporal":"Local stem","scattered_control":"Scattered control"}


def stage_report(folder,smoke=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows=[json.loads(line) for line in (folder/"results.jsonl").read_text().splitlines()]
    names=list(LABELS)
    summary={}
    for name in names:
        selected=[r for r in rows if r["model"]==name]
        ba=np.array([r["evaluation"]["balanced_accuracy"] for r in selected])
        acc=np.array([r["evaluation"]["accuracy"] for r in selected])
        summary[name]=dict(mean_ba=float(ba.mean()),sd_ba=float(ba.std()),
            mean_accuracy=float(acc.mean()),best_accuracy=float(acc.max()),worst_accuracy=float(acc.min()),
            mean_train_accuracy=float(np.mean([r["train"]["accuracy"] for r in selected])),
            mean_training_seconds=float(np.mean([r["training_seconds"] for r in selected])),
            collapsed_seeds=[r["seed"] for r in selected if r["evaluation"]["collapsed"]])
    local,vanilla,control=(summary[k] for k in ["local_temporal","vanilla_transformer","scattered_control"])
    criteria=dict(local_mean_ba_above_vanilla=local["mean_ba"]>vanilla["mean_ba"],
                  local_mean_ba_above_scattered=local["mean_ba"]>control["mean_ba"],
                  local_seed_sd_not_above_vanilla=local["sd_ba"]<=vanilla["sd_ba"],
                  local_mean_ba_above_chance=local["mean_ba"]>.5,
                  no_local_class_collapse=not local["collapsed_seeds"])
    pairing={}
    for comparator in ["vanilla_transformer","scattered_control"]:
        differences=[]
        for r in rows:
            if r["model"]!="local_temporal":continue
            other=next(q for q in rows if q["model"]==comparator and q["seed"]==r["seed"])
            differences.append(dict(seed=r["seed"],delta_balanced_accuracy=r["evaluation"]["balanced_accuracy"]-other["evaluation"]["balanced_accuracy"]))
        pairing[comparator]=differences
    result=dict(models=summary,criteria=criteria,full_descriptive_support=all(criteria.values()),paired_differences=pairing)
    (folder/"summary.json").write_text(json.dumps(result,indent=2))
    fig,axes=plt.subplots(2,3,figsize=(16,9))
    for i,name in enumerate(names):
        selected=[r for r in rows if r["model"]==name]
        histories=[json.loads((folder/f"{name}_seed{r['seed']}_history.json").read_text()) for r in selected]
        values=np.array([[h["train_loss"] for h in hist] for hist in histories]);x=np.arange(1,values.shape[1]+1)
        axes[0,0].plot(x,values.mean(0),label=LABELS[name],marker="." if len(x)==1 else None)
        axes[0,0].fill_between(x,values.mean(0)-values.std(0),values.mean(0)+values.std(0),alpha=.12)
        if "eval_loss" in histories[0][0]:
            ev=np.array([[h["eval_loss"] for h in hist] for hist in histories])
            axes[0,1].plot(x,ev.mean(0),label=LABELS[name],marker="." if len(x)==1 else None)
        axes[0,2].scatter([i]*len(selected),[r["evaluation"]["balanced_accuracy"] for r in selected])
        axes[1,0].scatter([i-.1]*len(selected),[r["train"]["accuracy"] for r in selected],marker="x",label=LABELS[name]+" train")
        axes[1,0].scatter([i+.1]*len(selected),[r["evaluation"]["accuracy"] for r in selected],label=LABELS[name]+" eval")
        axes[1,1].bar(i,summary[name]["mean_training_seconds"])
    if folder.name=="final":
        axes[0,1].axis("off")
        axes[0,1].text(.05,.65,f"Final subjects evaluated only after epoch {rows[0]['epochs']}.\nNo test-based epoch selection.\nSee development stage for evaluation-loss curves.",fontsize=11)
    else:axes[0,1].set(xlabel="Epoch",ylabel="Development cross-entropy",title="Development diagnostic only")
    axes[0,0].set(xlabel="Epoch",ylabel="Training loss",title="Mean ± seed SD")
    axes[0,0].legend(fontsize=7)
    for ax in [axes[0,2],axes[1,0],axes[1,1]]:
        ax.set(xticks=range(4),xticklabels=list(LABELS.values()));ax.tick_params(axis="x",labelrotation=15)
    for ax in [axes[0,2],axes[1,0]]:ax.axhline(.5,color="gray",linestyle="--");ax.set_ylim(0,1)
    axes[0,2].set_title("Primary: held-out balanced accuracy")
    axes[1,0].set_title("Training vs held-out accuracy")
    axes[1,1].set(ylabel="Seconds",title="Mean training time, diagnostics excluded")
    filters=np.load(folder/"temporal_filters.npy")
    frequencies=np.fft.rfftfreq(512,1/160)
    spectra=np.abs(np.fft.rfft(filters,n=512,axis=-1))
    spectra/=np.maximum(spectra.max(axis=-1,keepdims=True),1e-12)
    axes[1,2].plot(frequencies,spectra.T,alpha=.7)
    axes[1,2].axvspan(7,30,color="gray",alpha=.15)
    axes[1,2].set(xlabel="Hz",ylabel="Normalized magnitude",title="Local stem: first-seed filter responses",xlim=(0,80))
    if smoke:fig.suptitle("SYNTHETIC PLUMBING CHECK — NOT EEG RESULTS",color="red")
    fig.tight_layout();fig.savefig(folder/"experiment_dashboard.png",dpi=160);plt.close(fig)
    split=json.loads((folder/"split.json").read_text())
    lines=[f"# {folder.name.upper()} stage"+(" — synthetic check" if smoke else ""),"",
        f"Training subjects: {split['train_subjects']}; evaluation: {split['evaluation_subjects']}. "
        f"Trials: {split['n_train']} / {split['n_evaluation']}. Scaling fitted only on these training subjects.","",
        "| Model | Mean BA ± SD | Accuracy best / worst / mean | Mean train accuracy | Mean training | Collapsed seeds |",
        "| --- | ---: | ---: | ---: | ---: | --- |"]
    for name,s in summary.items():
        lines.append(f"| {LABELS[name]} | {s['mean_ba']:.2%} ± {100*s['sd_ba']:.2f} pp | "
            f"{s['best_accuracy']:.2%} / {s['worst_accuracy']:.2%} / {s['mean_accuracy']:.2%} | "
            f"{s['mean_train_accuracy']:.2%} | {s['mean_training_seconds']:.2f} s | {s['collapsed_seeds']} |")
    lines += ["","## Prespecified descriptive support checks","",
              *[f"- {key}: **{value}**" for key,value in criteria.items()],"",
              f"Full descriptive support: **{all(criteria.values())}**. This is not a statistical significance test.","",
              "| Model | Seed | Train acc | Eval acc | Eval BA | Eval loss | Per-subject acc | Predicted counts |",
              "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |"]
    for r in rows:
        e=r["evaluation"]
        lines.append(f"| {LABELS[r['model']]} | {r['seed']} | {r['train']['accuracy']:.2%} | {e['accuracy']:.2%} | "
                     f"{e['balanced_accuracy']:.2%} | {e['loss']:.3f} | {e['per_subject']} | {e['predicted_counts']} |")
    lines += ["","## Measured cost","",
              "| Model | Parameters | Tokens incl. CLS | Mean training | Mean eval inference | Mean peak GPU allocation |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name in names:
        selected=[r for r in rows if r["model"]==name]
        gpu=[r["peak_gpu_MiB"] for r in selected if r["peak_gpu_MiB"] is not None]
        memory=f"{np.mean(gpu):.1f} MiB" if gpu else "n/a (CPU)"
        tokens="n/a" if name=="eegnet_part1_config" else 482 if name=="vanilla_transformer" else 30
        lines.append(f"| {LABELS[name]} | {selected[0]['parameters']:,} | {tokens} | "
                     f"{summary[name]['mean_training_seconds']:.3f} s | "
                     f"{np.mean([r['inference_seconds'] for r in selected]):.4f} s | {memory} |")
    lines += ["","Inference includes metric calculation and transfer to CPU for the complete evaluation set, "
              "measured once after training. Peak memory is PyTorch allocated memory, not total device usage. "
              "Per-epoch development diagnostics are excluded from training time and retained in the separate "
              "training-and-diagnostics wall-clock field."]
    lines += ["","## Paired seed differences (local minus comparator, balanced accuracy)",""]
    for name,diffs in pairing.items():
        mean=np.mean([d["delta_balanced_accuracy"] for d in diffs])
        lines.append(f"- {LABELS[name]}: mean {100*mean:+.2f} pp; by seed: "+
                     ", ".join(f"{d['seed']}: {100*d['delta_balanced_accuracy']:+.2f}" for d in diffs)+" pp.")
    lines += ["","## Actual determinism checks",""]
    for path in sorted(folder.glob("*_repeat_check.json")):
        lines.append(f"- {path.stem}: {json.loads(path.read_text())}")
    lines += ["","Repeated runs are excluded from summaries. Seed SD reflects optimization variability on the same people, "
              "not a confidence interval over new subjects. The fewer attention entries are a structural fact; "
              "runtime and accuracy benefits must be measured. Filter responses show learned coefficients, not causal motor features.",""]
    (folder/"STAGE_REPORT.md").write_text("\n".join(lines))
    return result


def study_report(output,smoke=False):
    lines=["# Controlled temporal-tokenization study"+(" — SYNTHETIC CHECK" if smoke else ""),"",
       "Hypothesis and protocol were fixed in `EXPERIMENT_PLAN.md` before implementing this model. "
       "Earlier baseline test results had already been seen; this is exploratory research.","",
       "The proposed model and its scattered-time control both match vanilla at 135,746 parameters, "
       "retain its attention backbone, and use 30 rather than 482 tokens. Attention score entries fall "
       "from 232,324 to 900 per head/layer (258.14× fewer). This is not a claim of a 258× end-to-end speedup.","",
       "Development fits subjects 1–6 and diagnoses subjects 7–8. Final retrains on 1–8 and evaluates 9–10 "
       "once at the fixed final epoch. Neither stage selects hyperparameters, epochs or a winning seed. "
       "All four configurations are rerun under matched minibatch orders and the same numerical protocol.","",
       "A rejected hypothesis is a valid research outcome. Read every support check and class-collapse diagnostic; "
       "do not call reduced seed spread a success if the model predicts only one class.",""]
    for stage in ["dev","final"]:
        folder=output/stage
        if not (folder/"STAGE_REPORT.md").exists():continue
        lines += [(folder/"STAGE_REPORT.md").read_text(),""]
    lines += ["## Interpretation guide","",
       "If local beats vanilla but matches scattered, the evidence supports the practical tokenization package "
       "without isolating a locality benefit. If local beats scattered too, this supports ordered contiguous "
       "tokenization under these conditions; the control also changes temporal positional meaning.","",
       "If efficiency improves but held-out decoding does not, report an efficiency gain and an unsupported "
       "generalization hypothesis. If local overfits too, local structure alone did not solve cross-subject shift. "
       "Neither positive nor negative results establish the physiological signal or remove the visual/ocular confound.","",
       "No attention heatmap or scalp localization claim is inferred. Learned filter spectra can suggest "
       "frequency preferences, but input is already 7–30 Hz filtered and responses outside that band are not "
       "evidence of used out-of-band EEG. Windowing/striding can discard useful information.","",
       "The study is small: five seeds, two held-out people per stage, and historical test exposure. "
       "Replicate on additional subjects/datasets to make a stronger claim.",""]
    (output/"EXPERIMENT_REPORT.md").write_text("\n".join(lines))
    if (output/"final/experiment_dashboard.png").exists():
        import shutil
        shutil.copyfile(output/"final/experiment_dashboard.png",output/"figure.png")
    final=output/"final/summary.json"
    if final.exists():
        s=json.loads(final.read_text());m=s["models"]["local_temporal"]
        note=["# Part 2 measured write-up draft"+(" — SYNTHETIC, DO NOT SUBMIT" if smoke else ""),"",
          "**Hypothesis (written before implementation):** Shared local temporal filters should reduce attention "
          "cost and improve held-out-subject balanced accuracy and seed stability relative to per-sample attention.","",
          "**Change and mechanism:** Eight learned 200 ms filters shared across all electrodes, stride 100 ms, "
          "produce 29 overlapping temporal tokens plus CLS. A learned spatial projection and the unchanged "
          "three-layer attention backbone follow. The vanilla and local models have the same 135,746 parameters. "
          "An equally sized scattered-time control tests ordered local grouping.","",
          f"**Results:** Accuracy best / worst / mean = {m['best_accuracy']:.2%} / {m['worst_accuracy']:.2%} / "
          f"{m['mean_accuracy']:.2%}. Balanced accuracy mean ± seed SD = {m['mean_ba']:.2%} ± {100*m['sd_ba']:.2f} pp. "
          f"Full prespecified descriptive support: {s['full_descriptive_support']}. Inspect the report's individual "
          "criteria and controls before writing the interpretation; do not imply a significance test.","",
          "**Figure:** `figure.png` compares training losses, held-out scores, train/evaluation gaps and runtime, "
          "and shows the local model's first-seed learned temporal filter responses.","",
          "This was an exploratory follow-up after seeing baseline test results. Repeat checks, all seeds and "
          "subject-level results are retained. Add a short, evidence-based interpretation before submission.",""]
        (output/"ANSWER_DRAFT.md").write_text("\n".join(note))
