"""CPU baseline audit. No model changes and no test-driven checkpoint selection.

Run: ../.venv/bin/python -B -u audit_baselines.py
Data: run prepare_data.py first. All outputs remain under Part 2.
"""
import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

from prepare_data import ROOT  # establish local cache directories before imports
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

from models import EEGNet, VanillaEEGTransformer, count_params
from train import set_seed

CONFIGS = {
    "eegnet_starter": {"kern": 64, "p": 0.25},
    "eegnet_part1_config": {"kern": 80, "p": 0.5},
    "vanilla_transformer": {},
}


def build(name, channels, samples):
    constructor = VanillaEEGTransformer if name == "vanilla_transformer" else EEGNet
    return constructor(channels, samples, **CONFIGS[name])


def describe(name, channels, samples):
    set_seed(0)
    model = build(name, channels, samples)
    trace, handles = {}, []
    def hook(label):
        def record(module, inputs, output):
            trace[label] = list(output.shape)
        return record
    for label, module in model.named_children():
        handles.append(module.register_forward_hook(hook(label)))
    model.eval()
    with torch.no_grad():
        logits = model(torch.zeros(2, channels, samples))
        assert logits.shape == (2, 2) and torch.isfinite(logits).all()
    for handle in handles:
        handle.remove()
    result = dict(parameters=count_params(model), config=CONFIGS[name], trace_batch2=trace,
                  module_parameters={k: sum(p.numel() for p in v.parameters())
                                     for k, v in model.named_children()}, architecture=str(model))
    if name == "vanilla_transformer":
        n = samples + 1  # include CLS
        result.update(tokens=n, heads=4, layers=3, head_dim=16,
                      attention_entries_per_head=n*n,
                      attention_entries_batch64_all_heads_one_layer=64*4*n*n,
                      conceptual_attention_float32_MiB_per_layer=64*4*n*n*4/(1024**2),
                      attention_QK_AV_MACs_per_trial_all_layers=3*2*n*n*64)
    return result


@torch.no_grad()
def predict(model, X, batch=32):
    model.eval()
    return np.concatenate([model(torch.from_numpy(X[i:i+batch])).argmax(1).numpy()
                           for i in range(0, len(X), batch)])


def run(name, seed, data, args, output):
    set_seed(seed)
    Xtr, ytr, Xte, yte = (data[k] for k in ["Xtr", "ytr", "Xte", "yte"])
    model = build(name, Xtr.shape[1], Xtr.shape[2])
    loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                        batch_size=args.batch, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()
    history = []
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        loss_sum, correct, total = 0.0, 0, 0
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite loss: {name}, seed {seed}, epoch {epoch}")
            loss.backward()
            if epoch == 1 and total == 0:
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                assert grads and all(torch.isfinite(g).all() for g in grads)
                first_grad_norm = float(torch.sqrt(sum(g.square().sum() for g in grads)))
            optimizer.step()
            total += len(yb)
            loss_sum += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
        row = dict(epoch=epoch, loss=loss_sum/total, online_train_accuracy=correct/total,
                   seconds=time.perf_counter()-epoch_start)
        history.append(row)
        # Save progress every epoch so interrupted audits retain useful evidence.
        (output / f"{name}_seed{seed}_history.json").write_text(json.dumps(history, indent=2))
        print(f"{name} seed={seed} epoch={epoch}/{args.epochs} loss={row['loss']:.4f} "
              f"train={row['online_train_accuracy']:.3f} seconds={row['seconds']:.2f}", flush=True)
    train_seconds = time.perf_counter() - start
    train_pred = predict(model, Xtr)
    # Final epoch was chosen in advance. Test labels are first used below.
    pred = predict(model, Xte)
    groups = data["groups_te"]
    result = dict(model=name, seed=seed, epochs=args.epochs, parameters=count_params(model),
                  selection="prespecified_final_epoch", train_subjects=list(range(1,9)),
                  test_subjects=[9,10], train_accuracy=accuracy_score(ytr, train_pred),
                  test_accuracy=accuracy_score(yte, pred),
                  balanced_accuracy=balanced_accuracy_score(yte, pred),
                  confusion_matrix=confusion_matrix(yte, pred, labels=[0,1]).tolist(),
                  predicted_class_counts=np.bincount(pred,minlength=2).tolist(),
                  per_subject={str(int(s)): accuracy_score(yte[groups == s], pred[groups == s])
                               for s in np.unique(groups)},
                  train_seconds=train_seconds, total_seconds=time.perf_counter()-start,
                  first_batch_gradient_norm=first_grad_norm, history=history)
    np.savez(output / f"{name}_seed{seed}_predictions.npz", y=yte, pred=pred, subjects=groups)
    torch.save(model.state_dict(), ROOT / ".cache" / f"{name}_seed{seed}.pt")
    (output / f"{name}_seed{seed}.json").write_text(json.dumps(result, indent=2))
    print(f"RESULT {name} seed={seed}: train={result['train_accuracy']:.3f}, "
          f"test={result['test_accuracy']:.3f}, train_seconds={train_seconds:.1f}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0,1,2])
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--models", nargs="+", choices=list(CONFIGS), default=list(CONFIGS))
    args = parser.parse_args()
    if min(args.epochs,args.threads,args.batch) < 1:
        parser.error("epochs, threads and batch must be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    output = ROOT / "artifacts" / ("audit_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(parents=True)
    data = np.load(ROOT / ".cache/preprocessed.npz")
    metadata = dict(arguments=vars(args), versions=dict(python=platform.python_version(),
                    torch=torch.__version__, numpy=np.__version__), device="cpu",
                    source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
                                   for p in ["data.py","models.py","train.py","audit_baselines.py"]},
                    architectures={name:describe(name,*data["Xtr"].shape[1:]) for name in args.models})
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Audit output: {output}", flush=True)
    results = []
    for name in args.models:
        for seed in args.seeds:
            result = run(name, seed, data, args, output)
            results.append(result)
            with (output / "results.jsonl").open("a") as file:
                file.write(json.dumps({k:v for k,v in result.items() if k != "history"}) + "\n")
    summary = {}
    for name in args.models:
        selected = [r for r in results if r["model"] == name]
        scores = [r["test_accuracy"] for r in selected]
        summary[name] = dict(best=max(scores), worst=min(scores), mean=float(np.mean(scores)),
                             std=float(np.std(scores)),
                             mean_train_seconds=float(np.mean([r["train_seconds"] for r in selected])))
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(11,4))
    for name in args.models:
        selected = [r for r in results if r["model"] == name]
        losses = np.array([[h["loss"] for h in r["history"]] for r in selected])
        axes[0].plot(np.arange(1,args.epochs+1),losses.mean(0),label=name)
        axes[1].scatter([name]*len(selected),[r["test_accuracy"] for r in selected])
    axes[0].set(xlabel="Epoch",ylabel="Training cross-entropy")
    axes[0].legend(fontsize=7)
    axes[1].axhline(0.5,color="gray",linestyle="--",label="50% reference")
    axes[1].set(ylabel="Final-epoch test accuracy",ylim=(0,1))
    axes[1].tick_params(axis="x",labelrotation=20)
    fig.tight_layout()
    fig.savefig(output / "baseline_audit.png",dpi=150)
    plt.close(fig)
    print(json.dumps(summary,indent=2),flush=True)
    print(f"Completed: {output}",flush=True)


if __name__ == "__main__":
    main()
