"""T4-friendly audit of exactly Part-1-config EEGNet and the vanilla transformer.

No architecture changes. FP16 AMP, effective batch 64 via microbatches, local
data, final-epoch evaluation, synchronized GPU timing and per-seed Drive backup.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from prepare_data import ROOT
import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from models import EEGNet, VanillaEEGTransformer, count_params
from audit_baselines import describe
from train import set_seed

MODEL_NAMES = ("eegnet_part1_config", "vanilla_transformer")


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def backup(output, destination):
    if destination is not None:
        target = destination / output.name
        target.mkdir(parents=True, exist_ok=True)
        shutil.copytree(output, target, dirs_exist_ok=True)


def build(name, channels, samples):
    if name == "eegnet_part1_config":
        return EEGNet(channels, samples, kern=80, p=0.5)
    if name == "vanilla_transformer":
        return VanillaEEGTransformer(channels, samples)
    raise ValueError(name)


@torch.inference_mode()
def predictions(model, X, batch):
    # Evaluate all models in float32, including those trained with AMP.
    model.eval()
    return torch.cat([model(X[i:i+batch]).float().softmax(-1)
                      for i in range(0, len(X), batch)]).cpu().numpy()


def run_seed(name, seed, arrays, args, device, output):
    set_seed(seed)
    random.seed(seed)
    Xtr, ytr, Xte, yte, groups = arrays
    model = build(name, Xtr.shape[1], Xtr.shape[2]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    amp = device.type == "cuda" and args.precision == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    criterion = nn.CrossEntropyLoss()
    history, skipped_steps = [], 0
    first_grad_norm = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        synchronize(device)
        epoch_start = time.perf_counter()
        order = torch.randperm(len(ytr), device=device)
        loss_sum = torch.zeros((), device=device)
        correct = torch.zeros((), device=device, dtype=torch.long)
        for offset in range(0, len(order), args.batch):
            batch_ids = order[offset:offset+args.batch]
            optimizer.zero_grad(set_to_none=True)
            # Scale each microbatch by its actual fraction of this effective
            # batch, including the final (smaller) batch. One optimizer update.
            micro_size = len(batch_ids) if name == "eegnet_part1_config" else args.microbatch
            for micro in range(0, len(batch_ids), micro_size):
                ids = batch_ids[micro:micro+micro_size]
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                    logits = model(Xtr[ids])
                    loss = criterion(logits, ytr[ids])
                scaler.scale(loss * (len(ids) / len(batch_ids))).backward()
                loss_sum += loss.detach().float() * len(ids)
                correct += (logits.detach().argmax(-1) == ytr[ids]).sum()
            if first_grad_norm is None:
                scaler.unscale_(optimizer)
                squared = sum(p.grad.detach().float().square().sum()
                              for p in model.parameters() if p.grad is not None)
                norm = float(squared.sqrt().item())
                if np.isfinite(norm):
                    first_grad_norm = norm
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped_steps += int(scaler.get_scale() < old_scale)
        synchronize(device)
        row = dict(epoch=epoch, loss=float(loss_sum.item()/len(ytr)),
                   online_train_accuracy=float(correct.item()/len(ytr)),
                   seconds=time.perf_counter()-epoch_start,
                   amp_scale=float(scaler.get_scale()), skipped_optimizer_steps=skipped_steps)
        if not np.isfinite(row["loss"]):
            raise ValueError(f"Nonfinite training loss in {name}, seed {seed}")
        history.append(row)
        write_json(output / f"{name}_seed{seed}_history.json", history)
        print(f"{name} seed={seed} epoch={epoch}/{args.epochs} loss={row['loss']:.4f} "
              f"train={row['online_train_accuracy']:.3f} time={row['seconds']:.3f}s", flush=True)
    synchronize(device)
    training_seconds = time.perf_counter()-start
    train_probs = predictions(model, Xtr, args.microbatch)
    synchronize(device)
    inference_start = time.perf_counter()
    probs = predictions(model, Xte, args.microbatch)
    synchronize(device)
    inference_seconds = time.perf_counter()-inference_start
    truth = yte.cpu().numpy()
    pred = probs.argmax(-1)
    train_truth = ytr.cpu().numpy()
    result = dict(model=name, seed=seed, epochs=args.epochs, parameters=count_params(model),
                  selection="prespecified_final_epoch", synthetic_smoke=args.smoke,
                  train_accuracy=float(accuracy_score(train_truth, train_probs.argmax(-1))),
                  test_accuracy=float(accuracy_score(truth, pred)),
                  balanced_accuracy=float(balanced_accuracy_score(truth, pred)),
                  confusion_matrix=confusion_matrix(truth, pred, labels=[0,1]).tolist(),
                  predicted_class_counts=np.bincount(pred,minlength=2).tolist(),
                  per_subject={str(int(s)):float(accuracy_score(truth[groups==s],pred[groups==s]))
                               for s in np.unique(groups)},
                  train_seconds=training_seconds, test_inference_seconds=inference_seconds,
                  test_inference_trials=len(truth),
                  peak_gpu_allocated_MiB=(torch.cuda.max_memory_allocated(device)/2**20
                                          if device.type == "cuda" else None),
                  first_finite_gradient_norm=first_grad_norm,
                  skipped_optimizer_steps=skipped_steps, history=history)
    if first_grad_norm is None:
        raise ValueError("No finite gradient norm observed; inspect AMP stability")
    with (output / f"{name}_seed{seed}_predictions.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["subject","true_class","predicted_class","probability_feet"])
        writer.writerows(zip(groups.tolist(),truth.tolist(),pred.tolist(),probs[:,1].tolist()))
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    torch.save(dict(model=name, seed=seed, state_dict={k:v.cpu() for k,v in model.state_dict().items()},
                    input_shape=list(Xtr.shape[1:]), precision=args.precision,
                    epochs=args.epochs, selection="prespecified_final_epoch"),
               checkpoint_dir / f"{name}_seed{seed}.pt")
    write_json(output / f"{name}_seed{seed}.json",result)
    print(f"RESULT {name} seed={seed}: train={result['train_accuracy']:.2%}, "
          f"test={result['test_accuracy']:.2%}, training={training_seconds:.1f}s, "
          f"test inference={inference_seconds:.3f}s",flush=True)
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cuda","cpu"], default="cuda")
    parser.add_argument("--precision", choices=["fp16","fp32"], default="fp16")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0,1,2])
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--microbatch", type=int, default=16)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--sync-dir", type=Path)
    parser.add_argument("--strict-determinism", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verify-repeat", action=argparse.BooleanOptionalAction, default=True,
                        help="Repeat first seed for both models and require exact weights/losses/predictions")
    parser.add_argument("--smoke", action="store_true", help="Synthetic one-epoch plumbing check; not scientific results")
    args = parser.parse_args()
    if min(args.epochs,args.batch,args.microbatch,args.threads) < 1 or args.microbatch > args.batch:
        parser.error("Positive sizes required, with microbatch <= effective batch")
    if len(args.seeds) != len(set(args.seeds)):
        parser.error("Seeds must be unique")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU. In Colab select Runtime > Change runtime type > T4 GPU.")
    if device.type == "cpu":
        args.precision = "fp32"
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(args.strict_determinism)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.strict_determinism and device.type == "cuda":
        # T4-compatible deterministic fallback; do not assume FlashAttention.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            torch.backends.cuda.enable_cudnn_sdp(False)
    if args.smoke:
        args.epochs, args.seeds = 1, [0]
        rng = np.random.default_rng(2026)
        data = dict(Xtr=rng.normal(size=(8,64,481)).astype("float32"), ytr=np.arange(8)%2,
                    Xte=rng.normal(size=(4,64,481)).astype("float32"), yte=np.arange(4)%2,
                    groups_te=np.array([9,9,10,10]))
    else:
        with np.load(ROOT / ".cache/preprocessed.npz") as cached:
            data = {k:cached[k] for k in cached.files}
        assert data["Xtr"].shape == (360,64,481) and data["Xte"].shape == (90,64,481)
        assert set(data["groups_tr"]) == set(range(1,9)) and set(data["groups_te"]) == {9,10}
        assert all(np.isfinite(data[k]).all() for k in ["Xtr","Xte"])
    arrays = [torch.as_tensor(data[k],device=device) for k in ["Xtr","ytr","Xte","yte"]]
    arrays.append(data["groups_te"])
    prefix = "smoke_" if args.smoke else "colab_"
    output = ROOT / "artifacts" / (prefix + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(parents=True)
    metadata = dict(status="running",arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                    device=str(device), gpu=torch.cuda.get_device_name(device) if device.type=="cuda" else None,
                    compute_capability=list(torch.cuda.get_device_capability(device)) if device.type=="cuda" else None,
                    cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), python=platform.python_version(),
                    packages={k:importlib.metadata.version(k) for k in ["torch","mne","numpy","scipy","scikit-learn","matplotlib"]},
                    models=list(MODEL_NAMES), train_subjects=list(range(1,9)), test_subjects=[9,10],
                    evaluation_precision="fp32", normalization="Part 2 starter: training-only channel statistics, epsilon=1e-6 V",
                    reproducibility="Strict deterministic kernels by default; same stack/device required for bitwise repeats.",
                    attention_backend="Deterministic math SDPA" if args.strict_determinism else "Automatic SDPA dispatch",
                    cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"],
                    eegnet_training_microbatch=args.batch,
                    transformer_training_microbatch=args.microbatch,
                    source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
                                   ["models.py","data.py","train.py","colab_audit.py","colab_report.py","prepare_data.py"]},
                    architectures={name:describe(name,64,481) for name in MODEL_NAMES})
    write_json(output / "metadata.json",metadata)
    for name in ["dataset_manifest.json","data_summary.json"]:
        source = ROOT / "artifacts" / name
        if source.exists() and not args.smoke:
            shutil.copyfile(source,output / name)
    print(f"Output: {output}\nDevice: {metadata['gpu'] or device}; precision: {args.precision}",flush=True)
    results = []
    try:
        backup(output,args.sync_dir)
        for name in MODEL_NAMES:
            for seed in args.seeds:
                result = run_seed(name,seed,arrays,args,device,output)
                results.append(result)
                with (output / "results.jsonl").open("a") as file:
                    file.write(json.dumps({k:v for k,v in result.items() if k!="history"}) + "\n")
                backup(output,args.sync_dir)
                if args.verify_repeat and seed == args.seeds[0]:
                    repeat_dir = output / "repeat_checks" / f"{name}_seed{seed}"
                    repeat_dir.mkdir(parents=True)
                    repeated = run_seed(name,seed,arrays,args,device,repeat_dir)
                    original_state = torch.load(output/"checkpoints"/f"{name}_seed{seed}.pt",weights_only=True)["state_dict"]
                    repeated_state = torch.load(repeat_dir/"checkpoints"/f"{name}_seed{seed}.pt",weights_only=True)["state_dict"]
                    weights_equal = all(torch.equal(original_state[k],repeated_state[k]) for k in original_state)
                    predictions_equal = ((output/f"{name}_seed{seed}_predictions.csv").read_bytes() ==
                                         (repeat_dir/f"{name}_seed{seed}_predictions.csv").read_bytes())
                    fields = ["loss","online_train_accuracy","amp_scale","skipped_optimizer_steps"]
                    history_equal = all(all(a[k]==b[k] for k in fields)
                                        for a,b in zip(result["history"],repeated["history"]))
                    verification = dict(model=name,seed=seed,weights_exact=weights_equal,
                                        predictions_exact=predictions_equal,training_history_exact=history_equal,
                                        extra_training_seconds=repeated["train_seconds"])
                    write_json(output/f"{name}_repeat_verification.json",verification)
                    if not (weights_equal and predictions_equal and history_equal):
                        raise RuntimeError(f"Deterministic repeat check failed: {verification}")
                    print(f"REPRODUCIBILITY PASS: {name}, seed={seed}, exact weights/predictions/losses",flush=True)
                    backup(output,args.sync_dir)
        from colab_report import generate_report
        generate_report(output)
        metadata["status"] = "complete"
        print(f"COMPLETE: {output}",flush=True)
    except BaseException as error:
        metadata.update(status="interrupted" if isinstance(error,KeyboardInterrupt) else "failed",
                        error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output / "metadata.json",metadata)
        backup(output,args.sync_dir)


if __name__ == "__main__":
    main()
