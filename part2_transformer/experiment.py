"""Run the locked, controlled tokenization study described in EXPERIMENT_PLAN.md."""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import random
import shutil
import time
from datetime import datetime,timezone

from prepare_data import ROOT
import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score,balanced_accuracy_score,confusion_matrix
from colab_audit import write_json,synchronize,backup
from models import EEGNet,VanillaEEGTransformer,count_params
from experiment_models import YourModel,ScatteredTimeControl

NAMES = ["eegnet_part1_config","vanilla_transformer","local_temporal","scattered_control"]


def build(name):
    return {"eegnet_part1_config":lambda:EEGNet(64,481,kern=80,p=0.5),
            "vanilla_transformer":lambda:VanillaEEGTransformer(64,481),
            "local_temporal":lambda:YourModel(64,481),
            "scattered_control":lambda:ScatteredTimeControl(64,481)}[name]()


def structural_checks():
    models = {}
    for name in NAMES:
        torch.manual_seed(0)
        models[name] = build(name)
    local,control = models["local_temporal"],models["scattered_control"]
    assert count_params(local)==count_params(control)==count_params(models["vanilla_transformer"])==135746
    assert local.n_tokens==29 and local.tokenize(torch.zeros(2,64,481)).shape==(2,29,64)
    assert all(torch.equal(p,q) for p,q in zip(local.parameters(),control.parameters()))
    assert torch.equal(torch.sort(control.time_index[:480]).values,torch.arange(480))
    assert int(control.time_index[-1])==480
    return {name:dict(parameters=count_params(model),architecture=str(model),
                      tokens=(None if name=="eegnet_part1_config" else 482 if name=="vanilla_transformer" else 30))
            for name,model in models.items()}


@torch.inference_mode()
def evaluate(model,X,y,groups,batch):
    model.eval()
    logits = torch.cat([model(X[i:i+batch]).float() for i in range(0,len(X),batch)])
    loss = float(nn.functional.cross_entropy(logits,y).item())
    probs = logits.softmax(-1).cpu().numpy()
    labels,pred = y.cpu().numpy(),probs.argmax(-1)
    counts = np.bincount(pred,minlength=2)
    metrics = dict(accuracy=float(accuracy_score(labels,pred)),balanced_accuracy=float(balanced_accuracy_score(labels,pred)),
                   loss=loss,confusion=confusion_matrix(labels,pred,labels=[0,1]).tolist(),
                   predicted_counts=counts.tolist(),collapsed=bool(counts.min()<.1*len(y)),
                   per_subject={str(int(s)):float(accuracy_score(labels[groups==s],pred[groups==s]))
                                for s in np.unique(groups)})
    return metrics,probs


def datasets(smoke):
    if smoke:
        rng = np.random.default_rng(2026)
        stages = {}
        for stage,subjects in [("dev",[7,8]),("final",[9,10])]:
            stages[stage] = dict(Xtr=rng.normal(size=(8,64,481)).astype("float32"),ytr=np.arange(8)%2,
                Xev=rng.normal(size=(4,64,481)).astype("float32"),yev=np.arange(4)%2,
                train_groups=np.ones(8,dtype=int),eval_groups=np.repeat(subjects,2),
                normalization=dict(synthetic=True))
        return stages
    from data import load_data
    X,y,Xtest,ytest = load_data(standardize=False)
    summary = json.loads((ROOT/"artifacts/data_summary.json").read_text())
    counts = summary["per_subject_trials"]
    g = np.concatenate([np.full(counts[str(s)],s) for s in range(1,9)])
    gt = np.concatenate([np.full(counts[str(s)],s) for s in [9,10]])
    assert len(g)==len(y)==360 and len(gt)==len(ytest)==90
    stages = {}
    for stage in ["dev","final"]:
        tr = g<=6 if stage=="dev" else np.ones(len(g),dtype=bool)
        a,b = (X[~tr],y[~tr]) if stage=="dev" else (Xtest,ytest)
        mean = X[tr].mean(axis=(0,2),keepdims=True)
        std = X[tr].std(axis=(0,2),keepdims=True)+1e-6
        stages[stage] = dict(Xtr=(X[tr]-mean)/std,ytr=y[tr],Xev=(a-mean)/std,yev=b,
               train_groups=g[tr],eval_groups=g[~tr] if stage=="dev" else gt,
               normalization=dict(mean=mean.ravel().tolist(),std=std.ravel().tolist(),epsilon=1e-6))
        assert not set(stages[stage]["train_groups"]) & set(stages[stage]["eval_groups"])
    return stages


def train_one(name,seed,data,args,device,folder,stage):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    model = build(name).to(device)
    X,y,A,b = [torch.as_tensor(data[k],device=device) for k in ["Xtr","ytr","Xev","yev"]]
    order_rng = torch.Generator(device="cpu").manual_seed(10000+seed)
    optimizer = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    amp = device.type=="cuda"
    scaler = torch.amp.GradScaler("cuda",enabled=amp)
    history,skipped = [],0
    if amp:torch.cuda.reset_peak_memory_stats(device)
    synchronize(device);start=time.perf_counter()
    for epoch in range(1,args.epochs+1):
        model.train();synchronize(device);tick=time.perf_counter()
        order = torch.randperm(len(y),generator=order_rng).to(device)
        loss_sum,correct = torch.zeros((),device=device),torch.zeros((),device=device,dtype=torch.long)
        for offset in range(0,len(y),64):
            ids=order[offset:offset+64]
            micro=len(ids) if name=="eegnet_part1_config" else 16
            optimizer.zero_grad(set_to_none=True)
            for j in range(0,len(ids),micro):
                ix=ids[j:j+micro]
                with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp):
                    logits=model(X[ix]);loss=nn.functional.cross_entropy(logits,y[ix])
                scaler.scale(loss*(len(ix)/len(ids))).backward()
                loss_sum+=loss.detach().float()*len(ix)
                correct+=(logits.detach().argmax(-1)==y[ix]).sum()
            old=scaler.get_scale();scaler.step(optimizer);scaler.update()
            skipped+=int(scaler.get_scale()<old)
        synchronize(device)
        row=dict(epoch=epoch,train_loss=float(loss_sum.item()/len(y)),
                 online_train_accuracy=float(correct.item()/len(y)),train_seconds=time.perf_counter()-tick,
                 skipped_amp_steps=skipped,amp_scale=float(scaler.get_scale()))
        if not np.isfinite(row["train_loss"]):raise RuntimeError("Nonfinite training loss")
        if stage=="dev":
            diagnostic,_=evaluate(model,A,b,data["eval_groups"],16)
            row.update(eval_loss=diagnostic["loss"],eval_accuracy=diagnostic["accuracy"],
                       eval_balanced_accuracy=diagnostic["balanced_accuracy"])
        history.append(row)
        write_json(folder/f"{name}_seed{seed}_history.json",history)
        print(f"{stage} {name} seed={seed} epoch={epoch}/{args.epochs} loss={row['train_loss']:.4f} "
              f"seconds={row['train_seconds']:.3f}"+(f" dev_BA={row['eval_balanced_accuracy']:.3f}" if stage=="dev" else ""),flush=True)
    synchronize(device);wall=time.perf_counter()-start
    training,_=evaluate(model,X,y,data["train_groups"],16)
    synchronize(device);tick=time.perf_counter()
    evaluation,probs=evaluate(model,A,b,data["eval_groups"],16)
    synchronize(device);inference=time.perf_counter()-tick
    result=dict(model=name,seed=seed,stage=stage,epochs=args.epochs,parameters=count_params(model),
        train=training,evaluation=evaluation,training_seconds=sum(h["train_seconds"] for h in history),
        training_and_dev_diagnostics_seconds=wall,inference_seconds=inference,
        peak_gpu_MiB=torch.cuda.max_memory_allocated(device)/2**20 if amp else None,
        skipped_amp_steps=skipped,history=history)
    write_json(folder/f"{name}_seed{seed}.json",result)
    with (folder/f"{name}_seed{seed}_predictions.csv").open("w",newline="") as file:
        writer=csv.writer(file);writer.writerow(["subject","true","pred","prob_feet"])
        writer.writerows(zip(data["eval_groups"],data["yev"],probs.argmax(-1),probs[:,1]))
    checkpoints=folder/"checkpoints";checkpoints.mkdir(exist_ok=True)
    torch.save(dict(model=name,seed=seed,state_dict={k:v.cpu() for k,v in model.state_dict().items()},
                    normalization=data["normalization"],input_shape=[64,481]),checkpoints/f"{name}_seed{seed}.pt")
    # First-seed diagnostics inspect fitted temporal filters, without choosing a model.
    if name=="local_temporal" and seed==args.seeds[0]:
        np.save(folder/"temporal_filters.npy",model.temporal.weight.detach().cpu().numpy()[:,0])
    print(f"RESULT {stage} {name} seed={seed}: train={training['accuracy']:.2%}, "
          f"eval={evaluation['accuracy']:.2%}, BA={evaluation['balanced_accuracy']:.2%}",flush=True)
    del model,optimizer,X,y,A,b
    if amp:torch.cuda.empty_cache()
    return result


def verify_repeat(name,seed,original,repeated,folder,repeat_folder):
    a=torch.load(folder/"checkpoints"/f"{name}_seed{seed}.pt",weights_only=True)["state_dict"]
    b=torch.load(repeat_folder/"checkpoints"/f"{name}_seed{seed}.pt",weights_only=True)["state_dict"]
    fields=["train_loss","online_train_accuracy","amp_scale","skipped_amp_steps"]
    if original["stage"]=="dev":fields += ["eval_loss","eval_balanced_accuracy"]
    checks=dict(weights_exact=all(torch.equal(a[k],b[k]) for k in a),
        predictions_exact=(folder/f"{name}_seed{seed}_predictions.csv").read_bytes()==(repeat_folder/f"{name}_seed{seed}_predictions.csv").read_bytes(),
        losses_exact=all(all(x[k]==y[k] for k in fields) for x,y in zip(original["history"],repeated["history"])))
    write_json(folder/f"{name}_repeat_check.json",checks)
    if not all(checks.values()):raise RuntimeError(f"Exact repeat failed for {name}: {checks}")
    print(f"REPRODUCIBILITY PASS {name}: {checks}",flush=True)


def main(prepare_inputs=False):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device",choices=["cuda","cpu"],default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--stages",nargs="+",choices=["dev","final"],default=["dev","final"])
    parser.add_argument("--seeds",nargs="+",type=int,default=[0,1,2,3,4])
    parser.add_argument("--epochs",type=int,default=40)
    parser.add_argument("--sync-dir",type=__import__('pathlib').Path)
    parser.add_argument("--verify-repeat",action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument("--smoke",action="store_true",help="Synthetic data; not a scientific result")
    args=parser.parse_args()
    if args.epochs<1 or len(args.seeds)!=len(set(args.seeds)):parser.error("Positive epochs and unique seeds required")
    if len(args.stages)!=len(set(args.stages)):parser.error("Stages must be unique")
    if args.smoke:args.epochs,args.seeds=1,[0]
    device=torch.device(args.device)
    if device.type=="cuda" and not torch.cuda.is_available():raise RuntimeError("Select a Colab T4 GPU runtime")
    if prepare_inputs and not args.smoke:
        from prepare_data import prepare
        prepare()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if device.type=="cuda":
        torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        if hasattr(torch.backends.cuda,"enable_cudnn_sdp"):torch.backends.cuda.enable_cudnn_sdp(False)
    architecture=structural_checks()
    output=ROOT/"artifacts"/(("study_smoke_" if args.smoke else "study_")+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(parents=True)
    metadata=dict(status="running",arguments={k:str(v) if hasattr(v,'__fspath__') else v for k,v in vars(args).items()},
       gpu=torch.cuda.get_device_name(device) if device.type=="cuda" else None,
       python=platform.python_version(),cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
       packages={p:importlib.metadata.version(p) for p in ["torch","numpy","mne","scipy","scikit-learn","matplotlib"]},
       architectures=architecture,primary="held-out balanced accuracy mean and SD across seeds",
       historical_test_exposure=True,strict_determinism=True,
       source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
        ["main.py","experiment.py","experiment_models.py","experiment_report.py","EXPERIMENT_PLAN.md","models.py","data.py","colab_audit.py"]})
    write_json(output/"metadata.json",metadata)
    shutil.copy2(ROOT/"EXPERIMENT_PLAN.md",output/"EXPERIMENT_PLAN.md")
    print(f"Study output: {output}",flush=True)
    try:
        data_stages=datasets(args.smoke)
        for stage in args.stages:
            folder=output/stage;folder.mkdir()
            data=data_stages[stage]
            write_json(folder/"split.json",dict(train_subjects=np.unique(data["train_groups"]).tolist(),
              evaluation_subjects=np.unique(data["eval_groups"]).tolist(),n_train=len(data["ytr"]),
              n_evaluation=len(data["yev"]),normalization=data["normalization"]))
            for name in NAMES:
                for seed in args.seeds:
                    result=train_one(name,seed,data,args,device,folder,stage)
                    with (folder/"results.jsonl").open("a") as file:file.write(json.dumps({k:v for k,v in result.items() if k!="history"})+"\n")
                    backup(output,args.sync_dir)
                    if args.verify_repeat and seed==args.seeds[0]:
                        repeat_folder=folder/"repeats"/name;repeat_folder.mkdir(parents=True)
                        repeated=train_one(name,seed,data,args,device,repeat_folder,stage)
                        verify_repeat(name,seed,result,repeated,folder,repeat_folder)
                        backup(output,args.sync_dir)
            from experiment_report import stage_report
            stage_report(folder,args.smoke)
            backup(output,args.sync_dir)
        from experiment_report import study_report
        study_report(output,args.smoke)
        metadata["status"]="complete"
        print(f"COMPLETE: {output}",flush=True)
    except BaseException as error:
        metadata.update(status="interrupted" if isinstance(error,KeyboardInterrupt) else "failed",error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output/"metadata.json",metadata);backup(output,args.sync_dir)


if __name__=="__main__":main()
