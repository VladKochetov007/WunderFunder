"""Freeze exact argv and source hashes; refuses to overwrite a plan version."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PREFIX = str(HERE.relative_to(ROOT))
PYTHON = ".venv/bin/python"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def argv(run, seed, fold, stage, rho, parent=None):
    command = [PYTHON, PREFIX+"/train_sam.py", "--name",run,"--sam-stage",stage,
               "--sam-rho",str(rho),"--valid-fold",str(fold),"--seed",str(seed),
               "--hidden","128","--layers","2","--chunk","2000","--n-valid-folds","4",
               "--full-valid","--aux-struct","0.3","--aux-next","0.3","--precision","fp16",
               "--wd","0.01","--fast","--eval-source","holdout" if fold<0 else "vfold"]
    if fold < 0:
        command += ["--include-heldout-valid"]
    if stage == "parent":
        command += ["--train-source","train+valid","--loss","wmse","--batch","32",
                    "--epochs","4","--lr","0.001","--propensity",f"prop_f{max(fold,0)}",
                    "--prop-power","2","--prop-eps","0.03","--valid-true-mask","0.03",
                    "--ema-teacher","1.0","--ema-decay","0.999","--eval-ema",
                    "--since-weight","32","--since-floor","0.25"]
    else:
        command += ["--train-source","valid","--loss","gcorr","--batch","16","--epochs","2",
                    "--lr","0.0005","--mask-weight","--freeze","inp,gru","--init",parent+"@ep3"]
    return command


def make_job(seed, fold, arm, stage, final=False):
    end = f"final_s{seed}" if final else f"s{seed}_f{fold}"
    parent = f"on_sam_p{arm}_{end}"
    name = parent if stage == "parent" else f"on_sam_h{arm}_{end}"
    epoch = 3 if stage == "parent" else 1
    rho = .05 if stage == "parent" and arm == "sam" else 0.
    check = [PYTHON,PREFIX+"/verify_checkpoint.py","--run",name,"--epoch",str(epoch),
             "--stage",stage,"--fold",str(fold),"--rho",str(rho)]
    predict = check if stage == "parent" or final else [
        PYTHON,PREFIX+"/predict_fixed.py",name,"--fold",str(fold),"--epoch","1"]
    return {"name":name,"seed":seed,"fold":fold,"arm":arm,"stage":stage,
            "train_argv":argv(name,seed,fold,stage,rho,parent),"predict_argv":predict,
            "endpoint":f"artifacts/runs/{name}/model_ep{epoch}.pt",
            "depends_on":[] if stage=="parent" else [parent]}


def main():
    sources = list(HERE.glob("*.py")) + [HERE/"trainer-source.json", HERE/"spec.md",
              ROOT/"scripts/train_rnn.py", HERE.parent/"run_queue.py"]
    sources += list((ROOT/"src/wnn").glob("*.py"))
    codes = {os.path.relpath(p,HERE):sha(p) for p in sorted(set(sources))}
    shared = {
      "id":"SAM-PARENT-01", "created_utc":datetime.now(timezone.utc).isoformat(),
      "status":"frozen-preexecution; independent review required; not authorization to start",
      "code_sha256":codes, "scientific_configurations":1,
      "control":"Matched rerun from random initialization, identical code/args except sam_rho=0; no old ref reuse",
      "fixed_epochs":{"parent":3,"head":1}, "fit_count_total_all_seeds":16,"finalist_fit_reserve":2,
      "metric":"global pooled clipped |target|-weighted Pearson average on pure-chain need_prediction AND is_scored rows",
      "exposure":"exposed discovery folds0/1; not a fresh holdout",
      "screen_gate":"Seed3 paired fixed .25 slot replacement: pooled >=.0003 and both folds >=0",
      "promotion_gate":"Both seed pooled deltas >0; all four seed/fold deltas >=0; mean seed pooled delta >=.0003",
      "serving_selection":"Predetermined seed3 h128 slot replacement only; never choose the better seed or a blend weight",
      "anchors_by_fold":{str(f):[f"on_ctld256_f{f}",f"on_ctl128s3_f{f}",f"on_ctl128s4_f{f}",f"on_ctlc500_f{f}"] for f in [0,1]},
      "paired_comparison":"Replace seed-matched plain-h128 slot by either new control H2 or new SAM H2; other three anchors fixed, all weights.25",
      "deployment_comparison":"Also compare seed3 SAM slot against unmodified original V1-like anchor, not only rerun control; must be nonnegative both folds",
      "readout_policy":"Only fixed H2 ep1; no parent or intermediate epoch score; one registered pooled+fold report per seed plus fixed original-anchor serving comparison",
      "resources":{"max_simultaneous_GPU_jobs":2,"GPU_fraction_per_job":.34,"max_GPU_GiB_per_job":5.5,
                   "CPU_quota_per_lane":"600%","torch_threads_per_lane":4,"loader_workers_per_lane":2,
                   "host_memory_scope_GiB":20,"timeout_seconds_per_stage":1800,
                   "new_parent_cutoff_utc":"2026-10-01T06:00:00Z","finalist_cutoff_utc":"2026-10-01T07:00:00Z"},
      "inputs":"Existing authorized train+valid datasets, fixed chain_folds.npz and fold-specific propensity caches/checkpoints; hashes to be recorded by root preflight before data-backed launch",
      "validation_policy":"suppressed; exact sequential DataLoader base-seed RNG draw retained; no best-epoch selection",
      "external_actions":"No Git push/public submission/messages",
    }
    for seed in [3,4]:
        plan = dict(shared)
        plan["id"] += f"-seed{seed}-v1"
        plan["allocation_condition"] = "root authorization after independent review" if seed==3 else "seed3 screen gate passed and root records allocation"
        plan["new_forecasting_fits"] = 8
        # run_queue's two strided lanes each own one fold, so every head follows
        # its two parents. No cross-lane checkpoint dependency exists.
        plan["jobs"] = [make_job(seed,f,arm,stage) for stage in ["parent","head"]
                        for arm in ["ctl","sam"] for f in [0,1]]
        path = HERE/f"execution-plan.seed{seed}.v1.json"
        with path.open("x") as stream:
            json.dump(plan,stream,indent=2); stream.write("\n")
    plan = dict(shared)
    plan.update(id="SAM-PARENT-01-finalist-v1",allocation_condition="full promotion and root allocation",
                new_forecasting_fits=2,required_lanes=1,
                jobs=[make_job(3,-1,"sam",stage,True) for stage in ["parent","head"]])
    with (HERE/"execution-plan.finalist.v1.json").open("x") as stream:
        json.dump(plan,stream,indent=2);stream.write("\n")
    print(json.dumps({p.name:sha(p) for p in sorted(HERE.glob("execution-plan*.json"))},indent=2))


if __name__ == "__main__":
    main()
