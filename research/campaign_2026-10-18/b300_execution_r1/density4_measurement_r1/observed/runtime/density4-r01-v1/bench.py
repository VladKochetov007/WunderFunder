"""Compare three and four co-located fits without changing existing training."""
from __future__ import annotations
import csv
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path("/data/epanov_s/tabdata")
PACKET=ROOT/"runtime/pr15-host-v1"
STATE=ROOT/"state/density4-r01-v1"
sys.path.insert(0,str(PACKET))
from worker import _load_common,_replace_flag,_make_request,_phase_environment
from workspace import prepare_workspace
from validation import verify_source_tree
import controller as prior_controller

TARGETS={
  2:dict(uuid="GPU-df6fe7d0-64e5-7af2-467d-c1699cb6fc67",cpus=[56,58],
         r01="tabdata_b300_r01_v1_deep_f1_s1003",new_arm="wide"),
  7:dict(uuid="GPU-be85f55c-71a1-f389-eeed-55c63a0e3ce7",cpus=[61,63],
         r01="tabdata_b300_r01_v1_wide_f3_s1003",new_arm="deep"),
}
PATTERN=re.compile(r"ep ([0-9]+) batch ([0-9]+)/([0-9]+).*?seq/s ([0-9.]+)")

def write(path:Path,value:dict|list)->None:
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2)+"\n")
    temporary.replace(path)

def sample(paths:dict[str,Path],events:dict[str,list],phase:str)->None:
    for name,path in paths.items():
        if not path.exists():
            continue
        text=path.read_text(errors="replace")
        matches=PATTERN.findall(text)
        if not matches:
            continue
        epoch,batch,total,_=matches[-1]
        index=int(epoch)*int(total)+int(batch)
        previous=events.setdefault(name,[])
        if not previous or previous[-1]["batch_index"]!=index:
            previous.append(dict(time_unix=path.stat().st_mtime,batch_index=index,phase=phase))

def rates(events:dict[str,list],names:list[str],start:float,end:float)->dict:
    result={}
    for name in names:
        points=[r for r in events.get(name,[]) if start<=r["time_unix"]<=end]
        if len(points)<2:
            raise ValueError(f"Insufficient progress points for {name}: {len(points)}")
        span=points[-1]["time_unix"]-points[0]["time_unix"]
        if span<35:
            raise ValueError(f"Short progress window for {name}: {span}")
        delta=points[-1]["batch_index"]-points[0]["batch_index"]
        result[name]=dict(sequences_per_second=delta*8/span,observed_seconds=span,
                          completed_batches=delta,completed_optimizer_updates=delta*10)
    return result

def terminate(process:subprocess.Popen)->None:
    if process.poll() is not None:
        return
    os.killpg(process.pid,signal.SIGTERM)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGKILL)
        process.wait()

def main()->None:
    STATE.mkdir(parents=True,exist_ok=False)
    (STATE/"logs").mkdir()
    frozen=prior_controller.frozen_identity()
    verify_source_tree(ROOT/"runtime/pr15-source-e238-v1",PACKET/"source_manifest.json")
    old_state=ROOT/"state/pr15-execution-v1"
    old_events=[json.loads(x) for x in (old_state/"events.jsonl").read_text().splitlines()]
    starts=[r for r in old_events if r["event"]=="fit_started"]
    source=ROOT/"runtime/source-a266-lf-subset-v1"
    rpacket=ROOT/"runtime/b300-r01-v1"
    rm=json.loads((rpacket/"EXECUTION_MANIFEST.json").read_text())
    for relative,expected in rm["files"].items():
        assert hashlib.sha256((rpacket/relative).read_bytes()).hexdigest()==expected,relative
    sys.path.insert(0,str(rpacket/"code"))
    from fit_recipe import build_training_argv
    from fit_workspace import prepare_workspace as prepare_r01_workspace
    from adapter_common import _RECIPE,_json,_TREE_MANIFEST,verify_deployed_tree_manifest
    verify_deployed_tree_manifest(source,_json(_TREE_MANIFEST),manifest_sha256=hashlib.sha256(_TREE_MANIFEST.read_bytes()).hexdigest())
    paths={}
    by_gpu={}
    actors={}
    handles={}
    for gpu,target in TARGETS.items():
        names=[r["job"]["id"] for r in starts if r["job"]["gpu_index"]==gpu]
        assert len(names)==2,(gpu,names)
        names.append(target["r01"])
        by_gpu[gpu]=names
        for name in names:
            campaign="r01-execution-v4" if name==target["r01"] else "pr15-execution-v1"
            paths[name]=ROOT/"state"/campaign/"logs"/(name+".log")
        name=f"density4_gpu{gpu}"
        work=STATE/"work"/name
        work.mkdir(parents=True)
        split=ROOT/"splits/fold0.json"
        workspace=prepare_r01_workspace(work,ROOT/"data",split,0)
        args=build_training_argv(_json(_RECIPE),arm=target["new_arm"],fold=0,seed=1003,run_name=name,split_manifest=split,mode="profile")
        args=_replace_flag(args,"--profile-steps","32")
        args=_replace_flag(args,"--profile-warmup","2")
        request=work/"worker_request.json"
        _make_request(request,mode="profile",name=name,source_root=source,work=work,train_argv=args,predict_argv=None)
        env=_phase_environment(dict(os.environ),work=work,gpu_uuid=target["uuid"],phase="profile")
        command=[str(ROOT/".venv/bin/python"),"-u",str(PACKET/"gpu_entrypoint.py"),str(PACKET/"train_cell.py"),"--request",str(request)]
        actors[gpu]=dict(name=name,work=str(work),command=command,env=env)
    events={}
    baseline=json.loads((ROOT/"state/density4-v3/BASELINE.json").read_text())
    last_report=0
    write(STATE/"BASELINE.json",baseline)
    processes={}
    try:
        for gpu,actor in actors.items():
            log=STATE/"logs"/f"gpu{gpu}.log"
            handle=log.open("xb")
            handles[gpu]=handle
            cpus=set(TARGETS[gpu]["cpus"])
            process=subprocess.Popen(actor["command"],env=actor["env"],cwd=source,stdin=subprocess.DEVNULL,
                                     stdout=handle,stderr=subprocess.STDOUT,start_new_session=True,
                                     preexec_fn=lambda cpus=cpus:os.sched_setaffinity(0,cpus))
            processes[gpu]=process
            paths[actor["name"]]=log
        deadline=time.time()+600
        training_starts={}
        exit_times={}
        while len(exit_times)<len(processes):
            now=time.time()
            if now>deadline:
                raise TimeoutError("Bounded density profiles exceeded ten minutes")
            sample(paths,events,"four")
            for gpu,process in processes.items():
                log=paths[actors[gpu]["name"]]
                text=log.read_text(errors="replace") if log.exists() else ""
                if "loader workers" in text and gpu not in training_starts:
                    training_starts[gpu]=now
                if process.poll() is not None and gpu not in exit_times:
                    exit_times[gpu]=now
                    if process.returncode:
                        raise RuntimeError(f"Profile GPU{gpu} failed {process.returncode}: {text[-1800:]}")
            if now-last_report>20:
                write(STATE/"PROGRESS.json",dict(phase="four_process_profiles",time_unix=now,profile_pids={str(k):p.pid for k,p in processes.items()}))
                last_report=now
            time.sleep(1)
        report={}
        for gpu,names in by_gpu.items():
            name=actors[gpu]["name"]
            text=paths[name].read_text()
            def record(prefix:str)->dict:
                rows=[json.loads(line[len(prefix)+1:]) for line in text.splitlines() if line.startswith(prefix+" ")]
                assert len(rows)==1,(prefix,len(rows))
                return rows[0]
            updates=record("PROFILE_UPDATE_EVIDENCE")
            resource=record("PROFILE_RESOURCE_EVIDENCE")
            timings=record("PROFILE")
            assert updates["optimizer_completed_updates"]==340 and updates["valid_for_throughput"]
            assert resource["precision"]=="fp32" and not resource["matmul_tf32"] and not resource["cudnn_tf32"]
            assert resource["float32_matmul_precision"]=="highest"
            start=training_starts[gpu]
            end=exit_times[gpu]-1
            during=rates(events,names,start,end)
            fourth=34*8/(exit_times[gpu]-training_starts[gpu])
            during[name]=dict(sequences_per_second=fourth,completed_optimizer_updates=340,observed_seconds=exit_times[gpu]-training_starts[gpu])
            for original in names:
                if original.startswith("pr15_"):
                    fit_receipt=ROOT/"state/pr15-execution-v1/work"/original/"fit_receipt.json"
                    if fit_receipt.exists() and fit_receipt.stat().st_mtime<=end:
                        raise ValueError("Original PR15 fit transitioned during profile: "+original)
            before=sum(r["sequences_per_second"] for r in baseline[str(gpu)].values())
            after=sum(r["sequences_per_second"] for r in during.values())
            report[str(gpu)]=dict(baseline_three=baseline[str(gpu)],during_four=during,
                                  before_sequences_per_second=before,after_sequences_per_second=after,
                                  gain_fraction=after/before-1,new_r01_arm=TARGETS[gpu]["new_arm"],updates=updates,resource=resource,timings=timings,
                                  training_window_seconds=end-start)
        assert prior_controller.frozen_identity()==frozen
        write(STATE/"RESULT.json",dict(schema="tabdata-density4.v1",results=report,
                                     profile_packet_sha256=frozen,time_unix=time.time(),
                                     scientific_models_not_changed=True,
                                     note="Engineering only. Baseline reused from earlier density4-v3 three-process window; drift and short-window uncertainty remain. Added R01 profile walltime includes warmup/overhead. No model quality claim."))
        write(STATE/"COMPLETE.json",dict(time_unix=time.time()))
    finally:
        for process in processes.values():
            terminate(process)
        for handle in handles.values():
            handle.close()
        write(STATE/"SAMPLES.json",events)

if __name__=="__main__":
    try:
        main()
    except Exception as error:
        if STATE.exists():
            write(STATE/"FAILED.json",dict(error=str(error),kind=type(error).__name__,time_unix=time.time()))
        raise
