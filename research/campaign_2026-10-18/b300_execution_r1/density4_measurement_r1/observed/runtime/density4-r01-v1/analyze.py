from __future__ import annotations
import hashlib,json,subprocess,time
from pathlib import Path
ROOT=Path("/data/epanov_s/tabdata")
S=ROOT/"state/density4-r01-v1"

def rate(points:list,start:float,end:float)->dict:
    rows=[r for r in points if start<=r["time_unix"]<=end]
    if len(rows)<2:raise ValueError("insufficient incumbent points")
    span=rows[-1]["time_unix"]-rows[0]["time_unix"]
    if span<35:raise ValueError("short incumbent interval")
    return dict(sequences_per_second=(rows[-1]["batch_index"]-rows[0]["batch_index"])*8/span,span_seconds=span,point_count=len(rows),coverage=span/(end-start))

def main()->None:
    baseline=json.loads((S/"BASELINE.json").read_text())
    samples=json.loads((S/"SAMPLES.json").read_text())
    ticks=int(subprocess.check_output(["systemctl","--user","show","tabdata-density4-r01-v1","-p","ExecMainStartTimestampMonotonic","--value"],text=True))/1e6
    start=time.time()-(time.monotonic()-ticks)
    results={}
    for gpu,arm in ((2,"wide"),(7,"deep")):
        p=S/"logs"/f"gpu{gpu}.log";text=p.read_text();end=p.stat().st_mtime
        def record(label:str)->dict:
            r=[json.loads(l[len(label)+1:]) for l in text.splitlines() if l.startswith(label+" ")]
            if len(r)!=1:raise ValueError((label,len(r)))
            return r[0]
        u=record("PROFILE_UPDATE_EVIDENCE");timing=record("PROFILE");env=record("PROCESS_ENVELOPE")
        assert u["optimizer_completed_updates"]==340 and u["valid_for_throughput"] and not u["grad_scaler_enabled"]
        assert u["gradient_stats_nonfinite_steps"]==0 and env["allocator_budget_bytes"]==6*1024**3
        added=272/(end-start)
        during={n:rate(samples[n],start,end) for n in baseline[str(gpu)]}
        for n in during:
            if n.startswith("pr15_"):
                p=ROOT/"state/pr15-execution-v1/work"/n/"fit_receipt.json"
                if p.exists() and p.stat().st_mtime<=end:raise ValueError("fit transition in window")
        before_r01=sum(r["sequences_per_second"] for n,r in baseline[str(gpu)].items() if n.startswith("tabdata_b300_r01"))
        after_r01=sum(r["sequences_per_second"] for n,r in during.items() if n.startswith("tabdata_b300_r01"))+added
        before_pr15=sum(r["sequences_per_second"] for n,r in baseline[str(gpu)].items() if n.startswith("pr15_"))
        after_pr15=sum(r["sequences_per_second"] for n,r in during.items() if n.startswith("pr15_"))
        results[str(gpu)]=dict(added_r01_arm=arm,added_sequences_per_second_lower_estimate=added,
                              observed_unit_start_to_profile_exit_seconds=end-start,incumbents=during,
                              before_r01_sequences_per_second=before_r01,after_r01_sequences_per_second=after_r01,
                              r01_family_rate_gain_fraction=after_r01/before_r01-1,
                              before_pr15_sequences_per_second=before_pr15,after_pr15_sequences_per_second=after_pr15,
                              pr15_family_rate_change_fraction=after_pr15/before_pr15-1,
                              updates=u,profile=timing,envelope=env)
    result=dict(schema="tabdata-density4-r01-feasibility.v1",time_unix=time.time(),results=results,
                precision_basis="Original frozen a266 FP32 recipe and frozen process envelope; this older trainer has no PROFILE_RESOURCE_EVIDENCE record.",
                limitations="Short sequential windows, reused earlier baseline and sparse incumbent progress. Added actor rate includes unit startup, warmup and process exit. Engineering feasibility only, not sustained makespan or quality evidence.",
                scientific_recipe_unchanged=True,original_fits_and_controllers_untouched=True)
    p=S/"FEASIBILITY.json";p.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2));print("SHA256");print(hashlib.sha256(p.read_bytes()).hexdigest())

if __name__=="__main__":main()
