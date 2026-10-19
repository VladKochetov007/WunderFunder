B=pathlib.Path("/home/grinchenkoeg/computer-vision-homework-20261018")
DEST=B/"support-source/eda19-r6-v2"
OUT=B/"research_tmp/nf19-r6-v2"
UNIT="computer-vision-homework-eda19-r6-v2.service"
PY="/home/grinchenkoeg/WunderFunder/.venv/bin/python"
def sha(p):
 return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
 with p.open("x") as f:json.dump(x,f,indent=2);f.write("\n")
def props(unit):
 p=subprocess.run(["systemctl","--user","show",unit,"-p","LoadState","-p","ActiveState","-p","SubState","-p","ControlGroup","-p","CPUQuotaPerSecUSec","-p","MemoryHigh","-p","MemoryMax","-p","MemorySwapMax","-p","MemoryCurrent","-p","ExecMainStatus"],capture_output=True,text=True,timeout=10)
 return dict(s.split("=",1) for s in p.stdout.splitlines() if "=" in s)
def age(s):
 return (datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(s)).total_seconds()
assert not DEST.exists() and not OUT.exists(),"create-only destination/output collision"
assert props(UNIT).get("LoadState")=="not-found","support unit already exists"
planpath=B/"research/campaign_2026-10-18/priority25_execution_r4/f3_beleriand_preparation_r1/F3_BELERIAND_PLAN.json"
assert sha(planpath)=="b303146cf9db05e0a3bddf8f245d3c9b880d055ce7e97ebe5dd13510b5f4a2d8"
plan=json.loads(planpath.read_bytes())
seq=json.loads((B/"runtime/receipts/RANK01_F3_S1001_SEQUENCE.json").read_bytes())
assert seq["status"]=="running" and seq["index"]==3 and seq["attempt"]==plan["steps"][3]["attempt"]
assert 0<=age(seq["heartbeat_utc"])<120 and 0<=age(seq["worker_heartbeat"])<120
controller="computer-vision-homework-r01-f3-sequence.service"
worker="wnn-bandit-"+seq["attempt"]+".service"
parent=props("wnn-campaign.slice")
assert parent["CPUQuotaPerSecUSec"]=="2s"
assert [parent[k] for k in ("MemoryHigh","MemoryMax","MemorySwapMax")]==["18000000000","20000000000","0"]
cg=pathlib.Path("/sys/fs/cgroup")/parent["ControlGroup"].lstrip("/")
assert not (cg/"cgroup.procs").read_text().strip(),"parent contains direct processes"
populated={}
for path in cg.rglob("cgroup.procs"):
 if path.read_text().strip():
  rel=path.parent.relative_to(cg)
  assert len(rel.parts)>0
  populated[str(rel)]=path.read_text().split()
assert set(x.split("/")[0] for x in populated)=={controller,worker},populated
units={u:props(u) for u in (controller,worker)}
assert all(d["ActiveState"]=="active" and d["SubState"]=="running" for d in units.values()),units
assert units[controller]["CPUQuotaPerSecUSec"]=="100ms" and units[worker]["CPUQuotaPerSecUSec"]=="1s"
assert units[controller]["MemoryMax"]=="512000000" and units[worker]["MemoryMax"]=="8000000000"
assert all(d["MemorySwapMax"]=="0" for d in units.values())
assert 0.1+1+0.5<=2 and 512000000+8000000000+1500000000<=17000000000
assert int(parent["MemoryCurrent"])+1500000000<17000000000
events=(cg/"memory.events").read_text()
events_local=(cg/"memory.events.local").read_text()
assert dict(x.split() for x in events.splitlines())["oom"]=="0"
assert dict(x.split() for x in events.splitlines())["oom_kill"]=="0"
scratch=B/"research_tmp"
fs=subprocess.run(["findmnt","-n","-o","FSTYPE","-T",str(scratch)],capture_output=True,text=True,timeout=5)
assert fs.returncode==0 and fs.stdout.strip() not in ("tmpfs","ramfs")
assert shutil.disk_usage(scratch).free>2000000000
meminfo=dict(line.replace(":","").split()[:2] for line in pathlib.Path("/proc/meminfo").read_text().splitlines())
assert int(meminfo["MemAvailable"])*1024>5000000000
gpu=subprocess.run(["nvidia-smi","-i","GPU-a50d0c3b-d28f-a2a7-8b43-3e02b543fc0c","--query-compute-apps=pid,gpu_uuid,used_memory","--format=csv,noheader"],capture_output=True,text=True,timeout=10)
assert gpu.returncode==0
gpu_rows=[]
for line in gpu.stdout.splitlines():
 cols=[x.strip() for x in line.split(",")]
 if len(cols)!=3:continue
 proc=pathlib.Path("/proc")/cols[0]
 try:pcg=(proc/"cgroup").read_text()
 except FileNotFoundError:continue
 gpu_rows.append(dict(pid=int(cols[0]),uuid=cols[1],memory=cols[2],cgroup=pcg,campaign="wnn-campaign.slice" in pcg))
ours=[d for d in gpu_rows if d["campaign"]]
assert len(ours)==1 and worker in ours[0]["cgroup"],gpu_rows
sample=B/"research_tmp/nf19-sample-r3-v2/SAMPLE.json"
assert sha(sample)=="b89455ba449f31bb9ebe815976d4da23bb1a8e499e421b6bb5f8d44566f76e64"
checkpoint=B/"checkpoint_snapshots/rr18_r01_wide_f0_s1001_e6/epoch5_terminal/model_ep5.pt"
assert checkpoint.is_file() and not checkpoint.is_symlink() and checkpoint.stat().st_size==8905959
assert sha(checkpoint)=="78c1413de8eeaa51f803cf59d6b0d7d2eddaad74cc91f744b8b2f85466b61b7f"
canonical={"data.py":"a93da840137adbd7afdb79a22687fb73150e77107f91ca2adc5211b854bc7e00","gmetric.py":"2cf10e740a831379f62b7d1269456053f5c1f9efa2dbab03156e366972bed5aa"}
for name,digest in canonical.items():assert sha(B/"source-a266b036/src/wnn"/name)==digest
DEST.mkdir()
for name,entry in CFG["files"].items():
 rel=pathlib.PurePosixPath(name)
 assert not rel.is_absolute() and ".." not in rel.parts and len(rel.parts)==2
 target=DEST/name
 target.parent.mkdir(exist_ok=True)
 payload=base64.b64decode(entry["base64"],validate=True)
 assert len(payload)==entry["bytes"] and hashlib.sha256(payload).hexdigest()==entry["sha256"]
 with target.open("xb") as f:f.write(payload)
 assert sha(target)==entry["sha256"]
# Independent pre-invocation full byte check, before importing any staged module.
for name,entry in CFG["files"].items():assert sha(DEST/name)==entry["sha256"]
for name,digest in canonical.items():assert sha(B/"source-a266b036/src/wnn"/name)==digest
latest=json.loads((B/"runtime/receipts/RANK01_F3_S1001_SEQUENCE.json").read_bytes())
assert latest["index"]==seq["index"] and latest["attempt"]==seq["attempt"] and latest["status"]=="running"
assert not OUT.exists()
command=["systemd-run","--user","--unit="+UNIT,"--slice=wnn-campaign.slice","--property=Type=exec","--property=CPUQuota=50%","--property=MemoryHigh=1400000000","--property=MemoryMax=1500000000","--property=MemorySwapMax=0","--property=RuntimeMaxSec=600","--property=RemainAfterExit=yes","--property=StandardOutput=append:"+str(DEST/"probe.log"),"--property=StandardError=append:"+str(DEST/"probe.log"),"--working-directory="+str(DEST),"/usr/bin/env","CUDA_VISIBLE_DEVICES=","OMP_NUM_THREADS=1","OPENBLAS_NUM_THREADS=1","MKL_NUM_THREADS=1","NUMEXPR_NUM_THREADS=1","PYTHONPATH="+str(B/"source-a266b036/src")+":"+str(DEST),"TMPDIR="+str(scratch),PY,"-B",str(DEST/"idea_screens_r6/run_probe.py"),"--sample",str(B/"research_tmp/nf19-sample-r3-v2"),"--checkpoint",str(checkpoint),"--output",str(OUT),"--expected-sample-sha256","b89455ba449f31bb9ebe815976d4da23bb1a8e499e421b6bb5f8d44566f76e64","--expected-checkpoint-sha256","78c1413de8eeaa51f803cf59d6b0d7d2eddaad74cc91f744b8b2f85466b61b7f","--expected-preparation-manifest-sha256","__R6_MANIFEST_SHA__"]
record=dict(schema="nf19.r6_probe_launch.v2",utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),unit=UNIT,sequence=latest,source_sha256={n:v["sha256"] for n,v in CFG["files"].items()},canonical_source_sha256=canonical,sample_sha256=sha(sample),checkpoint_sha256=sha(checkpoint),admission=dict(parent=parent,units=units,populated=populated,memory_events=events,memory_events_local=events_local,gpu_processes=gpu_rows,scratch_fs=fs.stdout.strip(),scratch_free=shutil.disk_usage(scratch).free,host_mem_available=int(meminfo["MemAvailable"])*1024),command=command,output=str(OUT))
write(DEST/"ROOT_ADMISSION.json",record)
p=subprocess.run(command,capture_output=True,text=True,timeout=20)
record.update(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
write(DEST/"ROOT_LAUNCH.json",record)
print(json.dumps(record,indent=2))
