B=pathlib.Path("/home/grinchenkoeg/computer-vision-homework-20261018")
DEST=B/"support-source/eda19-r6-v3"
OUT=B/"research_tmp/nf19-r6-v3"
UNIT="computer-vision-homework-eda19-r6-v3.service"
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
canonical={'__init__.py': '2dca10ee7e2e8c1139bed2165b322d2077fdbeac8095428e8c796efe47ea66c6', 'block_memory_attention.py': 'a05c04d49f258d808ddb4fc4f7373a7bef57963050619f96698822e61840cc39', 'data.py': 'a93da840137adbd7afdb79a22687fb73150e77107f91ca2adc5211b854bc7e00', 'event_content.py': '75beff387c4f9312dc3bc7e065d76e7df2ceeb666fe1bb3e93cc245dfca445cd', 'features.py': 'c2b317d1f148d0cad9632726a1436b94f4ae76016458deaa70aa36fdca21ecb2', 'nn.py': '7fca0e21fc0da2a40e2a39bed069a871b404adf6a0e41a49dcf24ddcab878d48', 'numeric_ple.py': '229a7e0f032cd65bbf6700fa9372c08016b20e4719cb2f9eddd540ff8c468720', 'optim.py': '549831f55be352c46d740a46e73386d41db8a8c2247b4a6d3f646177bd3cdd97', 'privileged.py': '89dded31558dde890f671f7a4394c429346f77bdfafa1a5b078876e0c94b76c4', 'propensity.py': 'c92f7bf067c54615439b67cc38c343b9d18bc0ba84e19a7343c61ba424ad414d', 'quantization.py': 'fbca8fff8af5ae488eea2c2b53e304fd3620fdaf32f1f213ac499378b0261298', 'research_features.py': '5dfc31d01c6a5934f66fd6029f457ae2a85a22d6f169d32b89e2160e38c84b50', 'sentinel.py': '114bf5afa346ee2af6b9dc5de642de5467b7d5893629ffcd91519d72bea6a9ec', 'specialist.py': '2937ca9a32561c150a19401eec30dab5e0dd81f507f1fc8365d7f4fb90b0a400', 'splits.py': '24024fbb82d33afdc2c709912404f7c1c26504a0816a050428c9f66e04e951fe', 'structured_state.py': '227a5b8fe8ada7b34d61463ad2f022a0d802011aa985b215ef880d5a469c1756', 'training/__init__.py': '686df59460b49e15730ed1c2c7d4496d8a516c1c505ed84752a56d12262591e7', 'training/environment.py': 'f1e150f7473c8cd317ef489d7c3e9e796f22b6a9715ed2f02ae94505f5b8e18b', 'training/forward.py': '7e7e23051bea314a3547bc26f8232ebe56616f79c1872cafc480667e873794b9', 'training/inputs.py': '40d046cc7a3377ceacfea5409b19a3d69054b607fc0ea723e64e70e5968da2eb', 'training/model_validation.py': '2326c3e39793753d077be580e3953dc1b2e2a09bf455f63d7576497868d557ba', 'training/models.py': '95d5bc6cce784448203717c38070c00f2e677ba6d5bc340737d84b03605dd473', 'training_dynamics.py': '041e3afdf11b064ba35b6ee021116789c6222a4c73a992ea1398ad69f65c0463', 'training_factorized.py': 'b1e7edb7a780e6c2539b298bc30294c03882e04c9dcb27eae323b0576fa6f2b2', 'training_qat.py': '65d770f1ee5bc4cb5b88d7e1b42d63eb77fae4704010f0b263966e0de58578e2'}
for name,digest in canonical.items():assert sha(B/"source-a266b036/src/wnn"/name)==digest
DEST.mkdir()
for name,entry in CFG["files"].items():
 rel=pathlib.PurePosixPath(name)
 assert not rel.is_absolute() and ".." not in rel.parts
 assert len(rel.parts)==2 or (len(rel.parts)==3 and rel.parts[:2]==("idea_screens_r6","records"))
 target=DEST/name
 target.parent.mkdir(parents=True,exist_ok=True)
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
inline=__INLINE_REPR__
command=["systemd-run","--user","--unit="+UNIT,"--slice=wnn-campaign.slice","--property=Type=exec","--property=CPUQuota=50%","--property=MemoryHigh=1400000000","--property=MemoryMax=1500000000","--property=MemorySwapMax=0","--property=RuntimeMaxSec=600","--property=RemainAfterExit=yes","--property=StandardOutput=append:"+str(DEST/"probe.log"),"--property=StandardError=append:"+str(DEST/"probe.log"),"--working-directory="+str(DEST),"/usr/bin/env","CUDA_VISIBLE_DEVICES=","OMP_NUM_THREADS=1","OPENBLAS_NUM_THREADS=1","MKL_NUM_THREADS=1","NUMEXPR_NUM_THREADS=1","PYTHONPATH="+str(B/"source-a266b036/src")+":"+str(DEST),"TMPDIR="+str(scratch),PY,"-B","-c",inline]
record=dict(schema="nf19.r6_probe_launch.v3",utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),unit=UNIT,sequence=latest,source_sha256={n:v["sha256"] for n,v in CFG["files"].items()},canonical_source_sha256=canonical,sample_sha256=sha(sample),checkpoint_sha256=sha(checkpoint),admission=dict(parent=parent,units=units,populated=populated,memory_events=events,memory_events_local=events_local,gpu_processes=gpu_rows,scratch_fs=fs.stdout.strip(),scratch_free=shutil.disk_usage(scratch).free,host_mem_available=int(meminfo["MemAvailable"])*1024),command=command,output=str(OUT))
write(DEST/"ROOT_ADMISSION.json",record)
p=subprocess.run(command,capture_output=True,text=True,timeout=20)
record.update(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
write(DEST/"ROOT_LAUNCH.json",record)
print(json.dumps(record,indent=2))
