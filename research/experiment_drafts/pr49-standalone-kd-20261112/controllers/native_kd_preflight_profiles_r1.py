"""Admit exact canonical cache/config consumers before four new KD profiles."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

root = Path("/data/lc-epanov_s/tabdata-final-quality-20261112-r1")
source = root / "kd-intake-r2/source-kd-r1"
packet = root / "kd-intake-r2/pr49-standalone-kd-r2"
python = "/tmp/tabdata-lc-epanov_s-architecture-20261107-env/bin/python"
preflight = """
import hashlib,json
from pathlib import Path
from wnn.training.sequence_config import load_config
from wnn.training.sequence_data import build_data
from wnn.training.sequence_source_capsule import verify_source_capsule
from wnn.training.sequence_provenance import source_identity
from wnn.training.sequence_ensemble_config import requires_teacher_cache
r=Path('/data/lc-epanov_s/tabdata-final-quality-20261112-r1')
s=r/'kd-intake-r2/source-kd-r1';p=r/'kd-intake-r2/pr49-standalone-kd-r2'
d=Path('/data/lc-epanov_s/tabdata-stem-successors-20261111-r1/project')
t=Path('/data2/lc-epanov_s/tabdata-final-quality-20261112-r1/teacher-cache')
panel=json.loads((p/'PANEL_MANIFEST_R2.json').read_bytes())
verify_source_capsule(s,s/'SOURCE_MANIFEST.json',panel['source_manifest_sha256'],panel['source_commit'])
assert source_identity(s,s/'scripts/train_sequence.py')['sha256']==panel['source_identity_sha256']
assert hashlib.sha256((t/'manifest.json').read_bytes()).hexdigest()=='138ef43c6ef03755a3b6195ca83f2c346f7dca4e3fe365f9e8e151ad2c71792d'
configs={}
for e in panel['cells']:
 path=p/e['config'];assert hashlib.sha256(path.read_bytes()).hexdigest()==e['config_sha256']
 c=load_config(path);assert c['seed']==e['seed'] and c['kd_weight']==e['kd_weight']
 assert c['since_weight']==0 and requires_teacher_cache(c,False)
 assert c['model']['hidden']==256 and c['model']['layers']==2 and c['model']['input_stem_kernel']==16
 configs[e['cell']]=c
for seed in (203501,203502):
 pair=[c for c in configs.values() if c['seed']==seed];assert len(pair)==2
 assert [k for k in pair[0] if pair[0][k]!=pair[1][k]]==['kd_weight']
data=build_data(next(iter(configs.values())),d,d/'chronological_primary.json',t)
assert len(data.fit.members)==10325 and len(data.evaluation.members)==1873
assert all(member[0]=='train' for member in data.fit.members)
teacher=data.provenance['teacher']
assert teacher['checkpoint_sha256']=='759165147d99b33034b74f4af63a30e8a202dafc78fd5e5699d05b6cfe9bef90'
cache=data.fit.cache
for member in (data.fit.members[0],data.fit.members[5162],data.fit.members[-1]):
 target,hints=cache.read(*member,0);assert tuple(target.shape)==(20000,2)
print(json.dumps({'status':'passed','cells':list(configs),'source_sha256':panel['source_identity_sha256'],
 'source_commit':panel['source_commit'],'cache_manifest_sha256':teacher['manifest_sha256'],
 'actual_teacher_sha256':teacher['checkpoint_sha256'],'fitting_members':10325,'valid_members':1873,
 'paired_treatment_changed_fields':['kd_weight'],'cache_required_even_at_lambda_zero':True}))
"""
compile(preflight, "<actual-KD-cache-consumer>", "exec")
env = dict(
    os.environ,
    PYTHONPATH=str(source / "src"),
    OMP_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    PYTHONDONTWRITEBYTECODE="1",
)
checked = subprocess.run(
    [python, "-c", preflight],
    cwd=source,
    env=env,
    capture_output=True,
    text=True,
    timeout=180,
    check=False,
)
receipt = {
    "returncode": checked.returncode,
    "stdout": checked.stdout,
    "stderr": checked.stderr,
    "at_epoch": time.time(),
}
with (root / "control/NEW_KD_CANONICAL_PREFLIGHT_R1.json").open("x") as handle:
    json.dump(receipt, handle, indent=2)
if checked.returncode:
    print(json.dumps(receipt))
else:
    command = [python, str(packet / "l4_standalone_kd_wave_r1.py"), "profile-newkd"]
    with (root / "control/newkd-profiles-r1.log").open("xb") as log:
        process = subprocess.Popen(
            command,
            cwd=source,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    owner = {"pid": process.pid, "command": command, "at_epoch": time.time()}
    with (root / "control/NEW_KD_PROFILE_LAUNCH_R1.json").open("x") as handle:
        json.dump(owner, handle, indent=2)
    print(json.dumps({"preflight": receipt, "profile_owner": owner}))
