"""Admit the single fixed seed-3 teacher using already-reviewed profile gates."""

from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import zlib
from pathlib import Path

C = Path(__file__).resolve().parents[1]
P = C / "preparation"
R = C / "records/teacher_s3_v1"
PROFILE = C / "records/teacher_profile_v3"


def main() -> None:
    """Require accepted profile bytes before creating one actual fit admission."""
    acceptance_raw = (PROFILE / "ACCEPTANCE.json").read_bytes()
    acceptance = json.loads(acceptance_raw)
    assert acceptance["status"] == "PASS"
    assert acceptance["successful_updates"] == 120
    assert acceptance["source_commit"] == "af21d6d8409c46b36830f0bdddd50e67ee17025f"
    recipe_raw = (P / "TEACHER_S3_FIT_BINDING.json").read_bytes()
    assert (
        hashlib.sha256(recipe_raw).hexdigest()
        == "6a50960bd650111c1da64a4f89694d7d992c88473f139f295c7a54f4fa5547f3"
    )
    recipe = json.loads(recipe_raw)
    assert recipe["run"]["name"] == "fg19_t3_h384x2_e3_s3_v1"
    assert recipe["resource_envelope"]["pytorch_allocator_limit_bytes"] == 4294967296
    reused = (C / "records/teacher_profile_v2/REMOTE_SUBMISSION.py").read_bytes()
    assert hashlib.sha256(reused).hexdigest() == acceptance["reused_guard_sha256"]
    launch_raw = (PROFILE / "LAUNCH.json").read_bytes()
    assert hashlib.sha256(launch_raw).hexdigest() == acceptance["launch_sha256"]
    launch = json.loads(launch_raw)
    payload = {
        "recipe": recipe,
        "recipe_sha256": hashlib.sha256(recipe_raw).hexdigest(),
        "acceptance": acceptance,
        "acceptance_sha256": hashlib.sha256(acceptance_raw).hexdigest(),
        "profile_launch": launch,
        "reused_source_zlib_b64": base64.b64encode(zlib.compress(reused)).decode(),
        "reused_source_sha256": hashlib.sha256(reused).hexdigest(),
    }
    body = r"""import base64,zlib,hashlib,json,subprocess,datetime,os,stat
from pathlib import Path
P=PAYLOAD
raw=zlib.decompress(base64.b64decode(P['reused_source_zlib_b64'],validate=True));assert hashlib.sha256(raw).hexdigest()==P['reused_source_sha256']
G={'__name__':'fg19_readonly_profile_gates'}
exec(compile(raw,'accepted_profile_admission_reuse','exec'),G)
G['GPU']=P['recipe']['runtime_environment']['cuda_visible_device'];G['MIN_GPU_FREE']=4294967296+1000000000
B=G['B'];SOURCE=G['SOURCE'];OUT=B/'research_tmp/fg19_teacher_s3_v1';U='computer-vision-homework-fg19-teacher-s3-v1.service';RUN=B/'data/artifacts/runs/fg19_t3_h384x2_e3_s3_v1';A=P['acceptance'];recipe=P['recipe']
assert A['status']=='PASS' and A['successful_updates']==120 and A['source_commit']==P['recipe']['source_binding']['git_commit']
log=B/'research_tmp/fg19_teacher_profile_v3/unit.log';assert G['_sha'](log)==A['log_sha256']
assert not os.path.lexists(OUT) and not os.path.lexists(RUN) and not (B/'runtime/STOP_ADMISSION.json').exists()
assert G['_unit'](U,['LoadState'])['LoadState']=='not-found'
expected={v['path']:v['sha256'] for v in G['PAYLOAD']['source_manifest']['files']}
actual={str(p.relative_to(SOURCE)) for p in SOURCE.rglob('*') if p.is_file()};assert actual==set(expected)
for n,d in expected.items():
 q=SOURCE/n;assert not q.is_symlink() and G['_sha'](q)==d
prop=G['_check_terminal_propensity']();data=G['_data_identity']();parent,pop,cgroup=G['_parent']();gpu,foreign=G['_gpu']()
assert not pop and int(parent['MemoryCurrent'])+8000000000<17000000000
assert os.statvfs(B).f_bavail*os.statvfs(B).f_frsize>20000000000
assert G['_run'](['findmnt','-T',str(B/'research_tmp'),'-n','-o','FSTYPE']).strip() not in ['tmpfs','ramfs']
assert recipe['fit_scope']['split_manifest_sha256']==G['PAYLOAD']['split_sha256'] and recipe['fit_scope']['train_source']=='train+valid' and recipe['fit_scope']['include_heldout_valid'] is False
split=Path(recipe['fit_scope']['split_manifest_path']);before=split.stat();assert not split.is_symlink() and stat.S_ISREG(before.st_mode) and before.st_size<2000000
splitraw=split.read_bytes();after=split.stat();assert (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
assert str(split.resolve())==str(B/'splits/fold0.json') and hashlib.sha256(splitraw).hexdigest()==recipe['fit_scope']['split_manifest_sha256']=='c09f7240396a33ca82977115af8d6067e571c0ec25c58a3c99b438ca2d2294b7'
split_identity={'sha256':hashlib.sha256(splitraw).hexdigest(),'bytes':len(splitraw),'device':after.st_dev,'inode':after.st_ino,'mtime_ns':after.st_mtime_ns}
args=recipe['argv'];assert '--save-training-state' in args and '--no-eval-during-fit' in args and '--profile-steps' not in args and '--max-batches' not in args
assert args[args.index('--name')+1]=='fg19_t3_h384x2_e3_s3_v1' and args[args.index('--seed')+1]=='3'
assert args[args.index('--precision')+1]=='fp32' and args[args.index('--muon-ortho-dtype')+1]=='fp32' and '--strict-muon-fp32' in args
OUT.mkdir()
G['_write'](OUT/'RECIPE.json',recipe)
admission={'utc':datetime.datetime.now(datetime.UTC).isoformat(),'unit':U,'recipe_sha256':P['recipe_sha256'],'source_commit':recipe['source_binding']['git_commit'],'source_files':expected,'split_identity':split_identity,'profile_acceptance_sha256':P['acceptance_sha256'],'profile':A,'reused_guard_sha256':P['reused_source_sha256'],'propensity':prop,'data':data,'parent':parent,'populated':pop,'gpu':gpu,'foreign_preserved':foreign,'events':(cgroup/'memory.events').read_text(),'events_local':(cgroup/'memory.events.local').read_text()}
G['_write'](OUT/'ADMISSION.json',admission)
old=P['profile_launch']['command'];i=old.index('/home/grinchenkoeg/WunderFunder/.venv/bin/python');prefix=old[:i]
prefix=[('--unit='+U if a.startswith('--unit=') else '--property=RuntimeMaxSec=86400' if a=='--property=RuntimeMaxSec=1200' else '--property=StandardOutput=append:'+str(OUT/'unit.log') if a.startswith('--property=StandardOutput=') else '--property=StandardError=append:'+str(OUT/'unit.log') if a.startswith('--property=StandardError=') else a) for a in prefix]
assert '--setenv=WNN_GPU_MEMORY_BYTES=4294967296' in prefix
assert '--setenv=CUDA_VISIBLE_DEVICES='+recipe['runtime_environment']['cuda_visible_device'] in prefix
for required in ['--property=CPUQuota=100%','--property=MemoryHigh=7500000000','--property=MemoryMax=8000000000','--property=MemorySwapMax=0','--property=RuntimeMaxSec=86400','--property=Slice=wnn-campaign.slice']:
 assert prefix.count(required)==1
assert sum(v.startswith('--setenv=CUDA_VISIBLE_DEVICES=') for v in prefix)==1
for target in [OUT,RUN]:
 for ancestor in target.parents:
  if ancestor==B:break
  assert not ancestor.is_symlink()
cmd=prefix+['/home/grinchenkoeg/WunderFunder/.venv/bin/python','-u',str(SOURCE/'scripts/train_rnn.py'),*args[2:]]
r=subprocess.run(cmd,capture_output=True,text=True,timeout=30)
launch={'utc':datetime.datetime.now(datetime.UTC).isoformat(),'unit':U,'command':cmd,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'post':G['_unit'](U,['InvocationID','ActiveState','SubState','MainPID','ControlGroup','ExecStart']),'unknown_outcome_policy':'never blindly retry; inspect exact unit and existing run/log receipts'}
G['_write'](OUT/'LAUNCH.json',launch)
print(json.dumps({'admission':admission,'launch':launch}))
""".replace("P=PAYLOAD", "P=" + repr(payload))
    compile(body, "teacher_s3_remote", "exec")
    R.mkdir(exist_ok=False)
    (R / "REMOTE_SUBMISSION.py").write_text(body)
    (R / "RECIPE.json").write_bytes(recipe_raw)
    try:
        result = subprocess.run(
            [
                "ssh",
                "-T",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=10",
                "-o",
                "ConnectionAttempts=1",
                "beleriand",
                "python3 -B -",
            ],
            input=body.encode(),
            capture_output=True,
            timeout=90,
        )
    except subprocess.TimeoutExpired as error:
        (R / "STDOUT.raw.json").write_bytes(error.stdout or b"")
        (R / "STDERR.raw").write_bytes(error.stderr or b"")
        (R / "TRANSPORT.json").write_text(
            json.dumps(
                {
                    "outcome": "unknown",
                    "timed_out": True,
                    "policy": "Inspect exact unit and durable paths; NEVER retry blindly",
                }
            )
            + "\n"
        )
        raise
    (R / "TRANSPORT.json").write_text(
        json.dumps({"returncode": result.returncode, "outcome": "known_response_received"}) + "\n"
    )
    (R / "STDOUT.raw.json").write_bytes(result.stdout)
    (R / "STDERR.raw").write_bytes(result.stderr)
    assert result.returncode == 0, (result.returncode, result.stderr)
    value = json.loads(result.stdout)
    for key, item in value.items():
        (R / (key.upper() + ".json")).write_text(json.dumps(item, indent=2) + "\n")
    print(
        json.dumps(
            {
                "unit": value["launch"]["unit"],
                "returncode": value["launch"]["returncode"],
                "post": value["launch"]["post"],
            }
        )
    )


if __name__ == "__main__":
    main()
