"""Admit the single fixed seed-5 teacher using already-reviewed profile gates."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import subprocess
import zlib
from pathlib import Path

C = Path(__file__).resolve().parents[1]
P = C / "preparation"
R = C / "records/teacher_s5_v1"
PROFILE = C / "records/teacher_profile_v3"


def _read_s4_acceptance(path: Path, expected_sha256: str) -> tuple[bytes, dict[str, object], str]:
    """Require the exact root-accepted terminal receipt for the completed seed-4 fit."""
    if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise ValueError("the S4 terminal acceptance SHA-256 must be 64 lowercase hex characters")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256:
        raise ValueError("the S4 terminal acceptance does not match the caller-provided digest")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("the S4 terminal acceptance must be a JSON object")
    expected = {
        "schema": "fg19.teacher-s4-terminal-root-acceptance.v1",
        "status": "ACCEPTED",
        "run_name": "fg19_t3_h384x2_e3_s4_v1",
        "source_commit": "af21d6d8409c46b36830f0bdddd50e67ee17025f",
        "recipe_sha256": "c460b098d09773786552e5b2c98596fab3319a05108d49bd5693da43ee8a7d88",
        "successful_updates": 22380,
        "skipped_updates": 0,
        "nonfinite_updates": 0,
    }
    for key, item in expected.items():
        if value.get(key) != item:
            raise ValueError(f"S4 terminal acceptance {key} differs from the fixed predecessor")
    native = value.get("native_unit")
    if not isinstance(native, dict):
        raise ValueError("S4 terminal acceptance native_unit is missing")
    native_expected = {
        "unit": "computer-vision-homework-fg19-teacher-s4-v1.service",
        "InvocationID": "b6d71599e0044970b8641e3e241c40a4",
        "ActiveState": "active",
        "SubState": "exited",
    }
    for key, item in native_expected.items():
        if native.get(key) != item:
            raise ValueError(f"S4 terminal native unit {key} differs from the completed fit")
    if native.get("ExecMainCode") not in (1, "1") or native.get("ExecMainStatus") not in (0, "0"):
        raise ValueError("S4 native unit did not exit successfully")
    verifier = value.get("verifier_unit")
    if not isinstance(verifier, dict):
        raise ValueError("S4 root acceptance does not bind the successful terminal verifier unit")
    verifier_unit = verifier.get("unit")
    verifier_invocation = verifier.get("InvocationID")
    if (
        not isinstance(verifier_unit, str)
        or not verifier_unit.startswith("computer-vision-homework-fg19-teacher-")
        or not verifier_unit.endswith(".service")
        or verifier_unit == native.get("unit")
        or not isinstance(verifier_invocation, str)
        or re.fullmatch(r"[0-9a-f]{32}", verifier_invocation) is None
        or verifier.get("ActiveState") != "active"
        or verifier.get("SubState") != "exited"
        or verifier.get("ExecMainCode") not in (1, "1")
        or verifier.get("ExecMainStatus") not in (0, "0")
    ):
        raise ValueError("S4 root acceptance terminal verifier identity/status is invalid")
    verification_sha = value.get("verification_sha256")
    if (
        not isinstance(verification_sha, str)
        or re.fullmatch(r"[0-9a-f]{64}", verification_sha) is None
    ):
        raise ValueError("S4 acceptance lacks the actual B-side verification digest")
    model_sha = value.get("model_sha256")
    checkpoint_sha = value.get("checkpoint_sha256")
    files = value.get("files")
    fingerprint = value.get("model_tensor_fingerprint")
    if not isinstance(model_sha, str) or re.fullmatch(r"[0-9a-f]{64}", model_sha) is None:
        raise ValueError("S4 selected model file digest is missing")
    if not isinstance(checkpoint_sha, str) or re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha) is None:
        raise ValueError("S4 epoch-2 checkpoint file digest is missing")
    if not isinstance(fingerprint, str) or re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None:
        raise ValueError("S4 common model tensor fingerprint is missing")
    if (
        not isinstance(files, dict)
        or not isinstance(files.get("model.pt"), dict)
        or not isinstance(files.get("model_ep2.pt"), dict)
        or files["model.pt"].get("sha256") != model_sha
        or files["model_ep2.pt"].get("sha256") != checkpoint_sha
    ):
        raise ValueError("S4 file hashes do not match the successful verifier's files map")
    return raw, value, digest


def _read_s4_stop(
    path: Path, expected_sha256: str, acceptance: dict[str, object]
) -> tuple[bytes, dict[str, object], str]:
    """Require the hash-bound exact stop capture for the accepted S4 invocation."""
    if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise ValueError("the S4 exact-stop SHA-256 must be 64 lowercase hex characters")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256:
        raise ValueError("the S4 exact-stop capture does not match the caller-provided digest")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("the S4 exact-stop capture must be a JSON object")
    before_all = value.get("before")
    stop_all = value.get("stop")
    after_all = value.get("after")
    if not all(isinstance(item, dict) for item in (before_all, stop_all, after_all)):
        raise ValueError("S4 stop capture is missing before/stop/after unit records")
    native = acceptance["native_unit"]
    verifier = acceptance["verifier_unit"]
    expected_units = {
        native["unit"]: native["InvocationID"],
        verifier["unit"]: verifier["InvocationID"],
    }
    if any(set(records) != set(expected_units) for records in (before_all, stop_all, after_all)):
        raise ValueError("S4 exact-stop receipt must contain both accepted owned units only")
    for unit, invocation in expected_units.items():
        before = before_all[unit]
        stopped = stop_all[unit]
        after = after_all[unit]
        if (
            before.get("InvocationID") != invocation
            or before.get("ActiveState") != "active"
            or before.get("SubState") != "exited"
            or before.get("ExecMainCode") not in (1, "1")
            or before.get("ExecMainStatus") not in (0, "0")
            or stopped.get("returncode") != 0
            or after.get("LoadState") != "not-found"
            or after.get("ActiveState") != "inactive"
            or after.get("SubState") != "dead"
            or after.get("InvocationID") != ""
        ):
            raise ValueError(f"S4 exact stop capture does not bind successful owned unit {unit}")
    if value.get("populated") != {}:
        raise ValueError("S4 exact stop capture leaves a populated campaign cgroup")
    return raw, value, digest


def main() -> None:
    """Require accepted S4 terminal bytes and profile before creating one S5 fit admission."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s4-terminal-acceptance-sha256", required=True)
    parser.add_argument("--s4-stop-sha256", required=True)
    cli = parser.parse_args()
    predecessor_path = C / "records/teacher_s4_v1/TERMINAL_ACCEPTANCE.json"
    predecessor_raw, predecessor, predecessor_sha = _read_s4_acceptance(
        predecessor_path, cli.s4_terminal_acceptance_sha256
    )
    stop_path = C / "records/teacher_s4_v1/terminal_exact_stop/STDOUT.raw.json"
    stop_raw, stop_capture, stop_sha = _read_s4_stop(stop_path, cli.s4_stop_sha256, predecessor)
    acceptance_raw = (PROFILE / "ACCEPTANCE.json").read_bytes()
    acceptance = json.loads(acceptance_raw)
    assert acceptance["status"] == "PASS"
    assert acceptance["successful_updates"] == 120
    assert acceptance["source_commit"] == "af21d6d8409c46b36830f0bdddd50e67ee17025f"
    recipe_raw = (P / "TEACHER_S5_FIT_BINDING.json").read_bytes()
    assert (
        hashlib.sha256(recipe_raw).hexdigest()
        == "6d9d91b9d19b4d2cd6c957433e534f74880f1bf8414a06e3434adf8eb56d734a"
    )
    recipe = json.loads(recipe_raw)
    assert recipe["run"]["name"] == "fg19_t3_h384x2_e3_s5_v1"
    assert recipe["resource_envelope"]["pytorch_allocator_limit_bytes"] == 4294967296
    reused = (C / "records/teacher_profile_v2/REMOTE_SUBMISSION.py").read_bytes()
    assert hashlib.sha256(reused).hexdigest() == acceptance["reused_guard_sha256"]
    launch_raw = (PROFILE / "LAUNCH.json").read_bytes()
    assert hashlib.sha256(launch_raw).hexdigest() == acceptance["launch_sha256"]
    launch = json.loads(launch_raw)
    payload = {
        "recipe": recipe,
        "predecessor_acceptance": predecessor,
        "predecessor_acceptance_raw_b64": base64.b64encode(predecessor_raw).decode(),
        "predecessor_acceptance_sha256": predecessor_sha,
        "predecessor_stop": stop_capture,
        "predecessor_stop_raw_b64": base64.b64encode(stop_raw).decode(),
        "predecessor_stop_sha256": stop_sha,
        "predecessor_recipe_sha256": "c460b098d09773786552e5b2c98596fab3319a05108d49bd5693da43ee8a7d88",
        "recipe_sha256": hashlib.sha256(recipe_raw).hexdigest(),
        "acceptance": acceptance,
        "acceptance_sha256": hashlib.sha256(acceptance_raw).hexdigest(),
        "profile_launch": launch,
        "reused_source_zlib_b64": base64.b64encode(zlib.compress(reused)).decode(),
        "reused_source_sha256": hashlib.sha256(reused).hexdigest(),
    }
    body = r"""import base64,zlib,hashlib,json,subprocess,datetime,os,stat,re
from pathlib import Path
P=PAYLOAD
raw=zlib.decompress(base64.b64decode(P['reused_source_zlib_b64'],validate=True));assert hashlib.sha256(raw).hexdigest()==P['reused_source_sha256']
G={'__name__':'fg19_readonly_profile_gates'}
exec(compile(raw,'accepted_profile_admission_reuse','exec'),G)
G['GPU']=P['recipe']['runtime_environment']['cuda_visible_device'];G['MIN_GPU_FREE']=4294967296+1000000000
B=G['B'];DATA=B/'data';SOURCE=G['SOURCE'];OUT=B/'research_tmp/fg19_teacher_s5_v1';U='computer-vision-homework-fg19-teacher-s5-v1.service';RUN=B/'data/artifacts/runs/fg19_t3_h384x2_e3_s5_v1';A=P['acceptance'];recipe=P['recipe'];parent_raw=base64.b64decode(P['predecessor_acceptance_raw_b64'],validate=True);assert hashlib.sha256(parent_raw).hexdigest()==P['predecessor_acceptance_sha256'];parent_acceptance=json.loads(parent_raw);assert parent_acceptance==P['predecessor_acceptance'];assert parent_acceptance['status']=='ACCEPTED' and parent_acceptance['schema']=='fg19.teacher-s4-terminal-root-acceptance.v1';assert parent_acceptance['native_unit']['InvocationID']=='b6d71599e0044970b8641e3e241c40a4';assert parent_acceptance['recipe_sha256']==P['predecessor_recipe_sha256']
model_sha=parent_acceptance['model_sha256'];checkpoint_sha=parent_acceptance['checkpoint_sha256'];files=parent_acceptance['files'];assert files['model.pt']['sha256']==model_sha and files['model_ep2.pt']['sha256']==checkpoint_sha;assert re.fullmatch(r'[0-9a-f]{64}',parent_acceptance['model_tensor_fingerprint'])
stop_raw=base64.b64decode(P['predecessor_stop_raw_b64'],validate=True);assert hashlib.sha256(stop_raw).hexdigest()==P['predecessor_stop_sha256'];stop=json.loads(stop_raw);S4='computer-vision-homework-fg19-teacher-s4-v1.service';VFY=parent_acceptance['verifier_unit']['unit'];assert VFY.startswith('computer-vision-homework-fg19-teacher-') and VFY.endswith('.service') and VFY!=S4;assert parent_acceptance['verifier_unit']['ActiveState']=='active' and parent_acceptance['verifier_unit']['SubState']=='exited' and parent_acceptance['verifier_unit']['ExecMainCode'] in (1,'1') and parent_acceptance['verifier_unit']['ExecMainStatus'] in (0,'0');assert stop==P['predecessor_stop'];expected_units={S4:'b6d71599e0044970b8641e3e241c40a4',VFY:parent_acceptance['verifier_unit']['InvocationID']};assert len(expected_units)==2;assert all(set(stop[k])==set(expected_units) for k in ('before','stop','after'));assert stop['populated']=={}
for unit,inv in expected_units.items():
 assert stop['before'][unit]['InvocationID']==inv and stop['before'][unit]['ExecMainStatus'] in (0,'0');assert stop['stop'][unit]['returncode']==0;assert stop['after'][unit]['LoadState']=='not-found' and stop['after'][unit]['ActiveState']=='inactive' and stop['after'][unit]['SubState']=='dead' and stop['after'][unit]['InvocationID']==''
assert A['status']=='PASS' and A['successful_updates']==120 and A['source_commit']==P['recipe']['source_binding']['git_commit']
log=B/'research_tmp/fg19_teacher_profile_v3/unit.log';assert G['_sha'](log)==A['log_sha256']
assert not os.path.lexists(OUT) and not os.path.lexists(RUN) and not (B/'runtime/STOP_ADMISSION.json').exists()
assert G['_unit'](U,['LoadState'])['LoadState']=='not-found'
expected={v['path']:v['sha256'] for v in G['PAYLOAD']['source_manifest']['files']}
actual={str(p.relative_to(SOURCE)) for p in SOURCE.rglob('*') if p.is_file()};assert actual==set(expected)
for n,d in expected.items():
 q=SOURCE/n;assert not q.is_symlink() and G['_sha'](q)==d
prop_model=DATA/'artifacts/runs/fg19_propensity_clean_f0_seed0_e4/model.pt';ps=prop_model.stat();prop_sha='6facf4ca9045118b8e18ca590053c97e2f5f5122c92f228f75002f0c76a2800e';assert not prop_model.is_symlink() and prop_model.is_file() and ps.st_size==174545 and G['_sha'](prop_model)==prop_sha;prop={'run_name':'fg19_propensity_clean_f0_seed0_e4','model_sha256':prop_sha,'bytes':ps.st_size,'verification_sha256':'73b1155011bee53ecba84da8005d3ef2f70b1a5117bb0ae7a01cd145af6cc139','reused_from_s4_admission':True};data=G['_data_identity']();parent,pop,cgroup=G['_parent']();gpu,foreign=G['_gpu']();assert G['_unit'](S4,['LoadState'])['LoadState']=='not-found' and G['_unit'](VFY,['LoadState'])['LoadState']=='not-found'
assert not pop and int(parent['MemoryCurrent'])+8000000000<17000000000
assert os.statvfs(B).f_bavail*os.statvfs(B).f_frsize>20000000000
assert G['_run'](['findmnt','-T',str(B/'research_tmp'),'-n','-o','FSTYPE']).strip() not in ['tmpfs','ramfs']
assert recipe['fit_scope']['split_manifest_sha256']==G['PAYLOAD']['split_sha256'] and recipe['fit_scope']['train_source']=='train+valid' and recipe['fit_scope']['include_heldout_valid'] is False
split=Path(recipe['fit_scope']['split_manifest_path']);before=split.stat();assert not split.is_symlink() and stat.S_ISREG(before.st_mode) and before.st_size<2000000
splitraw=split.read_bytes();after=split.stat();assert (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
assert str(split.resolve())==str(B/'splits/fold0.json') and hashlib.sha256(splitraw).hexdigest()==recipe['fit_scope']['split_manifest_sha256']=='c09f7240396a33ca82977115af8d6067e571c0ec25c58a3c99b438ca2d2294b7'
split_identity={'sha256':hashlib.sha256(splitraw).hexdigest(),'bytes':len(splitraw),'device':after.st_dev,'inode':after.st_ino,'mtime_ns':after.st_mtime_ns}
args=recipe['argv'];assert '--save-training-state' in args and '--no-eval-during-fit' in args and '--profile-steps' not in args and '--max-batches' not in args
assert args[args.index('--name')+1]=='fg19_t3_h384x2_e3_s5_v1' and args[args.index('--seed')+1]=='5'
assert args[args.index('--precision')+1]=='fp32' and args[args.index('--muon-ortho-dtype')+1]=='fp32' and '--strict-muon-fp32' in args
OUT.mkdir()
G['_write'](OUT/'RECIPE.json',recipe)
admission={'utc':datetime.datetime.now(datetime.UTC).isoformat(),'unit':U,'recipe_sha256':P['recipe_sha256'],'predecessor_terminal_acceptance_sha256':P['predecessor_acceptance_sha256'],'predecessor_terminal_acceptance':parent_acceptance,'predecessor_terminal_stop_sha256':P['predecessor_stop_sha256'],'predecessor_terminal_stop':stop,'source_commit':recipe['source_binding']['git_commit'],'source_files':expected,'split_identity':split_identity,'profile_acceptance_sha256':P['acceptance_sha256'],'profile':A,'reused_guard_sha256':P['reused_source_sha256'],'propensity':prop,'data':data,'parent':parent,'populated':pop,'gpu':gpu,'foreign_preserved':foreign,'events':(cgroup/'memory.events').read_text(),'events_local':(cgroup/'memory.events.local').read_text()}
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
    compile(body, "teacher_s5_remote", "exec")
    R.mkdir(exist_ok=False)
    (R / "REMOTE_SUBMISSION.py").write_text(body)
    (R / "RECIPE.json").write_bytes(recipe_raw)
    (R / "PREDECESSOR_TERMINAL_ACCEPTANCE.json").write_bytes(predecessor_raw)
    (R / "PREDECESSOR_TERMINAL_STOP.json").write_bytes(stop_raw)
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
    (R / "STDOUT.raw.json").write_bytes(result.stdout)
    (R / "STDERR.raw").write_bytes(result.stderr)
    (R / "TRANSPORT.json").write_text(
        json.dumps({"returncode": result.returncode, "outcome": "known_response_received"}) + "\n"
    )
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
