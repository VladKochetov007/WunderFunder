"""Prepare, but do not transmit, the bounded teacher terminal verifier V2."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

C = Path(__file__).resolve().parents[1]
P = C / "preparation"
RECORDS = C / "records/teacher_s3_v1"
OUT = P / "teacher_terminal_verifier_v2"
REMOTE_OUTPUT = (
    "/home/grinchenkoeg/computer-vision-homework-20261018/research_tmp/fg19_teacher_s3_verify_v2"
)
VERIFIER_UNIT = "computer-vision-homework-fg19-teacher-s3-verify-v2.service"
CHECKER_NAME = "teacher_terminal_check_v2.py"
EXPECTED = {
    "teacher_terminal_check_v2.py": "d3abd965ec31e27c24629304fc28016fe4b51fe0cc03af6db60d105aca1d6202",
    "propensity_terminal_support.py": "958d973b17a2180c954ca7e0953ce20d7b5a3120124cd3cf964a97606604f409",
    "SOURCE_MANIFEST.json": "0c83367d4ee3afd7e9f382111060d8c125b52420ce27cd5cc309d338287af7fd",
    "ADMISSION.json": "dd3cbb8416c6cf14c29fb5937e664b41b71496420e44200c42248fa4d3109c17",
    "LAUNCH.json": "6538d9502bdcd8cd3bdac0b50c4dc46b48ca80ca15d45ea25365c5313ecf8144",
    "LAUNCH_ACCEPTANCE.json": "67479aef04bcca80d4e9d8bfbe0de71b180439a4d48fd135bffc9ccd2b94a1c7",
    "FIT_BINDING.json": "6a50960bd650111c1da64a4f89694d7d992c88473f139f295c7a54f4fa5547f3",
    "CHECKER_PREPARATION_V2.json": "d2b1c92b59841b66e9fb587d9c097de3d50f3b63f1aa3741762dbe8ffe87d69b",
    "TERMINAL_EVIDENCE.json": "5601d8f071823962da15fdf789751c5c3620fd1712fa2224cd5008e6b03d3431",
    "V1_FAILURE_CAPTURE_VALIDATION.json": "406d41c4cf80150e60a466e4fca55d441cced16a40883575011d1b9d82878699",
    "V1_FAILURE_UNIT_LOG.txt": "4c2a24c40b0c8e73846e9a18aa3d954292c5d3fae54161bde5e2a704b95e1caa",
    "V1_FAILURE_VERIFIER_LAUNCH.json": "1679107e001a4d9cb893cc9147f7a645d2d4110ff9b85bed116c038dbb055817",
    "V1_FAILURE_STDOUT.json": "3292bf363e96dfab6d60827f40eb5db2fe6721b5d12f5e437f97225ea5b22f65",
}
OUTPUT_NAMES = {"VERIFIER_ADMISSION.json", "VERIFIER_LAUNCH.json", "VERIFICATION.json", "unit.log"}
NATIVE_STDOUT_SHA = "42d141a40361c936046934f6226f94e1e54eac04960524f196c9dbfb3190064f"
BODY = r"""import base64,datetime,hashlib,json,os,stat,subprocess
from pathlib import Path

P=PAYLOAD
B=Path('/home/grinchenkoeg/computer-vision-homework-20261018')
D=B/'research_tmp/fg19_teacher_s3_verify_v2'
RUN=B/'data/artifacts/runs/fg19_t3_h384x2_e3_s3_v1'
SOURCE=B/'source-fg19-teacher-af21d6d8'
SPLIT=B/'splits/fold0.json'
U='computer-vision-homework-fg19-teacher-s3-verify-v2.service'
N='computer-vision-homework-fg19-teacher-s3-v1.service'
INV='02a8c00b433b49a185c2700624397769'
NATIVE_STDOUT_SHA='42d141a40361c936046934f6226f94e1e54eac04960524f196c9dbfb3190064f'
SPLIT_SHA='c09f7240396a33ca82977115af8d6067e571c0ec25c58a3c99b438ca2d2294b7'

def require(ok, message):
    if not ok:
        raise RuntimeError(message)

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def show(unit, fields):
    cmd=['systemctl','--user','show','--all',unit]
    for key in fields:
        cmd.extend(['-p',key])
    result=subprocess.run(cmd,capture_output=True,text=True,timeout=15)
    require(result.returncode==0,'systemctl show failed for '+unit)
    return dict(line.split('=',1) for line in result.stdout.splitlines() if '=' in line)

def sha_file(path, limit):
    before=path.lstat()
    require(stat.S_ISREG(before.st_mode) and not path.is_symlink() and before.st_size<=limit,'unsafe or oversized file: '+str(path))
    h=hashlib.sha256()
    with path.open('rb') as stream:
        while block:=stream.read(1<<20):
            h.update(block)
    after=path.stat()
    require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns),'file changed while hashing: '+str(path))
    return h.hexdigest(),after

def no_symlink_ancestors(path, stop):
    current=path
    while current!=stop:
        require(not current.is_symlink(),'symlink in registered path: '+str(current))
        current=current.parent
        require(stop in current.parents or current==stop,'path escaped trusted root')
    return path.resolve(strict=True)

def unit_native_snapshot():
    keys=['LoadState','InvocationID','ActiveState','SubState','MainPID','ExecMainCode','ExecMainStatus','ControlGroup','CPUUsageNSec','ExecMainStartTimestamp','ExecMainExitTimestamp','ExecMainStartTimestampMonotonic','ExecMainExitTimestampMonotonic','MemoryPeak']
    native=show(N,keys)
    require(native.get('InvocationID')==INV,'native invocation mismatch')
    require((native.get('ActiveState'),native.get('SubState'))==('active','exited'),'native unit is not exited')
    require((native.get('MainPID'),native.get('ExecMainCode'),native.get('ExecMainStatus'))==('0','1','0'),'native unit did not exit successfully')
    require(native.get('CPUUsageNSec','').isdigit() and int(native['CPUUsageNSec'])>0,'native CPU usage is missing')
    for key in ('ExecMainStartTimestamp','ExecMainExitTimestamp','ExecMainStartTimestampMonotonic','ExecMainExitTimestampMonotonic','MemoryPeak'):
        require(native.get(key),'native terminal field missing: '+key)
    ev=json.loads(base64.b64decode(P['TERMINAL_EVIDENCE.json']['base64'],validate=True))
    require(digest(base64.b64decode(P['TERMINAL_EVIDENCE.json']['base64'],validate=True))==P['TERMINAL_EVIDENCE.json']['sha256'],'terminal snapshot package digest differs')
    require(ev.get('raw_sha256')==NATIVE_STDOUT_SHA and ev.get('unit')==N and ev.get('invocation_id')==INV,'captured native evidence binding differs')
    require((ev.get('active_state'),ev.get('sub_state'),ev.get('main_pid'),ev.get('exec_main_code'),ev.get('exec_main_status'))==('active','exited',0,1,0),'captured terminal evidence is not successful')
    return native,ev

def check_registered_data(admission):
    data=admission.get('data')
    require(isinstance(data,dict) and len(data)==6,'registered data map is incomplete')
    summaries={}
    for rel,item in data.items():
        require(isinstance(item,dict),'registered data entry is malformed')
        path=B/'data'/rel
        require(str(path)==item.get('path'),'registered path differs: '+rel)
        before=path.lstat()
        require(stat.S_ISREG(path.stat().st_mode),'registered path is not regular: '+rel)
        require(path.is_symlink() is item.get('islink'),'registered symlink status differs: '+rel)
        resolved=path.resolve(strict=True)
        st=path.stat()
        actual={'resolved':str(resolved),'device':st.st_dev,'bytes':st.st_size,'inode':st.st_ino,'mtime_ns':st.st_mtime_ns}
        expected={key:item[key] for key in actual}
        require(actual==expected,'registered file identity changed: '+rel)
        summaries[rel]=actual
    return summaries

def check_run_artifacts():
    expected={'model.pt':32_000_000,'model_ep0.pt':32_000_000,'model_ep1.pt':32_000_000,'model_ep2.pt':32_000_000,'training_state.pt':64_000_000,'provenance.json':16_000_000,'history.json':16_000_000}
    require(RUN.is_dir() and not RUN.is_symlink(),'registered run directory is missing or linked')
    no_symlink_ancestors(RUN,B)
    rows={}
    total=0
    for name,limit in expected.items():
        path=RUN/name
        require(path.is_file() and not path.is_symlink(),'required terminal artifact missing: '+name)
        info=path.stat()
        require(info.st_size<=limit,'terminal artifact exceeds bound: '+name)
        rows[name]={'bytes':info.st_size,'device':info.st_dev,'inode':info.st_ino,'mtime_ns':info.st_mtime_ns}
        total+=info.st_size
    require(total<=128_000_000,'terminal artifact aggregate exceeds bound')
    return rows

def check_source(admission,manifest):
    require(manifest.get('source_commit')=='af21d6d8409c46b36830f0bdddd50e67ee17025f','source commit differs')
    entries=manifest.get('files')
    require(isinstance(entries,list) and len(entries)==164,'source manifest entry count differs')
    pinned={entry['path']:entry['sha256'] for entry in entries}
    require(pinned==admission.get('source_files'),'source/admission maps differ')
    actual={str(path.relative_to(SOURCE)) for path in SOURCE.rglob('*') if path.is_file()}
    require(actual==set(pinned),'source file inventory differs')
    rows={}
    total=0
    for entry in entries:
        rel=entry['path'];size=entry['bytes'];path=SOURCE/rel
        require(type(size) is int and 0<=size<=128_000_000,'invalid source size')
        no_symlink_ancestors(path,SOURCE)
        actual_sha,after=sha_file(path,size)
        require(after.st_size==size and actual_sha==entry['sha256'],'source file digest differs: '+rel)
        total+=size
        require(total<=128_000_000,'source byte budget exceeded')
        rows[rel]=actual_sha
    return {'count':len(rows),'bytes':total,'files_sha256':digest(json.dumps(rows,sort_keys=True,separators=(',',':')).encode())}

def main():
    require(B.is_dir() and not B.is_symlink(),'project root is unavailable')
    require(D.parent.is_dir() and not D.parent.is_symlink() and not os.path.lexists(D),'verification output path already exists or is linked')
    require(not (B/'runtime/STOP_ADMISSION.json').exists(),'global stop-admission marker exists')
    require(show(U,['LoadState']).get('LoadState')=='not-found','verification unit already exists')
    require(SOURCE.is_dir() and not SOURCE.is_symlink(),'pinned source tree is unavailable')
    require(str(SOURCE.resolve(strict=True))==str(SOURCE),'source path is not physical')

    admission_raw=base64.b64decode(P['ADMISSION.json']['base64'],validate=True)
    launch_raw=base64.b64decode(P['LAUNCH.json']['base64'],validate=True)
    accept_raw=base64.b64decode(P['LAUNCH_ACCEPTANCE.json']['base64'],validate=True)
    manifest_raw=base64.b64decode(P['SOURCE_MANIFEST.json']['base64'],validate=True)
    binding_raw=base64.b64decode(P['FIT_BINDING.json']['base64'],validate=True)
    require(digest(admission_raw)==P['ADMISSION.json']['sha256']=='dd3cbb8416c6cf14c29fb5937e664b41b71496420e44200c42248fa4d3109c17','admission bytes differ')
    require(digest(launch_raw)==P['LAUNCH.json']['sha256']=='6538d9502bdcd8cd3bdac0b50c4dc46b48ca80ca15d45ea25365c5313ecf8144','raw launch bytes differ')
    require(digest(accept_raw)==P['LAUNCH_ACCEPTANCE.json']['sha256']=='67479aef04bcca80d4e9d8bfbe0de71b180439a4d48fd135bffc9ccd2b94a1c7','launch acceptance bytes differ')
    require(digest(manifest_raw)==P['SOURCE_MANIFEST.json']['sha256']=='0c83367d4ee3afd7e9f382111060d8c125b52420ce27cd5cc309d338287af7fd','source manifest bytes differ')
    admission=json.loads(admission_raw);launch=json.loads(launch_raw);accept=json.loads(accept_raw);manifest=json.loads(manifest_raw);binding=json.loads(binding_raw)
    require(admission.get('unit')==N and admission.get('recipe_sha256')==P['FIT_BINDING.json']['sha256'],'fit admission binding differs')
    require(binding.get('run',{}).get('name')=='fg19_t3_h384x2_e3_s3_v1' and binding.get('source_binding',{}).get('git_commit')==manifest.get('source_commit'),'fit binding metadata differs')
    require(launch.get('unit')==N and launch.get('post',{}).get('InvocationID')==INV,'raw launch binding differs')
    require(accept.get('launch_sha256')==P['LAUNCH.json']['sha256'] and accept.get('admission_sha256')==P['ADMISSION.json']['sha256'] and accept.get('invocation_id')==INV,'launch acceptance links differ')
    require(accept.get('status')=='ADMITTED_RUNNING_NOT_TERMINAL','unexpected launch-acceptance status')
    split_before=SPLIT.lstat()
    require(stat.S_ISREG(split_before.st_mode) and not SPLIT.is_symlink() and split_before.st_size<2_000_000,'split path is not a bounded regular file')
    split_sha,split_after=sha_file(SPLIT,2_000_000)
    require(split_sha==SPLIT_SHA==admission.get('split_identity',{}).get('sha256'),'split digest differs')
    split_identity={'sha256':split_sha,'bytes':split_after.st_size,'device':split_after.st_dev,'inode':split_after.st_ino,'mtime_ns':split_after.st_mtime_ns}
    require(split_identity==admission.get('split_identity'),'split registration identity changed')
    data=check_registered_data(admission)
    artifacts=check_run_artifacts()
    source=check_source(admission,manifest)
    native,terminal=unit_native_snapshot()

    parent=show('wnn-campaign.slice',['ControlGroup','MemoryCurrent','CPUQuotaPerSecUSec','MemoryHigh','MemoryMax','MemorySwapMax'])
    cg=Path('/sys/fs/cgroup')/parent.get('ControlGroup','').lstrip('/')
    require(parent.get('ControlGroup') not in ('','/') and '/wnn-campaign.slice' in parent['ControlGroup'],'parent cgroup identity is unscoped')
    require((parent.get('CPUQuotaPerSecUSec'),parent.get('MemoryHigh'),parent.get('MemoryMax'),parent.get('MemorySwapMax'))==('2s','18000000000','20000000000','0'),'parent resource envelope differs')
    require(cg.is_dir() and cg.resolve(strict=True)==cg,'parent cgroup path unavailable or redirected')
    procs=[cg/'cgroup.procs',*cg.rglob('cgroup.procs')]
    require(len(procs)<128,'parent cgroup inventory exceeds bound')
    populated={str(path):path.read_text() for path in procs if path.read_text().strip()}
    require(not populated,'parent campaign cgroup is populated')
    require(cg.joinpath('memory.current').is_file(),'parent memory counter missing')
    current=int(cg.joinpath('memory.current').read_text())
    require(current+1_500_000_000<17_000_000_000,'parent memory headroom is insufficient')
    events={name:(cg/name).read_text() for name in ('memory.events','memory.events.local')}
    free=os.statvfs(B).f_bavail*os.statvfs(B).f_frsize
    require(free>20_000_000_000,'project filesystem free space is below the established guard')
    fs=subprocess.run(['findmnt','-T',str(D.parent),'-n','-o','FSTYPE'],capture_output=True,text=True,timeout=10,check=True).stdout.strip()
    require(fs not in ('tmpfs','ramfs'),'research_tmp is not persistent storage')
    terminal_bytes=base64.b64decode(P['TERMINAL_EVIDENCE.json']['base64'],validate=True)
    require(digest(terminal_bytes)==P['TERMINAL_EVIDENCE.json']['sha256'],'captured terminal evidence digest differs')
    require(terminal.get('raw_sha256')==NATIVE_STDOUT_SHA,'native raw stdout digest differs')
    require((native['InvocationID'],native['ActiveState'],native['SubState'],native['MainPID'],native['ExecMainCode'],native['ExecMainStatus'])==(INV,'active','exited','0','1','0'),'fresh native exit gate differs')

    D.mkdir(exist_ok=False)
    for name,item in P.items():
        raw=base64.b64decode(item['base64'],validate=True)
        require(digest(raw)==item['sha256'],'local payload digest differs: '+name)
        with (D/name).open('xb') as stream:
            stream.write(raw)
    verifier_admission={
      'schema':'fg19.teacher-s3-verifier-admission.v2',
      'utc':datetime.datetime.now(datetime.UTC).isoformat(),
      'run_name':'fg19_t3_h384x2_e3_s3_v1','unit':U,'native_unit':N,'invocation_id':INV,
      'previous_v1_failure':{
        'unit':'computer-vision-homework-fg19-teacher-s3-verify-v1.service',
        'invocation_id':'2a0e38a6515643e1b8ae9a167de10e3f',
        'cause':'weights_only=True rejected GLOBAL torch.torch_version.TorchVersion',
        'capture_files':{name:P[name]['sha256'] for name in (
          'V1_FAILURE_CAPTURE_VALIDATION.json','V1_FAILURE_UNIT_LOG.txt',
          'V1_FAILURE_VERIFIER_LAUNCH.json','V1_FAILURE_STDOUT.json')},
      },
      'native_snapshot':native,'terminal_evidence_sha256':digest(terminal_bytes),
      'source':source,'split_identity':split_identity,'registered_data_stat_only':data,
      'terminal_artifact_stat_only':artifacts,'parent':parent,'parent_cgroup':str(cg),
      'parent_populated':populated,'parent_memory_current_bytes':current,
      'parent_memory_events':events,'free_bytes':free,'research_tmp_fstype':fs,
      'staged_files':{name:item['sha256'] for name,item in P.items()},
      'payload_scope':'Code and JSON metadata only; terminal model/checkpoint contents are not opened by this admission script.'
    }
    with (D/'VERIFIER_ADMISSION.json').open('x',encoding='utf-8') as stream:
        json.dump(verifier_admission,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')

    py='/home/grinchenkoeg/WunderFunder/.venv/bin/python'
    cmd=['systemd-run','--user','--unit='+U,'--property=Type=exec','--property=RemainAfterExit=yes',
      '--property=Slice=wnn-campaign.slice','--property=CPUQuota=50%',
      '--property=MemoryHigh=1400000000','--property=MemoryMax=1500000000',
      '--property=MemorySwapMax=0','--property=RuntimeMaxSec=600',
      '--property=StandardOutput=append:'+str(D/'unit.log'),
      '--property=StandardError=append:'+str(D/'unit.log'),
      '--working-directory='+str(D)]
    env={'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1',
      'MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1',
      'TMPDIR':str(B/'research_tmp')}
    for key,value in env.items():cmd.append('--setenv='+key+'='+value)
    cmd += [py,'-u',str(D/'teacher_terminal_check_v2.py'),
      '--run-dir',str(RUN),'--source-root',str(SOURCE),'--split',str(SPLIT),
      '--source-manifest',str(D/'SOURCE_MANIFEST.json'),'--admission',str(D/'ADMISSION.json'),
      '--launch',str(D/'LAUNCH.json'),'--launch-acceptance',str(D/'LAUNCH_ACCEPTANCE.json'),
      '--terminal-evidence',str(D/'TERMINAL_EVIDENCE.json'),'--output',str(D/'VERIFICATION.json')]
    try:
        result=subprocess.run(cmd,capture_output=True,text=True,timeout=25)
        launch_result={'utc':datetime.datetime.now(datetime.UTC).isoformat(),'unit':U,'command':cmd,
          'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr,
          'post':show(U,['InvocationID','ActiveState','SubState','MainPID','ControlGroup','ExecMainCode','ExecMainStatus']),
          'unknown_outcome_policy':'Never retry blindly; inspect this unit and durable output first.'}
    except subprocess.TimeoutExpired as error:
        launch_result={'utc':datetime.datetime.now(datetime.UTC).isoformat(),'unit':U,'command':cmd,
          'outcome':'unknown','stdout':error.stdout or b'','stderr':error.stderr or b'',
          'unknown_outcome_policy':'Never retry blindly; inspect this unit and durable output first.'}
    with (D/'VERIFIER_LAUNCH.json').open('x',encoding='utf-8') as stream:
        json.dump(launch_result,stream,sort_keys=True,indent=2,default=lambda x:x.decode(errors='replace') if isinstance(x,bytes) else str(x));stream.write('\n')
    print(json.dumps({'unit':U,'launch':launch_result,'admission_sha256':digest(json.dumps(verifier_admission,sort_keys=True,separators=(',',':')).encode())},sort_keys=True))

main()
"""


def _read_payload() -> dict[str, bytes]:
    """Read only the frozen checker, helper, and metadata bindings locally."""
    paths = {
        "teacher_terminal_check_v2.py": P / "teacher_terminal_check_v2.py",
        "propensity_terminal_support.py": P / "propensity_terminal_support.py",
        "SOURCE_MANIFEST.json": P / "TEACHER_SOURCE_MANIFEST.json",
        "ADMISSION.json": RECORDS / "ADMISSION.json",
        "LAUNCH.json": RECORDS / "LAUNCH.json",
        "LAUNCH_ACCEPTANCE.json": RECORDS / "LAUNCH_ACCEPTANCE.json",
        "FIT_BINDING.json": P / "TEACHER_S3_FIT_BINDING.json",
        "CHECKER_PREPARATION_V2.json": P / "TEACHER_TERMINAL_PREPARATION_V2.json",
        "TERMINAL_EVIDENCE.json": RECORDS / "terminal_native_v1/TERMINAL_EVIDENCE.json",
        "V1_FAILURE_CAPTURE_VALIDATION.json": RECORDS
        / "terminal_verification_v1/capture/CAPTURE_VALIDATION.json",
        "V1_FAILURE_UNIT_LOG.txt": RECORDS / "terminal_verification_v1/capture/unit.log",
        "V1_FAILURE_VERIFIER_LAUNCH.json": RECORDS
        / "terminal_verification_v1/capture/VERIFIER_LAUNCH.json",
        "V1_FAILURE_STDOUT.json": RECORDS / "terminal_verification_v1/capture/STDOUT.raw.json",
    }
    payload = {name: path.read_bytes() for name, path in paths.items()}
    for name, raw in payload.items():
        actual = hashlib.sha256(raw).hexdigest()
        if actual != EXPECTED[name]:
            raise ValueError(f"frozen input digest differs for {name}: {actual}")
    collisions = set(payload).intersection(OUTPUT_NAMES)
    if collisions:
        raise ValueError(f"staged inputs collide with generated outputs: {sorted(collisions)}")
    admission = json.loads(payload["ADMISSION.json"])
    manifest = json.loads(payload["SOURCE_MANIFEST.json"])
    accept = json.loads(payload["LAUNCH_ACCEPTANCE.json"])
    if len(manifest.get("files", [])) != 164 or admission.get("source_files") != {
        item["path"]: item["sha256"] for item in manifest["files"]
    }:
        raise ValueError("admitted source map differs from the frozen source manifest")
    if accept.get("invocation_id") != "02a8c00b433b49a185c2700624397769":
        raise ValueError("launch acceptance belongs to a different invocation")
    failure = json.loads(payload["V1_FAILURE_CAPTURE_VALIDATION.json"])
    failure_unit = failure.get("unit", {})
    failure_launch = json.loads(payload["V1_FAILURE_VERIFIER_LAUNCH.json"])
    checker_receipt = json.loads(payload["CHECKER_PREPARATION_V2.json"])
    if (
        checker_receipt.get("checker", {}).get("sha256") != EXPECTED[CHECKER_NAME]
        or checker_receipt.get("status")
        != "SOURCE_AND_SYNTHETIC_CHECKS_PASS_NOT_TERMINAL_ACCEPTANCE"
        or checker_receipt.get("previous_attempt", {}).get("invocation_id")
        != "2a0e38a6515643e1b8ae9a167de10e3f"
        or failure_unit.get("InvocationID") != "2a0e38a6515643e1b8ae9a167de10e3f"
        or (failure_unit.get("ActiveState"), failure_unit.get("SubState")) != ("failed", "failed")
        or (failure_unit.get("ExecMainCode"), failure_unit.get("ExecMainStatus")) != ("1", "1")
        or failure_launch.get("unit")
        != "computer-vision-homework-fg19-teacher-s3-verify-v1.service"
        or failure_launch.get("post", {}).get("InvocationID") != "2a0e38a6515643e1b8ae9a167de10e3f"
        or "Unsupported global: GLOBAL torch.torch_version.TorchVersion"
        not in payload["V1_FAILURE_UNIT_LOG.txt"].decode("utf-8", errors="replace")
    ):
        raise ValueError("preserved V1 failure evidence does not match the scoped loader failure")
    return payload


def main() -> None:
    """Write one local, create-only remote-submission body; make no remote call."""
    payload = _read_payload()
    encoded = {
        name: {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "base64": base64.b64encode(raw).decode("ascii"),
        }
        for name, raw in payload.items()
    }
    body = BODY.replace("P=PAYLOAD", "P=" + repr(encoded))
    compile(body, "teacher_terminal_remote_submission", "exec")
    required_body_bindings = (
        f"D=B/'research_tmp/{Path(REMOTE_OUTPUT).name}'",
        f"U='{VERIFIER_UNIT}'",
        f"str(D/'{CHECKER_NAME}')",
        "--property=MemoryHigh=1400000000",
        "--property=MemoryMax=1500000000",
        "--property=RuntimeMaxSec=600",
    )
    if any(binding not in body for binding in required_body_bindings):
        raise ValueError("V2 body does not match its fixed checker/output/resource bindings")
    OUT.mkdir(exist_ok=False)
    body_path = OUT / "REMOTE_SUBMISSION.py"
    with body_path.open("x", encoding="utf-8") as stream:
        stream.write(body)
    receipt = {
        "schema": "fg19.teacher-s3-terminal-verifier-preparation.v2",
        "status": "PREPARED_LOCALLY_NOT_TRANSMITTED_OR_LAUNCHED",
        "run_name": "fg19_t3_h384x2_e3_s3_v1",
        "native_invocation_id": "02a8c00b433b49a185c2700624397769",
        "verifier_unit": VERIFIER_UNIT,
        "remote_output": REMOTE_OUTPUT,
        "remote_body_sha256": hashlib.sha256(body.encode()).hexdigest(),
        "remote_body_bytes": len(body.encode()),
        "embedded_inputs": {name: value["sha256"] for name, value in encoded.items()},
        "checker_sha256": EXPECTED["teacher_terminal_check_v2.py"],
        "support_sha256": EXPECTED["propensity_terminal_support.py"],
        "terminal_stdout_sha256": NATIVE_STDOUT_SHA,
        "previous_v1_failure": {
            "unit": "computer-vision-homework-fg19-teacher-s3-verify-v1.service",
            "invocation_id": "2a0e38a6515643e1b8ae9a167de10e3f",
            "capture_files": {
                name: EXPECTED[name]
                for name in (
                    "V1_FAILURE_CAPTURE_VALIDATION.json",
                    "V1_FAILURE_UNIT_LOG.txt",
                    "V1_FAILURE_VERIFIER_LAUNCH.json",
                    "V1_FAILURE_STDOUT.json",
                )
            },
        },
        "resource_limits": {
            "cpu_quota": "50%",
            "memory_high_bytes": 1_400_000_000,
            "memory_max_bytes": 1_500_000_000,
            "swap_max_bytes": 0,
            "runtime_seconds": 600,
        },
        "remote_behavior": "Bind the preserved V1 TorchVersion restricted-loader failure; recheck the exact successful teacher native exit, source/split/registered-path/headroom and terminal file stats; create V2 output once; stage only pinned checker/helper and JSON evidence; run the checker once with weights_only=True plus the exact TorchVersion allowlist under unchanged caps.",
        "no_actor_remote_action": True,
    }
    receipt_path = OUT / "PREPARATION.json"
    with receipt_path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
