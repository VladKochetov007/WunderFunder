from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
from pathlib import Path

REPO = Path("/Users/nb/Documents/wunder/WunderFunder-maintenance")
RECORDS = REPO / "research/campaign_2026-10-18/priority25_execution_r4/receipts"
SSH = [
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
]
REMOTE = r"""
import pathlib,subprocess,json,hashlib,base64,datetime
B=pathlib.Path('/home/grinchenkoeg/computer-vision-homework-20261018')
fields=['Id','ActiveState','SubState','Result','ExecMainStatus','ExecMainCode','ExecMainStartTimestampMonotonic','ExecMainExitTimestampMonotonic','CPUUsageNSec','MemoryPeak','MemoryCurrent','ControlGroup','CPUQuotaPerSecUSec','MemoryMax','MemorySwapMax','RuntimeMaxUSec']
result=subprocess.run(['systemctl','--user','show',unit,'-p',','.join(fields)],capture_output=True,text=True,timeout=10,check=True)
props=dict(line.split('=',1) for line in result.stdout.splitlines())
assert props['Id']==unit
out={'schema':'priority25.f3_stage1_support_capture.v1','captured_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'unit':unit,'properties':props}
if props['SubState'] not in ['exited','failed']:
 out['status']='STILL_RUNNING_NO_CAPTURE'
 print(json.dumps(out));raise SystemExit(0)
out['journal']=subprocess.run(['journalctl','--user','-u',unit,'-n','30','--no-pager'],capture_output=True,text=True,timeout=10,check=True).stdout
out['raw_files']={}
for name in names:
 path=B/'receipts'/name
 if path.is_file():
  assert not path.is_symlink() and path.stat().st_size<=2000000
  raw=path.read_bytes();out['raw_files'][name]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'base64':base64.b64encode(raw).decode()}
out['status']='TERMINAL_CAPTURE'
cg=subprocess.check_output(['systemctl','--user','show','wnn-campaign.slice','-p','ControlGroup','--value'],text=True,timeout=10).strip()
out['parent_memory_events']=(pathlib.Path('/sys/fs/cgroup')/cg.lstrip('/')/'memory.events').read_text()
print(json.dumps(out,indent=2,sort_keys=True))
"""


def main() -> None:
    """Capture raw terminal evidence before stopping only our exact exited support unit."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["wide", "deep"], required=True)
    parser.add_argument("--operation", choices=["collector", "verifier"], required=True)
    args = parser.parse_args()
    unit = f"computer-vision-homework-r01-{args.arm}-f3-stage1-{args.operation}-v1.service"
    prefix = f"RANK01_{args.arm.upper()}_F3_S1001_STAGE1"
    suffix = "COLLECTION" if args.operation == "collector" else "VERIFICATION"
    receipt_name = f"{prefix}_{suffix}.json"
    names = [
        f"{prefix}_{args.operation.upper()}_ADMISSION_V1.json",
        f"{prefix}_{args.operation.upper()}_LAUNCH_V1.json",
        receipt_name,
    ]
    header = f"unit={unit!r}\nnames={names!r}\n"
    result = subprocess.run(
        SSH, input=(header + REMOTE).encode(), capture_output=True, timeout=45, check=True
    )
    evidence = json.loads(result.stdout)
    if evidence["status"] != "TERMINAL_CAPTURE":
        print(json.dumps(evidence))
        return
    files = evidence.pop("raw_files")
    for name, value in files.items():
        assert name in names
        raw = base64.b64decode(value.pop("base64"), validate=True)
        assert len(raw) == value["bytes"] and hashlib.sha256(raw).hexdigest() == value["sha256"]
        json.loads(raw)
        with (RECORDS / name).open("xb") as stream:
            stream.write(raw)
    props = evidence["properties"]
    evidence["raw_files"] = files
    start = int(props["ExecMainStartTimestampMonotonic"])
    end = int(props["ExecMainExitTimestampMonotonic"])
    evidence["execution_seconds"] = (end - start) / 1e6 if end >= start > 0 else None
    evidence["unit_success"] = (
        props["SubState"] == "exited"
        and props["Result"] == "success"
        and props["ExecMainStatus"] == "0"
    )
    evidence["all_expected_files_captured"] = set(files) == set(names)
    capture_path = RECORDS / f"{prefix}_{suffix}_EXECUTION_V1.json"
    with capture_path.open("x") as stream:
        json.dump(evidence, stream, indent=2, sort_keys=True)
        stream.write("\n")
    stop_script = (
        f"unit={unit!r}\n"
        + r"""
import subprocess,json,datetime
raw=subprocess.check_output(['systemctl','--user','show',unit,'-p','Id,SubState,ExecMainStatus'],text=True,timeout=10)
props=dict(line.split('=',1) for line in raw.splitlines())
assert props['Id']==unit and props['SubState'] in ['exited','failed']
result=subprocess.run(['systemctl','--user','stop',unit],capture_output=True,text=True,timeout=15)
print(json.dumps({'stopped_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'unit':unit,'pre_stop':props,'after_durable_capture':True,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr},indent=2))
assert result.returncode==0
"""
    )
    stopped = subprocess.run(
        SSH, input=stop_script.encode(), capture_output=True, timeout=35, check=False
    )
    with (RECORDS / f"{prefix}_{suffix}_STOP_V1.json").open("xb") as stream:
        stream.write(stopped.stdout)
    assert stopped.returncode == 0, stopped.stderr.decode()
    print(
        json.dumps(
            {
                "status": "CAPTURED_AND_OWN_TERMINAL_STOPPED",
                "unit_success": evidence["unit_success"],
                "all_expected_files_captured": evidence["all_expected_files_captured"],
                "execution_seconds": evidence["execution_seconds"],
                "receipt_sha256": files.get(receipt_name, {}).get("sha256"),
            }
        )
    )
    assert evidence["unit_success"] and evidence["all_expected_files_captured"]


if __name__ == "__main__":
    main()
