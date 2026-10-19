"""Capture exact admitted profile metadata and original scalar receipt bytes."""

from __future__ import annotations

import base64
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Make one bounded read-only observation, preserving raw output before parsing."""
    launch = json.loads((ROOT / "records/profile_v1/LAUNCH.json").read_bytes())
    properties = dict(
        line.split("=", 1) for line in launch["post_launch_systemctl"]["stdout"].splitlines()
    )
    invocation = properties["InvocationID"]
    body = """
import base64,datetime,json,subprocess,pathlib,hashlib
UNIT='computer-vision-homework-fg19-propensity-profile-v1.service'
OUT=pathlib.Path('/home/grinchenkoeg/computer-vision-homework-20261018/research_tmp/fg19_propensity_profile_v1')
expected=EXPECTED
p=subprocess.run(['systemctl','--user','show','--all',UNIT,'-p','InvocationID','-p','ExecStart','-p','MainPID','-p','ControlGroup','-p','ActiveState','-p','SubState','-p','Result','-p','ExecMainCode','-p','ExecMainStatus','-p','CPUUsageNSec','-p','MemoryPeak','-p','ExecMainStartTimestamp','-p','ExecMainExitTimestamp'],capture_output=True,text=True,timeout=15)
assert p.returncode==0,(p.returncode,p.stderr)
props=dict(line.split('=',1) for line in p.stdout.splitlines())
assert props['InvocationID']==expected
parent=subprocess.run(['systemctl','--user','show','--all','wnn-campaign.slice','-p','ControlGroup','-p','MemoryCurrent'],capture_output=True,text=True,timeout=15)
assert parent.returncode==0
pp=dict(line.split('=',1) for line in parent.stdout.splitlines());assert pp['ControlGroup']
cg=pathlib.Path('/sys/fs/cgroup')/pp['ControlGroup'].lstrip('/')
paths=[cg/'cgroup.procs',*cg.rglob('cgroup.procs')];assert len(paths)<128
children={str(x):x.read_text() for x in paths if x.read_text().strip()}
files={}
for name in ['PROFILE_RESULT.json','UPDATE_PROGRESS.jsonl','unit.log']:
 f=OUT/name
 if f.exists():
  assert f.stat().st_size<2000000
  b=f.read_bytes();files[name]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'base64':base64.b64encode(b).decode()}
record={'created_utc':datetime.datetime.now(datetime.UTC).isoformat(),'expected_invocation':expected,'properties':props,'raw_systemctl':p.stdout,'parent_properties':pp,'parent_events':(cg/'memory.events').read_text(),'parent_events_local':(cg/'memory.events.local').read_text(),'populated_descendants':children,'own_cgroup_files':None,'files':files}
if props.get('ControlGroup'):
 own=pathlib.Path('/sys/fs/cgroup')/props['ControlGroup'].lstrip('/')
 record['own_cgroup_files']={n:(own/n).read_text() if (own/n).exists() else None for n in ['memory.current','memory.events','memory.events.local','cgroup.procs']}
print(json.dumps(record))
""".replace("expected=EXPECTED", "expected=" + repr(invocation))
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    directory = ROOT / "records/profile_v1" / ("capture_" + stamp)
    directory.mkdir(exist_ok=False)
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
        timeout=60,
    )
    (directory / "REMOTE.py").write_text(body)
    (directory / "STDOUT.raw.json").write_bytes(result.stdout)
    (directory / "STDERR.raw").write_bytes(result.stderr)
    if result.returncode:
        raise RuntimeError("Read-only capture failed; preserved raw output; no compute retry")
    value = json.loads(result.stdout)
    for name, item in value["files"].items():
        data = base64.b64decode(item["base64"], validate=True)
        import hashlib

        assert len(data) == item["bytes"] and hashlib.sha256(data).hexdigest() == item["sha256"]
        (directory / name).write_bytes(data)
    print(
        json.dumps(
            {
                "capture": str(directory),
                "properties": value["properties"],
                "files": list(value["files"]),
            }
        )
    )


if __name__ == "__main__":
    main()
