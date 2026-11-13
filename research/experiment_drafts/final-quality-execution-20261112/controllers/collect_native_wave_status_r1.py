"""Collect bounded live telemetry and exact engineering receipts over existing RPC."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path


def tail_json(path: Path, count: int = 2) -> list[dict]:
    """Read complete trailing metric records without loading an entire training log."""
    if not path.is_file():
        return []
    with path.open('rb') as handle:
        handle.seek(max(0, path.stat().st_size - 65536))
        lines = handle.read().splitlines()
    result = []
    for line in lines:
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return result[-count:]


def main() -> None:
    """Report both legitimate failures and active fits, keeping full receipts separate."""
    l4 = Path('/data/lc-epanov_s/tabdata-final-quality-20261112-r1')
    root = l4 if l4.is_dir() else Path('/data/epanov_s/tabdata-final-quality-20261112-r1')
    fits = []
    for progress in sorted((root/'runs').glob('**/monitor_progress.json')):
        run = progress.parent
        evaluations = tail_json(run/'monitor_evaluations.jsonl', 2)
        steps = tail_json(run/'rank0.jsonl', 1)
        fits.append({'path':str(run), 'progress':json.loads(progress.read_bytes()),
                     'evaluations':evaluations, 'last_step':steps,
                     'terminal':json.loads((run/'terminal_status.json').read_bytes())['status']
                     if (run/'terminal_status.json').is_file() else None})
    profiles = []
    for path in sorted((root/'runs').glob('age*/profile/profile_receipt.json')):
        receipt = json.loads(path.read_bytes())
        profiles.append({'path':str(path), 'receipt':receipt})
    control = root/'control/profile-age'
    complete = json.loads((control/'complete.json').read_bytes()) if (control/'complete.json').is_file() else None
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw',
                          '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
    processes = subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
                                '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
    logs = []
    if control.is_dir() and complete is None:
        for path in control.glob('*.log'):
            with path.open('rb') as handle:
                handle.seek(max(0,path.stat().st_size-1800))
                logs.append({'path':str(path), 'tail':handle.read().decode(errors='replace')})
    result = {'at_epoch':time.time(), 'root':str(root),'gpu':gpu.stdout,'processes':processes.stdout,
              'physical_disk_free':shutil.disk_usage(root).free, 'fits':fits,
              'age_profile_complete':complete,'age_profiles':profiles,'age_logs':logs}
    print(json.dumps(result))


main()
