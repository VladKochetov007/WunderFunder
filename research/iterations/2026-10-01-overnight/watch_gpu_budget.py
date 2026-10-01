"""Monitor only explicitly registered campaign training scopes; never touch other jobs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import fcntl
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--limit-mib", type=int, default=5632)
    a = ap.parse_args()
    raw = a.plan.read_bytes()
    jobs = {j["name"] for j in json.loads(raw)["jobs"]}
    assert jobs and all(name.startswith("on_") and name.replace("_", "").isalnum() for name in jobs)
    assert a.limit_mib == 5632
    peaks = {name: 0 for name in jobs}
    samples = {name: 0 for name in jobs}
    stopped = set()
    with a.output.open("x") as out:
        def emit(event, **kwargs):
            out.write(json.dumps({"utc": datetime.now(timezone.utc).isoformat(), "event": event, **kwargs}) + "\n")
            out.flush()
        emit("started", plan_sha256=hashlib.sha256(raw).hexdigest(), jobs=sorted(jobs), limit_MiB=a.limit_mib)
        while True:
            assert a.plan.read_bytes() == raw, "Plan changed while monitoring"
            with (HERE / "job-ledger.jsonl").open() as ledger:
                fcntl.flock(ledger, fcntl.LOCK_SH)
                rows = [json.loads(x) for x in ledger if x.strip()]
            done = {r["job"] for r in rows if r.get("stage") == "train" and r["event"] in ("completed", "failed")}
            if jobs <= done:
                emit("finished", peaks_MiB=peaks, sample_counts=samples, stopped=sorted(stopped))
                return
            if datetime.now(timezone.utc).isoformat() >= "2026-10-01T08:23:30+00:00":
                emit("monitor_deadline", peaks_MiB=peaks, incomplete=sorted(jobs - done))
                return
            active = {r["job"] for r in rows if r.get("stage") == "train" and r["event"] == "started"} & jobs - done
            listing = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], text=True, timeout=10)
            usage = {}
            for line in listing.splitlines():
                fields = [x.strip() for x in line.split(",")]
                if len(fields) == 2 and all(x.isdigit() for x in fields):
                    pid, memory = map(int, fields)
                    usage[pid] = usage.get(pid, 0) + memory
            for name in active:
                unit = f"wunder-{name}-train.scope"
                cg = subprocess.check_output(["systemctl", "--user", "show", unit, "-p", "ControlGroup", "--value"], text=True, timeout=5).strip()
                if not cg:
                    continue
                group = Path("/sys/fs/cgroup") / cg.lstrip("/")
                process_file = group / "cgroup.procs"
                if not process_file.exists():
                    continue
                pids = {int(x) for x in process_file.read_text().split()}
                memory = sum(usage.get(pid, 0) for pid in pids)
                peaks[name] = max(peaks[name], memory)
                samples[name] += 1
                if memory > a.limit_mib and name not in stopped:
                    emit("own_scope_GPU_limit_exceeded", job=name, scope=unit, observed_MiB=memory)
                    subprocess.run(["systemctl", "--user", "stop", unit], check=True, timeout=20)
                    stopped.add(name)
            time.sleep(5)


if __name__ == "__main__":
    main()
