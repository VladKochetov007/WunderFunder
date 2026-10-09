"""Local controller for one receipt-owned Vast teacher run.

Subcommands:
  plan     write the private signed-URL plan (never printed; mode 600)
  create   rent one offer, write the ownership receipt (attempt id + instance id + account scope)
  start    copy plan.json/remote_run.py to the owned host and start the worker detached
  status   print the remote status summary (one ssh call)
  collect  copy checkpoints, valid predictions and logs back; verify byte counts
  destroy  destroy the receipt-owned instance and verify it is gone

Only the instance named in this attempt's receipt is ever touched.  The Vast key is read from the
private .env inside this process and passed only to the local vastai CLI; it never reaches the host.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
RUNTIME = ROOT / ".runtime/vast_teachers_r1"
RECEIPT = HERE / "OWNERSHIP_RECEIPT.json"
PLAN = RUNTIME / "plan.private.json"
KEY = Path.home() / ".ssh/vast_key"
VASTAI = str(Path.home() / ".local/bin/vastai")


def now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def api_key() -> str:
    for line in (ROOT / ".env").read_text().splitlines():
        if line.strip().startswith(("vastai=", "VAST_API_KEY=")):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("no Vast key in .env")


def vast(*args: str, timeout: int = 120) -> str:
    result = subprocess.run([VASTAI, *args, "--api-key", api_key(), "--raw"], capture_output=True,
                            text=True, timeout=timeout, check=False)
    if result.returncode != 0:
        raise SystemExit(f"vastai {args[0]} failed: {result.stderr[-400:]}")
    return result.stdout


def receipt() -> dict:
    return json.loads(RECEIPT.read_text())


def ssh_target(instance_id: int) -> tuple[str, int]:
    info = json.loads(vast("show", "instance", str(instance_id)))
    host = info.get("public_ipaddr") or info.get("ssh_host")
    ports = (info.get("ports") or {}).get("22/tcp")
    if ports:
        return host.strip(), int(ports[0]["HostPort"])
    return info["ssh_host"], int(info["ssh_port"])


def ssh(command: str, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    host, port = ssh_target(receipt()["instance_id"])
    return subprocess.run(["ssh", "-i", str(KEY), "-p", str(port), "-o", "StrictHostKeyChecking=no",
                           "-o", "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=20", "-o", "LogLevel=ERROR",
                           f"root@{host}", command], capture_output=True, text=True, timeout=timeout, check=False)


def scp(sources: list[str], destination: str, upload: bool, timeout: int = 1800) -> None:
    host, port = ssh_target(receipt()["instance_id"])
    base = ["scp", "-i", str(KEY), "-P", str(port), "-o", "StrictHostKeyChecking=no", "-o",
            "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR", "-r"]
    argv = base + sources + [f"root@{host}:{destination}"] if upload else base + [f"root@{host}:{s}" for s in sources] + [destination]
    result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
    if result.returncode != 0:
        raise SystemExit(f"scp failed: {result.stderr[-400:]}")


def cmd_plan(_: argparse.Namespace) -> None:
    from kaggle.api.kaggle_api_extended import ApiDownloadDatasetRequest, KaggleApi

    registry = json.loads((ROOT / "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
                           "multifold_r1/portable_inputs_fold1.json").read_text())["sources"]
    assets = {row["path"]: row for row in json.loads(
        (ROOT / "research/campaign_2026-10-04/kaggle/fold0_assets/assets-manifest.json").read_text())["files"]}
    bundle = json.loads((HERE / "bundle/BUNDLE_RECEIPT.json").read_text())
    wanted = [
        ("datasets/train.parquet", "wunder-funder", "train.parquet"),
        ("datasets/valid.parquet", "wunder-funder", "valid.parquet"),
        *[(p, "wnn-c20261004-fold0-assets", assets[p]["filename"]) for p in
          ("cache/chain_folds.npz", "cache/chains.npz", "cache/train_targets.npz", "cache/valid_targets.npz")],
        ("bundle.tar.gz", "wnn-vast-teacher-bundle-20261009-r2", "bundle.wnn"),
    ]
    api = KaggleApi()
    api.authenticate()
    inputs = []
    with api.build_kaggle_client() as kaggle:
        for remote_path, dataset, filename in wanted:
            request = ApiDownloadDatasetRequest()
            request.owner_slug, request.dataset_slug, request.file_name = "vladkochetov07", dataset, filename
            response = kaggle.datasets.dataset_api_client.download_dataset(request)
            url = response.request.url
            parts = urlsplit(url)
            if parts.port == 443:
                url = parts._replace(netloc=parts.hostname).geturl()
            if remote_path == "bundle.tar.gz":
                digest, size = bundle["sha256"], bundle["bytes"]
            elif remote_path.startswith("datasets/"):
                digest, size = registry[remote_path]["sha256"], registry[remote_path]["original_identity"]["bytes"]
            else:
                digest, size = assets[remote_path]["sha256"], assets[remote_path]["size"]
            inputs.append({"remote_path": remote_path, "sha256": digest, "size_bytes": size, "url": url})
    RUNTIME.mkdir(parents=True, exist_ok=True)
    PLAN.write_text(json.dumps({"inputs": inputs}))
    os.chmod(PLAN, 0o600)
    print(json.dumps({"plan": str(PLAN), "files": [(i["remote_path"], i["size_bytes"]) for i in inputs],
                      "total_gb": round(sum(i["size_bytes"] for i in inputs) / 1e9, 2)}))


def cmd_create(args: argparse.Namespace) -> None:
    if RECEIPT.exists():
        raise SystemExit("receipt exists: this attempt already created an instance")
    attempt = {"attempt_id": args.attempt, "offer_id": args.offer, "requested_utc": now(), "state": "creating",
               "image": args.image, "disk_gb": args.disk, "hard_stop_utc": args.hard_stop}
    RECEIPT.write_text(json.dumps(attempt, indent=1))
    out = vast("create", "instance", str(args.offer), "--image", args.image, "--disk", str(args.disk), "--ssh",
               "--direct", "--label", args.attempt)
    response = json.loads(out)
    attempt |= {"state": "created", "instance_id": int(response["new_contract"]), "create_response": response,
                "created_utc": now()}
    RECEIPT.write_text(json.dumps(attempt, indent=1))
    print(json.dumps({k: attempt[k] for k in ("attempt_id", "instance_id", "created_utc")}))


def cmd_start(_: argparse.Namespace) -> None:
    result = ssh("mkdir -p /root/run && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader && df -h /root | tail -1 && nproc && free -g | head -2")
    print(result.stdout, result.stderr[-300:])
    if result.returncode != 0:
        raise SystemExit("host not reachable yet")
    scp([str(PLAN), str(HERE / "remote_run.py")], "/root/run/", upload=True, timeout=600)
    ssh("mv /root/run/plan.private.json /root/run/plan.json && chmod 600 /root/run/plan.json")
    started = ssh("(cd /root/run && setsid nohup python remote_run.py > worker.out 2>&1 < /dev/null &) ; echo started")
    print(started.stdout.strip())


def cmd_status(_: argparse.Namespace) -> None:
    result = ssh("cat /root/run/status.json 2>/dev/null; ls /root/run/DONE.json /root/run/FAILED.json 2>/dev/null; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader")
    text = result.stdout
    try:
        status = json.loads(text[: text.rindex("}") + 1])
    except ValueError:
        print(text[-1500:], result.stderr[-300:])
        return
    summary = {"phase": status.get("phase"), "elapsed_min": round((status.get("updated", 0) - status.get("started", 0)) / 60, 1),
               "downloads": {k: (v.get("state"), v.get("mb_per_s"), v.get("download_seconds")) for k, v in status.get("downloads", {}).items()},
               "jobs": {k: (v.get("state"), v.get("gpu"), v.get("fit_seconds"), v.get("last_line", "")[-90:]) for k, v in status.get("jobs", {}).items()},
               "error": status.get("error")}
    print(json.dumps(summary, indent=1))
    print(text[text.rindex("}") + 1:].strip())


def cmd_collect(_: argparse.Namespace) -> None:
    dest = HERE / "collected"
    (dest / "runs").mkdir(parents=True, exist_ok=True)
    (dest / "cache").mkdir(exist_ok=True)
    listing = ssh("cd /root/wnn && ls -d artifacts/runs/vt_* 2>/dev/null; ls cache/pred_vt_* 2>/dev/null")
    runs = [line for line in listing.stdout.split() if line.startswith("artifacts/runs/")]
    preds = [line for line in listing.stdout.split() if line.startswith("cache/")]
    for run in runs:
        scp([f"/root/wnn/{run}"], str(dest / "runs"), upload=False)
    if preds:
        scp([f"/root/wnn/{p}" for p in preds], str(dest / "cache"), upload=False, timeout=3600)
    scp(["/root/run/status.json", "/root/run/logs"], str(dest), upload=False)
    print(json.dumps({"runs": runs, "predictions": preds}))


def cmd_queue(_: argparse.Namespace) -> None:
    """Ship refill daemon + job files to the owned host and start the daemon (idempotent)."""
    jobs = sorted((HERE / "refill_jobs").glob("*.json"))
    ssh("mkdir -p /root/run/refill/pending /root/run/refill/running /root/run/refill/done")
    scp([str(HERE / "remote_refill.py")], "/root/run/", upload=True)
    have = {re.sub(r"^p\d\d_", "", name) for name in
            ssh("ls /root/run/refill/pending /root/run/refill/running /root/run/refill/done").stdout.split()}
    new = [str(j) for j in jobs if j.name not in have and not any(h.startswith(j.stem) for h in have)]
    if new:
        scp(new, "/root/run/refill/pending/", upload=True)
    alive = ssh("pgrep -fc '^python remote_refill.py' || true").stdout.strip()
    if alive in ("", "0"):
        ssh("(cd /root/run && setsid nohup python remote_refill.py > refill.out 2>&1 < /dev/null &) ; echo started", timeout=30)
    print(json.dumps({"queued": [Path(n).name for n in new], "daemon_was_running": alive not in ("", "0")}))


FAMILIES = [
    ("fold", re.compile(r"vt_f[023]_")), ("sched", re.compile(r"sgdr|trap|wsd|lineard2z|ep6")),
    ("optim", re.compile(r"adamw|muonhidden")), ("weighting", re.compile(r"lw15|lwfloor|cls3")),
    ("arch", re.compile(r"gated|tcn|structured|eventcontent|twoclock|dual|lstm|rawonly|encmlp")),
    ("width", re.compile(r"gru(256x3|320|384|448|512)")), ("loss", re.compile(r"corr|huber|wmsegcorr")),
    ("regularize", re.compile(r"drop2|wd05|swa|b16|tr_lr|tr_base|seed")),
]


def family_of(name: str) -> str:
    for family, pattern in FAMILIES:
        if pattern.search(name):
            return family
    return "other"


def interleave(names: list[str]) -> list[str]:
    """Round-robin across mechanism families so concurrently running jobs differ; stable within a family."""
    buckets: dict[str, list[str]] = {}
    for name in sorted(names):
        buckets.setdefault(family_of(re.sub(r"^p\d\d_", "", name)), []).append(name)
    order = []
    while any(buckets.values()):
        for family in sorted(buckets, key=lambda f: -len(buckets[f])):
            if buckets[family]:
                order.append(buckets[family].pop(0))
    return order


def cmd_reorder(_: argparse.Namespace) -> None:
    pending = ssh("ls /root/run/refill/pending").stdout.split()
    order = interleave(pending)
    commands = [f"mv /root/run/refill/pending/{old} /root/run/refill/pending/p{i:02d}_{re.sub(r'^p[0-9][0-9]_', '', old)}"
                for i, old in enumerate(order)]
    if commands:
        ssh(" ; ".join(commands) + " ; true", timeout=120)
    print(json.dumps([f"p{i:02d}:{family_of(re.sub(r'^p[0-9][0-9]_', '', n))}" for i, n in enumerate(order)]))


def cmd_pull(_: argparse.Namespace) -> None:
    """Fetch every finished job not yet pulled; append bank rows.  Serialized by a file lock."""
    import fcntl

    RUNTIME.mkdir(parents=True, exist_ok=True)
    with (RUNTIME / "pull.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _pull_locked()


def _pull_locked() -> None:
    bank_dir = ROOT / "research/campaign_2026-10-06/bandit_continuation/prediction_bank"
    banks = {1: bank_dir / "BANK_FOLD1.jsonl", **{k: bank_dir / f"BANK_FOLD{k}_VAST.jsonl" for k in (0, 2, 3)}}
    for path in banks.values():
        path.touch()
    have = {json.loads(line)["name"] for path in banks.values() for line in path.read_text().splitlines() if line.strip()}
    listing = ssh("cd /root/wnn && ls cache/pred_vt_*_valid.json 2>/dev/null").stdout.split()
    pulled = []
    for item in listing:
        name = item.removeprefix("cache/pred_").removesuffix("_valid.json")
        if name in have:
            continue
        run = name.split("__")[0]
        sizes = ssh(f"cd /root/wnn && stat -c %s cache/pred_{name}_valid.npy && ls artifacts/runs/{run}/model_ep3.pt").stdout
        if "model_ep3.pt" not in sizes:
            continue
        busy = ssh(f"pgrep -f '[p]redict_run.py.* {name} ' || true").stdout.strip()
        if busy:
            continue
        if not (ROOT / "artifacts/runs" / run / "model_ep3.pt").exists():
            scp([f"/root/wnn/artifacts/runs/{run}"], str(ROOT / "artifacts/runs"), upload=False)
        scp([f"/root/wnn/cache/pred_{name}_valid.npy", f"/root/wnn/cache/pred_{name}_valid.json"], str(ROOT / "cache"), upload=False)
        fold_match = re.match(r"vt_f(\d)_", name)
        fold = int(fold_match.group(1)) if fold_match else 1
        row = {"name": name, "fold": fold, "prediction": f"cache/pred_{name}_valid.npy",
               "tags": {"scale": "full", "family": "teacher_scratch", "vast": "r1"}, "source": "vast_teachers_r1"}
        with banks[fold].open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        pulled.append(name)
    print(json.dumps({"pulled": pulled}))


REMOTE_PROBE = r"""
cd /root/run
echo "GPU $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | tr '\n' ';')"
echo "GPUPROCS $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"
echo "PENDING $(ls refill/pending 2>/dev/null | wc -l) RUNNING $(ls refill/running 2>/dev/null | wc -l) DONEJOBS $(ls refill/done 2>/dev/null | tr '\n' ',')"
echo "TERMINAL $(ls DONE.json FAILED.json 2>/dev/null | tr '\n' ',')"
echo "PREDS $(ls /root/wnn/cache/pred_vt_*_valid.json 2>/dev/null | wc -l)"
for f in logs/vt_*.fit.log; do n=$(basename $f .fit.log); l=$(grep '^ep ' $f | tail -1); [ -n "$l" ] && echo "JOB $n $l"; done
"""


def probe() -> dict:
    out = ssh(REMOTE_PROBE, timeout=60).stdout.splitlines()
    info: dict = {"jobs": {}}
    for line in out:
        key, _, rest = line.partition(" ")
        if key == "JOB":
            name, _, log = rest.partition(" ")
            info["jobs"][name] = log
        else:
            info[key] = rest
    return info


def spend() -> tuple[float, float]:
    row = json.loads(vast("show", "instance", str(receipt()["instance_id"])))
    hours = (time.time() - float(row["start_date"])) / 3600
    return hours, hours * float(row["dph_total"])


def eta_minutes(log: str) -> str:
    import re

    match = re.match(r"ep (\d+) batch (\d+)/(\d+) loss ([\d.]+) lr \S+ seq/s ([\d.]+)", log)
    if not match:
        return "?"
    epoch, batch, total, _, rate = int(match[1]), int(match[2]), int(match[3]), match[4], float(match[5])
    remaining = ((3 - epoch) * total + (total - batch)) * 32 / max(rate, 1e-3) / 60
    return f"ep{epoch} {batch}/{total} {rate:.0f}seq/s eta{remaining:.0f}m"


def cmd_dash(_: argparse.Namespace) -> None:
    info = probe()
    hours, cost = spend()
    print(f"GPU util%,MiB: {info.get('GPU')}  compute procs: {info.get('GPUPROCS')}  spend: {hours:.2f}h ${cost:.2f}")
    print(f"queue pending={info.get('PENDING')}  terminal={info.get('TERMINAL')}  preds on host={info.get('PREDS')}")
    for name, log in sorted(info["jobs"].items()):
        print(f"  {name:26s} {eta_minutes(log)}")


def cmd_tick(args: argparse.Namespace) -> None:
    """Unattended step: pull finished jobs; destroy the owned instance when drained or over budget."""
    if receipt().get("state") == "destroyed_verified":
        print("already destroyed")
        return
    info = probe()
    cmd_pull(args)
    hours, cost = spend()
    pending = int(info.get("PENDING", "1").split()[0])
    running = int(info.get("GPUPROCS", "1"))
    in_flight = int(info.get("PENDING", "1 RUNNING 1").split("RUNNING")[1].split()[0])
    primary_done = bool(info.get("TERMINAL", "").strip(", "))
    local = {p.name for p in (ROOT / "cache").glob("pred_vt_*_valid.npy")}
    remote = int(info.get("PREDS", "0"))
    held = cost >= args.cap
    if held and pending > 0:
        ssh("mkdir -p /root/run/refill/held && mv /root/run/refill/pending/*.json /root/run/refill/held/ 2>/dev/null || true")
        print(f"spend cap ${args.cap} reached: queued jobs moved to held/, running jobs finish gracefully")
    drained = (pending == 0 or held) and in_flight == 0 and running == 0 and primary_done and len(local) >= remote
    print(json.dumps({"hours": round(hours, 2), "cost": round(cost, 2), "pending": pending, "gpu_procs": running,
                      "held": held, "drained": drained, "gpu": info.get("GPU")}))
    if drained:
        cmd_destroy(args)


def cmd_destroy(_: argparse.Namespace) -> None:
    data = receipt()
    vast("destroy", "instance", str(data["instance_id"]), "-y")
    for _ in range(12):
        time.sleep(10)
        listing = json.loads(vast("show", "instances"))
        if not any(int(row.get("id", -1)) == data["instance_id"] for row in listing):
            data |= {"state": "destroyed_verified", "destroyed_utc": now()}
            RECEIPT.write_text(json.dumps(data, indent=1))
            print("destroyed_verified")
            return
    raise SystemExit("instance still listed after destroy request")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan")
    create = sub.add_parser("create")
    create.add_argument("--offer", type=int, required=True)
    create.add_argument("--attempt", required=True)
    create.add_argument("--image", default="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime")
    create.add_argument("--disk", type=int, default=150)
    create.add_argument("--hard-stop", required=True)
    for name in ("start", "status", "collect", "destroy", "queue", "pull", "dash", "tick", "reorder"):
        node = sub.add_parser(name)
        if name == "tick":
            node.add_argument("--cap", type=float, default=5.0, help="stop admitting queued jobs at this spend; running jobs always finish, then the instance is destroyed")
    args = parser.parse_args()
    {"plan": cmd_plan, "create": cmd_create, "start": cmd_start, "status": cmd_status,
     "collect": cmd_collect, "destroy": cmd_destroy, "queue": cmd_queue, "pull": cmd_pull, "dash": cmd_dash, "tick": cmd_tick, "reorder": cmd_reorder}[args.command](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
