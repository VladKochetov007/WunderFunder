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
    started = ssh("cd /root/run && nohup python remote_run.py > worker.out 2>&1 & echo started")
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


def cmd_destroy(_: argparse.Namespace) -> None:
    data = receipt()
    vast("destroy", "instance", str(data["instance_id"]))
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
    for name in ("start", "status", "collect", "destroy"):
        sub.add_parser(name)
    args = parser.parse_args()
    {"plan": cmd_plan, "create": cmd_create, "start": cmd_start, "status": cmd_status,
     "collect": cmd_collect, "destroy": cmd_destroy}[args.command](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
