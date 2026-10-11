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
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
# Lane "" is the first Vast account (.env `vastai`); lane "b" is the second account (.env `vastai2`).  Each lane has its own
# ownership receipt, runtime directory, bundle and job directory; no operation ever crosses lanes.
LANE = os.environ.get("VT_LANE", "")
SUFFIX = f"_{LANE}" if LANE else ""
KEY_PREFIXES = ("vastai2=",) if LANE == "b" else ("vastai=", "VAST_API_KEY=")
RUNTIME = ROOT / f".runtime/vast_teachers_r1{SUFFIX}"
RECEIPT = HERE / f"OWNERSHIP_RECEIPT{SUFFIX}.json"
PLAN = RUNTIME / "plan.private.json"
BUNDLE_DIR = HERE / f"bundle{SUFFIX}"
JOBS_DIR = HERE / f"refill_jobs{SUFFIX}"
KEY = Path.home() / ".ssh/vast_key"
VASTAI = str(Path.home() / ".local/bin/vastai")
COLLECTED_DIR = HERE / f"collected{SUFFIX}"
COLLECTION_MANIFEST = COLLECTED_DIR / "collection_manifest.json"
COLLECTION_RECEIPT = COLLECTED_DIR / "collection_receipt.json"
COLLECTION_VERIFICATION = RUNTIME / "collection_verified.json"
COLLECTION_MANIFEST_SCHEMA = "vast-teacher-remote-manifest-v1"
COLLECTION_RECEIPT_SCHEMA = "vast-teacher-collection-receipt-v1"
MAX_COLLECTION_JSON_BYTES = 8 * 1024 * 1024
MAX_COLLECTION_ARTIFACTS = 4096


def now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def api_key() -> str:
    for line in (ROOT / ".env").read_text().splitlines():
        if line.strip().startswith(KEY_PREFIXES):
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


def _safe_create_response(response: object) -> dict[str, bool | int]:
    """Keep only the scalar create fields needed for the ownership receipt."""
    if not isinstance(response, dict):
        raise SystemExit("Vast create response is not an object")
    safe: dict[str, bool | int] = {}
    success = response.get("success")
    if type(success) is bool:
        safe["success"] = success
    contract = response.get("new_contract")
    if type(contract) is int:
        safe["new_contract"] = contract
    elif isinstance(contract, str) and contract.isdecimal():
        safe["new_contract"] = int(contract)
    return safe


def _collection_inventory_sha256(files: list[dict[str, object]]) -> str:
    """Hash the sorted remote path, byte count and SHA-256 inventory."""
    inventory = [
        {"remote_path": item["remote_path"], "size_bytes": item["size_bytes"], "sha256": item["sha256"]}
        for item in sorted(files, key=lambda value: str(value["remote_path"]))
    ]
    payload = json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _collection_local_path(remote_path: str) -> str | None:
    """Map an allowed remote output path to its fixed collected/ destination."""
    path = PurePosixPath(remote_path)
    if path.is_absolute() or ".." in path.parts or not path.parts or path.as_posix() != remote_path:
        return None
    if path.parts[0] == "wnn" and len(path.parts) >= 3:
        if path.parts[1] == "cache" and path.name.startswith("pred_vt_") and path.name.endswith(("_valid.npy", "_valid.json")):
            return str(PurePosixPath("cache", *path.parts[2:]))
        if path.parts[1:3] == ("artifacts", "runs") and len(path.parts) >= 4 and path.parts[3].startswith("vt_"):
            return str(PurePosixPath("runs", *path.parts[3:]))
    if path.parts[:2] == ("run", "logs") and len(path.parts) >= 3:
        return str(PurePosixPath("logs", *path.parts[2:]))
    if path.as_posix() in {"run/status.json", "run/DONE.json", "run/FAILED.json"}:
        return path.name
    return None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _collection_stat(path: Path) -> tuple[int, int, int, int, int] | None:
    """Return a stable local identity without reading a potentially large artifact."""
    try:
        relative = path.relative_to(COLLECTED_DIR)
        if COLLECTED_DIR.is_symlink():
            return None
        current = COLLECTED_DIR
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                return None
        identity = path.lstat()
    except OSError:
        return None
    except ValueError:
        return None
    if not stat.S_ISREG(identity.st_mode):
        return None
    return (identity.st_dev, identity.st_ino, identity.st_size, identity.st_mtime_ns, identity.st_ctime_ns)


def _read_bounded_json(path: Path) -> tuple[dict[str, object] | None, bytes | None]:
    try:
        raw = path.read_bytes()
    except OSError:
        return None, None
    if len(raw) > MAX_COLLECTION_JSON_BYTES:
        return None, None
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None, raw
    if not isinstance(value, dict):
        return None, raw
    return value, raw


def _collection_status(ownership: dict[str, object]) -> tuple[bool, str]:
    """Verify the remote inventory and local hashes, caching unchanged files.

    The remote manifest has ``schema``, ``inventory_complete``,
    ``inventory_sha256`` and ``files`` fields. Each file records a path relative
    to ``/root`` plus its byte count and SHA-256. The adjacent collector receipt
    binds the manifest's raw SHA-256 and inventory identity, the owning
    attempt/instance, ``collector_reviewed_complete: true``, and an ``artifacts``
    list with matching remote paths/hashes and deterministic local paths.
    Supported roots map ``wnn/cache`` to ``collected/cache``,
    ``wnn/artifacts/runs`` to ``collected/runs``, ``run/logs`` to
    ``collected/logs``, and run status/terminal files to ``collected``. Local
    files are hashed once; the durable cache reuses that result while their
    device, inode, size, mtime and ctime remain unchanged.
    """
    manifest, manifest_raw = _read_bounded_json(COLLECTION_MANIFEST)
    collection, collection_raw = _read_bounded_json(COLLECTION_RECEIPT)
    if manifest is None or manifest_raw is None:
        return False, "remote collection manifest is missing or invalid"
    if collection is None or collection_raw is None:
        return False, "collector receipt is missing or invalid"
    if manifest.get("schema") != COLLECTION_MANIFEST_SCHEMA or manifest.get("inventory_complete") is not True:
        return False, "remote collection manifest is not marked complete"
    if collection.get("schema") != COLLECTION_RECEIPT_SCHEMA:
        return False, "collector receipt schema is unsupported"
    if collection.get("collector_reviewed_complete") is not True:
        return False, "collector has not reviewed complete artifact collection"
    if collection.get("attempt_id") != ownership.get("attempt_id") or collection.get("instance_id") != ownership.get("instance_id"):
        return False, "collector receipt belongs to another Vast attempt"
    manifest_sha256 = hashlib.sha256(manifest_raw).hexdigest()
    if collection.get("remote_manifest_sha256") != manifest_sha256:
        return False, "collector receipt does not bind the remote manifest bytes"

    raw_files = manifest.get("files")
    if not isinstance(raw_files, list) or not raw_files or len(raw_files) > MAX_COLLECTION_ARTIFACTS:
        return False, "remote collection manifest has no bounded artifact inventory"
    files: list[dict[str, object]] = []
    seen_remote: set[str] = set()
    predictions: set[str] = set()
    sidecars: set[str] = set()
    for item in raw_files:
        if not isinstance(item, dict) or set(item) != {"remote_path", "size_bytes", "sha256"}:
            return False, "remote collection manifest has a malformed artifact entry"
        remote_path = item.get("remote_path")
        size_bytes = item.get("size_bytes")
        sha256 = item.get("sha256")
        if not isinstance(remote_path, str) or not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes < 0:
            return False, "remote collection manifest has an invalid path or byte count"
        if not isinstance(sha256, str) or re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
            return False, "remote collection manifest has an invalid SHA-256"
        if remote_path in seen_remote or _collection_local_path(remote_path) is None:
            return False, "remote collection manifest has a duplicate or unsupported path"
        seen_remote.add(remote_path)
        if remote_path.startswith("wnn/cache/") and remote_path.endswith("_valid.npy"):
            predictions.add(remote_path.removesuffix(".npy"))
        if remote_path.startswith("wnn/cache/") and remote_path.endswith("_valid.json"):
            sidecars.add(remote_path.removesuffix(".json"))
        files.append({"remote_path": remote_path, "size_bytes": size_bytes, "sha256": sha256})
    if predictions != sidecars:
        return False, "remote prediction arrays and sidecars do not match by name"
    inventory_sha256 = _collection_inventory_sha256(files)
    if manifest.get("inventory_sha256") != inventory_sha256 or collection.get("remote_inventory_sha256") != inventory_sha256:
        return False, "collector receipt does not bind the exact remote inventory"

    raw_artifacts = collection.get("artifacts")
    if not isinstance(raw_artifacts, list) or len(raw_artifacts) != len(files):
        return False, "collector receipt artifact count differs from the remote inventory"
    by_remote: dict[str, dict[str, object]] = {}
    for item in raw_artifacts:
        if not isinstance(item, dict) or set(item) != {"remote_path", "local_path", "size_bytes", "sha256"}:
            return False, "collector receipt has a malformed local artifact entry"
        remote_path = item.get("remote_path")
        if not isinstance(remote_path, str) or remote_path in by_remote:
            return False, "collector receipt has a duplicate or invalid remote path"
        by_remote[remote_path] = item
    if set(by_remote) != seen_remote:
        return False, "collector receipt names differ from the remote inventory"

    local_files: list[tuple[dict[str, object], Path]] = []
    for remote_file in sorted(files, key=lambda item: str(item["remote_path"])):
        remote_path = str(remote_file["remote_path"])
        artifact = by_remote[remote_path]
        local_path = _collection_local_path(remote_path)
        if local_path is None:
            return False, f"remote artifact path is unsupported: {remote_path}"
        if artifact != {**remote_file, "local_path": local_path}:
            return False, f"collector receipt does not exactly bind {remote_path}"
        path = COLLECTED_DIR.joinpath(*PurePosixPath(local_path).parts)
        stat_identity = _collection_stat(path)
        if stat_identity is None or stat_identity[2] != remote_file["size_bytes"]:
            return False, f"collected artifact is missing or has the wrong size: {local_path}"
        local_files.append((remote_file, path))

    receipt_sha256 = hashlib.sha256(collection_raw).hexdigest()
    cached, _ = _read_bounded_json(COLLECTION_VERIFICATION)
    current_stats = [
        {"local_path": _collection_local_path(str(item["remote_path"])), "stat": list(_collection_stat(path) or ())}
        for item, path in local_files
    ]
    if cached is not None and cached.get("schema") == "vast-teacher-collection-verification-v1" and cached.get("receipt_sha256") == receipt_sha256 and cached.get("manifest_sha256") == manifest_sha256 and cached.get("files") == current_stats:
        return True, "verified_cached"
    for remote_file, path in local_files:
        try:
            local_sha256 = _file_sha256(path)
        except OSError:
            return False, f"could not hash collected artifact: {path.name}"
        if local_sha256 != remote_file["sha256"]:
            return False, f"collected artifact hash differs from remote manifest: {path.name}"

    verified = {"schema": "vast-teacher-collection-verification-v1", "receipt_sha256": receipt_sha256,
                "manifest_sha256": manifest_sha256, "verified_utc": now(), "files": current_stats}
    try:
        COLLECTION_VERIFICATION.parent.mkdir(parents=True, exist_ok=True)
        temporary = COLLECTION_VERIFICATION.with_suffix(".tmp")
        temporary.write_text(json.dumps(verified, separators=(",", ":")))
        os.chmod(temporary, 0o600)
        temporary.replace(COLLECTION_VERIFICATION)
    except OSError:
        return False, "could not persist verified collection evidence"
    return True, "verified"


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
    bundle = json.loads((BUNDLE_DIR / "BUNDLE_RECEIPT.json").read_text())
    wanted = [
        ("datasets/train.parquet", "wunder-funder", "train.parquet"),
        ("datasets/valid.parquet", "wunder-funder", "valid.parquet"),
        *[(p, "wnn-c20261004-fold0-assets", assets[p]["filename"]) for p in
          ("cache/chain_folds.npz", "cache/chains.npz", "cache/train_targets.npz", "cache/valid_targets.npz")],
        ("bundle.tar.gz", bundle.get("dataset_slug", "wnn-vast-teacher-bundle-20261009-r2"), bundle.get("dataset_file", "bundle.wnn")),
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
    response = _safe_create_response(json.loads(out))
    if "new_contract" not in response:
        raise SystemExit("Vast create response omitted a valid new_contract")
    attempt |= {"state": "created", "instance_id": response["new_contract"], "create_response": response,
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
    dest = COLLECTED_DIR
    (dest / "runs").mkdir(parents=True, exist_ok=True)
    (dest / "cache").mkdir(exist_ok=True)
    listing = ssh("cd /root/wnn && ls -d artifacts/runs/vt_* 2>/dev/null; ls cache/pred_vt_* 2>/dev/null")
    runs = [line for line in listing.stdout.split() if line.startswith("artifacts/runs/")]
    preds = [line for line in listing.stdout.split() if line.startswith("cache/")]
    for run in runs:
        scp([f"/root/wnn/{run}"], str(dest / "runs"), upload=False)
    if preds:
        scp([f"/root/wnn/{p}" for p in preds], str(dest / "cache"), upload=False, timeout=3600)
    terminal = ssh("ls /root/run/DONE.json /root/run/FAILED.json 2>/dev/null").stdout.split()
    scp(["/root/run/status.json", "/root/run/logs", *terminal], str(dest), upload=False)
    print(json.dumps({"runs": runs, "predictions": preds, "terminal": terminal}))


def job_key(name: str) -> str:
    """Job identity without ordering prefix, wave/index prefix, outcome suffix or extension."""
    name = re.sub(r"^p\d\d_", "", name)
    name = re.sub(r"^w\d+b?_\d+_", "", name)
    return re.sub(r"(\.try\d+)?(\.rc\d+)?\.json$", "", name)


def cmd_queue(_: argparse.Namespace) -> None:
    """Ship refill daemon + job files to the owned host and start the daemon (idempotent)."""
    jobs = sorted(JOBS_DIR.glob("*.json"))
    ssh("mkdir -p /root/run/refill/pending /root/run/refill/running /root/run/refill/done")
    scp([str(HERE / "remote_refill.py")], "/root/run/", upload=True)
    have = {job_key(name) for name in
            ssh("ls /root/run/refill/pending /root/run/refill/running /root/run/refill/done").stdout.split()}
    new = [str(j) for j in jobs if job_key(j.name) not in have]
    if new:
        scp(new, "/root/run/refill/pending/", upload=True)
    alive = ssh("pgrep -fc '^python remote_refill.py' || true").stdout.strip()
    if alive in ("", "0"):
        ssh("(cd /root/run && setsid nohup python remote_refill.py > refill.out 2>&1 < /dev/null &) ; echo started", timeout=30)
    print(json.dumps({"queued": [Path(n).name for n in new], "daemon_was_running": alive not in ("", "0")}))


FAMILIES = [
    ("fold", re.compile(r"vt_f[023]_")), ("sched", re.compile(r"sgdr|trap|wsd|lineard2z|ep6")),
    ("optim", re.compile(r"adamw|muonhidden|_mu[36]")), ("weighting", re.compile(r"lw15|lwfloor|cls3|since|proppow")),
    ("arch", re.compile(r"gated|tcn|structured|eventcontent|twoclock|dual|lstm|rawonly|encmlp|encconv|gru256x4|gru320x3|depth")),
    ("width", re.compile(r"gru(256x3|320|384|448|512)")), ("loss", re.compile(r"corr|huber|wmsegcorr")),
    ("regularize", re.compile(r"drop2|wd05|swa|b16|tr_lr|tr_base|seed|_ema|ctrl")), ("distill", re.compile(r"dsq")), ("init", re.compile(r"chrono|orth(?!ogonal)")), ("training", re.compile(r"chunk|lossmix|headlr")), ("data", re.compile(r"heur|onorm|thin")),
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


BACKLOG = ROOT / "research/campaign_2026-10-06/bandit_continuation/backlog/IDEA_BACKLOG.json"
LANE_ASSETS = {"a": {"fold0", "fold1", "fold2", "fold3", "teachers_f0"}, "b": {"fold0", "fold1", "fold2", "fold3"}}


def ready_items(lane: str) -> list[dict]:
    items = json.loads(BACKLOG.read_text())["items"]
    ok = [i for i in items if i["status"] == "ready" and lane in i["lanes"] and set(i["needs"]) <= LANE_ASSETS[lane]]
    return sorted(ok, key=lambda i: -i["priority"])


def cmd_feed(args: argparse.Namespace) -> None:
    """Move the best eligible backlog items into this lane's job directory, then queue and interleave."""
    lane = LANE or "a"
    sys.path.insert(0, str(HERE))
    import make_refill_jobs

    chosen = ready_items(lane)[: args.n]
    data = json.loads(BACKLOG.read_text())
    JOBS_DIR.mkdir(exist_ok=True)
    for item in chosen:
        job = make_refill_jobs.build_job(item["name"], item["change"], "bl")
        (JOBS_DIR / f"bl_{item['id']}.json").write_text(json.dumps(job, indent=1))
        for row in data["items"]:
            if row["id"] == item["id"]:
                row.update(status="queued", lane=lane, queued_utc=now())
    print(json.dumps({"lane": lane, "fed": [i["id"] for i in chosen], "est_hours": round(sum(i["hours"] for i in chosen), 1),
                      "ready_left": len(ready_items(lane)) - len(chosen)}))
    if args.dry_run or not chosen:
        for item in chosen:
            (JOBS_DIR / f"bl_{item['id']}.json").unlink(missing_ok=True)
        return
    BACKLOG.write_text(json.dumps(data, indent=1))
    cmd_queue(args)
    cmd_reorder(args)


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
        sizes = ssh(f"cd /root/wnn && stat -c %s cache/pred_{name}_valid.npy && ls artifacts/runs/{run}/model_ep*.pt | head -1").stdout
        if "model_ep" not in sizes:
            continue
        busy = ssh(f"pgrep -f '[p]redict_run.py.* {name} ' || true").stdout.strip()
        if busy:
            continue
        if not list((ROOT / "artifacts/runs" / run).glob("model_ep*.pt")):
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
echo "DISKFREE $(df -BG /root | tail -1 | awk '{print $4}' | tr -d G)"
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
    if not getattr(args, "no_pull", False):
        cmd_pull(args)
    hours, cost = spend()
    pending = int(info.get("PENDING", "1").split()[0])
    running = int(info.get("GPUPROCS", "1"))
    in_flight = int(info.get("PENDING", "1 RUNNING 1").split("RUNNING")[1].split()[0])
    primary_done = bool(info.get("TERMINAL", "").strip(", "))
    held = cost >= args.cap
    if held and pending > 0:
        ssh("mkdir -p /root/run/refill/held && mv /root/run/refill/pending/*.json /root/run/refill/held/ 2>/dev/null || true")
        print(f"spend cap ${args.cap} reached: queued jobs moved to held/, running jobs finish gracefully")
    remote_drained = (pending == 0 or held) and in_flight == 0 and running == 0 and primary_done
    collection_required = False
    collection_state = "awaiting_remote_drain"
    drained = False
    if remote_drained:
        collection_ok, collection_state = _collection_status(receipt())
        collection_required = not collection_ok
        drained = collection_ok
    print(json.dumps({"hours": round(hours, 2), "cost": round(cost, 2), "pending": pending, "gpu_procs": running,
                      "held": held, "drained": drained, "collection_required": collection_required,
                      "collection_status": collection_state, "gpu": info.get("GPU")}))
    if collection_required:
        print(f"collection required before teardown: {collection_state}")
    if not held and pending < 3 and ready_items(LANE or "a"):
        # the backlog must never run dry while eligible items exist (the optimizer alone would otherwise fill the lane)
        try:
            cmd_feed(argparse.Namespace(n=2, dry_run=False))
        except Exception as error:  # noqa: BLE001 - feeding must not break the unattended tick
            print(f"auto-feed failed: {type(error).__name__}: {str(error)[:200]}")
    keepalive = ROOT / ".runtime/hpo_keepalive"
    if drained and not held and keepalive.exists() and time.time() - keepalive.stat().st_mtime < 1800:  # a lane at its spend cap cannot take optimizer work
        print("drained but the hyper-parameter optimizer is active: keeping the instance")
        drained = False
    if drained:
        cmd_destroy(args)


def cmd_destroy(_: argparse.Namespace) -> None:
    data = receipt()
    collection_ok, collection_state = _collection_status(data)
    if not collection_ok:
        print(json.dumps({"collection_required": True, "collection_status": collection_state}))
        raise SystemExit(f"collection required before teardown: {collection_state}")
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
    for name in ("start", "status", "collect", "destroy", "queue", "pull", "dash", "tick", "reorder", "feed"):
        node = sub.add_parser(name)
        if name == "feed":
            node.add_argument("--n", type=int, default=4, help="backlog items to queue")
            node.add_argument("--dry-run", action="store_true")
        if name == "tick":
            node.add_argument("--no-pull", action="store_true", help="skip result pulls (a separate pull timer does them) so a slow scp cannot block cap/feed/drain")
            node.add_argument("--cap", type=float, default=5.0, help="stop admitting queued jobs at this spend; running jobs always finish, then the instance is destroyed")
    args = parser.parse_args()
    {"plan": cmd_plan, "create": cmd_create, "start": cmd_start, "status": cmd_status,
     "collect": cmd_collect, "destroy": cmd_destroy, "queue": cmd_queue, "pull": cmd_pull, "dash": cmd_dash, "tick": cmd_tick, "reorder": cmd_reorder, "feed": cmd_feed}[args.command](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
