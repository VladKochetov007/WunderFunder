"""Kaggle 2xT4 long-run worker: stage mounted inputs into the trainer layout, run the packaged jobs (one per GPU), keep outputs."""
import glob, json, os, shutil, subprocess, sys, tarfile, threading, time, traceback
from pathlib import Path

OUT = Path("/kaggle/working/out"); OUT.mkdir(parents=True, exist_ok=True)
W = Path("/root/wnn"); W.mkdir(parents=True, exist_ok=True)
STATUS = OUT / "status.json"
state = {"phase": "start", "t0": time.time(), "jobs": {}}


def save(**kw):
    state.update(kw); state["updated"] = time.time(); STATUS.write_text(json.dumps(state, indent=1, default=str))


def find(name, under="/kaggle/input"):
    hits = glob.glob(f"{under}/**/{name}", recursive=True)
    if not hits:
        raise FileNotFoundError(name)
    return hits[0]


def main():
    save(phase="locate", inputs=sorted(os.listdir("/kaggle/input")))
    bundle = find("bundle.tar.gz")
    with tarfile.open(bundle) as t:
        t.extractall(W)
    jobs = json.loads((W / "kaggle_jobs.json").read_text())
    amap = {"cache/chain_folds.npz": "a2cfe843a7ce3085d383524a9d1438154fba4f2e0af6df48ee1e1b015e5c942f.wnn", "cache/chains.npz": "39aba46fad7b717830e90687e1c4a6991794864cd2cf621b0bbf341f2d9a7766.wnn", "cache/train_targets.npz": "ead1a05e7aa5ac3d1c10cde65986be9f48fd342cadb95330611e58c3a8f817e9.wnn", "cache/valid_targets.npz": "dbb7a1a48dd0af9fa524a80c2694008a56cb91d5dae80e8d0d4984c0c5622955.wnn"}
    (W / "datasets").mkdir(exist_ok=True); (W / "cache").mkdir(exist_ok=True)
    for name in ("train.parquet", "valid.parquet", "valid_mask.parquet"):
        link = W / "datasets" / name
        if not link.exists():
            link.symlink_to(find(name))
    for remote, fname in amap.items():
        link = W / remote
        if not link.exists():
            link.symlink_to(find(fname))
    save(phase="gpus", gpus=subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout)
    env = os.environ | {"WNN_PROJECT_ROOT": str(W), "PYTHONPATH": str(W / "src"), "PYTHONUNBUFFERED": "1", **jobs.get("env", {})}
    logs = OUT / "logs"; logs.mkdir(exist_ok=True)

    def run(argv, log, e, timeout):
        with (logs / log).open("ab") as h:
            p = subprocess.Popen(argv, stdout=h, stderr=subprocess.STDOUT, env=e, cwd=W)
            try:
                return p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                p.terminate()
                try:
                    p.wait(120)
                except subprocess.TimeoutExpired:
                    p.kill()
                return 124

    save(phase="prepare")
    for step in jobs["prepare"]:
        rc = run(step["argv"], f"prepare.{step['label']}.log", env | {"CUDA_VISIBLE_DEVICES": "0"}, step["timeout"])
        state.setdefault("prepare_rc", {})[step["label"]] = rc; save()
    queue = list(jobs["jobs"]); lock = threading.Lock()

    def worker(slot):
        while True:
            with lock:
                if not queue:
                    return
                job = queue.pop(0)
            e = env | {"CUDA_VISIBLE_DEVICES": str(slot["gpu"]), "OMP_NUM_THREADS": str(slot["threads"]), "WNN_THREADS": str(slot["threads"])}
            rec = state["jobs"].setdefault(job["name"], {})
            for step in job["steps"]:
                t = time.time()
                rec["state"] = step["label"]; save()
                rc = run(step["argv"], f"{job['name']}.{step['label']}.log", e, step["timeout"])
                rec[f"{step['label']}_rc"] = rc; rec[f"{step['label']}_s"] = round(time.time() - t); save()
                if rc not in (0,) and step["label"] == "fit":
                    break  # still export whatever epoch exists is skipped; checkpoints are collected below
            rec["state"] = "finished"; save()

    save(phase="jobs")
    threads = []
    for i, slot in enumerate(jobs["slots"]):
        th = threading.Thread(target=worker, args=(slot,)); th.start(); threads.append(th); time.sleep(jobs.get("stagger_seconds", 30))
    for th in threads:
        th.join()
    save(phase="collect")
    runs = OUT / "runs"; runs.mkdir(exist_ok=True)
    for job in jobs["jobs"]:
        src = W / "artifacts/runs" / job["name"]
        if src.exists():
            shutil.copytree(src, runs / job["name"], dirs_exist_ok=True, ignore=shutil.ignore_patterns("training_state.pt"))
    for p in glob.glob(str(W / "cache/pred_kg_*")):
        shutil.copy(p, OUT)
    save(phase="done")


try:
    main()
except Exception:
    save(phase="failed", error=traceback.format_exc()[-3000:])
    raise
