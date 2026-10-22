"""Publish bounded PR37 wide status and prove only this cgroup's CUDA clients."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path("/data/epanov_s/tabdata")
PACKET = ROOT / "runtime/next4h-20261022-nllfixed-h256"
SIDE_PLAN = PACKET / "SIDEPLAN.json"
BASE_PACKET = ROOT / "runtime/next4h-20261022-r2"
BASE_STATE = ROOT / "state/next4h-20261022-nllfixed"
STATE = ROOT / "state/next4h-20261022-nllfixed-h256"
UNIT = "tabdata-next4h-20261022-nllfixed-h256.service"
MARKER = "<!-- tabdata-nllfixed-h256-20261022:PR37 -->"


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def progress(job: dict[str, Any], side: bool) -> str:
    if side:
        job_file = STATE / "jobs" / f"{job['id']}.json"
        if job_file.exists():
            record = read(job_file)
            phase = record.get("phase", "unknown")
        else:
            phase = "profiled/admission or not yet started"
        run_root = BASE_STATE / "nll_project/artifacts/runs" / job["id"]
    else:
        job_file = BASE_STATE / "jobs" / f"{job['id']}.json"
        record = read(job_file) if job_file.exists() else {}
        phase = record.get("phase", "not yet started")
        if job.get("family") == "nll":
            run_root = BASE_STATE / "nll_project/artifacts/runs" / job["id"]
        else:
            run_root = BASE_STATE / "numeric/runs_r2_e3" / f"f{job['fold']}_{job['arm']}"

    history = run_root / "history.json"
    if history.exists():
        try:
            return f"{phase}; {len(read(history))}/3 epoch records"
        except (OSError, ValueError, TypeError):
            pass
    for path in [*run_root.glob("metrics.jsonl"), *run_root.rglob("rank0.jsonl")]:
        if path.exists():
            try:
                rows = [
                    line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
                ]
                if rows:
                    last = json.loads(rows[-1])
                    step = last.get("global_step", last.get("step", last.get("update", "?")))
                    return f"{phase}; {len(rows)} rank0/metric records, last step {step}"
            except (OSError, ValueError, TypeError):
                return f"{phase}; progress log present"
    return f"{phase}; no epoch/update receipt yet"


def gpu_proof(final: bool) -> dict[str, Any]:
    import subprocess

    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
            "--format=csv,noheader",
        ],
        text=True,
        timeout=10,
    )
    owned = []
    for row in output.splitlines():
        try:
            pid = int(row.split(",", 1)[0].strip())
            group = Path(f"/proc/{pid}/cgroup").read_text(encoding="utf-8")
        except (ValueError, OSError):
            continue
        if UNIT in group:
            owned.append(row)
    return {"observed_unix": time.time(), "owned_cuda_clients": owned, "final": final}


def report(final: bool) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT / "runtime/github-reporting-v1/code"))
    from github_api import GitHub, read_token

    config = read(ROOT / "runtime/github-reporting-v1/CONFIG.json")
    api = GitHub(read_token(Path(config["token_file"])), request_limit=8)
    if api.request("GET", "/user")["login"] != "SemyonEpanov":
        raise ValueError("Unexpected GitHub publisher identity")
    proof = gpu_proof(final)
    plan = read(SIDE_PLAN)
    start = read(STATE / "START.json")
    lines = [
        MARKER,
        "### Additional cold-label NLL wide — " + ("final" if final else "progress update"),
        "",
        f"Side-plan SHA256 `{hashlib.sha256(SIDE_PLAN.read_bytes()).hexdigest()}`; base-plan SHA256 `{start['base_plan_sha256']}`.",
        "Recovery after SSH logout stopped the original attempt (Linger was disabled; now verified enabled). Original partial artifacts are preserved. These are fresh retries, not additional independent replications. Six numeric fits continue in a separate recovery service. This is a prospective H256 capacity arm of the cold-label experiment. Each cell reuses the matched H256 fold/seed; H256 results remain separately reported. Each matched NLL/WMSE pair uses the same fresh-init seed and source-bound fold; 3 strict-FP32 epochs, fixed epoch 2, H256x2 GRU and canonical 25-output head, full frozen training membership, and unchanged official grader.",
        f"The original node deadline remains `{start['deadline_unix']:.3f}`; profiles +5400s, fit +11100s, evaluation +12900s, grade +13140s. Scaleout service `{UNIT}` owns only its own CUDA work.",
        "",
        "#### Added H256 wide pairs",
        "| Pair | GPU | WMSE progress | NLL progress | Grade |",
        "|---|---:|---|---|---|",
    ]
    pairs: dict[tuple[int, int], dict[str, Any]] = {}
    for job in plan["jobs"]:
        pairs.setdefault((int(job["fold"]), int(job["seed"])), {})[job["arm"]] = job
    for (fold, seed), arms in sorted(pairs.items()):
        gpu = int(arms["nll"]["gpu_index"])
        wmse = progress(arms["wmse"], True)
        nll = progress(arms["nll"], True)
        grade_path = STATE / "grades" / f"pair_f{fold}_s{seed}.json"
        grade = "pending"
        if grade_path.exists():
            value = read(grade_path)
            panel = value["panels"][0]
            grade = f"ΔWP {panel['paired_component_bootstrap']['mean_diff']:.6g}"
        lines.append(f"| F{fold}/s{seed} | {gpu} | {wmse} | {nll} | {grade} |")
    lines += [
        "",
        f"Owned wide CUDA clients at observation: {len(proof['owned_cuda_clients'])}.",
        "Partial epochs are progress only; paired grades condition on these fixed fits and their component bootstrap does not estimate seed uncertainty. No public-score claim.",
        f"Logs, hashes, grades and cleanup receipt remain in `{STATE}`. Uniquely named wide checkpoints and predictions use the verified shared project `{BASE_STATE / 'nll_project'}`; existing run paths are rejected before admission.",
    ]
    if final and not proof["owned_cuda_clients"]:
        lines += [
            "",
            "Terminal cleanup proof: no CUDA clients remain in this wide systemd cgroup.",
        ]
    body = "\n".join(lines)
    outbox = STATE / "PR37_WIDE_LATEST.md"
    outbox.parent.mkdir(parents=True, exist_ok=True)
    outbox.write_text(body, encoding="utf-8")
    previous = [
        comment
        for comment in api.comments(37)
        if MARKER in comment.get("body", "")
        and comment.get("user", {}).get("login") == "SemyonEpanov"
    ]
    response = api.patch(previous[-1]["id"], body) if previous else api.post(37, body)
    result = {
        "url": response["html_url"],
        "body_sha256": hashlib.sha256(body.encode()).hexdigest(),
        "proof": proof,
        "created_unix": time.time(),
    }
    write(STATE / "PUBLICATION_LATEST.json", result)
    if final:
        write(STATE / "FINAL_GPU_PROOF.json", proof)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true")
    options = parser.parse_args()
    with (STATE / "publication.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        print(json.dumps(report(options.final)))


if __name__ == "__main__":
    main()
