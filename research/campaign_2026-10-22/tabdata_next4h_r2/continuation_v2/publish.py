"""Idempotent GitHub updates with observed progress and terminal cgroup proof."""

from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path

from delivery import PACKET, ROOT, STATE, UNIT, read, sha, write


def gpu_proof(final: bool) -> dict:
    """Identify only this exact unit's CUDA clients; never signal foreign jobs."""
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
        pid = int(row.split(",")[0])
        try:
            group = Path(f"/proc/{pid}/cgroup").read_text()
        except FileNotFoundError:
            continue
        if "/" + UNIT in group:
            owned.append(row)
    result = {"observed_unix": time.time(), "owned_cuda_clients": owned, "final": final}
    write(STATE / ("FINAL_GPU_PROOF.json" if final else "GPU_STATUS.json"), result)
    return result


def partial(job: dict) -> str:
    """Read checkpoint-independent epoch/update progress without executing models."""
    if job["family"] == "nll":
        path = STATE / "nll_project/artifacts/runs" / job["id"] / "history.json"
        if path.exists():
            return f"{len(read(path))}/3 epoch records"
    folder = STATE / "numeric/runs_r2_e3" / f"f{job['fold']}_{job['arm']}"
    log = folder / "metrics.jsonl"
    if log.exists():
        return f"{len(log.read_text().splitlines())} metric records; inspect safe-epoch state"
    return "no completed-epoch receipt yet"


def report(final: bool) -> dict:
    """Update each authorized PR as SemyonEpanov and preserve a local outbox."""
    sys.path.insert(0, str(ROOT / "runtime/github-reporting-v1/code"))
    from github_api import GitHub, read_token

    config = read(ROOT / "runtime/github-reporting-v1/CONFIG.json")
    api = GitHub(read_token(Path(config["token_file"])), request_limit=20)
    if api.request("GET", "/user")["login"] != "SemyonEpanov":
        raise ValueError("Unexpected GitHub publisher identity")
    proof = gpu_proof(final)
    plan = read(PACKET / "PLAN.json")
    start = read(STATE / "START.json") if (STATE / "START.json").exists() else None
    results = []
    for family, number in [("numeric", 35), ("nll", 37), ("mask", 36)]:
        jobs = [j for j in plan["jobs"] if j["family"] == family]
        marker = f"<!-- tabdata-next4h-20261022-r2:PR{number} -->"
        lines = [
            marker,
            "### " + ("Final bounded execution" if final else "Bounded execution update"),
            "",
            f"Plan SHA256 `{sha(PACKET / 'PLAN.json')}`. Source commits: numeric `20faaa1ab`, NLL `6577b88e` (test-only fix over `643c5cc5`), mask `eceb95c0`.",
            "New three-epoch amendments; these results do not complete the original long panels.",
            "Ten planned scientific fits: six Transformer input-encoding fits on GPUs2–7, four cold GRU WMSE/NLL fits as two workers each on GPUs0–1. All use strict FP32; no autocast, TF32, quantization, teacher or warm-parent substitution.",
            "All source-qualified fitting components, complete 20k-row sequences and full declared held-out evaluation; folds remain exposed development evidence.",
            f"Owned CUDA clients at this observation: {len(proof['owned_cuda_clients'])}. "
            + (
                "All CUDA clients owned by this service have stopped."
                if final and not proof["owned_cuda_clients"]
                else "This is not terminal cleanup proof."
            ),
            "",
        ]
        if start:
            lines += [
                f"Start Unix `{start['start_unix']:.3f}`; absolute four-hour deadline `{start['deadline_unix']:.3f}`. Preparation and profiles count inside this wall. systemd RuntimeMaxSec=13500, KillMode=control-group, bounded stop/finalizer. Fit cutoff +11100s, export +12900s, grade +13140s.",
                "",
            ]
        if family == "mask":
            lines += [
                "**Not admitted; zero new PR36 fits.** Exact pinned parent checkpoint/provenance assets were not found under the project. No parent replacement or new parent fitting was performed. The prepared R2 remains available separately."
            ]
        else:
            lines += ["| Cell | GPU | Phase | Progress |", "|---|---:|---|---|"]
            for job in jobs:
                path = STATE / "jobs" / (job["id"] + ".json")
                status = read(path) if path.exists() else {"phase": "preparing_or_profiling"}
                lines.append(
                    f"| F{job['fold']}/s{job['seed']} {job['arm']} | {job['gpu_index']} | {status['phase']} | {partial(job)} |"
                )
                if status.get("error"):
                    lines += ["", f"Failure `{job['id']}`: `{status['error']}`", ""]
            if family == "numeric":
                lines += ["", "| Fold | Arm | Official WP |", "|---|---|---:|"]
                for job in jobs:
                    path = (
                        STATE / "numeric/runs_r2_e3" / f"f{job['fold']}_{job['arm']}" / "score.json"
                    )
                    if path.exists():
                        value = read(path)
                        lines.append(
                            f"| F{job['fold']} | {job['arm']} | {value['score']['wp']:.9f} |"
                        )
                panel = STATE / "numeric/panel_r2_e3.json"
                if panel.exists():
                    lines += [
                        "",
                        "Complete paired panel:",
                        "```json",
                        json.dumps(read(panel)["contrasts"], indent=2),
                        "```",
                    ]
            else:
                for fold in (0, 2):
                    path = STATE / f"nll_grade_f{fold}.json"
                    if path.exists():
                        lines += [
                            "",
                            f"F{fold} paired official grade:",
                            "```json",
                            json.dumps(read(path), indent=2)[:12000],
                            "```",
                        ]
        for name in (
            "PROFILES.json",
            "ADMISSION.json",
            "FAILURE.json",
            "NUMERIC_GRADE_FAILURE.json",
            "NLL_GRADE_FAILURE_F0.json",
            "NLL_GRADE_FAILURE_F2.json",
        ):
            path = STATE / name
            if path.exists():
                record = read(path)
                if name == "PROFILES.json":
                    record = {
                        "profile_failures": record.get("failures", {}),
                        "profiles_passed": len(record.get("results", {})),
                    }
                lines += [
                    "",
                    name + ":",
                    "```json",
                    json.dumps(record, indent=2)[:6000],
                    "```",
                ]
        lines += [
            "",
            "Partial epochs are progress only, not completed fixed-epoch comparisons. Component intervals condition on fitted models and exclude seed/unseen-fold uncertainty. No public-score or significance claim.",
            f"Commands, logs, partial checkpoints, exact source/input hashes, outcomes and cleanup evidence are retained at `{STATE}`. Credentials, datasets and weights are not posted.",
        ]
        body = "\n".join(lines)
        outbox = STATE / f"PR{number}_LATEST.md"
        outbox.write_text(body)
        previous = [
            c
            for c in api.comments(number)
            if marker in c.get("body", "") and c.get("user", {}).get("login") == "SemyonEpanov"
        ]
        response = api.patch(previous[-1]["id"], body) if previous else api.post(number, body)
        results.append(
            {
                "pr": number,
                "url": response["html_url"],
                "body_sha256": sha(outbox),
                "final": final,
            }
        )
    write(
        STATE / "PUBLICATION_LATEST.json",
        {"results": results, "gpu_proof": proof, "observed_unix": time.time()},
    )
    return {"results": results, "owned_cuda_clients": len(proof["owned_cuda_clients"])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true")
    options = parser.parse_args()
    with (STATE / "publication.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        print(json.dumps(report(options.final)))
