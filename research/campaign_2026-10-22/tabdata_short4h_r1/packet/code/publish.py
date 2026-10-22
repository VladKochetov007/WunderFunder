from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path

from bootstrap import PACKET, ROOT, STATE


def report(final: bool) -> dict:
    """Publish bounded lifecycle and paired outcomes as SemyonEpanov, preserving an outbox."""
    sys.path.insert(0, str(ROOT / "runtime/github-reporting-v1/code"))
    from feature_common import read, sha, write
    from github_api import GitHub, read_token

    config = read(ROOT / "runtime/github-reporting-v1/CONFIG.json")
    api = GitHub(read_token(Path(config["token_file"])), request_limit=20)
    identity = api.request("GET", "/user")
    if identity["login"] != "SemyonEpanov":
        raise ValueError("Publisher is not the authorized account")
    plan = read(PACKET / "PLAN.json")
    own_gpu = []
    observed = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
            "--format=csv,noheader",
        ],
        text=True,
        timeout=10,
    )
    for row in observed.splitlines():
        pid = int(row.split(",")[0])
        try:
            group = Path(f"/proc/{pid}/cgroup").read_text()
        except FileNotFoundError:
            continue
        if "/tabdata-short4h-20261022-r1.service" in group:
            own_gpu.append(row)
    gpu_proof = dict(observed_unix=time.time(), owned_gpu_processes=own_gpu, final=final)
    write(STATE / "FINAL_GPU_PROOF.json" if final else STATE / "GPU_STATUS.json", gpu_proof)
    results = []
    for mechanism, number in [("age", 26), ("innovation", 27)]:
        jobs = [j for j in plan["jobs"] if j["mechanism"] == mechanism]
        start = read(STATE / "START.json") if (STATE / "START.json").exists() else None
        label = "Final bounded run" if final else "Bounded run status"
        lines = [
            f"<!-- tabdata-short4h-20261022-r1:PR{number} -->",
            f"### {label}: 24-slot / strict four-hour node allocation",
            "",
            f"Plan SHA256: `{sha(PACKET / 'PLAN.json')}`. Source `a266b036b02e8e012c6eb31072291dad618a169d`.",
            "Explicit new short-screen amendment; this does not complete or resume the historical CP30 panel.",
            "Two mechanisms × two fitting-data fidelities × three paired fold/seed cells × two arms = 24 planned fits total, 12 relevant to this PR.",
            "Fidelities: 10% / 25% complete fitting components (nested per fold), full 20k-row sequences, full unchanged exposed evaluation folds. Cells F0/s1201, F2/s1202, F3/s1203.",
            "Cold H512×1, strict FP32, autocast/TF32 off, label WMSE, AdamW/OneCycle, six epochs / fixed epoch5; no validation checkpoint selection.",
            f"GPU state: {len(own_gpu)} campaign CUDA clients at this observation. "
            + (
                "All campaign GPU computation is stopped."
                if final and not own_gpu
                else "Status is not terminal stop proof."
            ),
            "",
        ]
        if start:
            lines.append(
                f"Common start Unix {start['start_unix']:.3f}; absolute four-hour deadline {start['deadline_unix']:.3f}. Profiles count inside the allocation. RuntimeMaxSec=13500 / control-group stop, leaving time for publication."
            )
        counts = {
            stage: sum((STATE / "work" / j["id"] / file).exists() for j in jobs)
            for stage, file in [
                ("fit_integrity_accepted", "FIT.json"),
                ("exported", "EXPORTED.json"),
                ("failed_or_cutoff", "FAILURE.json"),
            ]
        }
        partial = []
        for job in jobs:
            h = STATE / "work" / job["id"] / "artifacts/runs" / job["id"] / "history.json"
            try:
                history = json.loads(h.read_text()) if h.exists() else []
            except json.JSONDecodeError:
                history = []
            if not (STATE / "work" / job["id"] / "EXPORTED.json").exists():
                partial.append(f"{job['id']}: {len(history)}/6 epoch records; partial/unscored")
        lines += [
            "",
            "Lifecycle counts: `" + json.dumps(counts, sort_keys=True) + "`.",
            "",
            "| Data fidelity | Fold / seed | Control WP | Candidate WP | Paired delta |",
            "|---|---|---:|---:|---:|",
        ]
        for gradefile in sorted((STATE / "scored").glob(mechanism + "_p*.json")):
            r = read(gradefile)
            lines.append(
                f"| {r['family']} | F{r['fold']} / {r['seed']} | {r['control_wp']:.8f} | {r['candidate_wp']:.8f} | {r['candidate_wp'] - r['control_wp']:+.8f} |"
            )
        lines += [
            "",
            "All individual deltas are exploratory development evidence, not public-transfer or significance claims. Fidelity panels share cells; they are not independent discoveries. Component intervals in original grade.json condition on fitted models.",
            "",
            "<details><summary>Partial / unscored endpoints</summary>",
            "",
            *partial,
            "",
            "</details>",
            "",
            f"Artifacts, commands, epochs, source/input hashes, negatives and exact failures are preserved at `{STATE}`. No datasets, private credentials or model weights are posted.",
        ]
        body = "\n".join(lines)
        outbox = STATE / f"PR{number}_LATEST.md"
        outbox.write_text(body)
        marker = lines[0]
        comments = api.comments(number)
        own = [
            c
            for c in comments
            if marker in c.get("body", "") and c.get("user", {}).get("login") == "SemyonEpanov"
        ]
        result = api.patch(own[-1]["id"], body) if own else api.post(number, body)
        results.append(
            dict(
                pr=number,
                url=result["html_url"],
                comment_id=result["id"],
                final=final,
                body_sha256=sha(outbox),
            )
        )
    write(
        STATE / "PUBLICATION_LATEST.json",
        dict(results=results, time_unix=time.time(), final=final, gpu_proof=gpu_proof),
    )
    return dict(results=results, owned_gpu_clients=len(own_gpu))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    with (STATE / "publication.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            print(json.dumps(report(args.final)))
        except Exception as error:
            from feature_common import write

            write(
                STATE / "PUBLICATION_FAILURE.json",
                dict(error=repr(error), observed_unix=time.time()),
            )
            raise
