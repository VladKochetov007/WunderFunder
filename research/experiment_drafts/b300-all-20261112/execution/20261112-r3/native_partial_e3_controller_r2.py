from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

END = 1794471619.060462
ROOT = Path("/data/epanov_s/tabdata-all-20261112-r1")
CELLS = (
    "ip01_control_s204101",
    "ip01_treatment_s204101",
    "ip02_control_s204101",
    "ip03_control_s204101",
    "ip03_treatment_s204101",
    "ip04_control_s204101",
    "ip04_treatment_s204101",
    "ip07_treatment_s204101",
)
UUIDS = (
    "daf615f5-ea55-860a-bfe1-41c36ab96c3e",
    "25ae492c-db98-e1ae-0f5e-c56365a422c7",
    "df6fe7d0-64e5-7af2-467d-c1699cb6fc67",
    "76951d88-add3-0d6b-5bee-c0390915ca47",
    "e6d65cb1-aba4-14f5-d4e5-bf24ef1ee3d8",
    "1fc4d921-dc78-6711-c2a8-ec0589b7f106",
    "2913ac69-eccb-63b0-1112-e09ea9598b57",
    "be85f55c-71a1-f389-eeed-55c63a0e3ce7",
)


def main() -> None:
    """Run eight bounded inference-only readers after the original fit queue stops."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--adapter-sha256", required=True)
    args = parser.parse_args()
    if hashlib.sha256(args.adapter.read_bytes()).hexdigest() != args.adapter_sha256:
        raise ValueError("retained reader source changed")
    queue = json.loads(
        (ROOT / "control/impact81-transformer-queue-r2/status.json").read_bytes()
    )
    if queue["status"] != "completed_with_cell_outcomes":
        raise ValueError("original fit queue has not settled")
    if time.time() >= END - 300:
        raise TimeoutError("no new retained reader starts in the final five minutes")
    sys.path.insert(
        0, "/data/epanov_s/tabdata-qknorm-b300-20261111-r1/control/profile-receiver-r1"
    )
    import b300_driver_core_r1 as core
    import b300_driver_support_r1 as support
    import b300_owned_runner_r2 as owned

    control = ROOT / "control/partial-e3-readout-r1"
    control.mkdir(parents=True, exist_ok=False)
    owner = support._identity(ROOT)
    core.atomic_json(control / "owner.json", owner)
    guard = owned.ResourceGuard(
        ROOT,
        control,
        {
            "owned_rss_bytes": 512 * 1024**3,
            "output_bytes": 512 * 1024**3,
            "minimum_free_bytes": 64 * 1024**3,
        },
    )
    packet = ROOT / "packet-r5-transformer/research/experiment_drafts/b300-all-20261112"
    runtime = ROOT / "runtimes/pr75"
    python = "/data/epanov_s/tabdata/.venv/bin/python"
    clock = args.adapter.parent / "CLOCK_B300_R2.json"

    def run(cell: str, uuid: tuple[str, ...]) -> dict:
        """Keep each report separate from fixed E4 artifacts and retain failure receipts."""
        output = ROOT / "runs" / cell / "readout-e3-r2"
        command = [
            python,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc-per-node=4",
            str(args.adapter),
            "--packet-root",
            str(packet),
            "--runtime-root",
            str(runtime),
            "--cell",
            str(packet / "configs" / f"{cell}.json"),
            "--fit-dir",
            str(ROOT / "runs" / cell / "fit"),
            "--data-root",
            "/data/epanov_s/tabdata-qknorm-b300-20261111-r1/project",
            "--split-manifest",
            "/data/epanov_s/tabdata-qknorm-b300-20261111-r1/project/chronological_primary.json",
            "--clock",
            str(clock),
            "--output",
            str(output),
            "--batch-sequences",
            "32",
            "--owner",
            "SemyonEpanov-root-e3-report-only-20261112",
        ]
        result = {
            "cell": cell,
            "gpu_uuid": uuid,
            "output": str(output),
            "adapter_sha256": args.adapter_sha256,
            "fixed_endpoint_epoch": 4,
            "retained_epoch": 3,
        }
        try:
            if output.exists():
                raise FileExistsError("preserving existing retained report output")
            env = support._environment(runtime, ROOT, list(uuid))
            env["PATH"] = (
                str(ROOT / "tools/git-r2/usr/bin") + os.pathsep + env.get("PATH", "")
            )
            owned.run_owned(
                command,
                root=runtime,
                control=control,
                cell=cell,
                stage="e3-readout",
                gpu_uuids=list(uuid),
                owner=owner,
                environment=env,
                deadline=END - 35,
                stage_timeout=max(1, int(END - 35 - time.time())),
                guard=guard,
                progress=lambda: None,
            )
            report = json.loads((output / "report_only.json").read_bytes())
            result.update(status="report_only_complete", report=report)
        except (owned.OwnedRunError, OSError, ValueError, KeyError) as error:
            result.update(
                status="report_only_failed_or_partial",
                error=f"{type(error).__name__}: {error}",
            )
        core.atomic_json(control / f"{cell}-result.json", result)
        return result

    results = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [
            pool.submit(run, cell, uuid)
            for cell, uuid in zip(CELLS[:2], (UUIDS[:4], UUIDS[4:]), strict=True)
        ]
        for job in as_completed(jobs):
            results.append(job.result())
    core.atomic_json(
        control / "complete.json",
        {
            "at": time.time(),
            "status": "settled",
            "inference_only": True,
            "cells": results,
        },
    )


if __name__ == "__main__":
    main()
