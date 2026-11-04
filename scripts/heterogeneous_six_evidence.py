from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

if __package__:
    from scripts.heterogeneous_six_support import (
        Runtime,
        gpu_snapshot,
        legacy_record,
        ticks,
        write_json,
    )
else:
    from heterogeneous_six_support import Runtime, gpu_snapshot, legacy_record, ticks, write_json

from wnn.heterogeneous_pool_cache import validate_pool_cache
from wnn.heterogeneous_pool_sources import INPUT_FORMAT, read_object, validate_source
from wnn.splits import CampaignSplit, file_sha256


def arithmetic_fixture(cache: Path, output: Path, split: CampaignSplit) -> dict[str, object]:
    """Compare actual targets with independent float64 scalar arithmetic on fixed rows."""
    metadata = read_object(cache / "metadata.json")
    sources = metadata["sources"]
    qualified = [validate_source(record, split) for record in sources]
    rows = metadata["files"]["main_source_train"]["row_groups"]
    values = np.load(cache / "main_source_train.npy", mmap_mode="r", allow_pickle=False)
    samples = []
    for index in (0, len(rows) // 2, len(rows) - 1):
        for step in (0, 99, 999, 1000, 19999):
            raw = [
                array[position[rows[index]], step].astype(float).tolist()
                for array, position in qualified
            ]
            expected, bounds = [], []
            for target in (0, 1):
                terms = [
                    record["coefficient"] * prediction[target]
                    for record, prediction in zip(sources, raw, strict=True)
                ]
                expected.append(math.fsum(terms))
                bounds.append(
                    8 * float(np.finfo(np.float32).eps) * math.fsum(abs(term) for term in terms)
                    + 1e-7
                )
            actual = values[index, step].astype(float).tolist()
            errors = [abs(a - b) for a, b in zip(actual, expected, strict=True)]
            if any(error > bound for error, bound in zip(errors, bounds, strict=True)):
                raise ValueError("independent cache arithmetic check failed")
            samples.append(
                {
                    "group": rows[index],
                    "step": step,
                    "source_values": raw,
                    "stored": actual,
                    "independent_float64": expected,
                    "absolute_errors": errors,
                    "roundoff_bounds": bounds,
                }
            )
    result = {
        "schema": "wnn.heterogeneous-pool-arithmetic-fixture.v1",
        "profile": metadata["profile"],
        "metadata_sha256": file_sha256(cache / "metadata.json"),
        "coefficients": metadata["source_coefficients"],
        "samples": samples,
        "passed": True,
        "quality_claim": False,
    }
    write_json(output, result)
    return result


def external_status(root: Path, key: str, binding: dict[str, object] | None) -> dict[str, object]:
    """Adopt an existing owned phase only through its exact request/source/birth record."""
    status = read_object(root / "status" / f"{key}.json")
    request_path = root / "requests" / f"{key}.json"
    request = read_object(request_path)
    if (
        status.get("request_sha256") != file_sha256(request_path)
        or (
            binding is not None
            and status.get("source_identity") not in (binding, binding["sha256"])
        )
        or request.get("source_identity") != status.get("source_identity")
    ):
        raise ValueError(f"external owned request/source differs: {key}")
    if status["status"] in {"running", "launching"}:
        if ticks(status["runner_pid"]) not in (None, status["runner_start_ticks"]):
            raise ValueError(f"external runner PID was reused: {key}")
        if status.get("pid") and ticks(status["pid"]) not in (None, status.get("start_ticks")):
            raise ValueError(f"external child PID was reused: {key}")
    if status["status"] in {"failed", "timed_out", "interrupted"}:
        raise RuntimeError(f"external phase failed without retry: {key}")
    if status["status"] == "complete" and status.get("return_code") != 0:
        raise ValueError(f"external phase claims completion without exit0: {key}")
    return {"status": status, "request": request}


def adopt_owned_job(
    runtime: Runtime, key: str, script: str, args: list[str], gpu: int | None, phase: str, role: str
) -> dict[str, object]:
    """Adopt the preserved R1 child through its unchanged exact request and ownership."""
    observed = external_status(runtime.root, key, runtime.binding)
    request, status = observed["request"], observed["status"]
    affinity = "56-63" if gpu is None else f"{gpu * 8}-{gpu * 8 + 7}"
    command = [
        "taskset",
        "-c",
        affinity,
        str(runtime.python),
        "-u",
        str(runtime.source / "scripts" / script),
        *args,
    ]
    uuid = None if gpu is None else gpu_snapshot()[gpu]["uuid"]
    if (
        request.get("command") != command
        or request.get("cwd") != str(runtime.source)
        or request.get("phase") != phase
        or request.get("role") != role
        or status.get("phase") != phase
        or status.get("role") != role
        or status.get("gpu_uuid") != uuid
        or request.get("gpu_uuid") != uuid
        or request.get("env", {}).get("WNN_PROJECT_ROOT") != str(runtime.root)
        or request.get("env", {}).get("CUDA_VISIBLE_DEVICES") != (uuid or "")
        or status.get("command_sha256")
        != hashlib.sha256(json.dumps(command, separators=(",", ":")).encode()).hexdigest()
    ):
        raise ValueError(f"existing owned phase differs from the exact reviewed command: {key}")
    return {
        "status_path": str(runtime.root / "status" / f"{key}.json"),
        "request_path": str(runtime.root / "requests" / f"{key}.json"),
        "runner_pid": status["runner_pid"],
        "runner_start_ticks": status["runner_start_ticks"],
        "phase": phase,
        "role": role,
        "gpu": gpu,
        "adopted": True,
        "observed": status,
    }


def profile_gate(path: Path) -> dict[str, object]:
    """Require exactly80 finite successful updates from the declared disposable profile."""
    if path.stat().st_size > 64 << 20:
        raise ValueError("profile log exceeds its bound")
    lines = [
        line.partition(" ")[2]
        for line in path.read_text().splitlines()
        if line.startswith("PROFILE_UPDATE_EVIDENCE ")
    ]
    if len(lines) != 1:
        raise ValueError("profile lacks one unambiguous update receipt")
    evidence = json.loads(lines[0])
    if (
        evidence.get("valid_for_throughput") is not True
        or any(
            evidence.get(key) != 80
            for key in (
                "optimizer_attempts",
                "optimizer_completed_updates",
                "gradient_stats_steps",
                "loss_checks",
            )
        )
        or evidence.get("loss_finite") is not True
        or evidence.get("gradient_finite") is not True
    ):
        raise ValueError("profile is incomplete, nonfinite or changed from80updates")
    return evidence


def export_args(run: str, name: str, split: Path) -> list[str]:
    """Export the exact runtime model.pt endpoint with explicit feature-only coverage."""
    flag = (
        "--campaign-full-train-index-feature-only"
        if name == "train"
        else "--campaign-evaluation-only"
    )
    return [
        run,
        name,
        "--campaign-split-manifest",
        str(split),
        flag,
        "--inference-precision",
        "fp32",
    ]


def advance_cache(
    runtime: Runtime,
    state: dict,
    split: CampaignSplit,
    args: argparse.Namespace,
    nodes: list[dict],
    profile: str,
) -> None:
    """Advance only the declared control/heterogeneous cache dependencies."""
    root, binding = runtime.root, runtime.binding
    if profile in state["caches"]:
        return
    output = root / ("gru-only-targets" if profile == "gru-only" else "heterogeneous-targets")
    status = root / "status" / f"{args.control_cache_key}.json"
    if profile == "gru-only":
        if not status.exists():
            return
        if external_status(root, args.control_cache_key, binding)["status"]["status"] != "complete":
            return
    elif not output.exists():
        if profile == "heterogeneous" and (
            not all(f"export-{node['id']}-train-r1" in state["exports"] for node in nodes[:2])
            or not (args.galam_export / "source-record.json").exists()
        ):
            return
        records = [
            legacy_record(
                args.legacy_root,
                root / "controller-six-qualifications",
                f"tl3c_fin256_s{seed}",
                "gru",
                0.5 if profile == "gru-only" else 0.35,
                split,
            )
            for seed in (3, 4)
        ]
        if profile == "heterogeneous":
            external = root / "status/galam-fit-export-r1.json"
            if (
                not external.exists()
                or external_status(root, "galam-fit-export-r1", None)["status"]["status"]
                != "complete"
            ):
                return
            records += [read_object(args.galam_export / "source-record.json")]
            records += [
                legacy_record(
                    root,
                    root / "controller-six-qualifications",
                    node["run"],
                    family,
                    0.1,
                    split,
                )
                for node, family in zip(nodes[:2], ("lstm", "tcn"), strict=True)
            ]
        manifest = root / f"controller-six-{profile}-inputs.json"
        write_json(manifest, {"format": INPUT_FORMAT, "profile": profile, "sources": records})
        result = runtime.job(
            f"cache-{profile}-controller-r1",
            "build_heterogeneous_pool_cache.py",
            [
                "--inputs",
                str(manifest),
                "--data-root",
                str(root),
                "--split-manifest",
                str(args.split),
                "--output",
                str(output),
                "--min-free-gib",
                "12",
            ],
            gpu=None,
            phase="cache",
            role=profile,
            seconds=1800,
        )
        if result != "complete":
            return
    validated = validate_pool_cache(output, root, split, "source", require_families=False)
    arithmetic_fixture(output, root / "status" / f"{profile}-arithmetic-fixture.json", split)
    state["caches"][profile] = {
        "metadata_sha256": validated["metadata_sha256"],
        "verified_files": validated["verified_files"],
    }
