"""Receiving-host admission from actual check/profile evidence; no model imports."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from acceptance_receipts import validate_muon_receipt
from profile_analysis import timing_summary

ROOT = Path(__file__).resolve().parents[4]


def digest(value: object) -> str:
    """Match the runtime's canonical JSON identity."""
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()


def file_hash(path: Path) -> str:
    """Stream evidence hashes without reading model files into memory."""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def read(path: Path) -> dict[str, Any]:
    """Read a bounded receipt object."""
    if path.stat().st_size > 32_000_000:
        raise ValueError(f"Oversized receipt: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Receipt must be an object: {path}")
    return value


def require(condition: bool, reason: str) -> None:
    """Fail admission without assertions that optimization could disable."""
    if not condition:
        raise ValueError(reason)


def source_hash() -> str:
    """Match train_sequence's source identity without importing training code."""
    paths = sorted((ROOT / "src").rglob("*.py")) + [ROOT / "scripts/train_sequence.py"]
    return digest({p.relative_to(ROOT).as_posix(): file_hash(p) for p in paths})


def write_new(path: Path, value: dict[str, Any]) -> None:
    """Never replace an existing admission or its failure evidence."""
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def accept_checks(directory: Path) -> None:
    """Require actual completed pytest and two-rank objective receipts."""
    xml = ET.parse(directory / "pytest.xml").getroot()
    cases = list(xml.iter("testcase"))
    require(bool(cases), "No pytest cases were executed")
    required = {
        "test_transformer_epoch_checkpoint_resume_matches_uninterrupted_fit[adamw]",
        "test_transformer_epoch_checkpoint_resume_matches_uninterrupted_fit[muon]",
    }
    require(required <= {case.get("name") for case in cases}, "Both CUDA recovery paths must run")
    require(
        not any(list(xml.iter(tag)) for tag in ("failure", "error", "skipped")),
        "Receiver checks failed or were skipped (CUDA recovery must execute)",
    )
    ddp = read(directory / "ddp.json")
    require(ddp.get("passed") is True and ddp.get("world_size") == 2, "DDP check failed")
    require(
        bool(ddp.get("cases")) and all(case.get("passed") is True for case in ddp["cases"]),
        "DDP cases are incomplete",
    )
    muon = read(directory / "muon-ddp.json")
    validate_muon_receipt(muon)
    require(
        (muon["torch"], muon["cuda_runtime"]) == (ddp["torch"], ddp["cuda_runtime"]),
        "Check software identities differ",
    )
    paths = [
        directory / name
        for name in (
            "pytest.xml",
            "pytest.log",
            "ddp.json",
            "ddp.log",
            "muon-ddp.json",
            "muon-ddp.log",
            "nvidia-smi.txt",
            "topology.txt",
        )
    ]
    paths += [
        ROOT / "tests" / name
        for name in (
            "test_b300_sequence_models.py",
            "test_b300_sequence_contracts.py",
            "test_large_teacher_runtime_guards.py",
            "test_sequence_full_fit_data.py",
            "test_sequence_recovery_cuda.py",
            "test_sequence_muon.py",
        )
    ]
    paths += [
        ROOT / "scripts" / name for name in ("check_sequence_ddp.py", "check_sequence_muon_ddp.py")
    ]
    write_new(
        directory / "acceptance.json",
        {
            "schema": "wnn.large-teacher-checks.v1",
            "status": "passed",
            "source_sha256": source_hash(),
            "files": {str(p.resolve()): file_hash(p) for p in paths},
            "pytest_cases": len(cases),
            "torch": ddp["torch"],
            "cuda_runtime": ddp["cuda_runtime"],
            "limits": "Synthetic correctness/recovery checks; no predictive or full-shape performance result.",
        },
    )


def checked_profile(path: Path, config: dict[str, Any], world: int) -> dict[str, Any]:
    """Check that a complete bounded profile belongs to this exact source/config."""
    value = read(path)
    identity = value["identity"]
    require(
        value.get("status") == "profile_complete" and value.get("updates") == 12,
        "Need twelve completed engineering updates",
    )
    require(value.get("errors") == [] and value.get("error_count") == 0, "Profile has errors")
    require(value.get("scientific_checkpoint_written") is False, "Profile wrote scientific weights")
    require(identity["source_sha256"] == source_hash(), "Profile source differs from checkout")
    require(
        identity["config_sha256"] == digest(config), "Profile config differs from frozen recipe"
    )
    require(identity["world_size"] == world, "Wrong profile world size")
    require(identity["data"]["fold"] == config["expected_fold"], "Wrong profile fold")
    require(all(math.isfinite(x) and x >= 0 for x in value["losses"].values()), "Nonfinite losses")
    memory = value["memory_by_rank"]
    require([item["rank"] for item in memory] == list(range(world)), "Missing rank memory records")
    for item in memory:
        require(
            0
            < item["peak_allocated_bytes"]
            <= item["peak_reserved_bytes"]
            <= 0.8 * item["device_total_memory_bytes"],
            "Less than 20% GPU memory headroom",
        )
    return value


def admit(args: argparse.Namespace) -> None:
    """Publish a reviewed, source-bound eight-rank profile only after all checks pass."""
    config = read(args.config)
    require(bool(args.reviewer.strip()), "A receiving reviewer is required")
    require(math.isfinite(args.host_ram_gib) and args.host_ram_gib > 0, "Invalid host RAM budget")
    checks = read(args.checks)
    require(
        checks.get("status") == "passed" and checks["source_sha256"] == source_hash(),
        "Missing checks for current runtime",
    )
    for name, expected in checks["files"].items():
        require(file_hash(Path(name)) == expected, f"Acceptance evidence changed: {name}")
    one_path = args.directory / "profile-one/profile_receipt.json"
    eight_path = args.directory / "profile-eight/profile_receipt.json"
    one = checked_profile(one_path, config | {"required_world_size": 1}, 1)
    eight = checked_profile(eight_path, config, 8)
    peak_path = args.directory / "profile-peak/profile_receipt.json"
    peak = checked_profile(peak_path, config | {"warmup_fraction": 0.0}, 8)
    require(
        peak["identity"]["data_sha256"] == eight["identity"]["data_sha256"],
        "Peak stress and production data differ",
    )
    require(
        peak["identity"]["runtime"] == eight["identity"]["runtime"],
        "Peak stress and production hardware/software differ",
    )
    for rank in range(8):
        records = [
            json.loads(line)
            for line in (args.directory / "profile-peak" / f"rank{rank}.jsonl")
            .read_text()
            .splitlines()
        ]
        require([row["step"] for row in records] == list(range(1, 13)), "Incomplete peak stress")
        require(
            math.isclose(records[0]["learning_rate"], config["lr"], rel_tol=1e-10),
            "Peak stress did not start at peak LR",
        )
        require(
            all(
                math.isfinite(row["gradient_norm_preclip"])
                and row["learning_rate"] >= 0.95 * config["lr"]
                for row in records
            ),
            "Peak stress has invalid gradients or non-peak learning rate",
        )
    require(
        one["identity"]["data_sha256"] == eight["identity"]["data_sha256"],
        "One- and eight-GPU data/preprocessing differ",
    )
    runtime = eight["identity"]["runtime"]
    require(
        runtime["torch"] == checks["torch"] and runtime["cuda_runtime"] == checks["cuda_runtime"],
        "Check and fit runtimes differ",
    )
    require(all("B300" in item["name"] for item in runtime["devices"]), "B300 hardware required")
    parameters = eight["model_parameters"]["total"]
    # Conservative upper bound: model + up to two FP32 optimizer slots per host rank.
    host_floor = (
        sum(row["host_peak_rss_bytes"] for row in eight["memory_by_rank"]) + 12 * parameters * 8
    )
    require(
        host_floor <= 0.8 * args.host_ram_gib * 2**30,
        "Host budget cannot cover conservative all-rank resume staging plus 20% headroom",
    )
    data = eight["identity"]["data"]
    disk_floor = 32 * parameters + 160_000 * (
        len(data["fit_members"]) + len(data["evaluation_members"])
    )
    require(
        shutil.disk_usage(args.directory).free >= disk_floor,
        "Insufficient scratch for checkpoints, separate archive and prediction/cache arrays",
    )
    eight.update(
        approved=True,
        admission={
            "reviewer": args.reviewer,
            "checks_sha256": file_hash(args.checks),
            "raw_profile_sha256": file_hash(eight_path),
            "one_gpu_profile_sha256": file_hash(one_path),
            "peak_stress_profile_sha256": file_hash(peak_path),
            "peak_rank_logs_sha256": {
                f"rank{rank}": file_hash(args.directory / "profile-peak" / f"rank{rank}.jsonl")
                for rank in range(8)
            },
            "host_ram_budget_gib": args.host_ram_gib,
            "resume_host_floor_bytes": host_floor,
            "cell_scratch_floor_bytes": disk_floor,
            "timing": timing_summary(args.directory, config, eight),
            "limits": "Resource estimates exclude other jobs and non-tensor overhead. No quality approval.",
        },
    )
    write_new(args.directory / "approved_profile.json", eight)


def main() -> None:
    """Dispatch explicit receiving-host evidence stages."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    checks = sub.add_parser("checks")
    checks.add_argument("--directory", type=Path, required=True)
    profile = sub.add_parser("profile")
    for name in ("directory", "checks", "config"):
        profile.add_argument(f"--{name}", type=Path, required=True)
    profile.add_argument("--reviewer", required=True)
    profile.add_argument("--host-ram-gib", required=True, type=float)
    args = parser.parse_args()
    if args.stage == "checks":
        accept_checks(args.directory)
    else:
        admit(args)


if __name__ == "__main__":
    main()
