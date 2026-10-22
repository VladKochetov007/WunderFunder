"""Verify the R4 source amendment before running the unchanged F1 profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
AMENDMENT_REL = "research/campaign_2026-10-21/next_cycle_r1/i1_warm_history_extension_r1/PROFILE_F1_SOURCE_AMENDMENT_R4.json"
ADMISSION_REL = "research/campaign_2026-10-21/next_cycle_r1/WARM_F1_PROFILE_ADMISSION_R4.json"
RECEIPT_REL = "research/campaign_2026-10-21/next_cycle_r1/i1_warm_history_extension_r1/PROFILE_F1_LOCAL_EXECUTION_R4.json"
LOG_REL = "research/campaign_2026-10-21/next_cycle_r1/warm_f1_profile_r4.log"


def sha256(path: Path) -> str:
    """Hash source and receipt files with bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_object(path: Path) -> dict[str, Any]:
    """Read a small JSON manifest after checking its size and stable length."""
    size = path.stat().st_size
    if size <= 0 or size > 1_000_000:
        raise ValueError(f"manifest size is outside bounds: {path} ({size})")
    raw = path.read_bytes()
    if len(raw) != size:
        raise ValueError(f"manifest changed while reading: {path}")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def check_map(root: Path, pins: dict[str, str]) -> dict[str, str]:
    """Verify repository-relative file digests under one pinned root."""
    root = root.resolve(strict=True)
    checked: dict[str, str] = {}
    for name, expected in pins.items():
        path = (root / name).resolve(strict=True)
        if root not in path.parents or sha256(path) != expected:
            raise ValueError(f"source or data pin differs: {name}")
        checked[name] = expected
    return checked


def git(source: Path, *args: str) -> str:
    """Read a small identity string from a source checkout."""
    return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()


def main() -> None:
    """Fail closed on amended source, historical identity and output paths."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--source-amendment", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()

    amendment_path = args.source_amendment.resolve(strict=True)
    admission_path = args.admission.resolve(strict=True)
    if amendment_path != ROOT / AMENDMENT_REL or admission_path != ROOT / ADMISSION_REL:
        raise ValueError("R4 amendment/admission path differs from the indexed packet")
    amendment = json_object(amendment_path)
    admission = json_object(admission_path)
    if sha256(amendment_path) != admission["source_amendment_sha256"]:
        raise ValueError("source amendment digest differs from admission")
    if sha256(Path(__file__)) != admission["adapter_sha256"]:
        raise ValueError("R4 adapter digest differs from admission")
    if (
        admission["status"] not in {"prepared_for_root_review_not_run", "admitted_for_execution"}
        or admission["admission_id"] != "wnn-warm-f1-profile-r4-20261022"
        or admission["source_commit"] != args.source_sha
        or admission["source_root"] != amendment["source"]["root"]
        or admission["profile_id"] != amendment["profile_id"]
        or admission["plan_path"] != amendment["base_plan"]["path"]
        or admission["plan_sha256"] != amendment["base_plan"]["sha256"]
        or admission["profile_recipe_path"] != amendment["base_profile_recipe"]["path"]
        or admission["profile_recipe_sha256"] != amendment["base_profile_recipe"]["sha256"]
        or admission["unchanged_argv_sha256"] != amendment["profile"]["unchanged_argv_sha256"]
        or admission["fold"] != amendment["profile"]["fold"]
        or admission["profile_warmup_batches"] != amendment["profile"]["profile_warmup"]
        or admission["profile_measured_batches"] != amendment["profile"]["profile_steps"]
        or admission["expected_optimizer_updates"]
        != amendment["profile"]["expected_optimizer_updates"]
        or admission["receipt_path"] != RECEIPT_REL
        or admission["log_path"] != LOG_REL
        or admission["output_run_path"] != "artifacts/runs/i1warm_profile_f1_seed4_r2"
    ):
        raise ValueError("R4 admission fields differ from source amendment")
    limits = admission["resource_limits"]
    if (
        limits["cpu_quota_percent"] != 100
        or limits["ram_bytes"] != 4_000_000_000
        or limits["swap_bytes"] != 0
        or limits["gpu_allocator_bytes"] != 4_000_000_000
        or limits["vram_stop_bytes"] != 5_000_000_000
        or limits["max_wall_seconds"] != 900
        or limits["loader_workers"] != 0
        or limits["physical_disk_scratch"] is not True
    ):
        raise ValueError("R4 resource contract differs from the bounded admission")
    runtime = admission["runtime_environment"]
    if (
        runtime["CUDA_VISIBLE_DEVICES"] != "0"
        or runtime["WNN_GPU_MEMORY_BYTES"] != "4000000000"
        or runtime["WNN_THREADS"] != "1"
        or runtime["WNN_LOADER_WORKERS"] != "0"
        or runtime["WNN_LOADER_RAM_GB"] != "0.5"
        or not Path(runtime["TMPDIR"])
        .resolve()
        .is_relative_to(Path(admission["scratch_root"]).resolve(strict=True))
    ):
        raise ValueError("R4 runtime environment differs from the admitted limits")

    source = amendment["source"]
    source_root = Path(source["root"]).resolve(strict=True)
    if (
        args.source_sha != source["commit"]
        or git(source_root, "rev-parse", "HEAD") != args.source_sha
    ):
        raise ValueError("required source SHA does not match the clean frozen source")
    if git(source_root, "status", "--porcelain"):
        raise ValueError("repaired source worktree is dirty")
    prior_root = Path(source["prior_source_root"]).resolve(strict=True)
    if git(prior_root, "rev-parse", "HEAD") != source["prior_source_commit"]:
        raise ValueError("preserved R3 source commit changed")
    if (
        git(prior_root, "status", "--porcelain")
        or sha256(prior_root / "src/wnn/distillation_cache.py")
        != source["prior_cache_reader_sha256"]
    ):
        raise ValueError("preserved R3 source is dirty or its old cache reader changed")
    changed = git(
        source_root,
        "diff",
        "--name-only",
        f"{source['prior_source_commit']}..{args.source_sha}",
    ).splitlines()
    if changed != ["src/wnn/distillation_cache.py"]:
        raise ValueError(f"source amendment changed unexpected paths: {changed}")

    plan_path = ROOT / amendment["base_plan"]["path"]
    recipe_path = ROOT / amendment["base_profile_recipe"]["path"]
    if sha256(plan_path) != amendment["base_plan"]["sha256"]:
        raise ValueError("frozen R1 plan changed")
    if sha256(recipe_path) != amendment["base_profile_recipe"]["sha256"]:
        raise ValueError("base profile recipe changed")
    plan, recipe = json_object(plan_path), json_object(recipe_path)
    old_source_pins = plan["source"]["current_source_sha256"]
    if (
        len(old_source_pins) != 41
        or recipe["source_manifest"]["canonical_source_sha256"] != old_source_pins
    ):
        raise ValueError("frozen R1 source manifest differs")
    source_pins = check_map(source_root, old_source_pins)
    additions = source["required_additions_sha256"]
    if set(additions) & set(old_source_pins):
        raise ValueError("R4 source addition overlaps the frozen source map")
    source_pins.update(check_map(source_root, additions))

    prior = amendment["prior_attempt"]
    r3_admission, r3_receipt = ROOT / prior["admission_path"], ROOT / prior["execution_path"]
    r3_log = ROOT / prior["log_path"]
    if any(
        sha256(path) != expected
        for path, expected in (
            (r3_admission, prior["admission_sha256"]),
            (r3_receipt, prior["execution_sha256"]),
            (r3_log, prior["log_sha256"]),
        )
    ):
        raise ValueError("preserved R3 admission, receipt or failure log changed")
    r3 = json_object(r3_receipt)
    if (
        r3["source_head"] != prior["source_commit"]
        or r3["adapter_sha256"] != prior["adapter_sha256"]
        or r3["status"] != "failed"
        or r3["fold"] != 1
        or r3["wall_seconds"] != prior["wall_seconds"]
        or "metadata exceeds its 512000-byte read bound" not in r3_log.read_text()
    ):
        raise ValueError("R3 failure identity changed")

    f0 = amendment["preserved_f0"]
    f0_receipt, f0_log = ROOT / f0["execution_path"], ROOT / f0["log_path"]
    if sha256(f0_receipt) != f0["execution_sha256"] or sha256(f0_log) != f0["log_sha256"]:
        raise ValueError("completed F0 R2 receipt/log changed")
    f0_result = json_object(f0_receipt)
    if (
        f0_result["source_head"] != f0["source_commit"]
        or f0_result["status"] != "complete"
        or f0_result["fold"] != 0
        or f0_result["wall_seconds"] != f0["wall_seconds"]
    ):
        raise ValueError("completed F0 R2 profile identity changed")

    profile = next(item for item in plan["profiles"] if item["fold"] == 1)
    argv = list(profile["argv"])
    argv[0] = str(source_root / "scripts/train_rnn.py")
    argv_hash = hashlib.sha256(
        json.dumps(argv[1:], separators=(",", ":"), ensure_ascii=True).encode("ascii")
    ).hexdigest()
    profile_pin = amendment["profile"]
    if (
        profile["profile_id"] != profile_pin["id"]
        or profile["measured_batches"] != 100
        or profile["warmup_batches"] != 10
        or argv_hash != profile_pin["unchanged_argv_sha256"]
        or argv[1:] != r3["argv"][1:]
        or "--save-training-state" in argv
        or "--resume-state" in argv
    ):
        raise ValueError("F1 profile argv or bounded-step contract changed")

    cell = next(item for item in plan["cells"] if item["fold"] == 1)
    parent, cache, prop = cell["parent"], cell["target_cache"], cell["propensity"]
    variant = cache["variant"]
    data_pins = {
        parent["checkpoint"]: parent["checkpoint_sha256"],
        cell["split"]["path"]: cell["split"]["sha256"],
        cache["root"] + "/metadata.json": cache["metadata_sha256"],
        cache["root"] + f"/main_{variant}_train.npy": cache["main_train_sha256"],
        cache["root"] + f"/main_{variant}_valid.npy": cache["main_valid_sha256"],
        f"artifacts/runs/{prop['run']}/model.pt": prop["checkpoint_sha256"],
        f"artifacts/runs/{parent['run']}/provenance.json": parent["provenance_sha256"],
        f"artifacts/runs/{prop['run']}/provenance.json": prop["provenance_sha256"],
    }
    checked_data = check_map(ROOT, data_pins)
    receipt_path, log_path = ROOT / RECEIPT_REL, ROOT / LOG_REL
    run_name = argv[argv.index("--name") + 1]
    if receipt_path.exists() or log_path.exists() or (ROOT / "artifacts/runs" / run_name).exists():
        raise FileExistsError("R4 receipt, log or output run already exists")

    env = os.environ.copy()
    env.update(
        {
            "WNN_PROJECT_ROOT": str(ROOT),
            "PYTHONPATH": str(source_root / "src") + os.pathsep + str(source_root),
            **runtime,
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
    evidence = {
        "schema_version": 1,
        "status": "pins_passed_profile_starting",
        "source_head": args.source_sha,
        "source_root": str(source_root),
        "source_amendment_sha256": sha256(amendment_path),
        "adapter_sha256": sha256(Path(__file__)),
        "profile_id": profile["profile_id"],
        "fold": 1,
        "source_pins": source_pins,
        "data_pins": checked_data,
        "preserved_f0_receipt_sha256": sha256(f0_receipt),
        "preserved_f0_log_sha256": sha256(f0_log),
        "preserved_r3_receipt_sha256": sha256(r3_receipt),
        "preserved_r3_log_sha256": sha256(r3_log),
        "argv": argv,
        "profile_batches": 110,
        "expected_optimizer_updates": 1100,
        "scientific_fit_performed": False,
        "preflight_seconds": time.monotonic() - started,
    }
    with receipt_path.open("x", encoding="utf-8") as stream:
        json.dump(evidence, stream, indent=2, allow_nan=False)
        stream.write("\n")
    result = subprocess.run([sys.executable, "-u", *argv], cwd=ROOT, env=env, check=False)
    evidence.update(
        status="complete" if result.returncode == 0 else "failed",
        returncode=result.returncode,
        wall_seconds=time.monotonic() - started,
    )
    receipt_path.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
