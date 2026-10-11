"""Bounded offline validation for the immutable R3 packet and bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any

PACKET_REL = (
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
    "ops_continuation_20261011/clean_t3_fold0_h390_r3_teacher_cohort"
)
R2_REL = PACKET_REL.replace("_r3_teacher_cohort", "_r2_teacher_cohort")
PROFILE_SOURCE_SHA = "41a5452c752bb10bee28af2eb2e06e7a224da8d72f307b4d5bec6fb5b3b236fa"
SOURCE_DELTA_SHA = "912205eba101c8539c372206e5ae0e56f2d0a39f8997b271189b3d2f7d601406"
PRODUCER_PREFLIGHT_SHA = "fc16040084e041a8e458f80463f9096bcd9a3c946928580b55aa4468cde35afc"


def sha256_stream(stream) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1 << 20), b""):
        digest.update(block)
    return digest.hexdigest()


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return sha256_stream(stream)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate(root: Path, bundle: Path) -> dict[str, Any]:
    packet = root / PACKET_REL
    manifest = read_json(bundle / "bundle-manifest.json")
    config = read_json(packet / "KERNEL_CONFIGURATION_R3.json")
    plan = read_json(packet / "TEACHER_COHORT_PLAN_R3.json")
    gate = read_json(packet / "CONDITIONAL_FIT_ADMISSION_R3.json")
    guard = read_json(packet / "PROFILE_GUARD_CONTRACT_R3.json")
    input_binding = read_json(packet / "INPUT_BINDING_R3.json")
    source_binding = read_json(packet / "SOURCE_BINDING_R3.json")
    profile_reference = read_json(packet / "PROFILE_REFERENCE_R3.json")
    source_manifest = packet / "source_snapshot/SOURCE_MANIFEST_H390.json"

    assert manifest["dataset"] == "vladkochetov07/wnn-clean-t3-f0-teachers-20261011-r3"
    assert manifest["private"] is True
    assert (
        json.loads((bundle / "kernel/kernel-metadata.json").read_text())["id"]
        == f"{manifest['dataset']}-runner"
    )
    assert sha256(bundle / "bundle.wnn") == manifest["archive_sha256"]
    assert manifest["archive_sha256"] == config["bundle_archive_sha256"]
    assert sha256(bundle / "bundle-manifest.json") == config["bundle_manifest_sha256"]
    kernel_path = bundle / "kernel/kernel.py"
    compile(kernel_path.read_text(), str(kernel_path), "exec")
    assert sha256(kernel_path) == config["bootstrap_sha256"]
    assert kernel_path.stat().st_size < 1_000_000
    assert sha256(source_manifest) == PROFILE_SOURCE_SHA
    assert sha256(packet / "SOURCE_DELTA_R3.json") == SOURCE_DELTA_SHA
    assert sha256(packet / "ACTUAL_PRODUCER_PREFLIGHT_R2_RECEIPT.json") == PRODUCER_PREFLIGHT_SHA
    producer = read_json(packet / "ACTUAL_PRODUCER_PREFLIGHT_R2_RECEIPT.json")
    delta = read_json(packet / "SOURCE_DELTA_R3.json")
    assert producer["status"] == "pass_static_actual_producer_fixtures_only"
    assert producer["split_manifest_sha256"] == plan["split_manifest_sha256"]
    assert delta["parent_source_manifest_sha256"] == source_binding["source_manifest_parent_sha256"]
    assert delta["new_source_manifest_sha256"] == PROFILE_SOURCE_SHA
    assert source_binding["source_manifest_sha256"] == PROFILE_SOURCE_SHA
    assert source_binding["source_delta_receipt_sha256"] == SOURCE_DELTA_SHA
    assert source_binding["actual_producer_preflight_sha256"] == PRODUCER_PREFLIGHT_SHA
    assert profile_reference["source_manifest_sha256"] == PROFILE_SOURCE_SHA
    assert profile_reference["scope"].startswith("one-worker")
    assert profile_reference["observed"]["optimizer_completed_updates"] == 599
    assert profile_reference["observed"]["amp_skipped_updates"] == 1
    assert input_binding["split_manifest_sha256"] == plan["split_manifest_sha256"]
    assert (
        input_binding["own_propensity_checkpoint_sha256"]
        == plan["own_propensity_checkpoint_sha256"]
    )
    assert len(input_binding["external_inputs"]) == 13
    input_rows = {row["path"]: row for row in input_binding["external_inputs"]}
    external_paths = set(manifest["external_mounts"])
    assert set(input_rows) == external_paths
    manifest_rows = {row["path"]: row for row in manifest["files"]}
    for path, row in input_rows.items():
        assert row["size"] == manifest_rows[path]["size"]
        assert row["sha256"] == manifest_rows[path]["sha256"]
        assert row["mount"] == manifest["external_mounts"][path]

    source_doc = read_json(source_manifest)
    source_files = source_doc["sources"]
    for relative, expected in source_files.items():
        source_path = packet / "source_snapshot" / relative
        assert source_path.is_file() and sha256(source_path) == expected, relative

    with tarfile.open(bundle / "bundle.wnn", "r:gz") as archive:
        members = {item.name: item for item in archive.getmembers()}
        assert len(members) == len(archive.getmembers())
        allowed = {
            row["path"]
            for row in manifest["files"]
            if row["path"] not in manifest["external_mounts"]
        } | set(manifest["spec_paths"])
        assert set(members) == allowed
        for row in manifest["files"]:
            if row["path"] in manifest["external_mounts"]:
                continue
            member = members[row["path"]]
            assert member.size == row["size"], row["path"]
            stream = archive.extractfile(member)
            assert stream is not None and sha256_stream(stream) == row["sha256"], row["path"]
        specs = {
            name: json.loads(archive.extractfile(path).read())
            for name, path in ((Path(path).stem, path) for path in manifest["spec_paths"])
        }

    profiles = [
        specs[f"profile_clean_t3_f0_muonall_h384_e3_s{seed}_paired_t4_r3"] for seed in (3, 4)
    ]
    assert len(profiles) == 2
    for seed, profile in zip((3, 4), profiles, strict=True):
        argv = profile["argv"]
        assert argv[argv.index("--seed") + 1] == str(seed)
        assert argv[argv.index("--profile-warmup") + 1] == "5"
        assert argv[argv.index("--profile-steps") + 1] == "5"
        assert "--grad-stats" in argv
        assert profile["artifacts"] == []
        assert profile["limits"]["ram_bytes"] == 6_000_000_000
        assert profile["limits"]["gpu_memory_bytes"] == 5_000_000_000

    fits = [specs[f"fit_clean_t3_f0_muonall_h384_e3_s{seed}_r3"] for seed in (3, 4, 5)]
    base_r2 = read_json(root / R2_REL / "jobs/fit_clean_t3_f0_muonall_h384_e3_s3_r2.json")
    for seed, fit in zip((3, 4, 5), fits, strict=True):
        argv = fit["argv"]
        assert argv[argv.index("--seed") + 1] == str(seed)
        assert argv[argv.index("--hidden") + 1] == "384"
        assert argv[argv.index("--layers") + 1] == "2"
        assert argv[argv.index("--epochs") + 1] == "3"
        assert argv[argv.index("--checkpoint-epoch") + 1] == "2"
        assert argv[argv.index("--optim") + 1] == "muon"
        assert argv[argv.index("--muon-scope") + 1] == "all"
        assert argv[argv.index("--batch") + 1] == "32"
        assert argv[argv.index("--chunk") + 1] == "1000"
        assert "--save-training-state" in argv
        assert "--initializer" not in argv
        assert len(fit["artifacts"]) == 5
        expected_argv = list(base_r2["argv"])
        expected_argv[1] = f"{PACKET_REL}/source_snapshot/scripts/cleanfold_exec.py"
        expected_argv[expected_argv.index("--name") + 1] = f"clean_t3_f0_muonall_h384_e3_s{seed}_r3"
        expected_argv[expected_argv.index("--seed") + 1] = str(seed)
        assert argv == expected_argv, "R3 fit argv has an unreviewed scientific change"

    for entry in plan["export_stage"]:
        name = Path(entry["spec"]).stem
        export = specs[name]
        assert export["artifacts"] == [entry["npy"]["path"], entry["sidecar"]]
        if entry["kind"] == "full_valid":
            assert export["argv"][-1] == "--campaign-full-valid-feature-only"
            assert entry["npy"]["shape"] == [1873, 20000, 2]
        else:
            assert "--campaign-full-valid-feature-only" not in export["argv"]

    assert len(plan["fit_stage"]) == 3 and len(plan["export_stage"]) == 9
    assert plan["fit_membership"] == {"train": 10535, "valid": 1405, "excluded_valid_eval": 468}
    assert guard["require_exactly_two_distinct_t4_devices"] is True
    assert guard["profile_batches"] == {
        "warmup_complete_batches": 5,
        "measured_complete_batches": 5,
    }
    assert gate["status"] == "pending_root_review_and_release"
    assert gate["release_authorized"] is False
    assert config["full_cohort_fit_admission"] is False
    assert config["total_eta_seconds"] is None
    assert plan["actual_total_eta_seconds"] is None

    return {
        "schema": "wnn.kaggle.teacher_cohort_static_validation.r3",
        "status": "PASS",
        "archive_sha256": manifest["archive_sha256"],
        "bundle_manifest_sha256": sha256(bundle / "bundle-manifest.json"),
        "bootstrap_sha256": sha256(kernel_path),
        "source_manifest_sha256": sha256(source_manifest),
        "archive_members": len(members),
        "profile_jobs": len(profiles),
        "fit_jobs": len(fits),
        "export_jobs": len(plan["export_stage"]),
        "fit_admission": False,
        "provider_actions": 0,
        "checks": [
            "all source snapshot hashes and opaque bundle members match",
            "profile stage is two 5-warmup/5-measured profile-only specs with resource caps",
            "fit seeds/recipe/checkpoint/state artifacts match R2 science recipe",
            "fold-0 fit membership excludes the 468 evaluation groups",
            "all nine prediction exports are declared; full-valid exports are 1873-row feature-only endpoints",
            "conditional root gate remains pending and ETA fields remain null",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    receipt = validate(args.root.resolve(strict=True), args.bundle.resolve(strict=True))
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
